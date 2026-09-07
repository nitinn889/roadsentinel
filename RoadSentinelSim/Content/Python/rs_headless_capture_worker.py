"""
rs_headless_capture_worker.py
-----------------------------
Unreal Engine worker script for the automated headless inspection capture.

This script is designed to be launched by Unreal Editor via:
    UnrealEditor project.uproject -ExecutePythonScript=<this_file> -RenderOffscreen ...

It reads a capture plan JSON from env/output/manual_inspections/capture_plan.json
(or the path in ROADSENTINEL_INSPECTION_CAPTURE_PLAN env var), and for each
(day, segment, weather) triple it:

  1. Materialises the temporal manifest for that day.
  2. Applies the assigned weather/lighting preset.
  3. Renders the segment via SceneCapture2D (genuine UE render).
  4. Validates the resulting PNG (non-blank, correct dimensions).
  5. Writes raw.png + metadata.json to the deterministic output path.

On completion it writes a completion marker JSON.  The host-side orchestrator
polls for this file.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any, Dict, List

try:
    import unreal
except ImportError:
    unreal = None

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = WORKSPACE_ROOT / "env" / "output"
PLAN_ENV = "ROADSENTINEL_INSPECTION_CAPTURE_PLAN"
DEFAULT_PLAN_PATH = OUTPUT_ROOT / "manual_inspections" / "capture_plan.json"


def _load_plan() -> Dict[str, Any]:
    plan_path = Path(os.environ.get(PLAN_ENV, str(DEFAULT_PLAN_PATH)))
    if not plan_path.is_file():
        raise FileNotFoundError(f"Capture plan not found: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "RoadSentinelInspectionCapturePlan/v1":
        raise ValueError(f"Unsupported plan schema: {plan.get('schema_version')}")
    return plan


def _load_phase2():
    """Import the existing scene builder / materialiser."""
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import rs_phase2_surface_defects as phase2
    if not phase2.HAS_UNREAL:
        raise RuntimeError("Worker must run inside Unreal Editor")
    return phase2


def _load_inspection_helper():
    helper_dir = str(WORKSPACE_ROOT / "env" / "scripts")
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    import rs_inspection_capture as ic
    return ic


def _editor_world():
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
    except Exception:
        subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        world = subsystem.get_editor_world()
    if not world:
        raise RuntimeError("Unreal Editor world is not available")
    return world


def _render_target(world, width: int, height: int):
    render_format = getattr(
        unreal.TextureRenderTargetFormat,
        "RTF_RGBA8_SRGB",
        unreal.TextureRenderTargetFormat.RTF_RGBA8,
    )
    clear = unreal.LinearColor(0.0, 0.0, 0.0, 1.0)
    try:
        target = unreal.RenderingLibrary.create_render_target2d(
            world, int(width), int(height),
            render_format, clear_color=clear, auto_generate_mips=False,
        )
    except TypeError:
        target = unreal.RenderingLibrary.create_render_target2d(
            world, int(width), int(height), render_format,
        )
    if not target:
        raise RuntimeError("Could not create render target")
    return target


def _capture_png(world, camera_pose: Dict, camera_config: Dict, output_path: Path) -> None:
    """Full SceneCapture2D render to PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = _render_target(world, int(camera_config["width"]), int(camera_config["height"]))

    loc = unreal.Vector(
        float(camera_pose["x_cm"]),
        float(camera_pose["y_cm"]),
        float(camera_pose["z_cm"]),
    )
    rot = unreal.Rotator(
        roll=float(camera_pose.get("roll_deg", 0.0)),
        pitch=float(camera_pose["pitch_deg"]),
        yaw=float(camera_pose["yaw_deg"]),
    )
    cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
    if not cam:
        raise RuntimeError("Could not spawn SceneCapture2D")

    try:
        comp = None
        try:
            comp = cam.get_capture_component2d()
        except Exception:
            pass
        if not comp:
            comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
        if not comp:
            raise RuntimeError("No capture component")

        comp.set_editor_property("texture_target", target)
        comp.set_editor_property("fov_angle", float(camera_config["fov_deg"]))
        comp.set_editor_property("capture_every_frame", False)
        comp.set_editor_property("capture_on_movement", False)
        try:
            comp.set_editor_property(
                "capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
            )
        except (AttributeError, TypeError):
            pass

        # Double-capture for material streaming
        comp.capture_scene()
        time.sleep(0.30)
        comp.capture_scene()

        unreal.RenderingLibrary.export_render_target(
            world, target, str(output_path.parent), output_path.name,
        )

        # Wait for PNG to appear
        deadline = time.time() + 20.0
        alt_path = output_path.with_name(output_path.name + ".png")
        while time.time() < deadline:
            if output_path.is_file() and output_path.stat().st_size > 0:
                break
            if alt_path.is_file() and alt_path.stat().st_size > 0:
                alt_path.replace(output_path)
                break
            time.sleep(0.10)
        else:
            raise RuntimeError(f"UE did not export PNG within 20s: {output_path}")
    finally:
        try:
            unreal.EditorLevelLibrary.destroy_actor(cam)
        except Exception:
            pass


def _process_day(phase2, ic, day: int, captures: List[Dict], ground_truth_path: str) -> List[Dict]:
    """Materialise one day and capture all segments with their assigned weather."""
    manifest_path = str(WORKSPACE_ROOT / "env" / "output" / "temporal_20_day" / "manifests" / f"day_{day:02d}.json")
    scene = phase2.materialize_temporal_manifest(manifest_path, ground_truth_path)
    camera_config = scene["camera_config"]
    segments_by_id = {s["road_segment_id"]: s for s in scene["segments"]}
    world = _editor_world()

    try:
        unreal.EditorLevelLibrary.redraw_all_viewports()
    except Exception:
        pass

    results = []
    for capture in captures:
        segment_id = capture["segment_id"]
        weather = capture["weather"]

        seg = segments_by_id.get(segment_id)
        if not seg:
            results.append({
                "day": day, "segment_id": segment_id, "weather": weather,
                "status": "error", "error": f"Segment {segment_id} not in scene",
            })
            continue

        # Apply weather for this segment
        phase2.apply_environment_lighting(weather)
        # Small settle for lighting to take effect
        time.sleep(0.15)

        out_dir = ic.build_output_dir(day, segment_id)
        raw_path = out_dir / "raw.png"

        try:
            _capture_png(world, seg["camera_pose"], camera_config, raw_path)

            val = ic.validate_png(
                raw_path,
                expected_width=int(camera_config["width"]),
                expected_height=int(camera_config["height"]),
            )

            meta_path = ic.write_metadata(
                output_dir=out_dir,
                day=day,
                segment_id=segment_id,
                camera_pose=seg["camera_pose"],
                world_coordinates={
                    "x_cm": seg["world_location_cm"].get("x", 0.0),
                    "y_cm": seg["world_location_cm"].get("y", 0.0),
                    "z_cm": seg["world_location_cm"].get("z", 0.0),
                    "along_m": seg["persistent_location"].get("along_m", 0.0),
                    "across_m": seg["persistent_location"].get("across_m", 0.0),
                },
                condition_score=seg.get("renderer_condition_score", 0.0),
                extra={
                    "weather_preset": weather,
                    "capture_mode": "headless_automated",
                    "total_defects_day": len(scene.get("defects", [])),
                },
            )

            results.append({
                "day": day, "segment_id": segment_id, "weather": weather,
                "status": "ok" if val["valid"] else "warning",
                "path": str(raw_path),
                "metadata_path": str(meta_path),
                "validated": val["valid"],
                "validation_reason": val.get("reason"),
                "file_size_bytes": val.get("file_size_bytes", 0),
            })
            unreal.log(
                f"[RoadSentinel] Headless capture day {day:02d} {segment_id} "
                f"[{weather}] — {'OK' if val['valid'] else 'WARN: ' + str(val.get('reason'))}"
            )
        except Exception as exc:
            results.append({
                "day": day, "segment_id": segment_id, "weather": weather,
                "status": "error", "error": str(exc),
            })
            unreal.log_error(
                f"[RoadSentinel] Headless capture FAILED day {day:02d} {segment_id}: {exc}"
            )

    return results


def main():
    if unreal is None:
        raise RuntimeError("This script must be executed by Unreal Editor")

    plan = _load_plan()
    phase2 = _load_phase2()
    ic = _load_inspection_helper()

    # Group captures by day for efficient scene materialisation
    captures_by_day: Dict[int, List[Dict]] = {}
    for entry in plan["captures"]:
        day = int(entry["day"])
        captures_by_day.setdefault(day, []).append(entry)

    all_results = []
    total = len(plan["captures"])
    completed = 0

    for day in sorted(captures_by_day.keys()):
        day_captures = captures_by_day[day]
        gt_path = str(OUTPUT_ROOT / "manual_inspections" / f"day_{day:02d}_ground_truth.json")
        results = _process_day(phase2, ic, day, day_captures, gt_path)
        all_results.extend(results)
        completed += len(day_captures)
        unreal.log(
            f"[RoadSentinel] Progress: {completed}/{total} captures "
            f"({completed * 100 // total}%)"
        )

    # Write completion marker
    marker_path = Path(plan.get("completion_marker", str(
        OUTPUT_ROOT / "manual_inspections" / "capture_complete.json"
    )))
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({
        "status": "complete",
        "total_captures": total,
        "successful": sum(1 for r in all_results if r.get("validated")),
        "warnings": sum(1 for r in all_results if r.get("status") == "warning"),
        "errors": sum(1 for r in all_results if r.get("status") == "error"),
        "results": all_results,
    }, indent=2), encoding="utf-8")
    unreal.log(f"[RoadSentinel] Headless capture complete: {total} captures written.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        marker = OUTPUT_ROOT / "manual_inspections" / "capture_complete.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }, indent=2), encoding="utf-8")
        traceback.print_exc()
        raise
