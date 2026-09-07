"""Native Unreal Engine capture worker for RoadSentinel's 20-day experiment.

This file is launched by Unreal Editor with ``-ExecutePythonScript``.  It
materialises the requested condition manifest inside RoadSentinelSim and uses
``SceneCapture2D`` plus an RGBA8 render target for every fixed road segment.
The resulting PNGs are genuine UE renders: they are not CARLA frames, PIL
drawings, or screenshots of the Unreal Editor chrome.

The host runner writes a request JSON below ``env/output`` and supplies its
path through ``ROADSENTINEL_TEMPORAL_CAPTURE_REQUEST``.  Keeping that contract
file-based makes the capture process fully automatic even though Unreal closes
after an ExecutePythonScript job finishes.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple


try:
    import unreal
except ImportError:  # Allows syntax checks outside Unreal Engine.
    unreal = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = WORKSPACE_ROOT / "env" / "output"
REQUEST_ENV = "ROADSENTINEL_TEMPORAL_CAPTURE_REQUEST"
REQUEST_SCHEMA = "RoadSentinelUnrealTemporalCaptureRequest/v1"


def _require_output_path(value: Any, description: str) -> Path:
    """Resolve a request artifact while constraining it to project output."""
    if not value:
        raise ValueError(f"Missing {description}")
    path = Path(str(value)).expanduser().resolve()
    try:
        path.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{description} must be below {OUTPUT_ROOT}: {path}") from exc
    return path


def _load_request() -> Tuple[Path, Dict[str, Any]]:
    default_path = OUTPUT_ROOT / "temporal_20_day" / "unreal_capture_request.json"
    request_path = _require_output_path(os.environ.get(REQUEST_ENV, default_path), "capture request")
    if not request_path.is_file():
        raise FileNotFoundError(
            f"Unreal temporal capture request not found: {request_path}. "
            f"Set {REQUEST_ENV} before starting Unreal."
        )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("Unsupported Unreal temporal capture request schema")
    entries = request.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Capture request needs at least one entry")
    return request_path, request


def _load_phase2_module() -> Any:
    """Import the existing RoadSentinel UE scene builder, rather than clone it."""
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import rs_phase2_surface_defects as phase2

    if not phase2.HAS_UNREAL:
        raise RuntimeError("rs_temporal_20_day_capture.py must run in an Unreal Editor Python process")
    return phase2


def _editor_world() -> Any:
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
    except Exception:
        subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        world = subsystem.get_editor_world()
    if not world:
        raise RuntimeError("Unreal Editor world is not available for SceneCapture2D")
    return world


def _render_target(world: Any, width: int, height: int) -> Any:
    """Create a transient LDR render target suitable for lossless PNG export."""
    render_format = getattr(
        unreal.TextureRenderTargetFormat,
        "RTF_RGBA8_SRGB",
        unreal.TextureRenderTargetFormat.RTF_RGBA8,
    )
    clear = unreal.LinearColor(0.0, 0.0, 0.0, 1.0)
    try:
        target = unreal.RenderingLibrary.create_render_target2d(
            world,
            int(width),
            int(height),
            render_format,
            clear_color=clear,
            auto_generate_mips=False,
        )
    except TypeError:
        # UE Python API signatures vary slightly between point releases.
        target = unreal.RenderingLibrary.create_render_target2d(world, int(width), int(height), render_format)
    if not target:
        raise RuntimeError("Could not create Unreal SceneCapture2D render target")
    return target


def _capture_component(actor: Any) -> Any:
    component = None
    try:
        component = actor.get_capture_component2d()
    except Exception:
        pass
    if not component:
        component = actor.get_component_by_class(unreal.SceneCaptureComponent2D)
    if not component:
        raise RuntimeError("Spawned SceneCapture2D actor has no capture component")
    return component


def _wait_for_png(expected_path: Path, timeout_s: float = 20.0) -> Path:
    """Wait for UE's image exporter and tolerate its rare duplicate suffix."""
    candidates = (expected_path, expected_path.with_name(expected_path.name + ".png"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 0:
                if candidate != expected_path:
                    candidate.replace(expected_path)
                return expected_path
        time.sleep(0.1)
    raise RuntimeError(f"Unreal did not export capture within {timeout_s:.0f}s: {expected_path}")


def _capture_png(
    world: Any,
    camera_pose: Dict[str, Any],
    camera_config: Dict[str, Any],
    output_path: Path,
) -> None:
    """Render one actual RoadSentinelSim camera view to a clean PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = _render_target(world, int(camera_config["width"]), int(camera_config["height"]))
    location = unreal.Vector(
        float(camera_pose["x_cm"]),
        float(camera_pose["y_cm"]),
        float(camera_pose["z_cm"]),
    )
    rotation = unreal.Rotator(
        roll=float(camera_pose.get("roll_deg", 0.0)),
        pitch=float(camera_pose["pitch_deg"]),
        yaw=float(camera_pose["yaw_deg"]),
    )
    camera = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, location, rotation)
    if not camera:
        raise RuntimeError("Could not spawn Unreal SceneCapture2D camera")

    try:
        component = _capture_component(camera)
        component.set_editor_property("texture_target", target)
        component.set_editor_property("fov_angle", float(camera_config["fov_deg"]))
        component.set_editor_property("capture_every_frame", False)
        component.set_editor_property("capture_on_movement", False)
        try:
            component.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        except (AttributeError, TypeError):
            # The target remains a real UE render even if an older engine uses
            # its default final-colour capture source.
            pass

        # SceneCapture2D renders immediately to the target.  A second capture
        # after a short settle gives streamed materials one render tick without
        # reverting to an editor-viewport screenshot.
        component.capture_scene()
        time.sleep(0.20)
        component.capture_scene()
        unreal.RenderingLibrary.export_render_target(
            world,
            target,
            str(output_path.parent),
            output_path.name,
        )
        _wait_for_png(output_path)
    finally:
        try:
            unreal.EditorLevelLibrary.destroy_actor(camera)
        except Exception:
            pass


def _gps_for_world(world_x_m: float, world_y_m: float) -> Dict[str, float]:
    """Deterministic local WGS-84 reference for dashboard/work-order location."""
    base_lat = 13.0827
    base_lon = 80.2707
    latitude = base_lat + world_y_m / 111_320.0
    longitude = base_lon + world_x_m / (111_320.0 * math.cos(math.radians(base_lat)))
    return {"latitude": round(latitude, 7), "longitude": round(longitude, 7), "altitude_m": 25.0}


def _capture_day(phase2: Any, entry: Dict[str, Any]) -> None:
    day = int(entry.get("day", 0))
    if not 1 <= day <= 20:
        raise ValueError(f"Temporal capture day must be in 1..20, got {day}")
    manifest_path = _require_output_path(entry.get("manifest_path"), f"day {day} manifest")
    capture_dir = _require_output_path(entry.get("capture_dir"), f"day {day} capture directory")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Day {day} manifest does not exist: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("day", 0)) != day:
        raise ValueError(f"Requested day {day} does not match its manifest")
    manifest_segments = payload.get("segments", [])
    if not isinstance(manifest_segments, list) or not manifest_segments:
        raise ValueError(f"Day {day} manifest has no road segments")
    segment_manifest = {str(item.get("road_segment_id")): item for item in manifest_segments}
    if len(segment_manifest) != len(manifest_segments):
        raise ValueError(f"Day {day} manifest has duplicate road segment IDs")

    image_dir = capture_dir / "images"
    ground_truth_path = capture_dir / "ground_truth.json"
    # This existing helper rebuilds the native UE road, creates only manifest
    # defects at stable coordinates, and returns the corresponding fixed poses.
    scene = phase2.materialize_temporal_manifest(str(manifest_path), str(ground_truth_path))
    if int(scene.get("day", 0)) != day:
        raise RuntimeError(f"Unreal materialised the wrong day for request {day}")
    camera_config = scene["camera_config"]
    world = _editor_world()
    try:
        unreal.EditorLevelLibrary.redraw_all_viewports()
    except Exception:
        pass

    metadata: List[Dict[str, Any]] = []
    for order, segment in enumerate(scene["segments"], start=1):
        segment_id = str(segment["road_segment_id"])
        raw_segment = segment_manifest.get(segment_id)
        if raw_segment is None:
            raise RuntimeError(f"Unreal returned unknown segment {segment_id} on day {day}")
        image_name = f"{segment_id}.png"
        _capture_png(world, segment["camera_pose"], camera_config, image_dir / image_name)
        world_location = segment["world_location_cm"]
        world_x_m = float(world_location["x"]) / 100.0
        world_y_m = float(world_location["y"]) / 100.0
        gps = _gps_for_world(world_x_m, world_y_m)
        metadata.append({
            "day": day,
            "road_segment_id": segment_id,
            "image_name": image_name,
            "route_order": order,
            "sim_time_s": float(raw_segment.get("viewpoint_time_s", order - 1)),
            "world_x_m": round(world_x_m, 3),
            "world_y_m": round(world_y_m, 3),
            "world_z_m": float(world_location["z"]) / 100.0,
            "camera_x_m": float(segment["camera_pose"]["x_cm"]) / 100.0,
            "camera_y_m": float(segment["camera_pose"]["y_cm"]) / 100.0,
            "altitude_m": float(camera_config["altitude_m"]),
            "yaw_deg": float(segment["camera_pose"]["yaw_deg"]),
            "pitch_deg": float(segment["camera_pose"]["pitch_deg"]),
            "latitude": gps["latitude"],
            "longitude": gps["longitude"],
            "capture_source": "Unreal Engine SceneCapture2D render target",
        })

    capture_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = capture_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0].keys()))
        writer.writeheader()
        writer.writerows(metadata)

    # The scene helper records renderer provenance.  Add only capture location
    # metadata; no ground truth is used by the ML inference score downstream.
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    metadata_by_segment = {row["road_segment_id"]: row for row in metadata}
    for defect in ground_truth.get("defects", []):
        row = metadata_by_segment.get(str(defect.get("road_segment_id")))
        if row:
            defect["gps_coordinates"] = {
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "altitude_m": row["altitude_m"],
            }
    ground_truth.update({
        "engine": "Unreal Engine SceneCapture2D render target",
        "capture_source": "Unreal Engine SceneCapture2D render target",
        "capture_directory": str(capture_dir),
        "captures": metadata,
    })
    ground_truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    unreal.log(f"[RoadSentinel] Temporal UE capture day {day:02d} complete: {len(metadata)} PNGs")


def _write_failure_marker(request_path: Path, exc: BaseException) -> None:
    marker = request_path.parent / "unreal_capture_failure.json"
    marker.write_text(json.dumps({
        "status": "error",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }, indent=2), encoding="utf-8")


def main() -> None:
    if unreal is None:
        raise RuntimeError("This script must be executed by Unreal Editor, not system Python")
    request_path, request = _load_request()
    phase2 = _load_phase2_module()
    entries = request["entries"]
    for entry in sorted(entries, key=lambda item: int(item.get("day", 0))):
        if not isinstance(entry, dict):
            raise ValueError("Every capture request entry must be an object")
        _capture_day(phase2, entry)
    unreal.log("[RoadSentinel] Unreal 20-day capture batch complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            path = _require_output_path(
                os.environ.get(REQUEST_ENV, OUTPUT_ROOT / "temporal_20_day" / "unreal_capture_request.json"),
                "capture request",
            )
            _write_failure_marker(path, exc)
        except Exception:
            pass
        traceback.print_exc()
        raise
