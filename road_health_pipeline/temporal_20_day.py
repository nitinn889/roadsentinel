"""Automated, leakage-aware 20-day Unreal Engine road-deterioration experiment.

This is an integration runner, not a synthetic dashboard fixture.  Unreal
Engine materialises the same RoadSentinelSim road and deterministic drone
camera anchors for every day, then exports clean SceneCapture2D images.  Those
native Unreal renders are passed through the existing DINOv2 + SAM2 perception
stack and aggregated by persistent road-patch ID.

The condition manifests are renderer input only.  They preserve physical
locations and control the Unreal scene; they are never used to calculate an ML
severity score or to manufacture a detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "env" / "output" / "temporal_20_day"
EVALUATION_START_DAY = 15
UNREAL_PROJECT = ROOT / "RoadSentinelSim" / "RoadSentinelSim.uproject"
UNREAL_LAUNCHER = ROOT / "launch_unreal_editor.sh"
UNREAL_CAPTURE_SCRIPT = ROOT / "RoadSentinelSim" / "Content" / "Python" / "rs_temporal_20_day_capture.py"
UNREAL_OUTPUT_ROOT = ROOT / "env" / "output"


@dataclass(frozen=True)
class SegmentSpec:
    segment_id: str
    along_m: float
    across_m: float
    profile: str
    description: str

    @property
    def viewpoint_time_s(self) -> float:
        # All inspection flights use the same 30km/h (8.333m/s) fixed route.
        return self.along_m / (30.0 / 3.6)


SEGMENTS: Tuple[SegmentSpec, ...] = (
    SegmentSpec("SEG_001", 35.0, -1.35, "stable", "Control patch: remains healthy"),
    SegmentSpec("SEG_002", 95.0, 1.35, "slow", "Slow aggregate wear and late minor cracking"),
    SegmentSpec("SEG_003", 155.0, -1.35, "gradual", "Gradual fatigue crack progressing to a dry pothole"),
    SegmentSpec("SEG_004", 215.0, 1.35, "rapid", "Accelerated failure progressing to a water-filled pothole"),
    SegmentSpec("SEG_005", 275.0, -1.35, "wet", "Slow onset then wet pothole and hydroplaning risk"),
    SegmentSpec("SEG_006", 335.0, 1.35, "healthy", "Resilient healthy comparison patch"),
)


def severity_tier(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def condition_label(score: float, defect_type: Optional[str], water: bool = False) -> str:
    if water:
        return "Water-filled pothole"
    if score < 0.10:
        return "Healthy"
    if defect_type in {"crack", "crack_or_damage"} or score < 0.35:
        return "Minor wear" if score < 0.22 else "Crack"
    if score >= 0.82:
        return "Severe deterioration"
    return "Pothole"


def latent_severity(profile: str, day: int) -> float:
    """Deterministic condition curve used solely by the Unreal surface renderer."""
    t = (day - 1) / 19.0
    if profile == "stable":
        return 0.025 + 0.012 * t
    if profile == "healthy":
        return 0.015 + 0.008 * t
    if profile == "slow":
        return 0.025 + 0.25 * max(0.0, (day - 4) / 16.0)
    if profile == "gradual":
        return 0.03 + 0.57 * max(0.0, (day - 2) / 18.0) ** 1.12
    if profile == "rapid":
        return 0.04 + 0.88 * max(0.0, (day - 1) / 19.0) ** 1.20
    if profile == "wet":
        return 0.025 + 0.84 * max(0.0, (day - 5) / 15.0) ** 1.35
    raise ValueError(f"Unknown deterioration profile: {profile}")


def render_defect_for_segment(segment: SegmentSpec, day: int, seed: int) -> Optional[Dict[str, Any]]:
    """Build one persistent visual defect record for a segment/day.

    Coordinates and IDs never change.  Only condition attributes (dimensions,
    cracking and water state) evolve.  The returned record is written to the
    condition manifest consumed by the native Unreal scene materialiser.
    """
    severity = round(float(np.clip(latent_severity(segment.profile, day), 0.0, 0.98)), 3)
    if severity < 0.085:
        return None

    if severity < 0.34:
        defect_type = "crack"
        length_m = 0.45 + 1.60 * severity
        width_m = 0.035 + 0.045 * severity
        depth_m = 0.008 + 0.020 * severity
        is_water = False
        shape = "elongated_longitudinal"
    else:
        is_water = segment.profile in {"rapid", "wet"} and severity >= 0.66
        defect_type = "water_filled_pothole" if is_water else "pothole"
        length_m = 0.42 + 1.55 * severity
        width_m = 0.32 + 1.20 * severity
        depth_m = 0.020 + 0.145 * severity
        shape = "irregular_natural" if segment.profile != "rapid" else "compound_cluster"

    area = max(0.01, math.pi * length_m * width_m / 4.0)
    orientation = 0.0 if defect_type == "crack" else (25.0 if segment.segment_id in {"SEG_004", "SEG_005"} else 8.0)
    return {
        "defect_id": f"{segment.segment_id}_PRIMARY",
        "road_segment_id": segment.segment_id,
        "along_m": segment.along_m,
        "across_m": segment.across_m,
        "defect_type": defect_type,
        "shape_category": shape,
        "lane_position": "left_wheel_track" if segment.across_m < 0 else "right_wheel_track",
        "dimensions": {
            "length_m": round(length_m, 3),
            "width_m": round(width_m, 3),
            "diameter_m": round(math.sqrt(length_m * width_m), 3),
            "depth_m": round(depth_m, 3),
            "area_m2": round(area, 3),
            "aspect_ratio": round(max(length_m, width_m) / max(0.01, min(length_m, width_m)), 2),
            "orientation_deg": orientation,
        },
        "surface_properties": {
            "irregularity": round(0.20 + 0.62 * severity, 3),
            "edge_breakup": round(0.12 + 0.75 * severity, 3),
            "roughness": round(0.15 + 0.70 * severity, 3),
        },
        "water_state": {
            "is_water_filled": is_water,
            "water_level_m": round(depth_m * 0.65, 3) if is_water else 0.0,
            "water_coverage_frac": round(min(0.96, 0.48 + severity * 0.48), 2) if is_water else 0.0,
            "turbidity": round(0.22 + 0.30 * severity, 2) if is_water else 0.0,
            "wet_halo_radius_m": round(0.16 + 0.32 * severity, 2) if is_water else 0.0,
        },
        "associated_defects": {
            "has_cracks": defect_type != "crack" and severity >= 0.43,
            "crack_pattern": "alligator" if severity >= 0.66 else "radial",
            "has_road_patch": False,
        },
        "severity_category": severity_tier(severity),
        "true_severity_score": severity,
        "generation_seed": seed + int(segment.segment_id[-3:]),
    }


def build_day_manifest(day: int, seed: int) -> Dict[str, Any]:
    defects = [d for seg in SEGMENTS if (d := render_defect_for_segment(seg, day, seed)) is not None]
    return {
        "schema_version": "RoadSentinelTemporalManifest/v1",
        "day": day,
        "map_name": "RoadSentinelSim / SimBlank",
        "fixed_flight": {
            "altitude_m": 25.0,
            "speed_kmph": 30.0,
            "weather": "Clear Noon (70° Sun)",
            "camera_fov_deg": 70.0,
            "route": "RoadSentinelSim straight highway then curve; six fixed segment camera anchors",
        },
        "unreal_capture": {
            "altitude_m": 25.0,
            "pitch_deg": -89.0,
            "fov_deg": 70.0,
            "width": 1920,
            "height": 1080,
            "source": "Unreal SceneCapture2D render target",
        },
        "segments": [
            {
                "road_segment_id": seg.segment_id,
                "along_m": seg.along_m,
                "across_m": seg.across_m,
                "profile": seg.profile,
                "description": seg.description,
                "viewpoint_time_s": round(seg.viewpoint_time_s, 3),
                "renderer_condition_score": latent_severity(seg.profile, day),
            }
            for seg in SEGMENTS
        ],
        "defects": defects,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_capture_rows(capture_dir: Path) -> List[Dict[str, Any]]:
    csv_path = capture_dir / "metadata.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Unreal capture did not produce metadata.csv: {csv_path}")
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    image_dir = capture_dir / "images"
    valid = [r for r in rows if r.get("image_name") and (image_dir / r["image_name"]).is_file()]
    if not valid:
        raise RuntimeError(f"Unreal Engine produced no captured RGB images in {image_dir}")
    return valid


def pick_segment_frame(capture_dir: Path, rows: List[Dict[str, Any]], segment: SegmentSpec) -> Tuple[Path, Dict[str, Any]]:
    # Native UE export has one deterministic image per persistent segment.  Do
    # not pick a neighbouring flight frame merely because it is nearby in time.
    explicit_rows = [item for item in rows if item.get("road_segment_id")]
    if explicit_rows:
        matched = [item for item in explicit_rows if item.get("road_segment_id") == segment.segment_id]
        if len(matched) != 1:
            raise RuntimeError(
                f"Expected exactly one Unreal capture for {segment.segment_id}; found {len(matched)} in {capture_dir}"
            )
        row = matched[0]
        return capture_dir / "images" / row["image_name"], row

    # Compatibility with pre-existing non-temporal artifacts.  New temporal
    # runs never use this branch.
    row = min(rows, key=lambda item: abs(float(item.get("sim_time_s", 0.0)) - segment.viewpoint_time_s))
    return capture_dir / "images" / row["image_name"], row


def telemetry_for_frame(row: Dict[str, Any], day: int) -> Dict[str, Any]:
    return {
        "timestamp": f"Day {day:02d}",
        "latitude": float(row.get("latitude", 13.0827)),
        "longitude": float(row.get("longitude", 80.2707)),
        "altitude_m": float(row.get("altitude_m", 25.0)),
        "heading_deg": float(row.get("yaw_deg", 0.0)),
        "world_x": float(row.get("world_x_m", row.get("local_x_m", 0.0))),
        "world_y": float(row.get("world_y_m", row.get("local_y_m", 0.0))),
        "speed_mps": 30.0 / 3.6,
        "frame_id": row.get("image_name", ""),
    }


def normalized_detections(result: Any, image_path: Path, overlay_path: Path) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for det, legacy in zip(result.detections, result.potholes):
        d = det.to_dict()
        d["bbox_xyxy"] = d.pop("bbox")
        d["pothole_id"] = d["defect_id"]
        d["severity_score"] = float(d.get("severity", {}).get("severity_score", 0.0))
        d["water_flag"] = bool(d.get("is_water_filled", False))
        d["source_image"] = image_path.name
        d["image_path"] = str(image_path.resolve())
        d["overlay_path"] = str(overlay_path.resolve())
        d["area_m2"] = d.get("estimated_area_m2")
        d["estimated_depth_m"] = d.get("estimated_depth_m")
        d["anomaly_score"] = legacy.anomaly_score
        output.append(d)
    return output


def _require_unreal_output_path(path: Path, description: str) -> Path:
    """Keep Unreal-side request paths inside the project's designated output tree."""
    resolved = path.resolve()
    try:
        resolved.relative_to(UNREAL_OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{description} must be inside {UNREAL_OUTPUT_ROOT}; got {resolved}. "
            "The Unreal renderer only accepts project output paths."
        ) from exc
    return resolved


def _capture_day_complete(capture_dir: Path) -> bool:
    """Return true only when the real Unreal exports for all persistent patches exist."""
    if not (capture_dir / "metadata.csv").is_file() or not (capture_dir / "ground_truth.json").is_file():
        return False
    image_dir = capture_dir / "images"
    return all((image_dir / f"{segment.segment_id}.png").is_file() for segment in SEGMENTS)


def run_unreal_capture_batch(
    requests: Sequence[Tuple[int, Path, Path]],
    request_path: Path,
    unreal_gui: bool = False,
) -> None:
    """Ask the actual RoadSentinelSim Unreal project to render capture days.

    ``-ExecutePythonScript`` runs the supplied scene-capture script in a real
    Unreal Editor process.  It uses a render target, not CARLA, a PIL mock, or
    an editor-viewport screenshot.  A single Editor launch processes the
    supplied day list, keeping the temporal run automatic and avoiding twenty
    costly manual GUI launches.
    """
    if not requests:
        return
    if not UNREAL_PROJECT.is_file():
        raise FileNotFoundError(f"RoadSentinel Unreal project is missing: {UNREAL_PROJECT}")
    if not UNREAL_LAUNCHER.is_file():
        raise FileNotFoundError(f"Unreal launcher is missing: {UNREAL_LAUNCHER}")
    if not UNREAL_CAPTURE_SCRIPT.is_file():
        raise FileNotFoundError(f"Unreal temporal capture script is missing: {UNREAL_CAPTURE_SCRIPT}")

    request_entries = []
    for day, manifest_path, capture_dir in requests:
        manifest_path = _require_unreal_output_path(manifest_path, "Manifest")
        capture_dir = _require_unreal_output_path(capture_dir, "Capture directory")
        request_entries.append({
            "day": int(day),
            "manifest_path": str(manifest_path),
            "capture_dir": str(capture_dir),
        })
    request_path = _require_unreal_output_path(request_path, "Unreal capture request")
    write_json(request_path, {
        "schema_version": "RoadSentinelUnrealTemporalCaptureRequest/v1",
        "project": str(UNREAL_PROJECT),
        "entries": request_entries,
    })

    cmd = [
        str(UNREAL_LAUNCHER),
        f"-ExecutePythonScript={UNREAL_CAPTURE_SCRIPT}",
        "-NoSound",
        "-NoSplash",
        "-Unattended",
    ]
    if not unreal_gui:
        # This is still the actual UE renderer; it simply avoids a visible
        # viewport while automation is running.  Never use -NullRHI here.
        cmd.append("-RenderOffscreen")
    env = os.environ.copy()
    env["ROADSENTINEL_TEMPORAL_CAPTURE_REQUEST"] = str(request_path)
    mode = "Unreal Editor GUI" if unreal_gui else "Unreal Engine offscreen renderer"
    day_text = ", ".join(f"{day:02d}" for day, _, _ in requests)
    print(f"[Temporal] Rendering day(s) {day_text} with {mode}; no CARLA process is used.")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    incomplete = [str(capture_dir) for _, _, capture_dir in requests if not _capture_day_complete(capture_dir)]
    if incomplete:
        raise RuntimeError(
            "Unreal Engine exited without all expected SceneCapture2D exports: " + ", ".join(incomplete)
        )


def predict_held_out(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit only days 1–14 and evaluate days 15–20 without temporal leakage."""
    train = [x for x in history if int(x["day"]) < EVALUATION_START_DAY]
    if len(train) < 2:
        return {
            "training_days": [],
            "prediction_horizon_days": 7,
            "predicted_future_severity": 0.0,
            "predicted_future_condition": "Healthy",
        }
    x = np.array([float(item["day"]) for item in train], dtype=np.float32)
    y = np.array([float(item["severity_score"]) for item in train], dtype=np.float32)
    slope, intercept = np.polyfit(x, y, 1)

    heldout = []
    errors = []
    for item in history:
        day = int(item["day"])
        if day < EVALUATION_START_DAY:
            continue
        predicted = float(np.clip(intercept + slope * day, 0.0, 1.0))
        actual = float(item["severity_score"])
        errors.append(abs(predicted - actual))
        heldout.append({"day": day, "predicted_severity": round(predicted, 3), "observed_severity": round(actual, 3)})

    future_day = 20 + 7
    future = float(np.clip(intercept + slope * future_day, 0.0, 1.0))
    return {
        "model": "bounded_linear_severity_trend",
        "training_days": [int(i) for i in x.tolist()],
        "unseen_evaluation_days": list(range(EVALUATION_START_DAY, 21)),
        "prediction_horizon_days": 7,
        "slope_per_day": round(float(slope), 4),
        "predicted_future_day": future_day,
        "predicted_future_severity": round(future, 3),
        "predicted_future_condition": condition_label(
            future,
            "pothole" if future >= 0.35 else ("crack" if future >= 0.10 else None),
        ),
        "held_out_predictions": heldout,
        "unseen_mae": round(float(np.mean(errors)), 4) if errors else None,
    }


def make_work_orders(segments: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    from vlm_work_order_gen import generate_fallback_work_order

    orders: List[Dict[str, Any]] = []
    for segment in segments:
        history = segment["history"]
        forecast = segment["prediction"]
        trigger = next((x for x in history if float(x["severity_score"]) >= threshold), None)
        if trigger is None and float(forecast.get("predicted_future_severity", 0.0)) >= threshold:
            trigger = history[EVALUATION_START_DAY - 2]  # last observed training day (day 14)
        if trigger is None:
            continue

        evidence = trigger.get("detections", [])
        primary = max(evidence, key=lambda d: float(d.get("severity_score", 0.0)), default={})
        current = float(trigger["severity_score"])
        predicted = float(forecast.get("predicted_future_severity", current))
        severity = max(current, predicted)
        tier = severity_tier(severity)
        water = bool(trigger.get("has_water_hazard", False))
        defect_class = primary.get("defect_type") or trigger.get("condition", "road deterioration").lower().replace(" ", "_")
        fallback = generate_fallback_work_order(
            defect_class=defect_class,
            severity_tier=tier,
            severity_score=severity,
            area_m2=primary.get("area_m2"),
            estimated_depth_m=primary.get("estimated_depth_m"),
            is_water_filled=water,
            road_segment_id=segment["road_segment_id"],
            pothole_id=primary.get("pothole_id", f"{segment['road_segment_id']}-condition"),
        )
        priority = "P1 - Immediate" if water or severity >= 0.85 else "P2 - High" if severity >= threshold else "P3 - Planned"
        orders.append({
            "work_order_id": f"WO-20D-{segment['road_segment_id']}",
            "dispatch_status": "DISPATCHED",
            "dispatch_target": "municipal_maintenance_queue",
            "trigger_day": trigger["day"],
            "road_segment_id": segment["road_segment_id"],
            "location": segment["location"],
            "detected_defects": [d.get("defect_type") for d in evidence],
            "bounding_box_evidence": [{"bbox_xyxy": d.get("bbox_xyxy"), "image": trigger["overlay_path"]} for d in evidence],
            "current_severity_score": round(current, 3),
            "predicted_future_severity": round(predicted, 3),
            "priority": priority,
            "recommended_maintenance_action": fallback["work_order_text"],
            **fallback,
        })
    return orders


def run_temporal_simulation(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    days: int = 20,
    duration_s: Optional[float] = None,
    seed: int = 2026,
    work_order_threshold: float = 0.65,
    resume: bool = False,
    unreal_gui: bool = False,
) -> Path:
    if days != 20:
        raise ValueError("This experiment is intentionally fixed at 20 inspection iterations.")
    if duration_s is not None:
        print("[Temporal] --duration is ignored for Unreal fixed-anchor capture; every SEG_* pose is rendered once per day.")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Output directory already has data: {output_dir}. Use --resume to continue safely.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    captures_dir = output_dir / "captures"
    analyses_dir = output_dir / "analysis"

    # Build all deterministic day manifests before Unreal starts.  The UE
    # script then re-materialises the same map once per day and captures the
    # fixed SEG_001...SEG_006 camera route with SceneCapture2D.
    capture_requests: List[Tuple[int, Path, Path]] = []
    for day in range(1, 21):
        manifest_path = manifests_dir / f"day_{day:02d}.json"
        write_json(manifest_path, build_day_manifest(day, seed))
        capture_dir = captures_dir / f"day_{day:02d}"
        if not resume or not _capture_day_complete(capture_dir):
            capture_requests.append((day, manifest_path, capture_dir))
    run_unreal_capture_batch(
        capture_requests,
        output_dir / "unreal_capture_request.json",
        unreal_gui=unreal_gui,
    )

    # Load the existing perception stack once; it is reused for every frame.
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))
    from inference.run_inference import generate_visual_overlays, infer, load_pipeline
    from common.io_utils import load_rgb
    from config import CONFIG

    print("[Temporal] Loading existing RoadSentinel DINOv2/SAM2 pipeline once for 120 segment inspections...")
    ml_pipeline = load_pipeline(device=CONFIG.device, memory_bank_dir=CONFIG.memory_bank_dir)
    all_segments: Dict[str, Dict[str, Any]] = {
        s.segment_id: {
            "road_segment_id": s.segment_id,
            "profile": s.profile,
            "description": s.description,
            "persistent_location": {"along_m": s.along_m, "across_m": s.across_m},
            "location": {},
            "history": [],
        }
        for s in SEGMENTS
    }

    for day in range(1, 21):
        capture_dir = captures_dir / f"day_{day:02d}"
        rows = read_capture_rows(capture_dir)
        gt = json.loads((capture_dir / "ground_truth.json").read_text(encoding="utf-8"))
        gt_by_segment = {d.get("road_segment_id"): d for d in gt.get("defects", [])}

        for spec in SEGMENTS:
            image_path, row = pick_segment_frame(capture_dir, rows, spec)
            analysis_dir = analyses_dir / spec.segment_id / f"day_{day:02d}"
            telemetry_path = analysis_dir / "frame_metadata.json"
            write_json(telemetry_path, telemetry_for_frame(row, day))
            result = infer(
                image_path,
                metadata_path=telemetry_path,
                pipeline=ml_pipeline,
                road_segment_id=spec.segment_id,
                test_mode_2d=False,
            )
            rgb = load_rgb(image_path)
            overlays = generate_visual_overlays(rgb, result.potholes, result.road_health)
            analysis_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = analysis_dir / "detection_overlay.jpg"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlays["detection_overlay"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(analysis_dir / "severity_overlay.jpg"), cv2.cvtColor(overlays["severity_overlay"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(analysis_dir / "road_health_overlay.jpg"), cv2.cvtColor(overlays["road_health_overlay"], cv2.COLOR_RGB2BGR))

            detections = normalized_detections(result, image_path, overlay_path)
            max_severity = max((float(d.get("severity_score", 0.0)) for d in detections), default=0.0)
            defect_type = max(detections, key=lambda d: float(d.get("severity_score", 0.0)), default={}).get("defect_type")
            ground_truth = gt_by_segment.get(spec.segment_id, {})
            # Every persistent patch has a geographic reference even when it
            # remains visually healthy and therefore has no renderer defect.
            all_segments[spec.segment_id]["location"] = {
                "latitude": float(row.get("latitude", 13.0827)),
                "longitude": float(row.get("longitude", 80.2707)),
                "altitude_m": float(row.get("altitude_m", 25.0)),
            }
            if ground_truth.get("gps_coordinates"):
                all_segments[spec.segment_id]["location"] = ground_truth.get("gps_coordinates", {})
            observation = {
                "day": day,
                "condition": condition_label(max_severity, defect_type, any(d.get("water_flag") for d in detections)),
                "severity_score": round(max_severity, 3),
                "road_health_score": result.road_health.road_health_score,
                "detections": detections,
                "detection_count": len(detections),
                "has_water_hazard": any(bool(d.get("water_flag")) for d in detections),
                "image_path": str(image_path.resolve()),
                "overlay_path": str(overlay_path.resolve()),
                "frame_time_s": float(row.get("sim_time_s", 0.0)),
                "ml_anomaly_score": round(float(result.anomaly_score), 4),
                "ml_anomaly_threshold": round(float(result.anomaly_threshold), 4),
                "calibration": [x for x in result.warnings if x.startswith("Domain-adaptive")],
            }
            write_json(analysis_dir / "result.json", observation)
            all_segments[spec.segment_id]["history"].append(observation)
        print(f"[Temporal] Day {day:02d}/20 complete: Unreal Engine capture + ML inference + bbox overlays")

    segment_list = list(all_segments.values())
    for segment in segment_list:
        segment["prediction"] = predict_held_out(segment["history"])
        segment["current"] = segment["history"][-1]
    work_orders = make_work_orders(segment_list, work_order_threshold)
    work_order_by_seg = {w["road_segment_id"]: w for w in work_orders}
    for segment in segment_list:
        segment["work_order"] = work_order_by_seg.get(segment["road_segment_id"])

    result = {
        "schema_version": "RoadSentinelTemporal20Day/v1",
        "metadata": {
            "days": 20,
            "map_name": "RoadSentinelSim / SimBlank",
            "execution": "automated Unreal Engine SceneCapture2D fixed-route inspection (no CARLA)",
            "capture_source": "Unreal Engine SceneCapture2D render target",
            "fixed_flight": build_day_manifest(1, seed)["fixed_flight"],
            "model_training_days": list(range(1, EVALUATION_START_DAY)),
            "unseen_evaluation_days": list(range(EVALUATION_START_DAY, 21)),
            "work_order_threshold": work_order_threshold,
            "ground_truth_policy": "Rendering/location only; ML scores and detections come from RGB inference.",
        },
        "segments": segment_list,
        "work_orders": work_orders,
        "overall_health": {
            "segments_deteriorated": [s["road_segment_id"] for s in segment_list if s["current"]["severity_score"] >= 0.35],
            "segments_healthy": [s["road_segment_id"] for s in segment_list if s["current"]["severity_score"] < 0.10],
        },
    }
    results_path = output_dir / "temporal_results.json"
    write_json(results_path, result)
    write_json(output_dir / "work_orders.json", {"metadata": result["metadata"], "work_orders": work_orders})
    return results_path


def dashboard_port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def launch_existing_dashboard(port: int) -> None:
    """Start the existing FastAPI Government Dashboard without replacing one already running."""
    url = f"http://127.0.0.1:{port}/temporal"
    if dashboard_port_is_open(port):
        print(f"[Temporal] Existing RoadSentinel dashboard is active: {url}")
        return
    python_bin = ROOT / ".venv" / "bin" / "python"
    if not python_bin.is_file():
        python_bin = Path(sys.executable)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PIPELINE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.Popen(
        [str(python_bin), str(PIPELINE_ROOT / "inference" / "server.py"), "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(PIPELINE_ROOT),
        env=env,
        start_new_session=True,
    )
    print(f"[Temporal] Dashboard started: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fully automated RoadSentinel 20-day Unreal Engine deterioration simulation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Deprecated compatibility option; Unreal renders one fixed camera pose per segment each day.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--work-order-threshold", type=float, default=0.65)
    parser.add_argument("--resume", action="store_true", help="Reuse only days with complete Unreal SceneCapture2D exports")
    parser.add_argument(
        "--unreal-gui",
        action="store_true",
        help="Show the Unreal Editor while it renders; default uses the same UE renderer offscreen.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Existing dashboard server port")
    parser.add_argument("--no-dashboard", action="store_true", help="Do not auto-launch the existing dashboard after completion")
    args = parser.parse_args()
    result = run_temporal_simulation(
        output_dir=args.output_dir,
        duration_s=args.duration,
        seed=args.seed,
        work_order_threshold=args.work_order_threshold,
        resume=args.resume,
        unreal_gui=args.unreal_gui,
    )
    print(f"[Temporal] Complete 20-day dashboard dataset: {result}")
    if not args.no_dashboard:
        launch_existing_dashboard(args.port)


if __name__ == "__main__":
    main()
