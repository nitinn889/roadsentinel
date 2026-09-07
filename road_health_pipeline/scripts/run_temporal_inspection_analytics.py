#!/usr/bin/env python3
"""
run_temporal_inspection_analytics.py
------------------------------------
Execute ML analysis, deterioration tracking, held-out prediction,
work-order generation, and dashboard integration on the 20-day
Unreal Engine inspection capture dataset.

Preserves existing valid outputs and computes complete end-to-end
results with DINOv2 / SAM2 perception, visual overlays, and FastAPI
dashboard integration.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = WORKSPACE_ROOT / "road_health_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config import CONFIG
from common.io_utils import load_rgb
from inference.run_inference import generate_visual_overlays, infer, load_pipeline
from temporal_20_day import (
    EVALUATION_START_DAY,
    SEGMENTS,
    condition_label,
    make_work_orders,
    normalized_detections,
    predict_held_out,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("temporal_analytics")


def gps_for_world(world_x_m: float, world_y_m: float) -> Dict[str, float]:
    """Deterministic local WGS-84 reference for dashboard/work-order location."""
    base_lat = 13.0827
    base_lon = 80.2707
    latitude = base_lat + world_y_m / 111_320.0
    longitude = base_lon + world_x_m / (111_320.0 * math.cos(math.radians(base_lat)))
    return {"latitude": round(latitude, 7), "longitude": round(longitude, 7), "altitude_m": 25.0}


def run_analytics(
    inspections_dir: Path,
    output_dir: Path,
    work_order_threshold: float = 0.65,
    device: str = CONFIG.device,
    force: bool = False,
) -> Path:
    inspections_dir = Path(inspections_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses_dir = inspections_dir / "analysis"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading RoadSentinel DINOv2 / SAM2 pipeline on {device}...")
    t0 = time.time()
    ml_pipeline = load_pipeline(device=device, memory_bank_dir=CONFIG.memory_bank_dir)
    log.info(f"Pipeline initialized in {time.time() - t0:.2f}s")

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

    total_images = 20 * len(SEGMENTS)
    processed_count = 0
    reused_count = 0

    for day in range(1, 21):
        day_dir = inspections_dir / f"day_{day:02d}"
        gt_file = inspections_dir / f"day_{day:02d}_ground_truth.json"
        gt_defects = {}
        if gt_file.is_file():
            try:
                gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
                gt_defects = {d.get("road_segment_id"): d for d in gt_data.get("defects", [])}
            except Exception:
                pass

        log.info(f"--- Processing Day {day:02d}/20 ---")

        for spec in SEGMENTS:
            seg_id = spec.segment_id
            seg_capture_dir = day_dir / seg_id
            raw_image_path = seg_capture_dir / "raw.png"
            meta_path = seg_capture_dir / "metadata.json"

            if not raw_image_path.is_file():
                log.warning(f"Missing raw image: {raw_image_path}")
                continue

            # Load metadata
            meta = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            coords = meta.get("world_coordinates", {})
            along_m = float(coords.get("along_m", spec.along_m))
            across_m = float(coords.get("across_m", spec.across_m))
            gps = gps_for_world(along_m, across_m)
            all_segments[seg_id]["location"] = gps

            weather = meta.get("weather_preset", "Unknown")

            analysis_seg_dir = analyses_dir / seg_id / f"day_{day:02d}"
            analysis_seg_dir.mkdir(parents=True, exist_ok=True)

            telemetry_path = analysis_seg_dir / "frame_metadata.json"
            telemetry = {
                "timestamp": f"Day {day:02d}",
                "latitude": gps["latitude"],
                "longitude": gps["longitude"],
                "altitude_m": gps["altitude_m"],
                "heading_deg": float(meta.get("camera_pose", {}).get("yaw_deg", 0.0)),
                "world_x": along_m,
                "world_y": across_m,
                "speed_mps": 30.0 / 3.6,
                "frame_id": f"{seg_id}_day_{day:02d}",
                "weather_preset": weather,
            }
            write_json(telemetry_path, telemetry)

            result_json_path = analysis_seg_dir / "result.json"
            overlay_path = analysis_seg_dir / "detection_overlay.jpg"
            severity_overlay_path = analysis_seg_dir / "severity_overlay.jpg"
            health_overlay_path = analysis_seg_dir / "road_health_overlay.jpg"

            if (
                not force
                and result_json_path.is_file()
                and overlay_path.is_file()
                and severity_overlay_path.is_file()
                and health_overlay_path.is_file()
            ):
                # Reuse existing valid observation
                try:
                    observation = json.loads(result_json_path.read_text(encoding="utf-8"))
                    # Ensure paths are updated to current locations
                    observation["image_path"] = str(raw_image_path.resolve())
                    observation["overlay_path"] = str(overlay_path.resolve())
                    all_segments[seg_id]["history"].append(observation)
                    reused_count += 1
                    processed_count += 1
                    continue
                except Exception:
                    pass  # Fall through to re-run

            # Run ML inference
            t_infer_start = time.time()
            result = infer(
                raw_image_path,
                metadata_path=telemetry_path,
                pipeline=ml_pipeline,
                road_segment_id=seg_id,
                test_mode_2d=False,
            )

            rgb = load_rgb(raw_image_path)
            overlays = generate_visual_overlays(rgb, result.potholes, result.road_health)

            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlays["detection_overlay"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(severity_overlay_path), cv2.cvtColor(overlays["severity_overlay"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(health_overlay_path), cv2.cvtColor(overlays["road_health_overlay"], cv2.COLOR_RGB2BGR))

            detections = normalized_detections(result, raw_image_path, overlay_path)
            max_severity = max((float(d.get("severity_score", 0.0)) for d in detections), default=0.0)
            defect_type = max(detections, key=lambda d: float(d.get("severity_score", 0.0)), default={}).get("defect_type")

            has_water = any(bool(d.get("water_flag")) for d in detections)
            cond = condition_label(max_severity, defect_type, has_water)

            observation = {
                "day": day,
                "condition": cond,
                "severity_score": round(max_severity, 3),
                "road_health_score": round(result.road_health.road_health_score, 2),
                "detections": detections,
                "detection_count": len(detections),
                "has_water_hazard": has_water,
                "image_path": str(raw_image_path.resolve()),
                "overlay_path": str(overlay_path.resolve()),
                "weather_preset": weather,
                "ml_anomaly_score": round(float(result.anomaly_score), 4),
                "ml_anomaly_threshold": round(float(result.anomaly_threshold), 4),
                "calibration": [x for x in result.warnings if x.startswith("Domain-adaptive")],
            }
            write_json(result_json_path, observation)
            all_segments[seg_id]["history"].append(observation)

            processed_count += 1
            infer_dur = time.time() - t_infer_start
            log.info(
                f"  [{processed_count:3d}/{total_images}] Day {day:02d} {seg_id} ({weather[:15]}): "
                f"Sev={max_severity:.3f} Health={result.road_health.road_health_score:.1f} "
                f"Detections={len(detections)} [{infer_dur:.2f}s]"
            )

    log.info(f"Inference complete: {processed_count} inspections processed ({reused_count} reused from cache).")

    # Predict deterioration progression
    segment_list = list(all_segments.values())
    for segment in segment_list:
        segment["prediction"] = predict_held_out(segment["history"])
        segment["current"] = segment["history"][-1] if segment["history"] else {}

    # Generate prioritized work orders
    work_orders = make_work_orders(segment_list, work_order_threshold)
    work_order_by_seg = {w["road_segment_id"]: w for w in work_orders}
    for segment in segment_list:
        segment["work_order"] = work_order_by_seg.get(segment["road_segment_id"])

    # Unique weather presets used
    all_weathers = sorted(list({
        obs.get("weather_preset", "Unknown")
        for seg in segment_list
        for obs in seg.get("history", [])
    }))

    result_dataset = {
        "schema_version": "RoadSentinelTemporal20Day/v1",
        "metadata": {
            "days": 20,
            "map_name": "RoadSentinelSim / SimBlank",
            "execution": "automated Unreal Engine SceneCapture2D inspection with varying weather conditions",
            "capture_source": "Unreal Engine SceneCapture2D render target",
            "model_training_days": list(range(1, EVALUATION_START_DAY)),
            "unseen_evaluation_days": list(range(EVALUATION_START_DAY, 21)),
            "work_order_threshold": work_order_threshold,
            "weather_presets_used": all_weathers,
            "ground_truth_policy": "Rendering/location only; ML scores and detections come from DINOv2/SAM2 RGB inference.",
        },
        "segments": segment_list,
        "work_orders": work_orders,
        "overall_health": {
            "segments_deteriorated": [s["road_segment_id"] for s in segment_list if s["current"].get("severity_score", 0.0) >= 0.35],
            "segments_healthy": [s["road_segment_id"] for s in segment_list if s["current"].get("severity_score", 0.0) < 0.10],
        },
    }

    # Save to manual_inspections
    write_json(inspections_dir / "temporal_results.json", result_dataset)
    write_json(inspections_dir / "work_orders.json", {"metadata": result_dataset["metadata"], "work_orders": work_orders})

    # Also update temporal_20_day for dashboard compatibility
    output_results_path = output_dir / "temporal_results.json"
    write_json(output_results_path, result_dataset)
    write_json(output_dir / "work_orders.json", {"metadata": result_dataset["metadata"], "work_orders": work_orders})

    # Sync analysis directories to output_dir so dashboard can serve overlays from either
    target_analysis_dir = output_dir / "analysis"
    target_analysis_dir.mkdir(parents=True, exist_ok=True)
    for seg_dir in analyses_dir.iterdir():
        if seg_dir.is_dir():
            target_seg = target_analysis_dir / seg_dir.name
            target_seg.mkdir(parents=True, exist_ok=True)
            for day_subdir in seg_dir.iterdir():
                if day_subdir.is_dir():
                    dest = target_seg / day_subdir.name
                    dest.mkdir(parents=True, exist_ok=True)
                    for file_path in day_subdir.iterdir():
                        if file_path.is_file():
                            shutil.copy2(file_path, dest / file_path.name)

    log.info(f"Results successfully saved to:")
    log.info(f"  - {inspections_dir / 'temporal_results.json'}")
    log.info(f"  - {output_results_path}")
    log.info(f"Work orders generated: {len(work_orders)}")
    for wo in work_orders:
        log.info(f"  [WO-{wo['work_order_id']}] Segment: {wo['road_segment_id']} | Priority: {wo['priority']} | Action: {wo['recommended_maintenance_action']}")

    return output_results_path


def main():
    parser = argparse.ArgumentParser(description="Run ML analytics, prediction, and work-order generation on inspection captures.")
    parser.add_argument(
        "--inspections-dir",
        type=Path,
        default=WORKSPACE_ROOT / "env" / "output" / "manual_inspections",
        help="Directory containing the 20-day manual inspection captures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "env" / "output" / "temporal_20_day",
        help="Directory where temporal_results.json and work_orders.json are written for dashboard integration",
    )
    parser.add_argument("--work-order-threshold", type=float, default=0.65, help="Severity threshold for work order generation")
    parser.add_argument("--device", type=str, default=CONFIG.device, help="Compute device (cuda / cpu)")
    parser.add_argument("--force", action="store_true", help="Re-run inference even if cached result.json exists")
    args = parser.parse_args()

    results_path = run_analytics(
        inspections_dir=args.inspections_dir,
        output_dir=args.output_dir,
        work_order_threshold=args.work_order_threshold,
        device=args.device,
        force=args.force,
    )
    print(f"\n[✓] ML Analysis & Dashboard Integration Complete: {results_path}")


if __name__ == "__main__":
    main()
