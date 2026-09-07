#!/usr/bin/env python3
"""
ingest_manual_inspections.py
----------------------------
Ingestion runner for user-captured Unreal Engine inspection images.

Ingests captures from:
    env/output/manual_inspections/day_XX/SEG_XXX/raw.png
    env/output/manual_inspections/day_XX/SEG_XXX/metadata.json

Key features:
1. Scans Day 1-20 and SEG_001-SEG_008.
2. Validates inputs before inference:
   - Rejects missing, empty, or corrupt images.
   - Rejects nearly all-white, all-black, or blank captures with a visible diagnostic error.
   - Records rejected captures as invalid rather than treating them as healthy roads.
3. For every valid image:
   - Runs the existing ML inference pipeline (DINOv2 patch embeddings, domain adaptation, SAM2).
   - Generates ML bounding-box overlays (detection, severity, road health).
   - Calculates observed severity and condition strictly from ML results (no Unreal ground-truth fabrication).
   - Keeps source metadata, camera pose, and GPS coordinates from metadata.json.
4. Builds persistent history per segment across days:
   - Fits deterioration trend only on Days 1-14.
   - Evaluates on unseen Days 15-20.
   - Forecasts future severity and condition after Day 20.
5. Generates prioritized work orders when observed or forecast severity reaches threshold.
6. Writes final results conforming to RoadSentinelTemporal20Day/v1 schema for the dashboard.
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

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = WORKSPACE_ROOT / "road_health_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config import CONFIG
from common.io_utils import load_rgb
from inference.run_inference import generate_visual_overlays, infer, load_pipeline
from vlm_work_order_gen import generate_fallback_work_order
from temporal_20_day import (
    EVALUATION_START_DAY,
    SegmentSpec,
    condition_label,
    normalized_detections,
    severity_tier,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_manual_inspections")

ALL_SEGMENT_SPECS: Tuple[SegmentSpec, ...] = (
    SegmentSpec("SEG_001", 35.0, -1.35, "stable", "Control patch: baseline structural integrity"),
    SegmentSpec("SEG_002", 95.0, 1.35, "slow", "Slow aggregate wear and late minor cracking"),
    SegmentSpec("SEG_003", 155.0, -1.35, "gradual", "Gradual fatigue crack progressing to a dry pothole"),
    SegmentSpec("SEG_004", 215.0, 1.35, "rapid", "Accelerated failure progressing to a water-filled pothole"),
    SegmentSpec("SEG_005", 275.0, -1.35, "wet", "Slow onset then wet pothole and hydroplaning risk"),
    SegmentSpec("SEG_006", 335.0, 1.35, "healthy", "Resilient comparison patch"),
    SegmentSpec("SEG_007", 395.0, -1.35, "moderate", "Subsurface fatigue and longitudinal cracking"),
    SegmentSpec("SEG_008", 455.0, 1.35, "edge_wear", "Shoulder degradation and edge subsidence"),
)


def gps_for_world(world_x_m: float, world_y_m: float) -> Dict[str, float]:
    """Deterministic local WGS-84 reference for dashboard/work-order location."""
    base_lat = 13.0827
    base_lon = 80.2707
    latitude = base_lat + world_y_m / 111_320.0
    longitude = base_lon + world_x_m / (111_320.0 * math.cos(math.radians(base_lat)))
    return {"latitude": round(latitude, 7), "longitude": round(longitude, 7), "altitude_m": 25.0}


def validate_inspection_image(
    image_path: Path,
    min_size_bytes: int = 1000,
    max_extreme_fraction: float = 0.98,
) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
    """Validate raw inspection capture. Rejects empty, corrupt, all-white, or all-black frames.

    Returns:
        (is_valid, rejection_reason, loaded_cv2_image_bgr)
    """
    if not image_path.is_file():
        return False, f"Capture file not found: {image_path.name}", None

    file_size = image_path.stat().st_size
    if file_size < min_size_bytes:
        return False, f"File too small ({file_size} bytes < {min_size_bytes}); empty or incomplete render", None

    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Corrupted or unreadable image file (failed decoding)", None

    if len(img.shape) != 3 or img.shape[0] < 200 or img.shape[1] < 200:
        return False, f"Invalid image dimensions: {img.shape}; expected 1920x1080 RGB", img

    # Pixel intensity distribution checks
    # 1. Pitch-black frame check (SceneCapture2D nighttime failure / unlit)
    if img.max() <= 1 or (img < 3).mean() >= max_extreme_fraction:
        dark_pct = (img < 3).mean() * 100.0
        return False, f"Frame rejected: image is pitch black ({dark_pct:.1f}% zero/dark pixels); SceneCapture2D exposure failure", img

    # 2. Overexposed all-white check (SceneCapture2D blown-out lighting)
    if img.min() >= 253 or (img > 252).mean() >= max_extreme_fraction:
        white_pct = (img > 252).mean() * 100.0
        return False, f"Frame rejected: image is overexposed / all-white ({white_pct:.1f}% 0xFF pixels); SceneCapture2D sun lux blown out", img

    # 3. Flat blank canvas (near-zero standard deviation)
    if img.std() < 1.0 and (img.mean() > 240.0 or img.mean() < 15.0):
        return False, f"Frame rejected: image has near-zero variance (std={img.std():.2f}, mean={img.mean():.1f}); blank render target", img

    return True, None, img


def create_diagnostic_error_overlay(
    raw_img: Optional[np.ndarray],
    segment_id: str,
    day: int,
    weather: str,
    reason: str,
    target_width: int = 1920,
    target_height: int = 1080,
) -> np.ndarray:
    """Create a high-visibility diagnostic error overlay image for rejected frames."""
    if raw_img is not None and raw_img.shape[:2] == (target_height, target_width):
        canvas = (raw_img.astype(np.float32) * 0.4).astype(np.uint8)
    else:
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        canvas[:, :] = (20, 20, 30)

    # 2. Top Banner
    cv2.rectangle(canvas, (0, 0), (target_width, 140), (10, 10, 20), -1)
    cv2.line(canvas, (0, 140), (target_width, 140), (0, 0, 255), 3)

    # 1. Bright red outer border (drawn on top)
    border_thick = 14
    cv2.rectangle(canvas, (0, 0), (target_width - 1, target_height - 1), (0, 0, 255), border_thick)

    cv2.putText(
        canvas,
        "ROAD SENTINEL -- REJECTED INSPECTION FRAME",
        (40, 55),
        cv2.FONT_HERSHEY_DUPLEX,
        1.3,
        (0, 77, 255),  # Amber-red
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Segment: {segment_id}  |  Day {day:02d}/20  |  Preset: {weather}",
        (40, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200, 220, 255),
        2,
        cv2.LINE_AA,
    )

    # 3. Center Error Card
    card_top = 400
    card_bottom = 680
    card_left = 120
    card_right = target_width - 120
    cv2.rectangle(canvas, (card_left, card_top), (card_right, card_bottom), (15, 18, 28), -1)
    cv2.rectangle(canvas, (card_left, card_top), (card_right, card_bottom), (0, 77, 255), 2)

    cv2.putText(
        canvas,
        "[!] INPUT VALIDATION FAILURE (FRAME EXCLUDED FROM ML INFERENCE)",
        (card_left + 40, card_top + 60),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )

    # Word wrap reason
    words = reason.split()
    lines = []
    curr_line = []
    for w in words:
        curr_line.append(w)
        if len(" ".join(curr_line)) > 75:
            lines.append(" ".join(curr_line))
            curr_line = []
    if curr_line:
        lines.append(" ".join(curr_line))

    y_offset = card_top + 120
    for l in lines[:3]:
        cv2.putText(
            canvas,
            l,
            (card_left + 40, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (240, 240, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 40

    cv2.putText(
        canvas,
        "Policy: Not treated as healthy road. Skips zero-defect assumption to prevent distortion.",
        (card_left + 40, card_bottom - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (140, 160, 180),
        1,
        cv2.LINE_AA,
    )

    return canvas


def predict_held_out_robust(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit polynomial trend only on valid observations in Days 1-14; evaluate on unseen Days 15-20."""
    valid_train = [
        x for x in history
        if int(x["day"]) < EVALUATION_START_DAY
        and x.get("is_valid", True)
        and x.get("severity_score") is not None
    ]

    if len(valid_train) < 2:
        # Fallback if too few valid training days
        last_val = valid_train[-1]["severity_score"] if valid_train else 0.0
        return {
            "training_days": [x["day"] for x in valid_train],
            "unseen_evaluation_days": [],
            "held_out_predictions": [],
            "unseen_mae": None,
            "prediction_horizon_days": 7,
            "predicted_future_day": 27,
            "predicted_future_severity": round(last_val, 3),
            "predicted_future_condition": "Pending Validation" if not valid_train else ("Severe" if last_val >= 0.65 else "Moderate" if last_val >= 0.35 else "Healthy"),
            "status": "insufficient_train_data" if len(valid_train) == 0 else "single_point_estimate",
        }

    x = np.array([float(item["day"]) for item in valid_train], dtype=np.float32)
    y = np.array([float(item["severity_score"]) for item in valid_train], dtype=np.float32)
    slope, intercept = np.polyfit(x, y, 1)

    heldout = []
    errors = []
    for item in history:
        day = int(item["day"])
        if day >= EVALUATION_START_DAY:
            predicted = float(np.clip(slope * day + intercept, 0.0, 1.0))
            record = {
                "day": day,
                "predicted_severity": round(predicted, 3),
                "is_valid_actual": item.get("is_valid", True),
            }
            if item.get("is_valid", True) and item.get("severity_score") is not None:
                actual = float(item["severity_score"])
                record["actual_severity"] = round(actual, 3)
                errors.append(abs(actual - predicted))
            heldout.append(record)

    future_day = 27
    future_severity = float(np.clip(slope * future_day + intercept, 0.0, 1.0))
    future_condition = (
        "Severe Pothole Risk"
        if future_severity >= 0.65
        else "Moderate Deterioration"
        if future_severity >= 0.35
        else "Acceptable"
    )

    mae = round(float(np.mean(errors)), 3) if errors else None

    return {
        "training_days": [int(item["day"]) for item in valid_train],
        "unseen_evaluation_days": [int(item["day"]) for item in history if int(item["day"]) >= EVALUATION_START_DAY],
        "held_out_predictions": heldout,
        "unseen_mae": mae,
        "slope_per_day": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "prediction_horizon_days": 7,
        "predicted_future_day": future_day,
        "predicted_future_severity": round(future_severity, 3),
        "predicted_future_condition": future_condition,
        "status": "valid_forecast",
    }


def generate_work_orders_for_inspections(
    segment_list: List[Dict[str, Any]],
    threshold: float = 0.65,
) -> List[Dict[str, Any]]:
    """Generate work orders when observed current OR predicted future severity meets threshold."""
    orders = []
    for segment in segment_list:
        seg_id = segment["road_segment_id"]
        current = segment.get("current", {})
        forecast = segment.get("prediction", {})

        current_sev = current.get("severity_score")
        future_sev = forecast.get("predicted_future_severity")

        # Determine trigger
        trigger_sev = max(
            current_sev if current_sev is not None else 0.0,
            future_sev if future_sev is not None else 0.0,
        )

        water_hazard = bool(current.get("has_water_hazard", False))

        # Check if threshold reached
        if trigger_sev < threshold and not water_hazard:
            continue

        trigger_day = current.get("day", 20)
        evidence = current.get("detections", [])
        primary = max(evidence, key=lambda d: float(d.get("severity_score", 0.0)), default={})

        tier = severity_tier(trigger_sev)
        defect_class = primary.get("defect_type") or current.get("condition", "pothole").lower().replace(" ", "_")

        fallback = generate_fallback_work_order(
            defect_class=defect_class,
            severity_tier=tier,
            severity_score=trigger_sev,
            area_m2=primary.get("area_m2", 2.5),
            estimated_depth_m=primary.get("estimated_depth_m", 0.08),
            is_water_filled=water_hazard,
            road_segment_id=seg_id,
            pothole_id=primary.get("pothole_id", f"{seg_id}-defect"),
        )

        priority = (
            "P1 - Immediate"
            if water_hazard or trigger_sev >= 0.85
            else "P2 - High"
            if trigger_sev >= threshold
            else "P3 - Planned"
        )

        orders.append({
            "work_order_id": f"WO-INSPECT-{seg_id}",
            "dispatch_status": "DISPATCHED",
            "dispatch_target": "municipal_maintenance_queue",
            "trigger_day": trigger_day,
            "road_segment_id": seg_id,
            "location": segment.get("location", {}),
            "detected_defects": [d.get("defect_type", "pothole") for d in evidence] or ["surface_defect"],
            "bounding_box_evidence": [
                {"bbox_xyxy": d.get("bbox_xyxy"), "image": current.get("overlay_path", "")}
                for d in evidence
            ],
            "current_severity_score": round(current_sev, 3) if current_sev is not None else None,
            "predicted_future_severity": round(future_sev, 3) if future_sev is not None else None,
            "priority": priority,
            "recommended_maintenance_action": fallback["work_order_text"],
            **fallback,
        })
    return orders


def ingest_all_manual_inspections(
    inspections_dir: Path,
    output_dir: Path,
    segment_ids: Optional[List[str]] = None,
    work_order_threshold: float = 0.65,
    device: str = CONFIG.device,
    force: bool = False,
    rebuild_dashboard: bool = True,
) -> Dict[str, Any]:
    """Execute complete ingestion across Days 1-20 and SEG_001-SEG_008."""
    inspections_dir = Path(inspections_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses_dir = inspections_dir / "analysis"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    target_segment_ids = segment_ids or [s.segment_id for s in ALL_SEGMENT_SPECS]
    spec_by_id = {s.segment_id: s for s in ALL_SEGMENT_SPECS}

    log.info(f"Target Segments ({len(target_segment_ids)}): {', '.join(target_segment_ids)}")
    log.info(f"Inspections Directory: {inspections_dir}")
    log.info(f"Dashboard Output Directory: {output_dir}")

    # Initialize segments
    all_segments: Dict[str, Dict[str, Any]] = {}
    for seg_id in target_segment_ids:
        spec = spec_by_id.get(seg_id, SegmentSpec(seg_id, 35.0, 0.0, "custom", f"Road segment {seg_id}"))
        along = spec.along_m
        across = spec.across_m
        all_segments[seg_id] = {
            "road_segment_id": seg_id,
            "profile": spec.profile,
            "description": spec.description,
            "persistent_location": {"along_m": along, "across_m": across},
            "location": gps_for_world(along, across),
            "history": [],
        }

    # Pipeline container (loaded lazily on first valid image)
    ml_pipeline = None

    valid_frames_count = 0
    rejected_frames_count = 0
    missing_frames_count = 0

    for day in range(1, 21):
        day_dir = inspections_dir / f"day_{day:02d}"

        for seg_id in target_segment_ids:
            # Check segment-wise layout: env/output/temporal_segments/SEG_XXX/day_XX.png
            alt_raw_path = inspections_dir / seg_id / f"day_{day:02d}.png"
            alt_history_path = inspections_dir / seg_id / "segment_history.json"

            if alt_raw_path.is_file():
                raw_path = alt_raw_path
                meta_path = alt_history_path
            else:
                seg_dir = day_dir / seg_id
                raw_path = seg_dir / "raw.png"
                meta_path = seg_dir / "metadata.json"

            analysis_seg_dir = analyses_dir / seg_id / f"day_{day:02d}"
            analysis_seg_dir.mkdir(parents=True, exist_ok=True)

            telemetry_path = analysis_seg_dir / "frame_metadata.json"
            result_json_path = analysis_seg_dir / "result.json"
            overlay_path = analysis_seg_dir / "detection_overlay.jpg"
            severity_overlay_path = analysis_seg_dir / "severity_overlay.jpg"
            health_overlay_path = analysis_seg_dir / "road_health_overlay.jpg"

            # Check if capture exists on disk
            if not raw_path.is_file():
                missing_frames_count += 1
                obs = {
                    "day": day,
                    "is_valid": False,
                    "validation_error": f"No inspection capture found for {seg_id} on Day {day:02d}",
                    "condition": "No Capture",
                    "severity_score": None,
                    "road_health_score": None,
                    "detections": [],
                    "detection_count": 0,
                    "has_water_hazard": False,
                    "image_path": str(raw_path.resolve()),
                    "overlay_path": str(overlay_path.resolve()),
                    "weather_preset": "Unrecorded",
                    "ml_anomaly_score": None,
                    "ml_anomaly_threshold": None,
                }
                # Diagnostic placeholder
                placeholder = create_diagnostic_error_overlay(
                    None, seg_id, day, "Unrecorded", obs["validation_error"]
                )
                cv2.imwrite(str(overlay_path), placeholder)
                write_json(result_json_path, obs)
                all_segments[seg_id]["history"].append(obs)
                continue

            # Load metadata
            meta = {}
            if meta_path.is_file():
                try:
                    meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta_path.name == "segment_history.json":
                        hist_entries = meta_raw.get("history", [])
                        matching = next((e for e in hist_entries if int(e.get("day", -1)) == day), {})
                        meta = matching
                    else:
                        meta = meta_raw
                except Exception:
                    pass

            coords = meta.get("world_coordinates", {})
            camera_pose = meta.get("camera_pose", {})
            along_m = float(coords.get("along_m", camera_pose.get("x_cm", 0.0) / 100.0 if "x_cm" in camera_pose else all_segments[seg_id]["persistent_location"]["along_m"]))
            across_m = float(coords.get("across_m", camera_pose.get("y_cm", 0.0) / 100.0 if "y_cm" in camera_pose else all_segments[seg_id]["persistent_location"]["across_m"]))
            gps = gps_for_world(along_m, across_m)
            all_segments[seg_id]["location"] = gps

            weather = meta.get("weather_preset", "Unknown")

            # Write standard frame telemetry
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

            # Step 2 & 3: Validate input before inference
            is_valid, reject_reason, loaded_img = validate_inspection_image(raw_path)

            if not is_valid:
                rejected_frames_count += 1
                # Generate diagnostic error overlay
                diag_overlay = create_diagnostic_error_overlay(
                    loaded_img, seg_id, day, weather, reject_reason or "Input rejected"
                )
                cv2.imwrite(str(overlay_path), diag_overlay)
                cv2.imwrite(str(severity_overlay_path), diag_overlay)
                cv2.imwrite(str(health_overlay_path), diag_overlay)

                obs = {
                    "day": day,
                    "is_valid": False,
                    "validation_error": reject_reason,
                    "condition": "Capture Error",
                    "severity_score": None,
                    "road_health_score": None,
                    "detections": [],
                    "detection_count": 0,
                    "has_water_hazard": False,
                    "image_path": str(raw_path.resolve()),
                    "overlay_path": str(overlay_path.resolve()),
                    "weather_preset": weather,
                    "ml_anomaly_score": None,
                    "ml_anomaly_threshold": None,
                }
                write_json(result_json_path, obs)
                all_segments[seg_id]["history"].append(obs)
                continue

            # Frame is valid genuine image
            valid_frames_count += 1

            # Check for cached valid observation
            if (
                not force
                and result_json_path.is_file()
                and overlay_path.is_file()
                and severity_overlay_path.is_file()
                and health_overlay_path.is_file()
            ):
                try:
                    cached_obs = json.loads(result_json_path.read_text(encoding="utf-8"))
                    if cached_obs.get("is_valid") is True and cached_obs.get("severity_score") is not None:
                        cached_obs["image_path"] = str(raw_path.resolve())
                        cached_obs["overlay_path"] = str(overlay_path.resolve())
                        all_segments[seg_id]["history"].append(cached_obs)
                        continue
                except Exception:
                    pass

            # Lazy-load pipeline on first valid execution
            if ml_pipeline is None:
                log.info(f"Loading RoadSentinel DINOv2 / SAM2 pipeline on {device}...")
                t_load = time.time()
                ml_pipeline = load_pipeline(device=device, memory_bank_dir=CONFIG.memory_bank_dir)
                log.info(f"ML Pipeline ready in {time.time() - t_load:.2f}s")

            # Run ML inference
            t_infer = time.time()
            result = infer(
                raw_path,
                metadata_path=telemetry_path,
                pipeline=ml_pipeline,
                road_segment_id=seg_id,
                test_mode_2d=False,
            )

            rgb = load_rgb(raw_path)
            overlays = generate_visual_overlays(rgb, result.potholes, result.road_health)

            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlays["detection_overlay"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(severity_overlay_path), cv2.cvtColor(overlays["severity_overlay"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(health_overlay_path), cv2.cvtColor(overlays["road_health_overlay"], cv2.COLOR_RGB2BGR))

            detections = normalized_detections(result, raw_path, overlay_path)
            max_severity = max((float(d.get("severity_score", 0.0)) for d in detections), default=0.0)
            defect_type = max(detections, key=lambda d: float(d.get("severity_score", 0.0)), default={}).get("defect_type")

            has_water = any(bool(d.get("water_flag")) for d in detections)
            cond = condition_label(max_severity, defect_type, has_water)

            obs = {
                "day": day,
                "is_valid": True,
                "validation_error": None,
                "condition": cond,
                "severity_score": round(max_severity, 3),
                "road_health_score": round(result.road_health.road_health_score, 2),
                "detections": detections,
                "detection_count": len(detections),
                "has_water_hazard": has_water,
                "image_path": str(raw_path.resolve()),
                "overlay_path": str(overlay_path.resolve()),
                "weather_preset": weather,
                "ml_anomaly_score": round(float(result.anomaly_score), 4),
                "ml_anomaly_threshold": round(float(result.anomaly_threshold), 4),
                "calibration": [x for x in result.warnings if x.startswith("Domain-adaptive")],
            }
            write_json(result_json_path, obs)
            all_segments[seg_id]["history"].append(obs)
            log.info(
                f"  [ML Ingest] Day {day:02d} {seg_id} ({weather[:15]}): "
                f"Sev={max_severity:.3f} Health={result.road_health.road_health_score:.1f} "
                f"Detections={len(detections)} [{time.time() - t_infer:.2f}s]"
            )

    log.info(
        f"Ingestion complete: {valid_frames_count} valid frames, "
        f"{rejected_frames_count} rejected frames, {missing_frames_count} unrecorded."
    )

    # Step 4: Persistent history & trend forecasting per segment
    segment_list = list(all_segments.values())
    for segment in segment_list:
        segment["prediction"] = predict_held_out_robust(segment["history"])
        # Find latest observation with a valid capture if available, else latest day
        valid_obs = [x for x in segment["history"] if x.get("is_valid")]
        segment["current"] = valid_obs[-1] if valid_obs else (segment["history"][-1] if segment["history"] else {})

    # Step 5: Work order generation
    work_orders = generate_work_orders_for_inspections(segment_list, work_order_threshold)
    work_order_by_seg = {w["road_segment_id"]: w for w in work_orders}
    for segment in segment_list:
        segment["work_order"] = work_order_by_seg.get(segment["road_segment_id"])

    # Unique weather presets
    all_weathers = sorted(list({
        obs.get("weather_preset", "Unknown")
        for seg in segment_list
        for obs in seg.get("history", [])
        if obs.get("weather_preset") and obs["weather_preset"] != "Unrecorded"
    }))

    result_dataset = {
        "schema_version": "RoadSentinelTemporal20Day/v1",
        "metadata": {
            "days": 20,
            "map_name": "RoadSentinelSim / SimBlank",
            "execution": "user-captured Unreal Engine inspection ingestion with input validation & domain-adaptive ML",
            "capture_source": "Unreal Engine SceneCapture2D inspection frames",
            "model_training_days": list(range(1, EVALUATION_START_DAY)),
            "unseen_evaluation_days": list(range(EVALUATION_START_DAY, 21)),
            "work_order_threshold": work_order_threshold,
            "weather_presets_used": all_weathers,
            "validation_summary": {
                "valid_frames": valid_frames_count,
                "rejected_frames": rejected_frames_count,
                "missing_frames": missing_frames_count,
                "total_expected": 20 * len(target_segment_ids),
            },
            "ground_truth_policy": "Strict ML observation: no Unreal condition manifests used for detections or severity.",
        },
        "segments": segment_list,
        "work_orders": work_orders,
        "overall_health": {
            "segments_deteriorated": [
                s["road_segment_id"] for s in segment_list
                if s["current"].get("severity_score") is not None and s["current"]["severity_score"] >= 0.35
            ],
            "segments_healthy": [
                s["road_segment_id"] for s in segment_list
                if s["current"].get("severity_score") is not None and s["current"]["severity_score"] < 0.10
            ],
            "segments_uninspected": [
                s["road_segment_id"] for s in segment_list
                if s["current"].get("severity_score") is None
            ],
        },
    }

    # Step 7: Write final results in temporal-results format
    write_json(inspections_dir / "temporal_results.json", result_dataset)
    write_json(inspections_dir / "work_orders.json", {"metadata": result_dataset["metadata"], "work_orders": work_orders})

    if rebuild_dashboard:
        output_results_path = output_dir / "temporal_results.json"
        write_json(output_results_path, result_dataset)
        write_json(output_dir / "work_orders.json", {"metadata": result_dataset["metadata"], "work_orders": work_orders})

        # Sync analysis directory
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
                        for f in day_subdir.iterdir():
                            if f.is_file():
                                shutil.copy2(f, dest / f.name)

        log.info(f"Dashboard dataset synced to: {output_results_path}")

    return result_dataset


def main():
    default_dir = WORKSPACE_ROOT / "env" / "output" / "temporal_segments"
    if not default_dir.exists():
        default_dir = WORKSPACE_ROOT / "env" / "output" / "manual_inspections"

    parser = argparse.ArgumentParser(description="Ingest manual Unreal inspection captures for Days 1-20 and SEG_001-SEG_008.")
    parser.add_argument(
        "--inspections-dir",
        type=Path,
        default=default_dir,
        help="Path to manual inspections directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "env" / "output" / "temporal_20_day",
        help="Path where temporal_results.json is served to the dashboard",
    )
    parser.add_argument(
        "--work-order-threshold",
        type=float,
        default=0.65,
        help="Severity score threshold to trigger maintenance work orders",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CONFIG.device,
        help="Compute device (cuda / cpu)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run inference on valid captures",
    )
    parser.add_argument(
        "--no-rebuild-dashboard",
        action="store_true",
        help="Skip syncing to temporal_20_day dashboard directory",
    )
    args = parser.parse_args()

    result = ingest_all_manual_inspections(
        inspections_dir=args.inspections_dir,
        output_dir=args.output_dir,
        work_order_threshold=args.work_order_threshold,
        device=args.device,
        force=args.force,
        rebuild_dashboard=not args.no_rebuild_dashboard,
    )

    print("\n=======================================================")
    print("  RoadSentinel Manual Inspection Ingestion Summary")
    print("=======================================================")
    summary = result["metadata"]["validation_summary"]
    print(f"  Valid Captures Processed:    {summary['valid_frames']}")
    print(f"  Rejected / Blank Frames:     {summary['rejected_frames']}")
    print(f"  Unrecorded Frames:           {summary['missing_frames']}")
    print(f"  Total Work Orders Generated: {len(result['work_orders'])}")
    print(f"  Segments in Dataset:         {len(result['segments'])}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
