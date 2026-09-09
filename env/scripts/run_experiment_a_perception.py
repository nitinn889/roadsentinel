#!/usr/bin/env python3
"""run_experiment_a_perception.py
---------------------------------
RoadSentinel Phase 2 — Experiment A Perception Processing.

Executes the frozen Marion Day-5 NORMAL DINOv2 + SAM2 perception pipeline
over the 40 physical captures (39 primary metadata-complete + 1 diagnostic
image-only capture for SEG_003 Day 10).

ABSOLUTE RESTRICTIONS:
- NEVER touches Unreal Engine / CARLA / simulation.
- FILE ACCESS ONLY over existing captures in env/output/temporal_segments/.
- NO model tuning, NO threshold adjustments, NO memory bank modifications.
- NO 2x tiled DINOv2 mode (rejected in Day-5 GPU follow-up).
- NO YOLO, NO XGBoost, NO temporal tracking across days.
- Raw captures are IMMUTABLE.

Outputs:
- integration/experiment_a/perception/<segment_id>/day_<day:02d>/
    - features.json
    - overlay.png
    - masks/
        - defect_001_mask.png, ...
    - inference_metadata.json
    - diagnostics.json
- integration/experiment_a/perception_results.csv
- integration/experiment_a/PHASE2_PROGRESS.md
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = ROOT / "road_health_pipeline"
SAM2_DINO_ROOT = ROOT / "sam2_dino"

for p in (str(ROOT), str(PIPELINE_ROOT), str(SAM2_DINO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.io_utils import load_rgb
from current_condition import describe_scores, mask_geometry
from export_features import build_feature_record
from feature_contract import SCHEMA_VERSION, validate_feature_record, save_feature_record
from inference.run_inference import infer, load_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("phase2_perception")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise OSError(f"Could not write mask: {path}")


def process_single_capture(
    *,
    image_path: Path,
    output_dir: Path,
    pipeline: Any,
    segment_id: str,
    day: int,
    meta_row: Dict[str, str],
    device_name: str,
) -> Dict[str, Any]:
    """Process a single physical image through the frozen DINOv2 + SAM2 pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_id = f"{segment_id}_day_{day:02d}"
    is_missing_metadata = (meta_row.get("metadata_path") == "MISSING" or
                           meta_row.get("sidecar_valid", "").lower() != "true")

    # The perception model receives ONLY the image; simulation metadata is NOT fed into model
    diagnostics: Dict[str, Any] = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = infer(
        image_path,
        pipeline=pipeline,
        road_segment_id=segment_id,
        test_mode_2d=True,
        diagnostics=diagnostics,
    )

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    rgb = load_rgb(image_path)
    height, width = rgb.shape[:2]

    # Process candidates and final SAM2 masks
    candidates = diagnostics.get("candidates", [])
    final_masks: List[np.ndarray] = []
    mask_rel_paths: List[str] = []
    geometries: List[Dict[str, Any]] = []
    union_mask = np.zeros((height, width), dtype=bool)

    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tint = np.zeros_like(overlay)

    for idx, candidate in enumerate(candidates, start=1):
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {(height, width)}")

        mask_name = f"defect_{idx:03d}_mask.png"
        mask_file = masks_dir / mask_name
        _write_mask(mask_file, mask)

        rel_mask_path = f"masks/{mask_name}"
        final_masks.append(mask)
        mask_rel_paths.append(rel_mask_path)
        union_mask |= mask

        geom = mask_geometry(mask)
        geometries.append(geom)

        if geom["bbox"] is not None:
            x1, y1, x2, y2 = geom["bbox"]
            cx = int(round(geom["centroid_x"]))
            cy = int(round(geom["centroid_y"]))

            # Draw bounding box and centroid
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)

            # Label above bounding box
            label_text = f"#{idx:02d} road_defect"
            label_y = max(18, y1 - 6)
            cv2.putText(
                overlay,
                label_text,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        tint[mask] = (0, 0, 255)

    if np.any(union_mask):
        overlay[union_mask] = cv2.addWeighted(
            overlay[union_mask], 0.65, tint[union_mask], 0.35, 0.0
        )

    # Build schema-compliant feature record
    feature_record = build_feature_record(
        result.to_dict(),
        segment_id=segment_id,
        day=day,
        original_filename=image_path.name,
        condition=None,  # Not fed into model; kept unconditioned
        defect_masks=final_masks,
        mask_paths=mask_rel_paths,
    )
    feature_record["image_id"] = image_id

    # Annotate defect types explicitly as 'road_defect'
    for defect in feature_record["defects"]:
        if not defect.get("defect_type"):
            defect["defect_type"] = "road_defect"

    # Add overlay status banner
    defect_count = feature_record["defect_count"]
    curr_sev = feature_record["current_severity"] or 0.0
    anom_score = feature_record["surface_anomaly_score"] or 0.0
    defect_area_ratio = feature_record["defect_area_ratio"] or 0.0

    banner_h = 36
    cv2.rectangle(overlay, (0, 0), (width, banner_h), (25, 25, 25), -1)
    if defect_count > 0:
        banner_text = (
            f"{segment_id} Day {day:02d} | Defects: {defect_count} | "
            f"Severity: {curr_sev:.4f} | Area Ratio: {defect_area_ratio:.4f} | "
            f"Anomaly: {anom_score:.4f}"
        )
        banner_color = (50, 220, 255)
    else:
        banner_text = (
            f"{segment_id} Day {day:02d} | Zero Defects Detected | "
            f"Severity: 0.0000 | Anomaly: {anom_score:.4f}"
        )
        banner_color = (120, 255, 120)

    cv2.putText(
        overlay,
        banner_text,
        (15, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        banner_color,
        2,
        cv2.LINE_AA,
    )

    overlay_path = output_dir / "overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not write overlay: {overlay_path}")

    # Validate against frozen schema v1.0.0
    validate_feature_record(feature_record)
    save_feature_record(feature_record, output_dir / "features.json")

    # Save diagnostic JSON
    localization = diagnostics.get("localization", {})
    candidate_mask = np.asarray(localization.get("candidate_mask", np.zeros((height, width), dtype=bool)), dtype=bool)
    diagnostic_record = {
        "image_id": image_id,
        "segment_id": segment_id,
        "day": day,
        "original_filename": image_path.name,
        "image_shape": [height, width, int(rgb.shape[2])],
        "road_mask_ratio": diagnostics.get("road_mask_ratio"),
        "road_mask": diagnostics.get("road_mask_diagnostics"),
        "raw_anomaly_scores": describe_scores(diagnostics.get("raw_patch_scores", np.array([]))),
        "normalized_anomaly_scores": describe_scores(diagnostics.get("normalized_patch_scores", np.array([]))),
        "normalization": diagnostics.get("normalization"),
        "threshold_selection": diagnostics.get("threshold_selection"),
        "threshold": diagnostics.get("threshold"),
        "threshold_mask_ratio": float(np.mean(localization.get("threshold_mask", np.zeros((1, 1))))),
        "candidate_mask_ratio": float(np.mean(candidate_mask)),
        "connected_component_count_before_filters": localization.get("connected_component_count_before_filters", 0),
        "accepted_candidate_count_after_filters": len(candidates),
        "sam2_refined_count": int(sum(c.sam2_result is not None for c in candidates)),
        "sam2_prompts": localization.get("sam2_prompts"),
        "final_mask_geometry": geometries,
        "union_area_pixels": int(union_mask.sum()),
        "union_area_ratio": float(union_mask.mean()),
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostic_record, indent=2) + "\n", encoding="utf-8"
    )

    # Save preserved simulation metadata & execution record
    if is_missing_metadata:
        sim_conditions = {
            "lighting_preset": None,
            "road_health_state": None,
            "pothole_sizing_spectrum": None,
            "pothole_density_per_100m2": None,
            "pothole_moisture_state": None,
            "camera_preset": None,
            "capture_timestamp": None,
            "notes": "Missing metadata sidecar; processed image-only diagnostic",
        }
        metadata_status = "MISSING"
    else:
        density_str = meta_row.get("pothole_density_per_100m2", "")
        try:
            density_val = float(density_str) if density_str and density_str != "UNKNOWN" else None
        except ValueError:
            density_val = None

        sim_conditions = {
            "lighting_preset": meta_row.get("lighting_preset"),
            "road_health_state": meta_row.get("road_health_state"),
            "pothole_sizing_spectrum": meta_row.get("pothole_sizing_spectrum"),
            "pothole_density_per_100m2": density_val,
            "pothole_moisture_state": meta_row.get("pothole_moisture_state"),
            "camera_preset": meta_row.get("camera_preset"),
            "capture_timestamp": meta_row.get("capture_timestamp"),
        }
        metadata_status = "VALID"

    inference_metadata = {
        "image_id": image_id,
        "segment_id": segment_id,
        "day": day,
        "metadata_status": metadata_status,
        "simulation_conditions": sim_conditions,
        "inference_execution": {
            "pipeline": "frozen_marion_day5_normal_dinov2_sam2",
            "tiled_mode": False,
            "device": device_name,
            "inference_time_ms": round(elapsed_ms, 2),
            "defect_count": defect_count,
            "current_severity": curr_sev,
            "crack_area_ratio": feature_record["crack_area_ratio"],
            "defect_area_ratio": defect_area_ratio,
            "surface_anomaly_score": anom_score,
            "water_flag": feature_record["water_flag"],
            "schema_valid": True,
            "accepted_mask_count": len(final_masks),
        },
    }
    (output_dir / "inference_metadata.json").write_text(
        json.dumps(inference_metadata, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "image_id": image_id,
        "segment_id": segment_id,
        "day": day,
        "lighting_preset": sim_conditions["lighting_preset"] or "UNKNOWN",
        "road_health_state": sim_conditions["road_health_state"] or "UNKNOWN",
        "camera_preset": sim_conditions["camera_preset"] or "UNKNOWN",
        "moisture_state": sim_conditions["pothole_moisture_state"] or "UNKNOWN",
        "density": sim_conditions["pothole_density_per_100m2"] if sim_conditions["pothole_density_per_100m2"] is not None else "",
        "current_severity": curr_sev,
        "defect_count": defect_count,
        "crack_area_ratio": feature_record["crack_area_ratio"] or 0.0,
        "defect_area_ratio": defect_area_ratio,
        "surface_anomaly_score": anom_score,
        "water_flag": feature_record["water_flag"] if feature_record["water_flag"] is not None else False,
        "inference_time_ms": round(elapsed_ms, 2),
        "schema_valid": True,
        "metadata_status": metadata_status,
        "result_path": str(output_dir.relative_to(ROOT)),
    }


def run_experiment_a():
    inventory_path = ROOT / "integration/temporal/actual_capture_inventory.csv"
    output_root = ROOT / "integration/experiment_a/perception"
    results_csv_path = ROOT / "integration/experiment_a/perception_results.csv"

    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing inventory: {inventory_path}")

    with open(inventory_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        captures = list(reader)

    log.info("Loaded %d captures from %s", len(captures), inventory_path)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log.info("Using device: %s (%s)", device, device_name)

    log.info("Loading frozen Marion Day-5 NORMAL DINOv2 + SAM2 pipeline...")
    pipeline = load_pipeline(device=device)
    log.info("Pipeline loaded successfully.")

    processed_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    total_t0 = time.perf_counter()

    for idx, row in enumerate(captures, start=1):
        seg = row["segment_id"]
        day = int(row["day"])
        img_rel = row["image_path"]
        img_path = ROOT / img_rel
        out_dir = output_root / seg / f"day_{day:02d}"

        log.info("[%02d/%02d] Processing %s Day %02d (%s)", idx, len(captures), seg, day, img_path.name)
        if not img_path.exists():
            log.error("Image file missing: %s", img_path)
            failures.append({"segment_id": seg, "day": day, "error": "file_not_found"})
            continue

        try:
            res = process_single_capture(
                image_path=img_path,
                output_dir=out_dir,
                pipeline=pipeline,
                segment_id=seg,
                day=day,
                meta_row=row,
                device_name=device_name,
            )
            processed_rows.append(res)
            log.info(
                "  Done in %.1f ms | defects=%d | severity=%.4f | anomaly=%.4f",
                res["inference_time_ms"],
                res["defect_count"],
                res["current_severity"],
                res["surface_anomaly_score"],
            )
        except Exception as exc:
            log.error("  FAILED processing %s Day %02d: %s", seg, day, exc, exc_info=True)
            failures.append({"segment_id": seg, "day": day, "error": str(exc)})

    total_duration_s = time.perf_counter() - total_t0
    log.info(
        "Phase 2 inference complete: %d attempted, %d succeeded, %d failed. Total duration: %.2fs",
        len(captures),
        len(processed_rows),
        len(failures),
        total_duration_s,
    )

    # Write master perception results CSV
    fieldnames = [
        "image_id",
        "segment_id",
        "day",
        "lighting_preset",
        "road_health_state",
        "camera_preset",
        "moisture_state",
        "density",
        "current_severity",
        "defect_count",
        "crack_area_ratio",
        "defect_area_ratio",
        "surface_anomaly_score",
        "water_flag",
        "inference_time_ms",
        "schema_valid",
        "metadata_status",
        "result_path",
    ]
    with open(results_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in processed_rows:
            writer.writerow(r)
    log.info("Saved master perception results to %s", results_csv_path)

    return processed_rows, failures, total_duration_s, device_name


if __name__ == "__main__":
    run_experiment_a()
