#!/usr/bin/env python3
"""run_sam2_dino.py
------------------
General-purpose batch and single-image inference runner for DINOv2 + SAM2.

Provides comparison-ready inference over arbitrary images, directories, or
manifest CSVs (e.g. benchmark/common_candidate/manifest.csv).

Preserves original filenames, computes mask-derived boxes, saves contract-compliant
features.json, individual defect masks, overlays, diagnostics, and machine-readable
timing in run_summary.json.

Usage:
    python run_sam2_dino.py --input <image-or-folder-or-manifest> --output <output-dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent          # sam2_dino/
REPO_ROOT = ROOT.parent                         # repository root (for manifest-relative paths)
PIPELINE_ROOT = REPO_ROOT / "road_health_pipeline"
SAM2_DINO_ROOT = ROOT

for p in (str(ROOT), str(PIPELINE_ROOT), str(SAM2_DINO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.io_utils import load_rgb
from current_condition import persist_current_condition
from feature_contract import validate_feature_record, load_feature_record
from inference.run_inference import load_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("run_sam2_dino")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images_from_input(input_path: Path) -> list[tuple[str, Path, Optional[str]]]:
    """Collect (image_id, image_path, condition) tuples from input."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    items: list[tuple[str, Path, Optional[str]]] = []

    # Case 1: CSV manifest (e.g. benchmark/common_candidate/manifest.csv)
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        log.info("Loading manifest CSV from %s", input_path)
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path_str = row.get("image_path") or row.get("path") or row.get("image")
                if not img_path_str:
                    continue
                resolved_path = (REPO_ROOT / img_path_str).resolve() if not Path(img_path_str).is_absolute() else Path(img_path_str)
                if not resolved_path.exists():
                    log.warning("Image path from manifest not found: %s", resolved_path)
                    continue
                image_id = row.get("image_id") or resolved_path.stem
                condition = row.get("condition") or None
                items.append((image_id, resolved_path, condition))
        return items

    # Case 2: Single image file
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [(input_path.stem, input_path.resolve(), None)]

    # Case 3: Directory of images
    if input_path.is_dir():
        log.info("Scanning directory for images: %s", input_path)
        found = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
        for p in found:
            items.append((p.stem, p.resolve(), None))
        return items

    raise ValueError(f"Unrecognized input target: {input_path}")


def run_inference_batch(
    *,
    input_path: Path,
    output_dir: Path,
    device: str = "cuda",
    camera_mode: str = "nadir",
    segment_id: str = "SEG_BENCHMARK",
    day: int = 1,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Run pipeline across all collected images and persist outputs."""
    image_items = collect_images_from_input(input_path)
    if not image_items:
        raise RuntimeError(f"No valid images found for input: {input_path}")

    if limit is not None and limit > 0:
        image_items = image_items[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Processing %d images with device=%s, camera_mode=%s", len(image_items), device, camera_mode)

    # Load models once
    pipeline = load_pipeline(device=device)

    summary_records: list[dict[str, Any]] = []
    start_all = time.perf_counter()

    for idx, (image_id, img_file, condition) in enumerate(image_items, start=1):
        log.info("[%d/%d] Processing %s (%s)", idx, len(image_items), image_id, img_file.name)
        img_out_dir = output_dir / image_id
        img_out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            summary = persist_current_condition(
                image_path=img_file,
                output_dir=img_out_dir,
                pipeline=pipeline,
                segment_id=segment_id,
                day=day,
                condition=condition,
                camera_mode=camera_mode,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Validate feature contract
            feat_rec = load_feature_record(img_out_dir / "features.json")
            validate_feature_record(feat_rec)

            diag_file = img_out_dir / "diagnostics.json"
            diag = json.loads(diag_file.read_text(encoding="utf-8")) if diag_file.exists() else {}

            # Extract mask-derived bounding boxes
            defect_boxes = [g["bbox"] for g in diag.get("final_mask_geometry", []) if g.get("bbox") is not None]

            item_summary = {
                "image_id": image_id,
                "original_filename": img_file.name,
                "output_dir": str(img_out_dir),
                "status": "SUCCESS",
                "inference_time_ms": round(elapsed_ms, 2),
                "defect_count": feat_rec["defect_count"],
                "current_severity": feat_rec["current_severity"],
                "defect_area_ratio": feat_rec["defect_area_ratio"],
                "threshold": diag.get("threshold"),
                "road_mask_ratio": diag.get("road_mask_ratio"),
                "defect_boxes_xyxy": defect_boxes,
            }
            summary_records.append(item_summary)
            log.info("Finished %s in %.1f ms | defects=%d | severity=%.4f", image_id, elapsed_ms, feat_rec["defect_count"], feat_rec["current_severity"])

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.error("Failed processing %s: %s", image_id, exc, exc_info=True)
            summary_records.append({
                "image_id": image_id,
                "original_filename": img_file.name,
                "output_dir": str(img_out_dir),
                "status": "FAILED",
                "error": str(exc),
                "inference_time_ms": round(elapsed_ms, 2),
            })

    total_time_s = time.perf_counter() - start_all
    avg_ms = (total_time_s * 1000.0 / len(image_items)) if image_items else 0.0

    run_summary = {
        "runner": "run_sam2_dino.py",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "device": device,
        "camera_mode": camera_mode,
        "total_images": len(image_items),
        "successful_images": sum(1 for r in summary_records if r.get("status") == "SUCCESS"),
        "failed_images": sum(1 for r in summary_records if r.get("status") == "FAILED"),
        "total_duration_s": round(total_time_s, 2),
        "mean_inference_time_ms": round(avg_ms, 2),
        "records": summary_records,
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    log.info("Saved run summary to %s (Mean time: %.1f ms/frame)", summary_path, avg_ms)
    return run_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DINOv2 + SAM2 inference on an image, folder, or benchmark manifest CSV."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to image, directory, or manifest.csv")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Destination directory for outputs")
    parser.add_argument("--device", default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--camera-mode", default="nadir", choices=["nadir", "forward"], help="Camera geometry mode")
    parser.add_argument("--segment-id", default="SEG_BENCHMARK", help="Segment identifier for feature contract")
    parser.add_argument("--day", type=int, default=1, help="Day index for feature contract")
    parser.add_argument("--limit", type=int, default=None, help="Optional max images to process")

    args = parser.parse_args()
    summary = run_inference_batch(
        input_path=args.input,
        output_dir=args.output,
        device=args.device,
        camera_mode=args.camera_mode,
        segment_id=args.segment_id,
        day=args.day,
        limit=args.limit,
    )
    print(json.dumps({
        "status": "COMPLETED",
        "total_images": summary["total_images"],
        "successful": summary["successful_images"],
        "mean_ms": summary["mean_inference_time_ms"],
        "summary_file": str(args.output / "run_summary.json")
    }, indent=2))
    return 0 if summary["failed_images"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
