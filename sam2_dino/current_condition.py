#!/usr/bin/env python3
"""Run the existing perception pipeline and persist Day 2 diagnostics/artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PIPELINE_ROOT = REPO_ROOT / "road_health_pipeline"
for entry in (str(HERE), str(PIPELINE_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from common.io_utils import load_rgb
from export_features import build_feature_record
from feature_contract import save_feature_record
from inference.run_inference import infer, load_pipeline


def describe_scores(scores: np.ndarray) -> dict[str, Any]:
    """Return bounded distribution diagnostics; never serialize score arrays."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25.0)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75.0)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "positive_count": int(np.count_nonzero(values > 1e-6)),
    }


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    """Measure one boolean mask in original-image pixel coordinates."""
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return {
            "bbox": None,
            "centroid_x": None,
            "centroid_y": None,
            "area_pixels": 0,
            "area_ratio": 0.0,
        }
    return {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "area_pixels": int(xs.size),
        "area_ratio": float(xs.size / mask_bool.size),
    }


def _write_mask(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise OSError(f"Could not write mask: {path}")


def persist_current_condition(
    *,
    image_path: Path,
    output_dir: Path,
    pipeline,
    segment_id: str,
    day: int,
    condition: str | None,
    camera_mode: str,
) -> dict[str, Any]:
    """Run one image and persist masks, geometry, diagnostics, and features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.masker.camera_mode = camera_mode
    diagnostics: dict[str, Any] = {}
    result = infer(
        image_path,
        pipeline=pipeline,
        road_segment_id=segment_id,
        test_mode_2d=True,
        diagnostics=diagnostics,
    )
    rgb = load_rgb(image_path)
    height, width = rgb.shape[:2]

    original_copy = output_dir / f"original{image_path.suffix}"
    shutil.copy2(image_path, original_copy)

    anomaly_map = np.asarray(diagnostics["anomaly_map"], dtype=np.float32)
    heatmap = cv2.applyColorMap(
        np.round(np.clip(anomaly_map, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    if not cv2.imwrite(str(output_dir / "anomaly_heatmap.png"), heatmap):
        raise OSError("Could not write anomaly heatmap")

    localization = diagnostics["localization"]
    candidate_mask = np.asarray(localization["candidate_mask"], dtype=bool)
    _write_mask(output_dir / "candidate_mask.png", candidate_mask)
    _write_mask(output_dir / "raw_road_mask.png", diagnostics["raw_road_mask"])
    _write_mask(output_dir / "road_mask.png", diagnostics["road_mask"])

    candidates = diagnostics["candidates"]
    if len(candidates) != len(result.detections):
        raise RuntimeError("Candidate/detection order mismatch; feature export stopped")

    final_masks: list[np.ndarray] = []
    mask_paths: list[str] = []
    geometries: list[dict[str, Any]] = []
    union_mask = np.zeros((height, width), dtype=bool)
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tint = np.zeros_like(overlay)

    for index, candidate in enumerate(candidates, start=1):
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError("Final SAM2/candidate mask is not in original-image pixel space")
        mask_name = f"defect_{index:03d}_mask.png"
        _write_mask(output_dir / mask_name, mask)
        final_masks.append(mask)
        mask_paths.append(mask_name)
        union_mask |= mask

        geometry = mask_geometry(mask)
        geometries.append(geometry)
        if geometry["bbox"] is not None:
            x1, y1, x2, y2 = geometry["bbox"]
            cx = int(round(geometry["centroid_x"]))
            cy = int(round(geometry["centroid_y"]))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)
        tint[mask] = (0, 0, 255)

    if np.any(union_mask):
        mask_pixels = union_mask
        overlay[mask_pixels] = cv2.addWeighted(
            overlay[mask_pixels], 0.65, tint[mask_pixels], 0.35, 0.0
        )
    _write_mask(output_dir / "union_mask.png", union_mask)
    if not cv2.imwrite(str(output_dir / "overlay.png"), overlay):
        raise OSError("Could not write overlay")

    feature_record = build_feature_record(
        result.to_dict(),
        segment_id=segment_id,
        day=day,
        original_filename=image_path.name,
        condition=condition,
        defect_masks=final_masks,
        mask_paths=mask_paths,
    )
    save_feature_record(feature_record, output_dir / "features.json")

    diagnostic_record = {
        "image_id": feature_record["image_id"],
        "segment_id": feature_record["segment_id"],
        "day": feature_record["day"],
        "original_filename": image_path.name,
        "image_shape": [height, width, int(rgb.shape[2])],
        "camera_mode": camera_mode,
        "road_mask_ratio": diagnostics["road_mask_ratio"],
        "road_mask": diagnostics["road_mask_diagnostics"],
        "raw_anomaly_scores": describe_scores(diagnostics["raw_patch_scores"]),
        "normalized_anomaly_scores": describe_scores(diagnostics["normalized_patch_scores"]),
        "normalization": diagnostics["normalization"],
        "threshold_selection": diagnostics["threshold_selection"],
        "threshold": diagnostics["threshold"],
        "threshold_mask_ratio": float(np.mean(localization["threshold_mask"])),
        "candidate_mask_ratio": float(np.mean(candidate_mask)),
        "connected_component_count_before_filters": localization["connected_component_count_before_filters"],
        "accepted_candidate_count_after_filters": len(candidates),
        "sam2_refined_count": int(sum(c.sam2_result is not None for c in candidates)),
        "sam2_prompts": localization["sam2_prompts"],
        "final_mask_geometry": geometries,
        "union_area_pixels": int(union_mask.sum()),
        "union_area_ratio": float(union_mask.mean()),
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostic_record, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "image_id": feature_record["image_id"],
        "segment_id": feature_record["segment_id"],
        "day": feature_record["day"],
        "original_filename": image_path.name,
        "output_dir": str(output_dir),
        "defect_count": feature_record["defect_count"],
        "sam2_refined_count": diagnostic_record["sam2_refined_count"],
        "threshold": diagnostic_record["threshold"],
        "road_mask_ratio": diagnostic_record["road_mask_ratio"],
        "candidate_mask_ratio": diagnostic_record["candidate_mask_ratio"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "fixed_test_set.json")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segment-id", default="SEG_DAY2_TEST")
    parser.add_argument("--day", type=int, default=1)
    parser.add_argument(
        "--memory-bank",
        type=Path,
        default=PIPELINE_ROOT / "output" / "real_memory_bank",
    )
    parser.add_argument("--only", help="Run only one manifest category")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if args.only:
        items = [item for item in items if item.get("category") == args.only]
    if not items:
        raise ValueError("No fixed test-set items selected")

    pipeline = load_pipeline(device=args.device, memory_bank_dir=args.memory_bank)
    summary = []
    for item in items:
        image_path = REPO_ROOT / item["image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        item_output = args.output_root / image_path.stem
        row = persist_current_condition(
            image_path=image_path,
            output_dir=item_output,
            pipeline=pipeline,
            segment_id=args.segment_id,
            day=args.day,
            condition=item.get("condition"),
            camera_mode=item["camera_mode"],
        )
        row["category"] = item["category"]
        row["evidence"] = item["evidence"]
        summary.append(row)
        print(
            f"{item['category']}: defects={row['defect_count']} "
            f"sam2={row['sam2_refined_count']} threshold={row['threshold']:.4f}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_summary.json").write_text(
        json.dumps({"test_set_status": manifest["test_set_status"], "items": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
