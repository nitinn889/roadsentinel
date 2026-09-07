#!/usr/bin/env python3
"""Convert an existing RoadSentinel InferenceResult JSON to the shared record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from feature_contract import SCHEMA_VERSION, save_feature_record


def parse_temporal_image_path(path: Path | str) -> tuple[str, int, str]:
    """Parse ``SEG_001/day_01/original.jpg`` without changing the filename."""
    image_path = Path(path)
    day_dir = image_path.parent.name
    segment_id = image_path.parent.parent.name
    if not day_dir.startswith("day_"):
        raise ValueError("Temporal image parent must be named day_01 through day_10")
    try:
        day = int(day_dir.removeprefix("day_"))
    except ValueError as exc:
        raise ValueError("Temporal image parent must be named day_01 through day_10") from exc
    if not 1 <= day <= 10 or day_dir != f"day_{day:02d}" or not segment_id:
        raise ValueError("Expected temporal path SEGMENT_ID/day_01..day_10/original_filename")
    return segment_id, day, image_path.name


def _normalised_severity(value: Any) -> Optional[float]:
    if value is None:
        return None
    score = float(value)
    if 0.0 <= score <= 1.0:
        return score
    if 1.0 < score <= 100.0:
        return score / 100.0
    raise ValueError("Existing severity must use either the [0,1] or [0,100] scale")


def _image_pixels(inference: Mapping[str, Any]) -> Optional[int]:
    shape = inference.get("image_shape")
    if isinstance(shape, list) and len(shape) >= 2:
        height, width = shape[:2]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
            return height * width
    return None


def build_feature_record(
    inference: Mapping[str, Any],
    *,
    segment_id: str,
    day: int,
    original_filename: str,
    condition: Optional[str] = None,
    defect_masks: Optional[list[np.ndarray]] = None,
    mask_paths: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build a record using only values present in, or derivable from, inference.

    When final masks are supplied, geometry and total ratios are derived from
    those masks in original-image pixel space. Their union prevents overlap
    from being counted more than once.
    """
    detections = inference.get("detections")
    if not isinstance(detections, list):
        detections = inference.get("potholes")
    if not isinstance(detections, list):
        detections = []
    if defect_masks is not None and len(defect_masks) != len(detections):
        raise ValueError("defect_masks length must equal serialized detections length")
    if mask_paths is not None and len(mask_paths) != len(detections):
        raise ValueError("mask_paths length must equal serialized detections length")

    image_pixels = _image_pixels(inference)
    exported_defects: list[dict[str, Any]] = []
    area_total = 0
    crack_area_total = 0
    severity_values: list[float] = []
    water_values: list[bool] = []
    area_complete = True
    defect_type_complete = True
    severity_complete = True
    water_complete = True
    union_mask: Optional[np.ndarray] = None
    crack_union_mask: Optional[np.ndarray] = None

    for detection_index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            continue
        defect: dict[str, Any] = {}
        bbox = detection.get("bbox", detection.get("bbox_xyxy"))
        if isinstance(bbox, list) and len(bbox) == 4:
            defect["bbox"] = bbox
        mask_path = mask_paths[detection_index] if mask_paths is not None else detection.get("mask_path")
        if isinstance(mask_path, str) and mask_path:
            defect["mask_path"] = mask_path
        for name in ("centroid_x", "centroid_y"):
            if isinstance(detection.get(name), (int, float)) and not isinstance(detection.get(name), bool):
                defect[name] = detection[name]
        defect_type = detection.get("defect_type")
        if isinstance(defect_type, str) and defect_type:
            defect["defect_type"] = defect_type
        else:
            defect_type_complete = False

        final_mask: Optional[np.ndarray] = None
        if defect_masks is not None:
            final_mask = np.asarray(defect_masks[detection_index], dtype=bool)
            shape = inference.get("image_shape")
            if not isinstance(shape, list) or len(shape) < 2 or final_mask.shape != tuple(shape[:2]):
                raise ValueError("Every final mask must match the original image height and width")
            ys, xs = np.nonzero(final_mask)
            if xs.size:
                defect["bbox"] = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ]
                defect["centroid_x"] = float(xs.mean())
                defect["centroid_y"] = float(ys.mean())
            union_mask = final_mask.copy() if union_mask is None else (union_mask | final_mask)

        mask_pixels = (
            int(final_mask.sum()) if final_mask is not None
            else detection.get("mask_area_pixels", detection.get("mask_area_px"))
        )
        if isinstance(mask_pixels, int) and not isinstance(mask_pixels, bool) and mask_pixels >= 0:
            area_total += mask_pixels
            if isinstance(defect_type, str) and "crack" in defect_type.lower():
                crack_area_total += mask_pixels
                if final_mask is not None:
                    crack_union_mask = (
                        final_mask.copy()
                        if crack_union_mask is None
                        else (crack_union_mask | final_mask)
                    )
            if image_pixels:
                defect["area_ratio"] = min(1.0, mask_pixels / image_pixels)
        else:
            area_complete = False

        severity = detection.get("severity_score")
        nested_severity = detection.get("severity")
        if severity is None and isinstance(nested_severity, Mapping):
            severity = nested_severity.get("severity_score")
        if severity is not None:
            severity_values.append(_normalised_severity(severity))
        else:
            severity_complete = False

        water = detection.get("water_flag", detection.get("is_water_filled"))
        if isinstance(water, bool):
            water_values.append(water)
        else:
            water_complete = False
        exported_defects.append(defect)

    anomaly = inference.get("anomaly_score")
    anomaly_score = float(anomaly) if isinstance(anomaly, (int, float)) and not isinstance(anomaly, bool) else None
    if anomaly_score is not None and not 0.0 <= anomaly_score <= 1.0:
        raise ValueError("anomaly_score must be in [0,1]")

    image_id = inference.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        image_id = Path(original_filename).stem

    if image_pixels and union_mask is not None:
        defect_area_ratio: Optional[float] = float(union_mask.sum() / image_pixels)
        crack_area_ratio: Optional[float] = (
            float(crack_union_mask.sum() / image_pixels)
            if crack_union_mask is not None
            else (0.0 if defect_type_complete else None)
        )
    else:
        defect_area_ratio = (
            min(1.0, area_total / image_pixels)
            if image_pixels and area_complete
            else (0.0 if image_pixels and not exported_defects else None)
        )
        crack_area_ratio = (
            min(1.0, crack_area_total / image_pixels)
            if image_pixels and area_complete and defect_type_complete
            else (0.0 if image_pixels and not exported_defects else None)
        )

    record = {
        "schema_version": SCHEMA_VERSION,
        "image_id": image_id,
        "segment_id": segment_id,
        "day": day,
        "original_filename": original_filename,
        "condition": condition,
        "current_severity": (
            max(severity_values) if severity_values and severity_complete
            else (0.0 if not exported_defects else None)
        ),
        "crack_area_ratio": crack_area_ratio,
        "defect_area_ratio": defect_area_ratio,
        "defect_count": len(exported_defects),
        "surface_anomaly_score": anomaly_score,
        "water_flag": (
            any(water_values) if water_values and water_complete
            else (False if not exported_defects else None)
        ),
        "defects": exported_defects,
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--source-image", type=Path, help="Path following SEG_ID/day_01/file convention")
    parser.add_argument("--segment-id")
    parser.add_argument("--day", type=int)
    parser.add_argument("--original-filename")
    parser.add_argument("--condition", help="Observed capture condition; omit when not verified")
    args = parser.parse_args()

    inference = json.loads(args.inference_json.read_text(encoding="utf-8"))
    if args.source_image:
        segment_id, day, original_filename = parse_temporal_image_path(args.source_image)
    else:
        if args.segment_id is None or args.day is None or args.original_filename is None:
            parser.error("provide --source-image or all of --segment-id, --day, --original-filename")
        segment_id, day, original_filename = args.segment_id, args.day, args.original_filename
    record = build_feature_record(
        inference,
        segment_id=segment_id,
        day=day,
        original_filename=original_filename,
        condition=args.condition,
    )
    save_feature_record(record, args.output_json)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
