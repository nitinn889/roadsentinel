"""Conservative one-to-one correspondence for adjacent daily defect records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MatchingConfig:
    mask_iou_min: float = 0.50
    bbox_iou_min: float = 0.30
    centroid_distance_max: float = 75.0
    area_ratio_max_change: float = 3.0


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return not (left.get("defect_type") and right.get("defect_type") and left["defect_type"] != right["defect_type"])


def _mask_iou(left_path: str, right_path: str) -> float | None:
    """Return mask IoU only for readable, shape-compatible stored masks."""
    try:
        import cv2
        import numpy as np
        left = cv2.imread(str(Path(left_path)), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(Path(right_path)), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None or left.shape != right.shape:
            return None
        left, right = left > 0, right > 0
        union = np.logical_or(left, right).sum()
        return float(np.logical_and(left, right).sum() / union) if union else 0.0
    except (ImportError, OSError):
        return None


def _candidate(left: dict[str, Any], right: dict[str, Any], config: MatchingConfig) -> tuple[str, float] | None:
    if not _compatible(left, right):
        return None
    lm, rm = left.get("mask_path"), right.get("mask_path")
    if lm and rm:
        score = _mask_iou(lm, rm)
        if score is not None and score >= config.mask_iou_min:
            return "mask_iou", score
    lb, rb = left.get("bbox"), right.get("bbox")
    if lb is not None and rb is not None:
        score = bbox_iou(lb, rb)
        if score >= config.bbox_iou_min:
            return "bbox_iou", score
    lx, ly, rx, ry = left.get("centroid_x"), left.get("centroid_y"), right.get("centroid_x"), right.get("centroid_y")
    if None not in (lx, ly, rx, ry):
        distance = ((float(lx) - float(rx)) ** 2 + (float(ly) - float(ry)) ** 2) ** 0.5
        if distance <= config.centroid_distance_max:
            la, ra = left.get("area_ratio"), right.get("area_ratio")
            if la not in (None, 0) and ra is not None and max(float(la), float(ra)) / min(float(la), float(ra)) > config.area_ratio_max_change:
                return None
            return "centroid_distance", 1.0 - distance / config.centroid_distance_max
    return None


def match_adjacent(previous: list[dict[str, Any]], current: list[dict[str, Any]], config: MatchingConfig = MatchingConfig()) -> list[dict[str, Any]]:
    """Return reliable greedy one-to-one matches; ambiguous/no-geometry defects remain unmatched."""
    candidates = []
    for old_index, old in enumerate(previous):
        for new_index, new in enumerate(current):
            candidate = _candidate(old, new, config)
            if candidate:
                method, score = candidate
                candidates.append((score, method, old_index, new_index))
    used_old, used_new, output = set(), set(), []
    for score, method, old_index, new_index in sorted(candidates, reverse=True):
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index); used_new.add(new_index)
        output.append({"previous_index": old_index, "current_index": new_index, "method": method, "confidence": round(float(score), 4)})
    return output
