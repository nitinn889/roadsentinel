#!/usr/bin/env python3
"""Build a small, deterministic synthetic robustness set from RDD2022 holdout images.

The transformations alter appearance only, so the original YOLO boxes remain valid.
They are photometric stress tests, not verified real-weather annotations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rdd2022"
OUT = ROOT / "test_conditions"
SAMPLES = [
    "China_Drone_000001.jpg",  # D00/D10
    "China_Drone_000004.jpg",  # D20/D10
    "China_Drone_000064.jpg",  # D40/Repair
    "China_Drone_000112.jpg",  # D00/D20
    "China_Drone_000160.jpg",  # D40/D20
    "China_Drone_001045.jpg",  # D00/D40
]
CONDITIONS = ("daylight", "low_light", "rain", "wet_road", "shadows", "fog_haze")


def transform(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "daylight":
        return image.copy()
    if condition == "low_light":
        return np.clip((image.astype(np.float32) / 255.0) ** 1.9 * 255.0, 0, 255).astype(np.uint8)
    if condition == "rain":
        canvas = cv2.convertScaleAbs(image, alpha=0.72, beta=-14)
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        for _ in range(170):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            length = int(rng.integers(10, 25))
            cv2.line(overlay, (x, y), (min(w - 1, x + 3), min(h - 1, y + length)), (185, 185, 185), 1)
        return cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)
    if condition == "wet_road":
        canvas = cv2.convertScaleAbs(image, alpha=0.70, beta=-18)
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        for _ in range(16):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            axes = (int(rng.integers(20, 80)), int(rng.integers(3, 12)))
            cv2.ellipse(overlay, (x, y), axes, int(rng.integers(-15, 15)), 0, 360, (190, 190, 190), -1)
        return cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
    if condition == "shadows":
        canvas = image.astype(np.float32)
        h, w = canvas.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        for _ in range(3):
            pts = np.array([[rng.integers(0, w), rng.integers(0, h)] for _ in range(5)], dtype=np.int32)
            cv2.fillConvexPoly(mask, pts, float(rng.uniform(0.42, 0.62)))
        return np.clip(canvas * mask[..., None], 0, 255).astype(np.uint8)
    if condition == "fog_haze":
        blur = cv2.GaussianBlur(image, (0, 0), 2.0)
        haze = np.full_like(blur, 210)
        return cv2.addWeighted(blur, 0.58, haze, 0.42, 0)
    raise ValueError(f"Unsupported condition: {condition}")


def main() -> None:
    rng = np.random.default_rng(20260908)
    rows = []
    for condition in CONDITIONS:
        (OUT / condition).mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        source = DATA / "images" / "val" / name
        label = DATA / "labels" / "val" / f"{Path(name).stem}.txt"
        image = cv2.imread(str(source))
        if image is None or not label.exists():
            raise FileNotFoundError(f"Required holdout image or label missing: {name}")
        for condition in CONDITIONS:
            target = OUT / condition / name
            if not cv2.imwrite(str(target), transform(image, condition, rng)):
                raise RuntimeError(f"Failed to write {target}")
            rows.append({
                "image_name": name,
                "condition": condition,
                "source": f"synthetic photometric variant of RDD2022 China_Drone validation image: {source}",
                "ground_truth_available": "true",
                "annotation_path": str(label),
                "notes": "Geometry-preserving synthetic stress test; not a verified real-world weather label.",
            })
    with (OUT / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} images and {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
