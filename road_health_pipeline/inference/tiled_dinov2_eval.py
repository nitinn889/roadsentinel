#!/usr/bin/env python3
"""tiled_dinov2_eval.py
--------------------
Experimental 2X Tiled / Sliding-Window DINOv2 inference wrapper.
Evaluated during Marion Day 5 GPU Emergency Follow-Up.

Outcome: REJECTED for production pipeline due to scale-domain mismatch with
the 1X healthy road memory bank (causes pothole regression and healthy false positives).
Preserved separately for research reproducibility.

Usage:
    python road_health_pipeline/inference/tiled_dinov2_eval.py \
        --image RoadSentinel_datasets/rdd2022_full/China_Drone/China_Drone/train/images/China_Drone_001267.jpg \
        --tile-size 256 \
        --stride 128 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "sam2_dino"))

from common.io_utils import load_rgb
from inference.run_inference import load_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("tiled_dinov2_eval")


def compute_tiled_anomaly_map(
    pipeline,
    rgb: np.ndarray,
    road_mask: np.ndarray,
    tile_size: int = 256,
    stride: int = 128,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Compute aggregated anomaly map using sliding window tiles."""
    h, w = rgb.shape[:2]
    raw_accum = np.zeros((h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    wy = np.hanning(tile_size).astype(np.float32)
    wx = np.hanning(tile_size).astype(np.float32)
    window = np.clip(np.outer(wy, wx), 0.1, 1.0)

    y_starts = sorted(list(set(list(range(0, h - tile_size + 1, stride)) + [h - tile_size])))
    x_starts = sorted(list(set(list(range(0, w - tile_size + 1, stride)) + [w - tile_size])))

    tiles_processed = 0
    for y0 in y_starts:
        y1 = y0 + tile_size
        for x0 in x_starts:
            x1 = x0 + tile_size
            tile_road = road_mask[y0:y1, x0:x1] if road_mask is not None else np.ones((tile_size, tile_size), bool)
            if not np.any(tile_road):
                continue

            crop = rgb[y0:y1, x0:x1]
            grid = pipeline.embedder.extract_patch_grid(crop)  # (37, 37, 384)
            flat = grid.reshape(-1, 384)
            scores = pipeline.detector.score_patches(flat).reshape(37, 37)
            tile_map = cv2.resize(scores, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

            raw_accum[y0:y1, x0:x1] += tile_map * window
            weight_accum[y0:y1, x0:x1] += window
            tiles_processed += 1

    raw_amap = raw_accum / np.maximum(weight_accum, 1e-6)

    # Local road normalization on aggregated raw anomaly map
    road_pixels = raw_amap[road_mask] if road_mask is not None else raw_amap.ravel()
    median = float(np.median(road_pixels))
    q25 = float(np.percentile(road_pixels, 25))
    q75 = float(np.percentile(road_pixels, 75))
    iqr = max(1e-4, q75 - q25)
    scale = max(1e-4, 3.0 * iqr)

    norm_amap = np.clip((raw_amap - median) / scale, 0.0, 1.0)
    pos = norm_amap[road_mask][norm_amap[road_mask] > 1e-6] if road_mask is not None else norm_amap[norm_amap > 1e-6]
    threshold = float(np.percentile(pos, 92.0)) if len(pos) > 0 else 1.0

    return raw_amap, norm_amap, threshold, tiles_processed


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 2X Tiled DINOv2 anomaly localization.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels (default 256 for 2X effective zoom)")
    parser.add_argument("--stride", type=int, default=128, help="Tile stride in pixels (default 128 for 50 percent overlap)")
    parser.add_argument("--device", default="cuda", help="Inference device")
    args = parser.parse_args()

    pipeline = load_pipeline(device=args.device)
    rgb = load_rgb(args.image)
    road_mask = pipeline.masker.get_road_mask(rgb)

    t0 = time.perf_counter()
    raw_amap, norm_amap, thresh, num_tiles = compute_tiled_anomaly_map(
        pipeline, rgb, road_mask, tile_size=args.tile_size, stride=args.stride
    )
    cands = pipeline.localizer.localize(rgb, norm_amap, road_mask, thresh, sam2=pipeline.masker, test_mode_2d=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(json.dumps({
        "status": "COMPLETED",
        "experiment": "2X_TILED_DINOV2",
        "image": str(args.image),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "tiles_processed": num_tiles,
        "threshold_p92": round(thresh, 4),
        "accepted_defects": len(cands),
        "elapsed_ms": round(elapsed_ms, 1),
        "decision": "REJECTED_FOR_PRODUCTION"
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
