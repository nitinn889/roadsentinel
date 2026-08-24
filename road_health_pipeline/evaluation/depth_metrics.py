from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.depth_estimator import depth_from_carla_ground_truth


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0)
    if not np.any(mask):
        raise ValueError("No valid depth pixels")
    p, g = pred[mask], gt[mask]
    err = p - g
    return {
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "relative_error": float(np.mean(np.abs(err) / np.maximum(g, 1e-6))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", type=Path, help="Predicted metric depth as .npy")
    ap.add_argument("gt", type=Path, help="CARLA raw depth PNG")
    ap.add_argument("--far-clip", type=float, default=1000.0)
    args = ap.parse_args()
    pred = np.load(args.pred).astype(np.float32)
    raw = cv2.cvtColor(cv2.imread(str(args.gt), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gt = depth_from_carla_ground_truth(raw, args.far_clip)
    print(metrics(pred, gt))


if __name__ == "__main__":
    main()
