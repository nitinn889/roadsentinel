from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


def compute_depth_metrics(pred_depth_m: np.ndarray,
                         gt_depth_m: np.ndarray,
                         mask: Optional[np.ndarray] = None,
                         min_depth_m: float = 0.1,
                         max_depth_m: float = 80.0) -> Dict[str, Optional[float]]:
    """Calculates standardized depth evaluation metrics comparing estimated metric depth to ground truth.

    Metrics computed:
    - MAE (Mean Absolute Error, in metres)
    - RMSE (Root Mean Squared Error, in metres)
    - AbsRel (Mean Absolute Relative Error: |pred - gt| / gt)
    - SqRel (Squared Relative Error: (pred - gt)^2 / gt)
    - Delta thresholds: % of pixels where max(pred/gt, gt/pred) < 1.25, 1.25^2, 1.25^3
    """
    if pred_depth_m is None or gt_depth_m is None:
        return {
            "mae_m": None,
            "rmse_m": None,
            "abs_rel": None,
            "sq_rel": None,
            "delta_1": None,
            "delta_2": None,
            "delta_3": None,
            "valid_pixels": 0,
        }

    # Validate shapes
    if pred_depth_m.shape != gt_depth_m.shape:
        raise ValueError(f"Shape mismatch between pred ({pred_depth_m.shape}) and gt ({gt_depth_m.shape})")

    # Build valid pixel mask
    valid = np.isfinite(pred_depth_m) & np.isfinite(gt_depth_m)
    valid &= (gt_depth_m >= min_depth_m) & (gt_depth_m <= max_depth_m)
    valid &= (pred_depth_m >= min_depth_m)

    if mask is not None:
        valid &= (mask > 0)

    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {
            "mae_m": None,
            "rmse_m": None,
            "abs_rel": None,
            "sq_rel": None,
            "delta_1": None,
            "delta_2": None,
            "delta_3": None,
            "valid_pixels": 0,
        }

    pred_v = pred_depth_m[valid].astype(np.float64)
    gt_v = gt_depth_m[valid].astype(np.float64)

    # MAE & RMSE
    diff = pred_v - gt_v
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Relative errors
    abs_rel = float(np.mean(np.abs(diff) / gt_v))
    sq_rel = float(np.mean((diff ** 2) / gt_v))

    # Threshold accuracy (delta)
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < 1.25 ** 2))
    delta_3 = float(np.mean(ratio < 1.25 ** 3))

    return {
        "mae_m": round(mae, 4),
        "rmse_m": round(rmse, 4),
        "abs_rel": round(abs_rel, 4),
        "sq_rel": round(sq_rel, 4),
        "delta_1": round(delta_1, 4),
        "delta_2": round(delta_2, 4),
        "delta_3": round(delta_3, 4),
        "valid_pixels": n_valid,
    }
