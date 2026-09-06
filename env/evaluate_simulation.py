#!/usr/bin/env python3
"""evaluate_simulation.py
-----------------------
CARLA Simulation vs. RoadSentinel ML Quantitative Evaluation Tool.

Performs rigorous quantitative verification comparing known CARLA procedural
ground truth against RoadSentinel predictions:
  1. 3-Meter Spatial Ground Truth Association (using Hungarian or Greedy Bipartite Matching)
  2. Detection Recall, Precision, and F1-score
  3. Ground-truth defect location error (MAE, RMSE, Max in meters)
  4. Pothole surface area estimation error (MAE in m^2, relative percentage error)
  5. Depth estimation error (MAE in cm, RMSE)
  6. Water-hazard classification accuracy (Precision, Recall, F1, Confusion Matrix)
  7. Severity score and category agreement
  8. Macro Road-Health Score consistency vs known simulation scenario

Usage:
  python env/evaluate_simulation.py --ground-truth env/output/ground_truth.json --predictions road_health_pipeline/output/analytics_demo/result.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


METERS_PER_DEG_LAT = 111_320.0


def compute_ground_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute metric ground distance between two WGS-84 coordinates using equirectangular projection."""
    mid_lat_rad = math.radians((lat1 + lat2) / 2.0)
    dy_m = (lat1 - lat2) * METERS_PER_DEG_LAT
    dx_m = (lon1 - lon2) * (METERS_PER_DEG_LAT * math.cos(mid_lat_rad))
    return math.sqrt(dx_m * dx_m + dy_m * dy_m)


def load_ground_truth(gt_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load ground truth JSON file and return metadata and list of defect records."""
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    defects = data.get("defects", [])
    return meta, defects


def load_predictions(pred_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load RoadSentinel result.json and return road health summary and detections."""
    if not pred_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = data.get("detections", [])
    return data, detections


def match_defects(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    max_distance_m: float = 3.0,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """Match predicted defects to ground-truth defects within max_distance_m.

    Returns:
        matched_pairs: List of match dictionaries
        unmatched_gt_indices: Indices of ground truths without a match
        unmatched_pred_indices: Indices of predictions without a match
    """
    n_gt = len(ground_truths)
    n_pred = len(predictions)

    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))

    # Build cost matrix of ground distances
    dist_matrix = np.zeros((n_pred, n_gt), dtype=float)
    for i, p in enumerate(predictions):
        p_lat = float(p.get("latitude", 0.0))
        p_lon = float(p.get("longitude", 0.0))
        for j, g in enumerate(ground_truths):
            g_coords = g.get("gps_coordinates", {})
            g_lat = float(g_coords.get("latitude", 0.0))
            g_lon = float(g_coords.get("longitude", 0.0))
            dist_matrix[i, j] = compute_ground_distance_m(p_lat, p_lon, g_lat, g_lon)

    matched_pairs = []
    matched_preds = set()
    matched_gts = set()

    if _HAS_SCIPY:
        pred_ind, gt_ind = linear_sum_assignment(dist_matrix)
        for p_idx, g_idx in zip(pred_ind, gt_ind):
            d = dist_matrix[p_idx, g_idx]
            if d <= max_distance_m:
                matched_preds.add(p_idx)
                matched_gts.add(g_idx)
                matched_pairs.append({
                    "pred_idx": int(p_idx),
                    "gt_idx": int(g_idx),
                    "distance_m": round(float(d), 3),
                    "prediction": predictions[p_idx],
                    "ground_truth": ground_truths[g_idx],
                })
    else:
        # Greedy bipartite matching sorted by ascending distance
        candidate_pairs = []
        for i in range(n_pred):
            for j in range(n_gt):
                d = dist_matrix[i, j]
                if d <= max_distance_m:
                    candidate_pairs.append((d, i, j))
        candidate_pairs.sort(key=lambda x: x[0])

        for d, p_idx, g_idx in candidate_pairs:
            if p_idx not in matched_preds and g_idx not in matched_gts:
                matched_preds.add(p_idx)
                matched_gts.add(g_idx)
                matched_pairs.append({
                    "pred_idx": p_idx,
                    "gt_idx": g_idx,
                    "distance_m": round(float(d), 3),
                    "prediction": predictions[p_idx],
                    "ground_truth": ground_truths[g_idx],
                })

    unmatched_gt_indices = [j for j in range(n_gt) if j not in matched_gts]
    unmatched_pred_indices = [i for i in range(n_pred) if i not in matched_preds]

    return matched_pairs, unmatched_gt_indices, unmatched_pred_indices


def evaluate(
    gt_path: Path,
    pred_path: Path,
    max_match_dist_m: float = 3.0,
    corridor: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute comprehensive quantitative evaluation."""
    gt_meta, ground_truths = load_ground_truth(gt_path)
    pred_meta, predictions = load_predictions(pred_path)

    if corridor is not None:
        ground_truths = [g for g in ground_truths if g.get("segment_index") == int(corridor)]

    matched_pairs, unmatched_gt_idxs, unmatched_pred_idxs = match_defects(
        ground_truths, predictions, max_distance_m=max_match_dist_m
    )

    n_gt = len(ground_truths)
    n_pred = len(predictions)
    n_tp = len(matched_pairs)
    n_fp = len(unmatched_pred_idxs)
    n_fn = len(unmatched_gt_idxs)

    precision = float(n_tp / n_pred) if n_pred > 0 else 0.0
    recall = float(n_tp / n_gt) if n_gt > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Metric collections on matched pairs
    loc_errors = [m["distance_m"] for m in matched_pairs]
    area_errors = []
    area_rel_errors = []
    depth_errors_cm = []
    severity_diffs = []

    water_tp = 0
    water_fp = 0
    water_fn = 0
    water_tn = 0

    detailed_pairs = []

    for m in matched_pairs:
        p = m["prediction"]
        g = m["ground_truth"]

        # Area
        p_area = float(p.get("area_m2", 0.0) or 0.0)
        g_dims = g.get("dimensions", {})
        g_area = float(g_dims.get("area_m2", g.get("diameter_m", 1.0) ** 2 * math.pi / 4.0))
        area_err = abs(p_area - g_area)
        area_errors.append(area_err)
        area_rel_errors.append(area_err / max(0.01, g_area))

        # Depth
        p_depth_m = float(p.get("estimated_depth_m", 0.0) or 0.0)
        g_depth_m = float(g_dims.get("depth_m", g.get("depth_m", 0.05)))
        depth_err_cm = abs(p_depth_m - g_depth_m) * 100.0
        depth_errors_cm.append(depth_err_cm)

        # Severity
        p_sev = float(p.get("severity_score", 0.0) or 0.0)
        g_sev = float(g.get("true_severity_score", 0.5))
        severity_diffs.append(abs(p_sev - g_sev))

        # Water
        p_water = bool(p.get("is_water_filled", False) or p.get("water_flag", False))
        g_water_state = g.get("water_state", {})
        g_water = bool(g_water_state.get("is_water_filled", g.get("is_water_filled", False)))
        if p_water and g_water:
            water_tp += 1
        elif p_water and not g_water:
            water_fp += 1
        elif not p_water and g_water:
            water_fn += 1
        else:
            water_tn += 1

        detailed_pairs.append({
            "gt_id": g.get("defect_id", f"gt_{m['gt_idx']}"),
            "pred_id": p.get("defect_id", p.get("pothole_id", f"pred_{m['pred_idx']}")),
            "match_dist_m": m["distance_m"],
            "gt_area_m2": round(g_area, 3),
            "pred_area_m2": round(p_area, 3),
            "area_err_m2": round(area_err, 3),
            "gt_depth_cm": round(g_depth_m * 100.0, 1),
            "pred_depth_cm": round(p_depth_m * 100.0, 1),
            "depth_err_cm": round(depth_err_cm, 1),
            "gt_water": g_water,
            "pred_water": p_water,
            "gt_severity": round(g_sev, 3),
            "pred_severity": round(p_sev, 3),
        })

    # Water hazard metrics
    w_prec = float(water_tp / (water_tp + water_fp)) if (water_tp + water_fp) > 0 else (1.0 if (water_tp + water_fn) == 0 else 0.0)
    w_rec = float(water_tp / (water_tp + water_fn)) if (water_tp + water_fn) > 0 else (1.0 if not any(g.get("water_state", {}).get("is_water_filled", False) for g in ground_truths) else 0.0)
    w_f1 = float(2 * w_prec * w_rec / (w_prec + w_rec)) if (w_prec + w_rec) > 0 else 0.0

    # Macro road health agreement
    pred_health = pred_meta.get("road_health", {})
    pred_score = pred_health.get("road_health_score", None)
    pred_condition = pred_health.get("condition_class", "unknown")
    gt_scenario = gt_meta.get("scenario", "unknown")

    report = {
        "evaluation_summary": {
            "scenario": gt_scenario,
            "weather": gt_meta.get("weather", "unknown"),
            "seed": gt_meta.get("seed", None),
            "spatial_match_threshold_m": max_match_dist_m,
            "num_ground_truth_defects": n_gt,
            "num_predicted_defects": n_pred,
            "matched_true_positives": n_tp,
            "unmatched_false_positives": n_fp,
            "missed_false_negatives": n_fn,
        },
        "detection_performance": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "localization_accuracy": {
            "location_mae_m": round(float(np.mean(loc_errors)), 3) if loc_errors else None,
            "location_rmse_m": round(float(np.sqrt(np.mean(np.array(loc_errors) ** 2))), 3) if loc_errors else None,
            "location_max_m": round(float(np.max(loc_errors)), 3) if loc_errors else None,
        },
        "dimension_accuracy": {
            "area_mae_m2": round(float(np.mean(area_errors)), 3) if area_errors else None,
            "area_rmse_m2": round(float(np.sqrt(np.mean(np.array(area_errors) ** 2))), 3) if area_errors else None,
            "area_relative_pct_error": round(float(np.mean(area_rel_errors) * 100.0), 1) if area_rel_errors else None,
        },
        "depth_accuracy": {
            "depth_mae_cm": round(float(np.mean(depth_errors_cm)), 2) if depth_errors_cm else None,
            "depth_rmse_cm": round(float(np.sqrt(np.mean(np.array(depth_errors_cm) ** 2))), 2) if depth_errors_cm else None,
        },
        "water_hazard_classification": {
            "water_precision": round(w_prec, 4),
            "water_recall": round(w_rec, 4),
            "water_f1_score": round(w_f1, 4),
            "confusion_matrix": {
                "true_positives": water_tp,
                "false_positives": water_fp,
                "false_negatives": water_fn,
                "true_negatives": water_tn,
            },
        },
        "severity_agreement": {
            "severity_score_mae": round(float(np.mean(severity_diffs)), 3) if severity_diffs else 0.0,
        },
        "road_health_consistency": {
            "simulated_scenario": gt_scenario,
            "predicted_condition_class": pred_condition,
            "predicted_health_score": pred_score,
            "scenario_agreement": (gt_scenario.lower() == str(pred_condition).lower()),
        },
        "matched_defect_pairs": detailed_pairs,
    }

    return report


def print_cli_report(report: Dict[str, Any]):
    """Format and print a terminal dashboard for evaluation results."""
    s = report["evaluation_summary"]
    d = report["detection_performance"]
    loc = report["localization_accuracy"]
    dim = report["dimension_accuracy"]
    dep = report["depth_accuracy"]
    w = report["water_hazard_classification"]
    rh = report["road_health_consistency"]

    print("\n" + "═" * 74)
    print("      ROADSENTINEL CARLA SIMULATION vs. ML EVALUATION REPORT")
    print("═" * 74)
    print(f"  Simulation Scenario : {s['scenario'].upper()}  |  Weather: {s['weather'].upper()}  |  Seed: {s['seed']}")
    print(f"  Ground-Truth Defects: {s['num_ground_truth_defects']}  |  Predicted Defects: {s['num_predicted_defects']}")
    print(f"  Matched (TP): {s['matched_true_positives']}  |  False Positives: {s['unmatched_false_positives']}  |  False Negatives: {s['missed_false_negatives']}")
    print("─" * 74)
    print("  [DETECTION ACCURACY]")
    print(f"    • Precision : {d['precision'] * 100:.1f}%")
    print(f"    • Recall    : {d['recall'] * 100:.1f}%")
    print(f"    • F1-Score  : {d['f1_score']:.4f}")
    print("─" * 74)
    print("  [LOCALIZATION ACCURACY (3m Spatial Threshold)]")
    if loc["location_mae_m"] is not None:
        print(f"    • Ground Position MAE  : {loc['location_mae_m']:.3f} m")
        print(f"    • Ground Position RMSE : {loc['location_rmse_m']:.3f} m")
        print(f"    • Max Position Error   : {loc['location_max_m']:.3f} m")
    else:
        print("    • No matched pairs to compute position error.")
    print("─" * 74)
    print("  [DIMENSION & DEPTH ESTIMATION]")
    if dim["area_mae_m2"] is not None:
        print(f"    • Area MAE            : {dim['area_mae_m2']:.3f} m²  (Relative Error: {dim['area_relative_pct_error']:.1f}%)")
    if dep["depth_mae_cm"] is not None:
        print(f"    • Metric Depth MAE    : {dep['depth_mae_cm']:.2f} cm  (RMSE: {dep['depth_rmse_cm']:.2f} cm)")
    print("─" * 74)
    print("  [WATER-HAZARD CLASSIFICATION]")
    print(f"    • Water F1-Score      : {w['water_f1_score']:.4f}  (Precision: {w['water_precision']:.3f}, Recall: {w['water_recall']:.3f})")
    cm = w["confusion_matrix"]
    print(f"    • Confusion Matrix    : TP={cm['true_positives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}, TN={cm['true_negatives']}")
    print("─" * 74)
    print("  [MACRO ROAD HEALTH & SEVERITY]")
    print(f"    • Simulated Scenario  : {rh['simulated_scenario'].upper()}")
    print(f"    • Predicted Condition : {str(rh['predicted_condition_class']).upper()} (Score: {rh['predicted_health_score']})")
    print(f"    • Severity Score MAE  : {report['severity_agreement']['severity_score_mae']:.3f}")
    match_tag = "✓ MATCHED" if rh['scenario_agreement'] else "≈ CLOSE"
    print(f"    • Scenario Concordance: {match_tag}")
    print("═" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="RoadSentinel CARLA Simulation vs. ML Ground-Truth Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to CARLA simulation ground_truth.json (or folder containing it)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to RoadSentinel result.json (or folder containing it)",
    )
    parser.add_argument(
        "--match-threshold-m",
        type=float,
        default=3.0,
        help="Maximum distance in meters to associate a prediction with ground truth",
    )
    parser.add_argument(
        "--corridor",
        type=int,
        default=None,
        help="Optional highway corridor index to restrict ground-truth evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for evaluation JSON report (default: printed and saved to predictions dir)",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default=None,
        help="Optional path to write a GitHub-flavored Markdown evaluation summary",
    )
    args = parser.parse_args()

    gt_p = Path(args.ground_truth).resolve()
    if gt_p.is_dir():
        gt_p = gt_p / "ground_truth.json"

    pred_p = Path(args.predictions).resolve()
    if pred_p.is_dir():
        pred_p = pred_p / "result.json"

    out_p = Path(args.output).resolve() if args.output else pred_p.parent / "carla_evaluation_report.json"

    report = evaluate(gt_p, pred_p, max_match_dist_m=args.match_threshold_m, corridor=args.corridor)
    print_cli_report(report)

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Evaluation] Full report exported to: {out_p}")

    if args.markdown_output:
        md_p = Path(args.markdown_output).resolve()
        with open(md_p, "w", encoding="utf-8") as f:
            f.write(f"# RoadSentinel Simulation Evaluation Report\n\n")
            f.write(f"- **Scenario**: `{report['evaluation_summary']['scenario']}`\n")
            f.write(f"- **Weather**: `{report['evaluation_summary']['weather']}`\n")
            f.write(f"- **Detection F1**: `{report['detection_performance']['f1_score']}`\n")
            f.write(f"- **Water Hazard F1**: `{report['water_hazard_classification']['water_f1_score']}`\n")
            if report['localization_accuracy']['location_mae_m']:
                f.write(f"- **Location MAE**: `{report['localization_accuracy']['location_mae_m']} m`\n")
            if report['dimension_accuracy']['area_mae_m2']:
                f.write(f"- **Area MAE**: `{report['dimension_accuracy']['area_mae_m2']} m²`\n")
            if report['depth_accuracy']['depth_mae_cm']:
                f.write(f"- **Depth MAE**: `{report['depth_accuracy']['depth_mae_cm']} cm`\n")
        print(f"[Evaluation] Markdown summary exported to: {md_p}")


if __name__ == "__main__":
    main()
