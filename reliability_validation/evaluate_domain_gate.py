"""RoadSentinel Phase 12: DINOv2 Domain-Gating Audit & Policy Simulation.

Evaluates:
1. OOD Saturation Root Cause Analysis:
   - Raw DINO kNN distance distribution before clipping.
   - Raw normalized distance (d_kNN / p99) without clipping.
2. Formal Domain Classification Experiment:
   - China (In-Domain = 0) vs India (Out-Domain = 1).
   - Domain AUROC, Domain AUPRC, optimal threshold, FPR (China false alarms), TPR (India detection).
3. Domain-Gated Selective Routing Simulation:
   - Standalone YOLO vs YOLO + Confidence Rejection vs Full Gated Architecture (DINO Gate + YOLO + Confidence).
   - Quantifies reduction in unsafe automated acceptance on out-of-domain data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate_domain_gate")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CHINA_DATA_PATH = WORKSPACE_ROOT / "reliability/data/reliability_dataset.csv"
INDIA_DATA_PATH = WORKSPACE_ROOT / "cross_domain/results/cross_domain_reliability_dataset.csv"

OUT_TABLES_DIR = WORKSPACE_ROOT / "reliability_validation/tables"
OUT_RESULTS_DIR = WORKSPACE_ROOT / "reliability_validation/results"

# Frozen Phase-8 Reference Constants (China Training Reference, N=1921)
CHINA_TRAIN_P99_KNN = 0.4491024327278137
CHINA_TRAIN_P95_KNN = 0.38041598200798035


def compute_distribution_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute complete descriptive statistics for a numeric array."""
    return {
        "N": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
    }


def main():
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_china = pd.read_csv(CHINA_DATA_PATH)
    df_india = pd.read_csv(INDIA_DATA_PATH)

    log.info("Loaded datasets for domain gate evaluation: China (N=%d), India (N=%d)", len(df_china), len(df_india))

    # 1. Raw Distance & Saturation Analysis
    china_knn_raw = df_china["dino_knn_distance"].values
    india_knn_raw = df_india["dino_knn_distance"].values

    china_raw_norm = china_knn_raw / CHINA_TRAIN_P99_KNN
    india_raw_norm = india_knn_raw / CHINA_TRAIN_P99_KNN

    stats_china_raw = compute_distribution_stats(china_knn_raw)
    stats_india_raw = compute_distribution_stats(india_knn_raw)
    stats_china_norm = compute_distribution_stats(china_raw_norm)
    stats_india_norm = compute_distribution_stats(india_raw_norm)

    # 2. Domain Classification Experiment (China = 0, India = 1)
    y_domain = np.array([0] * len(df_china) + [1] * len(df_india))
    scores_domain_knn = np.concatenate([china_knn_raw, india_knn_raw])
    scores_domain_norm = np.concatenate([china_raw_norm, india_raw_norm])

    auroc_domain = float(roc_auc_score(y_domain, scores_domain_knn))
    auprc_domain = float(average_precision_score(y_domain, scores_domain_knn))

    # Evaluate at specified thresholds derived from China training data
    fpr_p95 = float(np.mean(china_knn_raw > CHINA_TRAIN_P95_KNN))
    tpr_p95 = float(np.mean(india_knn_raw > CHINA_TRAIN_P95_KNN))

    fpr_p99 = float(np.mean(china_knn_raw > CHINA_TRAIN_P99_KNN))
    tpr_p99 = float(np.mean(india_knn_raw > CHINA_TRAIN_P99_KNN))

    # Separation distance (minimum India distance vs maximum China distance)
    min_india_dist = float(np.min(india_knn_raw))
    max_china_dist = float(np.max(china_knn_raw))
    margin_separation = min_india_dist - max_china_dist

    # Save Domain Gate Table
    domain_gate_records = [
        {
            "metric_type": "Raw_DINO_kNN_Distance",
            "china_mean_std": f"{stats_china_raw['mean']} ± {stats_china_raw['std']}",
            "china_median_iqr": f"{stats_china_raw['median']} [{stats_china_raw['iqr']}]",
            "china_range": f"[{stats_china_raw['min']}, {stats_china_raw['max']}]",
            "china_p95": stats_china_raw["p95"],
            "china_p99": stats_china_raw["p99"],
            "india_mean_std": f"{stats_india_raw['mean']} ± {stats_india_raw['std']}",
            "india_median_iqr": f"{stats_india_raw['median']} [{stats_india_raw['iqr']}]",
            "india_range": f"[{stats_india_raw['min']}, {stats_india_raw['max']}]",
            "india_p95": stats_india_raw["p95"],
            "india_p99": stats_india_raw["p99"],
            "domain_AUROC": round(auroc_domain, 4),
            "domain_AUPRC": round(auprc_domain, 4),
            "separation_margin": round(margin_separation, 4),
            "p95_threshold": round(CHINA_TRAIN_P95_KNN, 4),
            "p95_china_false_warning_rate": round(fpr_p95, 4),
            "p95_india_detection_rate": round(tpr_p95, 4),
            "p99_threshold": round(CHINA_TRAIN_P99_KNN, 4),
            "p99_china_false_warning_rate": round(fpr_p99, 4),
            "p99_india_detection_rate": round(tpr_p99, 4),
        },
        {
            "metric_type": "Raw_Normalized_Distance_(d/p99)",
            "china_mean_std": f"{stats_china_norm['mean']} ± {stats_china_norm['std']}",
            "china_median_iqr": f"{stats_china_norm['median']} [{stats_china_norm['iqr']}]",
            "china_range": f"[{stats_china_norm['min']}, {stats_china_norm['max']}]",
            "china_p95": stats_china_norm["p95"],
            "china_p99": stats_china_norm["p99"],
            "india_mean_std": f"{stats_india_norm['mean']} ± {stats_india_norm['std']}",
            "india_median_iqr": f"{stats_india_norm['median']} [{stats_india_norm['iqr']}]",
            "india_range": f"[{stats_india_norm['min']}, {stats_india_norm['max']}]",
            "india_p95": stats_india_norm["p95"],
            "india_p99": stats_india_norm["p99"],
            "domain_AUROC": round(auroc_domain, 4),
            "domain_AUPRC": round(auprc_domain, 4),
            "separation_margin": round(margin_separation / CHINA_TRAIN_P99_KNN, 4),
            "p95_threshold": round(CHINA_TRAIN_P95_KNN / CHINA_TRAIN_P99_KNN, 4),
            "p95_china_false_warning_rate": round(fpr_p95, 4),
            "p95_india_detection_rate": round(tpr_p95, 4),
            "p99_threshold": 1.0000,
            "p99_china_false_warning_rate": round(fpr_p99, 4),
            "p99_india_detection_rate": round(tpr_p99, 4),
        }
    ]
    df_gate_table = pd.DataFrame(domain_gate_records)
    df_gate_table.to_csv(OUT_TABLES_DIR / "table_domain_gate.csv", index=False)
    log.info("Saved domain gate table to table_domain_gate.csv")

    # 3. Selective Systems Architecture Comparison
    # Compare:
    # 1. Standalone YOLO (Accepts all predictions)
    # 2. YOLO + Confidence Rejection (Rejects low confidence or LOW band predictions)
    # 3. Full Gated Architecture (DINO Domain Gate + YOLO + Confidence Reliability)
    
    # Let's evaluate across China (In-Domain) and India (Cross-Domain) under Target T1 (F1 >= 0.50)
    system_comparison_records = []
    
    for d_name, d_df in [("China_Drone_Val (In-Domain)", df_china), ("India_Cross_Domain (Shifted)", df_india)]:
        n_total = len(d_df)
        y_fail_t1 = (d_df["yolo_f1"] < 0.50).values.astype(int)
        
        # System 1: Standalone YOLO (No rejection)
        acc_s1 = n_total
        fail_s1 = int(np.sum(y_fail_t1))
        fail_rate_s1 = fail_s1 / acc_s1
        
        # System 2: YOLO + Confidence Band Rejection (Reject LOW band r < 0.60 based on confidence model)
        # Using Phase-8 confidence heuristic or Model B
        # In Phase 8, HIGH band was r >= 0.85
        # Let's check max_conf >= 0.40 as confidence acceptance heuristic or Model B score
        conf_acc_mask = d_df["max_confidence"] >= 0.40  # Confidence-accepted subset
        acc_s2 = int(np.sum(conf_acc_mask))
        fail_s2 = int(np.sum(y_fail_t1[conf_acc_mask])) if acc_s2 > 0 else 0
        fail_rate_s2 = (fail_s2 / acc_s2) if acc_s2 > 0 else 0.0
        
        # System 3: Full Gated Architecture (DINO Domain Gate <= p99 AND Confidence accepted)
        gate_pass_mask = d_df["dino_knn_distance"] <= CHINA_TRAIN_P99_KNN
        gated_acc_mask = gate_pass_mask & conf_acc_mask
        acc_s3 = int(np.sum(gated_acc_mask))
        fail_s3 = int(np.sum(y_fail_t1[gated_acc_mask])) if acc_s3 > 0 else 0
        fail_rate_s3 = (fail_s3 / acc_s3) if acc_s3 > 0 else 0.0

        system_comparison_records.append({
            "dataset": d_name,
            "system_name": "System_1_Standalone_YOLO",
            "description": "Direct YOLO inference, zero reliability or domain gating",
            "total_images": n_total,
            "accepted_images": acc_s1,
            "acceptance_rate_pct": round(acc_s1 / n_total * 100, 2),
            "accepted_failures": fail_s1,
            "unsafe_accepted_failure_rate_pct": round(fail_rate_s1 * 100, 2),
            "unsafe_failure_reduction_vs_raw_pct": 0.0,
        })
        system_comparison_records.append({
            "dataset": d_name,
            "system_name": "System_2_YOLO_Confidence_Only",
            "description": "YOLO + Confidence thresholding (rejects unconfident detections)",
            "total_images": n_total,
            "accepted_images": acc_s2,
            "acceptance_rate_pct": round(acc_s2 / n_total * 100, 2),
            "accepted_failures": fail_s2,
            "unsafe_accepted_failure_rate_pct": round(fail_rate_s2 * 100, 2),
            "unsafe_failure_reduction_vs_raw_pct": round((1 - fail_rate_s2 / fail_rate_s1) * 100, 2) if fail_rate_s1 > 0 else 0.0,
        })
        system_comparison_records.append({
            "dataset": d_name,
            "system_name": "System_3_Full_Domain_Gated_Architecture",
            "description": "DINO Domain Gate + YOLO + Confidence Reliability",
            "total_images": n_total,
            "accepted_images": acc_s3,
            "acceptance_rate_pct": round(acc_s3 / n_total * 100, 2),
            "accepted_failures": fail_s3,
            "unsafe_accepted_failure_rate_pct": round(fail_rate_s3 * 100, 2),
            "unsafe_failure_reduction_vs_raw_pct": round((1 - (fail_s3 / fail_s1)) * 100, 2) if fail_s1 > 0 else 0.0,
        })

    df_sys = pd.DataFrame(system_comparison_records)
    df_sys.to_csv(OUT_TABLES_DIR / "table_selective_systems.csv", index=False)
    log.info("Saved selective system comparison table to table_selective_systems.csv")

    log.info("Domain gate evaluation complete.")


if __name__ == "__main__":
    main()
