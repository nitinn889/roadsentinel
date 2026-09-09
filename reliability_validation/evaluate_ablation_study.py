"""RoadSentinel Phase 12: Comprehensive Reliability Validation & Full Signal Ablation Study.

Evaluates:
1. Multiple Perceptual Targets (T0, T1, T2, T3, T4).
2. Signal Ablation Models (Models A to H):
   - Model A: YOLO Max Confidence Only
   - Model B: YOLO Confidence Features (max, mean, std, pred_count, num_low, num_high)
   - Model C: DINO OOD Only (ood_score, knn_dist, centroid_dist)
   - Model D: Image Quality Only (brightness, contrast, sharpness, edge_density)
   - Model E: YOLO Confidence + DINO OOD
   - Model F: YOLO Confidence + Image Quality
   - Model G: DINO OOD + Image Quality
   - Model H: Full Combined (Confidence + DINO OOD + Image Quality)
3. In-Domain 5-Fold Stratified Cross-Validation (China_Drone, N=480).
4. Cross-Domain Zero-Shot Transfer (India Dashcam, N=300) trained strictly on China development data.
5. 95% Bootstrap Confidence Intervals for AUROC and AUPRC.
6. Target Robustness Analysis.
7. Reliability Bands (HIGH, MEDIUM, LOW) across all targets and datasets.
8. Risk-Coverage Selective Prediction Analysis across multiple coverage levels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase12_ablation")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CHINA_DATA_PATH = WORKSPACE_ROOT / "reliability/data/reliability_dataset.csv"
INDIA_DATA_PATH = WORKSPACE_ROOT / "cross_domain/results/cross_domain_reliability_dataset.csv"

OUT_DATA_DIR = WORKSPACE_ROOT / "reliability_validation/data"
OUT_TABLES_DIR = WORKSPACE_ROOT / "reliability_validation/tables"
OUT_RESULTS_DIR = WORKSPACE_ROOT / "reliability_validation/results"
DASHBOARD_ASSETS_DIR = WORKSPACE_ROOT / "integration/dashboard_assets/reliability_validation"

# Feature Group Definitions
FEATURE_GROUPS: Dict[str, Tuple[str, List[str]]] = {
    "Model_A": ("YOLO Max Confidence Only", ["max_confidence"]),
    "Model_B": ("YOLO Confidence Features", [
        "max_confidence", "mean_confidence", "std_confidence",
        "yolo_pred_count", "num_low_conf", "num_high_conf"
    ]),
    "Model_C": ("DINO OOD Only", [
        "dino_ood_score", "dino_knn_distance", "dino_centroid_distance"
    ]),
    "Model_D": ("Image Quality Only", [
        "brightness", "contrast", "sharpness", "edge_density"
    ]),
    "Model_E": ("YOLO Confidence + DINO OOD", [
        "max_confidence", "mean_confidence", "std_confidence",
        "yolo_pred_count", "num_low_conf", "num_high_conf",
        "dino_ood_score", "dino_knn_distance", "dino_centroid_distance"
    ]),
    "Model_F": ("YOLO Confidence + Image Quality", [
        "max_confidence", "mean_confidence", "std_confidence",
        "yolo_pred_count", "num_low_conf", "num_high_conf",
        "brightness", "contrast", "sharpness", "edge_density"
    ]),
    "Model_G": ("DINO OOD + Image Quality", [
        "dino_ood_score", "dino_knn_distance", "dino_centroid_distance",
        "brightness", "contrast", "sharpness", "edge_density"
    ]),
    "Model_H": ("Full Combined (Conf + DINO + Image Quality)", [
        "max_confidence", "mean_confidence", "std_confidence",
        "yolo_pred_count", "num_low_conf", "num_high_conf",
        "dino_ood_score", "dino_knn_distance", "dino_centroid_distance",
        "brightness", "contrast", "sharpness", "edge_density"
    ]),
}

TARGET_DEFS = {
    "T0": {
        "name": "Historical F1 > 0",
        "desc": "image-level F1 > 0 (lenient baseline)",
        "succ_fn": lambda df: (df["yolo_f1"] > 0).astype(int),
    },
    "T1": {
        "name": "F1 >= 0.50",
        "desc": "image-level F1 >= 0.50 (balanced detection quality)",
        "succ_fn": lambda df: (df["yolo_f1"] >= 0.50).astype(int),
    },
    "T2": {
        "name": "Recall >= 0.50",
        "desc": "image-level Recall >= 0.50 (moderate defect capture)",
        "succ_fn": lambda df: (df["yolo_recall"] >= 0.50).astype(int),
    },
    "T3": {
        "name": "Recall >= 0.75",
        "desc": "image-level Recall >= 0.75 (high defect capture)",
        "succ_fn": lambda df: (df["yolo_recall"] >= 0.75).astype(int),
    },
    "T4": {
        "name": "Complete Recall (100% @ IoU>=0.5)",
        "desc": "all GT defects detected (Recall == 1.0 where GT > 0)",
        "succ_fn": lambda df: ((df["gt_count"] > 0) & (df["yolo_recall"] >= 1.0)).astype(int),
    },
}


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for AUROC and AUPRC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aurocs, auprcs = [], []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_t, y_p = y_true[idx], y_prob[idx]
        if len(np.unique(y_t)) < 2:
            continue
        aurocs.append(roc_auc_score(y_t, y_p))
        auprcs.append(average_precision_score(y_t, y_p))

    auroc_low = float(np.percentile(aurocs, 2.5)) if aurocs else 0.0
    auroc_high = float(np.percentile(aurocs, 97.5)) if aurocs else 0.0
    auprc_low = float(np.percentile(auprcs, 2.5)) if auprcs else 0.0
    auprc_high = float(np.percentile(auprcs, 97.5)) if auprcs else 0.0

    return {
        "AUROC_CI": (round(auroc_low, 4), round(auroc_high, 4)),
        "AUPRC_CI": (round(auprc_low, 4), round(auprc_high, 4)),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find decision threshold maximizing failure F1 score on training/OOF data."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def evaluate_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Calculate full suite of scientific failure prediction metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    unique_y = np.unique(y_true)
    auroc = float(roc_auc_score(y_true, y_prob)) if len(unique_y) > 1 else 0.5
    auprc = float(average_precision_score(y_true, y_prob)) if len(unique_y) > 1 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(unique_y) > 1 else float(np.mean(y_true == y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob, n_bins=10)

    return {
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "failure_precision": round(prec, 4),
        "failure_recall": round(rec, 4),
        "failure_F1": round(f1, 4),
        "Brier": round(brier, 4),
        "ECE": round(ece, 4),
    }


def compute_reliability_bands(
    y_true_failure: np.ndarray,
    reliability_scores: np.ndarray,
    high_th: float = 0.85,
    low_th: float = 0.60,
) -> Dict[str, Dict[str, Any]]:
    """Compute band statistics: HIGH (>=high_th), MEDIUM (low_th <= r < high_th), LOW (<low_th)."""
    bands = {}
    high_mask = reliability_scores >= high_th
    med_mask = (reliability_scores >= low_th) & (reliability_scores < high_th)
    low_mask = reliability_scores < low_th

    for b_name, mask in [("HIGH", high_mask), ("MEDIUM", med_mask), ("LOW", low_mask)]:
        n_b = int(np.sum(mask))
        if n_b > 0:
            n_fail = int(np.sum(y_true_failure[mask]))
            n_succ = n_b - n_fail
            succ_rate = round(n_succ / n_b, 4)
            fail_rate = round(n_fail / n_b, 4)
        else:
            n_succ = n_fail = 0
            succ_rate = fail_rate = 0.0
        bands[b_name] = {
            "count": n_b,
            "success_count": n_succ,
            "failure_count": n_fail,
            "success_rate": succ_rate,
            "failure_rate": fail_rate,
        }
    return bands


def compute_risk_coverage(
    df: pd.DataFrame,
    y_true_failure: np.ndarray,
    reliability_scores: np.ndarray,
    target_key: str,
    coverage_levels: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
) -> List[Dict[str, Any]]:
    """Compute risk-coverage selective prediction statistics."""
    rows = []
    n_total = len(df)
    total_failures = int(np.sum(y_true_failure))

    # Rank by reliability score descending
    order = np.argsort(-reliability_scores)
    
    for cov in coverage_levels:
        k = int(round(cov * n_total))
        k = max(1, min(k, n_total))
        accepted_idx = order[:k]
        
        acc_failures = int(np.sum(y_true_failure[accepted_idx]))
        acc_successes = k - acc_failures
        acc_fail_rate = acc_failures / k
        
        acc_df = df.iloc[accepted_idx]
        mean_f1 = float(acc_df["yolo_f1"].mean())
        mean_rec = float(acc_df["yolo_recall"].mean())
        
        rejected_failures = total_failures - acc_failures
        rej_fail_fraction = (rejected_failures / total_failures) if total_failures > 0 else 0.0

        rows.append({
            "target": target_key,
            "coverage_pct": int(round(cov * 100)),
            "accepted_count": k,
            "accepted_failures": acc_failures,
            "accepted_successes": acc_successes,
            "accepted_failure_rate": round(acc_fail_rate, 4),
            "mean_yolo_f1": round(mean_f1, 4),
            "mean_yolo_recall": round(mean_rec, 4),
            "rejected_failures": rejected_failures,
            "rejected_failure_fraction": round(rej_fail_fraction, 4),
            "min_reliability_threshold": round(float(reliability_scores[order[k - 1]]), 4),
        })
    return rows


def main():
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Data
    df_china = pd.read_csv(CHINA_DATA_PATH)
    df_india = pd.read_csv(INDIA_DATA_PATH)
    log.info("Loaded datasets: China validation (N=%d), India cross-domain (N=%d)", len(df_china), len(df_india))

    # Ensure consistent column naming for image quality and counts
    if "num_low_conf" not in df_india.columns and "num_low_confidence" in df_india.columns:
        df_india["num_low_conf"] = df_india["num_low_confidence"]
    if "num_high_conf" not in df_india.columns and "num_high_confidence" in df_india.columns:
        df_india["num_high_conf"] = df_india["num_high_confidence"]

    # 2. Compute Ground Truth Targets T0 - T4
    for t_key, t_info in TARGET_DEFS.items():
        df_china[f"{t_key}_success"] = t_info["succ_fn"](df_china)
        df_china[f"{t_key}_failure"] = 1 - df_china[f"{t_key}_success"]
        df_india[f"{t_key}_success"] = t_info["succ_fn"](df_india)
        df_india[f"{t_key}_failure"] = 1 - df_india[f"{t_key}_success"]

    # Target Prevalence Summary
    prevalence_records = []
    for d_name, d_df in [("China_Drone_Val", df_china), ("India_Cross_Domain", df_india)]:
        for t_key, t_info in TARGET_DEFS.items():
            n_succ = int(d_df[f"{t_key}_success"].sum())
            n_fail = int(d_df[f"{t_key}_failure"].sum())
            n_tot = len(d_df)
            prevalence_records.append({
                "dataset": d_name,
                "target": t_key,
                "target_name": t_info["name"],
                "target_description": t_info["desc"],
                "total_samples": n_tot,
                "success_count": n_succ,
                "failure_count": n_fail,
                "success_pct": round(n_succ / n_tot * 100, 2),
                "failure_pct": round(n_fail / n_tot * 100, 2),
            })
    df_prev = pd.DataFrame(prevalence_records)
    df_prev.to_csv(OUT_TABLES_DIR / "table_target_prevalence.csv", index=False)
    log.info("Saved target prevalence to table_target_prevalence.csv")

    # 3. Comprehensive Signal Ablation (Models A to H across T0 - T4)
    ablation_records = []
    raw_predictions_china = {}
    raw_predictions_india = {}
    fitted_models = {}

    for t_key in ["T0", "T1", "T2", "T3", "T4"]:
        y_china_fail = df_china[f"{t_key}_failure"].values.astype(int)
        y_india_fail = df_india[f"{t_key}_failure"].values.astype(int)

        for m_key, (m_name, feat_cols) in FEATURE_GROUPS.items():
            X_china = df_china[feat_cols].values
            X_india = df_india[feat_cols].values

            # --- A. In-Domain China 5-Fold Stratified Cross-Validation ---
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_fail_prob = np.zeros(len(df_china), dtype=float)

            for train_idx, val_idx in skf.split(X_china, y_china_fail):
                X_tr, X_val = X_china[train_idx], X_china[val_idx]
                y_tr, y_val = y_china_fail[train_idx], y_china_fail[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)

                clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=42, max_iter=500)
                clf.fit(X_tr_s, y_tr)
                oof_fail_prob[val_idx] = clf.predict_proba(X_val_s)[:, 1]

            # Find optimal threshold on development out-of-fold predictions
            opt_thresh = find_optimal_threshold(y_china_fail, oof_fail_prob)
            china_metrics = evaluate_metrics(y_china_fail, oof_fail_prob, threshold=opt_thresh)
            china_ci = bootstrap_ci(y_china_fail, oof_fail_prob, n_bootstraps=1000, seed=42)

            raw_predictions_china[f"{m_key}_{t_key}"] = oof_fail_prob
            df_china[f"prob_fail_{m_key}_{t_key}"] = oof_fail_prob
            df_china[f"reliability_{m_key}_{t_key}"] = 1.0 - oof_fail_prob

            # Record In-Domain Row
            ablation_records.append({
                "model": m_key,
                "model_name": m_name,
                "feature_groups": ", ".join(feat_cols),
                "target": t_key,
                "target_name": TARGET_DEFS[t_key]["name"],
                "dataset": "China_Drone_Val (In-Domain CV)",
                "N": len(df_china),
                "AUROC": china_metrics["AUROC"],
                "AUROC_95CI_low": china_ci["AUROC_CI"][0],
                "AUROC_95CI_high": china_ci["AUROC_CI"][1],
                "AUPRC": china_metrics["AUPRC"],
                "AUPRC_95CI_low": china_ci["AUPRC_CI"][0],
                "AUPRC_95CI_high": china_ci["AUPRC_CI"][1],
                "optimal_threshold": round(opt_thresh, 3),
                "balanced_accuracy": china_metrics["balanced_accuracy"],
                "failure_precision": china_metrics["failure_precision"],
                "failure_recall": china_metrics["failure_recall"],
                "failure_F1": china_metrics["failure_F1"],
                "Brier": china_metrics["Brier"],
                "ECE": china_metrics["ECE"],
            })

            # --- B. Cross-Domain Zero-Shot Evaluation on India (Trained on China) ---
            full_scaler = StandardScaler()
            X_china_s = full_scaler.fit_transform(X_china)
            X_india_s = full_scaler.transform(X_india)

            final_clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=42, max_iter=500)
            final_clf.fit(X_china_s, y_china_fail)

            india_fail_prob = final_clf.predict_proba(X_india_s)[:, 1]
            india_metrics = evaluate_metrics(y_india_fail, india_fail_prob, threshold=opt_thresh)
            india_ci = bootstrap_ci(y_india_fail, india_fail_prob, n_bootstraps=1000, seed=42)

            raw_predictions_india[f"{m_key}_{t_key}"] = india_fail_prob
            df_india[f"prob_fail_{m_key}_{t_key}"] = india_fail_prob
            df_india[f"reliability_{m_key}_{t_key}"] = 1.0 - india_fail_prob
            fitted_models[f"{m_key}_{t_key}"] = (final_clf, full_scaler)

            # Record Cross-Domain Row
            ablation_records.append({
                "model": m_key,
                "model_name": m_name,
                "feature_groups": ", ".join(feat_cols),
                "target": t_key,
                "target_name": TARGET_DEFS[t_key]["name"],
                "dataset": "India_Cross_Domain (Zero-Shot Transfer)",
                "N": len(df_india),
                "AUROC": india_metrics["AUROC"],
                "AUROC_95CI_low": india_ci["AUROC_CI"][0],
                "AUROC_95CI_high": india_ci["AUROC_CI"][1],
                "AUPRC": india_metrics["AUPRC"],
                "AUPRC_95CI_low": india_ci["AUPRC_CI"][0],
                "AUPRC_95CI_high": india_ci["AUPRC_CI"][1],
                "optimal_threshold": round(opt_thresh, 3),
                "balanced_accuracy": india_metrics["balanced_accuracy"],
                "failure_precision": india_metrics["failure_precision"],
                "failure_recall": india_metrics["failure_recall"],
                "failure_F1": india_metrics["failure_F1"],
                "Brier": india_metrics["Brier"],
                "ECE": india_metrics["ECE"],
            })

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(OUT_TABLES_DIR / "table_full_ablation.csv", index=False)
    log.info("Saved full ablation results to table_full_ablation.csv (Total rows: %d)", len(df_ablation))

    # 4. Target Robustness Summary Table
    robustness_records = []
    for t_key in ["T0", "T1", "T2", "T3", "T4"]:
        # Find best model on China by AUROC
        sub_china = df_ablation[(df_ablation["dataset"].str.contains("China")) & (df_ablation["target"] == t_key)]
        best_row = sub_china.sort_values(by="AUROC", ascending=False).iloc[0]
        best_model_key = best_row["model"]
        
        # Reliability band performance on best model
        rel_scores_china = df_china[f"reliability_{best_model_key}_{t_key}"].values
        bands_china = compute_reliability_bands(
            df_china[f"{t_key}_failure"].values, rel_scores_china, high_th=0.85, low_th=0.60
        )

        n_succ = int(df_china[f"{t_key}_success"].sum())
        n_fail = int(df_china[f"{t_key}_failure"].sum())

        robustness_records.append({
            "target": t_key,
            "definition": TARGET_DEFS[t_key]["desc"],
            "china_success_count": n_succ,
            "china_failure_count": n_fail,
            "china_failure_rate_pct": round(n_fail / len(df_china) * 100, 2),
            "best_model": best_model_key,
            "best_model_name": best_row["model_name"],
            "AUROC": best_row["AUROC"],
            "AUROC_95CI": f"[{best_row['AUROC_95CI_low']:.4f}, {best_row['AUROC_95CI_high']:.4f}]",
            "AUPRC": best_row["AUPRC"],
            "AUPRC_95CI": f"[{best_row['AUPRC_95CI_low']:.4f}, {best_row['AUPRC_95CI_high']:.4f}]",
            "balanced_accuracy": best_row["balanced_accuracy"],
            "failure_F1": best_row["failure_F1"],
            "high_band_success_rate": bands_china["HIGH"]["success_rate"],
            "high_band_sample_count": bands_china["HIGH"]["count"],
            "low_band_failure_rate": bands_china["LOW"]["failure_rate"],
            "low_band_sample_count": bands_china["LOW"]["count"],
        })
    df_robustness = pd.DataFrame(robustness_records)
    df_robustness.to_csv(OUT_TABLES_DIR / "table_target_robustness.csv", index=False)
    log.info("Saved target robustness table to table_target_robustness.csv")

    # 5. Multi-Target Reliability Bands Table (Original vs Recalibrated Bands)
    band_records = []
    for d_name, d_df in [("China_Drone_Val", df_china), ("India_Cross_Domain", df_india)]:
        for t_key in ["T0", "T1", "T2", "T3", "T4"]:
            # Evaluate using Model H (Full Combined) and Model B (Confidence Only)
            for m_key in ["Model_B", "Model_H"]:
                rel_scores = d_df[f"reliability_{m_key}_{t_key}"].values
                y_fail = d_df[f"{t_key}_failure"].values
                
                # Original Phase-8 bands: HIGH >= 0.85, MED [0.60, 0.85), LOW < 0.60
                orig_bands = compute_reliability_bands(y_fail, rel_scores, high_th=0.85, low_th=0.60)
                
                # Development-calibrated bands (China dev quartiles/percentiles e.g. p80, p40)
                p80 = float(np.percentile(df_china[f"reliability_{m_key}_{t_key}"], 75))
                p40 = float(np.percentile(df_china[f"reliability_{m_key}_{t_key}"], 25))
                recal_bands = compute_reliability_bands(y_fail, rel_scores, high_th=p80, low_th=p40)

                for b_type, b_dict, h_thresh, l_thresh in [
                    ("Original_Phase8_Bands", orig_bands, 0.85, 0.60),
                    ("Dev_Recalibrated_Bands", recal_bands, round(p80, 3), round(p40, 3)),
                ]:
                    band_records.append({
                        "dataset": d_name,
                        "target": t_key,
                        "model": m_key,
                        "band_calibration_type": b_type,
                        "high_threshold": h_thresh,
                        "low_threshold": l_thresh,
                        "high_band_count": b_dict["HIGH"]["count"],
                        "high_band_success_rate": b_dict["HIGH"]["success_rate"],
                        "medium_band_count": b_dict["MEDIUM"]["count"],
                        "medium_band_success_rate": b_dict["MEDIUM"]["success_rate"],
                        "low_band_count": b_dict["LOW"]["count"],
                        "low_band_failure_rate": b_dict["LOW"]["failure_rate"],
                    })
    df_bands = pd.DataFrame(band_records)
    df_bands.to_csv(OUT_TABLES_DIR / "table_reliability_bands_multitarget.csv", index=False)
    log.info("Saved reliability bands validation to table_reliability_bands_multitarget.csv")

    # 6. Risk-Coverage Selective Prediction for Strict Targets (T0, T1, T3)
    risk_cov_records = []
    for d_name, d_df in [("China_Drone_Val", df_china), ("India_Cross_Domain", df_india)]:
        for t_key in ["T0", "T1", "T3"]:
            # Evaluate using Model B (Confidence) and Model H (Full Combined)
            for m_key in ["Model_B", "Model_H"]:
                rel_scores = d_df[f"reliability_{m_key}_{t_key}"].values
                y_fail = d_df[f"{t_key}_failure"].values
                rows = compute_risk_coverage(d_df, y_fail, rel_scores, target_key=t_key)
                for r in rows:
                    r["dataset"] = d_name
                    r["model"] = m_key
                    risk_cov_records.append(r)
    df_risk_cov = pd.DataFrame(risk_cov_records)
    df_risk_cov.to_csv(OUT_TABLES_DIR / "table_risk_coverage_strict.csv", index=False)
    log.info("Saved strict risk-coverage table to table_risk_coverage_strict.csv")

    # 7. Save Consolidated Master Datasets
    df_china.to_csv(OUT_DATA_DIR / "china_validation_ablation_master.csv", index=False)
    df_india.to_csv(OUT_DATA_DIR / "india_cross_domain_ablation_master.csv", index=False)

    # 8. Comparison with Tree-based Random Forest on China Development Set
    rf_records = []
    for t_key in ["T0", "T1", "T2", "T3", "T4"]:
        y_china_fail = df_china[f"{t_key}_failure"].values.astype(int)
        for m_key, (m_name, feat_cols) in FEATURE_GROUPS.items():
            X_china = df_china[feat_cols].values
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_fail_prob = np.zeros(len(df_china), dtype=float)

            for train_idx, val_idx in skf.split(X_china, y_china_fail):
                X_tr, X_val = X_china[train_idx], X_china[val_idx]
                y_tr, y_val = y_china_fail[train_idx], y_china_fail[val_idx]

                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                rf.fit(X_tr, y_tr)
                oof_fail_prob[val_idx] = rf.predict_proba(X_val)[:, 1]

            metrics_rf = evaluate_metrics(y_china_fail, oof_fail_prob)
            rf_records.append({
                "model": m_key,
                "model_name": m_name,
                "classifier": "RandomForest",
                "target": t_key,
                "AUROC": metrics_rf["AUROC"],
                "AUPRC": metrics_rf["AUPRC"],
                "balanced_accuracy": metrics_rf["balanced_accuracy"],
                "failure_F1": metrics_rf["failure_F1"],
                "Brier": metrics_rf["Brier"],
                "ECE": metrics_rf["ECE"],
            })
    df_rf = pd.DataFrame(rf_records)
    df_rf.to_csv(OUT_TABLES_DIR / "table_random_forest_comparison.csv", index=False)
    log.info("Saved Random Forest comparison to table_random_forest_comparison.csv")

    log.info("Phase 12 Core Ablation & Evaluation Complete.")


if __name__ == "__main__":
    main()
