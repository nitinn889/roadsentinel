r"""Phase 1C: Temporal-Residual XGBoost Forecasting Pipeline.

Implements the complete modeling, ablation, nested grouped cross-validation,
and operational decision logic mandated by RoadSentinel Prompt 1C:
- M0: Persistence baseline (\Delta y = 0)
- M1: OLS Linear Regression (current condition + scenario)
- M2: XGBoost Residual (current condition + scenario)
- M3: XGBoost Residual (current condition + temporal history)
- M4: XGBoost Residual (current condition + temporal history + scenario)

Nested Grouped Cross-Validation:
- Outer: 5-fold GroupKFold by site_id (unbiased out-of-site generalization)
- Inner: 4-fold GroupKFold by site_id (target selection & hyperparameter tuning)
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("phase1c_forecasting")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = WORKSPACE_ROOT / "artifacts" / "phase1c" / "temporal_feature_manifest.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "artifacts" / "phase1c"

PREV_AUDIT_SITES = {"02-1002", "15-1006", "36-1008", "36-4017", "38-3006", "38-5002"}

SCENARIO_FEATURES = [
    "rainfall_level",
    "traffic_level",
    "temperature",
    "water_exposure",
    "days_ahead",
]

TEMPORAL_FEATURES = [
    "num_prior_inspections",
    "time_since_first_observation_days",
    "time_since_prev_inspection_days",
    "prev_severity",
    "second_prev_severity",
    "most_recent_severity_change",
    "annualized_severity_slope",
    "rolling_severity_mean",
    "rolling_severity_std",
    "same_construction_no",
]


def compute_severity_scaler(train_iris: np.ndarray) -> Tuple[float, float]:
    """Computes robust empirical quantiles on training fold IRIs."""
    q_low = float(np.quantile(train_iris, 0.01))
    q_high = float(np.quantile(train_iris, 0.99))
    if not (q_high > q_low):
        q_low = float(np.min(train_iris))
        q_high = float(np.max(train_iris)) + 1e-6
    return q_low, q_high


def scale_severity(iri: float, q_low: float, q_high: float) -> float:
    if np.isnan(iri):
        return np.nan
    return float(np.clip((iri - q_low) / (q_high - q_low), 0.0, 1.0))


def extract_features_and_targets(
    df: pd.DataFrame, q_low: float, q_high: float
) -> pd.DataFrame:
    """Prepares model feature matrices with fold-specific quantile scaling."""
    df_feat = df.copy()
    
    # Scale current and future severities
    df_feat["current_severity"] = df_feat["current_iri_m_per_km"].apply(
        lambda v: scale_severity(v, q_low, q_high)
    )
    df_feat["future_severity"] = df_feat["future_iri_m_per_km"].apply(
        lambda v: scale_severity(v, q_low, q_high)
    )
    
    # Direct severity change
    df_feat["delta_severity"] = df_feat["future_severity"] - df_feat["current_severity"]
    
    # Annualized severity change rate
    years_ahead = df_feat["observed_days_ahead"] / 365.25
    df_feat["annualized_change_rate"] = df_feat["delta_severity"] / years_ahead
    
    # Scale temporal features
    df_feat["prev_severity"] = df_feat["prev_iri_m_per_km"].apply(
        lambda v: scale_severity(v, q_low, q_high)
    )
    df_feat["second_prev_severity"] = df_feat["second_prev_iri_m_per_km"].apply(
        lambda v: scale_severity(v, q_low, q_high)
    )
    df_feat["rolling_severity_mean"] = df_feat["rolling_iri_mean"].apply(
        lambda v: scale_severity(v, q_low, q_high)
    )
    
    # Changes scaled by IRI scale factor
    scale_range = q_high - q_low
    df_feat["most_recent_severity_change"] = df_feat["most_recent_iri_change"] / scale_range
    df_feat["annualized_severity_slope"] = df_feat["annualized_iri_slope"] / scale_range
    df_feat["rolling_severity_std"] = df_feat["rolling_iri_std"] / scale_range
    
    return df_feat


def run_inner_cv(
    train_df: pd.DataFrame, q_low: float, q_high: float
) -> Tuple[str, Dict[str, Any]]:
    """Runs 4-fold GroupKFold inner CV on outer training sites.
    
    Evaluates Target A (direct delta) vs Target B (annualized rate)
    and searches a conservative hyperparameter grid for M4.
    """
    gkf_inner = GroupKFold(n_splits=4)
    sites = train_df["site_id"].values
    
    param_grid = [
        {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 80, "reg_lambda": 5.0, "subsample": 0.85},
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 100, "reg_lambda": 4.0, "subsample": 0.90},
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 80, "reg_lambda": 5.0, "subsample": 0.85},
        {"max_depth": 3, "learning_rate": 0.04, "n_estimators": 120, "reg_lambda": 4.0, "subsample": 0.90},
    ]
    
    targets = ["direct_delta", "annualized_rate"]
    
    m4_features = ["current_severity"] + SCENARIO_FEATURES + TEMPORAL_FEATURES
    
    best_target = "direct_delta"
    best_params = param_grid[0]
    best_inner_mae = float("inf")
    
    for target_name in targets:
        for params in param_grid:
            fold_maes = []
            
            for in_train_idx, in_val_idx in gkf_inner.split(train_df, groups=sites):
                in_train = train_df.iloc[in_train_idx]
                in_val = train_df.iloc[in_val_idx]
                
                # Model inputs
                x_in_train = in_train[m4_features].values
                x_in_val = in_val[m4_features].values
                
                years_val = (in_val["observed_days_ahead"] / 365.25).values
                c_sev_val = in_val["current_severity"].values
                f_sev_val = in_val["future_severity"].values
                
                xgb = XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                    **params
                )
                
                if target_name == "direct_delta":
                    y_in_train = in_train["delta_severity"].values
                    xgb.fit(x_in_train, y_in_train)
                    pred_delta = xgb.predict(x_in_val)
                else:
                    y_in_train = in_train["annualized_change_rate"].values
                    xgb.fit(x_in_train, y_in_train)
                    pred_rate = xgb.predict(x_in_val)
                    pred_delta = pred_rate * years_val
                    
                pred_fut = np.clip(c_sev_val + pred_delta, 0.0, 1.0)
                mae = float(mean_absolute_error(f_sev_val, pred_fut))
                fold_maes.append(mae)
                
            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_inner_mae:
                best_inner_mae = mean_mae
                best_target = target_name
                best_params = params
                
    log.info("Inner CV selected target: %s, best MAE: %.4f, params: %s", best_target, best_inner_mae, best_params)
    return best_target, best_params


def evaluate_forecasts(
    manifest_path: Path = MANIFEST_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    log.info("Starting Phase 1C Nested Grouped Forecasting Pipeline...")
    raw_df = pd.read_csv(manifest_path)
    
    # 5-fold outer GroupKFold across all 23 sites
    sites = raw_df["site_id"].values
    gkf_outer = GroupKFold(n_splits=5)
    
    oof_rows: List[Dict[str, Any]] = []
    fold_assignment_rows: List[Dict[str, Any]] = []
    
    m1_covariates = ["current_severity"] + SCENARIO_FEATURES
    m2_features = ["current_severity"] + SCENARIO_FEATURES
    m3_features = ["current_severity"] + TEMPORAL_FEATURES
    m4_features = ["current_severity"] + SCENARIO_FEATURES + TEMPORAL_FEATURES
    
    outer_fold_configs = []
    
    for fold_idx, (out_train_idx, out_test_idx) in enumerate(gkf_outer.split(raw_df, groups=sites)):
        raw_train = raw_df.iloc[out_train_idx].copy()
        raw_test = raw_df.iloc[out_test_idx].copy()
        
        train_sites = set(raw_train["site_id"].unique())
        test_sites = set(raw_test["site_id"].unique())
        assert train_sites.isdisjoint(test_sites), f"Leakage: Fold {fold_idx} sites overlap!"
        
        for s in test_sites:
            fold_assignment_rows.append({"site_id": s, "outer_fold": fold_idx, "sample_count": (raw_test["site_id"] == s).sum()})
            
        # Compute severity scale strictly on training sites
        all_train_iris = np.concatenate([raw_train["current_iri_m_per_km"].values, raw_train["future_iri_m_per_km"].values])
        q_low, q_high = compute_severity_scaler(all_train_iris)
        
        # Scale train and test sets
        train_df = extract_features_and_targets(raw_train, q_low, q_high)
        test_df = extract_features_and_targets(raw_test, q_low, q_high)
        
        # Inner CV on outer train for target & hyperparam selection
        best_target, best_params = run_inner_cv(train_df, q_low, q_high)
        outer_fold_configs.append({"fold": fold_idx, "best_target": best_target, "best_params": best_params})
        
        years_train = (train_df["observed_days_ahead"] / 365.25).values
        years_test = (test_df["observed_days_ahead"] / 365.25).values
        
        y_train_target = train_df["delta_severity"].values if best_target == "direct_delta" else train_df["annualized_change_rate"].values
        
        # 1. Model M0: Persistence
        # \Delta y = 0 => pred_future = current_severity
        m0_pred_delta = np.zeros(len(test_df), dtype=float)
        m0_pred_future = test_df["current_severity"].values.copy()
        
        # 2. Model M1: OLS Linear Regression (Current condition + Scenario)
        # OLS requires non-missing values; scenario features have zero missing values
        ols = LinearRegression()
        x_train_m1 = train_df[m1_covariates].values
        x_test_m1 = test_df[m1_covariates].values
        
        if best_target == "direct_delta":
            ols.fit(x_train_m1, train_df["delta_severity"].values)
            m1_pred_delta = ols.predict(x_test_m1)
        else:
            ols.fit(x_train_m1, train_df["annualized_change_rate"].values)
            m1_pred_delta = ols.predict(x_test_m1) * years_test
            
        m1_pred_future = np.clip(test_df["current_severity"].values + m1_pred_delta, 0.0, 1.0)
        
        # 3. Model M2: XGBoost (Current condition + Scenario)
        xgb_m2 = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            **best_params
        )
        xgb_m2.fit(train_df[m2_features].values, y_train_target)
        if best_target == "direct_delta":
            m2_pred_delta = xgb_m2.predict(test_df[m2_features].values)
        else:
            m2_pred_delta = xgb_m2.predict(test_df[m2_features].values) * years_test
        m2_pred_future = np.clip(test_df["current_severity"].values + m2_pred_delta, 0.0, 1.0)
        
        # 4. Model M3: XGBoost (Current condition + Temporal History)
        xgb_m3 = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            **best_params
        )
        xgb_m3.fit(train_df[m3_features].values, y_train_target)
        if best_target == "direct_delta":
            m3_pred_delta = xgb_m3.predict(test_df[m3_features].values)
        else:
            m3_pred_delta = xgb_m3.predict(test_df[m3_features].values) * years_test
        m3_pred_future = np.clip(test_df["current_severity"].values + m3_pred_delta, 0.0, 1.0)
        
        # 5. Model M4: XGBoost (Current condition + Temporal History + Scenario)
        xgb_m4 = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            **best_params
        )
        xgb_m4.fit(train_df[m4_features].values, y_train_target)
        if best_target == "direct_delta":
            m4_pred_delta = xgb_m4.predict(test_df[m4_features].values)
        else:
            m4_pred_delta = xgb_m4.predict(test_df[m4_features].values) * years_test
        m4_pred_future = np.clip(test_df["current_severity"].values + m4_pred_delta, 0.0, 1.0)
        
        # Collect out-of-fold prediction rows
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            oof_rows.append({
                "pair_id": row["pair_id"],
                "site_id": row["site_id"],
                "outer_fold": fold_idx,
                "is_previously_observed_audit_site": row["site_id"] in PREV_AUDIT_SITES,
                "current_date": row["current_date"],
                "future_date": row["future_date"],
                "observed_days_ahead": row["observed_days_ahead"],
                "days_ahead": row["days_ahead"],
                "current_severity": row["current_severity"],
                "future_severity_true": row["future_severity"],
                "delta_severity_true": row["delta_severity"],
                # M0
                "m0_persistence_pred_future": m0_pred_future[i],
                "m0_persistence_pred_delta": m0_pred_delta[i],
                "m0_error": abs(m0_pred_future[i] - row["future_severity"]),
                # M1
                "m1_ols_pred_future": m1_pred_future[i],
                "m1_ols_pred_delta": m1_pred_delta[i],
                "m1_error": abs(m1_pred_future[i] - row["future_severity"]),
                # M2
                "m2_xgb_scenario_pred_future": m2_pred_future[i],
                "m2_xgb_scenario_pred_delta": m2_pred_delta[i],
                "m2_error": abs(m2_pred_future[i] - row["future_severity"]),
                # M3
                "m3_xgb_temporal_pred_future": m3_pred_future[i],
                "m3_xgb_temporal_pred_delta": m3_pred_delta[i],
                "m3_error": abs(m3_pred_future[i] - row["future_severity"]),
                # M4
                "m4_xgb_combined_pred_future": m4_pred_future[i],
                "m4_xgb_combined_pred_delta": m4_pred_delta[i],
                "m4_error": abs(m4_pred_future[i] - row["future_severity"]),
                # Paired difference M4 vs Persistence (< 0 means M4 wins)
                "paired_diff_m4_minus_m0": abs(m4_pred_future[i] - row["future_severity"]) - abs(m0_pred_future[i] - row["future_severity"]),
                # Lag availability
                "num_prior_inspections": row["num_prior_inspections"],
                "has_at_least_2_prior": row["has_at_least_2_prior"],
            })
            
    oof_df = pd.DataFrame(oof_rows)
    fold_df = pd.DataFrame(fold_assignment_rows)
    
    # Save OOF and Fold assignments
    oof_df.to_csv(OUTPUT_DIR / "out_of_fold_predictions.csv", index=False)
    fold_df.to_csv(OUTPUT_DIR / "fold_assignments.csv", index=False)
    log.info("Saved out-of-fold predictions (%d rows) and fold assignments to %s", len(oof_df), OUTPUT_DIR)
    
    # -------------------------------------------------------------------------
    # Model Comparison & Ablation Metrics
    # -------------------------------------------------------------------------
    models = [
        ("M0_Persistence", "m0_persistence_pred_future", "m0_persistence_pred_delta"),
        ("M1_OLS_Scenario", "m1_ols_pred_future", "m1_ols_pred_delta"),
        ("M2_XGB_Scenario", "m2_xgb_scenario_pred_future", "m2_xgb_scenario_pred_delta"),
        ("M3_XGB_Temporal", "m3_xgb_temporal_pred_future", "m3_xgb_temporal_pred_delta"),
        ("M4_XGB_Combined", "m4_xgb_combined_pred_future", "m4_xgb_combined_pred_delta"),
    ]
    
    y_true = oof_df["future_severity_true"].values
    delta_true = oof_df["delta_severity_true"].values
    
    m0_mae = float(mean_absolute_error(y_true, oof_df["m0_persistence_pred_future"].values))
    m0_rmse = float(np.sqrt(mean_squared_error(y_true, oof_df["m0_persistence_pred_future"].values)))
    
    comparison_records: List[Dict[str, Any]] = []
    
    for model_name, pred_future_col, pred_delta_col in models:
        y_pred = oof_df[pred_future_col].values
        delta_pred = oof_df[pred_delta_col].values
        
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        delta_mae = float(mean_absolute_error(delta_true, delta_pred))
        
        mae_skill = 1.0 - (mae / m0_mae)
        paired_delta_mae = mae - m0_mae
        paired_delta_rmse = rmse - m0_rmse
        
        # Site win rate
        site_wins = 0
        total_sites = oof_df["site_id"].nunique()
        for site_id, site_grp in oof_df.groupby("site_id"):
            site_y = site_grp["future_severity_true"].values
            site_m = site_grp[pred_future_col].values
            site_m0 = site_grp["m0_persistence_pred_future"].values
            if mean_absolute_error(site_y, site_m) < mean_absolute_error(site_y, site_m0):
                site_wins += 1
        site_win_rate_pct = round((site_wins / total_sites) * 100.0, 2)
        
        # Direction accuracy
        if model_name == "M0_Persistence":
            direction_acc_pct = 0.0  # persistence predicts 0 change
        else:
            # Sign match where delta != 0
            nonzero = delta_true != 0.0
            if nonzero.sum() > 0:
                dir_match = (np.sign(delta_pred[nonzero]) == np.sign(delta_true[nonzero]))
                direction_acc_pct = round(float(dir_match.mean() * 100.0), 2)
            else:
                direction_acc_pct = 0.0
                
        # Spearman correlation
        if model_name == "M0_Persistence":
            spearman_corr = 0.0
        else:
            spearman_corr = float(spearmanr(delta_pred, delta_true).correlation)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
                
        comparison_records.append({
            "model_id": model_name,
            "inputs": (
                "None (Current state carried forward)" if "M0" in model_name else
                "Current condition + Scenario" if "M1" in model_name or "M2" in model_name else
                "Current condition + Temporal history" if "M3" in model_name else
                "Current condition + Temporal history + Scenario"
            ),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "change_mae": round(delta_mae, 4),
            "mae_skill_vs_persistence": round(mae_skill, 4),
            "paired_delta_mae": round(paired_delta_mae, 4),
            "paired_delta_rmse": round(paired_delta_rmse, 4),
            "site_win_rate_pct": site_win_rate_pct,
            "direction_accuracy_pct": direction_acc_pct,
            "spearman_rank_corr": round(spearman_corr, 4),
            "outperforms_persistence": mae < m0_mae,
        })
        
    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    log.info("Saved model comparison to %s", OUTPUT_DIR / "model_comparison.csv")
    
    # -------------------------------------------------------------------------
    # Paired Per-Site Metrics & Site Breakdown
    # -------------------------------------------------------------------------
    site_records: List[Dict[str, Any]] = []
    for site_id, grp in oof_df.groupby("site_id"):
        y_site = grp["future_severity_true"].values
        y_m0 = grp["m0_persistence_pred_future"].values
        y_m1 = grp["m1_ols_pred_future"].values
        y_m4 = grp["m4_xgb_combined_pred_future"].values
        
        m0_site_mae = float(mean_absolute_error(y_site, y_m0))
        m1_site_mae = float(mean_absolute_error(y_site, y_m1))
        m4_site_mae = float(mean_absolute_error(y_site, y_m4))
        
        delta_m4_minus_m0 = m4_site_mae - m0_site_mae
        
        site_records.append({
            "site_id": site_id,
            "stratum": "Previously_Observed_Audit_Site" if site_id in PREV_AUDIT_SITES else "Unseen_Generalization_Site",
            "sample_count": len(grp),
            "mean_days_ahead": round(float(grp["observed_days_ahead"].mean()), 1),
            "m0_persistence_mae": round(m0_site_mae, 4),
            "m1_ols_mae": round(m1_site_mae, 4),
            "m4_xgb_mae": round(m4_site_mae, 4),
            "paired_delta_mae_m4_vs_m0": round(delta_m4_minus_m0, 4),
            "m4_beats_persistence": m4_site_mae < m0_site_mae,
            "best_performing_model": (
                "M4_XGB" if (m4_site_mae < m0_site_mae and m4_site_mae < m1_site_mae)
                else ("M1_OLS" if m1_site_mae < m0_site_mae else "M0_Persistence")
            ),
        })
        
    site_df = pd.DataFrame(site_records)
    site_df.to_csv(OUTPUT_DIR / "paired_site_metrics.csv", index=False)
    log.info("Saved paired site metrics to %s", OUTPUT_DIR / "paired_site_metrics.csv")
    
    # -------------------------------------------------------------------------
    # Site-Clustered Bootstrap 95% CI on Paired Difference (M4 vs M0)
    # -------------------------------------------------------------------------
    unique_sites = list(oof_df["site_id"].unique())
    n_sites = len(unique_sites)
    rng = np.random.default_rng(42)
    B = 2000
    
    site_grouped_errors = {s: grp["paired_diff_m4_minus_m0"].values for s, grp in oof_df.groupby("site_id")}
    
    boot_paired_deltas = []
    for _ in range(B):
        sample_sites = rng.choice(unique_sites, size=n_sites, replace=True)
        boot_diffs = np.concatenate([site_grouped_errors[s] for s in sample_sites])
        boot_paired_deltas.append(float(np.mean(boot_diffs)))
        
    boot_paired_deltas = np.sort(boot_paired_deltas)
    ci_lower = float(np.percentile(boot_paired_deltas, 2.5))
    ci_upper = float(np.percentile(boot_paired_deltas, 97.5))
    point_est = float(np.mean(oof_df["paired_diff_m4_minus_m0"].values))
    
    log.info("Site-clustered 95%% CI for Paired Delta MAE (M4 - M0): %.4f [%.4f, %.4f]", point_est, ci_lower, ci_upper)
    
    # Horizon Breakdown
    horizon_groups = {
        "Short_Horizon_lt_365d": oof_df[oof_df["observed_days_ahead"] < 365],
        "Medium_Horizon_365_550d": oof_df[(oof_df["observed_days_ahead"] >= 365) & (oof_df["observed_days_ahead"] <= 550)],
        "Long_Horizon_gt_550d": oof_df[oof_df["observed_days_ahead"] > 550],
    }
    horizon_metrics = {}
    for h_name, h_df in horizon_groups.items():
        if len(h_df) > 0:
            h_y = h_df["future_severity_true"].values
            h_m0 = float(mean_absolute_error(h_y, h_df["m0_persistence_pred_future"].values))
            h_m4 = float(mean_absolute_error(h_y, h_df["m4_xgb_combined_pred_future"].values))
            horizon_metrics[h_name] = {
                "sample_count": len(h_df),
                "m0_mae": round(h_m0, 4),
                "m4_mae": round(h_m4, 4),
                "m4_skill_vs_m0": round(1.0 - (h_m4 / h_m0), 4),
                "m4_wins": h_m4 < h_m0,
            }
            
    # Subgroup: >= 2 prior inspections
    df_lag2 = oof_df[oof_df["has_at_least_2_prior"]]
    lag2_m0_mae = float(mean_absolute_error(df_lag2["future_severity_true"], df_lag2["m0_persistence_pred_future"]))
    lag2_m4_mae = float(mean_absolute_error(df_lag2["future_severity_true"], df_lag2["m4_xgb_combined_pred_future"]))
    
    forecast_skill_summary = {
        "evaluation_protocol": "Nested Grouped Cross-Validation (5 outer folds, 4 inner folds, grouped by site_id)",
        "total_pairs_evaluated": len(oof_df),
        "total_sites_evaluated": n_sites,
        "m0_persistence_baseline_mae": round(m0_mae, 4),
        "m0_persistence_baseline_rmse": round(m0_rmse, 4),
        "m4_proposed_xgboost_mae": round(float(comp_df[comp_df["model_id"] == "M4_XGB_Combined"]["mae"].values[0]), 4),
        "m4_proposed_xgboost_rmse": round(float(comp_df[comp_df["model_id"] == "M4_XGB_Combined"]["rmse"].values[0]), 4),
        "m4_mae_skill": round(float(comp_df[comp_df["model_id"] == "M4_XGB_Combined"]["mae_skill_vs_persistence"].values[0]), 4),
        "paired_difference_m4_minus_m0": {
            "point_estimate_delta_mae": round(point_est, 4),
            "site_clustered_95ci_lower": round(ci_lower, 4),
            "site_clustered_95ci_upper": round(ci_upper, 4),
            "ci_spans_zero": bool(ci_lower <= 0 <= ci_upper),
            "statistically_significantly_better_than_persistence": bool(ci_upper < 0),
        },
        "site_level_performance": {
            "total_sites": n_sites,
            "sites_won_by_m4": int((site_df["m4_beats_persistence"]).sum()),
            "site_win_rate_pct": round(float((site_df["m4_beats_persistence"]).mean() * 100), 2),
            "audit_set_6_sites_win_rate_pct": round(float(site_df[site_df["stratum"] == "Previously_Observed_Audit_Site"]["m4_beats_persistence"].mean() * 100), 2),
            "unseen_17_sites_win_rate_pct": round(float(site_df[site_df["stratum"] == "Unseen_Generalization_Site"]["m4_beats_persistence"].mean() * 100), 2),
        },
        "performance_by_horizon": horizon_metrics,
        "performance_on_full_lag_history_subset_n78": {
            "sample_count": len(df_lag2),
            "m0_mae": round(lag2_m0_mae, 4),
            "m4_mae": round(lag2_m4_mae, 4),
            "m4_skill": round(1.0 - (lag2_m4_mae / lag2_m0_mae), 4),
            "m4_wins": lag2_m4_mae < lag2_m0_mae,
        },
        "model_ablation_summary": comparison_records,
    }
    
    with open(OUTPUT_DIR / "forecast_skill.json", "w", encoding="utf-8") as f:
        json.dump(forecast_skill_summary, f, indent=2)
    log.info("Saved forecast skill summary to %s", OUTPUT_DIR / "forecast_skill.json")
    
    # -------------------------------------------------------------------------
    # Predefined Operational Selection Decision
    # -------------------------------------------------------------------------
    m4_overall_skill = forecast_skill_summary["m4_mae_skill"]
    m4_site_win_rate = forecast_skill_summary["site_level_performance"]["site_win_rate_pct"]
    m4_ci_upper = forecast_skill_summary["paired_difference_m4_minus_m0"]["site_clustered_95ci_upper"]
    
    if m4_overall_skill > 0 and m4_ci_upper < 0 and m4_site_win_rate > 50.0:
        selection_verdict = "SELECT_M4_PRIMARY_FORECASTER"
        operational_role = "Primary Autonomous Forecaster"
        justification = (
            "M4 achieves positive out-of-site MAE skill over persistence, and site-clustered 95% CI strictly excludes zero, "
            "confirming statistically significant generalization improvement across highway sites."
        )
    elif any(h["m4_wins"] for h in horizon_metrics.values()):
        selection_verdict = "RESTRICTED_BOUNDED_OPERATING_REGION_OR_SCENARIO_ESTIMATE"
        operational_role = "Secondary Scenario-Conditioned Estimate (Persistence Retained as Baseline)"
        winning_horizons = [h_name for h_name, h in horizon_metrics.items() if h["m4_wins"]]
        justification = (
            f"M4 does not demonstrate universal superiority across all sites (Overall MAE skill: {m4_overall_skill:+.4f}, "
            f"Site Win Rate: {m4_site_win_rate:.1f}%, Paired 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
            f"Per Prompt 1C Predefined Operational Selection Rule, Persistence is retained as the authoritative baseline forecast. "
            f"XGBoost is exposed strictly as an experimental scenario-conditioned projection (with positive skill observed in {winning_horizons} if applicable)."
        )
    else:
        selection_verdict = "RETAIN_PERSISTENCE_AUTOMATIC_FORECAST"
        operational_role = "Experimental Scenario-Conditioned Estimate (Persistence Retained as Sole Primary)"
        justification = (
            f"Persistence strictly outperforms XGBoost across held-out evaluation folds (Overall MAE skill: {m4_overall_skill:+.4f}, "
            f"Paired Delta MAE 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
            "Per Prompt 1C Predefined Operational Selection Rule, Persistence is retained as the automatic forecast. "
            "XGBoost is relegated to scenario-sensitivity analysis and must never supersede persistence in production decision-making."
        )
        
    final_decision = {
        "predefined_rule_applied": "Prompt 1C Section 8 Operational Selection Rule",
        "verdict": selection_verdict,
        "assigned_operational_role_for_xgboost": operational_role,
        "authoritative_primary_forecast": "M0_Persistence" if "Persistence" in operational_role else "M4_XGBoost",
        "evidence_summary": {
            "overall_mae_skill": round(m4_overall_skill, 4),
            "paired_delta_mae_point_estimate": round(point_est, 4),
            "paired_delta_mae_95ci": [round(ci_lower, 4), round(ci_upper, 4)],
            "site_win_rate_pct": m4_site_win_rate,
            "audit_set_win_rate_pct": forecast_skill_summary["site_level_performance"]["audit_set_6_sites_win_rate_pct"],
            "unseen_sites_win_rate_pct": forecast_skill_summary["site_level_performance"]["unseen_17_sites_win_rate_pct"],
        },
        "justification": justification,
        "prohibited_actions": [
            "Do not select whichever model performs best separately on each test record",
            "Do not label XGBoost scenario sensitivities as causal or counterfactual interventions",
            "Do not claim validated 30-90 day rapid forecasting from historical LTPP multi-year survey intervals",
        ],
        "decision_timestamp": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_DIR / "final_model_decision.json", "w", encoding="utf-8") as f:
        json.dump(final_decision, f, indent=2)
    log.info("Saved final model decision to %s", OUTPUT_DIR / "final_model_decision.json")
    
    return oof_df, comp_df, forecast_skill_summary, final_decision


if __name__ == "__main__":
    evaluate_forecasts()
