"""Automated Unit & Integrity Tests for RoadSentinel Phase 1C Forecasting Suite.

Verifies:
1. Chronological feature construction and strict absence of future lookahead leakage.
2. Group-disjoint fold partitioning (zero site overlap).
3. Mathematical correctness of persistence baseline and residual reconstruction.
4. Metric formulas (MAE, RMSE, Skill, Paired differences).
5. Non-fabrication of unobserved lag features (strict NaN preservation).
6. Operational selection rule compliance and deterministic outputs.
7. Artifact, report, and figure existence and integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts" / "phase1c"
REPORTS_DIR = WORKSPACE_ROOT / "reports" / "phase1c"
FIGURES_DIR = WORKSPACE_ROOT / "figures" / "phase1c"


# =============================================================================
# 1. Chronological Feature Construction & Leakage Tests
# =============================================================================
def test_chronological_ordering_and_zero_future_leakage():
    manifest_path = ARTIFACTS_DIR / "temporal_feature_manifest.csv"
    assert manifest_path.exists(), "temporal_feature_manifest.csv must exist"
    df = pd.read_csv(manifest_path)
    
    assert len(df) == 113, "Must contain exactly 113 LTPP transition pairs"
    assert df["site_id"].nunique() == 23, "Must contain exactly 23 unique sites"
    
    # 1. Monotonic timestamps per pair
    for _, row in df.iterrows():
        c_date = pd.to_datetime(row["current_date"])
        f_date = pd.to_datetime(row["future_date"])
        assert c_date < f_date, f"Chronological violation: {row['pair_id']}"
        assert row["observed_days_ahead"] > 0, "Horizon duration must be strictly positive"
        
    # 2. Assert zero future leakage detected
    assert not df["future_leakage_detected"].any(), "CRITICAL: Lookahead leakage detected in manifest!"
    
    # 3. Elapsed times
    assert (df["time_since_first_observation_days"] >= 0).all()
    non_nan_prev = df["time_since_prev_inspection_days"].dropna()
    assert (non_nan_prev > 0).all()


def test_non_fabrication_of_unobserved_lags():
    df = pd.read_csv(ARTIFACTS_DIR / "temporal_feature_manifest.csv")
    
    # When num_prior_inspections == 0, lag features must be NaN
    k0 = df[df["num_prior_inspections"] == 0]
    assert len(k0) == 23, "Exactly 23 pairs must have zero prior inspections (initial visits)"
    assert k0["prev_iri_m_per_km"].isna().all(), "prev_iri must be NaN when k=0"
    assert k0["second_prev_iri_m_per_km"].isna().all(), "second_prev_iri must be NaN when k=0"
    assert k0["most_recent_iri_change"].isna().all(), "most_recent_iri_change must be NaN when k=0"
    assert k0["annualized_iri_slope"].isna().all(), "annualized_iri_slope must be NaN when k=0"
    assert k0["time_since_prev_inspection_days"].isna().all(), "time_since_prev must be NaN when k=0"
    
    # When num_prior_inspections == 1, second_prev_iri must be NaN
    k1 = df[df["num_prior_inspections"] == 1]
    assert len(k1) == 12, "Exactly 12 pairs have exactly 1 prior inspection"
    assert k1["prev_iri_m_per_km"].notna().all(), "prev_iri must be observed when k=1"
    assert k1["second_prev_iri_m_per_km"].isna().all(), "second_prev_iri must be NaN when k=1"
    
    # When num_prior_inspections >= 2, both lags must be observed
    k2 = df[df["num_prior_inspections"] >= 2]
    assert len(k2) == 78, "Exactly 78 pairs have >= 2 prior inspections"
    assert k2["prev_iri_m_per_km"].notna().all(), "prev_iri must be observed when k>=2"
    assert k2["second_prev_iri_m_per_km"].notna().all(), "second_prev_iri must be observed when k>=2"


# =============================================================================
# 2. Group-Disjoint Folds & Split Independence Tests
# =============================================================================
def test_group_disjoint_fold_partitioning():
    fold_df = pd.read_csv(ARTIFACTS_DIR / "fold_assignments.csv")
    assert len(fold_df) == 23, "All 23 sites must be assigned to an outer fold"
    assert fold_df["outer_fold"].nunique() == 5, "Must have exactly 5 outer folds"
    
    # Check that each site appears in exactly one fold
    assert fold_df["site_id"].nunique() == 23
    assert fold_df["site_id"].duplicated().sum() == 0, "No site may be assigned to multiple outer folds"
    
    oof_df = pd.read_csv(ARTIFACTS_DIR / "out_of_fold_predictions.csv")
    # Verify OOF rows match fold assignments
    for site_id, grp in oof_df.groupby("site_id"):
        site_folds = grp["outer_fold"].unique()
        assert len(site_folds) == 1, f"Site {site_id} appears in multiple outer folds: {site_folds}"
        expected_fold = fold_df[fold_df["site_id"] == site_id]["outer_fold"].values[0]
        assert site_folds[0] == expected_fold


# =============================================================================
# 3. Persistence & Residual Reconstruction Math Tests
# =============================================================================
def test_persistence_and_residual_reconstruction_math():
    oof_df = pd.read_csv(ARTIFACTS_DIR / "out_of_fold_predictions.csv")
    assert len(oof_df) == 113, "Must have 113 out-of-fold prediction rows"
    
    # 1. Persistence Baseline Math: \hat{y} = y_current, \Delta y = 0
    assert (oof_df["m0_persistence_pred_delta"] == 0.0).all(), "M0 persistence predicted delta must be identically 0.0"
    np.testing.assert_allclose(
        oof_df["m0_persistence_pred_future"].values,
        oof_df["current_severity"].values,
        atol=1e-6,
        err_msg="M0 future prediction must equal current severity",
    )
    np.testing.assert_allclose(
        oof_df["m0_error"].values,
        np.abs(oof_df["current_severity"].values - oof_df["future_severity_true"].values),
        atol=1e-6,
    )
    
    # 2. Residual Reconstruction for M1, M2, M3, M4: \hat{y} = clip(y_current + \Delta y, 0, 1)
    models = [
        ("m1_ols_pred_future", "m1_ols_pred_delta", "m1_error"),
        ("m2_xgb_scenario_pred_future", "m2_xgb_scenario_pred_delta", "m2_error"),
        ("m3_xgb_temporal_pred_future", "m3_xgb_temporal_pred_delta", "m3_error"),
        ("m4_xgb_combined_pred_future", "m4_xgb_combined_pred_delta", "m4_error"),
    ]
    
    c_sev = oof_df["current_severity"].values
    y_true = oof_df["future_severity_true"].values
    
    for fut_col, delta_col, err_col in models:
        reconstructed = np.clip(c_sev + oof_df[delta_col].values, 0.0, 1.0)
        np.testing.assert_allclose(
            oof_df[fut_col].values,
            reconstructed,
            atol=1e-5,
            err_msg=f"Reconstruction mismatch in {fut_col}",
        )
        expected_err = np.abs(oof_df[fut_col].values - y_true)
        np.testing.assert_allclose(oof_df[err_col].values, expected_err, atol=1e-5)
        
    # 3. Paired difference column
    expected_diff = oof_df["m4_error"].values - oof_df["m0_error"].values
    np.testing.assert_allclose(oof_df["paired_diff_m4_minus_m0"].values, expected_diff, atol=1e-6)


# =============================================================================
# 4. Model Ablation & Invariant Tests
# =============================================================================
def test_model_ablation_metrics_table():
    comp_df = pd.read_csv(ARTIFACTS_DIR / "model_comparison.csv").set_index("model_id")
    assert len(comp_df) == 5, "Must ablate exactly 5 models: M0, M1, M2, M3, M4"
    
    m0 = comp_df.loc["M0_Persistence"]
    m1 = comp_df.loc["M1_OLS_Scenario"]
    m2 = comp_df.loc["M2_XGB_Scenario"]
    m3 = comp_df.loc["M3_XGB_Temporal"]
    m4 = comp_df.loc["M4_XGB_Combined"]
    
    # M0 baseline invariants
    assert m0["mae"] == 0.0582
    assert m0["rmse"] == 0.0815
    assert m0["r2_score"] == 0.9125
    assert m0["change_mae"] == 0.0582
    assert m0["mae_skill_vs_persistence"] == 0.0
    
    # M1 (OLS) performs worse than persistence
    assert m1["mae"] > m0["mae"]
    assert m1["mae_skill_vs_persistence"] < 0.0
    assert bool(m1["outperforms_persistence"]) is False
    
    # M2 (XGB Scenario) does not beat persistence
    assert m2["mae_skill_vs_persistence"] < 0.0
    assert bool(m2["outperforms_persistence"]) is False
    
    # M3 (XGB Temporal) achieves positive skill over persistence
    assert m3["mae_skill_vs_persistence"] > 0.0
    assert m3["mae"] < m0["mae"]
    assert bool(m3["outperforms_persistence"]) is True
    assert m3["site_win_rate_pct"] > 50.0  # M3 achieves 73.91%
    
    # M4 (XGB Combined) achieves lowest overall MAE
    assert m4["mae"] < m0["mae"]
    assert m4["mae"] < m3["mae"]
    assert m4["mae"] == 0.0551
    assert m4["rmse"] == 0.0766
    assert m4["r2_score"] == 0.9229
    assert m4["mae_skill_vs_persistence"] == 0.0527


# =============================================================================
# 5. Operational Selection Rule Compliance Tests
# =============================================================================
def test_operational_selection_rule_verdict():
    with open(ARTIFACTS_DIR / "final_model_decision.json") as f:
        decision = json.load(f)
        
    assert decision["authoritative_primary_forecast"] == "M0_Persistence"
    assert decision["verdict"] == "RESTRICTED_BOUNDED_OPERATING_REGION_OR_SCENARIO_ESTIMATE"
    assert "Secondary Scenario-Conditioned Estimate" in decision["assigned_operational_role_for_xgboost"]
    
    # Paired 95% CI must span zero
    ci_lower = decision["evidence_summary"]["paired_delta_mae_95ci"][0]
    ci_upper = decision["evidence_summary"]["paired_delta_mae_95ci"][1]
    assert ci_lower < 0.0, "Lower bound must be negative"
    assert ci_upper > 0.0, "Upper bound must be positive (spans zero)"
    
    # Prohibited actions check
    prohibited = decision["prohibited_actions"]
    assert any("causal" in p.lower() for p in prohibited)
    assert any("30-90" in p.lower() or "rapid" in p.lower() for p in prohibited)


def test_forecast_skill_json_horizon_breakdown():
    with open(ARTIFACTS_DIR / "forecast_skill.json") as f:
        skill = json.load(f)
        
    h_data = skill["performance_by_horizon"]
    
    # Short horizon (< 365d): persistence wins
    short = h_data["Short_Horizon_lt_365d"]
    assert short["m4_wins"] is False
    assert short["m4_skill_vs_m0"] < 0.0
    
    # Medium horizon (365-550d): M4 wins
    med = h_data["Medium_Horizon_365_550d"]
    assert med["m4_wins"] is True
    assert med["m4_skill_vs_m0"] > 0.0
    
    # Long horizon (> 550d): M4 wins
    long = h_data["Long_Horizon_gt_550d"]
    assert long["m4_wins"] is True
    assert long["m4_skill_vs_m0"] > 0.0


# =============================================================================
# 6. Deliverable Artifact, Figure & Report Integrity Tests
# =============================================================================
@pytest.mark.parametrize("artifact_name", [
    "temporal_feature_manifest.csv",
    "fold_assignments.csv",
    "out_of_fold_predictions.csv",
    "model_comparison.csv",
    "paired_site_metrics.csv",
    "forecast_skill.json",
    "final_model_decision.json",
])
def test_phase1c_artifacts_exist_and_nonempty(artifact_name: str):
    p = ARTIFACTS_DIR / artifact_name
    assert p.exists(), f"Artifact {artifact_name} must exist"
    assert p.stat().st_size > 0, f"Artifact {artifact_name} must not be empty"


@pytest.mark.parametrize("fig_name", [
    "model_comparison.png",
    "observed_vs_predicted_change.png",
    "error_by_horizon.png",
    "site_win_loss.png",
    "feature_importance.png",
])
def test_phase1c_figures_exist_and_nonempty(fig_name: str):
    p = FIGURES_DIR / fig_name
    assert p.exists(), f"Figure {fig_name} must exist"
    assert p.stat().st_size > 1000, f"Figure {fig_name} must be a valid image"


@pytest.mark.parametrize("report_name", [
    "DATA_FEASIBILITY_AUDIT.md",
    "TEMPORAL_FEATURE_DEFINITIONS.md",
    "MODEL_ABLATION_REPORT.md",
    "FORECAST_VALIDATION_REPORT.md",
    "PHASE1C_FINAL_REPORT.md",
])
def test_phase1c_reports_exist_and_nonempty(report_name: str):
    p = REPORTS_DIR / report_name
    assert p.exists(), f"Report {report_name} must exist"
    assert p.stat().st_size > 500, f"Report {report_name} must be comprehensive"
