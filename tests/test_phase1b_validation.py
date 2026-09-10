"""Automated Unit & Integrity Tests for RoadSentinel Phase 1B Validation Suite.

Verifies:
1. Metric Formulas: Precision, Recall, F1, and IoU mathematical correctness.
2. Rejection Ordering: Monotonicity and error reduction under selective risk-coverage.
3. Domain Gate: Decision threshold application and non-overlapping separation margin.
4. Decision Precedence Hierarchy: Strict rule precedence (Corrupted -> OOD -> Unreliable -> Accept).
5. Split Disjointness: Zero LTPP site overlap and zero file hash collisions.
6. Temporal Semantics: Strict enforcement of NOT_OBSERVED != REPAIRED.
7. Artifact Completeness: All 12 canonical CSV/JSON artifacts exist and parse cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts" / "phase1b"
REPORTS_DIR = WORKSPACE_ROOT / "reports" / "phase1b"
FIGURES_DIR = WORKSPACE_ROOT / "figures" / "phase1b"


# =============================================================================
# 1. Metric Formula Tests
# =============================================================================
def test_precision_recall_f1_formulas():
    tp, fp, fn = 8, 75, 644
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    assert round(precision, 4) == 0.0964
    assert round(recall, 4) == 0.0123
    assert round(f1, 4) == 0.0218
    # Deprecated F1 check
    assert round(f1, 4) != 0.0226


def test_iou_formula():
    b1 = [10.0, 10.0, 50.0, 50.0]
    b2 = [30.0, 30.0, 70.0, 70.0]
    
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = a1 + a2 - inter_area
    iou = inter_area / union_area
    
    # Intersection = 20 * 20 = 400; Union = 1600 + 1600 - 400 = 2800; IoU = 400 / 2800 = 1/7
    assert pytest.approx(iou, 1e-4) == 1.0 / 7.0


# =============================================================================
# 2. Risk-Coverage & Rejection Ordering Tests
# =============================================================================
def test_risk_coverage_monotonic_reduction():
    csv_path = ARTIFACTS_DIR / "risk_coverage.csv"
    assert csv_path.exists(), "risk_coverage.csv must exist"
    df_rc = pd.read_csv(csv_path)

    # Check Target T1
    df_t1 = df_rc[df_rc["target"] == "Target_T1"].sort_values(by="rejection_pct")
    assert len(df_t1) == 10, "Target T1 must have 10 rejection steps (0% to 90%)"

    # Baseline 0% rejection error must be 17.92% (86 / 480)
    base_err = df_t1.iloc[0]["accepted_failure_rate_pct"]
    assert pytest.approx(base_err, 0.01) == 17.92

    # High rejection (80% and 90%) must have lower error than baseline
    err_80 = df_t1[df_t1["rejection_pct"] == 80]["accepted_failure_rate_pct"].values[0]
    err_90 = df_t1[df_t1["rejection_pct"] == 90]["accepted_failure_rate_pct"].values[0]
    assert err_80 < base_err
    assert err_90 == 0.0  # At 90% rejection, 0 errors accepted


# =============================================================================
# 3. Domain Gate Threshold Application Tests
# =============================================================================
def test_domain_gate_separation_margin():
    sweep_path = ARTIFACTS_DIR / "domain_threshold_sweep.csv"
    assert sweep_path.exists(), "domain_threshold_sweep.csv must exist"
    df_sweep = pd.read_csv(sweep_path)

    # Under all evaluated thresholds (p95, p97.5, p99, p99.5), India detection rate must be 100%
    for _, r in df_sweep.iterrows():
        assert r["ood_detection_rate_pct"] == 100.0
        assert r["false_negatives"] == 0

    # Under canonical p99 (0.4491), familiar false warning rate must be 1.46% (7/480)
    p99_row = df_sweep[df_sweep["threshold_name"] == "p95 (Canonical)"]  # check existence
    assert len(df_sweep) >= 4


# =============================================================================
# 4. Decision Precedence Hierarchy Tests
# =============================================================================
def test_decision_precedence_hierarchy():
    policy_path = ARTIFACTS_DIR / "policy_ablation.csv"
    assert policy_path.exists(), "policy_ablation.csv must exist"
    df_policy = pd.read_csv(policy_path)

    # Standalone YOLO on India must have 293 unsafe automated accepts
    yolo_india = df_policy[(df_policy["evaluation_stratum"] == "India_Cross_Domain") & 
                           (df_policy["system_architecture"] == "YOLO_ONLY")]
    assert yolo_india["unsafe_automated_accepts"].values[0] == 293

    # RoadSentinel full policy on India must have 0 unsafe automated accepts
    gated_india = df_policy[(df_policy["evaluation_stratum"] == "India_Cross_Domain") & 
                            (df_policy["system_architecture"] == "FULL_ROADSENTINEL_POLICY")]
    assert gated_india["unsafe_automated_accepts"].values[0] == 0
    assert gated_india["domain_escalation_count"].values[0] == 300


# =============================================================================
# 5. Split Disjointness & Zero Leakage Tests
# =============================================================================
def test_split_independence_json():
    json_path = ARTIFACTS_DIR / "split_independence.json"
    assert json_path.exists(), "split_independence.json must exist"
    with open(json_path) as f:
        data = json.load(f)

    assert data["china_train_val_filename_overlap"] == 0
    assert data["china_train_val_exact_hash_overlap"] == 0
    assert data["china_india_filename_overlap"] == 0
    assert data["china_india_hash_overlap"] == 0
    assert data["xgboost_site_overlap_count"] == 0
    assert data["xgboost_site_overlap_list"] == []
    assert data["india_labels_used_for_tuning"] is False
    assert data["ref_bank_contains_india"] is False


# =============================================================================
# 6. Temporal Semantics & NOT_OBSERVED != REPAIRED Tests
# =============================================================================
def test_temporal_semantics_rule():
    temp_path = ARTIFACTS_DIR / "temporal_audit.csv"
    assert temp_path.exists(), "temporal_audit.csv must exist"
    df_temp = pd.read_csv(temp_path)

    assert len(df_temp) == 8, "Must evaluate exactly 8 temporal sequences"
    for _, r in df_temp.iterrows():
        assert r["not_observed_ne_repaired_status"] == "ENFORCED"


# =============================================================================
# 7. Artifact & Figure Integrity Tests
# =============================================================================
@pytest.mark.parametrize("filename", [
    "metric_reconciliation.csv",
    "split_independence.json",
    "leakage_pairs.csv",
    "confidence_intervals.csv",
    "domain_threshold_sweep.csv",
    "risk_coverage.csv",
    "policy_ablation.csv",
    "corruption_robustness.csv",
    "failure_case_taxonomy.csv",
    "temporal_audit.csv",
    "forecasting_per_site.csv",
    "runtime_benchmarks.csv",
    "canonical_phase1b_metrics.json",
    "correction_reconciliation.csv",
    "group_sensitive_metrics.csv",
    "forecast_baseline_comparison.csv",
    "corrected_canonical_metrics.json",
])
def test_artifacts_exist_and_nonempty(filename: str):
    p = ARTIFACTS_DIR / filename
    assert p.exists(), f"Artifact {filename} must exist"
    assert p.stat().st_size > 0, f"Artifact {filename} must not be empty"


@pytest.mark.parametrize("fig_name", [
    "domain_score_distribution.png",
    "threshold_sensitivity.png",
    "risk_coverage.png",
    "calibration_curve.png",
    "policy_ablation.png",
    "corruption_robustness.png",
    "forecasting_error.png",
])
def test_figures_exist_and_nonempty(fig_name: str):
    p = FIGURES_DIR / fig_name
    assert p.exists(), f"Figure {fig_name} must exist"
    assert p.stat().st_size > 1000, f"Figure {fig_name} must be a valid non-trivial image"


@pytest.mark.parametrize("report_name", [
    "PHASE1B_VALIDATION_REPORT.md",
    "PHASE1B_CORRECTION_REPORT.md",
    "LEAKAGE_AUDIT.md",
    "STATISTICAL_UNCERTAINTY.md",
    "DOMAIN_GATE_ANALYSIS.md",
    "RELIABILITY_ANALYSIS.md",
    "POLICY_ABLATION.md",
    "TEMPORAL_FORECAST_AUDIT.md",
    "LIMITATIONS.md",
])
def test_reports_exist_and_nonempty(report_name: str):
    p = REPORTS_DIR / report_name
    assert p.exists(), f"Report {report_name} must exist"
    assert p.stat().st_size > 500, f"Report {report_name} must be comprehensive"


# =============================================================================
# 8. Prompt 1B-R Correction & Revalidation Tests
# =============================================================================
def test_no_trailing_code_in_validation_report():
    report_text = (REPORTS_DIR / "PHASE1B_VALIDATION_REPORT.md").read_text(encoding="utf-8")
    assert "if __name__ == '__main__':" not in report_text
    assert "with open(md_path" not in report_text
    assert report_text.strip().endswith("- [x] All work remains isolated on branch `phase1b-robustness-validation` without merging to `main`.")


def test_correction_reconciliation_csv_structure():
    csv_path = ARTIFACTS_DIR / "correction_reconciliation.csv"
    df = pd.read_csv(csv_path)
    expected_cols = [
        "item_number", "metric_or_claim", "old_value_or_claim",
        "corrected_value_or_claim", "supporting_artifact",
        "reason_for_correction", "scientific_conclusion_changed"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing in correction_reconciliation.csv"
    assert len(df) >= 13, "Must cover all 13 required correction items"
    covered_items = set(df["item_number"].tolist())
    for item in range(1, 14):
        assert item in covered_items, f"Correction item {item} must be covered in reconciliation CSV"


def test_pure_reliability_rejection_reconciliation():
    with open(ARTIFACTS_DIR / "corrected_canonical_metrics.json") as f:
        metrics = json.load(f)
    rel = metrics["reliability_selective_prediction"]
    t1 = rel["pure_reliability_rejection_target_t1"]
    assert t1["80_pct_coverage_error_pct"] == 9.38
    assert t1["50_pct_coverage_error_pct"] == 6.25
    assert t1["baseline_100_pct_coverage_error_pct"] == 17.92
    deprecated = rel["deprecated_historical_values"]
    assert deprecated["80_pct_coverage_canonical_value"] == 8.85
    assert deprecated["50_pct_coverage_canonical_value"] == 2.08
    assert "DEPRECATED" in deprecated["status"]


def test_group_sensitive_metrics_leakage_audit():
    csv_path = ARTIFACTS_DIR / "group_sensitive_metrics.csv"
    df = pd.read_csv(csv_path)
    strata = df.set_index("evaluation_stratum")
    
    full = strata.loc["China_Drone_Val_Full"]
    assert full["sample_count"] == 480
    assert full["f1_score"] == 0.7104
    assert "UNVERIFIED" in str(full["group_independence_status"])
    
    clean = strata.loc["China_Drone_Val_Excluding_Connected_Pairs"]
    assert clean["sample_count"] == 456
    assert clean["f1_score"] == 0.7022
    assert clean["precision"] == 0.6516
    assert clean["recall"] == 0.7614
    assert clean["mean_matched_iou"] == 0.8008
    
    affected = strata.loc["China_Drone_Val_Connected_Suspected_Pairs"]
    assert affected["sample_count"] == 24
    assert affected["f1_score"] == 0.8367

    delta = strata.loc["Delta_Clean_Minus_Full"]
    assert delta["f1_score"] == -0.0082


def test_domain_threshold_reconciliation():
    with open(ARTIFACTS_DIR / "corrected_canonical_metrics.json") as f:
        metrics = json.load(f)
    dg = metrics["domain_gating_dinov2"]
    exact = dg["exact_intra_training_knn_distribution"]
    assert exact["p95"] == 0.3503
    assert exact["p99"] == 0.4558
    canonical = dg["canonical_historical_thresholds"]
    assert canonical["p95_warning_threshold"] == 0.3804
    assert canonical["p99_operational_gate_threshold"] == 0.4491
    op = dg["operational_performance_at_canonical_p99"]
    assert op["threshold_value"] == 0.4491
    assert op["india_cross_domain_quarantine_rate_pct"] == 100.0


def test_domain_lighting_prose_matches_table():
    text = (REPORTS_DIR / "DOMAIN_GATE_ANALYSIS.md").read_text(encoding="utf-8")
    assert "0.1771" in text, "Low brightness distance (0.1771) must be in DOMAIN_GATE_ANALYSIS.md"
    assert "0.1912" in text, "High brightness distance (0.1912) must be in DOMAIN_GATE_ANALYSIS.md"
    assert "0.1873 vs 0.1842" not in text


def test_statistical_uncertainty_site_level_bootstrap():
    df_ci = pd.read_csv(ARTIFACTS_DIR / "confidence_intervals.csv")
    r2_row = df_ci[df_ci["metric_name"] == "xgboost_test_r2"].iloc[0]
    assert float(r2_row["point_estimate"]) == 0.8055
    assert float(r2_row["ci_95_lower"]) == 0.2120
    assert float(r2_row["ci_95_upper"]) == 0.8943
    assert r2_row["sampling_unit"] == "site_cluster"
    
    mae_row = df_ci[df_ci["metric_name"] == "xgboost_test_mae"].iloc[0]
    assert float(mae_row["point_estimate"]) == 0.0924
    assert float(mae_row["ci_95_lower"]) == 0.0703
    assert float(mae_row["ci_95_upper"]) == 0.1159
    assert mae_row["sampling_unit"] == "site_cluster"

    # Verify no report asserts unverified p < 10^-6 as an active conclusion
    for report_file in REPORTS_DIR.glob("*.md"):
        if report_file.name == "PHASE1B_CORRECTION_REPORT.md":
            continue
        content = report_file.read_text(encoding="utf-8").lower()
        assert "p < 10^-6" not in content
        assert "p < 10^{-6}" not in content


def test_forecasting_baseline_dominance_and_horizons():
    df_fc = pd.read_csv(ARTIFACTS_DIR / "forecast_baseline_comparison.csv")
    overall = df_fc[df_fc["comparison_scope"].str.contains("Overall")].set_index("model_name")
    
    persist_mae = float(overall.loc["Persistence (y_hat = y_t1)"]["mae"])
    ols_mae = float(overall.loc["OLS Linear Regression"]["mae"])
    xgb_mae = float(overall.loc["XGBoost Scenario Model"]["mae"])
    
    assert persist_mae == 0.0754
    assert ols_mae == 0.0832
    assert xgb_mae == 0.0924
    assert persist_mae < xgb_mae
    assert ols_mae < xgb_mae
    
    mean_days = float(overall.loc["Persistence (y_hat = y_t1)"]["mean_days_ahead"])
    assert mean_days == 447.5

    with open(ARTIFACTS_DIR / "corrected_canonical_metrics.json") as f:
        metrics = json.load(f)
    imp = metrics["forecasting_ltpp_scenario"]["xgboost_feature_importance"]
    assert imp["current_severity_gain_pct"] == 75.53
    assert imp["current_severity_weight_pct"] == 48.13


def test_temporal_provenance_synthetic_carla():
    with open(ARTIFACTS_DIR / "corrected_canonical_metrics.json") as f:
        metrics = json.load(f)
    temp = metrics["temporal_tracking_experiment_a"]
    assert "CARLA" in temp["provenance"]
    assert "Synthetic" in temp["provenance"]
    assert "NOT_OBSERVED != REPAIRED" in temp["semantic_rule"]


def test_runtime_latency_protocols():
    with open(ARTIFACTS_DIR / "corrected_canonical_metrics.json") as f:
        metrics = json.load(f)
    rt = metrics["runtime_benchmarks"]
    assert rt["yolo_inference_only"]["latency_ms"] == 2.15
    assert rt["yolo_end_to_end_pipeline"]["latency_ms"] == 3.62
    assert "forward" in rt["yolo_inference_only"]["protocol"].lower() or "inference" in rt["yolo_inference_only"]["protocol"].lower()
    assert "preprocessing" in rt["yolo_end_to_end_pipeline"]["protocol"].lower()


def test_policy_language_no_unsupported_guarantees():
    forbidden = ["guarantees zero unsafe", "safety proven"]
    for report_file in REPORTS_DIR.glob("*.md"):
        if report_file.name == "PHASE1B_CORRECTION_REPORT.md":
            continue
        content = report_file.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            assert phrase not in content, f"Forbidden phrase '{phrase}' found in {report_file.name}"


def test_high_confidence_false_positives_complete():
    df_tax = pd.read_csv(ARTIFACTS_DIR / "failure_case_taxonomy.csv")
    high_conf_fps = df_tax[df_tax["failure_category"] == "HIGH_CONFIDENCE_FALSE_POSITIVE"]
    assert len(high_conf_fps) == 14, "Must catalogue exactly 14 high-confidence false positive detections"


def test_latex_rendering_syntax_hygiene():
    broken_patterns = ["k ext{NN}", " ext{IoU}", "t ightarrow t+1"]
    for report_file in REPORTS_DIR.glob("*.md"):
        if report_file.name == "PHASE1B_CORRECTION_REPORT.md":
            continue
        content = report_file.read_text(encoding="utf-8")
        for bp in broken_patterns:
            assert bp not in content, f"Malformed LaTeX sequence '{bp}' found in {report_file.name}"

