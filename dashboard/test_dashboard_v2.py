"""Automated Test Suite for RoadSentinel Dashboard V2 & Audited Metrics.

Validates:
1. Clean import and execution readiness of all 12 dashboard component modules.
2. Complete integrity of all cached data loaders and canonical CSV/JSON assets.
3. Headline result cards matching audited targets without hard-coding in UI.
4. Audited Phase 1B perception, reliability, domain gate, policy ablation, and CARLA temporal metrics.
5. Audited Phase 1C forecasting comparison, paired 95% CI, horizon subgroups, and dataset provenance.
6. Error handling and graceful fallback ("Not yet measured") when artifacts are missing.
7. Presence of mandatory caveat labels and absence of forbidden overclaims.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add workspace and dashboard to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = WORKSPACE_ROOT / "dashboard"
COMPONENTS_DIR = DASHBOARD_ROOT / "components"

for p in [WORKSPACE_ROOT, DASHBOARD_ROOT, COMPONENTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data_loader


class TestDashboardV2DataLoaders(unittest.TestCase):
    """Test all centralized data loader functions and verify artifact integrity."""

    def test_canonical_metrics_loaded(self):
        metrics = data_loader.load_canonical_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("perception_in_domain_china", metrics)
        self.assertIn("perception_cross_domain_india", metrics)
        self.assertIn("temporal_analytics_experiment_a", metrics)
        self.assertIn("forecasting_xgboost_goal1", metrics)
        self.assertIn("decision_engine_phase13", metrics)

    def test_decisions_table_loaded(self):
        df = data_loader.load_decisions_table()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 40)
        counts = df["decision"].value_counts().to_dict()
        self.assertEqual(counts.get("MONITOR", 0), 21)
        self.assertEqual(counts.get("REINSPECT", 0), 10)
        self.assertEqual(counts.get("PRIORITY_REVIEW", 0), 9)
        self.assertEqual(counts.get("AUTOMATED_ACCEPT", 0), 0)
        self.assertEqual(counts.get("DOMAIN_ESCALATION", 0), 0)

    def test_goal1_forecasts_loaded(self):
        df = data_loader.load_goal1_forecasts()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 600)
        scenarios = set(df["scenario"].unique())
        self.assertEqual(scenarios, {"NORMAL", "HEAVY_RAIN", "HEAVY_TRAFFIC", "HIGH_HEAT", "WET_EXPOSURE"})
        horizons = set(df["days_ahead"].unique())
        self.assertEqual(horizons, {30, 60, 90})

    def test_primary_results_loaded(self):
        df = data_loader.load_primary_results()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 40)
        self.assertEqual(set(df["segment_id"].unique()), {"SEG_001", "SEG_002", "SEG_003", "SEG_004"})

    def test_all_40_images_accessible(self):
        segments = ["SEG_001", "SEG_002", "SEG_003", "SEG_004"]
        for seg in segments:
            for day in range(1, 11):
                img_path = data_loader.get_image_path(seg, day)
                self.assertIsNotNone(img_path, f"Image path resolved to None for {seg} Day {day}")
                self.assertTrue(img_path.exists(), f"Physical image file does not exist at {img_path}")


class TestAuditedHeadlineResultCards(unittest.TestCase):
    """Test that all 7 headline cards match audited target values."""

    def setUp(self):
        self.cards_dict = data_loader.load_headline_result_cards()
        self.cards = list(self.cards_dict.values())
        self.card_map = {c["title"]: c for c in self.cards}

    def test_all_seven_cards_present(self):
        self.assertEqual(len(self.cards), 7)
        # Check that each expected card concept is present
        card_titles = list(self.card_map.keys())
        self.assertTrue(any("China UAV F1" in t for t in card_titles))
        self.assertTrue(any("India Dashcam F1" in t for t in card_titles))
        self.assertTrue(any("India" in t and "Escalation" in t for t in card_titles))
        self.assertTrue(any("China Warning Rate" in t for t in card_titles))
        self.assertTrue(any("Reliability Accepted Failure Rate" in t for t in card_titles))
        self.assertTrue(any("M4 Forecast" in t for t in card_titles))
        self.assertTrue(any("Raspberry Pi" in t for t in card_titles))

    def test_yolo_china_f1(self):
        card = self.cards_dict["yolo_china_f1"]
        self.assertEqual(card["value"], "0.7104")
        self.assertEqual(card["status"], "MEASURED")

    def test_yolo_india_f1(self):
        card = self.cards_dict["yolo_india_f1"]
        self.assertEqual(card["value"], "0.0218")
        self.assertEqual(card["status"], "MEASURED")

    def test_india_domain_escalation(self):
        card = self.cards_dict["india_domain_escalation"]
        self.assertIn("300/300", card["value"])
        self.assertIn("100%", card["value"])
        self.assertEqual(card["status"], "MEASURED")

    def test_china_warning_rate(self):
        card = self.cards_dict["china_warning_rate"]
        self.assertIn("7/480", card["value"])
        self.assertIn("1.46%", card["value"])
        self.assertEqual(card["status"], "MEASURED")

    def test_reliability_accepted_failure_rates(self):
        card = self.cards_dict["reliability_accepted_failure_rate"]
        self.assertIn("9.38%", card["value"])
        self.assertIn("6.25%", card["value"])
        self.assertEqual(card["status"], "MEASURED")

    def test_m4_forecast_mae_skill(self):
        card = self.cards_dict["m4_forecast_skill"]
        self.assertIn("+5.27%", card["value"])
        self.assertIn("Point estimate", card["delta"])
        self.assertEqual(card["status"], "DERIVED")

    def test_raspberry_pi_pending(self):
        card = self.cards_dict["raspberry_pi_profiling"]
        self.assertTrue("Pending" in card["value"])
        self.assertEqual(card["status"], "PENDING")


class TestAuditedPhase1BMetrics(unittest.TestCase):
    """Test Phase 1B audited artifacts (perception, gate, reliability, policy ablation, temporal)."""

    def test_risk_coverage_values(self):
        df_rc = data_loader.load_risk_coverage_table()
        self.assertFalse(df_rc.empty)
        # Find 80% and 50% coverage points on strict detection failure Target T1
        df_t1 = df_rc[df_rc["target"] == "Target_T1"]
        row_80 = df_t1[df_t1["retained_coverage_pct"] == 80.0]
        self.assertFalse(row_80.empty)
        self.assertAlmostEqual(float(row_80.iloc[0]["accepted_failure_rate_pct"]), 9.375, places=2)

        row_50 = df_t1[df_t1["retained_coverage_pct"] == 50.0]
        self.assertFalse(row_50.empty)
        self.assertAlmostEqual(float(row_50.iloc[0]["accepted_failure_rate_pct"]), 6.25, places=2)

    def test_policy_ablation_zero_unsafe_accepts_india(self):
        df_pol = data_loader.load_policy_ablation_table()
        self.assertFalse(df_pol.empty)
        ind_full = df_pol[(df_pol["evaluation_stratum"] == "India_Cross_Domain") & (df_pol["system_architecture"] == "FULL_ROADSENTINEL_POLICY")]
        self.assertFalse(ind_full.empty)
        self.assertEqual(int(ind_full.iloc[0]["unsafe_automated_accepts"]), 0)
        self.assertEqual(int(ind_full.iloc[0]["domain_escalation_count"]), 300)
        self.assertEqual(float(ind_full.iloc[0]["failures_prevented_pct"]), 100.0)

    def test_per_sample_domain_distances(self):
        dist_dict = data_loader.load_per_sample_domain_distances()
        china_dists = dist_dict["china_distances"]
        india_dists = dist_dict["india_distances"]
        self.assertEqual(len(china_dists), 480)
        self.assertEqual(len(india_dists), 300)
        # Verify China distances are lower than India distances
        self.assertTrue(max(china_dists) < min(india_dists))

    def test_carla_synthetic_temporal_summary(self):
        summary = data_loader.load_all_temporal_events_summary()
        self.assertEqual(summary["total_captures"], 40)
        self.assertEqual(summary["total_sequences"], 8)
        self.assertEqual(summary["total_tracks"], 48)
        self.assertEqual(summary["persistent_tracks"], 22)
        self.assertAlmostEqual(summary["persistence_rate_pct"], 45.83, places=2)
        self.assertEqual(summary["transitions"], 33)
        self.assertEqual(summary["area_increases"], 14)
        self.assertEqual(summary["area_decreases"], 19)
        self.assertEqual(summary["semantic_rule"], "NOT_OBSERVED != REPAIRED")


class TestAuditedPhase1CForecasting(unittest.TestCase):
    """Test Phase 1C forecasting results, subgroups, paired CI, and dataset metadata."""

    def test_model_comparison_values(self):
        df_models = data_loader.load_phase1c_model_comparison()
        self.assertFalse(df_models.empty)
        m0 = df_models[df_models["model_id"] == "M0_Persistence"].iloc[0]
        self.assertEqual(float(m0["mae"]), 0.0582)
        self.assertEqual(float(m0["rmse"]), 0.0815)
        self.assertEqual(float(m0["r2_score"]), 0.9125)

        m3 = df_models[df_models["model_id"] == "M3_XGB_Temporal"].iloc[0]
        self.assertEqual(float(m3["mae"]), 0.0563)
        self.assertEqual(float(m3["mae_skill_vs_persistence"]), 0.0326)

        m4 = df_models[df_models["model_id"] == "M4_XGB_Combined"].iloc[0]
        self.assertEqual(float(m4["mae"]), 0.0551)
        self.assertEqual(float(m4["rmse"]), 0.0766)
        self.assertEqual(float(m4["r2_score"]), 0.9229)
        self.assertEqual(float(m4["mae_skill_vs_persistence"]), 0.0527)

    def test_paired_confidence_interval_crosses_zero(self):
        dict_skill = data_loader.load_phase1c_forecast_skill()
        paired = dict_skill["paired_difference_m4_minus_m0"]
        self.assertEqual(paired["site_clustered_95ci_lower"], -0.0106)
        self.assertEqual(paired["site_clustered_95ci_upper"], 0.0043)
        self.assertTrue(paired["ci_spans_zero"])
        self.assertFalse(paired["statistically_significantly_better_than_persistence"])

    def test_horizon_subgroups(self):
        dict_skill = data_loader.load_phase1c_forecast_skill()
        horizons = dict_skill["performance_by_horizon"]
        
        short_h = horizons["Short_Horizon_lt_365d"]
        self.assertEqual(short_h["sample_count"], 43)
        self.assertEqual(short_h["m4_skill_vs_m0"], -0.0377)
        self.assertFalse(short_h["m4_wins"])

        med_h = horizons["Medium_Horizon_365_550d"]
        self.assertEqual(med_h["sample_count"], 35)
        self.assertEqual(med_h["m4_skill_vs_m0"], 0.0912)
        self.assertTrue(med_h["m4_wins"])

        long_h = horizons["Long_Horizon_gt_550d"]
        self.assertEqual(long_h["sample_count"], 35)
        self.assertEqual(long_h["m4_skill_vs_m0"], 0.1015)
        self.assertTrue(long_h["m4_wins"])

        prior_n78 = dict_skill["performance_on_full_lag_history_subset_n78"]
        self.assertEqual(prior_n78["sample_count"], 78)
        self.assertEqual(prior_n78["m4_skill"], 0.0918)
        self.assertTrue(prior_n78["m4_wins"])

    def test_forecast_dataset_metadata(self):
        dict_feas = data_loader.load_phase1c_feasibility_summary()
        self.assertEqual(dict_feas["total_pairs"], 113)
        self.assertEqual(dict_feas["total_sites"], 23)
        self.assertEqual(len(dict_feas["state_breakdown"]), 6)
        self.assertEqual(dict_feas["interval_days"]["minimum"], 31)
        self.assertEqual(dict_feas["interval_days"]["maximum"], 728)
        self.assertAlmostEqual(dict_feas["interval_days"]["mean"], 429.5, places=1)
        self.assertEqual(dict_feas["temporal_history_sufficiency"]["pairs_with_at_least_1_prior_inspection"], 90)
        self.assertEqual(dict_feas["temporal_history_sufficiency"]["pairs_with_at_least_2_prior_inspections"], 78)
        self.assertFalse(dict_feas["rapid_forecasting_30_90_days_supported"])

    def test_out_of_fold_predictions_integrity(self):
        df_oof = data_loader.load_phase1c_oof_predictions()
        self.assertFalse(df_oof.empty)
        self.assertEqual(len(df_oof), 113)
        required_cols = [
            "pair_id", "site_id", "future_severity_true", "delta_severity_true",
            "m4_xgb_combined_pred_future", "m4_xgb_combined_pred_delta", "m4_error"
        ]
        for col in required_cols:
            self.assertIn(col, df_oof.columns)


class TestDashboardV2ComponentImports(unittest.TestCase):
    """Verify all 12 component modules import cleanly and define render functions."""

    def test_component_imports(self):
        import overview
        self.assertTrue(callable(overview.render_overview))

        import single_image
        self.assertTrue(callable(single_image.render_single_image_assessment))

        import reliability_view
        self.assertTrue(callable(reliability_view.render_reliability_analysis))

        import forecast
        self.assertTrue(callable(forecast.render_forecast))

        import temporal
        self.assertTrue(callable(temporal.render_temporal_monitoring))

        import benchmark_view
        self.assertTrue(callable(benchmark_view.render_benchmark_view))

        import cross_domain_view
        self.assertTrue(callable(cross_domain_view.render_cross_domain_view))

        import decision_view
        self.assertTrue(callable(decision_view.render_decision_view))

        import failure_modes
        self.assertTrue(callable(failure_modes.render_failure_modes))

        import methodology
        self.assertTrue(callable(methodology.render_methodology))

        import experiment_b
        self.assertTrue(callable(experiment_b.render_experiment_b))

        import edge_deployment
        self.assertTrue(callable(edge_deployment.render_edge_deployment))


class TestGracefulMissingArtifactHandling(unittest.TestCase):
    """Verify that missing or malformed artifacts fail gracefully and report missing expected source."""

    def test_missing_csv_returns_empty_dataframe(self):
        # Pointing to non-existent path should return empty DataFrame, not throw uncaught exception
        res = data_loader.load_csv_artifact(Path("non_existent_artifact_xyz.csv"))
        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue(res.empty)

    def test_missing_json_returns_empty_dict(self):
        res = data_loader.load_json_artifact(Path("non_existent_artifact_xyz.json"))
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 0)

    def test_raspberry_pi_status_is_pending_not_fabricated(self):
        cards = data_loader.load_headline_result_cards()
        pi_card = cards["raspberry_pi_profiling"]
        self.assertEqual(pi_card["status"], "PENDING")
        self.assertIn("Pending", pi_card["value"])


class TestCaveatsAndLabels(unittest.TestCase):
    """Verify that all mandatory caveats, semantic rules, and interpretations are present in components."""

    def test_viewpoint_confounding_caveat(self):
        content = (COMPONENTS_DIR / "cross_domain_view.py").read_text(encoding="utf-8")
        self.assertIn("confounded by UAV-versus-dashcam viewpoint", content)

    def test_domain_gate_scope_caveat(self):
        content = (COMPONENTS_DIR / "cross_domain_view.py").read_text(encoding="utf-8")
        self.assertIn("not universal OOD detection", content)

    def test_policy_zero_unsafe_accepts_india_statement(self):
        content = (COMPONENTS_DIR / "decision_view.py").read_text(encoding="utf-8")
        self.assertIn("Zero unsafe automated accepts were observed on the evaluated India benchmark", content)
        self.assertIn("Escalation is safe routing, not successful defect detection", content)

    def test_temporal_semantic_rule_and_label(self):
        content = (COMPONENTS_DIR / "temporal.py").read_text(encoding="utf-8")
        self.assertIn("CARLA/Unreal Synthetic Temporal Evaluation", content)
        self.assertIn("NOT_OBSERVED != REPAIRED", content)
        self.assertIn("temporal-system functionality", content)

    def test_forecasting_zero_crossing_and_horizon_caveat(self):
        content = (COMPONENTS_DIR / "forecast.py").read_text(encoding="utf-8")
        self.assertIn("The interval crosses zero; universal statistical superiority over persistence is not established", content)
        self.assertIn("Do not claim validated 30–90-day forecasting", content)
        self.assertIn("365-day routing boundary", content)


if __name__ == "__main__":
    unittest.main()
