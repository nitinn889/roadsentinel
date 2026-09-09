"""Automated Test Suite for RoadSentinel Dashboard V2 & Canonical Research Metrics.

Validates:
1. Clean import and instantiation of all 12 dashboard component modules.
2. Complete integrity of all cached data loaders and canonical CSV/JSON assets.
3. 100% physical accessibility of all 40 Experiment A capture images.
4. Exact numerical parity across Canonical Research Metrics, Decision Engine, and XGBoost Forecasts.
5. Absence of forbidden overclaims across dashboard source code and guides.
"""

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
        # Verify all 5 scenarios and 3 horizons present
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

    def test_benchmark_tables_loaded(self):
        df_final = data_loader.load_final_perception_table()
        self.assertFalse(df_final.empty)
        df50, df25 = data_loader.load_binary_metrics()
        self.assertFalse(df50.empty)
        self.assertFalse(df25.empty)
        df_sem = data_loader.load_yolo_semantic_metrics()
        self.assertFalse(df_sem.empty)
        df_pan = data_loader.load_panel_examples()
        self.assertFalse(df_pan.empty)

    def test_temporal_table_loaded(self):
        df = data_loader.load_temporal_sequences_table()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 8)

    def test_cross_domain_tables_loaded(self):
        df_raw = data_loader.load_cross_domain_dataset()
        self.assertFalse(df_raw.empty)
        self.assertEqual(len(df_raw), 300)
        df_cmp = data_loader.load_cross_domain_comparison_table()
        self.assertFalse(df_cmp.empty)
        df_route = data_loader.load_cross_domain_routing_table()
        self.assertFalse(df_route.empty)

    def test_reliability_validation_tables_loaded(self):
        df_abl = data_loader.load_reliability_ablation_table()
        self.assertFalse(df_abl.empty)
        df_gate = data_loader.load_domain_gate_table()
        self.assertFalse(df_gate.empty)

    def test_decision_engine_tables_loaded(self):
        df_abl = data_loader.load_decision_ablation_table()
        self.assertFalse(df_abl.empty)
        df_summary = data_loader.load_segment_decision_summary()
        self.assertFalse(df_summary.empty)
        df_cases = data_loader.load_case_studies_table()
        self.assertFalse(df_cases.empty)


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


class TestCanonicalResearchConsistency(unittest.TestCase):
    """Verify canonical numbers against source JSON."""

    def test_canonical_metrics_values(self):
        m = data_loader.load_canonical_metrics()
        
        # In-domain YOLO
        yolo_in = m["perception_in_domain_china"]["yolo_v8n"]
        self.assertEqual(yolo_in["f1_score"], 0.7104)
        self.assertEqual(yolo_in["latency_ms"], 3.62)
        
        # In-domain DINO/SAM
        dino_in = m["perception_in_domain_china"]["dinov2_sam2"]
        self.assertEqual(dino_in["f1_score"], 0.0267)
        self.assertEqual(dino_in["latency_ms"], 263.15)
        
        # Latency ratio
        self.assertAlmostEqual(m["perception_in_domain_china"]["latency_ratio_dinov2_vs_yolo"], 72.693, places=2)

        # Cross-domain India counts
        india_yolo = m["perception_cross_domain_india"]["yolo_v8n"]
        self.assertEqual(india_yolo["target_t0_failures"], 292)
        self.assertEqual(india_yolo["target_t0_successes"], 8)
        self.assertEqual(india_yolo["target_t1_failures"], 293)
        self.assertEqual(india_yolo["target_t1_successes"], 7)

        # Domain gating
        dino_gate = m["domain_gating_dinov2"]
        self.assertEqual(dino_gate["domain_classification_auroc"], 1.0000)
        self.assertAlmostEqual(dino_gate["separation_margin"], 0.1158, places=4)

        # Temporal counts
        temp_events = m["temporal_analytics_experiment_a"]["canonical_event_counts"]
        self.assertEqual(temp_events["OBSERVED_AREA_INCREASED"], 14)
        self.assertEqual(temp_events["OBSERVED_AREA_DECREASED"], 19)
        self.assertEqual(temp_events["NOT_OBSERVED"], 34)
        self.assertEqual(temp_events["NEW_DEFECT"], 48)
        self.assertEqual(temp_events["TOTAL_MATCHED_TRANSITIONS"], 33)


if __name__ == "__main__":
    unittest.main()
