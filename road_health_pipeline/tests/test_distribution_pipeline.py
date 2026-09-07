"""Unit and integration tests for calibrated distribution and spatial filtering pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from inference.road_distribution import RoadDistributionModel, DomainCluster
from inference.spatial_filtering import SpatialContextFilter
from inference.hierarchical_classifier import (
    HierarchicalClassifier,
    HierarchicalDefectType,
    PipelineStage,
)
from calibration.calibrator import ProbabilityCalibrator
from datasets.benchmark_builder import BenchmarkBuilder


# ---------------------------------------------------------------------------
# 1. RoadDistributionModel Tests
# ---------------------------------------------------------------------------

class TestRoadDistributionModel:
    def test_multi_domain_clustering(self):
        model = RoadDistributionModel(embed_dim=16)
        rng = np.random.default_rng(42)

        # Real asphalt features around +1.0
        real_emb = rng.normal(loc=1.0, scale=0.1, size=(50, 16)).astype(np.float32)
        model.add_domain_cluster("real_asphalt", real_emb)

        # Sim CARLA features around -1.0
        sim_emb = rng.normal(loc=-1.0, scale=0.1, size=(50, 16)).astype(np.float32)
        model.add_domain_cluster("sim_asphalt", sim_emb)

        assert len(model.clusters) == 2

        # A query from real healthy should match real cluster
        q_real = rng.normal(loc=1.0, scale=0.1, size=(5, 16)).astype(np.float32)
        scores_real = model.score_patches(q_real, metric="cosine_min")
        assert (scores_real < 0.15).all()

        # A query from sim healthy should match sim cluster (no false anomaly!)
        q_sim = rng.normal(loc=-1.0, scale=0.1, size=(5, 16)).astype(np.float32)
        scores_sim = model.score_patches(q_sim, metric="cosine_min")
        assert (scores_sim < 0.15).all()

        # A true defect (divergent features with alternating pattern orthogonal to healthy clusters)
        pattern = np.array([1.0, -1.0] * 8, dtype=np.float32)
        q_defect = pattern[None, :] + rng.normal(scale=0.05, size=(5, 16)).astype(np.float32)
        scores_defect = model.score_patches(q_defect, metric="cosine_min")
        assert (scores_defect > 0.50).all()
        assert (scores_defect > scores_real.max()).all()

    def test_local_road_normalization_cancels_global_shift(self):
        # Scenario: a simulated frame has a global domain shift offset of +0.30 across all patches
        rng = np.random.default_rng(0)
        healthy_road = rng.normal(loc=0.35, scale=0.03, size=200).astype(np.float32)
        # Defect patch has localized elevation of +0.45 above the baseline
        defect = np.array([0.80, 0.85], dtype=np.float32)
        all_scores = np.concatenate([healthy_road, defect])

        norm_scores, stats = RoadDistributionModel.apply_local_road_normalization(
            all_scores, percentile_baseline=50.0
        )

        # Baseline of healthy road should be normalized close to 0
        healthy_norm = norm_scores[:200]
        assert np.percentile(healthy_norm, 50) == pytest.approx(0.0, abs=1e-3)
        assert np.percentile(healthy_norm, 90) < 0.35

        # True defect should have high normalized anomaly score
        defect_norm = norm_scores[200:]
        assert (defect_norm > 0.80).all()


# ---------------------------------------------------------------------------
# 2. SpatialContextFilter Tests
# ---------------------------------------------------------------------------

class TestSpatialContextFilter:
    @pytest.fixture
    def filter_suite(self):
        return SpatialContextFilter()

    def test_filter_isolated_spikes(self, filter_suite):
        # 100x100 mask with single isolated pixel vs 50-pixel defect blob
        mask = np.zeros((100, 100), dtype=bool)
        mask[10, 10] = True  # single spike
        mask[40:48, 40:48] = True  # 64-pixel blob

        cleaned = filter_suite.filter_isolated_spikes(mask, min_connected_pixels=20)
        assert not cleaned[10, 10]  # Isolated spike removed
        assert cleaned[44, 44]  # Coherent defect preserved

    def test_neighborhood_contrast(self, filter_suite):
        # Anomaly map: flat road = 0.20, defect center = 0.80
        amap = np.full((60, 60), 0.20, dtype=np.float32)
        candidate_mask = np.zeros((60, 60), dtype=bool)
        candidate_mask[25:35, 25:35] = True
        amap[candidate_mask] = 0.80

        contrast, in_mean, collar_mean = filter_suite.compute_neighborhood_contrast(
            amap, candidate_mask, dilation_radius=10
        )
        assert in_mean == pytest.approx(0.80, abs=1e-3)
        assert collar_mean == pytest.approx(0.20, abs=1e-3)
        assert contrast > 2.0

    def test_shadow_rejection(self, filter_suite):
        # Synthetic RGB image: smooth pavement with uniform dark cast shadow
        rgb = np.full((100, 100, 3), 140, dtype=np.uint8)
        shadow_mask = np.zeros((100, 100), dtype=bool)
        shadow_mask[30:70, 30:70] = True
        # Shadow is 60 units darker, but texture is completely smooth/uniform
        rgb[shadow_mask] = 80

        shadow_prob = filter_suite.evaluate_shadow_likelihood(rgb, shadow_mask)
        assert shadow_prob >= 0.50

    def test_road_marking_rejection(self, filter_suite):
        # Synthetic RGB with bright yellow road stripe
        rgb = np.full((100, 100, 3), 100, dtype=np.uint8)
        stripe_mask = np.zeros((100, 100), dtype=bool)
        stripe_mask[20:80, 48:52] = True  # thin vertical line
        # Bright yellow marking in RGB: (250, 210, 0)
        rgb[stripe_mask] = [250, 210, 0]

        marking_prob = filter_suite.evaluate_road_marking_likelihood(
            rgb, stripe_mask, aspect_ratio=15.0
        )
        assert marking_prob >= 0.70


# ---------------------------------------------------------------------------
# 3. HierarchicalClassifier Tests
# ---------------------------------------------------------------------------

class TestHierarchicalClassifier:
    @pytest.fixture
    def classifier(self):
        return HierarchicalClassifier(pothole_confirm_threshold=0.50)

    def test_stage_0_benign_rejection(self, classifier):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True
        rgb = np.full((50, 50, 3), 120, dtype=np.uint8)

        res = classifier.classify(
            rgb=rgb,
            mask=mask,
            raw_anomaly_score=0.40,
            spatial_consistency=0.2,
            filter_passed=False,
            rejection_reason="insufficient_neighborhood_contrast",
            calibrated_confidence=0.3,
        )
        assert res.stage == PipelineStage.STAGE_0_ANOMALY
        assert res.final_type == HierarchicalDefectType.BENIGN_ANOMALY
        assert not res.is_confirmed_pothole

    def test_stage_3_confirmed_pothole(self, classifier):
        mask = np.zeros((60, 60), dtype=bool)
        # Circular cavity
        cv2.circle(mask.view(np.uint8), (30, 30), 12, 1, -1)
        mask = mask.astype(bool)

        rgb = np.full((60, 60, 3), 140, dtype=np.uint8)
        rgb[mask] = 70  # Darker cavity depression

        res = classifier.classify(
            rgb=rgb,
            mask=mask,
            raw_anomaly_score=0.85,
            spatial_consistency=0.90,
            filter_passed=True,
            calibrated_confidence=0.88,
        )
        assert res.stage == PipelineStage.STAGE_3_CONFIRMED
        assert res.final_type == HierarchicalDefectType.CONFIRMED_POTHOLE
        assert res.is_confirmed_pothole

    def test_crack_differentiation(self, classifier):
        mask = np.zeros((80, 80), dtype=bool)
        # Thin elongated crack: 70 pixels long, 2 pixels wide
        mask[5:75, 39:41] = True

        rgb = np.full((80, 80, 3), 120, dtype=np.uint8)
        rgb[mask] = 50

        res = classifier.classify(
            rgb=rgb,
            mask=mask,
            raw_anomaly_score=0.75,
            spatial_consistency=0.80,
            filter_passed=True,
            calibrated_confidence=0.70,
        )
        assert res.final_type == HierarchicalDefectType.CRACK_OR_FATIGUE
        assert not res.is_confirmed_pothole


# ---------------------------------------------------------------------------
# 4. ProbabilityCalibrator Tests
# ---------------------------------------------------------------------------

class TestProbabilityCalibrator:
    def test_calibration_fit_and_metrics(self):
        calibrator = ProbabilityCalibrator()
        rng = np.random.default_rng(42)

        # Synthetic scores: healthy ~ 0.20, defect ~ 0.80
        s_neg = rng.beta(2, 8, size=100)
        s_pos = rng.beta(8, 2, size=100)
        scores = np.concatenate([s_neg, s_pos])
        labels = np.concatenate([np.zeros(100), np.ones(100)])

        metrics = calibrator.fit(scores, labels)
        assert calibrator.is_fitted
        assert 0.0 <= metrics.brier_score <= 0.25
        assert 0.0 <= metrics.expected_calibration_error <= 0.35
        assert metrics.max_f1 >= 0.85
        assert 0.20 <= metrics.optimal_threshold <= 0.80

        # Prediction test
        prob_neg = calibrator.predict_probability(0.1)
        prob_pos = calibrator.predict_probability(0.9)
        assert prob_neg < prob_pos
        assert prob_neg < 0.25
        assert prob_pos > 0.75
