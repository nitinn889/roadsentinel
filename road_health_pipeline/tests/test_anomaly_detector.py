"""Tests for inference/anomaly_detector.py.

Uses the synthetic memory bank from conftest (no model loading required).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config import CONFIG


@pytest.fixture
def detector(synthetic_memory_bank):
    from inference.anomaly_detector import AnomalyDetector
    return AnomalyDetector(memory_bank_dir=synthetic_memory_bank)


@pytest.fixture
def random_embeddings():
    """Batch of random (non-normalised) patch embeddings."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((20, 384)).astype(np.float32)


# ---------------------------------------------------------------------------
# score_patches
# ---------------------------------------------------------------------------

class TestScorePatches:
    def test_output_shape(self, detector, random_embeddings):
        scores = detector.score_patches(random_embeddings)
        assert scores.shape == (20,), f"Expected (20,), got {scores.shape}"

    def test_output_dtype(self, detector, random_embeddings):
        scores = detector.score_patches(random_embeddings)
        assert scores.dtype == np.float32

    def test_scores_in_range(self, detector, random_embeddings):
        scores = detector.score_patches(random_embeddings)
        assert (scores >= 0.0).all(), f"Min score {scores.min()} < 0"
        assert (scores <= 1.0 + 1e-5).all(), f"Max score {scores.max()} > 1"

    def test_scores_finite(self, detector, random_embeddings):
        scores = detector.score_patches(random_embeddings)
        assert np.isfinite(scores).all(), "Score contains NaN/Inf"

    def test_empty_input(self, detector):
        scores = detector.score_patches(np.empty((0, 384), dtype=np.float32))
        assert scores.shape == (0,), "Empty input should return empty scores"

    def test_single_patch(self, detector):
        emb = np.random.default_rng(0).standard_normal((1, 384)).astype(np.float32)
        scores = detector.score_patches(emb)
        assert scores.shape == (1,)
        assert 0.0 <= float(scores[0]) <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# build_anomaly_map
# ---------------------------------------------------------------------------

class TestBuildAnomalyMap:
    def test_output_shape(self, detector):
        H, W = 240, 320
        grid_size = 37
        rng = np.random.default_rng(0)
        n = 100
        coords = rng.integers(0, grid_size, size=(n, 2)).astype(np.int16)
        scores = rng.random(n).astype(np.float32)
        amap = detector.build_anomaly_map(coords, scores, (H, W), grid_size)
        assert amap.shape == (H, W), f"Expected ({H}, {W}), got {amap.shape}"

    def test_nonnegative(self, detector):
        H, W = 128, 128
        grid_size = 37
        rng = np.random.default_rng(0)
        n = 50
        coords = rng.integers(0, grid_size, size=(n, 2)).astype(np.int16)
        scores = rng.random(n).astype(np.float32)
        amap = detector.build_anomaly_map(coords, scores, (H, W), grid_size)
        assert (amap >= 0).all(), "Anomaly map must be non-negative"

    def test_finite(self, detector):
        H, W = 128, 128
        grid_size = 37
        rng = np.random.default_rng(0)
        n = 50
        coords = rng.integers(0, grid_size, size=(n, 2)).astype(np.int16)
        scores = rng.random(n).astype(np.float32)
        amap = detector.build_anomaly_map(coords, scores, (H, W), grid_size)
        assert np.isfinite(amap).all(), "Anomaly map contains NaN/Inf"

    def test_empty_coords(self, detector):
        """Empty patch coords → all-zero anomaly map."""
        H, W = 64, 64
        amap = detector.build_anomaly_map(
            np.empty((0, 2), dtype=np.int16),
            np.empty(0, dtype=np.float32),
            (H, W),
            37,
        )
        assert amap.shape == (H, W)
        assert (amap == 0).all()


# ---------------------------------------------------------------------------
# score_patch_grid
# ---------------------------------------------------------------------------

class TestScorePatchGrid:
    def test_output_shape(self, detector):
        rng = np.random.default_rng(0)
        grid = rng.standard_normal((37, 37, 384)).astype(np.float32)
        scores = detector.score_patch_grid(grid)
        assert scores.shape == (37, 37)

    def test_output_range(self, detector):
        rng = np.random.default_rng(0)
        grid = rng.standard_normal((37, 37, 384)).astype(np.float32)
        scores = detector.score_patch_grid(grid)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_empty_scores(self, detector):
        score, threshold = detector.summarize(np.array([], dtype=np.float32))
        assert score == 0.0
        assert threshold == 0.0

    def test_returns_same_value(self, detector):
        """Both return values must be equal (both use CONFIG.anomaly_percentile)."""
        scores = np.random.default_rng(0).random(100).astype(np.float32)
        image_score, threshold = detector.summarize(scores)
        assert image_score == threshold, (
            "summarize() should return the same value for image_score and threshold"
        )

    def test_matches_numpy_percentile(self, detector):
        scores = np.random.default_rng(0).random(100).astype(np.float32)
        image_score, _ = detector.summarize(scores)
        expected = float(np.percentile(scores, CONFIG.anomaly_percentile))
        assert abs(image_score - expected) < 1e-5


# ---------------------------------------------------------------------------
# normalize_anomaly_map
# ---------------------------------------------------------------------------

class TestNormalizeAnomalyMap:
    def test_range(self):
        from inference.anomaly_detector import AnomalyDetector
        amap = np.random.default_rng(0).random((64, 64)).astype(np.float32)
        normed = AnomalyDetector.normalize_anomaly_map(amap)
        assert float(normed.min()) >= 0.0
        assert float(normed.max()) <= 1.0 + 1e-5

    def test_uniform_map_returns_zeros(self):
        from inference.anomaly_detector import AnomalyDetector
        amap = np.full((32, 32), 0.5, dtype=np.float32)
        normed = AnomalyDetector.normalize_anomaly_map(amap)
        assert (normed == 0.0).all()


# ---------------------------------------------------------------------------
# Missing memory bank
# ---------------------------------------------------------------------------

def test_missing_memory_bank_raises_file_not_found(tmp_path):
    from inference.anomaly_detector import AnomalyDetector
    with pytest.raises(FileNotFoundError, match="Memory bank is incomplete"):
        AnomalyDetector(memory_bank_dir=tmp_path / "nonexistent")
