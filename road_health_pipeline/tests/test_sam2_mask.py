"""Tests for inference/sam2_mask.py.

Most tests use the mock SAM2 masker from conftest.py (no checkpoint needed).
Tests requiring a real SAM2 checkpoint are marked with:
    @pytest.mark.requires_checkpoint
and auto-skipped when the checkpoint file is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from common.schemas import SegmentationResult
from config import CONFIG

# Skip marker for checkpoint-dependent tests
_CHECKPOINT_AVAILABLE = CONFIG.sam2_checkpoint.exists()
requires_checkpoint = pytest.mark.skipif(
    not _CHECKPOINT_AVAILABLE,
    reason=f"SAM2 checkpoint not found: {CONFIG.sam2_checkpoint}",
)


# ---------------------------------------------------------------------------
# SegmentationResult schema tests
# ---------------------------------------------------------------------------

class TestSegmentationResult:
    def test_fields(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True
        result = SegmentationResult(
            mask=mask,
            confidence=0.85,
            bbox_xyxy=[10, 10, 30, 30],
            area_px=400,
        )
        assert result.mask.dtype == bool
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.bbox_xyxy) == 4
        assert result.area_px == 400

    def test_to_dict(self):
        mask = np.zeros((32, 32), dtype=bool)
        result = SegmentationResult(mask=mask, confidence=0.5, bbox_xyxy=[0, 0, 10, 10], area_px=0)
        d = result.to_dict()
        assert "confidence" in d
        assert "bbox_xyxy" in d
        assert "area_px" in d
        assert "mask" not in d  # mask is not serialised by to_dict


# ---------------------------------------------------------------------------
# Mock masker tests (no checkpoint)
# ---------------------------------------------------------------------------

class TestMockMasker:
    def test_get_road_mask_shape(self, mock_sam2_masker, synthetic_healthy_rgb):
        mask = mock_sam2_masker.get_road_mask(synthetic_healthy_rgb)
        h, w = synthetic_healthy_rgb.shape[:2]
        assert mask.shape == (h, w)
        assert mask.dtype == bool

    def test_get_road_mask_all_true(self, mock_sam2_masker, synthetic_healthy_rgb):
        mask = mock_sam2_masker.get_road_mask(synthetic_healthy_rgb)
        assert mask.all(), "Mock masker should return all-True mask"

    def test_refine_box_returns_segmentation_result(
        self, mock_sam2_masker, synthetic_healthy_rgb
    ):
        h, w = synthetic_healthy_rgb.shape[:2]
        result = mock_sam2_masker.refine_box(
            synthetic_healthy_rgb, [100, 100, 200, 200]
        )
        assert isinstance(result, SegmentationResult)
        assert result.mask.shape == (h, w)
        assert result.mask.dtype == bool
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.bbox_xyxy) == 4
        assert result.area_px == int(result.mask.sum())

    def test_refine_box_mask_covers_box(
        self, mock_sam2_masker, synthetic_healthy_rgb
    ):
        """The mock mask should cover the prompt box region."""
        box = [100, 100, 200, 200]
        result = mock_sam2_masker.refine_box(synthetic_healthy_rgb, box)
        # Check that the centre of the box is inside the mask
        cy, cx = 150, 150
        assert result.mask[cy, cx], "Mask should cover the centre of the prompt box"

    def test_area_px_matches_mask(self, mock_sam2_masker, synthetic_healthy_rgb):
        result = mock_sam2_masker.refine_box(
            synthetic_healthy_rgb, [50, 50, 150, 150]
        )
        assert result.area_px == int(result.mask.sum())


# ---------------------------------------------------------------------------
# Real SAM2 tests (checkpoint required)
# ---------------------------------------------------------------------------

@requires_checkpoint
@pytest.mark.slow
class TestRealSAM2:
    @pytest.fixture(scope="class")
    @classmethod
    def masker(cls):
        from inference.sam2_mask import RoadMasker
        return RoadMasker(device="cpu")

    def test_get_road_mask_shape(self, masker, synthetic_healthy_rgb):
        h, w = synthetic_healthy_rgb.shape[:2]
        mask = masker.get_road_mask(synthetic_healthy_rgb)
        assert mask.shape == (h, w)
        assert mask.dtype == bool

    def test_get_road_mask_nonzero(self, masker, synthetic_healthy_rgb):
        mask = masker.get_road_mask(synthetic_healthy_rgb)
        assert mask.sum() > 0, "Road mask should not be empty"

    def test_refine_box_returns_segmentation_result(
        self, masker, synthetic_pothole_rgb
    ):
        h, w = synthetic_pothole_rgb.shape[:2]
        # Use the centre quarter of the image as prompt
        result = masker.refine_box(
            synthetic_pothole_rgb, [w // 4, h // 4, 3 * w // 4, 3 * h // 4]
        )
        assert isinstance(result, SegmentationResult)
        assert result.mask.shape == (h, w)
        assert result.mask.dtype == bool
        assert result.area_px >= 0

    def test_refine_box_confidence_in_range(self, masker, synthetic_pothole_rgb):
        result = masker.refine_box(
            synthetic_pothole_rgb, [100, 100, 300, 300]
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_refine_box_area_matches_mask(self, masker, synthetic_pothole_rgb):
        result = masker.refine_box(
            synthetic_pothole_rgb, [100, 100, 300, 300]
        )
        assert result.area_px == int(result.mask.sum())

    def test_tiny_box_does_not_crash(self, masker, synthetic_pothole_rgb):
        """Very small prompt box should not raise; may return fallback mask."""
        result = masker.refine_box(synthetic_pothole_rgb, [200, 200, 205, 205])
        assert isinstance(result, SegmentationResult)


# ---------------------------------------------------------------------------
# Filter masks helper (unit test — no checkpoint needed)
# ---------------------------------------------------------------------------

class TestFilterMasks:
    def _make_masks(self, h: int, w: int, fractions: list[float]) -> np.ndarray:
        masks = []
        for f in fractions:
            m = np.zeros((h, w), dtype=bool)
            n_true = int(f * h * w)
            m.flat[:n_true] = True
            masks.append(m)
        return np.array(masks)

    def test_removes_tiny_masks(self):
        from inference.sam2_mask import RoadMasker
        h, w = 512, 512
        # 0.0001 fraction = ~26 pixels << candidate_min_area_px (100)
        masks = self._make_masks(h, w, [0.0001, 0.1, 0.2])
        scores = np.array([0.9, 0.7, 0.8], dtype=np.float32)
        filtered_m, filtered_s = RoadMasker._filter_masks(masks, scores, h, w)
        # Only the two larger masks should survive
        assert len(filtered_m) == 2

    def test_removes_huge_masks(self):
        from inference.sam2_mask import RoadMasker
        h, w = 512, 512
        # 0.5 fraction > candidate_max_area_fraction (0.35) → filtered
        masks = self._make_masks(h, w, [0.1, 0.5])
        scores = np.array([0.8, 0.9], dtype=np.float32)
        filtered_m, filtered_s = RoadMasker._filter_masks(masks, scores, h, w)
        assert len(filtered_m) == 1

    def test_all_filtered_returns_original(self):
        """When all masks fail the filter, return all masks to avoid crashing."""
        from inference.sam2_mask import RoadMasker
        h, w = 64, 64
        # All tiny (0 pixels)
        masks = np.zeros((2, h, w), dtype=bool)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        filtered_m, filtered_s = RoadMasker._filter_masks(masks, scores, h, w)
        assert len(filtered_m) == 2  # all returned unchanged
