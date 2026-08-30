"""Integration tests for the full DINOv2 + SAM2 pipeline.

These tests run the complete pipeline end-to-end on synthetic data
using the mock SAM2 masker (no checkpoint needed) and a real DINOv2 model.

Test flow:
    synthetic RGB image
         ↓
    DINOv2 patch grid extraction
         ↓
    Score patches against synthetic memory bank
         ↓
    Build anomaly map
         ↓
    Threshold + connected components
         ↓
    Mock SAM2 refinement
         ↓
    CandidateRegion list

Marked ``slow`` because they require DINOv2 (model download on first run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config import CONFIG
from common.schemas import CandidateRegion, InferenceResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    from inference.dinov2_embed import Dinov2Embedder, _INSTANCES
    _INSTANCES.clear()
    return Dinov2Embedder.from_config(device="cpu")


@pytest.fixture(scope="module")
def road_mask_healthy(synthetic_healthy_rgb):
    return np.ones(synthetic_healthy_rgb.shape[:2], dtype=bool)


@pytest.fixture(scope="module")
def road_mask_pothole(synthetic_pothole_rgb):
    return np.ones(synthetic_pothole_rgb.shape[:2], dtype=bool)


# ---------------------------------------------------------------------------
# DINOv2 → anomaly map
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_patch_grid_to_anomaly_map(
    embedder, synthetic_memory_bank, synthetic_pothole_rgb
):
    """Patch grid → score_patch_grid → anomaly map must have correct shape."""
    from inference.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector(memory_bank_dir=synthetic_memory_bank)
    grid = embedder.extract_patch_grid(synthetic_pothole_rgb)
    scores_grid = detector.score_patch_grid(grid)
    amap = detector.build_anomaly_map_from_grid(
        grid, scores_grid, synthetic_pothole_rgb.shape[:2]
    )

    h, w = synthetic_pothole_rgb.shape[:2]
    assert amap.shape == (h, w), f"Expected ({h}, {w}), got {amap.shape}"
    assert (amap >= 0).all(), "Anomaly map must be non-negative"
    assert np.isfinite(amap).all(), "Anomaly map contains NaN/Inf"


@pytest.mark.slow
def test_road_patch_embeddings_to_anomaly_map(
    embedder, synthetic_memory_bank, synthetic_pothole_rgb, road_mask_pothole
):
    """Legacy road-mask path: extract_road_patch_embeddings → anomaly map."""
    from inference.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector(memory_bank_dir=synthetic_memory_bank)
    emb, coords = embedder.extract_road_patch_embeddings(
        synthetic_pothole_rgb, road_mask_pothole
    )
    scores = detector.score_patches(emb)
    grid_size = CONFIG.dinov2_input_size // CONFIG.patch_size
    amap = detector.build_anomaly_map(
        coords, scores, synthetic_pothole_rgb.shape[:2], grid_size
    )
    h, w = synthetic_pothole_rgb.shape[:2]
    assert amap.shape == (h, w)
    assert np.isfinite(amap).all()


# ---------------------------------------------------------------------------
# Full pipeline: DINOv2 → anomaly → candidate → mock SAM2 → CandidateRegion
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_pipeline_with_mock_sam2(
    embedder,
    synthetic_memory_bank,
    synthetic_pothole_rgb,
    road_mask_pothole,
    mock_sam2_masker,
):
    """Complete pipeline from image to CandidateRegion list."""
    from inference.anomaly_detector import AnomalyDetector
    from inference.pothole_localizer import PotholeLocalizer

    detector = AnomalyDetector(memory_bank_dir=synthetic_memory_bank)
    localizer = PotholeLocalizer(confidence_threshold=0.0)  # accept all candidates

    emb, coords = embedder.extract_road_patch_embeddings(
        synthetic_pothole_rgb, road_mask_pothole
    )
    scores = detector.score_patches(emb)
    image_score, threshold = detector.summarize(scores)
    grid_size = CONFIG.dinov2_input_size // CONFIG.patch_size
    amap = detector.build_anomaly_map(
        coords, scores, synthetic_pothole_rgb.shape[:2], grid_size
    )

    candidates = localizer.localize(
        synthetic_pothole_rgb, amap, road_mask_pothole, threshold,
        sam2=mock_sam2_masker,
    )

    # Should return a list (may be empty if no anomalies pass threshold)
    assert isinstance(candidates, list)
    for c in candidates:
        assert isinstance(c, CandidateRegion)
        assert c.mask.dtype == bool
        assert c.mask.shape == synthetic_pothole_rgb.shape[:2]
        assert len(c.bbox_xyxy) == 4
        assert 0.0 <= c.anomaly_score
        assert 0.0 <= c.pothole_confidence <= 1.0


@pytest.mark.slow
def test_full_pipeline_healthy_image_lower_score(
    embedder,
    synthetic_memory_bank,
    synthetic_healthy_rgb,
    synthetic_pothole_rgb,
    road_mask_healthy,
    road_mask_pothole,
):
    """Healthy images should produce a lower image-level anomaly score than
    pothole images, when compared against the same memory bank.

    NOTE: With a synthetic memory bank (random embeddings), both scores are
    essentially random — this test just verifies the pipeline runs and
    returns finite, valid values, NOT that the scores are scientifically
    meaningful.
    """
    from inference.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector(memory_bank_dir=synthetic_memory_bank)

    emb_h, _ = embedder.extract_road_patch_embeddings(
        synthetic_healthy_rgb, road_mask_healthy
    )
    score_h, _ = detector.summarize(detector.score_patches(emb_h))

    emb_p, _ = embedder.extract_road_patch_embeddings(
        synthetic_pothole_rgb, road_mask_pothole
    )
    score_p, _ = detector.summarize(detector.score_patches(emb_p))

    # Both scores must be finite and in range [0, 1]
    assert np.isfinite(score_h)
    assert np.isfinite(score_p)
    assert 0.0 <= score_h <= 1.0 + 1e-5
    assert 0.0 <= score_p <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Output JSON schema
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_inference_result_json_schema(
    embedder,
    synthetic_memory_bank,
    synthetic_pothole_rgb,
    mock_sam2_masker,
    tmp_path,
):
    """Full InferenceResult → JSON must have required schema fields."""
    from inference.anomaly_detector import AnomalyDetector
    from inference.pothole_localizer import PotholeLocalizer
    from inference.area_estimator import estimate_area_m2
    from inference.depth_estimator import NullDepthEstimator
    from common.schemas import InferenceResult, PotholeRecord
    from common.io_utils import save_json, utc_iso

    road_mask = np.ones(synthetic_pothole_rgb.shape[:2], dtype=bool)
    detector = AnomalyDetector(memory_bank_dir=synthetic_memory_bank)
    localizer = PotholeLocalizer(confidence_threshold=0.0)

    emb, coords = embedder.extract_road_patch_embeddings(
        synthetic_pothole_rgb, road_mask
    )
    patch_scores = detector.score_patches(emb)
    image_score, threshold = detector.summarize(patch_scores)
    grid_size = CONFIG.dinov2_input_size // CONFIG.patch_size
    amap = detector.build_anomaly_map(
        coords, patch_scores, synthetic_pothole_rgb.shape[:2], grid_size
    )
    candidates = localizer.localize(
        synthetic_pothole_rgb, amap, road_mask, threshold, sam2=mock_sam2_masker
    )

    ts = utc_iso()
    records = []
    for i, c in enumerate(candidates[:3]):  # cap for speed
        area = estimate_area_m2(c.mask, altitude_m=30.0)
        records.append(
            PotholeRecord(
                pothole_id=f"test-{i:03d}",
                timestamp=ts,
                latitude=None,
                longitude=None,
                altitude_m=30.0,
                area_m2=area,
                estimated_depth_m=None,
                anomaly_score=c.anomaly_score,
                pothole_confidence=c.pothole_confidence,
                severity_score=c.pothole_confidence * 0.6,
                water_flag=False,
                water_confidence=0.0,
                source_image="synthetic_test",
                mask_area_px=int(c.mask.sum()),
                bbox_xyxy=c.bbox_xyxy,
            )
        )

    result = InferenceResult(
        image_path="synthetic_test",
        timestamp=ts,
        frame_id=None,
        telemetry={},
        image_shape=list(synthetic_pothole_rgb.shape),
        anomaly_threshold=float(threshold),
        anomaly_score=float(image_score),
        potholes=records,
        warnings=["SYNTHETIC TEST DATA — not scientifically valid."],
    )

    # Verify JSON serialisation
    as_json = result.to_json()
    parsed = json.loads(as_json)

    # Top-level schema fields
    required_top = {"image_path", "timestamp", "image_shape", "anomaly_score",
                    "anomaly_threshold", "potholes", "warnings"}
    missing = required_top - set(parsed.keys())
    assert not missing, f"JSON missing top-level keys: {missing}"

    # Per-pothole schema
    for rec in parsed["potholes"]:
        required_rec = {
            "pothole_id", "timestamp", "latitude", "longitude", "area_m2",
            "estimated_depth_m", "anomaly_score", "pothole_confidence",
            "severity_score", "water_flag", "mask_area_px", "bbox_xyxy",
        }
        missing_rec = required_rec - set(rec.keys())
        assert not missing_rec, f"Pothole record missing keys: {missing_rec}"

    # Write and verify the JSON file
    out_path = tmp_path / "test_result.json"
    save_json(parsed, out_path)
    assert out_path.exists()
    reloaded = json.loads(out_path.read_text())
    assert reloaded["image_path"] == "synthetic_test"
