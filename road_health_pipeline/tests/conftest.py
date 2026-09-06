"""Shared pytest fixtures for RoadSentinel pipeline tests.

Fixtures provided
-----------------
synthetic_healthy_rgb   — 512×512 uint8 RGB road image
synthetic_pothole_rgb   — 512×512 uint8 RGB image with a dark oval pothole
synthetic_road_mask     — 512×512 all-True boolean mask (whole image is "road")
synthetic_memory_bank   — (tmp_path) builds a tiny FAISS index without any model
mock_sam2_masker        — stub RoadMasker that returns a centred rectangle mask
"""

from __future__ import annotations

import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

# Ensure the pipeline root is on sys.path so imports work regardless of
# where pytest is invoked from.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from tests.make_fixtures import make_healthy, make_pothole
from common.schemas import SegmentationResult


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_healthy_rgb() -> np.ndarray:
    """512×512 uint8 RGB synthetic road image (no defects)."""
    return make_healthy(seed=0)


@pytest.fixture(scope="session")
def synthetic_pothole_rgb() -> np.ndarray:
    """512×512 uint8 RGB image with a simulated pothole."""
    return make_pothole(seed=999)


@pytest.fixture(scope="session")
def synthetic_road_mask(synthetic_healthy_rgb) -> np.ndarray:
    """512×512 all-True boolean mask (whole image treated as road)."""
    h, w = synthetic_healthy_rgb.shape[:2]
    return np.ones((h, w), dtype=bool)


# ---------------------------------------------------------------------------
# Memory bank fixture (no model required)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_memory_bank(tmp_path) -> Path:
    """Build a minimal FAISS memory bank from random embeddings.

    Dimension (384) matches dinov2_vits14.  No model loading required.
    Returns the directory path.
    """
    import json

    dim = 384
    n_vectors = 50
    rng = np.random.default_rng(42)

    # Simulate normalised healthy-road patch embeddings
    emb = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    bank_dir = tmp_path / "memory_bank"
    bank_dir.mkdir()
    faiss.write_index(index, str(bank_dir / "index.faiss"))
    np.save(str(bank_dir / "embeddings.npy"), emb)

    metadata = {
        "embedding_dim": dim,
        "num_source_images": 3,
        "num_source_patch_embeddings": n_vectors,
        "num_memory_embeddings": n_vectors,
        "model": "dinov2_vits14",
        "patch_size": 14,
        "sam2_checkpoint": "mock",
        "sam2_model_cfg": "mock",
        "healthy_dataset_definition": "synthetic test fixture",
        "faiss_metric": "inner_product_after_L2_normalization",
        "seed": 42,
    }
    (bank_dir / "metadata.json").write_text(json.dumps(metadata))
    return bank_dir


# ---------------------------------------------------------------------------
# Mock SAM2 masker
# ---------------------------------------------------------------------------

class _MockSAM2Masker:
    """Stub RoadMasker that returns deterministic masks without loading SAM2."""

    def get_road_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        return mask

    def refine_box(
        self, image_rgb: np.ndarray, box_xyxy: list[float]
    ) -> SegmentationResult:
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        # Expand slightly to simulate SAM2 border capture
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return SegmentationResult(
            mask=mask,
            confidence=0.80,
            bbox_xyxy=[x1, y1, x2, y2],
            area_px=int(mask.sum()),
        )


@pytest.fixture(scope="session")
def mock_sam2_masker() -> _MockSAM2Masker:
    """Stub RoadMasker that returns rectangular masks without loading SAM2."""
    return _MockSAM2Masker()


# ---------------------------------------------------------------------------
# pytest marks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_checkpoint: mark test as requiring SAM2 checkpoint file",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (requires model loading)",
    )
