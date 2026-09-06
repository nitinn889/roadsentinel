"""Tests for inference/dinov2_embed.py.

DINOv2 model reference (dinov2_vits14 @ 518px)
-----------------------------------------------
  Architecture : ViT-Small, patch size 14
  Feature dim  : 384
  Grid size    : 518 // 14 = 37
  Patch tokens : 37 * 37 = 1369 per image
  CLS token    : 1 per image, same dim 384

These tests verify tensor shapes, value finiteness, and model-reuse semantics.
They are marked ``slow`` because they require a real DINOv2 model download
(~330 MB) on first run; subsequent runs use the local torch.hub cache.
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


# ---------------------------------------------------------------------------
# Lazy embedder fixture (loads DINOv2 once for all tests in this module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    """Load DINOv2 once for all tests in this module."""
    from inference.dinov2_embed import Dinov2Embedder
    return Dinov2Embedder.from_config(device="cpu")  # CPU for CI safety


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_patch_grid_shape(embedder, synthetic_healthy_rgb):
    """extract_patch_grid must return (grid, grid, embed_dim).

    For dinov2_vits14 with input_size=518:
        grid  = 518 // 14 = 37
        dim   = 384
        shape = (37, 37, 384)
    """
    grid = embedder.extract_patch_grid(synthetic_healthy_rgb)
    assert grid.ndim == 3, f"Expected 3-D array, got {grid.ndim}-D"
    g = CONFIG.dinov2_input_size // CONFIG.patch_size  # 37
    assert grid.shape == (g, g, 384), (
        f"Expected ({g}, {g}, 384), got {grid.shape}"
    )


@pytest.mark.slow
def test_cls_token_shape(embedder, synthetic_healthy_rgb):
    """extract_cls_token must return shape (384,)."""
    cls = embedder.extract_cls_token(synthetic_healthy_rgb)
    assert cls.ndim == 1, f"Expected 1-D, got {cls.ndim}-D"
    assert cls.shape == (384,), f"Expected (384,), got {cls.shape}"


@pytest.mark.slow
def test_patch_values_finite(embedder, synthetic_healthy_rgb):
    """All patch token values must be finite (no NaN or Inf)."""
    grid = embedder.extract_patch_grid(synthetic_healthy_rgb)
    assert np.isfinite(grid).all(), "Non-finite values in patch grid"


@pytest.mark.slow
def test_cls_values_finite(embedder, synthetic_healthy_rgb):
    """CLS token must be finite."""
    cls = embedder.extract_cls_token(synthetic_healthy_rgb)
    assert np.isfinite(cls).all(), "Non-finite values in CLS token"


@pytest.mark.slow
def test_road_patch_embeddings_with_mask(embedder, synthetic_healthy_rgb, synthetic_road_mask):
    """extract_road_patch_embeddings must return (N, 384) and coords (N, 2)."""
    emb, coords = embedder.extract_road_patch_embeddings(
        synthetic_healthy_rgb, synthetic_road_mask
    )
    assert emb.ndim == 2, f"Expected 2-D embeddings, got {emb.ndim}-D"
    assert emb.shape[1] == 384, f"Expected dim 384, got {emb.shape[1]}"
    assert coords.ndim == 2, f"Expected 2-D coords, got {coords.ndim}-D"
    assert coords.shape[0] == emb.shape[0], "coords and emb row count must match"
    assert coords.shape[1] == 2, "coords must have 2 columns (row, col)"


@pytest.mark.slow
def test_road_patch_embeddings_finite(embedder, synthetic_healthy_rgb, synthetic_road_mask):
    """Road patch embeddings must be finite."""
    emb, _ = embedder.extract_road_patch_embeddings(
        synthetic_healthy_rgb, synthetic_road_mask
    )
    assert np.isfinite(emb).all(), "Non-finite values in road patch embeddings"


# ---------------------------------------------------------------------------
# Model reuse
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_model_reuse():
    """from_config() must return the same underlying model object for same args."""
    from inference.dinov2_embed import Dinov2Embedder, _INSTANCES
    _INSTANCES.clear()  # reset cache for a clean test
    a = Dinov2Embedder.from_config(device="cpu", model_name="dinov2_vits14")
    b = Dinov2Embedder.from_config(device="cpu", model_name="dinov2_vits14")
    assert a is b, "from_config() should return the cached instance"


@pytest.mark.slow
def test_different_devices_create_different_instances():
    """from_config() with different args must create independent instances."""
    from inference.dinov2_embed import Dinov2Embedder, _INSTANCES
    _INSTANCES.clear()
    a = Dinov2Embedder.from_config(device="cpu", model_name="dinov2_vits14")
    b = Dinov2Embedder.from_config(device="cpu", model_name="dinov2_vitb14")
    assert a is not b, "Different model names should produce different instances"


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def test_cuda_fallback_to_cpu_when_unavailable(monkeypatch, synthetic_healthy_rgb):
    """If CUDA is requested but unavailable, should fall back to CPU, not raise."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from inference.dinov2_embed import Dinov2Embedder, _INSTANCES
    _INSTANCES.clear()
    embedder = Dinov2Embedder(device="cuda")  # should not raise
    assert embedder.device.type == "cpu"


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_patch_grid_dtype(embedder, synthetic_healthy_rgb):
    """Output arrays must be float32."""
    grid = embedder.extract_patch_grid(synthetic_healthy_rgb)
    assert grid.dtype == np.float32, f"Expected float32, got {grid.dtype}"
