"""Tests for memory_bank/coreset.py and the memory bank build pipeline.

These tests do NOT require a DINOv2 model — they build a memory bank from
pre-computed random embeddings to verify coreset correctness, shapes, and
reproducibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


# ---------------------------------------------------------------------------
# Coreset tests
# ---------------------------------------------------------------------------

class TestRandomPresample:
    def test_no_op_when_small(self):
        from memory_bank.coreset import random_presample
        x = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
        result = random_presample(x, max_points=20, seed=42)
        assert result.shape == (10, 4)

    def test_subsamples_when_large(self):
        from memory_bank.coreset import random_presample
        x = np.random.default_rng(0).standard_normal((1000, 4)).astype(np.float32)
        result = random_presample(x, max_points=100, seed=42)
        assert result.shape == (100, 4)

    def test_deterministic(self):
        from memory_bank.coreset import random_presample
        x = np.random.default_rng(0).standard_normal((500, 8)).astype(np.float32)
        a = random_presample(x, max_points=50, seed=7)
        b = random_presample(x, max_points=50, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_result(self):
        from memory_bank.coreset import random_presample
        x = np.random.default_rng(0).standard_normal((500, 8)).astype(np.float32)
        a = random_presample(x, max_points=50, seed=1)
        b = random_presample(x, max_points=50, seed=2)
        assert not np.array_equal(a, b)


class TestKCenterGreedy:
    def test_output_count(self):
        from memory_bank.coreset import k_center_greedy
        x = np.random.default_rng(0).standard_normal((100, 16)).astype(np.float32)
        idx = k_center_greedy(x, n_select=10, seed=42)
        assert len(idx) == 10

    def test_indices_in_range(self):
        from memory_bank.coreset import k_center_greedy
        x = np.random.default_rng(0).standard_normal((100, 16)).astype(np.float32)
        idx = k_center_greedy(x, n_select=15, seed=42)
        assert (idx >= 0).all() and (idx < 100).all()

    def test_no_duplicates(self):
        from memory_bank.coreset import k_center_greedy
        x = np.random.default_rng(0).standard_normal((200, 16)).astype(np.float32)
        idx = k_center_greedy(x, n_select=20, seed=42)
        assert len(set(idx.tolist())) == 20, "Coreset indices must be unique"

    def test_deterministic(self):
        from memory_bank.coreset import k_center_greedy
        x = np.random.default_rng(0).standard_normal((100, 16)).astype(np.float32)
        a = k_center_greedy(x, n_select=10, seed=42)
        b = k_center_greedy(x, n_select=10, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_n_select_larger_than_points(self):
        from memory_bank.coreset import k_center_greedy
        x = np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)
        idx = k_center_greedy(x, n_select=20, seed=42)
        assert len(idx) == 5  # capped at len(x)

    def test_zero_select(self):
        from memory_bank.coreset import k_center_greedy
        x = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
        idx = k_center_greedy(x, n_select=0, seed=42)
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# Synthetic memory bank tests (uses conftest.synthetic_memory_bank fixture)
# ---------------------------------------------------------------------------

class TestSyntheticMemoryBank:
    def test_bank_exists(self, synthetic_memory_bank):
        assert (synthetic_memory_bank / "index.faiss").exists()
        assert (synthetic_memory_bank / "embeddings.npy").exists()
        assert (synthetic_memory_bank / "metadata.json").exists()

    def test_memory_bank_shape(self, synthetic_memory_bank):
        emb = np.load(synthetic_memory_bank / "embeddings.npy")
        assert emb.ndim == 2
        assert emb.shape[1] == 384, f"Expected dim 384, got {emb.shape[1]}"

    def test_faiss_dimension_matches(self, synthetic_memory_bank):
        emb = np.load(synthetic_memory_bank / "embeddings.npy")
        index = faiss.read_index(str(synthetic_memory_bank / "index.faiss"))
        assert emb.shape[0] == index.ntotal
        assert emb.shape[1] == index.d

    def test_no_nan_in_bank(self, synthetic_memory_bank):
        emb = np.load(synthetic_memory_bank / "embeddings.npy")
        assert np.isfinite(emb).all(), "Memory bank embeddings contain NaN/Inf"

    def test_metadata_fields(self, synthetic_memory_bank):
        import json
        meta = json.loads((synthetic_memory_bank / "metadata.json").read_text())
        required_keys = {
            "embedding_dim", "num_source_images", "num_memory_embeddings",
            "model", "patch_size",
        }
        missing = required_keys - set(meta.keys())
        assert not missing, f"Metadata missing keys: {missing}"

    def test_faiss_search_returns_results(self, synthetic_memory_bank):
        """Basic sanity: FAISS search on the synthetic bank returns k results."""
        emb = np.load(synthetic_memory_bank / "embeddings.npy")
        index = faiss.read_index(str(synthetic_memory_bank / "index.faiss"))
        query = emb[:1].copy()
        faiss.normalize_L2(query)
        D, I = index.search(query, 5)
        assert D.shape == (1, 5)
        assert I.shape == (1, 5)
        assert (D >= -1.0).all() and (D <= 1.0 + 1e-5).all()
