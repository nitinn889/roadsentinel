"""DINOv2 patch-level anomaly detection for RoadSentinel.

Algorithm overview
------------------
1. A memory bank of healthy-road DINOv2 patch features is built offline
   (see ``memory_bank/build_memory_bank.py``) and stored as a FAISS
   inner-product index with L2-normalised vectors.

2. At inference time, patch tokens from a query image are L2-normalised
   and searched against the memory bank (k-NN).

3. Anomaly score per patch = 1 - mean cosine similarity to the k nearest
   healthy neighbours.  Higher score → more anomalous.

4. Patch scores are mapped back to a 2D grid and upsampled to the original
   image resolution to produce a spatial anomaly heat-map.

Threshold
---------
The per-image anomaly threshold is taken as the ``CONFIG.anomaly_percentile``
(default 98th) of the road-patch score distribution for that image.  This is
an image-relative threshold: it highlights patches that are unusually anomalous
*compared to the rest of the same image's road patches*, which is more robust
than a global absolute threshold when road appearance varies between scenes.

Calibrating ``CONFIG.anomaly_percentile`` against a labelled validation set
is required before deployment; the default is a reasonable starting point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import json

import cv2
import numpy as np

from config import CONFIG

log = logging.getLogger(__name__)


class AnomalyDetector:
    """Query a healthy-road FAISS memory bank to score DINOv2 patch embeddings.

    Parameters
    ----------
    memory_bank_dir:
        Directory containing ``index.faiss``, ``embeddings.npy``, and
        ``metadata.json`` produced by ``build_memory_bank.py``.
    """

    def __init__(self, memory_bank_dir: Path = CONFIG.memory_bank_dir) -> None:
        import faiss

        self.memory_bank_dir = Path(memory_bank_dir)
        index_path = self.memory_bank_dir / "index.faiss"
        emb_path = self.memory_bank_dir / "embeddings.npy"
        meta_path = self.memory_bank_dir / "metadata.json"

        missing = [p for p in [index_path, emb_path, meta_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Memory bank is incomplete.  Missing files:\n"
                + "\n".join(f"  {p}" for p in missing)
                + "\nRun: python memory_bank/build_memory_bank.py --healthy-dir <path>"
            )

        self.index = faiss.read_index(str(index_path))
        self.embeddings = np.load(emb_path, mmap_mode="r")
        self.metadata = json.loads(meta_path.read_text())

        if self.embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"FAISS index dimension ({self.index.d}) does not match "
                f"embeddings.npy ({self.embeddings.shape[1]})"
            )

        log.info(
            "Memory bank loaded: %d vectors, dim=%d (built from %d images)",
            self.index.ntotal,
            self.index.d,
            self.metadata.get("num_source_images", "?"),
        )

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_patches(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for a batch of patch embeddings.

        Parameters
        ----------
        embeddings:
            Shape ``(N, embed_dim)`` float32.  These are the raw patch tokens
            (not necessarily L2-normalised; this method normalises them).

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` float32.  Each value is in ``[0, 1]``; higher means
            more anomalous.  0 = identical to a healthy-road patch;
            1 = maximally dissimilar.
        """
        import faiss

        if len(embeddings) == 0:
            return np.empty(0, dtype=np.float32)

        q = embeddings.astype(np.float32, copy=True)
        faiss.normalize_L2(q)
        # Inner product after L2 normalisation = cosine similarity
        similarity, _ = self.index.search(q, CONFIG.knn_k)
        # Mean cosine similarity to k nearest healthy neighbours
        # Clip similarity to [-1, 1] to guard against floating-point noise
        sim = np.clip(similarity, -1.0, 1.0).mean(axis=1)
        # Anomaly score: 0 = healthy, 1 = maximally anomalous
        return (1.0 - sim).astype(np.float32)

    # ------------------------------------------------------------------
    # Anomaly map
    # ------------------------------------------------------------------

    def build_anomaly_map(
        self,
        patch_coords: np.ndarray,
        scores: np.ndarray,
        image_shape: tuple[int, int],
        grid_size: int,
    ) -> np.ndarray:
        """Upsample patch-level anomaly scores to a full-resolution heat-map.

        Parameters
        ----------
        patch_coords:
            Shape ``(N, 2)`` int — ``(row, col)`` grid indices of each patch.
        scores:
            Shape ``(N,)`` float32 — anomaly score for each patch.
        image_shape:
            ``(H, W)`` of the original RGB image.
        grid_size:
            Number of patches per side (e.g. 37 for ViT-S/14 at 518 px).

        Returns
        -------
        np.ndarray
            Shape ``(H, W)`` float32 anomaly heat-map with values ≥ 0.
            Patches not covered by ``patch_coords`` have score 0.0.

        Notes
        -----
        Bilinear interpolation (``INTER_LINEAR``) is used for upsampling rather
        than cubic, to avoid the negative overshoot artifacts that bicubic can
        introduce when upsampling score maps with large local maxima.
        """
        h, w = image_shape
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        for (r, c), s in zip(patch_coords, scores):
            grid[int(r), int(c)] = max(grid[int(r), int(c)], float(s))
        amap = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.maximum(amap, 0.0)

    def build_anomaly_map_from_grid(
        self,
        patch_grid: np.ndarray,
        scores_grid: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Build anomaly map from a full patch grid (no road-mask filtering).

        Parameters
        ----------
        patch_grid:
            Shape ``(H_p, W_p, dim)`` — full patch feature grid from
            ``Dinov2Embedder.extract_patch_grid()``.
        scores_grid:
            Shape ``(H_p, W_p)`` — anomaly score per patch.
        image_shape:
            ``(H, W)`` of the original image.

        Returns
        -------
        np.ndarray
            Shape ``(H, W)`` float32.
        """
        h, w = image_shape
        amap = cv2.resize(scores_grid, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.maximum(amap, 0.0)

    def score_patch_grid(self, patch_grid: np.ndarray) -> np.ndarray:
        """Score every patch in a full grid.

        Parameters
        ----------
        patch_grid:
            Shape ``(H_p, W_p, dim)`` float32 from
            ``Dinov2Embedder.extract_patch_grid()``.

        Returns
        -------
        np.ndarray
            Shape ``(H_p, W_p)`` float32 anomaly scores.
        """
        h_p, w_p, dim = patch_grid.shape
        flat = patch_grid.reshape(-1, dim)
        scores_flat = self.score_patches(flat)
        return scores_flat.reshape(h_p, w_p)

    # ------------------------------------------------------------------
    # Summarisation and normalisation
    # ------------------------------------------------------------------

    def summarize(self, scores: np.ndarray) -> tuple[float, float]:
        """Summarise per-patch scores into an image-level anomaly score and threshold.

        Parameters
        ----------
        scores:
            1-D array of per-patch anomaly scores.

        Returns
        -------
        image_score : float
            The ``CONFIG.anomaly_percentile`` of the patch scores.  This is the
            "headline" anomaly score for the image.
        threshold : float
            Same value — used as the per-image threshold for candidate extraction.
            Downstream code compares individual patch scores against this value to
            identify anomalous regions.

        Note
        ----
        The threshold was previously hard-coded to the 75th percentile, which was
        inconsistent with ``CONFIG.anomaly_percentile`` (default 98).  Both values
        are now derived from the same configurable percentile.
        """
        if len(scores) == 0:
            return 0.0, 0.0
        image_score = float(np.percentile(scores, CONFIG.anomaly_percentile))
        return image_score, image_score

    @staticmethod
    def normalize_anomaly_map(amap: np.ndarray) -> np.ndarray:
        """Normalise an anomaly map to [0, 1] for visualisation.

        Returns an all-zero map if the input has no variation.
        """
        lo = float(amap.min())
        hi = float(amap.max())
        if hi <= lo:
            return np.zeros_like(amap, dtype=np.float32)
        return ((amap - lo) / (hi - lo)).astype(np.float32)
