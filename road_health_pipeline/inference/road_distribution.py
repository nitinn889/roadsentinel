"""Healthy-road feature distribution modeling and domain-adaptive normalization.

This module replaces fragile raw 1-NN Euclidean distance with statistical
distribution modeling of healthy road appearances across multiple domains
(real-world asphalt and synthetic CARLA/Unreal road surfaces).

Key Innovations
---------------
1. Multi-Domain Representation:
   Maintains clusters for distinct road material domains (e.g. real asphalt,
   simulated UE5/CARLA procedural surfaces, wet asphalt).

2. Statistical Distribution Scoring:
   Instead of an uncalibrated absolute distance, computes variance-scaled
   Mahalanobis or temperature-scaled cosine distance against healthy clusters:
       z_i = min_k (||x_i - mu_k|| / sigma_k)

3. Local Road-Adaptive Normalization (Self-Referential Contrast):
   For any individual survey frame (real or simulated), evaluates candidate
   patches relative to the scene's own healthy road baseline:
       Delta s_i = max(0, s_i - median_{road}(s))
   This step directly eliminates global domain gaps, camera sensor gain differences,
   and ambient illumination shifts between real-world and CARLA imagery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DomainCluster:
    """Statistical summary of healthy road features for a specific domain."""
    name: str
    centroid: np.ndarray  # Shape: (dim,)
    variance: np.ndarray  # Shape: (dim,) or scalar
    num_samples: int = 0
    std_norm: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "centroid": self.centroid.tolist(),
            "variance": self.variance.tolist(),
            "num_samples": self.num_samples,
            "std_norm": float(self.std_norm),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DomainCluster":
        return cls(
            name=d["name"],
            centroid=np.array(d["centroid"], dtype=np.float32),
            variance=np.array(d["variance"], dtype=np.float32),
            num_samples=d.get("num_samples", 0),
            std_norm=d.get("std_norm", 1.0),
        )


class RoadDistributionModel:
    """Learns, stores, and evaluates healthy road feature distributions."""

    def __init__(
        self,
        clusters: Optional[List[DomainCluster]] = None,
        global_mean: Optional[np.ndarray] = None,
        global_std: Optional[np.ndarray] = None,
        embed_dim: int = 384,
    ) -> None:
        self.clusters: List[DomainCluster] = clusters or []
        self.embed_dim = embed_dim
        self.global_mean: np.ndarray = (
            global_mean if global_mean is not None else np.zeros((embed_dim,), dtype=np.float32)
        )
        self.global_std: np.ndarray = (
            global_std if global_std is not None else np.ones((embed_dim,), dtype=np.float32)
        )

    def add_domain_cluster(
        self,
        name: str,
        embeddings: np.ndarray,
    ) -> DomainCluster:
        """Fit a healthy cluster from patch embeddings of a specific domain."""
        if len(embeddings) == 0:
            raise ValueError(f"Cannot fit domain cluster '{name}' with 0 embeddings")

        emb = embeddings.astype(np.float32)
        centroid = np.mean(emb, axis=0)
        variance = np.var(emb, axis=0) + 1e-6
        std_norm = float(np.mean(np.sqrt(variance)))

        cluster = DomainCluster(
            name=name,
            centroid=centroid,
            variance=variance,
            num_samples=len(emb),
            std_norm=std_norm,
        )
        self.clusters.append(cluster)
        self._recompute_global_stats()
        log.info(
            "Added healthy domain cluster '%s' with %d samples (dim=%d, avg std=%.4f)",
            name, len(emb), emb.shape[1], std_norm,
        )
        return cluster

    def _recompute_global_stats(self) -> None:
        if not self.clusters:
            return
        all_centroids = np.stack([c.centroid for c in self.clusters], axis=0)
        self.global_mean = np.mean(all_centroids, axis=0)
        all_vars = np.stack([c.variance for c in self.clusters], axis=0)
        self.global_std = np.sqrt(np.mean(all_vars, axis=0) + 1e-6)

    def score_patches(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine_min",
    ) -> np.ndarray:
        """Compute anomaly score for each patch embedding against healthy clusters.

        Parameters
        ----------
        embeddings:
            Shape (N, dim) float32 patch embeddings.
        metric:
            'cosine_min': 1 - max(cosine similarity across healthy clusters)
            'mahalanobis_min': minimum variance-normalized Euclidean distance

        Returns
        -------
        scores: Shape (N,) float32 in [0, 1] range (higher = more anomalous).
        """
        if len(embeddings) == 0:
            return np.empty(0, dtype=np.float32)

        q = embeddings.astype(np.float32)
        # Normalize queries
        q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)

        if not self.clusters:
            # Fallback if no clusters: score based on deviation from global mean
            diff = q - self.global_mean
            dist = np.linalg.norm(diff / self.global_std, axis=-1)
            return np.clip(dist / 10.0, 0.0, 1.0).astype(np.float32)

        if metric == "cosine_min":
            # For each cluster centroid, compute cosine similarity to query
            sims = []
            for c in self.clusters:
                c_norm = c.centroid / (np.linalg.norm(c.centroid) + 1e-8)
                sim = np.dot(q_norm, c_norm)
                sims.append(sim)
            max_sim = np.max(np.stack(sims, axis=0), axis=0)
            max_sim = np.clip(max_sim, -1.0, 1.0)
            # Anomaly score: 0 = matches best healthy cluster, 1 = distant
            scores = 1.0 - max_sim
            return np.clip(scores, 0.0, 1.0).astype(np.float32)

        elif metric == "mahalanobis_min":
            dists = []
            for c in self.clusters:
                diff = q - c.centroid
                scaled_diff = diff / np.sqrt(c.variance)
                d = np.sqrt(np.mean(scaled_diff ** 2, axis=-1))
                dists.append(d)
            min_dist = np.min(np.stack(dists, axis=0), axis=0)
            # Sigmoid/tanh mapping to [0, 1]
            scores = 1.0 - 1.0 / (1.0 + np.exp((min_dist - 2.5) * 1.5))
            return np.clip(scores, 0.0, 1.0).astype(np.float32)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def apply_local_road_normalization(
        raw_scores: np.ndarray,
        road_mask_indices: Optional[np.ndarray] = None,
        percentile_baseline: float = 50.0,
        spread_scale: float = 3.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Self-referential normalization: measures deviation from the road's own baseline.

        Cancels out global domain gap and uniform lighting offsets in both CARLA
        and real-world images.

        Parameters
        ----------
        raw_scores:
            Shape (N,) float32 raw patch anomaly scores.
        road_mask_indices:
            Optional boolean mask or index array indicating patches on valid road.
        percentile_baseline:
            Percentile to use as healthy road baseline (default: 50.0 = median).
        spread_scale:
            Multiplier on inter-quartile range or MAD to calibrate anomaly scale.

        Returns
        -------
        normalized_scores: Shape (N,) float32 in [0, 1] range.
        stats: Dictionary containing baseline, spread, and max elevation.
        """
        if len(raw_scores) == 0:
            return np.empty(0, dtype=np.float32), {}

        road_scores = (
            raw_scores[road_mask_indices]
            if road_mask_indices is not None and np.any(road_mask_indices)
            else raw_scores
        )

        baseline = float(np.percentile(road_scores, percentile_baseline))
        q75 = float(np.percentile(road_scores, 75.0))
        q25 = float(np.percentile(road_scores, 25.0))
        iqr = max(1e-4, q75 - q25)
        std = max(1e-4, float(np.std(road_scores)))

        # Effective scale factor
        scale = max(0.05, iqr * spread_scale)

        # Contrast elevation above the road's self-baseline
        elevations = raw_scores - baseline
        # Rectify non-elevated patches to 0
        rectified = np.maximum(0.0, elevations)

        # Smooth saturation curve mapping elevated score into [0, 1]
        normalized = 1.0 - np.exp(-rectified / scale)
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

        stats = {
            "baseline_median": baseline,
            "road_iqr": iqr,
            "road_std": std,
            "scale": scale,
            "max_elevation": float(np.max(elevations)) if len(elevations) > 0 else 0.0,
            "mean_normalized": float(np.mean(normalized)),
        }
        return normalized, stats

    def save(self, path: Path) -> None:
        """Serialize distribution model to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "embed_dim": self.embed_dim,
            "global_mean": self.global_mean.tolist(),
            "global_std": self.global_std.tolist(),
            "clusters": [c.to_dict() for c in self.clusters],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Saved RoadDistributionModel to %s (%d clusters)", path, len(self.clusters))

    @classmethod
    def load(cls, path: Path) -> "RoadDistributionModel":
        """Deserialize distribution model from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        clusters = [DomainCluster.from_dict(c) for c in data.get("clusters", [])]
        model = cls(
            clusters=clusters,
            global_mean=np.array(data["global_mean"], dtype=np.float32),
            global_std=np.array(data["global_std"], dtype=np.float32),
            embed_dim=data.get("embed_dim", 384),
        )
        log.info("Loaded RoadDistributionModel from %s (%d clusters)", path, len(clusters))
        return model
