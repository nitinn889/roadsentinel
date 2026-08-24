from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return dense metric depth in metres when available; otherwise None."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


@dataclass
class NullDepthEstimator(DepthEstimator):
    """Scientifically honest default: no metric RGB depth is claimed without a provided model."""

    @property
    def name(self) -> str:
        return "unavailable"

    def estimate(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        return None


class ExternalMetricDepthEstimator(DepthEstimator):
    """Adapter for a separately supplied model.

    The callable must have signature rgb -> HxW float32 depth in metres.
    """
    def __init__(self, predictor, name: str = "external_metric_depth"):
        self.predictor = predictor
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def estimate(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        depth = self.predictor(rgb)
        if depth is None:
            return None
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim != 2 or d.shape != rgb.shape[:2]:
            raise ValueError("Depth estimator must return HxW metric depth in metres")
        return d


def depth_from_carla_ground_truth(depth_rgb: np.ndarray, far_clip_m: float = 1000.0) -> np.ndarray:
    """Decode CARLA's standard 24-bit logarithmic depth encoding to metres.

    CARLA RGB camera depth images encode normalized depth in RGB bytes. This helper is
    intended only for evaluation/ground truth, never as an input to real inference.
    """
    r = depth_rgb[..., 0].astype(np.float32)
    g = depth_rgb[..., 1].astype(np.float32)
    b = depth_rgb[..., 2].astype(np.float32)
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1.0)
    return normalized * float(far_clip_m)


def masked_depth_stats(depth_m: np.ndarray, mask: np.ndarray) -> Optional[dict]:
    vals = depth_m[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return None
    return {
        "median_m": float(np.median(vals)),
        "min_m": float(np.min(vals)),
        "max_m": float(np.max(vals)),
        "p90_m": float(np.percentile(vals, 90)),
    }
