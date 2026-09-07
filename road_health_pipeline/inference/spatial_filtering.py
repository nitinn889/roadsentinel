"""Spatial consistency and contextual false-positive filtering.

This module enforces spatial coherence, neighborhood contrast, and contextual
constraints on candidate anomaly detections to eliminate false positives
caused by:
  - Isolated single-patch feature noise
  - Uniform cast shadows (building/tree shadows)
  - Painted road markings (white lane lines, yellow centerlines)
  - Camera border vignetting / horizon transitions
  - Smooth asphalt patches and road material transitions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Detailed audit of which filters passed or failed for a candidate."""
    passed: bool
    rejection_reason: Optional[str] = None
    spatial_score: float = 1.0
    neighborhood_contrast: float = 1.0
    shadow_prob: float = 0.0
    marking_prob: float = 0.0


class SpatialContextFilter:
    """Enforces spatial consistency, local neighborhood contrast, and scene context."""

    def __init__(
        self,
        min_cluster_patches: int = 2,
        min_neighborhood_contrast: float = 0.15,
        max_shadow_prob: float = 0.65,
        max_marking_prob: float = 0.35,
        border_margin_fraction: float = 0.03,
    ) -> None:
        self.min_cluster_patches = min_cluster_patches
        self.min_neighborhood_contrast = min_neighborhood_contrast
        self.max_shadow_prob = max_shadow_prob
        self.max_marking_prob = max_marking_prob
        self.border_margin_fraction = border_margin_fraction

    # ------------------------------------------------------------------
    # 1. 2D Spatial Consistency on Patch Grid
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_patch_scores(
        score_grid: np.ndarray,
        kernel_size: int = 3,
        sigma: float = 0.8,
    ) -> np.ndarray:
        """Apply 2D Gaussian smoothing to patch score grid to enforce spatial coherence."""
        if score_grid.size == 0:
            return score_grid
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(score_grid.astype(np.float32), (k, k), sigma)

    @staticmethod
    def filter_isolated_spikes(
        binary_mask: np.ndarray,
        min_connected_pixels: int = 40,
    ) -> np.ndarray:
        """Remove isolated single-pixel/single-patch noise spikes via connected components."""
        if not np.any(binary_mask):
            return binary_mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8), connectivity=8
        )
        cleaned = np.zeros_like(binary_mask, dtype=bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_connected_pixels:
                cleaned[labels == i] = True
        return cleaned

    # ------------------------------------------------------------------
    # 2. Neighborhood Collar Contrast
    # ------------------------------------------------------------------

    @staticmethod
    def compute_neighborhood_contrast(
        anomaly_map: np.ndarray,
        candidate_mask: np.ndarray,
        dilation_radius: int = 15,
    ) -> Tuple[float, float, float]:
        """Compute relative contrast between the candidate and its surrounding road collar.

        Returns
        -------
        contrast_ratio: (inside_mean - collar_mean) / max(collar_mean, 0.05)
        inside_mean: Mean anomaly score inside candidate
        collar_mean: Mean anomaly score in surrounding ring
        """
        if not np.any(candidate_mask):
            return 0.0, 0.0, 0.0

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1)
        )
        dilated = cv2.dilate(candidate_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        collar = dilated & ~candidate_mask

        inside_vals = anomaly_map[candidate_mask]
        inside_mean = float(np.mean(inside_vals)) if len(inside_vals) > 0 else 0.0

        collar_vals = anomaly_map[collar]
        collar_mean = float(np.mean(collar_vals)) if len(collar_vals) > 0 else inside_mean

        denom = max(collar_mean, 0.05)
        contrast = float(np.clip((inside_mean - collar_mean) / denom, -1.0, 5.0))
        return contrast, inside_mean, collar_mean

    # ------------------------------------------------------------------
    # 3. Contextual Rejectors
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_shadow_likelihood(
        rgb_image: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> float:
        """Estimate likelihood that the candidate is a benign cast shadow.

        Cast shadows have significant luminance drop but their internal texture
        variance is uniform (matching smooth or grainy pavement without edge disruption).
        """
        if not np.any(candidate_mask):
            return 0.0

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        inside = gray[candidate_mask]
        if len(inside) < 30:
            return 0.0

        kernel = np.ones((19, 19), np.uint8)
        collar_mask = cv2.dilate(candidate_mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~candidate_mask
        outside = gray[collar_mask] if np.any(collar_mask) else inside

        in_mean = float(np.mean(inside))
        out_mean = float(np.mean(outside))
        in_var = float(np.var(inside))
        out_var = float(np.var(outside))

        # Shadow check:
        # 1. Inside is darker than outside (out_mean - in_mean > 15)
        # 2. Inside texture variance is very low (< 25.0) or closely matches outside texture without broken rims
        luminance_drop = max(0.0, out_mean - in_mean)
        uniformity = 1.0 - min(1.0, in_var / 80.0)

        # Laplacian edge density inside candidate
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_inside = float(np.mean(np.abs(lap[candidate_mask])))

        # Deep uniform shadow has high drop, high uniformity, low laplacian
        shadow_score = (
            0.45 * min(1.0, luminance_drop / 60.0)
            + 0.35 * uniformity
            + 0.20 * (1.0 - min(1.0, lap_inside / 12.0))
        )
        return float(np.clip(shadow_score, 0.0, 1.0))

    @staticmethod
    def evaluate_road_marking_likelihood(
        rgb_image: np.ndarray,
        candidate_mask: np.ndarray,
        aspect_ratio: float,
    ) -> float:
        """Detect painted road markings (yellow centerlines, white lane dividers)."""
        if not np.any(candidate_mask):
            return 0.0

        pixels = rgb_image[candidate_mask]
        if len(pixels) < 20:
            return 0.0

        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

        # Yellow marking detection
        yellow_mask = (
            (hsv[:, 0] >= 11) & (hsv[:, 0] <= 42)
            & (hsv[:, 1] >= 45) & (hsv[:, 2] >= 60)
        )
        yellow_fraction = float(np.mean(yellow_mask))

        # White lane line detection
        white_mask = (hsv[:, 1] <= 35) & (hsv[:, 2] >= 140)
        white_fraction = float(np.mean(white_mask))

        # Lane markings are elongated (high aspect ratio)
        shape_factor = min(1.0, aspect_ratio / 3.0) if aspect_ratio > 1.5 else 0.5

        marking_score = max(
            yellow_fraction * 1.5,
            white_fraction * shape_factor * 1.3
        )
        return float(np.clip(marking_score, 0.0, 1.0))

    @staticmethod
    def is_boundary_or_horizon(
        bbox_xyxy: list[int],
        image_shape: tuple[int, int],
        margin_frac: float = 0.03,
    ) -> bool:
        """Check if candidate touches outer image borders or horizon vanishing zone."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        mx = max(15, int(w * margin_frac))
        my = max(15, int(h * margin_frac))

        # Horizon / distant sky perspective
        if y1 < 0.08 * h and y2 < 0.15 * h:
            return True

        # Frame border margins
        if x1 <= mx or x2 >= w - mx or y1 <= my or y2 >= h - my:
            return True

        return False

    # ------------------------------------------------------------------
    # Comprehensive Candidate Evaluation
    # ------------------------------------------------------------------

    def evaluate_candidate(
        self,
        rgb_image: np.ndarray,
        candidate_mask: np.ndarray,
        bbox_xyxy: list[int],
        anomaly_map: np.ndarray,
        aspect_ratio: float = 1.0,
        shape_circularity: float = 0.5,
    ) -> FilterResult:
        """Apply full filter suite to determine whether a candidate is a true defect."""
        h, w = rgb_image.shape[:2]

        # 1. Boundary / Horizon rejection
        if self.is_boundary_or_horizon(bbox_xyxy, (h, w), self.border_margin_fraction):
            return FilterResult(
                passed=False,
                rejection_reason="boundary_or_horizon_artifact",
                spatial_score=0.1,
            )

        # 2. Neighborhood contrast
        contrast, in_mean, collar_mean = self.compute_neighborhood_contrast(
            anomaly_map, candidate_mask
        )
        if contrast < self.min_neighborhood_contrast and in_mean < 0.60:
            return FilterResult(
                passed=False,
                rejection_reason="insufficient_neighborhood_contrast",
                neighborhood_contrast=contrast,
            )

        # 3. Road marking rejection
        marking_prob = self.evaluate_road_marking_likelihood(rgb_image, candidate_mask, aspect_ratio)
        if marking_prob > self.max_marking_prob:
            return FilterResult(
                passed=False,
                rejection_reason="painted_road_marking",
                marking_prob=marking_prob,
            )

        # 4. Shadow rejection
        shadow_prob = self.evaluate_shadow_likelihood(rgb_image, candidate_mask)
        if shadow_prob > self.max_shadow_prob:
            return FilterResult(
                passed=False,
                rejection_reason="benign_cast_shadow",
                shadow_prob=shadow_prob,
            )

        # 5. Composite spatial score
        spatial_score = float(
            np.clip(
                0.40 * min(1.0, contrast / 0.8)
                + 0.30 * shape_circularity
                + 0.30 * (1.0 - shadow_prob),
                0.0,
                1.0,
            )
        )

        return FilterResult(
            passed=True,
            spatial_score=spatial_score,
            neighborhood_contrast=contrast,
            shadow_prob=shadow_prob,
            marking_prob=marking_prob,
        )
