"""Pothole candidate localisation from a DINOv2 anomaly map.

This module translates a spatial DINOv2 anomaly heat-map into typed candidate
regions.  Each candidate is then optionally refined by SAM2.

Pipeline
--------
DINOv2 patch anomaly map
        ↓
  Thresholding (CONFIG.anomaly_percentile)
        ↓
  Morphological close + open (noise removal)
        ↓
  Connected components
        ↓
  Area filtering (too small / too large discarded)
        ↓
  Candidate bounding box
        ↓
  Heuristic confidence score
        ↓
  SAM2 refinement (optional, if ``sam2`` is provided)
        ↓
  CandidateRegion

Important distinction
---------------------
``anomaly candidate`` ≠ ``confirmed pothole``.

High DINOv2 anomaly scores can arise from road markings, shadows, repaired
asphalt, stains, debris, and lighting changes — not only potholes.  The
``pothole_confidence`` score is a *heuristic* that tries to down-weight these
false positives using shape, contrast, and area features.  It is NOT a trained
pothole classifier.

Confidence formula weights (heuristic — not from training data)
---------------------------------------------------------------
  0.40 × anomaly_score    — DINOv2-measured departure from healthy appearance
  0.20 × shape_score      — circularity (potholes tend to be blob-shaped)
  0.20 × darkness_score   — potholes appear darker than surrounding road
  0.20 × area_score       — log-scaled area (neither too tiny nor huge)

Adjust weights once labelled validation data is available.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from common.schemas import CandidateRegion, SegmentationResult
from config import CONFIG

log = logging.getLogger(__name__)


class PotholeLocalizer:
    """Localise pothole candidates from a DINOv2 anomaly map.

    Parameters
    ----------
    confidence_threshold:
        Minimum heuristic confidence for a candidate to be included in the
        output.  Default: ``CONFIG.pothole_confidence_threshold``.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIG.pothole_confidence_threshold,
    ) -> None:
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Heuristic scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_score(mask: np.ndarray) -> tuple[float, float]:
        """Compute circularity score in [0, 1] and aspect ratio."""
        area = float(mask.sum())
        if area <= 0:
            return 0.0, 1.0
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0, 1.0
        c = max(contours, key=cv2.contourArea)
        ca = max(1.0, cv2.contourArea(c))
        perimeter = max(1.0, cv2.arcLength(c, True))
        circularity = 4 * np.pi * ca / (perimeter * perimeter)
        circularity = float(np.clip(circularity, 0, 1))

        # Bounding rect aspect ratio
        _, _, bw, bh = cv2.boundingRect(c)
        aspect_ratio = float(max(bw, bh) / max(1, min(bw, bh)))
        return circularity, aspect_ratio

    def _classify_defect(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        shape_circ: float,
        aspect_ratio: float,
        is_water: bool,
    ) -> str:
        """Classify candidate into specific defect category."""
        if is_water:
            return "water_filled_pothole"

        # Cracks tend to have elongated aspect ratios (> 3.0) or very low circularity (< 0.25)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        region = gray[mask]
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F)[mask].var()) if len(region) > 0 else 0.0

        if aspect_ratio > 3.0 or shape_circ < 0.25:
            return "crack_or_damage"

        # Potholes are blob-like with moderate to high circularity
        if shape_circ >= 0.28:
            return "pothole"

        return "unknown_road_anomaly"

    def _is_benign_shadow_or_patch(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        road_mask: np.ndarray,
    ) -> bool:
        """Filter out benign uniform shadows and smooth asphalt patches."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        region = gray[mask]
        if len(region) < 30:
            return False

        # Internal texture variance
        var_internal = float(np.var(region))
        if var_internal < 15.0 and float(np.mean(region)) < 40.0:
            # Very uniform deep shadow with zero internal gradient
            return True

        return False

    def _candidate_confidence(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        anomaly_mean: float,
        anomaly_max: float,
    ) -> tuple[float, float, float, float]:
        """Heuristic likelihood score and shape features."""
        area_frac = float(mask.mean())
        shape, aspect_ratio = self._shape_score(mask)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        inside = gray[mask]
        kernel = np.ones((21, 21), np.uint8)
        outside_mask = (
            cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            & ~mask
        )
        outside = gray[outside_mask]

        darkness = 0.5
        surrounding_damage = 0.0
        if len(inside) > 0 and len(outside) > 0:
            delta = (float(outside.mean()) - float(inside.mean())) / 80.0
            darkness = float(np.clip(0.5 + delta, 0, 1))

            # Variance in outside ring indicates surrounding fatigue/cracks
            outside_var = float(np.var(outside))
            surrounding_damage = float(np.clip(outside_var / 800.0, 0, 1))

        area_score = float(np.clip(np.log1p(area_frac * 1000) / 5.0, 0, 1))
        anomaly_score = float(
            np.clip((anomaly_mean + anomaly_max) / 2.0 * 4.0, 0, 1)
        )

        conf = float(
            np.clip(
                0.40 * anomaly_score
                + 0.20 * shape
                + 0.20 * darkness
                + 0.20 * area_score,
                0,
                1,
            )
        )
        return conf, shape, aspect_ratio, surrounding_damage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(
        self,
        rgb: np.ndarray,
        anomaly_map: np.ndarray,
        road_mask: np.ndarray,
        threshold: float,
        sam2=None,
    ) -> list[CandidateRegion]:
        """Localise pothole candidates from the anomaly map."""
        candidate_bin = (anomaly_map >= threshold) & road_mask

        kernel_close = np.ones((7, 7), np.uint8)
        kernel_open = np.ones((3, 3), np.uint8)
        candidate_u8 = cv2.morphologyEx(
            candidate_bin.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close
        )
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel_open)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_u8, connectivity=8
        )

        out: list[CandidateRegion] = []
        for label in range(1, n):  # 0 = background
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h_cc = int(stats[label, cv2.CC_STAT_HEIGHT])

            if area < CONFIG.candidate_min_area_px:
                continue
            if area / candidate_u8.size > CONFIG.candidate_max_area_fraction:
                continue

            comp_mask = (labels == label)
            
            # False positive shadow suppression
            if self._is_benign_shadow_or_patch(rgb, comp_mask, road_mask):
                continue

            pixels = anomaly_map[comp_mask]
            conf, shape_circ, aspect_ratio, surrounding_damage = self._candidate_confidence(
                rgb, comp_mask, float(pixels.mean()), float(pixels.max())
            )

            if conf < self.confidence_threshold:
                continue

            # Classify defect type
            # Quick check for water pooling heuristic
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            val_mean = float(hsv[comp_mask, 2].mean()) if len(hsv[comp_mask]) > 0 else 100.0
            is_water = val_mean < 45.0 and conf > 0.65

            defect_type = self._classify_defect(
                rgb, comp_mask, shape_circ, aspect_ratio, is_water
            )

            refined_mask = comp_mask
            sam2_result: Optional[SegmentationResult] = None

            if sam2 is not None:
                try:
                    result = sam2.refine_box(rgb, [x, y, x + w, y + h_cc])
                    refined_road = result.mask & road_mask
                    if int(refined_road.sum()) >= CONFIG.candidate_min_area_px:
                        sam2_result = SegmentationResult(
                            mask=refined_road,
                            confidence=result.confidence,
                            bbox_xyxy=result.bbox_xyxy,
                            area_px=int(refined_road.sum()),
                        )
                        refined_mask = refined_road
                    else:
                        log.warning(
                            "SAM2 refinement for box [%d,%d,%d,%d] produced a "
                            "mask too small after road intersection (%d px); "
                            "keeping anomaly-map mask.",
                            x, y, x + w, y + h_cc, int(refined_road.sum()),
                        )
                except Exception as exc:
                    log.warning(
                        "SAM2 refinement failed for box [%d,%d,%d,%d]: %s — "
                        "falling back to anomaly-map mask.",
                        x, y, x + w, y + h_cc, exc,
                    )

            out.append(
                CandidateRegion(
                    mask=refined_mask,
                    bbox_xyxy=[x, y, x + w, y + h_cc],
                    anomaly_score=float(np.mean(pixels)),
                    pothole_confidence=conf,
                    sam2_result=sam2_result,
                    defect_type=defect_type,
                    shape_circularity=shape_circ,
                    aspect_ratio=aspect_ratio,
                    surrounding_damage=surrounding_damage,
                )
            )

        out.sort(key=lambda c: c.pothole_confidence, reverse=True)
        return out

