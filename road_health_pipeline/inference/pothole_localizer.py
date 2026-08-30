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
    def _shape_score(mask: np.ndarray) -> float:
        """Circularity score in [0, 1].

        Potholes tend to be roughly circular / blob-shaped.  A perfect circle
        has circularity 1.0; highly elongated or irregular shapes score lower.
        """
        area = float(mask.sum())
        if area <= 0:
            return 0.0
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        ca = max(1.0, cv2.contourArea(c))
        perimeter = max(1.0, cv2.arcLength(c, True))
        circularity = 4 * np.pi * ca / (perimeter * perimeter)
        return float(np.clip(circularity, 0, 1))

    def _candidate_confidence(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        anomaly_mean: float,
        anomaly_max: float,
    ) -> float:
        """Heuristic pothole-likelihood score in [0, 1].

        See module docstring for weight rationale.  This is NOT a trained
        classifier; do not report it as model accuracy.
        """
        area_frac = float(mask.mean())
        shape = self._shape_score(mask)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        inside = gray[mask]
        # Dilate the mask to get a surrounding-road annulus
        kernel = np.ones((21, 21), np.uint8)
        outside_mask = (
            cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            & ~mask
        )
        outside = gray[outside_mask]

        darkness = 0.5  # neutral default when regions are empty
        if len(inside) > 0 and len(outside) > 0:
            # Positive when inside is darker than outside (pothole-like)
            delta = (float(outside.mean()) - float(inside.mean())) / 80.0
            darkness = float(np.clip(0.5 + delta, 0, 1))

        # Log-scaled area score: penalises both tiny noise and huge false-positive regions
        area_score = float(np.clip(np.log1p(area_frac * 1000) / 5.0, 0, 1))

        # Combined anomaly signal from the heat-map
        anomaly_score = float(
            np.clip((anomaly_mean + anomaly_max) / 2.0 * 4.0, 0, 1)
        )

        # Weighted heuristic (see module docstring for weight explanation)
        return float(
            np.clip(
                0.40 * anomaly_score
                + 0.20 * shape
                + 0.20 * darkness
                + 0.20 * area_score,
                0,
                1,
            )
        )

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
        """Localise pothole candidates from the anomaly map.

        Parameters
        ----------
        rgb:
            HxWx3 uint8 RGB image.
        anomaly_map:
            HxW float32 anomaly heat-map from ``AnomalyDetector``.
        road_mask:
            HxW boolean mask restricting the search to road pixels.
        threshold:
            Anomaly score threshold (patches above this are considered
            anomalous).  Derived from ``AnomalyDetector.summarize()``.
        sam2:
            Optional ``RoadMasker`` instance.  When provided, each candidate
            bounding box is submitted to ``sam2.refine_box()`` to produce a
            more precise segmentation mask.

        Returns
        -------
        list[CandidateRegion]
            Candidates ordered by ``pothole_confidence`` descending.
            Only candidates with ``confidence >= self.confidence_threshold``
            are included.
        """
        # Threshold the anomaly map and restrict to road pixels
        candidate_bin = (anomaly_map >= threshold) & road_mask

        # Morphological operations to smooth/connect nearby detections and
        # remove isolated noise specks
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
            pixels = anomaly_map[comp_mask]
            conf = self._candidate_confidence(
                rgb, comp_mask, float(pixels.mean()), float(pixels.max())
            )

            if conf < self.confidence_threshold:
                continue

            refined_mask = comp_mask
            sam2_result: Optional[SegmentationResult] = None

            if sam2 is not None:
                try:
                    result = sam2.refine_box(rgb, [x, y, x + w, y + h_cc])
                    refined_road = result.mask & road_mask
                    if int(refined_road.sum()) >= CONFIG.candidate_min_area_px:
                        # Accept the SAM2 refinement
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
                )
            )

        # Sort by confidence descending
        out.sort(key=lambda c: c.pothole_confidence, reverse=True)
        return out
