"""Pothole candidate localisation from a DINOv2 anomaly map.

This module translates a spatial DINOv2 anomaly heat-map into typed candidate
regions.  Each candidate is then optionally refined by SAM2.

Pipeline
--------
DINOv2 patch anomaly map
        ↓
  Active threshold supplied by AnomalyDetector.select_active_threshold
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
from inference.spatial_filtering import SpatialContextFilter
from inference.hierarchical_classifier import HierarchicalClassifier
from calibration.calibrator import ProbabilityCalibrator

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
        self.spatial_filter = SpatialContextFilter()
        self.hierarchical_classifier = HierarchicalClassifier()
        self.calibrator = ProbabilityCalibrator()

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

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> list[int]:
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return [0, 0, 0, 0]
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        union = int(np.logical_or(a, b).sum())
        return float(np.logical_and(a, b).sum() / union) if union else 0.0

    @staticmethod
    def _centroid(mask: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(mask)
        return (float(xs.mean()), float(ys.mean())) if xs.size else (float("inf"), float("inf"))

    def _consolidate_candidates(self, candidates: list[CandidateRegion]) -> list[CandidateRegion]:
        """Union only overlapping or immediately adjacent refined masks.

        Adjacent masks are merged only when their centres are close *and* their
        bounding boxes touch/overlap after a small dilation.  This prevents the
        four SAM2 fragments of one physical depression becoming four records
        without indiscriminately joining separate road defects.
        """
        groups: list[list[CandidateRegion]] = []
        for candidate in candidates:
            placed = False
            for group in groups:
                group_mask = np.logical_or.reduce([item.mask for item in group])
                iou = self._mask_iou(candidate.mask, group_mask)
                touching = np.any(cv2.dilate(group_mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & candidate.mask)
                dilated = cv2.dilate(group_mask.astype(np.uint8), np.ones((49, 49), np.uint8)).astype(bool)
                cx, cy = self._centroid(candidate.mask)
                gx, gy = self._centroid(group_mask)
                nearby = np.any(dilated & candidate.mask) and np.hypot(cx - gx, cy - gy) <= CONFIG.road_mask_merge_centroid_px
                if iou >= CONFIG.road_mask_merge_iou or touching or nearby:
                    group.append(candidate)
                    placed = True
                    break
            if not placed:
                groups.append([candidate])

        consolidated: list[CandidateRegion] = []
        for group in groups:
            merged = np.logical_or.reduce([item.mask for item in group])
            representative = max(group, key=lambda item: item.pothole_confidence)
            shape, aspect = self._shape_score(merged)
            consolidated.append(CandidateRegion(
                mask=merged,
                bbox_xyxy=self._bbox_from_mask(merged),
                anomaly_score=max(item.anomaly_score for item in group),
                pothole_confidence=max(item.pothole_confidence for item in group),
                sam2_result=SegmentationResult(
                    mask=merged,
                    confidence=max((item.sam2_result.confidence if item.sam2_result else 0.0) for item in group),
                    bbox_xyxy=self._bbox_from_mask(merged),
                    area_px=int(merged.sum()),
                ),
                # No trained crack/pothole discriminator is validated here.
                defect_type="road_defect",
                shape_circularity=shape,
                aspect_ratio=aspect,
                surrounding_damage=max(item.surrounding_damage for item in group),
                hierarchical_classification=representative.hierarchical_classification,
                filter_result={"consolidated_fragments": len(group)},
            ))
        return consolidated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(
        self,
        rgb: np.ndarray,
        anomaly_map: np.ndarray,
        road_mask: Optional[np.ndarray],
        threshold: float,
        sam2=None,
        test_mode_2d: Optional[bool] = None,
        min_area_px: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        diagnostics: Optional[dict] = None,
    ) -> list[CandidateRegion]:
        """Localise pothole candidates from the anomaly map.
        
        Parameters
        ----------
        test_mode_2d:
            If True, bypasses 3D aerial road constraints, lowers confidence threshold
            to 0.15 (or specified), and filters strictly by pixel area (e.g. >= 50 px).
        """
        is_2d = test_mode_2d if test_mode_2d is not None else getattr(CONFIG, "test_mode_2d", True)
        effective_min_area = (
            min_area_px if min_area_px is not None
            else (getattr(CONFIG, "candidate_min_area_px_2d", 50) if is_2d else getattr(CONFIG, "candidate_min_area_px_3d", 100))
        )
        effective_conf_thresh = (
            confidence_threshold if confidence_threshold is not None
            else (getattr(CONFIG, "pothole_confidence_threshold_2d", 0.15) if is_2d else self.confidence_threshold)
        )

        if is_2d:
            # 2D Test Mode: evaluate road surface without rigid aerial bounds
            candidate_bin = anomaly_map >= threshold
            if road_mask is not None and np.any(road_mask):
                candidate_bin = candidate_bin & road_mask
        else:
            # 3D Simulation Mode: strictly intersect with aerial nadir road mask
            effective_road_mask = road_mask if road_mask is not None else np.ones(anomaly_map.shape, dtype=bool)
            candidate_bin = (anomaly_map >= threshold) & effective_road_mask

        kernel_close = np.ones((7, 7), np.uint8)
        kernel_open = np.ones((3, 3), np.uint8)
        candidate_u8 = cv2.morphologyEx(
            candidate_bin.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close
        )
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel_open)

        # Day 5: Pre-consolidate nearby candidate fragments of the same physical defect
        # (e.g. adjacent pothole fragments or crack segments) before SAM2 prompting.
        # This provides SAM2 with the unified candidate bounding box and prevents over-fragmentation.
        dilated_u8 = cv2.dilate(candidate_u8, np.ones((21, 21), np.uint8))
        n_grp, labels_grp, stats_grp, _ = cv2.connectedComponentsWithStats(
            dilated_u8, connectivity=8
        )

        if diagnostics is not None:
            diagnostics["threshold_mask"] = candidate_bin.copy()
            diagnostics["candidate_mask"] = candidate_u8.astype(bool)
            diagnostics["connected_component_count_before_filters"] = int(n_grp - 1)
            diagnostics["sam2_prompts"] = []

        out: list[CandidateRegion] = []
        for grp_label in range(1, n_grp):
            comp_mask = (labels_grp == grp_label) & (candidate_u8 > 0)
            area = int(comp_mask.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(comp_mask)
            x = int(xs.min())
            y = int(ys.min())
            x_max = int(xs.max()) + 1
            y_max = int(ys.max()) + 1
            w = x_max - x
            h_cc = y_max - y
            raw_bbox = [x, y, x_max, y_max]

            pixels = anomaly_map[comp_mask]
            raw_anomaly_mean = float(pixels.mean()) if len(pixels) > 0 else 0.0
            raw_anomaly_max = float(pixels.max()) if len(pixels) > 0 else 0.0

            conf, shape_circ, aspect_ratio, surrounding_damage = self._candidate_confidence(
                rgb, comp_mask, raw_anomaly_mean, raw_anomaly_max
            )

            log.debug(
                "[DEBUG DINOv2 RAW] Candidate Box: [%d, %d, %d, %d] | Confidence: %.4f | Raw Area: %d px",
                x, y, x + w, y + h_cc, conf, area
            )

            # 1. Road Mask Isolation (Reject off-road checkerboard grids, verges, and background)
            if road_mask is not None:
                cy, cx = int(y + h_cc / 2), int(x + w / 2)
                if 0 <= cy < road_mask.shape[0] and 0 <= cx < road_mask.shape[1]:
                    if not road_mask[cy, cx]:
                        continue
                overlap = np.sum(comp_mask & road_mask)
                if (overlap / max(1, area)) < 0.60:
                    continue

            # 2. Area filtering
            if area < effective_min_area:
                continue
            if (area / candidate_u8.size > getattr(CONFIG, "candidate_max_area_fraction", 0.35)):
                continue

            # 3. Shape & Artifact Filtering (Suppress simulation grid lines without suppressing organic cracks)
            if not is_2d and aspect_ratio > 8.0 and shape_circ < 0.05:
                # Linear artifact/grid line rather than organic road distress
                continue

            # 3b. Perspective Horizon & Boundary Rejection
            # Perspective horizon artifact only exists in forward camera mode, never in nadir/top-down views.
            img_h, img_w = rgb.shape[:2]
            is_forward = (getattr(CONFIG, "camera_mode", "nadir") == "forward") and not is_2d
            if is_forward and y < 0.10 * img_h and (y + h_cc) < 0.16 * img_h:
                # Distant perspective vanishing horizon artifact
                continue
            if not is_2d:
                margin_x = max(25, int(0.03 * img_w))
                margin_y = max(25, int(0.035 * img_h))
                if x <= margin_x or (x + w) >= img_w - margin_x or y <= margin_y or (y + h_cc) >= img_h - margin_y:
                    # Outer camera frame truncation / roadside verge transition
                    continue

            # 3c. Painted Road Marking Rejection (Yellow stripes and white lane lines)
            cand_pixels = rgb[comp_mask]
            if len(cand_pixels) > 0:
                hsv_c = cv2.cvtColor(cand_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                yellow_ratio = np.mean(
                    (hsv_c[:, 0] >= 12) & (hsv_c[:, 0] <= 38)
                    & (hsv_c[:, 1] >= 50) & (hsv_c[:, 2] >= 60)
                )
                white_ratio = np.mean(
                    (hsv_c[:, 1] <= 35) & (hsv_c[:, 2] >= 130)
                )
                if yellow_ratio > 0.30:
                    # Painted yellow centerline/curb marking
                    continue
                if white_ratio > 0.35 and aspect_ratio > 2.5:
                    # Painted white dashed/solid lane marking
                    continue

            # 4. Confidence thresholding
            if conf < effective_conf_thresh:
                continue

            # 5. Physical Water Puddle Detection (Texture smoothness + relative road absorption)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            in_pixels = gray[comp_mask]
            in_std = float(np.std(in_pixels)) if len(in_pixels) > 0 else 50.0

            kernel_dil = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(comp_mask.astype(np.uint8), kernel_dil, iterations=1).astype(bool)
            collar = dilated & ~comp_mask
            out_pixels = gray[collar] if np.any(collar) else in_pixels

            out_mean = float(np.mean(out_pixels)) if len(out_pixels) > 0 else float(np.mean(in_pixels))
            in_mean = float(np.mean(in_pixels)) if len(in_pixels) > 0 else out_mean

            # Liquid water is smoother than asphalt (low variance) and darker than surrounding dry pavement
            # Must NOT be yellow paint or bright white markings
            is_not_marking = True
            if len(cand_pixels) > 0:
                hsv_c = cv2.cvtColor(cand_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                if np.mean((hsv_c[:, 0] >= 12) & (hsv_c[:, 0] <= 38) & (hsv_c[:, 1] >= 45)) > 0.20:
                    is_not_marking = False

            is_water = (in_std < 26.0 and (out_mean - in_mean) > 5.0 and conf >= 0.40 and is_not_marking)

            # The current DINOv2/SAM2 geometry path has no validated semantic
            # classifier.  Preserve the neutral label rather than asserting a
            # crack or pothole type from shape heuristics.
            defect_type = "road_defect"

            refined_mask = comp_mask
            sam2_result: Optional[SegmentationResult] = None

            if sam2 is not None:
                # Day 5 adaptive prompt context:
                # Longitudinal cracks and elongated defects extend along their major axis.
                # Expand context along the elongation axis so SAM2 captures the full defect extent.
                if h_cc >= 1.3 * w and w <= 25:
                    pad_y = max(32, min(80, int(h_cc * 1.5)))
                    pad_x = 10
                elif w >= 1.3 * h_cc and h_cc <= 25:
                    pad_x = max(32, min(80, int(w * 1.5)))
                    pad_y = 10
                else:
                    pad_x = 12 if w < 40 else 6
                    pad_y = 12 if h_cc < 40 else 6

                prompt_box = [
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(img_w, x + w + pad_x),
                    min(img_h, y + h_cc + pad_y),
                ]
                prompt_diagnostic = {
                    "bbox_xyxy": raw_bbox,
                    "prompt_box_xyxy": prompt_box,
                    "sam2_status": "pending",
                }
                if diagnostics is not None:
                    diagnostics["sam2_prompts"].append(prompt_diagnostic)
                try:
                    result = sam2.refine_box(rgb, prompt_box)
                    prompt_diagnostic["sam2_status"] = "refined"
                    prompt_diagnostic["sam2_confidence"] = float(result.confidence)
                    prompt_diagnostic["sam2_bbox_xyxy"] = result.bbox_xyxy
                    prompt_diagnostic["sam2_area_px"] = int(result.area_px)
                    # Always keep a SAM2 region on the stabilized road mask,
                    # including 2D test mode.  This avoids a prompt seeded on
                    # road leaking into vegetation or buildings.
                    refined_road = result.mask & road_mask if road_mask is not None else result.mask
                    if int(refined_road.sum()) >= effective_min_area:
                        refined_mask = refined_road
                        sam2_result = SegmentationResult(
                            mask=refined_road,
                            confidence=result.confidence,
                            bbox_xyxy=self._bbox_from_mask(refined_road),
                            area_px=int(refined_road.sum()),
                        )
                    else:
                        prompt_diagnostic["sam2_status"] = "rejected_off_road_or_too_small"
                except Exception as exc:
                    prompt_diagnostic["sam2_status"] = "fallback_to_candidate_mask"
                    prompt_diagnostic["error"] = str(exc)
                    log.warning(
                        "SAM2 refinement failed for box [%d,%d,%d,%d]: %s — "
                        "falling back to anomaly-map mask.",
                        x, y, x + w, y + h_cc, exc,
                    )

            # Spatial consistency & neighborhood contrast evaluation
            filter_res = self.spatial_filter.evaluate_candidate(
                rgb, refined_mask, raw_bbox, anomaly_map, aspect_ratio, shape_circ
            )
            cal_conf = float(self.calibrator.predict_probability(conf))
            hier_res = self.hierarchical_classifier.classify(
                rgb,
                refined_mask,
                raw_anomaly_score=raw_anomaly_mean,
                spatial_consistency=filter_res.spatial_score,
                filter_passed=filter_res.passed,
                rejection_reason=filter_res.rejection_reason,
                calibrated_confidence=cal_conf,
            )
            # Refine defect type based on hierarchical classification
            # Keep the neutral Day 3 label until a typed classifier has been
            # tested against labelled crack and pothole examples.
            defect_type = "road_defect"

            out.append(
                CandidateRegion(
                    mask=refined_mask,
                    bbox_xyxy=raw_bbox,
                    anomaly_score=raw_anomaly_mean,
                    pothole_confidence=hier_res.calibrated_confidence,
                    sam2_result=sam2_result,
                    defect_type=defect_type,
                    shape_circularity=shape_circ,
                    aspect_ratio=aspect_ratio,
                    surrounding_damage=surrounding_damage,
                    hierarchical_classification=hier_res.to_dict(),
                    filter_result=filter_res.__dict__,
                )
            )

        out = self._consolidate_candidates(out)
        out.sort(key=lambda c: c.pothole_confidence, reverse=True)
        if diagnostics is not None:
            diagnostics["accepted_candidate_count"] = len(out)
            diagnostics["accepted_candidate_boxes"] = [c.bbox_xyxy for c in out]
            diagnostics["sam2_refined_count"] = sum(c.sam2_result is not None for c in out)
        return out
