"""
road_marking_suppressor.py
--------------------------
RoadSentinel: Lane Marking, Guardrail & Pavement Boundary Filter.

Prevents false-positive defect detections on:
- Painted double yellow centerlines
- Painted white dashed and solid lane lines
- High-contrast roadside guardrails and concrete barriers
- Perspective camera horizon and border artifacts
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class RoadMarkingSuppressor:
    """Detects painted road markings, edge barriers, and suppresses them from defect maps."""

    def __init__(
        self,
        yellow_hue_range: Tuple[int, int] = (12, 38),
        yellow_sat_min: int = 55,
        yellow_val_min: int = 65,
        white_sat_max: int = 35,
        white_val_min: int = 130,
        dilation_kernel_size: int = 7,
    ) -> None:
        self.yellow_hue_range = yellow_hue_range
        self.yellow_sat_min = yellow_sat_min
        self.yellow_val_min = yellow_val_min
        self.white_sat_max = white_sat_max
        self.white_val_min = white_val_min
        self.dilation_kernel_size = dilation_kernel_size

    def get_marking_mask(self, rgb: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of shape (H, W) where True indicates
        painted road markings (yellow or white) and roadside barriers.
        """
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        # 1. Yellow painted lines (tight hue + saturation)
        y_low, y_high = self.yellow_hue_range
        raw_yellow = (
            (h_chan >= y_low)
            & (h_chan <= y_high)
            & (s_chan >= max(60, self.yellow_sat_min))
            & (v_chan >= max(70, self.yellow_val_min))
        )

        # 2. White painted lines (high brightness, low saturation)
        road_v_median = float(np.median(v_chan))
        white_thresh = max(135, int(road_v_median + 40))
        raw_white = (s_chan <= self.white_sat_max) & (v_chan >= white_thresh)

        # Filter: only keep genuine continuous painted stripes (elongated or substantial linear structure)
        # Never mask individual gravel aggregate pebbles!
        clean_mask = np.zeros((h, w), dtype=bool)

        # Process yellow stripes
        nb_y, labels_y, stats_y, _ = cv2.connectedComponentsWithStats(raw_yellow.astype(np.uint8), connectivity=8)
        for i in range(1, nb_y):
            area = stats_y[i, cv2.CC_STAT_AREA]
            bw = stats_y[i, cv2.CC_STAT_WIDTH]
            bh = stats_y[i, cv2.CC_STAT_HEIGHT]
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if area >= 120 and (aspect >= 2.2 or max(bw, bh) >= 35):
                clean_mask[labels_y == i] = True

        # Process white stripes
        nb_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(raw_white.astype(np.uint8), connectivity=8)
        for i in range(1, nb_w):
            area = stats_w[i, cv2.CC_STAT_AREA]
            bw = stats_w[i, cv2.CC_STAT_WIDTH]
            bh = stats_w[i, cv2.CC_STAT_HEIGHT]
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if area >= 60 and (aspect >= 2.0 or max(bw, bh) >= 30):
                clean_mask[labels_w == i] = True

        # Dilate minimally (3x3 kernel) to tightly envelope painted line edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(clean_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        return dilated

    def is_marking_candidate(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        bbox: List[int],
    ) -> bool:
        """
        Evaluates whether a candidate region is a painted line or boundary artifact.
        """
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Check 1: Horizon artifact (perspective horizon in upper 12% of frame)
        if y1 < 0.10 * h and y2 < 0.16 * h:
            return True

        # Check 2: Border touching (camera truncation at borders, guardrails, curbs, verge transition)
        margin_x = max(35, int(0.048 * w))
        margin_y = max(30, int(0.040 * h))
        if x1 <= margin_x or x2 >= w - margin_x or y1 <= margin_y or y2 >= h - margin_y:
            return True

        # Check 3: Check color profile of candidate
        cand_pixels = rgb[mask] if np.any(mask) else rgb[y1:y2, x1:x2].reshape(-1, 3)
        if len(cand_pixels) == 0:
            return True

        hsv_pixels = cv2.cvtColor(cand_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        h_vals = hsv_pixels[:, 0]
        s_vals = hsv_pixels[:, 1]
        v_vals = hsv_pixels[:, 2]

        # Relative road brightness baseline
        hsv_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        road_v_median = float(np.median(hsv_full[:, :, 2]))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = max(bw, bh) / max(1, min(bw, bh))

        # Yellow paint check: high proportion of yellow pixels and linear geometry
        yellow_ratio = np.mean(
            (h_vals >= self.yellow_hue_range[0])
            & (h_vals <= self.yellow_hue_range[1])
            & (s_vals >= self.yellow_sat_min)
            & (v_vals >= self.yellow_val_min)
        )
        if yellow_ratio > 0.40 and aspect > 2.0:
            return True

        # White paint check: must be significantly brighter than road median AND linear
        white_thresh = max(190, int(road_v_median + 35))
        white_ratio = np.mean(
            (s_vals <= self.white_sat_max)
            & (v_vals >= white_thresh)
        )
        if white_ratio > 0.35 and aspect > 2.2:
            return True

        # Check 4: Extremely elongated linear stripe (painted line segment)
        if aspect > 5.5 and (yellow_ratio > 0.20 or white_ratio > 0.20):
            return True

        return False

    def filter_candidate_regions(
        self,
        candidates: list,
        rgb: np.ndarray,
        marking_mask: Optional[np.ndarray] = None,
    ) -> list:
        """Filters out defect candidates that fall on road markings or boundaries."""
        if marking_mask is None:
            marking_mask = self.get_marking_mask(rgb)

        filtered = []
        for cand in candidates:
            mask = cand.sam2_result.mask if (hasattr(cand, "sam2_result") and cand.sam2_result is not None) else cand.mask
            bbox = cand.bbox_xyxy

            # Check marking candidate heuristic
            if self.is_marking_candidate(rgb, mask, bbox):
                continue

            # Check overlap with marking mask
            if mask is not None and np.any(mask):
                overlap = np.sum(mask & marking_mask) / max(1, np.sum(mask))
                if overlap > 0.40:
                    continue

            filtered.append(cand)

        return filtered
