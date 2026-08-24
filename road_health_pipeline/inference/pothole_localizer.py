from __future__ import annotations

import cv2
import numpy as np

from config import CONFIG


class PotholeLocalizer:
    def __init__(self, confidence_threshold: float = CONFIG.pothole_confidence_threshold):
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _shape_score(mask: np.ndarray) -> float:
        area = float(mask.sum())
        if area <= 0:
            return 0.0
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        ca = max(1.0, cv2.contourArea(c))
        perimeter = max(1.0, cv2.arcLength(c, True))
        circularity = 4 * np.pi * ca / (perimeter * perimeter)
        return float(np.clip(circularity, 0, 1))

    def _candidate_confidence(self, rgb: np.ndarray, mask: np.ndarray, anomaly_mean: float, anomaly_max: float) -> float:
        h, w = mask.shape
        area_frac = float(mask.mean())
        shape = self._shape_score(mask)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        inside = gray[mask]
        outside_mask = cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1).astype(bool) & ~mask
        outside = gray[outside_mask]
        darkness = 0.5
        if len(inside) and len(outside):
            delta = (float(outside.mean()) - float(inside.mean())) / 80.0
            darkness = float(np.clip(0.5 + delta, 0, 1))
        area_score = float(np.clip(np.log1p(area_frac * 1000) / 5.0, 0, 1))
        anomaly_score = float(np.clip((anomaly_mean + anomaly_max) / 2.0 * 4.0, 0, 1))
        # This is a heuristic, not a trained pothole classifier.
        return float(np.clip(0.40 * anomaly_score + 0.20 * shape + 0.20 * darkness + 0.20 * area_score, 0, 1))

    def localize(self, rgb: np.ndarray, anomaly_map: np.ndarray, road_mask: np.ndarray, threshold: float,
                 sam2=None) -> list[dict]:
        candidate = (anomaly_map >= threshold) & road_mask
        kernel = np.ones((7, 7), np.uint8)
        candidate_u8 = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
        out = []
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < CONFIG.candidate_min_area_px:
                continue
            if area / candidate_u8.size > CONFIG.candidate_max_area_fraction:
                continue
            mask = labels == label
            pixels = anomaly_map[mask]
            conf = self._candidate_confidence(rgb, mask, float(pixels.mean()), float(pixels.max()))
            refined = mask
            if sam2 is not None:
                try:
                    refined = sam2.refine_box(rgb, [x, y, x + w, y + h]) & road_mask
                    if refined.sum() < CONFIG.candidate_min_area_px:
                        refined = mask
                except Exception:
                    refined = mask
            out.append({
                "mask": refined,
                "bbox_xyxy": [x, y, x + w, y + h],
                "anomaly_score": float(np.mean(pixels)),
                "pothole_confidence": conf,
            })
        return out
