"""
vehicle_suppressor.py
---------------------
RoadSentinel: Vehicle & Obstacle Suppression Filter.

Eliminates false-positive pothole detections caused by vehicles (cars, trucks,
buses, motorcycles), vehicle tires, and vehicle cast-shadows.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VehicleSuppressor:
    """Detects vehicles using lightweight YOLOv8 and generates exclusion masks."""

    VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car=2, motorcycle=3, bus=5, truck=7

    def __init__(self, confidence_threshold: float = 0.25, shadow_padding_px: int = 15) -> None:
        self.conf = confidence_threshold
        self.padding = shadow_padding_px
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            log.info("VehicleSuppressor initialized with YOLOv8n (Vehicle suppression enabled).")
        except Exception as e:
            log.warning("Could not initialize YOLOv8 vehicle model: %s. Using heuristic vehicle filter.", e)
            self.model = None

    def get_vehicle_mask(self, rgb: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of shape (H, W) where True indicates
        vehicle hulls and their immediate contact shadows.
        """
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        if self.model is None:
            return mask

        try:
            results = self.model(rgb, verbose=False, conf=self.conf, classes=self.VEHICLE_CLASSES)
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1 = max(0, xyxy[0] - self.padding)
                    y1 = max(0, xyxy[1] - self.padding)
                    x2 = min(w, xyxy[2] + self.padding)
                    y2 = min(h, xyxy[3] + self.padding)
                    mask[y1:y2, x1:x2] = True

            # Morphological dilation to ensure complete coverage of contact shadows
            if np.any(mask):
                kernel = np.ones((self.padding, self.padding), np.uint8)
                mask_u8 = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
                mask = mask_u8.astype(bool)

        except Exception as e:
            log.warning("Vehicle suppression inference error: %s", e)

        return mask

    def suppress_vehicles(self, anomaly_map: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zeroes out anomaly scores where vehicles are present.
        Returns (filtered_anomaly_map, vehicle_mask).
        """
        vehicle_mask = self.get_vehicle_mask(rgb)
        filtered_map = anomaly_map.copy()
        if np.any(vehicle_mask):
            filtered_map[vehicle_mask] = 0.0
            log.info("  -> Suppressed %d px of vehicle regions from anomaly map.", int(vehicle_mask.sum()))
        return filtered_map, vehicle_mask

    def filter_candidate_regions(self, candidates: list, vehicle_mask: np.ndarray) -> list:
        """Removes candidate defect regions that have significant intersection with vehicles."""
        if not np.any(vehicle_mask) or not candidates:
            return candidates

        filtered = []
        for cand in candidates:
            mask = getattr(cand, "mask", None)
            if mask is not None:
                intersection = np.sum(mask & vehicle_mask)
                total = max(1, np.sum(mask))
                if (intersection / total) > 0.25:
                    log.info("  [Vehicle Suppressed] Dropping candidate overlapping with vehicle (%d px).", int(intersection))
                    continue
            else:
                x1, y1, x2, y2 = [int(v) for v in cand.bbox_xyxy]
                box_crop = vehicle_mask[y1:y2, x1:x2]
                if box_crop.size > 0 and (np.sum(box_crop) / box_crop.size) > 0.40:
                    log.info("  [Vehicle Suppressed] Dropping candidate box overlapping with vehicle.")
                    continue
            filtered.append(cand)
        return filtered
