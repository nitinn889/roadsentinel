from __future__ import annotations

import math
from typing import Optional, Tuple
import cv2
import numpy as np

from common.schemas import DefectType


class DefectClassifier:
    """Classifies segmented road defect masks into distinct defect categories.

    Distinguishes:
    - pothole: compact, bowl-like road surface depression
    - water_filled_pothole: pothole exhibiting specular reflection / low internal variance / dark puddle signature
    - crack: thin, elongated, high-perimeter fracture or network
    - surface_wear: diffuse, low-contrast surface abrasion/deterioration
    - road_anomaly: unclassified road surface irregularity
    """

    def __init__(self,
                 water_variance_thresh: float = 220.0,
                 water_darkness_thresh: float = 75.0,
                 crack_aspect_ratio_thresh: float = 3.0,
                 crack_solidity_thresh: float = 0.55):
        self.water_variance_thresh = water_variance_thresh
        self.water_darkness_thresh = water_darkness_thresh
        self.crack_aspect_ratio_thresh = crack_aspect_ratio_thresh
        self.crack_solidity_thresh = crack_solidity_thresh

    def compute_morphological_properties(self, mask: np.ndarray) -> dict[str, float]:
        """Calculates solidity, aspect ratio, compactness, and extent for a binary mask."""
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "area_px": 0.0,
                "aspect_ratio": 1.0,
                "solidity": 1.0,
                "compactness": 1.0,
                "perimeter": 0.0,
            }

        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))

        # Convex hull and solidity
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(1.0, hull_area)

        # Bounding rect aspect ratio
        _, _, w, h = cv2.boundingRect(c)
        aspect_ratio = max(w, h) / max(1.0, min(w, h))

        # Compactness (isoperimetric quotient: 4 * pi * area / perimeter^2)
        compactness = (4.0 * math.pi * area) / max(1.0, perimeter ** 2)

        return {
            "area_px": area,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "compactness": compactness,
            "perimeter": perimeter,
        }

    def detect_water(self, rgb_image: Optional[np.ndarray], mask: np.ndarray) -> Tuple[bool, float]:
        """Detects whether a masked defect region contains standing water.

        Returns (is_water_filled, water_confidence).
        Water cues: low internal texture variance, specular glint / polarized highlights, or deep absorption.
        """
        if rgb_image is None or mask is None or not np.any(mask):
            return False, 0.0

        # Extract masked pixels
        h, w = mask.shape[:2]
        if rgb_image.shape[:2] != (h, w):
            resized_rgb = cv2.resize(rgb_image, (w, h))
        else:
            resized_rgb = rgb_image

        pixels = resized_rgb[mask]
        if len(pixels) < 10:
            return False, 0.0

        # Convert to HSV & Grayscale
        gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
        masked_gray = gray[mask]

        mean_val = float(np.mean(masked_gray))
        std_val = float(np.std(masked_gray))

        # Water in asphalt puddles typically has low spatial variance and either very dark appearance
        # or specular highlights with high local contrast gradient
        is_dark_puddle = mean_val < self.water_darkness_thresh and std_val < 35.0
        is_smooth_water = std_val < 18.0

        # Check for specular reflections (clipping in R, G, B channels simultaneously)
        specular_ratio = float(np.mean((pixels[:, 0] > 240) & (pixels[:, 1] > 240) & (pixels[:, 2] > 240)))
        has_specular = specular_ratio > 0.03

        water_score = 0.0
        if is_dark_puddle:
            water_score += 0.45
        if is_smooth_water:
            water_score += 0.35
        if has_specular:
            water_score += 0.40

        water_confidence = min(1.0, water_score)
        is_water = water_confidence >= 0.40
        return is_water, water_confidence

    def classify(self,
                 mask: np.ndarray,
                 rgb_image: Optional[np.ndarray] = None,
                 anomaly_score: float = 0.5,
                 confidence: float = 0.5) -> Tuple[str, bool, float, dict[str, float]]:
        """Classifies the defect mask.

        Returns:
            (defect_type, is_water_filled, water_confidence, morph_props)
        """
        morph = self.compute_morphological_properties(mask)
        is_water, water_conf = self.detect_water(rgb_image, mask)

        # Classification decision rules
        aspect_ratio = morph["aspect_ratio"]
        solidity = morph["solidity"]
        compactness = morph["compactness"]

        if is_water and water_conf > 0.50:
            defect_type = DefectType.WATER_FILLED_POTHOLE.value
        elif aspect_ratio >= self.crack_aspect_ratio_thresh or (solidity < self.crack_solidity_thresh and compactness < 0.25):
            defect_type = DefectType.CRACK.value
        elif compactness >= 0.20 and solidity >= 0.60:
            if is_water:
                defect_type = DefectType.WATER_FILLED_POTHOLE.value
            else:
                defect_type = DefectType.POTHOLE.value
        elif anomaly_score < 0.40:
            defect_type = DefectType.SURFACE_WEAR.value
        else:
            defect_type = DefectType.ROAD_ANOMALY.value

        return defect_type, is_water, water_conf, morph
