"""Hierarchical multi-stage defect taxonomy and classification.

Enforces a principled progression:
  Visual Anomaly
       ↓
  Road-Surface Defect
       ↓
  Candidate Specialization (Pothole Candidate, Crack, Water Hazard, Surface Wear)
       ↓
  Confirmed Pothole / Defect

This prevents treating every visual anomaly or shadow as a pothole.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    STAGE_0_ANOMALY = "visual_anomaly"
    STAGE_1_DEFECT = "road_surface_defect"
    STAGE_2_CANDIDATE = "candidate_defect"
    STAGE_3_CONFIRMED = "confirmed_defect"


class HierarchicalDefectType(str, Enum):
    CONFIRMED_POTHOLE = "confirmed_pothole"
    POTHOLE_CANDIDATE = "pothole_candidate"
    WATER_FILLED_POTHOLE = "water_filled_pothole"
    CRACK_OR_FATIGUE = "crack_or_fatigue"
    SURFACE_WEAR = "surface_wear_or_patch"
    BENIGN_ANOMALY = "benign_road_anomaly"
    UNKNOWN = "unknown_road_anomaly"


@dataclass
class HierarchicalClassification:
    """Full trace of hierarchical categorization through the multi-stage funnel."""
    stage: PipelineStage
    final_type: HierarchicalDefectType
    is_confirmed_pothole: bool
    calibrated_confidence: float
    raw_anomaly_score: float
    spatial_consistency: float
    circularity: float
    aspect_ratio: float
    is_water_hazard: bool
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "final_type": self.final_type.value,
            "is_confirmed_pothole": self.is_confirmed_pothole,
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "raw_anomaly_score": round(self.raw_anomaly_score, 4),
            "spatial_consistency": round(self.spatial_consistency, 4),
            "circularity": round(self.circularity, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "is_water_hazard": self.is_water_hazard,
            "reasoning": self.reasoning,
        }


class HierarchicalClassifier:
    """Classifies candidate anomalies across the multi-stage decision tree."""

    def __init__(
        self,
        pothole_confirm_threshold: float = 0.50,
        crack_aspect_ratio_threshold: float = 2.8,
        min_pothole_circularity: float = 0.22,
    ) -> None:
        self.pothole_confirm_threshold = pothole_confirm_threshold
        self.crack_aspect_ratio_threshold = crack_aspect_ratio_threshold
        self.min_pothole_circularity = min_pothole_circularity

    def classify(
        self,
        rgb_image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        raw_anomaly_score: float = 0.5,
        spatial_consistency: float = 0.5,
        filter_passed: bool = True,
        rejection_reason: Optional[str] = None,
        calibrated_confidence: float = 0.5,
        rgb: Optional[np.ndarray] = None,
    ) -> HierarchicalClassification:
        """Evaluate a candidate region through the multi-stage taxonomy."""
        img = rgb_image if rgb_image is not None else rgb
        if img is None:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
        if mask is None:
            mask = np.zeros((100, 100), dtype=bool)

        # Shape metrics
        circularity, aspect_ratio = self._calculate_shape_metrics(mask)

        # Stage 0 Check: Did it fail basic spatial/context filters?
        if not filter_passed:
            return HierarchicalClassification(
                stage=PipelineStage.STAGE_0_ANOMALY,
                final_type=HierarchicalDefectType.BENIGN_ANOMALY,
                is_confirmed_pothole=False,
                calibrated_confidence=min(calibrated_confidence, 0.2),
                raw_anomaly_score=raw_anomaly_score,
                spatial_consistency=spatial_consistency,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                is_water_hazard=False,
                reasoning=f"Filtered at Stage 0: {rejection_reason or 'benign visual anomaly'}",
            )

        # Stage 1: Verified Road Surface Defect
        # 1. Evaluate Crack vs Cavity geometry
        if aspect_ratio >= self.crack_aspect_ratio_threshold or circularity < self.min_pothole_circularity:
            return HierarchicalClassification(
                stage=PipelineStage.STAGE_2_CANDIDATE,
                final_type=HierarchicalDefectType.CRACK_OR_FATIGUE,
                is_confirmed_pothole=False,
                calibrated_confidence=calibrated_confidence,
                raw_anomaly_score=raw_anomaly_score,
                spatial_consistency=spatial_consistency,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                is_water_hazard=False,
                reasoning=f"Elongated linear defect (aspect_ratio={aspect_ratio:.2f}, circularity={circularity:.2f}) -> Crack/Fatigue",
            )

        # 2. Evaluate water content in cavity
        is_water, water_confidence = self._detect_water_sheen(img, mask)
        if is_water and circularity >= self.min_pothole_circularity:
            is_confirmed = calibrated_confidence >= self.pothole_confirm_threshold
            stage = PipelineStage.STAGE_3_CONFIRMED if is_confirmed else PipelineStage.STAGE_2_CANDIDATE
            return HierarchicalClassification(
                stage=stage,
                final_type=HierarchicalDefectType.WATER_FILLED_POTHOLE,
                is_confirmed_pothole=True,
                calibrated_confidence=calibrated_confidence,
                raw_anomaly_score=raw_anomaly_score,
                spatial_consistency=spatial_consistency,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                is_water_hazard=True,
                reasoning="Water-filled depression cavity with dark absorption and specular rim",
            )

        # 3. Evaluate Pothole depression cavity vs Surface wear
        darkness_delta = self._compute_depression_darkness(img, mask)

        if darkness_delta < -5.0 and raw_anomaly_score < 0.45:
            # Lighter than surrounding road with low anomaly score -> surface wear / aggregate loss
            return HierarchicalClassification(
                stage=PipelineStage.STAGE_1_DEFECT,
                final_type=HierarchicalDefectType.SURFACE_WEAR,
                is_confirmed_pothole=False,
                calibrated_confidence=calibrated_confidence * 0.5,
                raw_anomaly_score=raw_anomaly_score,
                spatial_consistency=spatial_consistency,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                is_water_hazard=False,
                reasoning="Planar surface variation with no cavity depression -> Surface Wear/Patch",
            )

        # It is a Pothole Candidate
        if calibrated_confidence >= self.pothole_confirm_threshold and darkness_delta >= 0.0:
            stage = PipelineStage.STAGE_3_CONFIRMED
            final_type = HierarchicalDefectType.CONFIRMED_POTHOLE
            is_confirmed = True
            reasoning = "Confirmed pothole cavity meeting confidence, shape, and depression criteria"
        else:
            stage = PipelineStage.STAGE_2_CANDIDATE
            final_type = HierarchicalDefectType.POTHOLE_CANDIDATE
            is_confirmed = False
            reasoning = f"Pothole candidate under evaluation (confidence={calibrated_confidence:.2f})"

        return HierarchicalClassification(
            stage=stage,
            final_type=final_type,
            is_confirmed_pothole=is_confirmed,
            calibrated_confidence=calibrated_confidence,
            raw_anomaly_score=raw_anomaly_score,
            spatial_consistency=spatial_consistency,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            is_water_hazard=False,
            reasoning=reasoning,
        )

    @staticmethod
    def _calculate_shape_metrics(mask: np.ndarray) -> Tuple[float, float]:
        """Compute circularity (4*pi*area / perim^2) and bounding box aspect ratio."""
        if not np.any(mask):
            return 0.0, 1.0

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0, 1.0

        c = max(contours, key=cv2.contourArea)
        area = max(1.0, cv2.contourArea(c))
        perimeter = max(1.0, cv2.arcLength(c, True))
        circularity = float(np.clip(4 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))

        _, _, w, h = cv2.boundingRect(c)
        aspect_ratio = float(max(w, h) / max(1, min(w, h)))
        return circularity, aspect_ratio

    @staticmethod
    def _compute_depression_darkness(rgb_image: np.ndarray, mask: np.ndarray) -> float:
        """Compute luminance drop between collar pavement and cavity interior."""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        inside = gray[mask]
        if len(inside) == 0:
            return 0.0

        kernel = np.ones((15, 15), np.uint8)
        collar_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
        outside = gray[collar_mask] if np.any(collar_mask) else inside

        return float(np.mean(outside) - np.mean(inside))

    @staticmethod
    def _detect_water_sheen(rgb_image: np.ndarray, mask: np.ndarray) -> Tuple[bool, float]:
        """Detect standing water puddles within the candidate mask."""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        inside = gray[mask]
        if len(inside) < 30:
            return False, 0.0

        in_std = float(np.std(inside))

        kernel = np.ones((15, 15), np.uint8)
        collar_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
        outside = gray[collar_mask] if np.any(collar_mask) else inside

        in_mean = float(np.mean(inside))
        out_mean = float(np.mean(outside))

        # Liquid water has smooth liquid texture with specular reflections or sky tone
        inside_rgb = rgb_image[mask]
        b_channel = inside_rgb[:, 2].astype(np.float32)
        r_channel = inside_rgb[:, 0].astype(np.float32)
        blue_sky_reflection = np.mean(b_channel > r_channel + 2.0) > 0.40
        has_specular_variance = 3.0 <= in_std < 22.0

        # White/yellow paint can be smooth and highly reflective as well.  This
        # guard is intentionally local to the candidate, complementing the
        # marking suppressor that runs earlier in the perception pipeline.
        hsv_inside = cv2.cvtColor(
            inside_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV
        ).reshape(-1, 3)
        yellow_ratio = float(np.mean(
            (hsv_inside[:, 0] >= 12) & (hsv_inside[:, 0] <= 38)
            & (hsv_inside[:, 1] >= 45) & (hsv_inside[:, 2] >= 60)
        ))
        white_ratio = float(np.mean(
            (hsv_inside[:, 1] <= 35) & (hsv_inside[:, 2] >= 150)
        ))
        is_marking = yellow_ratio > 0.25 or white_ratio > 0.55

        if (has_specular_variance or blue_sky_reflection) and (out_mean - in_mean) > 8.0 and not is_marking:
            conf = float(np.clip(1.0 - (in_std / 25.0), 0.5, 0.95))
            return True, conf

        return False, 0.0
