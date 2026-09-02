from __future__ import annotations

from typing import List, Optional
import cv2
import numpy as np

from common.schemas import (
    DefectMeasurement,
    DefectType,
    PredictionResult,
    RoadHealthScore,
    RoadSegmentAggregate,
)


class PipelineVisualizer:
    """Renders high-quality diagnostic visual overlays for RoadSentinel."""

    DEFECT_COLORS = {
        DefectType.POTHOLE.value: (255, 140, 0),         # Amber / Orange
        DefectType.WATER_FILLED_POTHOLE.value: (0, 195, 255), # Cyan / Sky Blue
        DefectType.CRACK.value: (220, 20, 60),           # Crimson / Magenta
        DefectType.SURFACE_WEAR.value: (255, 215, 0),    # Gold / Yellow
        DefectType.ROAD_ANOMALY.value: (180, 180, 180),  # Silver / Gray
    }

    SEVERITY_COLORS = {
        "low": (50, 205, 50),       # Lime Green
        "medium": (255, 215, 0),    # Yellow
        "high": (255, 140, 0),      # Orange
        "critical": (220, 20, 60),   # Red / Crimson
    }

    @staticmethod
    def draw_detection_overlay(rgb_image: np.ndarray,
                               detections: List[DefectMeasurement],
                               alpha: float = 0.45) -> np.ndarray:
        """Renders defect masks, bounding boxes, and metadata labels on the RGB image."""
        img = rgb_image.copy()
        overlay = img.copy()

        for d in detections:
            color = PipelineVisualizer.DEFECT_COLORS.get(d.defect_type, (200, 200, 200))
            x1, y1, x2, y2 = [int(v) for v in d.bbox]

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label text
            area_str = f"{d.estimated_area_m2:.2f}m²" if d.estimated_area_m2 is not None else f"{d.mask_area_pixels}px"
            water_str = " [WATER]" if d.is_water_filled else ""
            label = f"{d.defect_type.upper()}{water_str} ({d.confidence:.0%}) | {area_str}"

            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(img, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Alpha blend
        return cv2.addWeighted(overlay, 1.0 - alpha, img, alpha, 0)

    @staticmethod
    def draw_severity_overlay(rgb_image: np.ndarray,
                              detections: List[DefectMeasurement]) -> np.ndarray:
        """Renders color-coded defect severity indicators."""
        img = rgb_image.copy()

        for d in detections:
            sev_level = d.severity.severity.lower()
            color = PipelineVisualizer.SEVERITY_COLORS.get(sev_level, (200, 200, 200))
            x1, y1, x2, y2 = [int(v) for v in d.bbox]

            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # Severity score tag
            tag = f"SEVERITY: {sev_level.upper()} ({d.severity.severity_score:.0f}/100)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, max(0, y1)), color, -1)
            text_color = (0, 0, 0) if sev_level in ("low", "medium") else (255, 255, 255)
            cv2.putText(img, tag, (x1 + 4, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

        # Legend HUD top right
        h, w = img.shape[:2]
        legend_x = max(10, w - 240)
        cv2.rectangle(img, (legend_x - 10, 10), (w - 10, 120), (30, 30, 30), -1)
        cv2.putText(img, "SEVERITY SCALE", (legend_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_offset = 50
        for level, color in PipelineVisualizer.SEVERITY_COLORS.items():
            cv2.rectangle(img, (legend_x, y_offset - 10), (legend_x + 15, y_offset + 5), color, -1)
            cv2.putText(img, level.capitalize(), (legend_x + 25, y_offset + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            y_offset += 16

        return img

    @staticmethod
    def draw_road_health_overlay(rgb_image: np.ndarray,
                                 road_health: RoadHealthScore,
                                 road_segment_id: str = "SEG_001",
                                 prediction: Optional[PredictionResult] = None) -> np.ndarray:
        """Renders comprehensive HUD dashboard banner over the road imagery."""
        img = rgb_image.copy()
        h, w = img.shape[:2]

        # Top banner HUD
        banner_h = 110
        banner_overlay = img[:banner_h, :].copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (20, 24, 30), -1)
        img[:banner_h, :] = cv2.addWeighted(banner_overlay, 0.85, img[:banner_h, :], 0.15, 0)

        # Health score color
        score = road_health.road_health_score
        if score >= 80:
            status_color = (50, 205, 50)  # Green
        elif score >= 60:
            status_color = (255, 215, 0)  # Yellow
        elif score >= 40:
            status_color = (255, 140, 0)  # Orange
        else:
            status_color = (220, 20, 60)  # Red

        # Segment ID and Title
        cv2.putText(img, f"ROADSENTINEL ANALYTICS | SEGMENT: {road_segment_id}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 240), 1, cv2.LINE_AA)

        # Health Gauge
        cv2.putText(img, f"HEALTH SCORE: {score:.1f} / 100", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        cv2.putText(img, f"STATUS: {road_health.condition_class.upper()}", (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        # Deductions breakdown
        comps = road_health.components
        breakdown_str = f"Deductions: Pothole -{comps.get('pothole_penalty', 0):.1f} | Crack -{comps.get('crack_penalty', 0):.1f} | Water -{comps.get('water_penalty', 0):.1f}"
        cv2.putText(img, breakdown_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Prediction Panel (Right side)
        if prediction is not None:
            pred_x = max(w - 380, w // 2)
            cv2.putText(img, f"TEMPORAL FORECAST ({prediction.prediction_horizon_days}d)", (pred_x, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)

            det_p = prediction.deterioration_probability or 0.0
            pot_p = prediction.pothole_formation_probability or 0.0
            trend = (prediction.progression_trend or "stable").upper()

            cv2.putText(img, f"Deterioration Risk: {det_p:.1%} | Pothole Emergence: {pot_p:.1%}",
                        (pred_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(img, f"Trend: {trend} | Mode: {prediction.scientific_status}",
                        (pred_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)

        return img
