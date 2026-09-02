from __future__ import annotations

from typing import List, Optional
import numpy as np

from common.schemas import DefectMeasurement, DefectType, RoadHealthScore
from config import CONFIG, RoadHealthWeights


class RoadHealthScorer:
    """Calculates transparent, explainable 0-100 road health scores for road segments.

    100 = Brand new / healthy road
    0   = Severely degraded / impassable road
    """

    def __init__(self, weights: Optional[RoadHealthWeights] = None):
        self.cfg = weights or CONFIG.road_health

    def calculate_health(self,
                         detections: List[DefectMeasurement],
                         segment_area_m2: Optional[float] = None,
                         overall_confidence: float = 1.0) -> RoadHealthScore:
        """Calculates segment road health score and component deductions from detected defects."""
        if not detections:
            return RoadHealthScore(
                road_health_score=100.0,
                condition_class="Good",
                components={
                    "pothole_penalty": 0.0,
                    "crack_penalty": 0.0,
                    "water_penalty": 0.0,
                    "surface_penalty": 0.0,
                },
                confidence=overall_confidence,
                explanation="No defects detected. Road segment is in optimal condition.",
            )

        # Separate defects by type
        potholes = [d for d in detections if d.defect_type in (DefectType.POTHOLE.value, DefectType.WATER_FILLED_POTHOLE.value)]
        cracks = [d for d in detections if d.defect_type == DefectType.CRACK.value]
        water_hazards = [d for d in detections if d.is_water_filled]
        surface_anomalies = [d for d in detections if d.defect_type in (DefectType.SURFACE_WEAR.value, DefectType.ROAD_ANOMALY.value)]

        # 1. Pothole Penalty (up to max_pothole_penalty, e.g. 50.0)
        pothole_severity_sum = sum(d.severity.severity_score for d in potholes)
        raw_pothole_penalty = (len(potholes) * 8.0) + (pothole_severity_sum * 0.15)
        pothole_penalty = float(np.clip(raw_pothole_penalty, 0.0, self.cfg.max_pothole_penalty))

        # 2. Crack / Structural Penalty (up to max_crack_penalty, e.g. 25.0)
        crack_area_sum = sum(d.estimated_area_m2 or 0.05 for d in cracks)
        raw_crack_penalty = (len(cracks) * 4.0) + (crack_area_sum * 15.0)
        crack_penalty = float(np.clip(raw_crack_penalty, 0.0, self.cfg.max_crack_penalty))

        # 3. Water Hazard Penalty (up to max_water_penalty, e.g. 15.0)
        raw_water_penalty = len(water_hazards) * 7.5
        water_penalty = float(np.clip(raw_water_penalty, 0.0, self.cfg.max_water_penalty))

        # 4. Surface Deterioration Penalty (up to max_surface_penalty, e.g. 10.0)
        raw_surface_penalty = len(surface_anomalies) * 2.5
        surface_penalty = float(np.clip(raw_surface_penalty, 0.0, self.cfg.max_surface_penalty))

        # Aggregate total deductions
        total_deduction = pothole_penalty + crack_penalty + water_penalty + surface_penalty
        health_score = float(np.clip(100.0 - total_deduction, 0.0, 100.0))

        # Condition Class Mapping
        if health_score >= self.cfg.condition_good_min:
            condition_class = "Good"
        elif health_score >= self.cfg.condition_fair_min:
            condition_class = "Fair"
        elif health_score >= self.cfg.condition_poor_min:
            condition_class = "Poor"
        else:
            condition_class = "Critical"

        # Generate Explainable Summary
        explanations = []
        if potholes:
            explanations.append(f"{len(potholes)} pothole(s) (-{pothole_penalty:.1f} pts)")
        if cracks:
            explanations.append(f"{len(cracks)} crack formation(s) (-{crack_penalty:.1f} pts)")
        if water_hazards:
            explanations.append(f"{len(water_hazards)} water puddle hazard(s) (-{water_penalty:.1f} pts)")
        if surface_anomalies:
            explanations.append(f"{len(surface_anomalies)} surface wear area(s) (-{surface_penalty:.1f} pts)")

        explanation_str = f"Condition: {condition_class} ({health_score:.1f}/100). " + "; ".join(explanations)

        return RoadHealthScore(
            road_health_score=round(health_score, 2),
            condition_class=condition_class,
            components={
                "pothole_penalty": round(pothole_penalty, 2),
                "crack_penalty": round(crack_penalty, 2),
                "water_penalty": round(water_penalty, 2),
                "surface_penalty": round(surface_penalty, 2),
            },
            confidence=round(overall_confidence, 4),
            explanation=explanation_str,
        )
