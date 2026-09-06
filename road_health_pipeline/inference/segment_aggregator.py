from __future__ import annotations

import math
from typing import Dict, List, Optional
import numpy as np

from common.schemas import (
    DefectMeasurement,
    DefectType,
    PredictionResult,
    RoadHealthScore,
    RoadSegmentAggregate,
)
from config import CONFIG
from inference.road_health_scorer import RoadHealthScorer


class SegmentAggregator:
    """Aggregates individual defect measurements into segment-level records.

    Maintains full traceability: every segment contains the exact list of individual defect records.
    """

    def __init__(self,
                 grid_size_m: float = CONFIG.segment_grid_size_m,
                 scorer: Optional[RoadHealthScorer] = None):
        self.grid_size_m = grid_size_m
        self.scorer = scorer or RoadHealthScorer()

    def generate_segment_id(self,
                            latitude: Optional[float] = None,
                            longitude: Optional[float] = None,
                            world_x: Optional[float] = None,
                            world_y: Optional[float] = None,
                            fallback_id: str = "seg_001") -> str:
        """Generates a stable, reproducible segment identifier based on spatial coordinate binning."""
        if world_x is not None and world_y is not None:
            x_bin = int(math.floor(world_x / self.grid_size_m))
            y_bin = int(math.floor(world_y / self.grid_size_m))
            return f"SEG_X{x_bin:+04d}_Y{y_bin:+04d}"

        if latitude is not None and longitude is not None:
            meters_per_deg = 111_320.0
            lat_bin = int(math.floor((latitude * meters_per_deg) / self.grid_size_m))
            lon_bin = int(math.floor((longitude * meters_per_deg * math.cos(math.radians(latitude))) / self.grid_size_m))
            return f"SEG_L{lat_bin:+06d}_M{lon_bin:+06d}"

        return fallback_id

    def aggregate(self,
                  detections: List[DefectMeasurement],
                  road_segment_id: str,
                  timestamp: str,
                  latitude: Optional[float] = None,
                  longitude: Optional[float] = None,
                  prediction: Optional[PredictionResult] = None) -> RoadSegmentAggregate:
        """Aggregates a list of detections belonging to a segment into a RoadSegmentAggregate."""
        total_defects = len(detections)
        potholes = [d for d in detections if d.defect_type in (DefectType.POTHOLE.value, DefectType.WATER_FILLED_POTHOLE.value)]
        water_potholes = [d for d in detections if d.is_water_filled]
        cracks = [d for d in detections if d.defect_type == DefectType.CRACK.value]

        total_damaged_area = sum(d.estimated_area_m2 or 0.0 for d in detections)
        crack_damage_extent = sum(d.crack_or_damage_extent or d.estimated_area_m2 or 0.0 for d in cracks)

        severities = [d.severity.severity_score for d in detections]
        avg_severity = float(np.mean(severities)) if severities else 0.0
        max_severity = float(np.max(severities)) if severities else 0.0

        # Compute Road Health Score
        health_score = self.scorer.calculate_health(detections)

        # Default prediction if none supplied
        pred_res = prediction or PredictionResult(
            deterioration_probability=0.05 if health_score.road_health_score > 80 else 0.45,
            pothole_formation_probability=0.02 if len(cracks) == 0 else 0.35,
            prediction_horizon_days=30,
            progression_trend="stable" if health_score.road_health_score > 75 else "deteriorating",
            scientific_status="CARLA-SYNTHETIC ONLY",
        )

        return RoadSegmentAggregate(
            road_segment_id=road_segment_id,
            inspection_timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            total_defects=total_defects,
            total_potholes=len(potholes),
            total_water_potholes=len(water_potholes),
            total_cracks=len(cracks),
            total_damaged_area_m2=round(total_damaged_area, 4),
            avg_severity_score=round(avg_severity, 2),
            max_severity_score=round(max_severity, 2),
            water_hazard_count=len(water_potholes),
            crack_damage_extent_m2=round(crack_damage_extent, 4),
            road_health=health_score,
            prediction=pred_res,
            detections=detections,
        )

    def group_and_aggregate(self,
                            detections: List[DefectMeasurement],
                            timestamp: str) -> Dict[str, RoadSegmentAggregate]:
        """Groups detections by road_segment_id and aggregates each segment."""
        grouped: Dict[str, List[DefectMeasurement]] = {}
        for d in detections:
            grouped.setdefault(d.road_segment_id, []).append(d)

        results: Dict[str, RoadSegmentAggregate] = {}
        for seg_id, seg_detections in grouped.items():
            first_lat = next((d.latitude for d in seg_detections if d.latitude is not None), None)
            first_lon = next((d.longitude for d in seg_detections if d.longitude is not None), None)
            results[seg_id] = self.aggregate(
                detections=seg_detections,
                road_segment_id=seg_id,
                timestamp=timestamp,
                latitude=first_lat,
                longitude=first_lon,
            )
        return results
