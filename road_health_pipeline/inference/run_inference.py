from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import cv2
import numpy as np

from common.schemas import (
    CandidateRegion,
    DefectMeasurement,
    DefectType,
    InferenceResult,
    PotholeRecord,
    PredictionResult,
    RoadHealthScore,
    RoadSegmentAggregate,
    Telemetry,
)
from config import CONFIG, Config
from inference.area_estimator import estimate_area_m2
from inference.defect_classifier import DefectClassifier
from inference.depth_estimator import DepthEstimator, NullDepthEstimator
from inference.gps_localizer import GPSLocalizer, telemetry_from_dict
from inference.road_health_scorer import RoadHealthScorer
from inference.segment_aggregator import SegmentAggregator
from inference.severity_estimator import SeverityEstimator
from inference.visualizer import PipelineVisualizer
from prediction.progression_model import DeteriorationPredictor, TemporalInspectionRecord


class RoadSentinelPipeline:
    """Complete analytics pipeline processing SAM2 masks into physical measurements,

    explainable severity, segment-level road health scores, and temporal deterioration predictions.
    """

    def __init__(self,
                 config: Optional[Config] = None,
                 depth_estimator: Optional[DepthEstimator] = None,
                 localizer: Optional[GPSLocalizer] = None):
        self.cfg = config or CONFIG
        self.depth_estimator = depth_estimator or NullDepthEstimator()
        self.localizer = localizer or GPSLocalizer(
            origin_lat=self.cfg.carla_origin_lat,
            origin_lon=self.cfg.carla_origin_lon,
        )
        self.classifier = DefectClassifier()
        self.severity_estimator = SeverityEstimator(self.cfg.severity)
        self.health_scorer = RoadHealthScorer(self.cfg.road_health)
        self.segment_aggregator = SegmentAggregator(self.cfg.segment_grid_size_m, self.health_scorer)
        self.predictor = DeteriorationPredictor(self.cfg.prediction)
        self.visualizer = PipelineVisualizer()

    def process_candidates(self,
                           rgb_image: np.ndarray,
                           candidates: List[CandidateRegion],
                           telemetry: Telemetry,
                           road_segment_id: Optional[str] = None,
                           history: Optional[List[TemporalInspectionRecord]] = None
                           ) -> InferenceResult:
        """Processes candidate regions from DINOv2+SAM2 through the analytics layer.

        Steps:
        1. Localize frame coordinates / GPS
        2. Estimate metric depth (if depth sensor/model available)
        3. Extract defect measurements (area m^2, classification, water detection)
        4. Calculate transparent severity breakdown
        5. Score segment-level road health (0-100)
        6. Forecast temporal deterioration and pothole emergence
        7. Produce structured, traceable result
        """
        # 1. Attach / resolve GPS
        telemetry = self.localizer.attach(telemetry)
        h, w = rgb_image.shape[:2]

        # Determine segment ID
        seg_id = road_segment_id or self.segment_aggregator.generate_segment_id(
            latitude=telemetry.latitude,
            longitude=telemetry.longitude,
            world_x=telemetry.world_x,
            world_y=telemetry.world_y,
        )

        # 2. Depth estimation
        depth_map = self.depth_estimator.estimate(rgb_image)
        depth_source = self.depth_estimator.name

        detections: List[DefectMeasurement] = []
        legacy_potholes: List[PotholeRecord] = []

        # 3 & 4. Process each candidate
        for idx, cand in enumerate(candidates):
            mask = cand.sam2_result.mask if cand.sam2_result is not None else cand.mask
            conf = cand.sam2_result.confidence if cand.sam2_result is not None else cand.pothole_confidence
            bbox = cand.sam2_result.bbox_xyxy if cand.sam2_result is not None else cand.bbox_xyxy
            mask_px = int(np.sum(mask))

            # Physical area estimation
            area_m2 = estimate_area_m2(
                mask=mask,
                altitude_m=telemetry.altitude_m,
                horizontal_fov_deg=self.cfg.horizontal_fov_deg,
            )

            # Depth measurement within mask
            estimated_depth_m: Optional[float] = None
            if depth_map is not None:
                masked_depths = depth_map[mask]
                valid_d = masked_depths[np.isfinite(masked_depths) & (masked_depths > 0)]
                if len(valid_d) > 0:
                    estimated_depth_m = float(np.median(valid_d))

            # Fine-grained classification & water detection
            defect_type, is_water, water_conf, morph = self.classifier.classify(
                mask=mask,
                rgb_image=rgb_image,
                anomaly_score=cand.anomaly_score,
                confidence=conf,
            )

            # Crack or damage extent
            crack_extent = None
            if defect_type == DefectType.CRACK.value:
                # Approximate perimeter / length in metres if altitude known
                crack_extent = area_m2 * 2.0 if area_m2 is not None else float(morph["perimeter"]) * 0.001

            # Severity computation
            severity = self.severity_estimator.compute_severity(
                area_m2=area_m2,
                depth_m=estimated_depth_m,
                is_water_filled=is_water,
                water_confidence=water_conf,
                crack_or_damage_extent=crack_extent,
                confidence=conf,
                mask_area_px=mask_px,
            )

            defect_id = f"DEF_{telemetry.timestamp}_{idx:03d}"

            measurement = DefectMeasurement(
                defect_id=defect_id,
                defect_type=defect_type,
                confidence=float(conf),
                bbox=bbox,
                mask_area_pixels=mask_px,
                estimated_area_m2=area_m2,
                estimated_depth_m=estimated_depth_m,
                is_water_filled=is_water,
                water_confidence=float(water_conf),
                crack_or_damage_extent=crack_extent,
                road_segment_id=seg_id,
                timestamp=telemetry.timestamp,
                latitude=telemetry.latitude,
                longitude=telemetry.longitude,
                severity=severity,
                depth_source=depth_source,
            )
            detections.append(measurement)

            # Legacy compatibility record
            legacy_potholes.append(PotholeRecord(
                pothole_id=defect_id,
                timestamp=telemetry.timestamp,
                latitude=telemetry.latitude,
                longitude=telemetry.longitude,
                altitude_m=telemetry.altitude_m,
                area_m2=area_m2,
                estimated_depth_m=estimated_depth_m,
                anomaly_score=cand.anomaly_score,
                pothole_confidence=conf,
                severity_score=severity.severity_score,
                water_flag=is_water,
                water_confidence=water_conf,
                source_image=getattr(telemetry, "frame_id", "") or "frame",
                mask_area_px=mask_px,
                bbox_xyxy=bbox,
                depth_source=depth_source,
            ))

        # 5. Road Health Score
        road_health = self.health_scorer.calculate_health(detections)

        # 6. Temporal Deterioration Prediction
        hist_records = list(history or [])
        # Append current observation
        total_damaged_area = sum(d.estimated_area_m2 or 0.0 for d in detections)
        pothole_count = sum(1 for d in detections if "pothole" in d.defect_type)
        crack_count = sum(1 for d in detections if d.defect_type == DefectType.CRACK.value)
        max_sev = max((d.severity.severity_score for d in detections), default=0.0)
        has_water = any(d.is_water_filled for d in detections)

        current_record = TemporalInspectionRecord(
            timestamp_days=0.0,
            road_health_score=road_health.road_health_score,
            total_damaged_area_m2=total_damaged_area,
            pothole_count=pothole_count,
            crack_count=crack_count,
            max_severity_score=max_sev,
            water_present=has_water,
        )
        combined_hist = hist_records + [current_record]

        prediction = self.predictor.predict(
            road_segment_id=seg_id,
            history=combined_hist,
            horizon_days=self.cfg.prediction.default_horizon_days,
        )

        return InferenceResult(
            image_id=f"IMG_{telemetry.timestamp}",
            timestamp=telemetry.timestamp,
            road_segment_id=seg_id,
            geolocation={
                "lat": telemetry.latitude,
                "lon": telemetry.longitude,
            },
            image_shape=[h, w, 3],
            anomaly_threshold=self.cfg.anomaly_percentile,
            anomaly_score=float(np.mean([c.anomaly_score for c in candidates])) if candidates else 0.0,
            detections=detections,
            road_health=road_health,
            prediction=prediction,
            potholes=legacy_potholes,
            telemetry={
                "latitude": telemetry.latitude,
                "longitude": telemetry.longitude,
                "altitude_m": telemetry.altitude_m,
                "heading_deg": telemetry.heading_deg,
                "world_x": telemetry.world_x,
                "world_y": telemetry.world_y,
                "speed_mps": telemetry.speed_mps,
            },
            frame_id=telemetry.frame_id,
        )

    def generate_diagnostic_overlays(self,
                                     rgb_image: np.ndarray,
                                     result: InferenceResult,
                                     output_dir: Path) -> Dict[str, Path]:
        """Renders and saves detection_overlay.jpg, severity_overlay.jpg, and road_health_overlay.jpg."""
        output_dir.mkdir(parents=True, exist_ok=True)

        det_overlay = self.visualizer.draw_detection_overlay(rgb_image, result.detections)
        sev_overlay = self.visualizer.draw_severity_overlay(rgb_image, result.detections)
        health_overlay = self.visualizer.draw_road_health_overlay(
            rgb_image=rgb_image,
            road_health=result.road_health,
            road_segment_id=result.road_segment_id,
            prediction=result.prediction,
        )

        det_path = output_dir / "detection_overlay.jpg"
        sev_path = output_dir / "severity_overlay.jpg"
        health_path = output_dir / "road_health_overlay.jpg"
        json_path = output_dir / "result.json"

        # Save BGR for OpenCV
        cv2.imwrite(str(det_path), cv2.cvtColor(det_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sev_path), cv2.cvtColor(sev_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(health_path), cv2.cvtColor(health_overlay, cv2.COLOR_RGB2BGR))

        with open(json_path, "w") as f:
            f.write(result.to_json(indent=2))

        return {
            "detection_overlay": det_path,
            "severity_overlay": sev_path,
            "road_health_overlay": health_path,
            "result_json": json_path,
        }
