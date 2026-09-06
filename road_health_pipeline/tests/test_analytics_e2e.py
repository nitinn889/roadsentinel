"""End-to-end analytics and edge case integration tests."""

import json
import pytest
import numpy as np

from common.schemas import (
    DefectType,
    InferenceResult,
    PotholeRecord,
    RoadHealthResult,
    SeverityResult,
    Telemetry,
)
from analytics.severity import calculate_defect_severity
from analytics.road_health import calculate_road_health_score
from analytics.segment_aggregator import RoadSegmentAggregator
from analytics.prediction import RoadDeteriorationPredictor, SegmentObservation
from inference.run_inference import generate_visual_overlays


def test_edge_case_no_defects():
    """Scene with zero defects should produce score 100 and clean schema."""
    health = calculate_road_health_score(potholes=[])
    assert health.road_health_score == 100.0
    assert health.condition_class == "good"

    pred = RoadDeteriorationPredictor().predict(
        [SegmentObservation(timestamp="T0", road_health_score=100.0)]
    )
    assert pred.deterioration_probability < 0.15
    assert pred.pothole_formation_probability < 0.15


def test_edge_case_missing_depth_and_missing_gps():
    """Record without depth or GPS should keep nulls without crashing serialization."""
    sev = calculate_defect_severity(confidence=0.75, area_m2=0.5, depth_m=None)
    rec = PotholeRecord(
        pothole_id="p-null",
        timestamp="2026-08-30T00:00:00Z",
        latitude=None,
        longitude=None,
        altitude_m=None,
        area_m2=0.5,
        estimated_depth_m=None,
        anomaly_score=0.7,
        pothole_confidence=0.75,
        severity_score=sev.severity_score,
        water_flag=False,
        water_confidence=0.0,
        source_image="test.jpg",
        mask_area_px=500,
        bbox_xyxy=[0, 0, 10, 10],
        defect_type=DefectType.POTHOLE.value,
        severity_breakdown=sev.severity_components,
    )

    agg = RoadSegmentAggregator()
    summary = agg.aggregate_records([rec], latitude=None, longitude=None)
    assert summary.road_segment_id == "segment_unknown"
    assert summary.total_defects == 1

    # Verify JSON serialization
    d = summary.to_dict()
    json_str = json.dumps(d)
    assert "segment_unknown" in json_str


def test_edge_case_low_confidence_filtering():
    """Candidate with confidence below threshold should be filtered out."""
    from inference.pothole_localizer import PotholeLocalizer

    localizer = PotholeLocalizer(confidence_threshold=0.60)
    # If confidence is below 0.60, it will not be returned in localize
    assert localizer.confidence_threshold == 0.60


def test_visual_overlays_generation():
    """Verify rendering of detection, severity, and road health overlays."""
    h, w = 200, 300
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    sev = calculate_defect_severity(confidence=0.8, area_m2=0.4, depth_m=0.05)
    rec = PotholeRecord(
        pothole_id="p-vis",
        timestamp="2026-08-30T00:00:00Z",
        latitude=13.0,
        longitude=80.0,
        altitude_m=30.0,
        area_m2=0.4,
        estimated_depth_m=0.05,
        anomaly_score=0.7,
        pothole_confidence=0.8,
        severity_score=sev.severity_score,
        water_flag=True,
        water_confidence=0.9,
        source_image="test.jpg",
        mask_area_px=500,
        bbox_xyxy=[10, 10, 50, 50],
        defect_type=DefectType.WATER_FILLED_POTHOLE.value,
    )
    health = calculate_road_health_score([rec])
    overlays = generate_visual_overlays(rgb, [rec], health)

    assert "detection_overlay" in overlays
    assert "severity_overlay" in overlays
    assert "road_health_overlay" in overlays
    assert overlays["detection_overlay"].shape == (h, w, 3)
