"""Unit tests for 0-100 road health scoring."""

import pytest
from analytics.road_health import calculate_road_health_score, classify_road_condition
from common.schemas import PotholeRecord, RoadHealthResult


def make_pothole_record(sev=0.5, water=False, area=0.5):
    return PotholeRecord(
        pothole_id="p-01",
        timestamp="2026-08-30T00:00:00Z",
        latitude=13.0,
        longitude=80.0,
        altitude_m=30.0,
        area_m2=area,
        estimated_depth_m=0.05,
        anomaly_score=0.7,
        pothole_confidence=0.8,
        severity_score=sev,
        water_flag=water,
        water_confidence=1.0 if water else 0.0,
        source_image="test.jpg",
        mask_area_px=1000,
        bbox_xyxy=[10, 10, 50, 50],
    )


def test_road_health_pristine():
    """No defects -> score should be 100 / GOOD."""
    res = calculate_road_health_score(potholes=[], total_crack_area_m2=0.0, surface_anomaly_mean=0.0)
    assert isinstance(res, RoadHealthResult)
    assert res.road_health_score == 100.0
    assert res.condition_class == "good"
    assert res.components["total_penalty"] == 0.0


def test_road_health_single_mild_pothole():
    """Single mild pothole -> minor penalty -> FAIR or GOOD."""
    p = make_pothole_record(sev=0.3, water=False)
    res = calculate_road_health_score(potholes=[p])
    assert 70.0 <= res.road_health_score <= 95.0
    assert res.components["pothole_count_penalty"] > 0
    assert res.components["pothole_severity_penalty"] > 0


def test_road_health_multiple_severe_water_potholes():
    """Multiple severe water potholes -> POOR or CRITICAL."""
    potholes = [make_pothole_record(sev=0.9, water=True, area=1.5) for _ in range(4)]
    res = calculate_road_health_score(potholes=potholes, total_crack_area_m2=2.0)
    assert res.road_health_score < 50.0
    assert res.condition_class in {"poor", "critical"}
    assert res.components["water_hazard_penalty"] > 0
