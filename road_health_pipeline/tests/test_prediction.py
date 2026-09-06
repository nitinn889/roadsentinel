"""Unit tests for temporal feature extraction and deterioration prediction."""

import pytest
import numpy as np
from analytics.prediction import (
    RoadDeteriorationPredictor,
    SegmentObservation,
    extract_temporal_features,
)


def test_extract_temporal_features_empty():
    feats = extract_temporal_features([])
    assert feats["health_score_latest"] == 100.0
    assert feats["health_score_slope_per_day"] == 0.0


def test_extract_temporal_features_multi_point():
    obs = [
        SegmentObservation(timestamp="2026-08-01T00:00:00Z", road_health_score=95.0, day_offset=0.0, damaged_area_m2=0.1),
        SegmentObservation(timestamp="2026-08-15T00:00:00Z", road_health_score=80.0, day_offset=14.0, damaged_area_m2=0.8),
        SegmentObservation(timestamp="2026-08-29T00:00:00Z", road_health_score=65.0, day_offset=28.0, damaged_area_m2=1.5),
    ]
    feats = extract_temporal_features(obs)
    assert feats["health_score_latest"] == 65.0
    assert feats["timespan_days"] == 28.0
    assert feats["health_score_slope_per_day"] < 0  # degrading slope
    assert feats["damaged_area_growth_rate"] > 0


def test_predictor_single_observation_fallback():
    predictor = RoadDeteriorationPredictor(horizon_days=30)
    obs = [SegmentObservation(timestamp="T0", road_health_score=85.0)]
    res = predictor.predict(obs, road_segment_id="seg_test")
    assert 0.0 <= res.deterioration_probability <= 1.0
    assert res.prediction_horizon_days == 30
    assert len(res.notes) > 0


def test_predictor_severe_degradation_alert():
    predictor = RoadDeteriorationPredictor(horizon_days=30)
    obs = [
        SegmentObservation(timestamp="T0", road_health_score=90.0, day_offset=0.0),
        SegmentObservation(timestamp="T1", road_health_score=40.0, day_offset=14.0, pothole_count=3, has_water_hazard=True),
    ]
    res = predictor.predict(obs, road_segment_id="seg_test_severe")
    assert res.deterioration_probability >= 0.70
    assert res.progression_direction in {"degrading", "critical"}
