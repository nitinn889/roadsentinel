"""Unit tests for CARLA simulation ground truth evaluation."""

import pytest
from common.schemas import PotholeRecord
from evaluation.carla_eval import CarlaDefectGroundTruth, evaluate_carla_ground_truth


def test_carla_ground_truth_exact_match():
    pred = PotholeRecord(
        pothole_id="p-01",
        timestamp="2026-08-30T00:00:00Z",
        latitude=13.0,
        longitude=80.0,
        altitude_m=30.0,
        area_m2=0.50,
        estimated_depth_m=0.10,
        anomaly_score=0.8,
        pothole_confidence=0.9,
        severity_score=0.60,
        water_flag=True,
        water_confidence=0.8,
        source_image="test.jpg",
        mask_area_px=2000,
        bbox_xyxy=[10, 10, 50, 50],
    )
    gt = CarlaDefectGroundTruth(
        defect_id="gt-01",
        true_area_m2=0.50,
        true_depth_m=0.10,
        true_world_x=0.0,
        true_world_y=0.0,
        is_water_filled=True,
        true_severity=0.60,
    )

    metrics = evaluate_carla_ground_truth([pred], [gt])
    assert metrics["matched_pairs"] == 1
    assert metrics["detection_f1"] == 1.0
    assert metrics["area_mae_m2"] == 0.0
    assert metrics["depth_mae_m"] == 0.0
    assert metrics["water_f1"] == 1.0


def test_carla_ground_truth_water_false_positive():
    pred = PotholeRecord(
        pothole_id="p-01",
        timestamp="2026-08-30T00:00:00Z",
        latitude=13.0,
        longitude=80.0,
        altitude_m=30.0,
        area_m2=0.50,
        estimated_depth_m=0.08,
        anomaly_score=0.8,
        pothole_confidence=0.9,
        severity_score=0.50,
        water_flag=True,  # FP
        water_confidence=0.8,
        source_image="test.jpg",
        mask_area_px=2000,
        bbox_xyxy=[10, 10, 50, 50],
    )
    gt = CarlaDefectGroundTruth(
        defect_id="gt-01",
        true_area_m2=0.50,
        true_depth_m=0.08,
        true_world_x=0.0,
        true_world_y=0.0,
        is_water_filled=False,  # Dry
        true_severity=0.50,
    )

    metrics = evaluate_carla_ground_truth([pred], [gt])
    assert metrics["water_confusion_matrix"]["fp"] == 1
    assert metrics["water_precision"] == 0.0
