from __future__ import annotations

import json
import math
from pathlib import Path
import cv2
import numpy as np

try:
    import pytest
    approx = pytest.approx
except ImportError:
    def approx(expected, abs=1e-3):
        class ApproxVal:
            def __init__(self, exp, tol):
                self.exp = exp
                self.tol = tol
            def __eq__(self, other):
                return math.isclose(other, self.exp, abs_tol=self.tol)
            def __repr__(self):
                return f"approx({self.exp} ± {self.tol})"
        return ApproxVal(expected, abs)

from common.schemas import (
    CandidateRegion,
    DefectMeasurement,
    DefectType,
    InferenceResult,
    PredictionResult,
    RoadHealthScore,
    RoadSegmentAggregate,
    SegmentationResult,
    SeverityBreakdown,
    Telemetry,
)
from config import CONFIG, Config
from evaluation.depth_metrics import compute_depth_metrics
from evaluation.ground_truth_evaluator import CarlaGroundTruthEvaluator, GroundTruthDefect
from evaluation.prediction_metrics import PredictionEvaluator
from inference.area_estimator import estimate_area_m2
from inference.defect_classifier import DefectClassifier
from inference.depth_estimator import NullDepthEstimator
from inference.gps_localizer import GPSLocalizer
from inference.road_health_scorer import RoadHealthScorer
from inference.run_inference import RoadSentinelPipeline
from inference.segment_aggregator import SegmentAggregator
from inference.severity_estimator import SeverityEstimator
from inference.visualizer import PipelineVisualizer
from prediction.carla_temporal_dataset import CarlaTemporalSequenceGenerator
from prediction.progression_model import DeteriorationPredictor, TemporalInspectionRecord


def test_area_estimation():
    # 1280x720 frame at 30m altitude, 90 deg hfov
    mask = np.zeros((720, 1280), dtype=bool)
    mask[300:400, 500:600] = True  # 10,000 pixels
    area = estimate_area_m2(mask, altitude_m=30.0, horizontal_fov_deg=90.0)
    assert area is not None
    assert area > 0.0
    # Missing altitude returns None
    assert estimate_area_m2(mask, altitude_m=None) is None
    assert estimate_area_m2(mask, altitude_m=-1.0) is None


def test_depth_metrics_computation():
    gt = np.full((100, 100), 5.0, dtype=np.float32)
    pred = np.full((100, 100), 5.5, dtype=np.float32)
    metrics = compute_depth_metrics(pred, gt)
    assert metrics["mae_m"] == approx(0.5, abs=1e-3)
    assert metrics["rmse_m"] == approx(0.5, abs=1e-3)
    assert metrics["abs_rel"] == approx(0.1, abs=1e-3)
    assert metrics["delta_1"] == 1.0


def test_severity_calculation_with_and_without_depth():
    estimator = SeverityEstimator()

    # Case A: with depth and area
    sev_with_depth = estimator.compute_severity(
        area_m2=0.35,
        depth_m=0.06,
        is_water_filled=False,
        confidence=0.9,
    )
    assert sev_with_depth.severity in ("medium", "high", "critical")
    assert sev_with_depth.severity_components["depth"] is not None
    assert sev_with_depth.severity_components["area"] is not None

    # Case B: missing depth (RGB-only honest reweighting)
    sev_no_depth = estimator.compute_severity(
        area_m2=0.35,
        depth_m=None,
        is_water_filled=True,
        water_confidence=0.85,
        confidence=0.9,
    )
    assert sev_no_depth.severity_components["depth"] is None
    assert sev_no_depth.severity_components["water"] is not None
    assert sev_no_depth.severity_score > 0.0


def test_road_health_scoring():
    scorer = RoadHealthScorer()

    # Case 1: No defects (100 healthy)
    h_empty = scorer.calculate_health([])
    assert h_empty.road_health_score == 100.0
    assert h_empty.condition_class == "Good"

    # Case 2: Severe potholes and cracks
    sev_breakdown = SeverityBreakdown(severity="critical", severity_score=85.0)
    defects = [
        DefectMeasurement(
            defect_id="D1",
            defect_type=DefectType.POTHOLE.value,
            confidence=0.95,
            bbox=[10, 10, 100, 100],
            mask_area_pixels=5000,
            estimated_area_m2=0.45,
            estimated_depth_m=0.08,
            is_water_filled=True,
            water_confidence=0.9,
            crack_or_damage_extent=None,
            road_segment_id="SEG_01",
            timestamp="2026-08-30T12:00:00",
            latitude=13.0827,
            longitude=80.2707,
            severity=sev_breakdown,
        ),
        DefectMeasurement(
            defect_id="D2",
            defect_type=DefectType.CRACK.value,
            confidence=0.88,
            bbox=[120, 120, 300, 200],
            mask_area_pixels=3000,
            estimated_area_m2=0.25,
            estimated_depth_m=None,
            is_water_filled=False,
            water_confidence=0.0,
            crack_or_damage_extent=0.8,
            road_segment_id="SEG_01",
            timestamp="2026-08-30T12:00:00",
            latitude=13.0827,
            longitude=80.2707,
            severity=sev_breakdown,
        ),
    ]

    h_degraded = scorer.calculate_health(defects)
    assert h_degraded.road_health_score < 80.0
    assert h_degraded.components["pothole_penalty"] > 0
    assert h_degraded.components["crack_penalty"] > 0
    assert h_degraded.components["water_penalty"] > 0


def test_segment_aggregation_traceability():
    aggregator = SegmentAggregator()
    sev = SeverityBreakdown(severity="high", severity_score=65.0)
    d1 = DefectMeasurement(
        defect_id="D1",
        defect_type=DefectType.POTHOLE.value,
        confidence=0.9,
        bbox=[0, 0, 50, 50],
        mask_area_pixels=1000,
        estimated_area_m2=0.1,
        estimated_depth_m=0.03,
        is_water_filled=False,
        water_confidence=0.0,
        crack_or_damage_extent=None,
        road_segment_id="SEG_X000_Y000",
        timestamp="2026-08-30",
        latitude=13.0827,
        longitude=80.2707,
        severity=sev,
    )
    seg_agg = aggregator.aggregate([d1], "SEG_X000_Y000", "2026-08-30", 13.0827, 80.2707)
    assert seg_agg.total_defects == 1
    assert seg_agg.total_potholes == 1
    assert len(seg_agg.detections) == 1
    assert seg_agg.detections[0].defect_id == "D1"  # full traceability


def test_carla_ground_truth_comparison():
    evaluator = CarlaGroundTruthEvaluator()
    sev = SeverityBreakdown(severity="high", severity_score=70.0)
    pred = [
        DefectMeasurement(
            defect_id="D1",
            defect_type=DefectType.WATER_FILLED_POTHOLE.value,
            confidence=0.92,
            bbox=[100, 100, 200, 200],
            mask_area_pixels=2500,
            estimated_area_m2=0.22,
            estimated_depth_m=0.045,
            is_water_filled=True,
            water_confidence=0.95,
            crack_or_damage_extent=None,
            road_segment_id="SEG_01",
            timestamp="2026-08-30",
            latitude=13.0827,
            longitude=80.2707,
            severity=sev,
        )
    ]
    gt = [
        GroundTruthDefect(
            defect_id="D1",
            true_area_m2=0.20,
            true_depth_m=0.050,
            true_x_m=0.0,
            true_y_m=0.0,
            true_is_water=True,
            true_severity_level="high",
        )
    ]
    report = evaluator.evaluate_detections(pred, gt)
    assert report.num_samples == 1
    assert report.area_mae_m2 == approx(0.02, abs=1e-3)
    assert report.depth_mae_m == approx(0.005, abs=1e-3)
    assert report.water_accuracy == 1.0
    assert report.severity_exact_agreement == 1.0


def test_temporal_prediction_interface():
    predictor = DeteriorationPredictor()

    # Incomplete sequence (single step)
    single_step = [
        TemporalInspectionRecord(
            timestamp_days=0.0,
            road_health_score=85.0,
            total_damaged_area_m2=0.05,
            pothole_count=0,
            crack_count=1,
            max_severity_score=30.0,
            water_present=False,
        )
    ]
    p_single = predictor.predict("SEG_01", single_step, horizon_days=30)
    assert p_single.prediction_horizon_days == 30
    assert 0.0 <= p_single.deterioration_probability <= 1.0
    assert 0.0 <= p_single.pothole_formation_probability <= 1.0

    # Multi-step degrading sequence
    multi_step = [
        TemporalInspectionRecord(0.0, 90.0, 0.02, 0, 1, 20.0, False),
        TemporalInspectionRecord(15.0, 75.0, 0.10, 0, 2, 45.0, True),
        TemporalInspectionRecord(30.0, 55.0, 0.28, 1, 2, 70.0, True),
    ]
    p_multi = predictor.predict("SEG_01", multi_step, horizon_days=30)
    assert p_multi.progression_trend in ("deteriorating", "rapidly_deteriorating")
    assert p_multi.deterioration_probability > 0.5


def test_carla_synthetic_temporal_dataset_and_evaluator():
    gen = CarlaTemporalSequenceGenerator(seed=42)
    train_seqs, test_seqs = gen.generate_dataset(num_segments=20, train_ratio=0.70)
    assert len(train_seqs) == 14
    assert len(test_seqs) == 6

    # Verify no segment overlap (no leakage)
    train_ids = {s.road_segment_id for s in train_seqs}
    test_ids = {s.road_segment_id for s in test_seqs}
    assert len(train_ids.intersection(test_ids)) == 0

    # Run predictions on test set and evaluate
    predictor = DeteriorationPredictor()
    evaluator = PredictionEvaluator()

    true_dets = [s.ground_truth_deteriorated for s in test_seqs]
    true_pots = [s.ground_truth_pothole_formed for s in test_seqs]
    sim_trends = [s.progression_type for s in test_seqs]

    pred_dets = []
    pred_pots = []
    pred_trends = []

    for s in test_seqs:
        # Use first 3 inspections to predict future at day 60
        res = predictor.predict(s.road_segment_id, s.inspections[:3], horizon_days=30)
        pred_dets.append(res.deterioration_probability)
        pred_pots.append(res.pothole_formation_probability)
        pred_trends.append(res.progression_trend)

    report = evaluator.evaluate(
        true_deteriorated=true_dets,
        pred_deterioration_probs=pred_dets,
        true_pothole_formed=true_pots,
        pred_pothole_probs=pred_pots,
        simulated_progression_trends=sim_trends,
        predicted_progression_trends=pred_trends,
    )
    assert report.num_sequences == 6
    assert report.directional_trend_agreement_pct > 0.0


def test_end_to_end_pipeline_and_json_schema(tmp_path: Path):
    pipeline = RoadSentinelPipeline()
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(img, (640, 360), 40, (50, 50, 50), -1)

    # Synthetic candidate mask
    mask = np.zeros((720, 1280), dtype=bool)
    mask[320:400, 600:680] = True
    cand = CandidateRegion(
        mask=mask,
        bbox_xyxy=[600, 320, 680, 400],
        anomaly_score=0.82,
        pothole_confidence=0.88,
        sam2_result=SegmentationResult(
            mask=mask,
            confidence=0.91,
            bbox_xyxy=[600, 320, 680, 400],
            area_px=int(np.sum(mask)),
        ),
    )

    telem = Telemetry(
        timestamp="2026-08-30T12:00:00Z",
        latitude=13.0827,
        longitude=80.2707,
        altitude_m=30.0,
        heading_deg=0.0,
        world_x=10.0,
        world_y=20.0,
    )

    result = pipeline.process_candidates(img, [cand], telem)

    # Verify standard schema fields
    d = result.to_dict()
    assert "image_id" in d
    assert "road_segment_id" in d
    assert "geolocation" in d
    assert "detections" in d
    assert "road_health" in d
    assert "prediction" in d
    assert len(d["detections"]) == 1
    assert d["detections"][0]["defect_type"] in ("pothole", "water_filled_pothole")

    # Generate visual overlays
    out_files = pipeline.generate_diagnostic_overlays(img, result, tmp_path)
    assert Path(out_files["detection_overlay"]).exists()
    assert Path(out_files["severity_overlay"]).exists()
    assert Path(out_files["road_health_overlay"]).exists()
    assert Path(out_files["result_json"]).exists()
