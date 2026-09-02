from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure road_health_pipeline is in sys.path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from common.schemas import (
    CandidateRegion,
    SegmentationResult,
    Telemetry,
)
from evaluation.ground_truth_evaluator import CarlaGroundTruthEvaluator, GroundTruthDefect
from evaluation.prediction_metrics import PredictionEvaluator
from inference.run_inference import RoadSentinelPipeline
from prediction.carla_temporal_dataset import CarlaTemporalSequenceGenerator
from prediction.progression_model import DeteriorationPredictor


def run_full_demo():
    print("=================================================================")
    print("  ROADSENTINEL — ROAD HEALTH, SEVERITY & PREDICTION EVALUATION   ")
    print("=================================================================")

    start_time = time.time()
    pipeline = RoadSentinelPipeline()

    # 1. Simulate RGB Drone Image with Road Defects
    # At 30m altitude & 90 deg HFOV on 1280x720: 1 pixel ~ 0.0469m (4.69 cm)
    img_h, img_w = 720, 1280
    rgb = np.full((img_h, img_w, 3), 60, dtype=np.uint8)  # Asphalt base
    # Add road lane markings
    cv2.line(rgb, (0, 360), (1280, 360), (220, 220, 220), 4)

    # Defect 1: Open Pothole (radius 10 px ~ 0.47m radius, area ~ 0.69 m^2)
    cv2.circle(rgb, (400, 250), 10, (25, 25, 25), -1)
    mask1 = np.zeros((img_h, img_w), dtype=bool)
    cv2.circle(mask1.view(np.uint8), (400, 250), 10, 1, -1)

    # Defect 2: Water-filled Pothole (dark puddle + specular reflection)
    cv2.circle(rgb, (750, 420), 12, (15, 20, 30), -1)
    cv2.circle(rgb, (752, 418), 3, (250, 250, 255), -1)  # Specular glint
    mask2 = np.zeros((img_h, img_w), dtype=bool)
    cv2.circle(mask2.view(np.uint8), (750, 420), 12, 1, -1)

    # Defect 3: Longitudinal Crack (elongated line)
    pts = np.array([[900, 150], [920, 220], [940, 290], [960, 380]], dtype=np.int32)
    cv2.polylines(rgb, [pts], False, (20, 20, 20), 3)
    mask3 = np.zeros((img_h, img_w), dtype=bool)
    cv2.polylines(mask3.view(np.uint8), [pts], False, 1, 3)

    candidates = [
        CandidateRegion(
            mask=mask1,
            bbox_xyxy=[390, 240, 410, 260],
            anomaly_score=0.89,
            pothole_confidence=0.92,
            sam2_result=SegmentationResult(mask1, 0.94, [390, 240, 410, 260], int(mask1.sum())),
        ),
        CandidateRegion(
            mask=mask2,
            bbox_xyxy=[738, 408, 762, 432],
            anomaly_score=0.94,
            pothole_confidence=0.96,
            sam2_result=SegmentationResult(mask2, 0.95, [738, 408, 762, 432], int(mask2.sum())),
        ),
        CandidateRegion(
            mask=mask3,
            bbox_xyxy=[900, 150, 960, 380],
            anomaly_score=0.78,
            pothole_confidence=0.70,
            sam2_result=SegmentationResult(mask3, 0.82, [900, 150, 960, 380], int(mask3.sum())),
        ),
    ]

    telemetry = Telemetry(
        timestamp="2026-08-30T12:00:00Z",
        latitude=13.0827,
        longitude=80.2707,
        altitude_m=30.0,
        heading_deg=45.0,
        world_x=120.0,
        world_y=340.0,
        frame_id=101,
        speed_mps=8.33,
    )

    print("\n[1/4] Running Road Health Inference on drone capture...")
    result = pipeline.process_candidates(rgb, candidates, telemetry)

    output_dir = PIPELINE_ROOT / "outputs"
    saved_files = pipeline.generate_diagnostic_overlays(rgb, result, output_dir)
    print(f" -> Result JSON: {saved_files['result_json']}")
    print(f" -> Detection Overlay: {saved_files['detection_overlay']}")
    print(f" -> Severity Overlay: {saved_files['severity_overlay']}")
    print(f" -> Road Health Overlay: {saved_files['road_health_overlay']}")

    # 2. CARLA Ground Truth Evaluation
    print("\n[2/4] Evaluating against CARLA Ground Truth...")
    gt_evaluator = CarlaGroundTruthEvaluator()
    # Mask 1 area ~ 305 px * (0.046875)^2 ~ 0.670 m^2
    # Mask 2 area ~ 441 px * (0.046875)^2 ~ 0.969 m^2
    # Mask 3 crack area ~ 0.70 m^2
    gts = [
        GroundTruthDefect("DEF_001", true_area_m2=0.68, true_depth_m=0.065, true_x_m=120.0, true_y_m=340.0, true_is_water=False, true_severity_level="high"),
        GroundTruthDefect("DEF_002", true_area_m2=0.98, true_depth_m=0.080, true_x_m=121.5, true_y_m=342.0, true_is_water=True, true_severity_level="critical"),
        GroundTruthDefect("DEF_003", true_area_m2=0.72, true_depth_m=0.015, true_x_m=124.0, true_y_m=345.0, true_is_water=False, true_severity_level="medium"),
    ]
    gt_report = gt_evaluator.evaluate_detections(result.detections, gts)
    print(" CARLA Ground-Truth Results:")
    print(f"   Area MAE: {gt_report.area_mae_m2:.4f} m^2 (MAPE: {gt_report.area_mape_percent:.1f}%)")
    print(f"   Water Classification Accuracy: {gt_report.water_accuracy * 100.0:.1f}% (F1: {gt_report.water_f1:.2f})")
    print(f"   Severity Exact Agreement: {gt_report.severity_exact_agreement * 100.0:.1f}% (Within-1-class: {gt_report.severity_within_one_class * 100.0:.1f}%)")

    # 3. Temporal Deterioration Dataset & Evaluation
    print("\n[3/4] Running CARLA Synthetic Temporal Deterioration Benchmark...")
    gen = CarlaTemporalSequenceGenerator(seed=42)
    train_seqs, test_seqs = gen.generate_dataset(num_segments=30, train_ratio=0.70)
    
    predictor = DeteriorationPredictor()
    pred_evaluator = PredictionEvaluator()

    true_dets = [s.ground_truth_deteriorated for s in test_seqs]
    true_pots = [s.ground_truth_pothole_formed for s in test_seqs]
    sim_trends = [s.progression_type for s in test_seqs]

    pred_dets = []
    pred_pots = []
    pred_trends = []

    for s in test_seqs:
        p_res = predictor.predict(s.road_segment_id, s.inspections[:3], horizon_days=30)
        pred_dets.append(p_res.deterioration_probability)
        pred_pots.append(p_res.pothole_formation_probability)
        pred_trends.append(p_res.progression_trend)

    pred_report = pred_evaluator.evaluate(
        true_deteriorated=true_dets,
        pred_deterioration_probs=pred_dets,
        true_pothole_formed=true_pots,
        pred_pothole_probs=pred_pots,
        simulated_progression_trends=sim_trends,
        predicted_progression_trends=pred_trends,
    )

    print(" Temporal Deterioration Prediction Performance (Held-out Test Segments):")
    print(f"   Deterioration Detection Precision / Recall / F1: {pred_report.deterioration_precision:.2f} / {pred_report.deterioration_recall:.2f} / {pred_report.deterioration_f1:.2f}")
    print(f"   Deterioration Brier Score: {pred_report.deterioration_brier_score:.4f}")
    print(f"   Pothole Formation Precision / Recall / F1: {pred_report.pothole_precision:.2f} / {pred_report.pothole_recall:.2f} / {pred_report.pothole_f1:.2f}")
    print(f"   Directional Progression Agreement: {pred_report.directional_trend_agreement_pct:.1f}%")

    elapsed = time.time() - start_time
    print(f"\n[4/4] Pipeline evaluation complete in {elapsed:.3f} seconds.")
    print("=================================================================\n")


if __name__ == "__main__":
    run_full_demo()
