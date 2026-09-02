"""End-to-End Analytics, Road-Health Scoring, and Deterioration Demo for RoadSentinel.

Executes the complete post-inference analytics layer:
1. Generates mock/real road images with defects
2. Computes DINOv2 Anomaly Map + Defect Localisation
3. Calculates Multi-factor Severity Breakdown
4. Computes 0-100 Segment-level Road Health Score
5. Aggregates multi-inspection observations by Road Segment ID
6. Executes Temporal Deterioration & Pothole Prediction
7. Evaluates against CARLA Simulation Ground Truth
8. Exports JSON reports and visual overlays (detection, severity, health score).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.schemas import PotholeRecord, Telemetry
from common.io_utils import save_json, utc_iso
from analytics.severity import calculate_defect_severity
from analytics.road_health import calculate_road_health_score
from analytics.segment_aggregator import RoadSegmentAggregator
from analytics.prediction import RoadDeteriorationPredictor, SegmentObservation
from analytics.temporal_generator import generate_synthetic_segment_sequence
from evaluation.carla_eval import CarlaDefectGroundTruth, evaluate_carla_ground_truth
from evaluation.prediction_eval import evaluate_prediction_model
from inference.run_inference import generate_visual_overlays

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_analytics_e2e")


def create_demo_road_scene() -> tuple[np.ndarray, list[PotholeRecord], list[CarlaDefectGroundTruth]]:
    """Synthesize a realistic asphalt road scene with 2 distinct potholes for demo."""
    h, w = 720, 1280
    # Asphalt grey texture
    np.random.seed(42)
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Road lane markings (white)
    cv2.line(rgb, (w // 2, 0), (w // 2, h), (230, 230, 230), 8)

    # Defect 1: Medium Dry Pothole at (400, 450)
    cv2.circle(rgb, (400, 450), 45, (45, 45, 45), -1)
    cv2.circle(rgb, (400, 450), 50, (30, 30, 30), 3)

    # Defect 2: Severe Water-filled Pothole at (850, 520)
    cv2.ellipse(rgb, (850, 520), (75, 40), 20, 0, 360, (20, 25, 30), -1)
    cv2.ellipse(rgb, (850, 520), (78, 43), 20, 0, 360, (15, 20, 25), 4)

    timestamp = utc_iso()
    seg_id = "seg_carla_town10_0042"

    # Defect 1 record
    sev1 = calculate_defect_severity(
        confidence=0.88,
        area_m2=0.45,
        depth_m=0.08,
        is_water_filled=False,
        water_confidence=0.12,
        surrounding_damage=0.30,
        shape_circularity=0.85,
    )
    rec1 = PotholeRecord(
        pothole_id="demo-pothole-001",
        timestamp=timestamp,
        latitude=13.0827,
        longitude=80.2707,
        altitude_m=30.0,
        area_m2=0.45,
        estimated_depth_m=0.08,
        anomaly_score=0.82,
        pothole_confidence=0.88,
        severity_score=sev1.severity_score,
        water_flag=False,
        water_confidence=0.12,
        source_image="demo_scene.jpg",
        mask_area_px=6360,
        bbox_xyxy=[350, 400, 450, 500],
        defect_type="pothole",
        road_segment_id=seg_id,
        crack_or_damage_extent=0.30,
        shape_circularity=0.85,
        aspect_ratio=1.0,
        severity_breakdown=sev1.severity_components,
        depth_source="carla_synthetic",
    )

    # Defect 2 record (Water-filled)
    sev2 = calculate_defect_severity(
        confidence=0.92,
        area_m2=1.10,
        depth_m=0.14,
        is_water_filled=True,
        water_confidence=0.85,
        surrounding_damage=0.55,
        shape_circularity=0.62,
    )
    rec2 = PotholeRecord(
        pothole_id="demo-pothole-002",
        timestamp=timestamp,
        latitude=13.0831,
        longitude=80.2712,
        altitude_m=30.0,
        area_m2=1.10,
        estimated_depth_m=0.14,
        anomaly_score=0.94,
        pothole_confidence=0.92,
        severity_score=sev2.severity_score,
        water_flag=True,
        water_confidence=0.85,
        source_image="demo_scene.jpg",
        mask_area_px=10050,
        bbox_xyxy=[770, 475, 930, 565],
        defect_type="water_filled_pothole",
        road_segment_id=seg_id,
        crack_or_damage_extent=0.55,
        shape_circularity=0.62,
        aspect_ratio=1.85,
        severity_breakdown=sev2.severity_components,
        depth_source="carla_synthetic",
    )

    # CARLA Ground Truth targets
    gt1 = CarlaDefectGroundTruth(
        defect_id="gt-001",
        true_area_m2=0.48,
        true_depth_m=0.085,
        true_world_x=12.5,
        true_world_y=45.0,
        is_water_filled=False,
        true_severity=0.52,
    )
    gt2 = CarlaDefectGroundTruth(
        defect_id="gt-002",
        true_area_m2=1.15,
        true_depth_m=0.150,
        true_world_x=34.2,
        true_world_y=58.0,
        is_water_filled=True,
        true_severity=0.86,
    )

    return rgb, [rec1, rec2], [gt1, gt2]


def main():
    parser = argparse.ArgumentParser(description="Run RoadSentinel Analytics E2E Demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "analytics_demo",
        help="Directory to save demonstration artifacts",
    )
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("RoadSentinel Analytics Layer & Prediction E2E Pipeline")
    log.info("=" * 65)

    # 1. Generate demo road scene and detections
    log.info("Generating synthetic road scene and defect detections...")
    rgb, potholes, ground_truths = create_demo_road_scene()
    cv2.imwrite(str(out_dir / "input_frame.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # 2. Compute Segment-Level Road Health Score
    log.info("Computing 0-100 road health score...")
    health_result = calculate_road_health_score(potholes=potholes, surface_anomaly_mean=0.15)
    log.info("  Road Health Score: %.2f / 100 (%s)", health_result.road_health_score, health_result.condition_class.upper())
    for k, v in health_result.components.items():
        log.info("    %-28s: %.2f", k, v)

    # 3. Aggregate Road Segment Summary
    log.info("Aggregating detections into Road Segment Summary...")
    aggregator = RoadSegmentAggregator()
    summary = aggregator.aggregate_records(
        records=potholes,
        segment_id="seg_carla_town10_0042",
        timestamp=utc_iso(),
        latitude=13.0829,
        longitude=80.2709,
    )

    # 4. Generate Visual Overlays
    log.info("Rendering visualization overlays...")
    overlays = generate_visual_overlays(rgb, potholes, health_result)
    cv2.imwrite(str(out_dir / "detection_overlay.jpg"), overlays["detection_overlay"])
    cv2.imwrite(str(out_dir / "severity_overlay.jpg"), overlays["severity_overlay"])
    cv2.imwrite(str(out_dir / "road_health_overlay.jpg"), overlays["road_health_overlay"])
    log.info("  Saved overlays to: %s", out_dir)

    # 5. CARLA Ground-Truth Evaluation
    log.info("Evaluating detections against CARLA simulation ground truth...")
    gt_eval = evaluate_carla_ground_truth(potholes, ground_truths)
    save_json(gt_eval, out_dir / "carla_evaluation_report.json")
    log.info("  CARLA Ground Truth Match F1: %.3f", gt_eval["detection_f1"])
    log.info("  Area MAE                   : %.4f m^2", gt_eval["area_mae_m2"])
    log.info("  Depth MAE                  : %.4f m", gt_eval["depth_mae_m"])
    log.info("  Water Classification F1    : %.3f", gt_eval["water_f1"])

    # 6. Temporal Deterioration Prediction
    log.info("Running temporal deterioration prediction over 30-day horizon...")
    seq = generate_synthetic_segment_sequence("seg_carla_town10_0042", progression_type="accelerated_failure", num_timesteps=4)
    predictor = RoadDeteriorationPredictor(horizon_days=30)
    pred_result = predictor.predict(seq.observations, road_segment_id="seg_carla_town10_0042")
    summary.prediction = pred_result

    log.info("  Deterioration Probability    : %.3f", pred_result.deterioration_probability)
    log.info("  Pothole Formation Probability: %.3f", pred_result.pothole_formation_probability)
    log.info("  Progression Direction        : %s", pred_result.progression_direction.upper())

    # Save summary JSON
    save_json(summary.to_dict(), out_dir / "result.json")
    log.info("Full results exported to: %s", out_dir / "result.json")
    log.info("=" * 65)
    log.info("E2E Analytics Pipeline Execution Complete!")


if __name__ == "__main__":
    main()
