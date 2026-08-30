<<<<<<< HEAD
"""End-to-end RoadSentinel inference and analytics pipeline.

Usage (CLI)
-----------
python inference/run_inference.py path/to/image.jpg \\
    --device cuda \\
    --memory-bank output/memory_bank \\
    --output output/result.json \\
    --save-overlays

Usage (Python API — model reuse across images)
----------------------------------------------
from inference.run_inference import load_pipeline, infer

# Load all models once
pipeline = load_pipeline(device="cuda")

# Process many images reusing the same models
for path in image_list:
    result = infer(path, pipeline=pipeline)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG, Config
=======
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import cv2
import numpy as np

>>>>>>> 0e75352 (Add road health scoring and deterioration prediction)
from common.schemas import (
    CandidateRegion,
    DefectMeasurement,
    DefectType,
    InferenceResult,
    PotholeRecord,
    PredictionResult,
<<<<<<< HEAD
    RoadHealthResult,
    RoadHealthScore,
    RoadSegmentAggregate,
    SegmentationResult,
    SeverityBreakdown,
    SeverityResult,
    Telemetry,
)
from common.io_utils import load_rgb, load_json, save_json, utc_iso

from analytics.severity import calculate_defect_severity, classify_severity_label
from analytics.road_health import calculate_road_health_score
from analytics.segment_aggregator import generate_spatial_segment_id
from analytics.prediction import RoadDeteriorationPredictor, SegmentObservation

from inference.anomaly_detector import AnomalyDetector
from inference.area_estimator import estimate_area_m2
from inference.defect_classifier import DefectClassifier
from inference.depth_estimator import DepthEstimator, NullDepthEstimator
from inference.dinov2_embed import Dinov2Embedder
from inference.gps_localizer import GPSLocalizer, telemetry_from_dict
from inference.low_light_enhancer import LowLightEnhancer
from inference.pothole_localizer import PotholeLocalizer
from inference.road_health_scorer import RoadHealthScorer
from inference.sam2_mask import RoadMasker
from inference.segment_aggregator import SegmentAggregator
from inference.severity_estimator import SeverityEstimator
from inference.road_marking_suppressor import RoadMarkingSuppressor
from inference.vehicle_suppressor import VehicleSuppressor
from inference.visualizer import PipelineVisualizer
from prediction.progression_model import DeteriorationPredictor, TemporalInspectionRecord

log = logging.getLogger("run_inference")


# ---------------------------------------------------------------------------
# Pipeline container
# ---------------------------------------------------------------------------

@dataclass
class PipelineComponents:
    """Pre-loaded model container for batch inference."""
    masker: RoadMasker
    embedder: Dinov2Embedder
    detector: AnomalyDetector
    localizer: PotholeLocalizer
    depth_estimator: NullDepthEstimator
    vehicle_suppressor: VehicleSuppressor = field(default_factory=VehicleSuppressor)
    marking_suppressor: RoadMarkingSuppressor = field(default_factory=RoadMarkingSuppressor)
    low_light_enhancer: LowLightEnhancer = field(default_factory=LowLightEnhancer)
    crater_head: Optional[Any] = None
=======
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
>>>>>>> 0e75352 (Add road health scoring and deterioration prediction)

    def process_candidates(self,
                           rgb_image: np.ndarray,
                           candidates: List[CandidateRegion],
                           telemetry: Telemetry,
                           road_segment_id: Optional[str] = None,
                           history: Optional[List[TemporalInspectionRecord]] = None
                           ) -> InferenceResult:
        """Processes candidate regions from DINOv2+SAM2 through the analytics layer.

<<<<<<< HEAD
def load_pipeline(
    device: str = CONFIG.device,
    memory_bank_dir: Path = CONFIG.memory_bank_dir,
) -> PipelineComponents:
    """Load all pipeline models once and return a reusable container."""
    mb_dir = Path(memory_bank_dir)
    if not mb_dir.exists() or not (mb_dir / "index.faiss").exists():
        real_mb = ROOT / "output" / "real_memory_bank"
        if real_mb.exists() and (real_mb / "index.faiss").exists():
            mb_dir = real_mb

    log.info("Loading SAM2 on %s …", device)
    masker = RoadMasker(device=device)

    log.info("Loading DINOv2 on %s …", device)
    embedder = Dinov2Embedder.from_config(device=device)

    log.info("Loading memory bank from %s …", mb_dir)
    detector = AnomalyDetector(mb_dir)

    # Optional Fine-Tuned Crater Head Checkpoint
    crater_head = None
    head_ckpt = ROOT / "checkpoints" / "dinov2_crater_head.pt"
    if head_ckpt.is_file():
        try:
            import torch
            from training.finetune_dinov2_crater import CraterAnomalyHead
            c_head = CraterAnomalyHead(embed_dim=384, hidden_dim=256)
            ckpt_data = torch.load(head_ckpt, map_location=device, weights_only=False)
            c_head.load_state_dict(ckpt_data["head_state_dict"])
            c_head.to(device)
            c_head.eval()
            crater_head = c_head
            log.info("Loaded fine-tuned DINOv2 Crater Anomaly Head from %s (F1: %.3f)", head_ckpt.name, ckpt_data.get("f1_score", 0.99))
        except Exception as e:
            log.warning("Could not load crater head checkpoint: %s", e)

    return PipelineComponents(
        masker=masker,
        embedder=embedder,
        detector=detector,
        localizer=PotholeLocalizer(),
        depth_estimator=NullDepthEstimator(),
        vehicle_suppressor=VehicleSuppressor(),
        marking_suppressor=RoadMarkingSuppressor(),
        low_light_enhancer=LowLightEnhancer(),
        crater_head=crater_head,
    )


# ---------------------------------------------------------------------------
# Heuristic helpers & Visual Overlays
# ---------------------------------------------------------------------------

def water_heuristic(rgb: np.ndarray, mask: np.ndarray) -> Tuple[bool, float]:
    """RGB-based texture/intensity water heuristic."""
    if mask.sum() < 50:
        return False, 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    region = hsv[mask]
    val = region[:, 2].astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    local_std = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    low_texture = float(np.clip(1.0 - local_std / 1500.0, 0, 1))
    dark = float(np.clip(1.0 - val.mean() / 180.0, 0, 1))
    score = float(np.clip(0.55 * low_texture + 0.45 * dark, 0, 1))
    return score >= 0.70, score


def generate_visual_overlays(
    rgb: np.ndarray,
    potholes: List[PotholeRecord],
    road_health: Any,
    candidate_masks: Optional[List[np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Generate visual overlay images for detection, SAM2 segmentation, severity, and road health."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    # 1. Detection Overlay with SAM2 Segmentation Mask
    det_overlay = bgr.copy()
    mask_colored = np.zeros_like(bgr)

    for i, p in enumerate(potholes):
        x1, y1, x2, y2 = [int(v) for v in p.bbox_xyxy]
        is_water = getattr(p, "water_flag", False)
        # BGR: Orange for dry pothole/crater, Sky Blue for water-filled
        color = (0, 165, 255) if is_water else (0, 70, 255)
        mask_color = (0, 215, 255) if is_water else (30, 144, 255)

        # Draw SAM2 segmentation mask if available
        if candidate_masks and i < len(candidate_masks):
            cm = candidate_masks[i]
            if cm is not None and np.any(cm):
                mask_colored[cm] = mask_color
                contours, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(det_overlay, contours, -1, color, 2)

        # Bounding box
        cv2.rectangle(det_overlay, (x1, y1), (x2, y2), color, 2)
        
        # Label & confidence badge
        area_label = f"{p.area_m2:.2f}m²" if (p.area_m2 is not None and p.area_m2 > 0) else f"{p.mask_area_px}px"
        water_tag = " [WATER]" if is_water else ""
        label = f"{p.defect_type.upper()}{water_tag} ({p.pothole_confidence:.2f}) | {area_label}"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(det_overlay, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(0, y1)), color, -1)
        cv2.putText(det_overlay, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend SAM2 masks onto detection overlay
    if np.any(mask_colored):
        mask_bool = np.any(mask_colored > 0, axis=-1)
        det_overlay[mask_bool] = cv2.addWeighted(det_overlay, 0.55, mask_colored, 0.45, 0)[mask_bool]

    # 2. Severity Overlay
    sev_overlay = bgr.copy()
    for p in potholes:
        x1, y1, x2, y2 = [int(v) for v in p.bbox_xyxy]
        sev = p.severity_score
        # Continuous or normalized severity color mapping
        if sev >= 0.85 or (isinstance(sev, float) and sev > 75.0):
            color = (0, 0, 255)      # Red / Critical
            sev_label = "CRITICAL"
        elif sev >= 0.65 or (isinstance(sev, float) and sev > 50.0):
            color = (0, 140, 255)    # Orange / High
            sev_label = "HIGH"
        elif sev >= 0.35 or (isinstance(sev, float) and sev > 25.0):
            color = (0, 255, 255)    # Yellow / Medium
            sev_label = "MEDIUM"
        else:
            color = (0, 255, 0)      # Green / Low
            sev_label = "LOW"

        cv2.rectangle(sev_overlay, (x1, y1), (x2, y2), color, 3)
        tag = f"SEV: {sev_label} ({sev:.2f})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(sev_overlay, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(0, y1)), color, -1)
        text_color = (0, 0, 0) if sev_label in ("LOW", "MEDIUM") else (255, 255, 255)
        cv2.putText(sev_overlay, tag, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # 3. Road Health Overlay
    health_overlay = bgr.copy()
    score = getattr(road_health, "road_health_score", 100.0) if road_health else 100.0
    cond = getattr(road_health, "condition_class", "GOOD").upper() if road_health else "GOOD"
    h_color = (0, 255, 0) if score >= 80 else ((0, 255, 255) if score >= 60 else ((0, 140, 255) if score >= 40 else (0, 0, 255)))

    cv2.rectangle(health_overlay, (10, 10), (360, 60), (0, 0, 0), -1)
    cv2.putText(health_overlay, f"Road Health Score: {score:.1f}/100", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(health_overlay, f"Condition: {cond}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 1, cv2.LINE_AA)

    return {
        "detection_overlay": det_overlay,
        "severity_overlay": sev_overlay,
        "road_health_overlay": health_overlay,
    }


# ---------------------------------------------------------------------------
# High-level Pipeline Class
# ---------------------------------------------------------------------------

class RoadSentinelPipeline:
    """Complete analytics pipeline processing SAM2 masks into physical measurements,
    explainable severity, segment-level road health scores, and temporal deterioration predictions.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        depth_estimator: Optional[DepthEstimator] = None,
        localizer: Optional[GPSLocalizer] = None,
    ):
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

    def process_candidates(
        self,
        rgb_image: np.ndarray,
        candidates: List[CandidateRegion],
        telemetry: Telemetry,
        road_segment_id: Optional[str] = None,
        history: Optional[List[TemporalInspectionRecord]] = None,
    ) -> InferenceResult:
        """Processes candidate regions through physical measurements, severity, health score, and prediction."""
        # 1. Attach / resolve GPS
        telemetry = self.localizer.attach(telemetry)
        h, w = rgb_image.shape[:2]

=======
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

>>>>>>> 0e75352 (Add road health scoring and deterioration prediction)
        # Determine segment ID
        seg_id = road_segment_id or self.segment_aggregator.generate_segment_id(
            latitude=telemetry.latitude,
            longitude=telemetry.longitude,
            world_x=telemetry.world_x,
            world_y=telemetry.world_y,
        )

        # 2. Depth estimation
        depth_map = self.depth_estimator.estimate(rgb_image)
<<<<<<< HEAD
        depth_source = getattr(self.depth_estimator, "name", "unavailable")

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
            if defect_type == DefectType.CRACK.value or defect_type == "crack":
                crack_extent = area_m2 * 2.0 if area_m2 is not None else float(morph.get("perimeter", 0.0)) * 0.001

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
                source_image=str(getattr(telemetry, "frame_id", "") or "frame"),
                mask_area_px=mask_px,
                bbox_xyxy=bbox,
                defect_type=defect_type,
                road_segment_id=seg_id,
                crack_or_damage_extent=crack_extent,
                severity_breakdown=severity.severity_components,
                depth_source=depth_source,
            ))

        # 5. Road Health Score
        road_health = self.health_scorer.calculate_health(detections)

        # 6. Temporal Deterioration Prediction
        hist_records = list(history or [])
        total_damaged_area = sum(d.estimated_area_m2 or 0.0 for d in detections)
        pothole_count = sum(1 for d in detections if "pothole" in d.defect_type)
        crack_count = sum(1 for d in detections if d.defect_type in ("crack", DefectType.CRACK.value, DefectType.CRACK_OR_DAMAGE.value))
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
            image_path="",
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
            potholes=legacy_potholes,
            road_health=road_health,
            prediction=prediction,
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

    def generate_diagnostic_overlays(
        self,
        rgb_image: np.ndarray,
        result: InferenceResult,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Renders and saves detection_overlay.jpg, severity_overlay.jpg, and road_health_overlay.jpg."""
        output_dir.mkdir(parents=True, exist_ok=True)

        det_overlay = self.visualizer.draw_detection_overlay(rgb_image, result.detections)
        sev_overlay = self.visualizer.draw_severity_overlay(rgb_image, result.detections)
        health_overlay = self.visualizer.draw_road_health_overlay(
            rgb_image=rgb_image,
            road_health=result.road_health,
            road_segment_id=result.road_segment_id or "segment_0",
            prediction=result.prediction,
        )

        det_path = output_dir / "detection_overlay.jpg"
        sev_path = output_dir / "severity_overlay.jpg"
        health_path = output_dir / "road_health_overlay.jpg"
        json_path = output_dir / "result.json"

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


# ---------------------------------------------------------------------------
# Main Inference Entry Point (Functional API)
# ---------------------------------------------------------------------------

def infer(
    image_path: Path,
    metadata_path: Optional[Path] = None,
    device: str = CONFIG.device,
    memory_bank_dir: Path = CONFIG.memory_bank_dir,
    pipeline: Optional[PipelineComponents] = None,
    road_segment_id: Optional[str] = None,
    test_mode_2d: Optional[bool] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> InferenceResult:
    """Run inference; optionally expose bounded intermediate diagnostics."""
    is_2d = test_mode_2d if test_mode_2d is not None else getattr(CONFIG, "test_mode_2d", True)

    if pipeline is None:
        pipeline = load_pipeline(device=device, memory_bank_dir=memory_bank_dir)

    rgb = load_rgb(image_path)
    
    if not is_2d:
        # 3D Mode: Resolve telemetry and georeference via GPS localizer
        telemetry = (
            telemetry_from_dict(load_json(metadata_path))
            if metadata_path and Path(metadata_path).exists()
            else telemetry_from_dict({"timestamp": utc_iso()})
        )
        telemetry = GPSLocalizer().attach(telemetry)
    else:
        # 2D Mode: Minimal telemetry without requiring 3D altitude or coordinates
        telemetry = Telemetry(
            timestamp=utc_iso(),
            altitude_m=None,
            latitude=None,
            longitude=None,
            frame_id=Path(image_path).stem,
        )

    seg_id = road_segment_id or (
        generate_spatial_segment_id(telemetry.latitude, telemetry.longitude)
        if not is_2d and telemetry.latitude is not None
        else "segment_2d_test"
    )

    # Step 0: Low-light photography enhancement
    enhanced_rgb, was_enhanced = pipeline.low_light_enhancer.enhance(rgb)
    if was_enhanced:
        log.info("Applied low-light adaptive CLAHE + gamma recovery to frame.")

    # Step 1: Road mask (fallback to drivable highway corridor if road prompt is undersized)
    road_mask = pipeline.masker.get_road_mask(enhanced_rgb)
    if road_mask is None or road_mask.sum() < 0.05 * rgb.shape[0] * rgb.shape[1]:
        road_mask = pipeline.masker._extract_road_corridor_fallback(enhanced_rgb)
    if diagnostics is not None:
        diagnostics["road_mask"] = road_mask.copy()
        diagnostics["road_mask_ratio"] = float(np.mean(road_mask))
        diagnostics["low_light_enhancement_applied"] = bool(was_enhanced)

    # Step 2: DINOv2 patch embeddings
    embeddings, coords = pipeline.embedder.extract_road_patch_embeddings(enhanced_rgb, road_mask)

    # Step 3: Domain-adaptive anomaly scoring & map.  The raw healthy-memory
    # distance is first re-centred against the *current road surface* (median
    # and IQR), which prevents CARLA lighting/material shifts from making an
    # entire frame look anomalous or, conversely, suppressing every defect.
    raw_patch_scores, patch_scores, domain_calibration = pipeline.detector.score_patches_adaptive(
        embeddings
    )
    anomaly_map = pipeline.detector.build_anomaly_map(
        coords, patch_scores, rgb.shape[:2], pipeline.embedder.grid_size
    )

    # Suppress vehicle false positives (cars, trucks, motorcycles, shadows)
    vehicle_mask = pipeline.vehicle_suppressor.get_vehicle_mask(rgb)
    if vehicle_mask is not None and np.any(vehicle_mask):
        anomaly_map[vehicle_mask] = 0.0
        log.info("Suppressed %d px of vehicle hulls and shadows from anomaly map.", int(np.sum(vehicle_mask)))

    # Suppress painted road markings (double yellow lines, white lane stripes, barriers)
    marking_mask = pipeline.marking_suppressor.get_marking_mask(enhanced_rgb)
    if marking_mask is not None and np.any(marking_mask):
        anomaly_map[marking_mask] = 0.0
        log.info("Suppressed %d px of painted road markings and roadside barriers.", int(np.sum(marking_mask)))

    # Single source of truth: percentile/clipping constants and empty-score
    # behavior are defined by AnomalyDetector.select_active_threshold().
    threshold_px, image_score, threshold_diagnostics = (
        pipeline.detector.select_active_threshold(patch_scores)
    )

    localization_diagnostics: Optional[dict] = {} if diagnostics is not None else None

    # Step 4: Candidate localization (passes is_2d flag)
    candidates = pipeline.localizer.localize(
        rgb=enhanced_rgb,
        anomaly_map=anomaly_map,
        road_mask=road_mask,
        threshold=threshold_px,
        sam2=pipeline.masker,
        test_mode_2d=is_2d,
        diagnostics=localization_diagnostics,
    )

    # Filter any residual candidates overlapping with vehicles or road markings
    if vehicle_mask is not None and np.any(vehicle_mask):
        candidates = pipeline.vehicle_suppressor.filter_candidate_boxes(candidates, vehicle_mask)
    if marking_mask is not None and np.any(marking_mask):
        candidates = pipeline.marking_suppressor.filter_candidate_regions(candidates, enhanced_rgb, marking_mask)

    if diagnostics is not None:
        diagnostics["raw_patch_scores"] = raw_patch_scores.copy()
        diagnostics["normalized_patch_scores"] = patch_scores.copy()
        diagnostics["normalization"] = dict(domain_calibration)
        diagnostics["threshold"] = threshold_px
        diagnostics["image_score"] = image_score
        diagnostics["threshold_selection"] = threshold_diagnostics
        diagnostics["anomaly_map"] = anomaly_map.copy()
        diagnostics["localization"] = localization_diagnostics or {}
        diagnostics["candidates"] = candidates

    # Step 5: Depth estimation (bypassed in 2D mode, active in 3D mode)
    if not is_2d:
        depth = pipeline.depth_estimator.estimate(rgb)
    else:
        depth = None

    # Step 6: Build output records
    records: List[PotholeRecord] = []
    detections: List[DefectMeasurement] = []
    warnings: List[str] = []

    for idx, c in enumerate(candidates):
        m = c.mask
        mask_px = int(m.sum())

        # Area & Depth logic: evaluate based on 2D pixel area vs 3D metric telemetry
        if not is_2d:
            area = estimate_area_m2(m, altitude_m=telemetry.altitude_m, horizontal_fov_deg=CONFIG.horizontal_fov_deg)
            depth_m: Optional[float] = None
            if depth is not None:
                vals = depth[m]
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals):
                    depth_m = float(np.percentile(vals, 90) - np.percentile(vals, 10))
        else:
            # 2D Test Mode: bypass altitude and depth constraints entirely
            area = None
            depth_m = None

        water, water_conf = water_heuristic(rgb, m)
        conf = c.pothole_confidence

        sev_res = calculate_defect_severity(
            confidence=conf,
            area_m2=area,
            depth_m=depth_m,
            is_water_filled=water,
            water_confidence=water_conf,
            surrounding_damage=getattr(c, "surrounding_damage", 0.0),
            shape_circularity=getattr(c, "shape_circularity", 0.0),
        )

        pothole_id = f"pothole_{idx:03d}"
        rec = PotholeRecord(
            pothole_id=pothole_id,
            timestamp=telemetry.timestamp,
            latitude=telemetry.latitude,
            longitude=telemetry.longitude,
            altitude_m=telemetry.altitude_m,
            area_m2=area,
            estimated_depth_m=depth_m,
            anomaly_score=c.anomaly_score,
            pothole_confidence=conf,
            severity_score=sev_res.severity_score,
            water_flag=water,
            water_confidence=water_conf,
            source_image=str(image_path),
            mask_area_px=mask_px,
            bbox_xyxy=c.bbox_xyxy,
            defect_type=getattr(c, "defect_type", DefectType.POTHOLE.value),
            road_segment_id=seg_id,
            crack_or_damage_extent=round(getattr(c, "surrounding_damage", 0.0), 3),
            shape_circularity=round(getattr(c, "shape_circularity", 0.0), 3),
            aspect_ratio=round(getattr(c, "aspect_ratio", 1.0), 2),
            severity_breakdown=sev_res.severity_components,
            depth_source="2D_bypassed" if is_2d else getattr(pipeline.depth_estimator, "name", "unavailable"),
            notes=["2D test mode detection with SAM2 mask overlay" if is_2d else "3D CARLA telemetry detection"],
        )
        records.append(rec)

        detections.append(DefectMeasurement(
            defect_id=pothole_id,
            defect_type=getattr(c, "defect_type", DefectType.POTHOLE.value),
            confidence=conf,
            bbox=c.bbox_xyxy,
            mask_area_pixels=mask_px,
            estimated_area_m2=area,
            estimated_depth_m=depth_m,
            is_water_filled=water,
            water_confidence=water_conf,
            crack_or_damage_extent=getattr(c, "surrounding_damage", None),
            road_segment_id=seg_id,
            timestamp=telemetry.timestamp,
            latitude=telemetry.latitude,
            longitude=telemetry.longitude,
            severity=SeverityBreakdown(
                severity=sev_res.severity,
                severity_score=sev_res.severity_score,
                severity_components=sev_res.severity_components,
            ),
            depth_source="2D_bypassed" if is_2d else getattr(pipeline.depth_estimator, "name", "unavailable"),
        ))

    # Step 7: Road Health Score calculation
    road_health = calculate_road_health_score(
        potholes=records,
        total_crack_area_m2=0.0,
        surface_anomaly_mean=float(np.mean(patch_scores)) if len(patch_scores) else 0.0,
    )

    # Step 8: Deterioration Prediction Interface
    predictor = RoadDeteriorationPredictor()
    obs = SegmentObservation(
        timestamp=telemetry.timestamp,
        road_health_score=road_health.road_health_score,
        pothole_count=len(records),
        total_defects=len(records),
        damaged_area_m2=float(sum(r.area_m2 for r in records if r.area_m2 is not None)),
        max_severity=float(max((r.severity_score for r in records), default=0.0)),
        avg_severity=float(np.mean([r.severity_score for r in records])) if records else 0.0,
        has_water_hazard=any(r.water_flag for r in records),
    )
    prediction = predictor.predict([obs], road_segment_id=seg_id)

    return InferenceResult(
        image_id=Path(image_path).stem if image_path else "frame",
        image_path=str(image_path),
        timestamp=telemetry.timestamp,
        frame_id=telemetry.frame_id,
        road_segment_id=seg_id,
        geolocation={
            "lat": telemetry.latitude,
            "lon": telemetry.longitude,
        },
        image_shape=list(rgb.shape),
        anomaly_threshold=threshold_px,
        anomaly_score=image_score,
        detections=detections,
        potholes=records,
        road_health=road_health,
        prediction=prediction,
        telemetry={
            "latitude": telemetry.latitude,
            "longitude": telemetry.longitude,
            "altitude_m": telemetry.altitude_m,
            "heading_deg": telemetry.heading_deg,
            "world_x": telemetry.world_x,
            "world_y": telemetry.world_y,
            "speed_mps": telemetry.speed_mps,
        },
        warnings=warnings + [
            "Domain-adaptive anomaly normalization: "
            f"baseline={domain_calibration.get('baseline_median', 0.0):.4f}, "
            f"scale={domain_calibration.get('scale', 0.0):.4f}, "
            f"threshold={threshold_px:.4f}"
        ],
    )
=======
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
>>>>>>> 0e75352 (Add road health scoring and deterioration prediction)

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

<<<<<<< HEAD
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Run RoadSentinel inference on a single image.")
    ap.add_argument("image", type=Path, help="Input RGB image path")
    ap.add_argument("--metadata", type=Path, default=None, help="Telemetry JSON")
    ap.add_argument("--device", default=CONFIG.device, help="Torch device (cuda/cpu)")
    ap.add_argument("--memory-bank", type=Path, default=CONFIG.memory_bank_dir, help="Memory bank directory")
    ap.add_argument("--output", type=Path, default=CONFIG.output_dir / "inference.json", help="Output JSON path")
    ap.add_argument("--save-overlays", action="store_true", help="Save visual overlays")
    args = ap.parse_args()

    p = load_pipeline(device=args.device, memory_bank_dir=args.memory_bank)
    result = infer(args.image, args.metadata, pipeline=p)
    save_json(result.to_dict(), args.output)

    if args.save_overlays:
        rgb = load_rgb(args.image)
        overlays = generate_visual_overlays(rgb, result.potholes, result.road_health)
        out_dir = args.output.parent if args.output else CONFIG.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "detection_overlay.jpg"), overlays["detection_overlay"])
        cv2.imwrite(str(out_dir / "severity_overlay.jpg"), overlays["severity_overlay"])
        cv2.imwrite(str(out_dir / "road_health_overlay.jpg"), overlays["road_health_overlay"])
        log.info("Saved visual overlays to %s", out_dir)

    print(result.to_json())
=======
        det_path = output_dir / "detection_overlay.jpg"
        sev_path = output_dir / "severity_overlay.jpg"
        health_path = output_dir / "road_health_overlay.jpg"
        json_path = output_dir / "result.json"

        # Save BGR for OpenCV
        cv2.imwrite(str(det_path), cv2.cvtColor(det_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sev_path), cv2.cvtColor(sev_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(health_path), cv2.cvtColor(health_overlay, cv2.COLOR_RGB2BGR))
>>>>>>> 0e75352 (Add road health scoring and deterioration prediction)

        with open(json_path, "w") as f:
            f.write(result.to_json(indent=2))

        return {
            "detection_overlay": det_path,
            "severity_overlay": sev_path,
            "road_health_overlay": health_path,
            "result_json": json_path,
        }
