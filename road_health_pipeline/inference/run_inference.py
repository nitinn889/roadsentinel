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
    masker: RoadMasker
    embedder: Dinov2Embedder
    detector: AnomalyDetector
    localizer: PotholeLocalizer
    depth_estimator: NullDepthEstimator


def load_pipeline(
    device: str = CONFIG.device,
    memory_bank_dir: Path = CONFIG.memory_bank_dir,
) -> PipelineComponents:
    """Load all pipeline models once and return a reusable container.

    Parameters
    ----------
    device:
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    memory_bank_dir:
        Path to the directory produced by ``build_memory_bank.py``.

    Returns
    -------
    PipelineComponents
    """
    log.info("Loading SAM2 on %s …", device)
    masker = RoadMasker(device=device)

    log.info("Loading DINOv2 on %s …", device)
    embedder = Dinov2Embedder.from_config(device=device)

    log.info("Loading memory bank from %s …", memory_bank_dir)
    detector = AnomalyDetector(memory_bank_dir)

    return PipelineComponents(
        masker=masker,
        embedder=embedder,
        detector=detector,
        localizer=PotholeLocalizer(),
        depth_estimator=NullDepthEstimator(),
    )


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

def water_heuristic(rgb: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:
    """Low-risk RGB heuristic; not a trained water classifier."""
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


from analytics.severity import calculate_defect_severity
from analytics.road_health import calculate_road_health_score
from analytics.segment_aggregator import generate_spatial_segment_id
from analytics.prediction import RoadDeteriorationPredictor, SegmentObservation


def water_heuristic(rgb: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:
    """Low-risk RGB heuristic; not a trained water classifier."""
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
    potholes: list[PotholeRecord],
    road_health: Any,
) -> dict[str, np.ndarray]:
    """Generate visual overlay images for detection, severity, and road health."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1. Detection Overlay
    det_overlay = bgr.copy()
    for p in potholes:
        x1, y1, x2, y2 = p.bbox_xyxy
        color = (0, 165, 255) if p.water_flag else (0, 0, 255)
        cv2.rectangle(det_overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{p.defect_type} ({p.pothole_confidence:.2f})"
        cv2.putText(det_overlay, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 2. Severity Overlay
    sev_overlay = bgr.copy()
    for p in potholes:
        x1, y1, x2, y2 = p.bbox_xyxy
        sev = p.severity_score
        # Color code: green (low) -> yellow (med) -> orange (high) -> red (critical)
        if sev >= 0.85:
            color = (0, 0, 255)      # Red
        elif sev >= 0.65:
            color = (0, 140, 255)    # Orange
        elif sev >= 0.35:
            color = (0, 255, 255)    # Yellow
        else:
            color = (0, 255, 0)      # Green
        cv2.rectangle(sev_overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(sev_overlay, f"Sev: {sev:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 3. Road Health Overlay
    health_overlay = bgr.copy()
    score = road_health.road_health_score if road_health else 100.0
    cond = road_health.condition_class.upper() if road_health else "GOOD"
    h_color = (0, 255, 0) if score >= 80 else ((0, 255, 255) if score >= 60 else ((0, 140, 255) if score >= 40 else (0, 0, 255)))

    # Banner on top
    cv2.rectangle(health_overlay, (10, 10), (360, 60), (0, 0, 0), -1)
    cv2.putText(health_overlay, f"Road Health Score: {score:.1f}/100", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(health_overlay, f"Condition: {cond}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 1, cv2.LINE_AA)

    return {
        "detection_overlay": det_overlay,
        "severity_overlay": sev_overlay,
        "road_health_overlay": health_overlay,
    }


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer(
    image_path: Path,
    metadata_path: Optional[Path] = None,
    device: str = CONFIG.device,
    memory_bank_dir: Path = CONFIG.memory_bank_dir,
    pipeline: Optional[PipelineComponents] = None,
    road_segment_id: Optional[str] = None,
) -> InferenceResult:
    """Run the full RoadSentinel pipeline on a single image.

    Parameters
    ----------
    image_path:
        Path to the input RGB image.
    metadata_path:
        Optional JSON file with GPS/telemetry metadata.
    device:
        Torch device (ignored if ``pipeline`` is provided).
    memory_bank_dir:
        Memory bank directory (ignored if ``pipeline`` is provided).
    pipeline:
        Pre-loaded ``PipelineComponents``.
    road_segment_id:
        Optional identifier for the road segment.

    Returns
    -------
    InferenceResult
    """
    if pipeline is None:
        pipeline = load_pipeline(device=device, memory_bank_dir=memory_bank_dir)

    rgb = load_rgb(image_path)
    telemetry = (
        telemetry_from_dict(load_json(metadata_path))
        if metadata_path and metadata_path.exists()
        else telemetry_from_dict({"timestamp": utc_iso()})
    )
    telemetry = GPSLocalizer().attach(telemetry)

    seg_id = road_segment_id or generate_spatial_segment_id(telemetry.latitude, telemetry.longitude)

    # Step 1: Road mask
    road_mask = pipeline.masker.get_road_mask(rgb)

    # Step 2: DINOv2 patch embeddings (road pixels only)
    embeddings, coords = pipeline.embedder.extract_road_patch_embeddings(rgb, road_mask)

        # 2. Depth estimation
        depth_map = self.depth_estimator.estimate(rgb_image)
        depth_source = self.depth_estimator.name

        detections: List[DefectMeasurement] = []
        legacy_potholes: List[PotholeRecord] = []
    # Per-image threshold
    road_scores = patch_scores if len(patch_scores) else np.array([0.0])
    threshold_px = float(np.percentile(road_scores, CONFIG.anomaly_percentile))

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
    # Step 6: Build output records with severity breakdown
    records: list[PotholeRecord] = []
    warnings: list[str] = []

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
    for i, c in enumerate(candidates):
        m = c.mask
        area = estimate_area_m2(m, telemetry.altitude_m)
        depth_m: Optional[float] = None
        if depth is not None:
            vals = depth[m]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                depth_m = float(np.percentile(vals, 90) - np.percentile(vals, 10))

        water, water_conf = water_heuristic(rgb, m)
        conf = c.pothole_confidence
        
        # Calculate transparent severity
        sev_res = calculate_defect_severity(
            confidence=conf,
            area_m2=area,
            depth_m=depth_m,
            is_water_filled=water,
            water_confidence=water_conf,
            surrounding_damage=c.surrounding_damage,
            shape_circularity=c.shape_circularity,
        )

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
                severity_score=sev_res.severity_score,
                water_flag=water,
                water_confidence=water_conf,
                source_image=str(image_path),
                mask_area_px=int(m.sum()),
                bbox_xyxy=c.bbox_xyxy,
                defect_type=c.defect_type,
                road_segment_id=seg_id,
                crack_or_damage_extent=round(c.surrounding_damage, 3),
                shape_circularity=round(c.shape_circularity, 3),
                aspect_ratio=round(c.aspect_ratio, 2),
                severity_breakdown=sev_res.severity_components,
                depth_source=pipeline.depth_estimator.name,
                notes=[
                    "Post-FAISS tuned defect candidate with explainable severity breakdown."
                ],
            )
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
        image_path=str(image_path),
        timestamp=telemetry.timestamp,
        frame_id=telemetry.frame_id,
        telemetry=telemetry.__dict__,
        image_shape=list(rgb.shape),
        anomaly_threshold=threshold_px,
        anomaly_score=image_score,
        potholes=records,
        road_segment_id=seg_id,
        road_health=road_health,
        prediction=prediction,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run RoadSentinel inference on a single image."
    )
    ap.add_argument("image", type=Path, help="Input RGB image path")
    ap.add_argument("--metadata", type=Path, default=None, help="Telemetry JSON")
    ap.add_argument("--device", default=CONFIG.device, help="Torch device (cuda/cpu)")
    ap.add_argument(
        "--memory-bank",
        type=Path,
        default=CONFIG.memory_bank_dir,
        help="Memory bank directory",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=CONFIG.output_dir / "inference.json",
        help="Output JSON path",
    )
    ap.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save detection, severity, and road health visualization overlays",
    )
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
if __name__ == "__main__":
    main()

