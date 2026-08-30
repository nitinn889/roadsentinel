"""End-to-end RoadSentinel inference pipeline.

Usage (CLI)
-----------
python inference/run_inference.py path/to/image.jpg \\
    --device cuda \\
    --memory-bank output/memory_bank \\
    --output output/result.json

Usage (Python API — model reuse across images)
----------------------------------------------
from inference.run_inference import load_pipeline, infer

# Load all models once
pipeline = load_pipeline(device="cuda")

# Process many images reusing the same models
for path in image_list:
    result = infer(path, pipeline=pipeline)
    ...
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import load_rgb, load_json, save_json, utc_iso
from common.schemas import CandidateRegion, InferenceResult, PotholeRecord
from inference.sam2_mask import RoadMasker
from inference.dinov2_embed import Dinov2Embedder
from inference.anomaly_detector import AnomalyDetector
from inference.pothole_localizer import PotholeLocalizer
from inference.depth_estimator import NullDepthEstimator
from inference.area_estimator import estimate_area_m2
from inference.gps_localizer import GPSLocalizer, telemetry_from_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("run_inference")


# ---------------------------------------------------------------------------
# Pre-loaded pipeline components
# ---------------------------------------------------------------------------

@dataclass
class PipelineComponents:
    """Container for pre-loaded models.

    Pass this to ``infer()`` to avoid reloading models on every call.

    Attributes
    ----------
    masker:
        SAM2 road masker (also used for pothole refinement).
    embedder:
        DINOv2 feature extractor.
    detector:
        FAISS-based anomaly detector with a loaded memory bank.
    localizer:
        Connected-component pothole candidate localiser.
    depth_estimator:
        Depth estimator (defaults to NullDepthEstimator).
    """

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

    # Step 3: Anomaly scoring
    patch_scores = pipeline.detector.score_patches(embeddings)
    image_score, threshold = pipeline.detector.summarize(patch_scores)
    grid_size = CONFIG.dinov2_input_size // CONFIG.patch_size
    amap = pipeline.detector.build_anomaly_map(
        coords, patch_scores, rgb.shape[:2], grid_size
    )

    # Per-image threshold
    road_scores = patch_scores if len(patch_scores) else np.array([0.0])
    threshold_px = float(np.percentile(road_scores, CONFIG.anomaly_percentile))

    # Step 4: Candidate localisation + SAM2 refinement
    candidates: list[CandidateRegion] = pipeline.localizer.localize(
        rgb, amap, road_mask, threshold_px, sam2=pipeline.masker
    )

    # Step 5: Depth estimation
    depth = pipeline.depth_estimator.estimate(rgb)

    # Step 6: Build output records with severity breakdown
    records: list[PotholeRecord] = []
    warnings: list[str] = []

    if depth is None:
        warnings.append(
            "Metric RGB depth model was not provided; estimated_depth_m is null."
        )

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

        records.append(
            PotholeRecord(
                pothole_id=f"{telemetry.timestamp.replace(':', '').replace('-', '')}-{i:03d}",
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


if __name__ == "__main__":
    main()

