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


def severity(
    conf: float,
    area_m2: Optional[float],
    depth_m: Optional[float],
    water: bool,
) -> float:
    area_score = 0.0 if area_m2 is None else float(np.clip(area_m2 / 2.0, 0, 1))
    depth_score = 0.0 if depth_m is None else float(np.clip(depth_m / 0.15, 0, 1))
    s = 0.60 * conf + 0.25 * area_score + 0.15 * depth_score
    if water:
        s = max(s, min(1.0, s + 0.15))
    return float(np.clip(s, 0, 1))


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer(
    image_path: Path,
    metadata_path: Optional[Path] = None,
    device: str = CONFIG.device,
    memory_bank_dir: Path = CONFIG.memory_bank_dir,
    pipeline: Optional[PipelineComponents] = None,
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
        Pre-loaded ``PipelineComponents``.  If None, models are loaded fresh
        from ``device`` and ``memory_bank_dir``.  For batch inference, load
        once with ``load_pipeline()`` and pass here.

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

    # Per-image threshold (use patch scores from road region, not whole image)
    road_scores = patch_scores if len(patch_scores) else np.array([0.0])
    threshold_px = float(np.percentile(road_scores, CONFIG.anomaly_percentile))

    # Step 4: Candidate localisation + SAM2 refinement
    candidates: list[CandidateRegion] = pipeline.localizer.localize(
        rgb, amap, road_mask, threshold_px, sam2=pipeline.masker
    )

    # Step 5: Depth estimation
    depth = pipeline.depth_estimator.estimate(rgb)

    # Step 6: Build output records
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
                # Depth range within the mask as a proxy for pothole depth.
                # This is NOT a validated metric; it requires a ground-plane model.
                depth_m = float(np.percentile(vals, 90) - np.percentile(vals, 10))
        water, water_conf = water_heuristic(rgb, m)
        conf = c.pothole_confidence
        sev = severity(conf, area, depth_m, water)

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
                severity_score=sev,
                water_flag=water,
                water_confidence=water_conf,
                source_image=str(image_path),
                mask_area_px=int(m.sum()),
                bbox_xyxy=c.bbox_xyxy,
                depth_source=pipeline.depth_estimator.name,
                notes=[
                    "Heuristic pothole candidate; generic anomaly detection is not "
                    "a trained pothole classifier."
                ],
            )
        )

    return InferenceResult(
        image_path=str(image_path),
        timestamp=telemetry.timestamp,
        frame_id=telemetry.frame_id,
        telemetry=telemetry.__dict__,
        image_shape=list(rgb.shape),
        anomaly_threshold=threshold_px,
        anomaly_score=image_score,
        potholes=records,
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
    args = ap.parse_args()

    # Load pipeline once (important: no reload per image in CLI mode either)
    p = load_pipeline(device=args.device, memory_bank_dir=args.memory_bank)
    result = infer(args.image, args.metadata, pipeline=p)
    save_json(result.to_dict(), args.output)
    print(result.to_json())


if __name__ == "__main__":
    main()
