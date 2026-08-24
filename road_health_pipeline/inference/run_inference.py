from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import load_rgb, load_json, save_json, utc_iso
from common.schemas import InferenceResult, PotholeRecord
from inference.sam2_mask import RoadMasker
from inference.dinov2_embed import Dinov2Embedder
from inference.anomaly_detector import AnomalyDetector
from inference.pothole_localizer import PotholeLocalizer
from inference.depth_estimator import NullDepthEstimator
from inference.area_estimator import estimate_area_m2
from inference.gps_localizer import GPSLocalizer, telemetry_from_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_inference")


def water_heuristic(rgb: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:
    """Low-risk RGB heuristic; not a trained water classifier."""
    if mask.sum() < 50:
        return False, 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    region = hsv[mask]
    sat = region[:, 1].astype(np.float32)
    val = region[:, 2].astype(np.float32)
    # Water-like regions often have low texture and relatively dark/highly reflective appearance.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    local_std = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    low_texture = float(np.clip(1.0 - local_std / 1500.0, 0, 1))
    dark = float(np.clip(1.0 - val.mean() / 180.0, 0, 1))
    score = float(np.clip(0.55 * low_texture + 0.45 * dark, 0, 1))
    return score >= 0.70, score


def severity(conf: float, area_m2: float | None, depth_m: float | None, water: bool) -> float:
    area_score = 0.0 if area_m2 is None else float(np.clip(area_m2 / 2.0, 0, 1))
    depth_score = 0.0 if depth_m is None else float(np.clip(depth_m / 0.15, 0, 1))
    s = 0.60 * conf + 0.25 * area_score + 0.15 * depth_score
    if water:
        s = max(s, min(1.0, s + 0.15))
    return float(np.clip(s, 0, 1))


def infer(image_path: Path, metadata_path: Path | None = None, device: str = CONFIG.device,
          memory_bank_dir: Path = CONFIG.memory_bank_dir) -> InferenceResult:
    rgb = load_rgb(image_path)
    telemetry = telemetry_from_dict(load_json(metadata_path)) if metadata_path and metadata_path.exists() else telemetry_from_dict({"timestamp": utc_iso()})
    telemetry = GPSLocalizer().attach(telemetry)

    road_masker = RoadMasker(device=device)
    embedder = Dinov2Embedder(device=device)
    detector = AnomalyDetector(memory_bank_dir)
    localizer = PotholeLocalizer()
    depth_estimator = NullDepthEstimator()

    road_mask = road_masker.get_road_mask(rgb)
    embeddings, coords = embedder.extract_road_patch_embeddings(rgb, road_mask)
    patch_scores = detector.score_patches(embeddings)
    image_score, threshold = detector.summarize(patch_scores)
    grid_size = CONFIG.dinov2_input_size // CONFIG.patch_size
    amap = detector.build_anomaly_map(coords, patch_scores, rgb.shape[:2], grid_size)
    road_threshold_values = patch_scores if len(patch_scores) else np.array([0.0])
    threshold_px = float(np.percentile(road_threshold_values, CONFIG.anomaly_percentile))
    candidates = localizer.localize(rgb, amap, road_mask, threshold_px, sam2=road_masker)
    depth = depth_estimator.estimate(rgb)

    records = []
    warnings = []
    if depth is None:
        warnings.append("Metric RGB depth model was not provided; estimated_depth_m is null.")
    if candidates:
        for i, c in enumerate(candidates):
            m = c["mask"]
            area = estimate_area_m2(m, telemetry.altitude_m)
            depth_m = None
            if depth is not None:
                vals = depth[m]
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals):
                    # Do not treat camera-relative depth variation as exact pothole depth without a ground plane model.
                    depth_m = float(np.percentile(vals, 90) - np.percentile(vals, 10))
            water, water_conf = water_heuristic(rgb, m)
            conf = float(c["pothole_confidence"])
            sev = severity(conf, area, depth_m, water)
            records.append(PotholeRecord(
                pothole_id=f"{telemetry.timestamp.replace(':', '').replace('-', '')}-{i:03d}",
                timestamp=telemetry.timestamp,
                latitude=telemetry.latitude,
                longitude=telemetry.longitude,
                altitude_m=telemetry.altitude_m,
                area_m2=area,
                estimated_depth_m=depth_m,
                anomaly_score=float(c["anomaly_score"]),
                pothole_confidence=conf,
                severity_score=sev,
                water_flag=water,
                water_confidence=water_conf,
                source_image=str(image_path),
                mask_area_px=int(m.sum()),
                bbox_xyxy=[int(x) for x in c["bbox_xyxy"]],
                depth_source=depth_estimator.name,
                notes=["Heuristic pothole candidate; generic anomaly detection is not a trained pothole classifier."],
            ))

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--memory-bank", type=Path, default=CONFIG.memory_bank_dir)
    ap.add_argument("--output", type=Path, default=CONFIG.output_dir / "inference.json")
    args = ap.parse_args()
    result = infer(args.image, args.metadata, args.device, args.memory_bank)
    save_json(result.to_dict(), args.output)
    print(result.to_json())


if __name__ == "__main__":
    main()
