#!/usr/bin/env python3
"""RoadSentinel — 2D Pothole & Crater Detection Pipeline (SAM2 + DINOv2).

Designed for fast testing and immediate visual validation on flat 2D JPEG/PNG images.
Bypasses 3D CARLA simulation constraints (altitude, Z-depth, telemetry) and evaluates
craters directly using raw pixel area and tuned DINOv2 + SAM2 feature detection.

Key Features:
- Toggleable TEST_MODE_2D flag (defaults to True).
- Direct DINOv2 feature extraction with lowered zero-shot confidence threshold (0.15).
- Debug logging of raw candidate bounding boxes and confidence scores before filtering.
- SAM2ImagePredictor segmentation mask refinement.
- OpenCV visual rendering with semi-transparent mask overlays and bounding box annotations.
- Automated export to results/ directory for visual verification.

Usage:
  # Process a single 2D JPEG:
  python detect_2d_potholes.py --input path/to/pothole.jpg

  # Process a folder of 2D images:
  python detect_2d_potholes.py --input RoadSentinel_datasets/pothole_600/Pothole/rgb/ --limit 10

  # Run with custom confidence threshold and min pixel area:
  python detect_2d_potholes.py --input path/to/folder --conf 0.15 --min-area 50 --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure road_health_pipeline is in sys.path
PIPELINE_ROOT = Path(__file__).resolve().parent / "road_health_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Global 2D Test Mode Toggle & Calibrated Thresholds
# ---------------------------------------------------------------------------
TEST_MODE_2D: bool = True
DEFAULT_MIN_AREA_PX: int = 80
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40
DEFAULT_ANOMALY_PERCENTILE: float = 94.0

from config import CONFIG, Config
from common.io_utils import load_rgb, save_json, utc_iso
from common.schemas import CandidateRegion, DefectMeasurement, PotholeRecord
from inference.run_inference import load_pipeline, PipelineComponents, water_heuristic, generate_visual_overlays
from inference.low_light_enhancer import LowLightEnhancer
from inference.vehicle_suppressor import VehicleSuppressor
from inference.road_marking_suppressor import RoadMarkingSuppressor
from analytics.severity import calculate_defect_severity
from analytics.road_health import calculate_road_health_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("detect_2d")


# ---------------------------------------------------------------------------
# Visual Annotation & Overlay Rendering
# ---------------------------------------------------------------------------

def render_2d_pothole_overlay(
    rgb_image: np.ndarray,
    candidates: List[CandidateRegion],
    image_name: str = "image",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render OpenCV bounding boxes and alpha-blended SAM2 segmentation masks."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    mask_layer = np.zeros_like(bgr)
    h, w = rgb_image.shape[:2]

    detection_summaries = []

    for idx, cand in enumerate(candidates):
        x1, y1, x2, y2 = [int(v) for v in cand.bbox_xyxy]
        conf = cand.pothole_confidence
        mask = cand.mask
        mask_px = int(np.sum(mask)) if mask is not None else 0
        defect_type = cand.defect_type or "pothole"

        # Color scheme: Amber/Orange for dry crater, Cyan/Sky Blue for water hazard
        is_water = "water" in defect_type.lower()
        bbox_color = (0, 165, 255) if is_water else (0, 90, 255)       # BGR
        fill_color = (0, 215, 255) if is_water else (30, 144, 255)     # BGR
        contour_color = (255, 255, 255)

        # 1. Overlay SAM2 Segmentation Mask
        if mask is not None and np.any(mask):
            mask_layer[mask] = fill_color
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, contour_color, 2, cv2.LINE_AA)

        # 2. Draw Bounding Box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bbox_color, 2)

        # 3. Text Badge with Label, Confidence & Pixel Area
        badge_text = f"#{idx+1} {defect_type.upper()} ({conf:.2f}) | {mask_px}px"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        
        # Badge background pill
        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = max(0, y1)
        cv2.rectangle(overlay, (x1, badge_y1), (x1 + tw + 8, badge_y2), bbox_color, -1)
        cv2.putText(
            overlay,
            badge_text,
            (x1 + 4, badge_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        detection_summaries.append({
            "index": idx + 1,
            "bbox_xyxy": [x1, y1, x2, y2],
            "confidence": round(float(conf), 4),
            "mask_area_px": mask_px,
            "defect_type": defect_type,
            "anomaly_score": round(float(cand.anomaly_score), 4),
        })

    # Semi-transparent alpha blend for SAM2 masks
    if np.any(mask_layer):
        mask_active = np.any(mask_layer > 0, axis=-1)
        overlay[mask_active] = cv2.addWeighted(overlay, 0.60, mask_layer, 0.40, 0)[mask_active]

    # HUD Header Banner
    header_h = 45
    header_bg = overlay[:header_h, :].copy()
    cv2.rectangle(header_bg, (0, 0), (w, header_h), (25, 28, 35), -1)
    overlay[:header_h, :] = cv2.addWeighted(header_bg, 0.85, overlay[:header_h, :], 0.15, 0)

    title = f"ROADSENTINEL 2D VALIDATION | {image_name} | DETECTIONS: {len(candidates)}"
    cv2.putText(overlay, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay, {"detections": detection_summaries, "total_defects": len(candidates)}


# ---------------------------------------------------------------------------
# Core 2D Inference Pipeline Execution
# ---------------------------------------------------------------------------

def detect_potholes_2d(
    image_path: Path,
    pipeline: PipelineComponents,
    output_dir: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    anomaly_percentile: float = DEFAULT_ANOMALY_PERCENTILE,
    save_overlays: bool = True,
) -> Dict[str, Any]:
    """Execute SAM2 + DINOv2 crater detection in 2D test mode."""
    image_path = Path(image_path).resolve()
    rgb = load_rgb(image_path)
    h, w = rgb.shape[:2]

    log.info("Processing 2D image: %s (Resolution: %dx%d)", image_path.name, w, h)

    # 1. Low-Light Illumination Restoration
    enhancer = LowLightEnhancer()
    proc_rgb, was_enhanced = enhancer.enhance(rgb)
    if was_enhanced:
        log.info("  [Low-Light Restored] Applied adaptive CLAHE + gamma dynamic range expansion.")

    # 2. Road Surface Isolation (Road Corridor ROI)
    try:
        road_mask = pipeline.masker.get_road_mask(proc_rgb)
        if road_mask is None or road_mask.sum() < 0.08 * h * w:
            road_mask = pipeline.masker._extract_road_corridor_fallback(proc_rgb)
    except Exception as e:
        log.warning("SAM2 road surface prompt fallback to central corridor: %s", e)
        road_mask = pipeline.masker._extract_road_corridor_fallback(proc_rgb)

    # 3. DINOv2 Dense Patch Feature Extraction on restored road surface
    embeddings, coords = pipeline.embedder.extract_road_patch_embeddings(proc_rgb, road_mask)

    # 4. Anomaly Scoring & Heat-Map Generation
    patch_scores = pipeline.detector.score_patches(embeddings)
    anomaly_map = pipeline.detector.build_anomaly_map(
        coords, patch_scores, rgb.shape[:2], pipeline.embedder.grid_size
    )

    # 5. Vehicle & Obstacle Suppression Filter
    vehicle_suppressor = VehicleSuppressor()
    filtered_anomaly_map, vehicle_mask = vehicle_suppressor.suppress_vehicles(anomaly_map, proc_rgb)

    # 5b. Road Marking & Boundary Barrier Suppression Filter
    marking_suppressor = RoadMarkingSuppressor()
    marking_mask = marking_suppressor.get_marking_mask(proc_rgb)
    if np.any(marking_mask):
        filtered_anomaly_map[marking_mask] = 0.0
        log.info("  -> Suppressed %d px of painted road markings and roadside barriers.", int(np.sum(marking_mask)))

    # 6. Fine-Tuned DINOv2 Crater Head Verification & Adaptive Statistical Thresholding
    crater_presence_prob = 1.0
    if hasattr(pipeline, "crater_head") and pipeline.crater_head is not None:
        try:
            import torch
            from PIL import Image
            with torch.no_grad():
                t_img = pipeline.embedder.tf(Image.fromarray(proc_rgb)).unsqueeze(0).to(pipeline.embedder.device)
                features = pipeline.embedder.model.forward_features(t_img)
                c_logits, _, _ = pipeline.crater_head(features["x_norm_patchtokens"], features["x_norm_clstoken"])
                crater_presence_prob = float(torch.sigmoid(c_logits).item())
                log.info("  [DINOv2 Crater Head] Inferred road defect probability: %.2f%%", crater_presence_prob * 100.0)
        except Exception as e:
            log.warning("Crater head forward failed: %s", e)

    # Dynamic adaptive statistical thresholding:
    # With the 10,000-vector healthy road memory bank built from real highway imagery,
    # healthy tarmac sits cleanly around mean ~0.38 (std ~0.05).
    # Real road distress, cavities, and craters elevate above mean + 1.70 * std (~0.48 - 0.75).
    # Calibrated floor is set to 0.48 (not 0.62 which pruned genuine road cavities).
    road_scores = patch_scores if len(patch_scores) else np.array([0.0])
    mean_score = float(np.mean(road_scores))
    std_score = float(np.std(road_scores))
    threshold_px = max(0.48, mean_score + 1.70 * std_score)
    log.info("  DINOv2 patch anomaly threshold: %.4f (Mean: %.4f, Std: %.4f, Max: %.4f)",
             threshold_px, mean_score, std_score, float(np.max(road_scores)))

    # 7. Localize Candidates & SAM2 Refinement
    if crater_presence_prob < 0.12:
        log.info("  -> Image verified completely healthy by DINOv2 Crater Head (prob: %.1f%%). Skipping defect candidates.",
                 crater_presence_prob * 100.0)
        candidates = []
    else:
        candidates = pipeline.localizer.localize(
            rgb=proc_rgb,
            anomaly_map=filtered_anomaly_map,
            road_mask=road_mask,
            threshold=threshold_px,
            sam2=pipeline.masker,
            test_mode_2d=True,
            min_area_px=min_area_px,
            confidence_threshold=confidence_threshold,
        )

        # 8. Post-filtering against any residual vehicle overlap or road markings
        candidates = vehicle_suppressor.filter_candidate_regions(candidates, vehicle_mask)
        candidates = marking_suppressor.filter_candidate_regions(candidates, proc_rgb, marking_mask)

    log.info("  -> Found %d verified crater candidate(s) above confidence >= %.2f and area >= %d px",
             len(candidates), confidence_threshold, min_area_px)

    # 9. Render Overlays & Export Results
    output_dir.mkdir(parents=True, exist_ok=True)
    # Render overlay on enhanced RGB if low-light, so the user can visually verify the asphalt aggregate
    render_base = proc_rgb if was_enhanced else rgb
    overlay_img, summary = render_2d_pothole_overlay(render_base, candidates, image_name=image_path.name)

    overlay_filename = f"{image_path.stem}_detection_overlay.jpg"
    overlay_path = output_dir / overlay_filename
    cv2.imwrite(str(overlay_path), overlay_img)

    result_payload = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "test_mode_2d": True,
        "timestamp": utc_iso(),
        "image_shape": [h, w, 3],
        "anomaly_threshold": round(threshold_px, 4),
        "confidence_threshold": confidence_threshold,
        "min_area_px": min_area_px,
        "detections": summary["detections"],
        "total_defects": summary["total_defects"],
        "overlay_path": str(overlay_path),
    }

    return result_payload


def process_directory(
    input_path: Path,
    output_dir: Path,
    device: str = CONFIG.device,
    memory_bank_dir: Optional[Path] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    limit: Optional[int] = None,
) -> None:
    """Process a single image or directory of 2D images."""
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if limit:
            images = images[:limit]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not images:
        log.error("No images found to process in %s", input_path)
        return

    # Initialize models
    if memory_bank_dir is None:
        cand_mb = PIPELINE_ROOT / "output" / "real_memory_bank"
        memory_bank_dir = cand_mb if cand_mb.is_dir() else CONFIG.memory_bank_dir

    print("\n" + "=" * 70)
    print(f"  ROADSENTINEL 2D POTHOLE DETECTION PIPELINE (TEST_MODE_2D = {TEST_MODE_2D})")
    print(f"  Device: {device} | Memory Bank: {memory_bank_dir.name}")
    print(f"  Confidence Threshold: {confidence_threshold:.2f} | Min Area: {min_area_px} px")
    print(f"  Processing {len(images)} image(s)...")
    print("=" * 70 + "\n")

    pipeline = load_pipeline(device=device, memory_bank_dir=memory_bank_dir)

    all_results = []
    total_defects_found = 0
    t0 = time.time()

    for idx, img_p in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] Scanning: {img_p.name} ...")
        res = detect_potholes_2d(
            image_path=img_p,
            pipeline=pipeline,
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
            min_area_px=min_area_px,
        )
        all_results.append(res)
        total_defects_found += res["total_defects"]

    total_time = time.time() - t0
    results_json = output_dir / "detection_results.json"
    save_json(
        {
            "metadata": {
                "generated_at": utc_iso(),
                "total_images_processed": len(images),
                "total_defects_detected": total_defects_found,
                "confidence_threshold": confidence_threshold,
                "min_area_px": min_area_px,
                "elapsed_time_s": round(total_time, 2),
            },
            "results": all_results,
        },
        results_json,
    )

    print("\n" + "=" * 70)
    print("  2D DETECTION SUMMARY")
    print("=" * 70)
    print(f"  ✓ Processed Images      : {len(images)}")
    print(f"  ✓ Total Defects Found   : {total_defects_found}")
    print(f"  ✓ Time Elapsed          : {total_time:.2f}s ({total_time/max(1, len(images)):.2f}s/image)")
    print(f"  ✓ Results JSON Saved To : {results_json}")
    print(f"  ✓ Rendered Overlays In  : {output_dir}")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 2D Pothole Detection on flat JPEGs/PNGs with SAM2 + DINOv2."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("RoadSentinel_datasets/pothole_600/Pothole/rgb"),
        help="Path to 2D image file or directory of images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results"),
        help="Directory to save rendered visualization overlays and JSON results.",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Minimum DINOv2 confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-area",
        "-a",
        type=int,
        default=DEFAULT_MIN_AREA_PX,
        help=f"Minimum pixel area for anomaly detection (default: {DEFAULT_MIN_AREA_PX} px).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CONFIG.device,
        help="Computation device (cuda/cpu).",
    )
    parser.add_argument(
        "--memory-bank",
        type=Path,
        default=None,
        help="Path to healthy road memory bank.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of images processed from directory.",
    )
    args = parser.parse_args()

    process_directory(
        input_path=args.input,
        output_dir=args.output,
        device=args.device,
        memory_bank_dir=args.memory_bank,
        confidence_threshold=args.conf,
        min_area_px=args.min_area,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
