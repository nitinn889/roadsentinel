#!/usr/bin/env python3
"""Quantitative evaluation script for RoadSentinel SAM2 + DINOv2 pipeline.

Supports evaluating against:
  1. Segmentation Masks (e.g. Pothole-Mix / Pothole600 testing/validation sets)
     -> Computes Pixel IoU, Dice/F1, Precision, Recall, False Alarm Rate
  2. Bounding Box XMLs (e.g. RDD2022 China Drone aerial, India road)
     -> Computes Box Detection Recall, False Positives, Precision
  3. Image Folders (e.g. MWPD, Water-Filled Potholes)
     -> Computes anomaly score distributions, severity histograms, candidate detections

Usage:
  # 1. Fast Ground-Truth Segmentation Evaluation (Pothole600 test split)
  python scripts/evaluate_real_dataset.py --mode segmentation --limit 100

  # 2. Aerial / Drone Dataset Evaluation (RDD2022 China Drone)
  python scripts/evaluate_real_dataset.py --mode rdd_xml --subset china_drone --limit 100

  # 3. Custom directory
  python scripts/evaluate_real_dataset.py --images-dir /path/to/images --masks-dir /path/to/masks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import find_images, load_rgb, utc_iso
from inference.run_inference import load_pipeline, infer, PipelineComponents
from inference.anomaly_detector import AnomalyDetector
from common.schemas import CandidateRegion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_real_dataset")

DATASETS_ROOT = ROOT.parent / "RoadSentinel_datasets"


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Calculate binary segmentation metrics."""
    pred_bin = pred_mask.astype(bool)
    gt_bin = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = float(intersection / union) if union > 0 else (1.0 if not pred_bin.any() and not gt_bin.any() else 0.0)

    tp = float(intersection)
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not gt_bin.any() else 0.0)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else (1.0 if not pred_bin.any() and not gt_bin.any() else 0.0)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "gt_has_defect": bool(gt_bin.any()),
        "pred_has_defect": bool(pred_bin.any()),
    }


def parse_rdd_xml(xml_path: Path) -> list[dict[str, Any]]:
    """Parse Pascal VOC-style XML annotation from RDD2022."""
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text if obj.find("name") is not None else "unknown"
        bnd = obj.find("bndbox")
        if bnd is not None:
            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))
            objects.append({
                "class": name,
                "bbox": [xmin, ymin, xmax, ymax],
                "is_pothole": name == "D40",
            })
    return objects


def box_iou(box1: list[int], box2: list[int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def evaluate_segmentation(
    pipeline: PipelineComponents,
    images_dir: Path,
    masks_dir: Path,
    limit: Optional[int] = None,
    save_vis_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Evaluate against ground truth binary segmentation masks."""
    images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if limit:
        images = images[:limit]

    log.info("Running segmentation benchmark on %d image/mask pairs from %s...", len(images), images_dir.name)

    results = []
    t0 = time.monotonic()

    for idx, img_path in enumerate(images, 1):
        # Match mask filename
        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            continue

        rgb = load_rgb(img_path)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = gt_mask > 0

        # Run pipeline
        infer_res = infer(img_path, pipeline=pipeline)

        # Merge candidate masks
        pred_mask = np.zeros(rgb.shape[:2], dtype=bool)
        for p in infer_res.potholes:
            # We can use mask from candidates
            x1, y1, x2, y2 = p.bbox_xyxy
            pred_mask[y1:y2, x1:x2] = True

        m = compute_segmentation_metrics(pred_mask, gt_mask)
        m["image"] = img_path.name
        m["n_candidates"] = len(infer_res.potholes)
        m["anomaly_score"] = infer_res.anomaly_score
        results.append(m)

        # Optionally save visual comparison
        if save_vis_dir and idx <= 20:
            vis_row = []
            orig = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gt_overlay = orig.copy()
            gt_overlay[gt_mask] = [0, 255, 0]  # Green = Ground Truth
            gt_vis = cv2.addWeighted(orig, 0.6, gt_overlay, 0.4, 0)

            pred_overlay = orig.copy()
            pred_overlay[pred_mask] = [0, 0, 255]  # Red = Prediction
            pred_vis = cv2.addWeighted(orig, 0.6, pred_overlay, 0.4, 0)

            combined = np.hstack([orig, gt_vis, pred_vis])
            save_vis_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_vis_dir / f"eval_{img_path.stem}.jpg"), combined)

        if idx % 10 == 0 or idx == len(images):
            elapsed = time.monotonic() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            log.info("[%d/%d] (%.1f img/s) Current mean IoU: %.3f", idx, len(images), rate, np.mean([r["iou"] for r in results]))

    ious = [r["iou"] for r in results]
    dices = [r["dice"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    summary = {
        "dataset": str(images_dir),
        "num_evaluated_images": len(results),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "mean_dice_f1": float(np.mean(dices)) if dices else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "defect_detection_rate": float(np.mean([r["pred_has_defect"] == r["gt_has_defect"] for r in results])) if results else 0.0,
    }
    return summary, results


def evaluate_rdd_xml(
    pipeline: PipelineComponents,
    images_dir: Path,
    xml_dir: Path,
    limit: Optional[int] = None,
    iou_thresh: float = 0.3,
) -> dict[str, Any]:
    """Evaluate detection against RDD2022 Pascal VOC XML bounding boxes."""
    images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if limit:
        images = images[:limit]

    log.info("Running RDD2022 XML benchmark on %d images from %s...", len(images), images_dir.name)

    total_gt_potholes = 0
    total_gt_defects = 0
    tp_boxes = 0
    fp_boxes = 0
    total_pred_boxes = 0
    t0 = time.monotonic()

    per_image_results = []

    for idx, img_path in enumerate(images, 1):
        xml_path = xml_dir / f"{img_path.stem}.xml"
        gt_objects = parse_rdd_xml(xml_path)
        gt_boxes = [o["bbox"] for o in gt_objects]
        gt_potholes = [o["bbox"] for o in gt_objects if o["is_pothole"]]

        total_gt_defects += len(gt_boxes)
        total_gt_potholes += len(gt_potholes)

        infer_res = infer(img_path, pipeline=pipeline)
        pred_boxes = [p.bbox_xyxy for p in infer_res.potholes]
        total_pred_boxes += len(pred_boxes)

        matched_gt = set()
        for pbox in pred_boxes:
            matched = False
            for g_idx, gbox in enumerate(gt_boxes):
                if g_idx not in matched_gt and box_iou(pbox, gbox) >= iou_thresh:
                    matched_gt.add(g_idx)
                    tp_boxes += 1
                    matched = True
                    break
            if not matched:
                fp_boxes += 1

        per_image_results.append({
            "image": img_path.name,
            "n_gt_objects": len(gt_boxes),
            "n_gt_potholes": len(gt_potholes),
            "n_pred_candidates": len(pred_boxes),
            "anomaly_score": infer_res.anomaly_score,
        })

        if idx % 10 == 0 or idx == len(images):
            elapsed = time.monotonic() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            recall = tp_boxes / total_gt_defects if total_gt_defects > 0 else 0.0
            log.info("[%d/%d] (%.1f img/s) Recall: %.3f (TP=%d, GT=%d, Pred=%d)", idx, len(images), rate, recall, tp_boxes, total_gt_defects, total_pred_boxes)

    recall = float(tp_boxes / total_gt_defects) if total_gt_defects > 0 else 0.0
    precision = float(tp_boxes / total_pred_boxes) if total_pred_boxes > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    summary = {
        "dataset": str(images_dir),
        "num_evaluated_images": len(images),
        "total_gt_annotations": total_gt_defects,
        "total_gt_potholes": total_gt_potholes,
        "total_predicted_boxes": total_pred_boxes,
        "true_positives": tp_boxes,
        "false_positives": fp_boxes,
        "box_recall_at_iou_0_3": recall,
        "box_precision": precision,
        "box_f1": f1,
    }
    return summary, per_image_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate RoadSentinel on real datasets")
    ap.add_argument(
        "--mode",
        choices=["segmentation", "rdd_xml", "custom"],
        default="segmentation",
        help="Evaluation benchmark mode",
    )
    ap.add_argument(
        "--subset",
        choices=["pothole600_test", "pothole600_val", "china_drone", "india"],
        default="pothole600_test",
        help="Predefined dataset subset",
    )
    ap.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    ap.add_argument(
        "--memory-bank",
        type=Path,
        default=ROOT / "output" / "real_memory_bank",
        help="Path to memory bank (defaults to output/real_memory_bank)",
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Custom image directory (for custom mode)",
    )
    ap.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Custom masks directory (for custom mode)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images to evaluate (e.g. 50 or 100 for fast run)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "evaluation_results",
        help="Directory to save evaluation report and visualizations",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Determine default paths based on subset
    if args.mode == "segmentation":
        if args.subset == "pothole600_val":
            base = DATASETS_ROOT / "pothole_mix" / "pothole-mix" / "validation" / "pothole600"
        else:
            base = DATASETS_ROOT / "pothole_mix" / "pothole-mix" / "testing" / "pothole600"
        images_dir = base / "images"
        masks_dir = base / "masks"
    elif args.mode == "rdd_xml":
        if args.subset == "india":
            base = DATASETS_ROOT / "rdd2022_full" / "India" / "India" / "train"
        else:
            base = DATASETS_ROOT / "rdd2022_full" / "China_Drone" / "China_Drone" / "train"
        images_dir = base / "images"
        xml_dir = base / "annotations" / "xmls"
    else:
        images_dir = args.images_dir
        masks_dir = args.masks_dir

    log.info("=" * 65)
    log.info("RoadSentinel: Real Dataset Benchmark & Quantitative Evaluation")
    log.info("=" * 65)
    log.info("Mode        : %s (%s)", args.mode, args.subset)
    log.info("Images Dir  : %s", images_dir)
    log.info("Memory Bank : %s", args.memory_bank)
    log.info("Device      : %s", args.device)
    log.info("Limit       : %s", args.limit or "All")

    # Load pipeline once
    log.info("Loading SAM2 + DINOv2 pipeline components...")
    pipeline = load_pipeline(device=args.device, memory_bank_dir=args.memory_bank)

    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "segmentation":
        summary, details = evaluate_segmentation(
            pipeline=pipeline,
            images_dir=images_dir,
            masks_dir=masks_dir,
            limit=args.limit,
            save_vis_dir=vis_dir,
        )
    elif args.mode == "rdd_xml":
        summary, details = evaluate_rdd_xml(
            pipeline=pipeline,
            images_dir=images_dir,
            xml_dir=xml_dir,
            limit=args.limit,
        )
    else:
        raise NotImplementedError("Custom mode requires explicit flags.")

    # Save summary report
    summary["evaluated_at"] = utc_iso()
    summary_file = args.output_dir / f"evaluation_{args.mode}_{args.subset}.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    log.info("\n" + "=" * 65)
    log.info("BENCHMARK RESULTS SUMMARY:")
    log.info("=" * 65)
    for k, v in summary.items():
        if isinstance(v, float):
            log.info("  %-30s: %.4f", k, v)
        else:
            log.info("  %-30s: %s", k, v)
    log.info("=" * 65)
    log.info("Full metrics saved to: %s", summary_file)
    log.info("Visual samples saved to: %s", vis_dir)


if __name__ == "__main__":
    main()
