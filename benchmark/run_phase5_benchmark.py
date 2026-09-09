#!/usr/bin/env python3
"""RoadSentinel Phase 5: Formal YOLOv8n vs DINOv2+SAM2 Common Benchmark Runner.

Runs frozen YOLOv8n (supervised) and frozen DINOv2+SAM2 (Marion Day-5 NORMAL)
on the exact same 480 RDD2022 China_Drone validation images on RTX 5060 GPU.
Computes primary binary localization metrics at IoU >= 0.50, secondary sensitivity
at IoU >= 0.25, separate YOLO semantic metrics, hardware timing, condition subsets,
and regression checks.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parent.parent
SAM2_DINO_ROOT = ROOT / "sam2_dino"
PIPELINE_ROOT = ROOT / "road_health_pipeline"
for p in (str(ROOT), str(SAM2_DINO_ROOT), str(PIPELINE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.run_inference import load_pipeline, infer
from current_condition import mask_geometry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("phase5_benchmark")

# Constants
VAL_IMAGES_DIR = ROOT / "yolo/data/rdd2022/images/val"
VAL_LABELS_DIR = ROOT / "yolo/data/rdd2022/labels/val"
YOLO_WEIGHTS = ROOT / "yolo/weights/best.pt"
COMMON_CANDIDATE_MANIFEST = ROOT / "benchmark/common_candidate/manifest.csv"
OUT_DIR = ROOT / "benchmark/final_comparison"

CLASS_NAMES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "Repair",
}


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_boxes_greedy(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.50,
) -> Tuple[int, int, int, List[float]]:
    """Greedy 1-to-1 matching between predicted boxes and ground-truth boxes.

    Returns:
        (tp, fp, fn, matched_ious)
    """
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0, []
    if not pred_boxes:
        return 0, 0, len(gt_boxes), []
    if not gt_boxes:
        return 0, len(pred_boxes), 0, []

    candidates = []
    for p_idx, pb in enumerate(pred_boxes):
        for g_idx, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if v >= iou_threshold:
                candidates.append((v, p_idx, g_idx))

    # Sort descending by IoU
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    matched_ious = []

    for v, p_idx, g_idx in candidates:
        if p_idx not in matched_pred and g_idx not in matched_gt:
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            matched_ious.append(v)

    tp = len(matched_ious)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched_ious


def load_gt_for_image(label_path: Path, img_w: int = 512, img_h: int = 512) -> List[Dict[str, Any]]:
    """Parse YOLO format annotations [class_id xc yc w h normalized] into pixel xyxy."""
    if not label_path.exists():
        return []
    boxes = []
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    for line in lines:
        parts = line.split()
        cid = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w / 2.0) * img_w
        y1 = (yc - h / 2.0) * img_h
        x2 = (xc + w / 2.0) * img_w
        y2 = (yc + h / 2.0) * img_h
        boxes.append({
            "class_id": cid,
            "class_name": CLASS_NAMES.get(cid, str(cid)),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })
    return boxes


def run_benchmark():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log.info("Starting Phase 5 Benchmark on hardware: %s", device_name)

    # 1. Model Initialization
    log.info("--- Model Initialization ---")
    t0_yolo = time.perf_counter()
    yolo_model = YOLO(str(YOLO_WEIGHTS))
    t1_yolo = time.perf_counter()
    yolo_load_ms = (t1_yolo - t0_yolo) * 1000.0
    log.info("YOLOv8n loaded in %.1f ms", yolo_load_ms)

    t0_dino = time.perf_counter()
    dino_pipeline = load_pipeline(device="cuda" if torch.cuda.is_available() else "cpu")
    dino_pipeline.masker.camera_mode = "nadir"
    t1_dino = time.perf_counter()
    dino_load_ms = (t1_dino - t0_dino) * 1000.0
    log.info("DINOv2+SAM2 loaded in %.1f ms", dino_load_ms)

    # 2. Warmup
    log.info("--- GPU Warmup ---")
    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    dummy_path = OUT_DIR / "warmup_dummy.jpg"
    cv2.imwrite(str(dummy_path), dummy)
    _ = yolo_model.predict(source=dummy, conf=0.25, imgsz=512, device="0", verbose=False)
    _ = infer(dummy_path, pipeline=dino_pipeline, road_segment_id="WARMUP", test_mode_2d=True)
    dummy_path.unlink(missing_ok=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log.info("Warmup complete.")

    # 3. Regression Check
    log.info("--- Regression Verification ---")
    reg_1267 = ROOT / "benchmark/common_candidate/images/China_Drone_001267.jpg"
    reg_healthy = ROOT / "benchmark/common_candidate/images/original_healthy.jpg"
    
    yolo_1267 = yolo_model.predict(source=cv2.imread(str(reg_1267)), conf=0.25, imgsz=512, device="0", verbose=False)[0]
    yolo_healthy = yolo_model.predict(source=cv2.imread(str(reg_healthy)), conf=0.25, imgsz=512, device="0", verbose=False)[0]
    
    diag_1267 = {}
    _ = infer(reg_1267, pipeline=dino_pipeline, road_segment_id="REG", test_mode_2d=True, diagnostics=diag_1267)
    dino_1267_boxes = [mask_geometry(np.asarray(c.mask, dtype=bool))["bbox"] for c in diag_1267.get("candidates", []) if mask_geometry(np.asarray(c.mask, dtype=bool))["bbox"] is not None]
    
    diag_healthy = {}
    _ = infer(reg_healthy, pipeline=dino_pipeline, road_segment_id="REG", test_mode_2d=True, diagnostics=diag_healthy)
    dino_healthy_boxes = [mask_geometry(np.asarray(c.mask, dtype=bool))["bbox"] for c in diag_healthy.get("candidates", []) if mask_geometry(np.asarray(c.mask, dtype=bool))["bbox"] is not None]

    reg_pass = (len(yolo_1267.boxes) == 2 and len(yolo_healthy.boxes) == 0 and len(dino_healthy_boxes) == 0)
    log.info("Regression check: YOLO 1267=%d boxes, YOLO healthy=%d boxes, DINO healthy=%d boxes -> %s",
             len(yolo_1267.boxes), len(yolo_healthy.boxes), len(dino_healthy_boxes), "PASS" if reg_pass else "FAIL")

    # 4. Process all 480 Validation Images
    val_images = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    log.info("Found %d validation images in %s", len(val_images), VAL_IMAGES_DIR)
    assert len(val_images) == 480, f"Expected 480 validation images, found {len(val_images)}"

    per_image_records = []
    yolo_all_latencies = []
    dino_all_latencies = []

    # Accumulators for primary binary IoU 0.50
    yolo_bin_tp_50 = 0
    yolo_bin_fp_50 = 0
    yolo_bin_fn_50 = 0
    yolo_bin_ious_50 = []

    dino_bin_tp_50 = 0
    dino_bin_fp_50 = 0
    dino_bin_fn_50 = 0
    dino_bin_ious_50 = []

    # Accumulators for secondary binary IoU 0.25
    yolo_bin_tp_25 = 0
    yolo_bin_fp_25 = 0
    yolo_bin_fn_25 = 0
    yolo_bin_ious_25 = []

    dino_bin_tp_25 = 0
    dino_bin_fp_25 = 0
    dino_bin_fn_25 = 0
    dino_bin_ious_25 = []

    # Accumulators for YOLO semantic classes (IoU 0.50)
    yolo_sem_stats = {cid: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []} for cid in range(5)}

    total_gt_boxes_count = 0
    class_gt_distribution = {cid: 0 for cid in range(5)}

    start_bench = time.perf_counter()

    for idx, img_path in enumerate(val_images, start=1):
        stem = img_path.stem
        lbl_path = VAL_LABELS_DIR / f"{stem}.txt"
        
        # Load GT
        gt_boxes = load_gt_for_image(lbl_path, 512, 512)
        total_gt_boxes_count += len(gt_boxes)
        for g in gt_boxes:
            class_gt_distribution[g["class_id"]] += 1
        
        # Binary GT boxes
        gt_bin_boxes = [g["bbox"] for g in gt_boxes]

        # Raw image for YOLO
        bgr = cv2.imread(str(img_path))

        # --- YOLO Inference ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yolo_res = yolo_model.predict(source=bgr, conf=0.25, imgsz=512, device="0", verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        yolo_lat = (t1 - t0) * 1000.0
        yolo_all_latencies.append(yolo_lat)

        yolo_detections = []
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            confs = yolo_res.boxes.conf.cpu().numpy()
            clss = yolo_res.boxes.cls.cpu().numpy()
            for b, c, cl in zip(xyxy, confs, clss):
                cid = int(cl)
                yolo_detections.append({
                    "class_id": cid,
                    "class_name": CLASS_NAMES.get(cid, str(cid)),
                    "confidence": float(round(c, 4)),
                    "bbox": [round(float(v), 2) for v in b],
                })
        
        yolo_bin_boxes = [d["bbox"] for d in yolo_detections]

        # --- DINOv2 + SAM2 Inference ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dino_diag = {}
        _ = infer(img_path, pipeline=dino_pipeline, road_segment_id="VAL", test_mode_2d=True, diagnostics=dino_diag)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dino_lat = (t1 - t0) * 1000.0
        dino_all_latencies.append(dino_lat)

        dino_candidates = dino_diag.get("candidates", [])
        dino_boxes = []
        for c in dino_candidates:
            m = np.asarray(c.mask, dtype=bool)
            geom = mask_geometry(m)
            if geom["bbox"] is not None:
                dino_boxes.append(geom["bbox"])

        # --- Primary Matching: IoU 0.50 ---
        y_tp_50, y_fp_50, y_fn_50, y_ious_50 = match_boxes_greedy(yolo_bin_boxes, gt_bin_boxes, iou_threshold=0.50)
        d_tp_50, d_fp_50, d_fn_50, d_ious_50 = match_boxes_greedy(dino_boxes, gt_bin_boxes, iou_threshold=0.50)

        yolo_bin_tp_50 += y_tp_50
        yolo_bin_fp_50 += y_fp_50
        yolo_bin_fn_50 += y_fn_50
        yolo_bin_ious_50.extend(y_ious_50)

        dino_bin_tp_50 += d_tp_50
        dino_bin_fp_50 += d_fp_50
        dino_bin_fn_50 += d_fn_50
        dino_bin_ious_50.extend(d_ious_50)

        # --- Secondary Matching: IoU 0.25 ---
        y_tp_25, y_fp_25, y_fn_25, y_ious_25 = match_boxes_greedy(yolo_bin_boxes, gt_bin_boxes, iou_threshold=0.25)
        d_tp_25, d_fp_25, d_fn_25, d_ious_25 = match_boxes_greedy(dino_boxes, gt_bin_boxes, iou_threshold=0.25)

        yolo_bin_tp_25 += y_tp_25
        yolo_bin_fp_25 += y_fp_25
        yolo_bin_fn_25 += y_fn_25
        yolo_bin_ious_25.extend(y_ious_25)

        dino_bin_tp_25 += d_tp_25
        dino_bin_fp_25 += d_fp_25
        dino_bin_fn_25 += d_fn_25
        dino_bin_ious_25.extend(d_ious_25)

        # --- YOLO Semantic Class Matching (IoU 0.50) ---
        for cid in range(5):
            c_gt = [g["bbox"] for g in gt_boxes if g["class_id"] == cid]
            c_pred = [d["bbox"] for d in yolo_detections if d["class_id"] == cid]
            c_tp, c_fp, c_fn, c_ious = match_boxes_greedy(c_pred, c_gt, iou_threshold=0.50)
            yolo_sem_stats[cid]["gt"] += len(c_gt)
            yolo_sem_stats[cid]["pred"] += len(c_pred)
            yolo_sem_stats[cid]["tp"] += c_tp
            yolo_sem_stats[cid]["fp"] += c_fp
            yolo_sem_stats[cid]["fn"] += c_fn
            yolo_sem_stats[cid]["ious"].extend(c_ious)

        # Per-image precision, recall, f1 for IoU 0.50
        y_prec = y_tp_50 / (y_tp_50 + y_fp_50) if (y_tp_50 + y_fp_50) > 0 else 0.0
        y_rec = y_tp_50 / (y_tp_50 + y_fn_50) if (y_tp_50 + y_fn_50) > 0 else 0.0
        y_f1 = (2 * y_prec * y_rec / (y_prec + y_rec)) if (y_prec + y_rec) > 0 else 0.0
        y_mean_iou = float(np.mean(y_ious_50)) if y_ious_50 else 0.0

        d_prec = d_tp_50 / (d_tp_50 + d_fp_50) if (d_tp_50 + d_fp_50) > 0 else 0.0
        d_rec = d_tp_50 / (d_tp_50 + d_fn_50) if (d_tp_50 + d_fn_50) > 0 else 0.0
        d_f1 = (2 * d_prec * d_rec / (d_prec + d_rec)) if (d_prec + d_rec) > 0 else 0.0
        d_mean_iou = float(np.mean(d_ious_50)) if d_ious_50 else 0.0

        # Qualitative note/winner assignment
        if y_f1 > d_f1:
            winner = "YOLO"
        elif d_f1 > y_f1:
            winner = "DINO_SAM"
        elif y_f1 > 0 and y_f1 == d_f1:
            winner = "TIED_SUCCESS"
        elif len(gt_bin_boxes) == 0 and len(yolo_bin_boxes) == 0 and len(dino_boxes) == 0:
            winner = "TIED_HEALTHY_CORRECT"
        else:
            winner = "TIED_ZERO_OR_FAIL"

        per_image_records.append({
            "image_id": stem,
            "gt_count": len(gt_bin_boxes),
            "yolo_pred_count": len(yolo_bin_boxes),
            "yolo_tp": y_tp_50,
            "yolo_fp": y_fp_50,
            "yolo_fn": y_fn_50,
            "yolo_precision": round(y_prec, 4),
            "yolo_recall": round(y_rec, 4),
            "yolo_f1": round(y_f1, 4),
            "yolo_mean_iou": round(y_mean_iou, 4),
            "yolo_latency_ms": round(yolo_lat, 2),
            "dino_sam_pred_count": len(dino_boxes),
            "dino_sam_tp": d_tp_50,
            "dino_sam_fp": d_fp_50,
            "dino_sam_fn": d_fn_50,
            "dino_sam_precision": round(d_prec, 4),
            "dino_sam_recall": round(d_rec, 4),
            "dino_sam_f1": round(d_f1, 4),
            "dino_sam_mean_iou": round(d_mean_iou, 4),
            "dino_sam_latency_ms": round(dino_lat, 2),
            "winner_or_note": winner,
            "raw_gt": gt_boxes,
            "raw_yolo": yolo_detections,
            "raw_dino_boxes": dino_boxes,
        })

        if idx % 50 == 0 or idx == len(val_images):
            log.info("Processed [%d/%d] images... Current YOLO avg lat: %.2f ms | DINO avg lat: %.2f ms",
                     idx, len(val_images), np.mean(yolo_all_latencies), np.mean(dino_all_latencies))

    total_bench_duration = time.perf_counter() - start_bench
    log.info("Completed full 480 validation set in %.2f seconds.", total_bench_duration)

    # 5. Compute Aggregate Primary Metrics (IoU 0.50)
    y_p_50 = yolo_bin_tp_50 / (yolo_bin_tp_50 + yolo_bin_fp_50) if (yolo_bin_tp_50 + yolo_bin_fp_50) > 0 else 0.0
    y_r_50 = yolo_bin_tp_50 / (yolo_bin_tp_50 + yolo_bin_fn_50) if (yolo_bin_tp_50 + yolo_bin_fn_50) > 0 else 0.0
    y_f1_50 = (2 * y_p_50 * y_r_50 / (y_p_50 + y_r_50)) if (y_p_50 + y_r_50) > 0 else 0.0
    y_mean_iou_50 = float(np.mean(yolo_bin_ious_50)) if yolo_bin_ious_50 else 0.0
    y_med_iou_50 = float(np.median(yolo_bin_ious_50)) if yolo_bin_ious_50 else 0.0

    d_p_50 = dino_bin_tp_50 / (dino_bin_tp_50 + dino_bin_fp_50) if (dino_bin_tp_50 + dino_bin_fp_50) > 0 else 0.0
    d_r_50 = dino_bin_tp_50 / (dino_bin_tp_50 + dino_bin_fn_50) if (dino_bin_tp_50 + dino_bin_fn_50) > 0 else 0.0
    d_f1_50 = (2 * d_p_50 * d_r_50 / (d_p_50 + d_r_50)) if (d_p_50 + d_r_50) > 0 else 0.0
    d_mean_iou_50 = float(np.mean(dino_bin_ious_50)) if dino_bin_ious_50 else 0.0
    d_med_iou_50 = float(np.median(dino_bin_ious_50)) if dino_bin_ious_50 else 0.0

    # Secondary Metrics (IoU 0.25)
    y_p_25 = yolo_bin_tp_25 / (yolo_bin_tp_25 + yolo_bin_fp_25) if (yolo_bin_tp_25 + yolo_bin_fp_25) > 0 else 0.0
    y_r_25 = yolo_bin_tp_25 / (yolo_bin_tp_25 + yolo_bin_fn_25) if (yolo_bin_tp_25 + yolo_bin_fn_25) > 0 else 0.0
    y_f1_25 = (2 * y_p_25 * y_r_25 / (y_p_25 + y_r_25)) if (y_p_25 + y_r_25) > 0 else 0.0

    d_p_25 = dino_bin_tp_25 / (dino_bin_tp_25 + dino_bin_fp_25) if (dino_bin_tp_25 + dino_bin_fp_25) > 0 else 0.0
    d_r_25 = dino_bin_tp_25 / (dino_bin_tp_25 + dino_bin_fn_25) if (dino_bin_tp_25 + dino_bin_fn_25) > 0 else 0.0
    d_f1_25 = (2 * d_p_25 * d_r_25 / (d_p_25 + d_r_25)) if (d_p_25 + d_r_25) > 0 else 0.0

    # Timing Stats
    y_mean_lat = float(np.mean(yolo_all_latencies))
    y_med_lat = float(np.median(yolo_all_latencies))
    y_std_lat = float(np.std(yolo_all_latencies))
    y_min_lat = float(np.min(yolo_all_latencies))
    y_max_lat = float(np.max(yolo_all_latencies))
    y_fps = 1000.0 / y_mean_lat if y_mean_lat > 0 else 0.0

    d_mean_lat = float(np.mean(dino_all_latencies))
    d_med_lat = float(np.median(dino_all_latencies))
    d_std_lat = float(np.std(dino_all_latencies))
    d_min_lat = float(np.min(dino_all_latencies))
    d_max_lat = float(np.max(dino_all_latencies))
    d_fps = 1000.0 / d_mean_lat if d_mean_lat > 0 else 0.0

    # 6. Save Primary Comparison CSV
    # benchmark/final_comparison/common_binary_metrics.csv
    common_binary_csv = OUT_DIR / "common_binary_metrics.csv"
    with open(common_binary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "num_images", "num_gt_boxes", "num_predictions",
            "TP", "FP", "FN", "precision", "recall", "f1",
            "mean_matched_bbox_iou", "median_matched_bbox_iou",
            "mean_latency_ms", "median_latency_ms", "fps",
            "hardware", "notes"
        ])
        writer.writerow([
            "YOLOv8n", len(val_images), total_gt_boxes_count,
            yolo_bin_tp_50 + yolo_bin_fp_50,
            yolo_bin_tp_50, yolo_bin_fp_50, yolo_bin_fn_50,
            round(y_p_50, 4), round(y_r_50, 4), round(y_f1_50, 4),
            round(y_mean_iou_50, 4), round(y_med_iou_50, 4),
            round(y_mean_lat, 2), round(y_med_lat, 2), round(y_fps, 2),
            device_name, "Supervised detector frozen checkpoint best.pt (conf=0.25, imgsz=512)"
        ])
        writer.writerow([
            "DINOv2+SAM2", len(val_images), total_gt_boxes_count,
            dino_bin_tp_50 + dino_bin_fp_50,
            dino_bin_tp_50, dino_bin_fp_50, dino_bin_fn_50,
            round(d_p_50, 4), round(d_r_50, 4), round(d_f1_50, 4),
            round(d_mean_iou_50, 4), round(d_med_iou_50, 4),
            round(d_mean_lat, 2), round(d_med_lat, 2), round(d_fps, 2),
            device_name, "Frozen Marion Day-5 NORMAL pipeline; mask-derived bounding boxes"
        ])
    log.info("Saved %s", common_binary_csv)

    # Secondary Sensitivity CSV
    sensitivity_csv = OUT_DIR / "common_binary_sensitivity_iou25.csv"
    with open(sensitivity_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "iou_threshold", "num_images", "num_gt_boxes", "num_predictions",
            "TP", "FP", "FN", "precision", "recall", "f1",
            "mean_matched_bbox_iou", "median_matched_bbox_iou"
        ])
        writer.writerow([
            "YOLOv8n", 0.25, len(val_images), total_gt_boxes_count,
            yolo_bin_tp_25 + yolo_bin_fp_25,
            yolo_bin_tp_25, yolo_bin_fp_25, yolo_bin_fn_25,
            round(y_p_25, 4), round(y_r_25, 4), round(y_f1_25, 4),
            round(float(np.mean(yolo_bin_ious_25)), 4) if yolo_bin_ious_25 else 0.0,
            round(float(np.median(yolo_bin_ious_25)), 4) if yolo_bin_ious_25 else 0.0,
        ])
        writer.writerow([
            "DINOv2+SAM2", 0.25, len(val_images), total_gt_boxes_count,
            dino_bin_tp_25 + dino_bin_fp_25,
            dino_bin_tp_25, dino_bin_fp_25, dino_bin_fn_25,
            round(d_p_25, 4), round(d_r_25, 4), round(d_f1_25, 4),
            round(float(np.mean(dino_bin_ious_25)), 4) if dino_bin_ious_25 else 0.0,
            round(float(np.median(dino_bin_ious_25)), 4) if dino_bin_ious_25 else 0.0,
        ])
    log.info("Saved %s", sensitivity_csv)

    # 7. Save Per-Image Results CSV
    # benchmark/final_comparison/per_image_comparison.csv
    per_image_csv = OUT_DIR / "per_image_comparison.csv"
    fieldnames = [
        "image_id", "gt_count",
        "yolo_pred_count", "yolo_tp", "yolo_fp", "yolo_fn",
        "yolo_precision", "yolo_recall", "yolo_f1", "yolo_mean_iou", "yolo_latency_ms",
        "dino_sam_pred_count", "dino_sam_tp", "dino_sam_fp", "dino_sam_fn",
        "dino_sam_precision", "dino_sam_recall", "dino_sam_f1", "dino_sam_mean_iou", "dino_sam_latency_ms",
        "winner_or_note"
    ]
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in per_image_records:
            clean_row = {k: rec[k] for k in fieldnames}
            writer.writerow(clean_row)
    log.info("Saved %s", per_image_csv)

    # 8. Save YOLO Semantic Metrics CSV
    # Class coverage audit and semantic breakdown
    yolo_sem_csv = OUT_DIR / "yolo_semantic_metrics.csv"
    # Native YOLO mAP50 per class from training validation
    yolo_native_map50 = {0: 0.3435, 1: 0.4045, 2: 0.2046, 3: 0.2591, 4: 0.5867}
    with open(yolo_sem_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "class_name", "gt_count", "predicted_count",
            "TP", "FP", "FN", "precision", "recall", "f1",
            "mean_matched_iou", "native_yolo_map50", "support_note"
        ])
        for cid in range(5):
            st = yolo_sem_stats[cid]
            p = st["tp"] / (st["tp"] + st["fp"]) if (st["tp"] + st["fp"]) > 0 else 0.0
            r = st["tp"] / (st["tp"] + st["fn"]) if (st["tp"] + st["fn"]) > 0 else 0.0
            f1_val = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            m_iou = float(np.mean(st["ious"])) if st["ious"] else 0.0
            support_note = "ADEQUATE" if st["gt"] >= 50 else "INSUFFICIENT SUPPORT FOR STRONG CLASS-LEVEL CONCLUSION"
            writer.writerow([
                cid, CLASS_NAMES[cid], st["gt"], st["pred"],
                st["tp"], st["fp"], st["fn"],
                round(p, 4), round(r, 4), round(f1_val, 4),
                round(m_iou, 4), yolo_native_map50.get(cid, "N/A"),
                support_note
            ])
    log.info("Saved %s", yolo_sem_csv)

    # 9. Condition-Tagged Small Subset Analysis
    # Load manifest and match images
    manifest_rows = []
    if COMMON_CANDIDATE_MANIFEST.exists():
        with open(COMMON_CANDIDATE_MANIFEST, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)

    cond_groups = {}
    for row in manifest_rows:
        if row.get("gt_compatible", "").lower() != "true":
            continue
        cond = row.get("condition", "unknown")
        cond_groups.setdefault(cond, []).append(row["original_filename"])

    cond_subset_csv = OUT_DIR / "condition_subset_metrics.csv"
    with open(cond_subset_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition_tag", "provenance", "image_count", "gt_boxes",
            "yolo_tp", "yolo_fp", "yolo_fn", "yolo_precision", "yolo_recall", "yolo_f1",
            "dino_tp", "dino_fp", "dino_fn", "dino_precision", "dino_recall", "dino_f1"
        ])
        for cond, fnames in sorted(cond_groups.items()):
            subset_recs = [r for r in per_image_records if f"{r['image_id']}.jpg" in fnames]
            gt_cnt = sum(r["gt_count"] for r in subset_recs)
            
            y_tp = sum(r["yolo_tp"] for r in subset_recs)
            y_fp = sum(r["yolo_fp"] for r in subset_recs)
            y_fn = sum(r["yolo_fn"] for r in subset_recs)
            yp = y_tp / (y_tp + y_fp) if (y_tp + y_fp) > 0 else 0.0
            yr = y_tp / (y_tp + y_fn) if (y_tp + y_fn) > 0 else 0.0
            yf = 2 * yp * yr / (yp + yr) if (yp + yr) > 0 else 0.0

            d_tp = sum(r["dino_sam_tp"] for r in subset_recs)
            d_fp = sum(r["dino_sam_fp"] for r in subset_recs)
            d_fn = sum(r["dino_sam_fn"] for r in subset_recs)
            dp = d_tp / (d_tp + d_fp) if (d_tp + d_fp) > 0 else 0.0
            dr = d_tp / (d_tp + d_fn) if (d_tp + d_fn) > 0 else 0.0
            df = 2 * dp * dr / (dp + dr) if (dp + dr) > 0 else 0.0

            writer.writerow([
                cond, "VISUAL_LABEL_ONLY", len(subset_recs), gt_cnt,
                y_tp, y_fp, y_fn, round(yp, 4), round(yr, 4), round(yf, 4),
                d_tp, d_fp, d_fn, round(dp, 4), round(dr, 4), round(df, 4)
            ])
    log.info("Saved %s", cond_subset_csv)

    # 10. Raw Results JSON for Reproducibility & Qualitative Selection
    summary_json = OUT_DIR / "benchmark_summary.json"
    summary_payload = {
        "benchmark_metadata": {
            "title": "RoadSentinel Phase 5 Formal Common Benchmark",
            "dataset": "RDD2022 China_Drone Validation Split",
            "dataset_role": "COMMON VALIDATION BENCHMARK",
            "is_independent_unseen_test_set": False,
            "total_images": len(val_images),
            "total_gt_boxes": total_gt_boxes_count,
            "gt_class_distribution": class_gt_distribution,
            "hardware": device_name,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "yolo_load_ms": round(yolo_load_ms, 2),
            "dino_load_ms": round(dino_load_ms, 2),
            "regression_check": "PASS" if reg_pass else "FAIL",
        },
        "primary_binary_iou_050": {
            "yolo": {
                "predictions": yolo_bin_tp_50 + yolo_bin_fp_50,
                "tp": yolo_bin_tp_50,
                "fp": yolo_bin_fp_50,
                "fn": yolo_bin_fn_50,
                "precision": round(y_p_50, 4),
                "recall": round(y_r_50, 4),
                "f1": round(y_f1_50, 4),
                "mean_matched_iou": round(y_mean_iou_50, 4),
                "median_matched_iou": round(y_med_iou_50, 4),
                "mean_latency_ms": round(y_mean_lat, 2),
                "median_latency_ms": round(y_med_lat, 2),
                "std_latency_ms": round(y_std_lat, 2),
                "min_latency_ms": round(y_min_lat, 2),
                "max_latency_ms": round(y_max_lat, 2),
                "fps": round(y_fps, 2),
            },
            "dino_sam": {
                "predictions": dino_bin_tp_50 + dino_bin_fp_50,
                "tp": dino_bin_tp_50,
                "fp": dino_bin_fp_50,
                "fn": dino_bin_fn_50,
                "precision": round(d_p_50, 4),
                "recall": round(d_r_50, 4),
                "f1": round(d_f1_50, 4),
                "mean_matched_iou": round(d_mean_iou_50, 4),
                "median_matched_iou": round(d_med_iou_50, 4),
                "mean_latency_ms": round(d_mean_lat, 2),
                "median_latency_ms": round(d_med_lat, 2),
                "std_latency_ms": round(d_std_lat, 2),
                "min_latency_ms": round(d_min_lat, 2),
                "max_latency_ms": round(d_max_lat, 2),
                "fps": round(d_fps, 2),
            },
            "comparison": {
                "delta_precision": round(y_p_50 - d_p_50, 4),
                "delta_recall": round(y_r_50 - d_r_50, 4),
                "delta_f1": round(y_f1_50 - d_f1_50, 4),
                "delta_mean_iou": round(y_mean_iou_50 - d_mean_iou_50, 4),
                "latency_ratio": round(d_mean_lat / y_mean_lat, 2) if y_mean_lat > 0 else 0.0,
                "higher_precision": "YOLOv8n" if y_p_50 > d_p_50 else "DINOv2+SAM2",
                "higher_recall": "YOLOv8n" if y_r_50 > d_r_50 else "DINOv2+SAM2",
                "higher_f1": "YOLOv8n" if y_f1_50 > d_f1_50 else "DINOv2+SAM2",
                "higher_matched_iou": "YOLOv8n" if y_mean_iou_50 > d_mean_iou_50 else "DINOv2+SAM2",
                "faster_model": "YOLOv8n" if y_mean_lat < d_mean_lat else "DINOv2+SAM2",
            }
        },
        "secondary_binary_iou_025": {
            "yolo": {
                "tp": yolo_bin_tp_25, "fp": yolo_bin_fp_25, "fn": yolo_bin_fn_25,
                "precision": round(y_p_25, 4), "recall": round(y_r_25, 4), "f1": round(y_f1_25, 4),
            },
            "dino_sam": {
                "tp": dino_bin_tp_25, "fp": dino_bin_fp_25, "fn": dino_bin_fn_25,
                "precision": round(d_p_25, 4), "recall": round(d_r_25, 4), "f1": round(d_f1_25, 4),
            }
        },
        "per_image_records": per_image_records,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log.info("Saved %s", summary_json)

    print("\n" + "=" * 60)
    print("PHASE 5 BENCHMARK COMPLETE")
    print(f"Total Images: {len(val_images)} | Total GT Boxes: {total_gt_boxes_count}")
    print(f"YOLOv8n:     P={y_p_50:.4f}, R={y_r_50:.4f}, F1={y_f1_50:.4f}, Mean IoU={y_mean_iou_50:.4f}, Lat={y_mean_lat:.1f}ms ({y_fps:.1f} FPS)")
    print(f"DINOv2+SAM2: P={d_p_50:.4f}, R={d_r_50:.4f}, F1={d_f1_50:.4f}, Mean IoU={d_mean_iou_50:.4f}, Lat={d_mean_lat:.1f}ms ({d_fps:.1f} FPS)")
    print(f"Comparison:  Delta P={y_p_50 - d_p_50:+.4f}, Delta R={y_r_50 - d_r_50:+.4f}, Delta F1={y_f1_50 - d_f1_50:+.4f}, Lat Ratio={d_mean_lat / y_mean_lat:.1f}x")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()
