#!/usr/bin/env python3
"""Generate 3-panel comparative visual figures for Phase 5.

Produces side-by-side comparative panels across 9 representative qualitative categories:
- LEFT: Original image with Ground Truth bounding boxes (Green)
- CENTER: YOLOv8n detections (Orange/Blue)
- RIGHT: DINOv2 + SAM2 mask overlay + mask-derived bounding boxes (Magenta/Red)
Saves panels to benchmark/final_comparison/panels/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
log = logging.getLogger("generate_panels")

VAL_IMAGES_DIR = ROOT / "yolo/data/rdd2022/images/val"
OUT_DIR = ROOT / "benchmark/final_comparison/panels"
SUMMARY_JSON = ROOT / "benchmark/final_comparison/benchmark_summary.json"

CLASS_COLORS_GT = {
    0: (0, 200, 0),      # D00
    1: (0, 230, 100),    # D10
    2: (0, 255, 200),    # D20
    3: (0, 100, 255),    # D40
    4: (100, 255, 0),    # Repair
}


def draw_panel(
    image_bgr: np.ndarray,
    title: str,
    boxes: List[Dict[str, Any]],
    box_color: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None,
    mask_tint: Tuple[int, int, int] = (255, 0, 128),
) -> np.ndarray:
    """Render a single labeled panel with top title bar, bounding boxes, and optional mask."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    # Render mask if present
    if mask is not None and np.any(mask):
        tint_layer = np.zeros_like(canvas)
        tint_layer[mask] = mask_tint
        canvas[mask] = cv2.addWeighted(canvas[mask], 0.60, tint_layer[mask], 0.40, 0.0)

    # Render bounding boxes
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
        label = b.get("label", "")
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        if label:
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), box_color, -1)
            text_color = (0, 0, 0) if sum(box_color) > 400 else (255, 255, 255)
            cv2.putText(canvas, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # Header bar
    header_h = 36
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    (tw, th), bl = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(header, title, (int((w - tw) / 2), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    return np.vstack([header, canvas])


def generate_panels():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SUMMARY_JSON.exists():
        log.error("benchmark_summary.json not found at %s. Please run benchmark first.", SUMMARY_JSON)
        return

    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    records = data.get("per_image_records", [])
    log.info("Loaded %d per-image records from summary.", len(records))

    # Pre-select representative cases
    # Categories:
    # A. BOTH_CORRECT
    # B. YOLO_ONLY_SUCCESS
    # C. DINO_SAM_ONLY_SUCCESS
    # D. BOTH_FAIL
    # E. YOLO_FALSE_POSITIVE
    # F. DINO_SAM_FALSE_POSITIVE
    # G. THIN_CRACK_DIFFICULTY
    # H. POTHOLE_EXAMPLE
    # I. REPAIR_EXAMPLE

    # Explicit selection of representative cases across all 9 categories
    explicit_selection = {
        "A_BOTH_CORRECT": "China_Drone_001063",            # Both F1 = 1.0 (TP=1, FP=0, FN=0)
        "B_YOLO_ONLY_SUCCESS": "China_Drone_000010",        # YOLO F1 = 0.80, DINO F1 = 0.0 (YOLO TP=2, DINO TP=0)
        "C_DINO_SAM_ONLY_SUCCESS": "China_Drone_002162",    # DINO F1 = 0.50 (TP=1), YOLO F1 = 0.0 (TP=0)
        "D_BOTH_FAIL": "China_Drone_000033",                # Both models missed defect (Both F1 = 0.0)
        "E_YOLO_FALSE_POSITIVE": "China_Drone_000479",      # YOLO FP = 6 (over-predicted non-defects)
        "F_DINO_SAM_FALSE_POSITIVE": "China_Drone_000072",  # DINO FP = 8 on background road texture
        "G_THIN_CRACK_DIFFICULTY": "China_Drone_001380",    # Thin D00 crack; YOLO FP=2, FN=1; DINO FP=1, FN=1
        "H_POTHOLE_EXAMPLE": "China_Drone_000064",          # Rare D40 pothole class
        "I_REPAIR_EXAMPLE": "China_Drone_000023",           # Repair patch class
    }

    rec_map = {r["image_id"]: r for r in records}
    cases = {cat: rec_map.get(iid) for cat, iid in explicit_selection.items()}

    log.info("Selected case distribution:")
    for k, v in cases.items():
        log.info("  %s: %s", k, v["image_id"] if v else "None")

    # Load DINO pipeline to get exact masks for selected cases
    dino_pipeline = load_pipeline(device="cuda")
    dino_pipeline.masker.camera_mode = "nadir"

    generated_panels_info = []

    for cat_name, rec in cases.items():
        if rec is None:
            log.warning("No candidate found for category %s; skipping.", cat_name)
            continue
        image_id = rec["image_id"]
        img_path = VAL_IMAGES_DIR / f"{image_id}.jpg"
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue
        h, w = raw_bgr.shape[:2]

        # 1. Left: Ground Truth
        gt_boxes = []
        for g in rec.get("raw_gt", []):
            gt_boxes.append({
                "bbox": g["bbox"],
                "label": f"GT: {g['class_name']}",
            })
        panel_left = draw_panel(
            raw_bgr,
            f"1. Ground Truth (n={len(gt_boxes)})",
            gt_boxes,
            box_color=(0, 220, 0),
        )

        # 2. Center: YOLOv8n
        yolo_boxes = []
        for d in rec.get("raw_yolo", []):
            yolo_boxes.append({
                "bbox": d["bbox"],
                "label": f"{d['class_name']} {d['confidence']:.2f}",
            })
        panel_center = draw_panel(
            raw_bgr,
            f"2. YOLOv8n (P={rec['yolo_precision']:.2f}, R={rec['yolo_recall']:.2f}, F1={rec['yolo_f1']:.2f})",
            yolo_boxes,
            box_color=(0, 140, 255),
        )

        # 3. Right: DINOv2 + SAM2
        # Run inference to obtain exact mask
        dino_diag = {}
        _ = infer(img_path, pipeline=dino_pipeline, road_segment_id="PANEL", test_mode_2d=True, diagnostics=dino_diag)
        cands = dino_diag.get("candidates", [])
        union_mask = np.zeros((h, w), dtype=bool)
        dino_boxes = []
        for c in cands:
            m = np.asarray(c.mask, dtype=bool)
            union_mask |= m
            geom = mask_geometry(m)
            if geom["bbox"] is not None:
                dino_boxes.append({
                    "bbox": geom["bbox"],
                    "label": "road_defect",
                })
        panel_right = draw_panel(
            raw_bgr,
            f"3. DINOv2+SAM2 (P={rec['dino_sam_precision']:.2f}, R={rec['dino_sam_recall']:.2f}, F1={rec['dino_sam_f1']:.2f})",
            dino_boxes,
            box_color=(255, 0, 180),
            mask=union_mask,
        )

        # Concatenate horizontally
        combined = np.hstack([panel_left, panel_center, panel_right])

        # Overall Title Banner
        title_bar_h = 44
        title_bar = np.zeros((title_bar_h, combined.shape[1], 3), dtype=np.uint8)
        title_bar[:] = (20, 20, 20)
        main_title = f"{cat_name} — Image: {image_id}.jpg | GT Boxes: {rec['gt_count']} | YOLO F1: {rec['yolo_f1']:.2f} | DINO F1: {rec['dino_sam_f1']:.2f}"
        (tw, th), bl = cv2.getTextSize(main_title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(title_bar, main_title, (int((combined.shape[1] - tw) / 2), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        final_figure = np.vstack([title_bar, combined])

        out_path = OUT_DIR / f"{cat_name}_{image_id}.jpg"
        cv2.imwrite(str(out_path), final_figure)
        log.info("Saved visual panel: %s", out_path)
        generated_panels_info.append({
            "category": cat_name,
            "image_id": image_id,
            "panel_path": str(out_path.relative_to(ROOT)),
            "yolo_f1": rec["yolo_f1"],
            "dino_f1": rec["dino_sam_f1"],
        })

    index_json = OUT_DIR / "panels_index.json"
    index_json.write_text(json.dumps(generated_panels_info, indent=2), encoding="utf-8")
    log.info("Completed generating %d visual panels. Index saved to %s", len(generated_panels_info), index_json)


if __name__ == "__main__":
    generate_panels()
