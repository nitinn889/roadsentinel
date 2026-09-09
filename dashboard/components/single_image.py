"""Single-Image Road Assessment Component for RoadSentinel Dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from data_loader import (
    WORKSPACE_ROOT,
    get_curated_panel_images,
    load_benchmark_summary,
    load_experiment_a_perception,
)

CLASS_NAMES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "Repair",
}

YOLO_COLORS = {
    "D00": (255, 140, 0),     # Orange: Longitudinal Crack
    "D10": (255, 215, 0),     # Gold/Yellow: Transverse Crack
    "D20": (255, 0, 255),     # Magenta: Alligator Crack
    "D40": (255, 0, 0),       # Red: Pothole
    "Repair": (0, 255, 100),  # Green: Repair Patch
}


def draw_yolo_boxes(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    canvas = image_bgr.copy()
    for d in detections:
        box = d.get("bbox", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cname = d.get("class_name") or d.get("class", "defect")
        conf = d.get("confidence", 0.0)
        color = YOLO_COLORS.get(cname, (255, 128, 0))
        # Swap RGB to BGR for cv2
        bgr_color = (color[2], color[1], color[0])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr_color, 2)
        label = f"{cname} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), bgr_color, -1)
        cv2.putText(canvas, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_dino_boxes(image_bgr: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
    canvas = image_bgr.copy()
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        # Magenta / Violet for DINOv2 mask-derived box
        bgr_color = (180, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr_color, 2)
        label = "road_defect"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), bgr_color, -1)
        cv2.putText(canvas, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def render_single_image_assessment():
    st.markdown("## Single-Image Road Assessment")
    st.markdown(
        "Analyze individual road frames under supervised damage detection (YOLOv8n) "
        "and zero-shot foundation anomaly segmentation (DINOv2 + SAM2)."
    )

    # Mode Selector
    mode_col1, mode_col2 = st.columns([3, 1])
    with mode_col2:
        exec_mode = st.radio(
            "Execution Mode",
            ["Verified Demo Mode (Precomputed)", "Live YOLOv8n Inference Mode (GPU)"],
            index=0,
            help="Verified Demo Mode ensures fast, offline-safe demonstration using frozen assets. Live YOLOv8n Mode executes the frozen PyTorch detector on GPU."
        )

    curated_panels = get_curated_panel_images()
    benchmark_summary = load_benchmark_summary()
    exp_a_df = load_experiment_a_perception()
    per_image_recs = {r["image_id"]: r for r in benchmark_summary.get("per_image_records", [])}

    # Image Selector Tabs / Dropdown
    sample_options = [
        f"{item['id']} — {item['category']} ({item['source']})" for item in curated_panels
    ]
    if not sample_options:
        sample_options = ["China_Drone_001063", "China_Drone_000010", "China_Drone_002162"]

    selected_label = st.selectbox("Select Road Inspection Sample:", sample_options, index=0)
    selected_id = selected_label.split(" — ")[0].strip()

    # Find metadata
    panel_meta = next((item for item in curated_panels if item["id"] == selected_id), None)
    bench_rec = per_image_recs.get(selected_id, None)

    # Determine original image path
    val_img_path = WORKSPACE_ROOT / f"yolo/data/rdd2022/images/val/{selected_id}.jpg"
    cand_img_path = WORKSPACE_ROOT / f"benchmark/common_candidate/images/{selected_id}.jpg"
    if not cand_img_path.exists():
        cand_img_path = WORKSPACE_ROOT / f"benchmark/common_candidate/images/{selected_id}.png"

    active_img_path = val_img_path if val_img_path.exists() else cand_img_path

    if not active_img_path.exists():
        st.warning(f"Image asset for {selected_id} not found on disk at {active_img_path}. Falling back to panel montage.")
        if panel_meta and panel_meta.get("asset_path") and Path(panel_meta["asset_path"]).exists():
            st.image(panel_meta["asset_path"], caption="Precomputed 3-Panel Comparison")
        return

    raw_bgr = cv2.imread(str(active_img_path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # Prepare YOLO & DINO/SAM Results
    yolo_dets = []
    dino_boxes = []
    current_severity = None
    defect_count = 0
    defect_area_ratio = None
    surface_anomaly_score = None

    if exec_mode.startswith("Live") and torch_is_available():
        with st.spinner("Executing live YOLOv8n inference on GPU... (DINOv2+SAM2 uses frozen precomputed segmentations)"):
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(str(WORKSPACE_ROOT / "yolo/weights/best.pt"))
                y_res = yolo_model.predict(source=raw_bgr, conf=0.25, imgsz=512, device="0", verbose=False)[0]
                if y_res.boxes is not None:
                    for b, c, cl in zip(y_res.boxes.xyxy.cpu().numpy(), y_res.boxes.conf.cpu().numpy(), y_res.boxes.cls.cpu().numpy()):
                        cid = int(cl)
                        yolo_dets.append({
                            "class_name": CLASS_NAMES.get(cid, str(cid)),
                            "confidence": float(c),
                            "bbox": [float(v) for v in b],
                        })
            except Exception as e:
                st.error(f"Live YOLO inference error: {e}. Falling back to precomputed verified results.")
                yolo_dets = bench_rec.get("raw_yolo", []) if bench_rec else []
        if bench_rec:
            dino_boxes = bench_rec.get("raw_dino_boxes", [])
            defect_count = len(dino_boxes)
    else:
        # Precomputed Verified Demo
        if bench_rec:
            yolo_dets = bench_rec.get("raw_yolo", [])
            dino_boxes = bench_rec.get("raw_dino_boxes", [])
            defect_count = len(dino_boxes)

    # Check if this image has exact multi-modal severity from Experiment A
    if not exp_a_df.empty and selected_id in exp_a_df["image_id"].values:
        row_exp = exp_a_df[exp_a_df["image_id"] == selected_id].iloc[0]
        current_severity = float(row_exp["current_severity"])
        defect_count = int(row_exp["defect_count"])
        defect_area_ratio = float(row_exp["defect_area_ratio"])
        surface_anomaly_score = float(row_exp["surface_anomaly_score"])

    # If curated panel figure exists, check if pre-rendered panel can be shown
    panel_fig_path = None
    if panel_meta and panel_meta.get("asset_path"):
        p = Path(panel_meta["asset_path"])
        if p.exists():
            panel_fig_path = p

    # Render Side-by-Side Cards
    st.markdown("### Visual Perception Side-by-Side")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="image-card-header">1. Original Road Frame</div>', unsafe_allow_html=True)
        st.image(raw_rgb, use_container_width=True)
        st.caption(f"Source: {active_img_path.name} | Viewpoint: UAV / nadir")

    with col2:
        st.markdown('<div class="image-card-header">2. YOLOv8n Detections</div>', unsafe_allow_html=True)
        yolo_vis = draw_yolo_boxes(raw_bgr, yolo_dets) if yolo_dets else raw_rgb
        st.image(yolo_vis, use_container_width=True)
        st.caption(f"Supervised BBoxes: {len(yolo_dets)} detected (conf >= 0.25)")

    with col3:
        st.markdown('<div class="image-card-header">3. DINOv2 + SAM2 (Generic Defect)</div>', unsafe_allow_html=True)
        dino_vis = draw_dino_boxes(raw_bgr, dino_boxes) if dino_boxes else raw_rgb
        st.image(dino_vis, use_container_width=True)
        st.caption(f"Mask-Derived Boxes: {len(dino_boxes)} regions (class: road_defect)")

    # Metrics Row
    st.markdown("### Pavement State Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        sev_display = f"{current_severity:.4f}" if current_severity is not None else "N/A (Detection Benchmark)"
        st.metric("Current Severity Index", sev_display, delta=f"{defect_count} defect regions" if defect_count > 0 else "0 defects")
    with m2:
        st.metric("Defect Count", f"{defect_count}", delta=f"{len(yolo_dets)} YOLO boxes")
    with m3:
        area_display = f"{defect_area_ratio*100:.2f}%" if defect_area_ratio is not None else "N/A (BBox Only)"
        st.metric("Defect Area Ratio", area_display, delta="Pavement coverage" if defect_area_ratio is not None else None)
    with m4:
        anomaly_display = f"{surface_anomaly_score:.3f}" if surface_anomaly_score is not None else "N/A (Benchmark)"
        st.metric("Surface Anomaly Score", anomaly_display, delta="DINOv2 patch distance" if surface_anomaly_score is not None else None)

    # Pass current severity to session state for future forecast page if available
    if current_severity is not None:
        st.session_state["selected_current_severity"] = current_severity
    st.session_state["selected_image_id"] = selected_id

    # Educational Tooltip with Exact Severity Formula Description
    st.markdown("""
    <div class="callout-box">
        <strong>Educational Guide & Severity Calculation:</strong><br>
        • <strong>DINOv2</strong>: Compares vision transformer patch embeddings (14×14 px) against a clean asphalt memory bank to compute surface anomaly distances.<br>
        • <strong>SAM2</strong>: Refines anomalous candidate centroid prompts into detailed boundary segmentation masks.<br>
        • <strong>YOLOv8n</strong>: Directly predicts supervised damage classes (D00=Longitudinal, D10=Transverse, D20=Alligator, D40=Pothole, Repair) and rectangular bounding boxes.<br>
        • <strong>Severity Formula (Frozen Perception Pipeline)</strong>: Computed per defect as a weighted combination of defect area ($m^2$ or pixel mask extent), estimated depth (dynamically re-weighted when depth maps are unavailable in RGB-only mode), water hazard presence/confidence, and surrounding crack extent, scaled by detection confidence and normalized to $[0.0, 1.0]$. (RDD2022 validation frames evaluate bounding boxes only; multi-modal severity is marked N/A).
    </div>
    """, unsafe_allow_html=True)

    # Advanced Details Drawer
    with st.expander("Advanced Perception Diagnostics & Geometry"):
        st.json({
            "image_id": selected_id,
            "yolo_detections": yolo_dets,
            "dino_sam_mask_derived_boxes": dino_boxes,
            "current_severity": round(current_severity, 4) if current_severity is not None else "N/A",
            "defect_area_ratio": round(defect_area_ratio, 6) if defect_area_ratio is not None else "N/A",
            "verified_demo_mode": not exec_mode.startswith("Live"),
            "ground_truth_available": panel_meta.get("has_gt", True) if panel_meta else True,
        })


def torch_is_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
