"""Single-Image Road Assessment Component for RoadSentinel Dashboard V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch

from data_loader import (
    WORKSPACE_ROOT,
    get_image_path,
    load_decisions_table,
    load_primary_results,
)

YOLO_COLORS = {
    "D00": (255, 140, 0),     # Orange: Longitudinal Crack
    "D10": (255, 215, 0),     # Gold: Transverse Crack
    "D20": (255, 0, 255),     # Magenta: Alligator Crack
    "D40": (255, 0, 0),       # Red: Pothole
    "Repair": (0, 255, 100),  # Green: Repair Patch
}

TIER_COLORS = {
    "AUTOMATED_ACCEPT": "#16a34a",
    "MONITOR": "#2563eb",
    "REINSPECT": "#f59e0b",
    "PRIORITY_REVIEW": "#dc2626",
    "DOMAIN_ESCALATION": "#7c3aed",
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
        bgr_color = (color[2], color[1], color[0])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr_color, 2)
        label = f"{cname} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), bgr_color, -1)
        cv2.putText(canvas, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def render_single_image_assessment():
    st.markdown("## Single-Image Road Assessment")
    st.markdown(
        "Analyze individual physical road frames under **CURRENT MODEL ASSESSMENT**, "
        "YOLOv8n supervised distress detection, DINOv2 domain awareness, perception reliability, and decision triage."
    )

    df_dec = load_decisions_table()
    df_prim = load_primary_results()

    # Controls: Segment & Day Selectors
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 2])
    
    with ctrl_col1:
        seg_id = st.selectbox("Select Road Segment", ["SEG_001", "SEG_002", "SEG_003", "SEG_004"], index=3)
    with ctrl_col2:
        day_val = st.selectbox("Select Inspection Day", [f"Day {d:02d}" for d in range(1, 11)], index=4)
        day_num = int(day_val.split()[1])
    with ctrl_col3:
        exec_mode = st.radio(
            "Execution Mode",
            ["Verified Demo Mode (Precomputed)", "Live YOLOv8n GPU Inference"],
            index=0,
            help="Verified Demo Mode ensures fast, offline-safe operation from frozen research assets."
        )

    # Fetch Data Records
    dec_row = df_dec[(df_dec["segment"] == seg_id) & (df_dec["day"] == day_num)]
    prim_row = df_prim[(df_prim["segment_id"] == seg_id) & (df_prim["day"] == day_num)]

    if dec_row.empty or prim_row.empty:
        st.error(f"No records found for {seg_id} {day_val}.")
        return

    rec_dec = dec_row.iloc[0]
    rec_prim = prim_row.iloc[0]

    img_path = get_image_path(seg_id, day_num)
    if not img_path or not img_path.exists():
        st.warning(f"Image asset not found at expected path for {seg_id} {day_val}.")
        return

    # Image Display & Inference
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Live YOLO Inference if requested
    detections = []
    if "Live" in exec_mode:
        try:
            from ultralytics import YOLO
            yolo_weights = WORKSPACE_ROOT / "yolo/weights/best.pt"
            model = YOLO(str(yolo_weights))
            results = model(str(img_path), conf=0.25, imgsz=512, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls.item())
                conf_val = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                cname = {0: "D00", 1: "D10", 2: "D20", 3: "D40", 4: "Repair"}.get(cls_id, f"Class_{cls_id}")
                detections.append({"class_name": cname, "confidence": conf_val, "bbox": xyxy})
            display_img = draw_yolo_boxes(img_bgr, detections)
            st.caption(f"⚡ Live YOLOv8n GPU Inference: {len(detections)} distress detections found.")
        except Exception as err:
            st.warning(f"Live YOLO inference unavailable ({err}); displaying raw capture.")
            display_img = img_rgb
    else:
        # Precomputed visualization
        display_img = img_rgb

    # Main Visual Layout
    img_col, meta_col = st.columns([3, 2])

    with img_col:
        st.image(display_img, caption=f"{seg_id} — {day_val} ({rec_prim['camera_preset']})", use_container_width=True)

    with meta_col:
        st.markdown("### CURRENT MODEL ASSESSMENT")
        
        # Primary Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Model Severity", f"{rec_dec['current_severity']:.4f}", help="Marion Day-5 normalized severity score [0.0 to 1.0]")
            st.caption(f"Band: **{rec_dec['current_severity_band']}**")
        with m2:
            st.metric("Defect Count", int(rec_dec['defect_count']), help="Count of candidate distress regions")
            st.caption(f"Area Ratio: **{rec_dec['defect_area_ratio']:.6f}**")

        st.markdown("---")

        # Anomaly & Environmental Flags
        a1, a2 = st.columns(2)
        with a1:
            st.metric("Surface Anomaly Score", f"{rec_prim['surface_anomaly_score']:.4f}", help="DINOv2 patch embedding anomaly deviation")
            st.caption(f"State: **{rec_prim['road_health_state']}**")
        with a2:
            water_status = "DETECTED" if rec_prim['water_flag'] else "CLEAR"
            st.metric("Water Flag", water_status, help="Standing water or specular moisture reflection flag")
            st.caption(f"Lighting: **{rec_prim['lighting_preset']}**")

        st.markdown("---")

        # Reliability & Domain
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Perception Reliability", f"{rec_dec['reliability_score']:.4f}", help="Calibrated confidence reliability score [0.0 to 1.0]")
            st.caption(f"Reliability Band: **{rec_dec['reliability_band']}**")
        with r2:
            st.metric("DINO Domain Status", rec_dec['domain_status'], help="Macro domain familiarity check")
            st.caption(f"Raw Distance: **{rec_dec['raw_domain_distance']:.4f}** (Limit: 0.4491)")


    st.markdown("---")

    # Decision Engine Output Card
    dec_val = rec_dec["decision"]
    tier_col = TIER_COLORS.get(dec_val, "#2563eb")

    st.markdown(f"""
    <div style="background: #1e293b; border: 2px solid {tier_col}; border-radius: 10px; padding: 20px; margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; font-weight: bold; color: #94a3b8; text-transform: uppercase;">Decision Engine Recommended Action</span>
            <span style="background: {tier_col}; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 13px;">{dec_val}</span>
        </div>
        <div style="color: #f8fafc; font-size: 15px; font-weight: 500; margin-top: 12px;">
            <b>Primary Rationale:</b> {rec_dec['primary_reason']}
        </div>
        <div style="color: #cbd5e1; font-size: 13px; margin-top: 8px;">
            <b>Secondary Context:</b> {rec_dec['secondary_reason']}
        </div>
        <div style="color: #94a3b8; font-size: 12px; margin-top: 8px; font-style: italic;">
            <b>Limitations & Disclaimers:</b> {rec_dec['limitations']}
        </div>
    </div>
    """, unsafe_allow_html=True)
