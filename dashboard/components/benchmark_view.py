"""YOLOv8n vs DINOv2+SAM2 Benchmark View Component for RoadSentinel Dashboard."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from data_loader import (
    BENCHMARK_DIR,
    WORKSPACE_ROOT,
    load_binary_metrics,
    load_final_perception_table,
    load_group_sensitive_metrics,
    load_panel_examples,
    load_yolo_semantic_metrics,
)

FIGURES_DIR = BENCHMARK_DIR / "figures"
PANELS_DIR = BENCHMARK_DIR / "panels"


def render_benchmark_view():
    st.markdown("## Perception Benchmark: YOLOv8n vs DINOv2 + SAM2")
    st.markdown(
        "Formal comparative evaluation conducted on the **RDD2022 China_Drone validation split** "
        "(480 images, 742 ground-truth damage boxes) evaluated on an **NVIDIA RTX 5060 Laptop GPU**."
    )

    # 1. Comparison Context Callout (Section 17)
    st.markdown("""
    <div class="callout-box warn">
        <strong>CRITICAL EVALUATION CONTEXT & SCIENTIFIC HONESTY:</strong><br>
        • <strong>YOLOv8n</strong>: Supervised detector fine-tuned directly on RDD2022 China_Drone road-damage bounding boxes.<br>
        • <strong>DINOv2 + SAM2</strong>: Unsupervised foundation anomaly localization comparing patch embeddings against a clean pavement memory bank, followed by zero-shot prompt segmentation.<br>
        • <strong>Benchmark Domain</strong>: Because this common validation benchmark consists of China_Drone images, the evaluation is <strong>substantially closer to YOLO's training distribution</strong>.<br>
        • <strong>Split Definition</strong>: This is a <strong>COMMON VALIDATION BENCHMARK</strong>, not an independent blind cross-domain test set.
    </div>
    """, unsafe_allow_html=True)

    # 2. Primary Comparative Metrics Table (Section 16)
    st.markdown("### 1. Primary Perception Performance Comparison")

    df_final = load_final_perception_table()
    df50, df25 = load_binary_metrics()

    if not df_final.empty:
        # Display clean styled table
        display_cols = [
            "Model", "Method Type", "Precision", "Recall", "F1", 
            "Mean Matched BBox IoU", "Mean Latency", "FPS", "Prediction Type"
        ]
        available_cols = [c for c in display_cols if c in df_final.columns]
        st.dataframe(
            df_final[available_cols],
            hide_index=True,
            use_container_width=True
        )

    # Key Metric Highlight Cards from Loaded Perception Artifacts
    df_group = load_group_sensitive_metrics()
    yolo_f1 = 0.7104
    dino_f1 = 0.0267
    yolo_lat = "3.62 ms"
    dino_lat = "263.15 ms"
    dup_f1 = 0.7022

    if not df_final.empty:
        yolo_rows = df_final[df_final["Model"].str.contains("YOLO", na=False)]
        dino_rows = df_final[df_final["Model"].str.contains("DINO", na=False)]
        if not yolo_rows.empty and "F1" in yolo_rows.columns:
            yolo_f1 = float(yolo_rows.iloc[0]["F1"])
            yolo_lat = str(yolo_rows.iloc[0].get("Mean Latency", "3.62 ms"))
        if not dino_rows.empty and "F1" in dino_rows.columns:
            dino_f1 = float(dino_rows.iloc[0]["F1"])
            dino_lat = str(dino_rows.iloc[0].get("Mean Latency", "263.15 ms"))

    if not df_group.empty and "f1_score" in df_group.columns:
        filtered_row = df_group[df_group["evaluation_stratum"].str.contains("Excluding_Connected", na=False)]
        if not filtered_row.empty:
            dup_f1 = float(filtered_row.iloc[0]["f1_score"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("YOLOv8n F1-Score", f"{yolo_f1:.4f}", delta=f"+{yolo_f1 - dino_f1:.4f} vs DINO/SAM")
    with c2:
        st.metric("Duplicate-Filtered China F1", f"{dup_f1:.4f}", delta="-0.0082 (N=456 disjoint)", help="Excludes 24 validation frames sharing pHash <= 5 with training set")
    with c3:
        st.metric("DINOv2+SAM2 F1-Score", f"{dino_f1:.4f}", delta=f"-{yolo_f1 - dino_f1:.4f} (High False Alarms)", delta_color="inverse")
    with c4:
        st.metric("YOLO Inference Latency", yolo_lat, delta="276.3 FPS on RTX 5060")

    st.caption("Direct DINOv2 + SAM2 defect-accuracy superiority has not been established on organic road distress. YOLOv8n dominates in-domain localization.")
    st.markdown("---")

    # 3. YOLO 5-Class Semantic Breakdown (Section 18)
    st.markdown("### 2. YOLOv8n Semantic Class Breakdown (IoU = 0.50)")
    st.markdown(
        "Analysis of YOLOv8n detection performance across the 5 Road Damage Dataset damage classes. "
        "Evaluates class imbalance, precision, recall, and localization quality."
    )

    df_yolo = load_yolo_semantic_metrics()
    if not df_yolo.empty:
        st.dataframe(
            df_yolo[["class_name", "gt_count", "predicted_count", "TP", "FP", "FN", "precision", "recall", "f1", "mean_matched_iou", "support_note"]],
            hide_index=True,
            use_container_width=True
        )

    col_note1, col_note2 = st.columns(2)
    with col_note1:
        st.markdown("""
        <div class="callout-box danger">
            <strong>D40 Low-Support Warning:</strong><br>
            <strong>D40 (Pothole)</strong> contains only <strong>15 ground truth instances</strong> in the validation set.
            Due to this low sample size, class-level metrics for D40 cannot support strong statistical conclusions.
        </div>
        """, unsafe_allow_html=True)

    with col_note2:
        st.markdown("""
        <div class="callout-box warn">
            <strong>D20 Structural Weakness:</strong><br>
            <strong>D20 (Alligator Fatigue Cracking)</strong> is the weakest meaningful class (<strong>F1 = 0.4259, Recall = 0.3966</strong>).
            Complex interwoven polygonal crack networks are frequently fragmented into partial boxes or missed.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Curated Qualitative Panels Browser (Section 19)
    st.markdown("### 3. Curated Qualitative Benchmark Panels")
    st.markdown(
        "Explore side-by-side comparative visualizations across distinct failure, success, and domain-shift scenarios. "
        "Each panel illustrates the original road scene alongside YOLO ground-truth/predictions and DINOv2+SAM2 segmentations."
    )

    panels_df = load_panel_examples()
    if not panels_df.empty:
        panel_options = [
            f"{row['image_id']} — {row['category']}" for _, row in panels_df.iterrows()
        ]
        selected_panel_str = st.selectbox("Select Benchmark Case Study:", panel_options, index=0)
        selected_img_id = selected_panel_str.split(" — ")[0].strip()

        row_match = panels_df[panels_df["image_id"] == selected_img_id].iloc[0]
        panel_asset_rel = row_match["asset_paths"]
        panel_asset_path = WORKSPACE_ROOT / panel_asset_rel if pd.notna(panel_asset_rel) else None

        # Display Panel Details
        col_p_info, col_p_meta = st.columns([2, 1])
        with col_p_info:
            st.markdown(f"**Case Category**: `{row_match['category']}` | **Source**: `{row_match['source']}`")
            st.markdown(f"**Research Rationale**: {row_match['panel_reason']}")
        with col_p_meta:
            st.markdown(f"**YOLO Metric**: `{row_match['yolo_summary']}`")
            st.markdown(f"**DINO/SAM Metric**: `{row_match['dino_sam_summary']}`")

        # Render panel image
        if panel_asset_path and panel_asset_path.exists():
            st.image(str(panel_asset_path), caption=f"Panel: {selected_img_id} — {row_match['category']}", use_container_width=True)
        else:
            st.warning(f"Panel image asset not found at {panel_asset_path}.")

    st.markdown("---")

    # 5. Scientific Research Figures
    st.markdown("### 4. Benchmark Publication Figures")
    fig_tabs = st.tabs([
        "PR & F1 Comparison", "Latency & Throughput", "BBox IoU Overlap", "YOLO Class Metrics"
    ])

    with fig_tabs[0]:
        p = FIGURES_DIR / "fig1_precision_recall_f1.png"
        if p.exists():
            st.image(str(p), caption="Figure 1: Precision, Recall, and F1 Comparison across YOLOv8n vs DINOv2+SAM2", use_container_width=True)

    with fig_tabs[1]:
        p = FIGURES_DIR / "fig2_latency_throughput_log.png"
        if p.exists():
            st.image(str(p), caption="Figure 2: GPU Latency and Frame Rate (Logarithmic Scale)", use_container_width=True)

    with fig_tabs[2]:
        p = FIGURES_DIR / "fig3_matched_bbox_iou.png"
        if p.exists():
            st.image(str(p), caption="Figure 3: True Positive Bounding Box IoU Distribution against Ground Truth", use_container_width=True)

    with fig_tabs[3]:
        p = FIGURES_DIR / "fig4_yolo_class_wise_metrics.png"
        if p.exists():
            st.image(str(p), caption="Figure 4: YOLOv8n Per-Class Precision, Recall, and F1 Breakdown", use_container_width=True)
