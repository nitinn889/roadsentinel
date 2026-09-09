"""Cross-Domain Generalization & Zero-Shot Transfer Component for RoadSentinel Dashboard."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from data_loader import (
    WORKSPACE_ROOT,
    load_canonical_metrics,
    load_cross_domain_comparison_table,
    load_cross_domain_dataset,
    load_cross_domain_routing_table,
)

FIGURES_DIR = WORKSPACE_ROOT / "cross_domain/figures"


def render_cross_domain_view():
    st.markdown("## Cross-Domain Generalization & Zero-Shot Transfer")
    st.markdown(
        "Independent evaluation of frozen RoadSentinel perception pipelines and domain gating on the **RDD2022 India "
        "cross-domain dashcam benchmark** (300 forward-facing dashcam images, 652 ground-truth damage boxes)."
    )

    metrics = load_canonical_metrics()

    # 1. Scientific Disclaimer & Scope Boundary
    st.markdown("""
    <div class="callout-box warn">
        <strong>CRITICAL RESEARCH CONTEXT & DEFENSIBLE CLAIMS:</strong><br>
        • <strong>Domain Shift Scope</strong>: Evaluation transfers models trained/calibrated on top-down UAV drone surveys (China_Drone) to forward-facing oblique vehicle dashcams in unconstrained Indian traffic conditions.<br>
        • <strong>Defensible Generalization Claim</strong>: <em>"DINOv2 patch embeddings perfectly separated the evaluated China-Drone and India-dashcam benchmark domains (AUROC = 1.0000, margin +0.1158). This does NOT imply universal out-of-distribution detection across all unseen road environments."</em><br>
        • <strong>Perception Localization Finding</strong>: <em>"Zero-shot foundation anomaly segmentation (DINOv2+SAM2) did NOT solve cross-domain defect localization on the evaluated benchmark (F1 = 0.0000 at IoU ≥ 0.50)."</em>
    </div>
    """, unsafe_allow_html=True)

    # 2. Headline In-Domain vs Cross-Domain Comparison
    st.markdown("### 1. In-Domain vs Cross-Domain Perception Transfer")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("YOLO In-Domain F1", "0.7104", help="China_Drone validation split (N=480)")
    with col2:
        st.metric("YOLO Cross-Domain F1", "0.0218", delta="-96.9% Collapse", delta_color="inverse", help="India dashcam benchmark (N=300)")
    with col3:
        st.metric("DINOv2 Domain Separation", "1.0000", delta="AUROC (Perfect)", help="Zero overlap between China and India kNN distance distributions")
    with col4:
        st.metric("Unsafe Failure Quarantine", "100.0%", delta="300/300 Escalated", help="All evaluated India frames routed away from automated acceptance")

    # In vs Cross Table
    df_cmp = load_cross_domain_comparison_table()
    if not df_cmp.empty:
        st.dataframe(df_cmp, hide_index=True, use_container_width=True)

    st.markdown("---")

    # 3. DINOv2 Macro Domain Awareness & Distance Distributions
    st.markdown("### 2. DINOv2 Macro Domain Gating ($k$-NN Distance)")
    st.markdown(
        "DINOv2 ViT-S/14 384-dimensional CLS token embeddings compute nearest-neighbor cosine distance against "
        "the healthy training reference bank ($N=1,921$ China_Drone frames). On the evaluated datasets, domain shift "
        "is detected with complete statistical separation."
    )

    col_g1, col_g2 = st.columns([1, 1.2])
    with col_g1:
        st.markdown("""
        #### Empirical Domain Distance Distributions
        - **In-Domain (China Validation, $N=480$)**:
          - Mean Distance: `0.1841 ± 0.0847`
          - Median Distance: `0.1693`
          - Range: `[0.0393, 0.5247]`
          - $P_{95}$ Threshold: `0.3804`
          - $P_{99}$ Threshold: `0.4491`
        - **Cross-Domain (India Benchmark, $N=300$)**:
          - Mean Distance: `0.8432 ± 0.0441`
          - Median Distance: `0.8486`
          - Range: `[0.6405, 0.9261]`
        - **Separation Margin**: `+0.1158` (between China Max $P_{99}$ and India Min Distance)
        - **Domain Shift Identification**: **100.0%** of India frames exceed the $P_{99}$ operational shift threshold.
        """)

        st.markdown("""
        <div class="callout-box success">
            <strong>Key Research Finding:</strong><br>
            While DINOv2 patch tokens fail to localize small organic cracks reliably without supervised fine-tuning, 
            the global CLS representation serves as an <strong>exceptionally reliable Macro Domain Gate</strong>, 
            identifying severe sensor and geographic shifts prior to downstream decision routing.
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        p_ood = FIGURES_DIR / "fig3_ood_distribution_shift.png"
        if p_ood.exists():
            st.image(str(p_ood), caption="Figure 3: Empirical DINOv2 kNN Distance Separation between China and India Domains", use_container_width=True)

    st.markdown("---")

    # 4. Cross-Domain Failure Mode & Decision Quarantine
    st.markdown("### 3. Reliability & Decision Engine Cross-Domain Safety")
    st.markdown(
        "When applied to unseen domains where perception collapses, a naive system without reliability gating "
        "would accept severe misdetections. RoadSentinel routes out-of-distribution frames to `DOMAIN_ESCALATION`."
    )

    df_route = load_cross_domain_routing_table()
    if not df_route.empty:
        st.dataframe(df_route, hide_index=True, use_container_width=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        p_montage = FIGURES_DIR / "fig8_cross_domain_case_montage.png"
        if p_montage.exists():
            st.image(str(p_montage), caption="Figure 8: Curated Cross-Domain Transfer Failure Case Studies (India)", use_container_width=True)
    with col_f2:
        p_roc = FIGURES_DIR / "fig4_reliability_roc_cross_domain.png"
        if p_roc.exists():
            st.image(str(p_roc), caption="Figure 4: Cross-Domain Reliability ROC and Selective Risk Coverage", use_container_width=True)

    # 5. Full Dataset Browser
    with st.expander("Inspect Raw India Benchmark Dataset (300 frames)"):
        df_raw = load_cross_domain_dataset()
        if not df_raw.empty:
            st.dataframe(
                df_raw[["image_id", "gt_box_count", "yolo_pred_count", "image_f1_t0", "yolo_max_conf", "dino_ood_score", "dino_domain_flag"]],
                use_container_width=True
            )
