"""Domain Awareness & Perception Reliability Component for RoadSentinel Dashboard V2."""

from __future__ import annotations

import streamlit as st
import pandas as pd
from data_loader import (
    WORKSPACE_ROOT,
    load_canonical_metrics,
    load_domain_gate_table,
    load_reliability_ablation_table,
)


def render_reliability_analysis():
    st.markdown("## Domain Awareness & Perception Reliability")
    st.markdown(
        "Evaluate the two-tier perception safety framework: **DINOv2 Macro Domain Gating** at the input boundary "
        "and **YOLO Confidence-Based Reliability Estimation** at the sample prediction boundary."
    )

    canon = load_canonical_metrics()
    df_gate = load_domain_gate_table()
    df_ablation = load_reliability_ablation_table()

    # Two-Tier Concept Tabs
    tab_overview, tab_domain_gate, tab_sample_rel, tab_risk_cov = st.tabs([
        "Two-Tier Framework",
        "1. DINOv2 Domain Gating",
        "2. Sample Reliability Ablation",
        "3. Risk-Coverage Control"
    ])

    with tab_overview:
        st.markdown("### Decoupled Safety Architecture")
        st.markdown("""
        RoadSentinel separates perception failure defense into two mathematically distinct tasks:
        1. **Domain-Level Warning (DINOv2)**: *Does this image belong to the detector's operational design domain (ODD)?*
        2. **Sample-Level Failure Prediction (YOLO Confidence)**: *Is this specific bounding box detection likely to be correct?*
        """)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div style="background: #1e293b; padding: 18px; border-radius: 8px; border-left: 4px solid #8b5cf6;">
                <h4 style="color: #a78bfa; margin-top: 0;">Tier 1: Macro Domain Gate</h4>
                <p style="color: #cbd5e1; font-size: 13px;">
                    • <b>Mechanism:</b> DINOv2 ViT-S/14 Foundation Embeddings (kNN Cosine Distance).<br>
                    • <b>Domain AUROC:</b> <b>1.0000</b> (Zero distribution overlap).<br>
                    • <b>India Detection Rate:</b> <b>100.0%</b> (300/300 flagged at p99=0.4491).<br>
                    • <b>China False Alarm Rate:</b> <b>1.46%</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div style="background: #1e293b; padding: 18px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <h4 style="color: #60a5fa; margin-top: 0;">Tier 2: Sample-Level Reliability</h4>
                <p style="color: #cbd5e1; font-size: 13px;">
                    • <b>Mechanism:</b> Model B Calibrated YOLO Confidence Vector.<br>
                    • <b>In-Domain AUROC:</b> <b>0.8166 to 0.8650</b> across strict targets.<br>
                    • <b>Error Reduction:</b> Up to <b>88.4%</b> error reduction via selective rejection.<br>
                    • <b>HIGH Band Accuracy:</b> <b>91.97%</b> success rate under Target T1.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab_domain_gate:
        st.markdown("### DINOv2 Foundation Domain Shift Separation")
        if not df_gate.empty:
            st.dataframe(df_gate, use_container_width=True)

        st.markdown("""
        **Key Empirical Finding**: Raw DINOv2 kNN distances for Indian dashcam frames (mean $0.8432$, min $0.6405$) 
        never overlap with in-domain China drone frames (mean $0.1841$, max $0.5247$), yielding a clean separation margin of **+0.1158**.
        """)

        fig5_path = WORKSPACE_ROOT / "reliability_validation/figures/fig5_domain_distance_raw.png"
        fig6_path = WORKSPACE_ROOT / "reliability_validation/figures/fig6_domain_gate_roc.png"
        
        gcol1, gcol2 = st.columns(2)
        if fig5_path.exists():
            with gcol1:
                st.image(str(fig5_path), caption="Fig 5: DINO Raw Distance Distributions", use_container_width=True)
        if fig6_path.exists():
            with gcol2:
                st.image(str(fig6_path), caption="Fig 6: Domain Gate ROC & PR Curves (AUROC = 1.0000)", use_container_width=True)

    with tab_sample_rel:
        st.markdown("### Signal Ablation Study (Models A to H across Targets T0 to T4)")
        if not df_ablation.empty:
            st.dataframe(
                df_ablation[["model", "model_name", "target", "dataset", "AUROC", "AUROC_95CI_low", "AUROC_95CI_high", "AUPRC", "failure_F1"]],
                use_container_width=True
            )

        st.markdown("""
        **Ablation Insights**:
        - **Model B (YOLO Confidence)** is the dominant sample failure predictor ($\text{AUROC} = 0.8166$ on $T_1$).
        - **Model C (DINO OOD alone)** achieves near-random sample failure prediction ($\text{AUROC} = 0.5428$).
        - DINO embeddings act as a macro domain gate, not a sample-level defect detector.
        """)

        fig2_path = WORKSPACE_ROOT / "reliability_validation/figures/fig2_ablation_auroc.png"
        if fig2_path.exists():
            st.image(str(fig2_path), caption="Fig 2: Signal Ablation AUROC with 95% Bootstrap CIs", use_container_width=True)

    with tab_risk_cov:
        st.markdown("### Risk-Coverage Selective Prediction Analysis")
        st.markdown(r"""
        By rejecting the bottom $10\%$ to $50\%$ lowest reliability observations, automated inspection error 
        is drastically reduced from the baseline rate:
        """)


        fig4_path = WORKSPACE_ROOT / "reliability_validation/figures/fig4_risk_coverage_strict_target.png"
        if fig4_path.exists():
            st.image(str(fig4_path), caption="Fig 4: Risk-Coverage Error Reduction Curves under Strict Targets", use_container_width=True)
