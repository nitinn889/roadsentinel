"""System Overview Component for RoadSentinel Dashboard."""

from __future__ import annotations

import streamlit as st
from data_loader import load_perception_manifest, load_final_perception_table, load_binary_metrics


def render_overview():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">RoadSentinel — Road Perception & Deterioration Intelligence</div>
        <div class="sub-title">
            Examiner-Facing Interactive Dashboard integrating Supervised Damage Detection (YOLOv8n),
            Zero-Shot Foundation Anomaly Perception (DINOv2 + SAM2), Multi-Day Temporal Analytics,
            and Scenario-Conditioned Deterioration Forecasting (XGBoost Model V2).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status Badges
    st.markdown("### System Pipeline Status")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="status-pill frozen">PERCEPTION — FROZEN</div>', unsafe_allow_html=True)
        st.caption("YOLOv8n & DINOv2/SAM2")
    with col2:
        st.markdown('<div class="status-pill complete">TEMPORAL — COMPLETE</div>', unsafe_allow_html=True)
        st.caption("Phase 4 Evidence Locked")
    with col3:
        st.markdown('<div class="status-pill complete">FORECAST — COMPLETE</div>', unsafe_allow_html=True)
        st.caption("XGBoost Model V2")
    with col4:
        st.markdown('<div class="status-pill pending">EDGE PI 5 — PENDING</div>', unsafe_allow_html=True)
        st.caption("Phase 8 Deployment")
    with col5:
        st.markdown('<div class="status-pill caution">EXP B — PENDING</div>', unsafe_allow_html=True)
        st.caption("Intervention Metadata")

    st.markdown("---")

    # High-level architecture flow
    st.markdown("### End-to-End System Architecture")
    st.markdown("""
    ```
    +---------------------------------------------------------------------------------------------------+
    |                                   RAW ROAD CAMERA / UAV SURVEY                                    |
    +-------------------------------------------------+-------------------------------------------------+
                                                      |
                             +------------------------+------------------------+
                             |                                                 |
                             v                                                 v
             +-------------------------------+                 +-------------------------------+
             |      SUPERVISED DETECTOR      |                 |    FOUNDATION ANOMALY MODEL   |
             |       YOLOv8n (512x512)       |                 |       DINOv2 + SAM2.1         |
             +---------------+---------------+                 +---------------+---------------+
                             |                                                 |
             * Crack Bounding Boxes (D00/D10/D20)               * Defect Segmentation Masks
             * Pothole Detection (D40)                         * Mask-Derived Bounding Boxes
             * Repaired Pavement Surface                       * Surface Anomaly Score
             * Fast Inference: 3.6ms (276 FPS)                 * Zero-Shot Viewpoint Invariance
                             |                                                 |
                             +------------------------+------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |    CURRENT SEVERITY ENGINE    |
                                      |   Normalized Index [0.0 - 1.0]|
                                      +---------------+---------------+
                                                      |
                             +------------------------+------------------------+
                             |                                                 |
                             v                                                 v
             +-------------------------------+                 +-------------------------------+
             |     TEMPORAL CORRESPONDENCE   |                 |    SCENARIO FORECAST MODEL    |
             |       Phase 4 Analytics       |                 |       XGBoost Model V2        |
             +---------------+---------------+                 +---------------+---------------+
             * Mask/BBox IoU Matching                          * Climate: Rain, Temp, Moisture
             * Multi-Day Track Persistence                     * Structural: Traffic Volume
             * Event Classification                            * Monotone Severity Constraints
             * Environmental Confound Analysis                 * Horizon Forecast: 30-90 Days
    +---------------------------------------------------------------------------------------------------+
    ```
    """)

    # Quick Stats Metric Cards
    st.markdown("### Core Empirical Findings at a Glance")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">In-Domain Benchmark F1</div>
            <div class="metric-value" style="color: #38bdf8;">0.7104</div>
            <div class="metric-delta positive">YOLOv8n (vs DINO/SAM 0.0267)</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Inference Latency Ratio</div>
            <div class="metric-value" style="color: #34d399;">72.7×</div>
            <div class="metric-delta positive">YOLO 3.6ms vs DINO/SAM 263ms</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Control Sequence CV</div>
            <div class="metric-value" style="color: #a855f7;">0.67%</div>
            <div class="metric-delta neutral">SEG_003 7-day stability test</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Observed Progression</div>
            <div class="metric-value" style="color: #f59e0b;">+0.7302</div>
            <div class="metric-delta positive">SEG_004 D01–D05 deterioration</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-box">
        <strong>Examiner Note:</strong> Use the sidebar navigation on the left to explore specific research areas.
        Every section is powered by frozen research data and offline-verified artifacts to ensure 100% demo reliability.
    </div>
    """, unsafe_allow_html=True)
