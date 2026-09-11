"""RoadSentinel — Examiner-Facing Interactive Research Dashboard V2.

Comprehensive integration platform connecting:
1. System Overview & Architecture Flow
2. Single-Image Road Assessment (All 40 Captures + Live GPU Option)
3. Domain Awareness & Perception Reliability (Phase 12 Two-Tier Safety)
4. XGBoost Future Deterioration Forecasting (Phase 11 Scenarios)
5. Temporal Road Monitoring & Model-Observed Change (Phase 4 Multi-Day Sequences)
6. Perception Benchmark: YOLOv8n vs DINOv2+SAM2 (Phase 5/6 Comparison)
7. Cross-Domain Generalization & Zero-Shot Transfer (Phase 10 India Evaluation)
8. Reliability-Aware Decision & Inspection Priority Engine (Phase 13 Synthesis)
9. Failure Mode & Sensitivity Taxonomy (Qualitative & IoU Diagnostics)
10. Research Methodology & Scientific Protocols
11. Controlled Experiment B Protocol (Planned Next Phase)
12. Raspberry Pi Edge Deployment Roadmap (Pending Hardware Phase)

Launch command:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure dashboard root and components are in sys.path
DASHBOARD_ROOT = Path(__file__).resolve().parent
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

COMPONENTS_DIR = DASHBOARD_ROOT / "components"
if str(COMPONENTS_DIR) not in sys.path:
    sys.path.insert(0, str(COMPONENTS_DIR))

# Import all page components
from overview import render_overview
from single_image import render_single_image_assessment
from reliability_view import render_reliability_analysis
from forecast import render_forecast
from temporal import render_temporal_monitoring
from benchmark_view import render_benchmark_view
from cross_domain_view import render_cross_domain_view
from decision_view import render_decision_view
from failure_modes import render_failure_modes
from methodology import render_methodology
from experiment_b import render_experiment_b
from edge_deployment import render_edge_deployment


def inject_custom_css():
    """Inject custom glassmorphic styling."""
    css_path = DASHBOARD_ROOT / "assets/style.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="RoadSentinel — Research Dashboard V2",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 4px 14px 4px; border-bottom: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 14px;">
            <div style="font-size: 1.4rem; font-weight: 800; color: #38bdf8; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em;">
                RoadSentinel V2
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">
                Final Research Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = [
            "1. System Overview",
            "2. Single-Image Assessment",
            "3. Domain & Reliability",
            "4. XGBoost Future Forecast",
            "5. Temporal Change Analysis",
            "6. YOLO vs DINO/SAM Research",
            "7. Cross-Domain Generalization",
            "8. Decision & Inspection Priority",
            "9. Failure Mode Taxonomy",
            "10. Research Methodology",
            "11. Controlled Experiment B [PLANNED]",
            "12. Edge Deployment [PENDING]",
        ]

        selected_page = st.radio(
            "Navigation Sections",
            pages,
            index=0,
            key="main_nav",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Sidebar Live Status & System Diagnostic Badges
        st.markdown("##### Audited Pipeline Status")
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.8rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">Perception Core</span>
                <span class="status-pill frozen" style="font-size: 0.68rem; padding: 2px 7px;">FROZEN</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">DINOv2 Gate</span>
                <span class="status-pill complete" style="font-size: 0.68rem; padding: 2px 7px;">AUROC 1.000</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">Reliability (80% cov)</span>
                <span class="status-pill complete" style="font-size: 0.68rem; padding: 2px 7px;">9.38% FAIL</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">CARLA Temporal</span>
                <span class="status-pill complete" style="font-size: 0.68rem; padding: 2px 7px;">48 TRACKS</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">M4 XGBoost Skill</span>
                <span class="status-pill complete" style="font-size: 0.68rem; padding: 2px 7px;">+5.27% (POINT)</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">Policy Safety</span>
                <span class="status-pill complete" style="font-size: 0.68rem; padding: 2px 7px;">0 UNSAFE (IND)</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">Pi 5 Edge Deploy</span>
                <span class="status-pill pending" style="font-size: 0.68rem; padding: 2px 7px;">PENDING PROFILING</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("RoadSentinel v2.0 | Canonical Phase 14 Package\nOffline-ready & deterministic empirical records.")

    # Page Routing
    if selected_page.startswith("1."):
        render_overview()
    elif selected_page.startswith("2."):
        render_single_image_assessment()
    elif selected_page.startswith("3."):
        render_reliability_analysis()
    elif selected_page.startswith("4."):
        render_forecast()
    elif selected_page.startswith("5."):
        render_temporal_monitoring()
    elif selected_page.startswith("6."):
        render_benchmark_view()
    elif selected_page.startswith("7."):
        render_cross_domain_view()
    elif selected_page.startswith("8."):
        render_decision_view()
    elif selected_page.startswith("9."):
        render_failure_modes()
    elif selected_page.startswith("10."):
        render_methodology()
    elif selected_page.startswith("11."):
        render_experiment_b()
    elif selected_page.startswith("12."):
        render_edge_deployment()


if __name__ == "__main__":
    main()
