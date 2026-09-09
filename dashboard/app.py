"""RoadSentinel — Examiner-Facing Interactive Research Dashboard.

Phase 7 primary entrypoint integrating Supervised Damage Detection (YOLOv8n),
Zero-Shot Foundation Anomaly Perception (DINOv2 + SAM2), Multi-Day Temporal
Tracking, and Scenario-Conditioned Deterioration Forecasting (XGBoost Model V2).

Launch command:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure dashboard root is in sys.path
DASHBOARD_ROOT = Path(__file__).resolve().parent
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

# Ensure components directory is in sys.path
COMPONENTS_DIR = DASHBOARD_ROOT / "components"
if str(COMPONENTS_DIR) not in sys.path:
    sys.path.insert(0, str(COMPONENTS_DIR))

# Import page components
from overview import render_overview
from single_image import render_single_image_assessment
from forecast import render_future_forecast
from temporal import render_temporal_monitoring
from benchmark_view import render_benchmark_view
from failure_modes import render_failure_modes
from edge_deployment import render_edge_deployment
from methodology import render_methodology


def inject_custom_css():
    """Inject custom glassmorphic styling."""
    css_path = DASHBOARD_ROOT / "assets/style.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="RoadSentinel — Road Perception & Deterioration Dashboard",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="padding: 12px 4px 18px 4px; border-bottom: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 16px;">
            <div style="font-size: 1.45rem; font-weight: 700; color: #38bdf8; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em;">
                RoadSentinel
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">
                Examiner Research Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = [
            "1. System Overview",
            "2. Single-Image Road Assessment",
            "3. Future Condition Forecast",
            "4. Temporal Road Monitoring",
            "5. YOLO vs DINOv2+SAM2",
            "6. Failure Mode Analysis",
            "7. Edge Deployment",
            "8. Research / Methodology",
        ]

        selected_page = st.radio(
            "Navigation Sections",
            pages,
            index=0,
            key="main_nav",
            label_visibility="collapsed"
        )

        st.markdown("---")

        # System Diagnostic & Freeze Badges in Sidebar
        st.markdown("##### System Status")
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.82rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">Perception</span>
                <span class="status-pill frozen" style="font-size: 0.7rem; padding: 2px 8px;">FROZEN</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">Temporal Tracking</span>
                <span class="status-pill complete" style="font-size: 0.7rem; padding: 2px 8px;">LOCKED</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">Forecast Model</span>
                <span class="status-pill complete" style="font-size: 0.7rem; padding: 2px 8px;">V2 FROZEN</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8;">Pi 5 Edge Deploy</span>
                <span class="status-pill pending" style="font-size: 0.7rem; padding: 2px 8px;">PENDING</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("RoadSentinel v1.0 | Phase 7 Examiner Package\nAll models, datasets, & benchmark metrics frozen.")

    # Page Routing
    if selected_page.startswith("1."):
        render_overview()
    elif selected_page.startswith("2."):
        render_single_image_assessment()
    elif selected_page.startswith("3."):
        render_future_forecast()
    elif selected_page.startswith("4."):
        render_temporal_monitoring()
    elif selected_page.startswith("5."):
        render_benchmark_view()
    elif selected_page.startswith("6."):
        render_failure_modes()
    elif selected_page.startswith("7."):
        render_edge_deployment()
    elif selected_page.startswith("8."):
        render_methodology()


if __name__ == "__main__":
    main()
