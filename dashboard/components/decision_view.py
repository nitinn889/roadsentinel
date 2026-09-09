"""Decision Engine & Inspection Priority Component for RoadSentinel Dashboard."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from data_loader import (
    DECISION_ENGINE_DIR,
    WORKSPACE_ROOT,
    load_canonical_metrics,
    load_case_studies_table,
    load_decision_ablation_table,
    load_decision_policy_table,
    load_decisions_table,
    load_segment_decision_summary,
)

FIGURES_DIR = DECISION_ENGINE_DIR / "figures"


def render_decision_view():
    st.markdown("## Reliability-Aware Road-Health Decision Engine")
    st.markdown(
        "Synthesizes the four decoupled RoadSentinel evidence streams (**Current Severity**, "
        "**Observed Temporal Trend**, **Scenario Forecasts**, and **Perception Reliability**) "
        "into actionable, transparent inspection priorities."
    )

    df_decisions = load_decisions_table()
    metrics = load_canonical_metrics()

    # 1. Action Tiers Definition & Legend
    st.markdown("### 1. Five-Tier Decision Taxonomy")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #ef4444;">
            <div style="font-weight: 700; color: #ef4444; font-size: 0.95rem;">DOMAIN_ESCALATION</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Macro OOD detected; perception uncalibrated. Immediate manual field inspection required.</div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #f97316;">
            <div style="font-weight: 700; color: #f97316; font-size: 0.95rem;">PRIORITY_REVIEW</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">High severity or accelerating deterioration confirmed by high-reliability perception.</div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #eab308;">
            <div style="font-weight: 700; color: #eab308; font-size: 0.95rem;">REINSPECT</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Low-reliability perception or severe environmental confounds prevent autonomous confirmation.</div>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #3b82f6;">
            <div style="font-weight: 700; color: #3b82f6; font-size: 0.95rem;">MONITOR</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Low/moderate severity with stable temporal trend; scheduled standard resurvey cycle.</div>
        </div>
        """, unsafe_allow_html=True)
    with t5:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #10b981;">
            <div style="font-weight: 700; color: #10b981; font-size: 0.95rem;">AUTOMATED_ACCEPT</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Pristine pavement confirmed under high reliability with zero forecast risk.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Experiment A Decision Distribution
    st.markdown("### 2. Experiment A Distribution Summary (40 Physical Captures)")
    
    m_mon, m_re, m_prio, m_acc, m_esc = st.columns(5)
    with m_mon:
        st.metric("MONITOR", "21 captures (52.5%)", help="Standard scheduled monitoring cycle")
    with m_re:
        st.metric("REINSPECT", "10 captures (25.0%)", help="Low reliability or weather confounds")
    with m_prio:
        st.metric("PRIORITY_REVIEW", "9 captures (22.5%)", help="High confirmed distress or rapid growth")
    with m_acc:
        st.metric("AUTOMATED_ACCEPT", "0 captures (0.0%)", help="Conservative thresholding requires pristine multi-day evidence")
    with m_esc:
        st.metric("DOMAIN_ESCALATION", "0 captures (0.0%)", help="All Experiment A captures remain in-domain")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        p_dist = FIGURES_DIR / "fig2_experiment_a_decision_distribution.png"
        if p_dist.exists():
            st.image(str(p_dist), caption="Figure 2: Experiment A Decision Action Tier Breakdown", use_container_width=True)
    with col_fig2:
        p_time = FIGURES_DIR / "fig3_segment_decision_timelines.png"
        if p_time.exists():
            st.image(str(p_time), caption="Figure 3: Multi-Day Decision State Trajectories (SEG_001 to SEG_004)", use_container_width=True)

    st.markdown("---")

    # 3. Interactive Capture Decision Inspector & Explanation Card
    st.markdown("### 3. Explainable Decision Inspector")
    st.markdown("Select any physical capture to examine its complete multi-evidence synthesis and diagnostic explanation.")

    if not df_decisions.empty:
        capture_list = df_decisions["image_id"].tolist()
        selected_cap = st.selectbox("Select Inspection Capture:", capture_list, index=capture_list.index("SEG_004_day_05") if "SEG_004_day_05" in capture_list else 0)

        row = df_decisions[df_decisions["image_id"] == selected_cap].iloc[0]

        # Styled Explanation Card
        tier_colors = {
            "PRIORITY_REVIEW": "#f97316",
            "REINSPECT": "#eab308",
            "MONITOR": "#3b82f6",
            "DOMAIN_ESCALATION": "#ef4444",
            "AUTOMATED_ACCEPT": "#10b981",
        }
        dec_color = tier_colors.get(row["decision"], "#38bdf8")

        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.9); border: 2px solid {dec_color}; border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.4rem; font-weight: 800; color: {dec_color}; font-family: 'Outfit', sans-serif;">
                    {row['decision']}
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8;">Capture: <code>{row['image_id']}</code></div>
            </div>
            <div style="margin-top: 12px; font-size: 1.05rem; color: #f8fafc; line-height: 1.5;">
                <strong>Diagnostic Explanation:</strong><br>
                <em>"{row['explanation']}"</em>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 0.85rem;">
                <div><span style="color: #94a3b8;">Current Severity:</span><br><strong style="color: #38bdf8;">{row['current_severity']:.4f} ({row['severity_band']})</strong></div>
                <div><span style="color: #94a3b8;">Reliability Score:</span><br><strong style="color: #a855f7;">{row['reliability_score']:.4f} ({row['reliability_band']})</strong></div>
                <div><span style="color: #94a3b8;">Temporal Evidence:</span><br><strong style="color: #34d399;">{row['temporal_trend']}</strong></div>
                <div><span style="color: #94a3b8;">Peak 90d Scenario:</span><br><strong style="color: #f59e0b;">{row['dominant_scenario']} ({row['forecast_delta_90d']:+.4f})</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"**Primary Decision Driver**: {row['primary_reason']}")
        with col_r2:
            st.markdown(f"**Secondary Contributing Factor**: {row['secondary_reason']}")

    st.markdown("---")

    # 4. Decision Ablation & Policy Benchmarking
    st.markdown("### 4. Decision Engine Ablation Study (Systems A to E)")
    st.markdown(
        "Ablation testing demonstrates the essential safety role of multi-evidence fusion compared to naive single-signal heuristics."
    )

    df_abl = load_decision_ablation_table()
    if not df_abl.empty:
        st.dataframe(df_abl, hide_index=True, use_container_width=True)

    # 5. Full Decisions Database Table
    with st.expander("View Full Road Health Decisions Table (All 40 Captures)"):
        st.dataframe(df_decisions, use_container_width=True)
