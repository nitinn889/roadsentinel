"""Phase 1C Pavement Deterioration Forecasting & Horizon Analysis Component.

Integrates audited FHWA LTPP results:
- M0 Persistence vs M3 Temporal vs M4 Combined XGBoost
- Subgroup horizon & history depth analysis
- Out-of-fold empirical prediction behavior (113 LTPP pairs across 23 sites)
- Operational forecast router and scientific limitation disclaimers
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import (
    load_goal1_forecasts,
    load_phase1c_decision,
    load_phase1c_feasibility_summary,
    load_phase1c_forecast_skill,
    load_phase1c_model_comparison,
    load_phase1c_oof_predictions,
    load_primary_results,
)


def render_oof_behavior_plots(df_oof: pd.DataFrame):
    """Render 4 empirical out-of-fold behavior plots from real LTPP test pairs."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_color("#334155")

    # 1. Observed vs Predicted Future Severity
    ax1 = axes[0, 0]
    ax1.scatter(df_oof["future_severity_true"], df_oof["m4_xgb_combined_pred_future"], color="#38bdf8", alpha=0.75, edgecolors="#0284c7", s=45)
    lims1 = [
        min(df_oof["future_severity_true"].min(), df_oof["m4_xgb_combined_pred_future"].min()) - 0.05,
        max(df_oof["future_severity_true"].max(), df_oof["m4_xgb_combined_pred_future"].max()) + 0.05,
    ]
    ax1.plot(lims1, lims1, "--", color="#ef4444", alpha=0.8, label="Ideal 1:1 Identity")
    ax1.set_xlim(lims1)
    ax1.set_ylim(lims1)
    ax1.set_title("Observed vs Predicted Future Severity (M4 OOF, N=113)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("True Observed Future Severity", color="#94a3b8", fontsize=9.5)
    ax1.set_ylabel("M4 Predicted Future Severity", color="#94a3b8", fontsize=9.5)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    # 2. Observed vs Predicted Deterioration Change (Delta)
    ax2 = axes[0, 1]
    ax2.scatter(df_oof["delta_severity_true"], df_oof["m4_xgb_combined_pred_delta"], color="#a855f7", alpha=0.75, edgecolors="#7e22ce", s=45)
    lims2 = [
        min(df_oof["delta_severity_true"].min(), df_oof["m4_xgb_combined_pred_delta"].min()) - 0.02,
        max(df_oof["delta_severity_true"].max(), df_oof["m4_xgb_combined_pred_delta"].max()) + 0.02,
    ]
    ax2.plot(lims2, lims2, "--", color="#ef4444", alpha=0.8, label="Ideal 1:1 Identity")
    ax2.set_xlim(lims2)
    ax2.set_ylim(lims2)
    ax2.set_title("Observed vs Predicted Deterioration Change (Δ Severity)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("True Observed Change (Δ)", color="#94a3b8", fontsize=9.5)
    ax2.set_ylabel("M4 Predicted Change (Δ)", color="#94a3b8", fontsize=9.5)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    # 3. Residual Error vs Forecast Horizon
    ax3 = axes[1, 0]
    ax3.scatter(df_oof["observed_days_ahead"], df_oof["m4_error"], color="#f59e0b", alpha=0.75, edgecolors="#b45309", s=45)
    ax3.axhline(0, color="#ef4444", linestyle="--", alpha=0.7)
    ax3.set_title("Forecast Error vs Horizon (Days Ahead)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("Observed Horizon (Days)", color="#94a3b8", fontsize=9.5)
    ax3.set_ylabel("M4 Absolute Error |y - ŷ|", color="#94a3b8", fontsize=9.5)

    # 4. Residual Error vs History Depth (Prior Inspections)
    ax4 = axes[1, 1]
    depth_groups = [df_oof[df_oof["num_prior_inspections"] == k]["m4_error"].values for k in sorted(df_oof["num_prior_inspections"].unique())]
    labels_depth = [f"{k} Prior\n(N={len(vals)})" for k, vals in zip(sorted(df_oof["num_prior_inspections"].unique()), depth_groups)]
    
    bplot = ax4.boxplot(depth_groups, tick_labels=labels_depth, patch_artist=True, medianprops=dict(color="#f8fafc", linewidth=1.5))
    for patch in bplot["boxes"]:
        patch.set_facecolor("#0284c7")
        patch.set_alpha(0.7)
    ax4.set_title("Error Distribution vs History Depth", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax4.set_xlabel("Prior Inspection History Depth", color="#94a3b8", fontsize=9.5)
    ax4.set_ylabel("M4 Absolute Error", color="#94a3b8", fontsize=9.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_forecast():
    st.markdown("## Audited Pavement Deterioration Forecasting (Phase 1C)")
    st.markdown(
        "Rigorous benchmarking of scenario-conditioned and temporal machine-learning models "
        "against authoritative persistence baselines on the FHWA Long-Term Pavement Performance (LTPP) dataset."
    )

    # Top Metadata Banner
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px;">
        <span class="status-pill measured">AUDITED BENCHMARK</span>
        <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 8px;">
            Source: <code>artifacts/phase1c/model_comparison.csv</code> & <code>forecast_skill.json</code>
        </span>
        <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 6px;">
            Protocol: Nested Grouped Cross-Validation (5 outer folds, 4 inner folds, grouped by <code>site_id</code>). 
            Total observations: <b>113 LTPP transition pairs</b> across <b>23 highway sections</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_models = load_phase1c_model_comparison()
    dict_skill = load_phase1c_forecast_skill()
    df_oof = load_phase1c_oof_predictions()
    dict_feas = load_phase1c_feasibility_summary()
    dict_decision = load_phase1c_decision()

    # 1. Audited Forecasting Model Comparison (Requirement 6)
    st.markdown("### 1. Forecasting Model Comparison & Benchmark Baselines")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">M0 Persistence Baseline</div>
            <div class="metric-value" style="font-size: 1.5rem; color: #94a3b8;">MAE: 0.0582</div>
            <div class="metric-delta neutral">RMSE: 0.0815 | R²: 0.9125</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">M3 Temporal XGBoost</div>
            <div class="metric-value" style="font-size: 1.5rem; color: #38bdf8;">MAE: 0.0563</div>
            <div class="metric-delta positive">Skill: +3.26% (Point Estimate)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">M4 Temporal + Scenario XGBoost</div>
            <div class="metric-value" style="font-size: 1.5rem; color: #10b981;">MAE: 0.0551</div>
            <div class="metric-delta positive">Skill: +5.27% (Point Estimate)</div>
        </div>
        """, unsafe_allow_html=True)

    # Paired Site-Clustered 95% CI Box
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.1); border: 2px solid #f59e0b; border-radius: 12px; padding: 18px 22px; margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #fbbf24;">
                PAIRED SITE-CLUSTERED 95% CONFIDENCE INTERVAL (M4 minus Persistence MAE):
            </div>
            <span class="status-pill derived">CLUSTERED 95% CI</span>
        </div>
        <div style="font-family: monospace; font-size: 1.3rem; font-weight: 800; color: #fef3c7; margin-top: 6px;">
            [-0.0106, +0.0043]
        </div>
        <div style="font-size: 0.95rem; color: #fde68a; margin-top: 8px; line-height: 1.5;">
            <strong>Mandatory Scientific Interpretation:</strong> “The interval crosses zero; universal statistical superiority over persistence is not established.”
        </div>
        <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
            Point estimate paired difference: <code>Δ MAE = -0.0031</code>. Because the 95% confidence interval spans zero, 
            M4 cannot be claimed as universally superior across all road sites.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Full Model Ablation Table
    if not df_models.empty:
        st.markdown("#### Full Model Ablation Comparison Table")
        st.dataframe(df_models, hide_index=True, use_container_width=True)

    st.markdown("---")

    # 2. Horizon and History Analysis & Operational Router (Requirement 7)
    st.markdown("### 2. Horizon and History Subgroup Analysis & Operational Router")

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Short Horizon (&lt; 365 Days)</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #ef4444;">-3.77% Skill</div>
            <div class="metric-delta negative">N=43 | Persistence Wins</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Medium Horizon (365–550 Days)</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #10b981;">+9.12% Skill</div>
            <div class="metric-delta positive">N=35 | Point Estimate</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Long Horizon (&gt; 550 Days)</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #10b981;">+10.15% Skill</div>
            <div class="metric-delta positive">N=35 | Point Estimate</div>
        </div>
        """, unsafe_allow_html=True)
    with h4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">&ge; 2 Prior Inspections</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #10b981;">+9.18% Skill</div>
            <div class="metric-delta positive">N=78 | Point Estimate</div>
        </div>
        """, unsafe_allow_html=True)

    # Caution label for subgroups
    st.markdown("""
    <div class="callout-box warn">
        <strong>Subgroup Statistical Caveat:</strong> Sample counts are displayed above (N=43, N=35, N=35, N=78). 
        Do not mark subgroup improvements as statistically significant unless subgroup site-clustered confidence intervals exist.
    </div>
    """, unsafe_allow_html=True)

    # Operational Forecast Router
    st.markdown("#### Operational Forecast Routing Policy")
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.9); border: 1px solid #3b82f6; border-radius: 10px; padding: 18px 22px; margin-bottom: 20px;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #60a5fa; margin-bottom: 8px;">
            Audited Dual-Track Forecast Router:
        </div>
        <ul style="margin: 0; padding-left: 20px; color: #e2e8f0; line-height: 1.6; font-size: 0.95rem;">
            <li><strong>Short horizon (&lt; 365 days) OR insufficient history (&lt; 2 prior visits):</strong> Route directly to <code>M0_Persistence</code>. ML models exhibit negative skill (-3.77%) on short intervals.</li>
            <li><strong>Supported longer horizon (&ge; 365 days) AND adequate history (&ge; 2 prior visits):</strong> Route to <code>M4_XGB_Combined</code> residual scenario forecast.</li>
        </ul>
        <div style="margin-top: 12px; padding: 10px 14px; background: rgba(234, 179, 8, 0.12); border-left: 4px solid #eab308; border-radius: 4px; font-size: 0.88rem; color: #fef08a;">
            <strong>ROUTER VALIDATION NOTICE:</strong> The 365-day routing boundary is <strong>pending confirmation</strong> of whether it was predefined or independently validated on a hold-out test stratum.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Empirical Forecast Behavior (Requirement 8)
    st.markdown("### 3. Empirical Forecast Behavior (Out-of-Fold Test Records)")
    st.markdown(
        "All visual plots below are rendered directly from audited out-of-fold test predictions "
        "(`artifacts/phase1c/out_of_fold_predictions.csv`, 113 points across 23 sites). "
        "<strong>No synthetic or aggregate points are used.</strong>",
        unsafe_allow_html=True
    )

    if not df_oof.empty:
        render_oof_behavior_plots(df_oof)
    else:
        st.warning("Missing out-of-fold predictions artifact.")

    st.markdown("---")

    # 4. Forecast Dataset Information & Scope (Requirement 9)
    st.markdown("### 4. Forecast Dataset Information & Structural Parameters")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Transition Pairs", "113 LTPP pairs", help="Total paired survey intervals")
    with d2:
        st.metric("Highway Sections", "23 sections", help="Unique LTPP SHRP sections")
    with d3:
        st.metric("US States Represented", "6 US states", help="NY (50), AK (19), ND (18), HI (12), RI (9), LA (5)")
    with d4:
        st.metric("Forecast Horizons", "31 to 728 days", help="Mean horizon: 429.5 days; Median: 420.0 days")

    m_prior1, m_prior2 = st.columns(2)
    with m_prior1:
        st.metric("Pairs with ≥1 Prior Inspection", "90 pairs (79.6%)", help="Historical context available")
    with m_prior2:
        st.metric("Pairs with ≥2 Prior Inspections", "78 pairs (69.0%)", help="Full lag depth available")

    st.markdown("""
    <div style="background: rgba(239, 68, 68, 0.12); border: 2px solid #ef4444; border-radius: 10px; padding: 16px 20px; margin: 16px 0;">
        <div style="font-size: 1.05rem; font-weight: 700; color: #f87171;">
            MANDATORY SCOPE RESTRICTION:
        </div>
        <div style="font-size: 0.95rem; color: #fecaca; margin-top: 4px;">
            <strong>Do not claim validated 30–90-day forecasting.</strong> The empirical dataset reflects standard highway profile survey cycles 
            spanning 31 to 728 days (mean 429.5 days). Short-cycle high-frequency deterioration forecasting remains unvalidated.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Interactive Scenario Projections Explorer (Preserved functionality)
    st.markdown("---")
    st.markdown("### 5. Scenario Deterioration Projections (LTPP Pavement Segments)")
    df_fc = load_goal1_forecasts()
    df_prim = load_primary_results()

    if not df_fc.empty and not df_prim.empty:
        c_seg, c_day, c_horiz = st.columns(3)
        with c_seg:
            seg_id = st.selectbox("Select Road Segment", ["SEG_001", "SEG_002", "SEG_003", "SEG_004"], index=3, key="fc_seg")
        with c_day:
            day_num = st.selectbox("Select Inspection Day", list(range(1, 11)), index=4, key="fc_day", format_func=lambda d: f"Day {d:02d}")
        with c_horiz:
            horizon_val = st.selectbox("Forecast Horizon", [30, 60, 90], index=2, format_func=lambda h: f"{h} Days Ahead")

        sub_fc = df_fc[(df_fc["segment_id"] == seg_id) & (df_fc["day"] == day_num) & (df_fc["days_ahead"] == horizon_val)]
        prim_row = df_prim[(df_prim["segment_id"] == seg_id) & (df_prim["day"] == day_num)]

        if not sub_fc.empty and not prim_row.empty:
            current_sev = float(prim_row.iloc[0]["current_severity"])
            scen_cols = st.columns(5)
            scenarios = ["NORMAL", "HIGH_HEAT", "HEAVY_TRAFFIC", "HEAVY_RAIN", "WET_EXPOSURE"]

            for idx, scen in enumerate(scenarios):
                row = sub_fc[sub_fc["scenario"] == scen]
                if not row.empty:
                    pred_sev = float(row.iloc[0]["predicted_future_severity"])
                    pred_delta = float(row.iloc[0]["predicted_change"])
                    with scen_cols[idx]:
                        st.markdown(f"""
                        <div style="background: #1e293b; padding: 12px; border-radius: 8px; border-top: 4px solid {'#2563eb' if scen == 'WET_EXPOSURE' else '#64748b'}; text-align: center;">
                            <b style="color: {'#60a5fa' if scen == 'WET_EXPOSURE' else '#f8fafc'}; font-size: 12px;">{scen}</b>
                            <div style="font-size: 20px; font-weight: bold; color: #f8fafc; margin: 4px 0;">{pred_sev:.4f}</div>
                            <div style="color: #10b981; font-size: 11px;">Δ +{pred_delta:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
