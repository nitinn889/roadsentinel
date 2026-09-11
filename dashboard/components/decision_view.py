"""Decision Engine & Inspection Priority Component for RoadSentinel Dashboard.

Implements Audited Policy Ablation (India Benchmark, China In-Domain, and Pooled Benchmark)
strictly loading machine-readable artifacts from artifacts/phase1b/policy_ablation.csv.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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
    load_policy_ablation_table,
    load_segment_decision_summary,
)

FIGURES_DIR = DECISION_ENGINE_DIR / "figures"


def render_policy_destination_chart(df_stratum: pd.DataFrame, stratum_name: str):
    """Render a clean stacked destination bar chart for policy ablation."""
    architectures = df_stratum["system_architecture"].tolist()
    
    # Map architectures to cleaner display names
    arch_labels = {
        "YOLO_ONLY": "YOLO-Only\nBaseline",
        "YOLO_PLUS_RELIABILITY": "YOLO +\nSelective Pred",
        "YOLO_PLUS_DOMAIN_GATE": "YOLO +\nDomain Gate",
        "YOLO_PLUS_DOMAIN_GATE_PLUS_RELIABILITY": "YOLO + Gate\n+ Reliability",
        "FULL_ROADSENTINEL_POLICY": "Full Integrated\nPolicy",
    }
    labels = [arch_labels.get(a, a) for a in architectures]
    
    auto_accept = df_stratum["automated_accept_count"].to_numpy()
    human_rev = df_stratum["human_review_count"].to_numpy()
    domain_esc = df_stratum["domain_escalation_count"].to_numpy()
    unsafe_acc = df_stratum["unsafe_automated_accepts"].to_numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f172a", gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")
    
    x = np.arange(len(labels))
    width = 0.55
    
    # Left plot: Decision destinations stacked
    p1 = ax1.bar(x, auto_accept, width, label="Automated Accept", color="#10b981", alpha=0.9)
    p2 = ax1.bar(x, human_rev, width, bottom=auto_accept, label="Human Review", color="#eab308", alpha=0.9)
    p3 = ax1.bar(x, domain_esc, width, bottom=auto_accept + human_rev, label="Domain Escalation (Reject)", color="#ef4444", alpha=0.9)
    
    ax1.set_title(f"Decision Destinations Breakdown ({stratum_name})", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Inspection Captures", color="#94a3b8", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="#cbd5e1", fontsize=8.5)
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle="--", alpha=0.15, color="#94a3b8", axis="y")
    leg1 = ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5, loc="upper right")
    
    # Right plot: Unsafe Automated Accepts count
    colors_unsafe = ["#ef4444" if u > 0 else "#10b981" for u in unsafe_acc]
    bars2 = ax2.bar(x, unsafe_acc, width=0.5, color=colors_unsafe, alpha=0.9)
    ax2.set_title("Unsafe Automated Accepts (Must Be 0)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax2.set_ylabel("Unsafe Accepts Count", color="#94a3b8", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#cbd5e1", fontsize=8.5)
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle="--", alpha=0.15, color="#94a3b8", axis="y")
    
    # Add count labels on bars
    for bar, val in zip(bars2, unsafe_acc):
        ax2.annotate(
            f"{int(val)}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            color="#f8fafc", fontsize=9, fontweight="bold"
        )
        
    for spine in ["top", "right", "bottom", "left"]:
        ax1.spines[spine].set_color("#334155")
        ax2.spines[spine].set_color("#334155")
        
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_decision_view():
    st.markdown("## Reliability-Aware Policy Ablation & Decision Engine")
    st.markdown(
        "Synthesizes the four decoupled RoadSentinel evidence streams (**Current Severity**, "
        "**Observed Temporal Trend**, **Scenario Forecasts**, and **Perception Reliability**) "
        "into audited, fail-safe inspection priorities."
    )

    # 1. Audited Policy Ablation Section (Phase 1B Requirements)
    st.markdown("### 1. Audited Policy Routing Ablation (Decision Destinations)")
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px;">
        <span class="status-pill measured">AUDITED BENCHMARK</span>
        <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 8px;">
            Source: <code>artifacts/phase1b/policy_ablation.csv</code>
        </span>
        <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 6px;">
            Evaluates safety routing behavior across 5 distinct system architectures on <b>300 India Dashcam</b> stress test captures, 
            <b>480 China UAV</b> in-domain captures, and the <b>780 Pooled Benchmark</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_policy = load_policy_ablation_table()
    if df_policy.empty:
        st.error("Missing machine-readable policy ablation artifact at `artifacts/phase1b/policy_ablation.csv`.")
        return

    # Stratum Selector
    available_strata = df_policy["evaluation_stratum"].unique().tolist()
    selected_stratum = st.selectbox(
        "Select Evaluation Stratum:",
        available_strata,
        index=available_strata.index("India_Cross_Domain") if "India_Cross_Domain" in available_strata else 0,
        help="India_Cross_Domain exhibits 293/300 baseline T1 failures. China_In_Domain exhibits 86/480 failures."
    )

    df_sub = df_policy[df_policy["evaluation_stratum"] == selected_stratum].copy()

    # Highlight Metric Cards for Selected Stratum
    full_policy_row = df_sub[df_sub["system_architecture"] == "FULL_ROADSENTINEL_POLICY"]
    yolo_only_row = df_sub[df_sub["system_architecture"] == "YOLO_ONLY"]

    if not full_policy_row.empty and not yolo_only_row.empty:
        f_row = full_policy_row.iloc[0]
        y_row = yolo_only_row.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Automated Accept Coverage</div>
                <div class="metric-value" style="font-size: 1.5rem; color: #38bdf8;">{f_row['automated_accept_coverage_pct']:.1f}%</div>
                <div class="metric-delta neutral">{int(f_row['automated_accept_count'])} / {int(f_row['total_observations'])} observations</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Escalation / Review Rate</div>
                <div class="metric-value" style="font-size: 1.5rem; color: #facc15;">{(f_row['human_review_coverage_pct'] + f_row['domain_escalation_coverage_pct']):.1f}%</div>
                <div class="metric-delta neutral">{int(f_row['human_review_count'] + f_row['domain_escalation_count'])} diverted to review</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Unsafe Automated Accepts</div>
                <div class="metric-value" style="font-size: 1.5rem; color: {'#10b981' if f_row['unsafe_automated_accepts'] == 0 else '#ef4444'};">
                    {int(f_row['unsafe_automated_accepts'])}
                </div>
                <div class="metric-delta {'positive' if f_row['unsafe_automated_accepts'] == 0 else 'negative'}">
                    vs {int(y_row['unsafe_automated_accepts'])} in YOLO-only
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Downstream Failures Prevented</div>
                <div class="metric-value" style="font-size: 1.5rem; color: #10b981;">{f_row['failures_prevented_pct']:.1f}%</div>
                <div class="metric-delta positive">{int(f_row['failures_prevented_downstream'])} failures routed safely</div>
            </div>
            """, unsafe_allow_html=True)

    # Mandatory Phrasing & Scientific Disclaimers
    if selected_stratum == "India_Cross_Domain":
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.12); border: 2px solid #10b981; border-radius: 12px; padding: 18px 22px; margin: 16px 0;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #34d399;">
                MANDATORY RESEARCH AUDIT FINDING:
            </div>
            <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 6px; font-weight: 600;">
                “Zero unsafe automated accepts were observed on the evaluated India benchmark.”
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 8px; line-height: 1.5;">
                <strong style="color: #facc15;">Safety Routing Caution:</strong> Do not phrase this as proof that the system is universally safe. 
                Escalation is safe routing, not successful defect detection. When DINOv2 triggers a domain escalation (300/300 on India), 
                the system safely refuses to issue an automated pavement clearance because perception is uncalibrated.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="callout-box warn">
            <strong>Safety Routing Note:</strong> Escalation is safe routing, not defect detection. Automated acceptance requires both 
            in-domain distribution confirmation and high perception reliability.
        </div>
        """, unsafe_allow_html=True)

    # Destination Visualization Chart
    render_policy_destination_chart(df_sub, selected_stratum.replace("_", " "))

    # Audited Policy Table
    st.markdown("#### Audited Policy Destinations Table")
    display_policy_cols = [
        "system_architecture",
        "automated_accept_count",
        "automated_accept_coverage_pct",
        "human_review_count",
        "human_review_coverage_pct",
        "domain_escalation_count",
        "domain_escalation_coverage_pct",
        "unsafe_automated_accepts",
        "unsafe_automated_accept_rate_pct",
        "failures_prevented_downstream",
        "failures_prevented_pct",
        "successful_cases_unnecessarily_rejected",
        "measured_latency_ms",
    ]
    st.dataframe(df_sub[display_policy_cols], hide_index=True, use_container_width=True)

    st.markdown("---")

    # 2. Action Tiers Definition & Taxonomy
    st.markdown("### 2. Five-Tier Decision Taxonomy")
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

    # 3. Explainable Decision Inspector (Experiment A Captures)
    st.markdown("### 3. Explainable Decision Inspector (Experiment A Multi-Evidence Synthesis)")
    df_decisions = load_decisions_table()

    if not df_decisions.empty:
        capture_list = df_decisions["image_id"].tolist()
        selected_cap = st.selectbox(
            "Select Inspection Capture:",
            capture_list,
            index=capture_list.index("SEG_004_day_05") if "SEG_004_day_05" in capture_list else 0
        )

        row = df_decisions[df_decisions["image_id"] == selected_cap].iloc[0]

        tier_colors = {
            "PRIORITY_REVIEW": "#f97316",
            "REINSPECT": "#eab308",
            "MONITOR": "#3b82f6",
            "DOMAIN_ESCALATION": "#ef4444",
            "AUTOMATED_ACCEPT": "#10b981",
        }
        dec_color = tier_colors.get(row["decision"], "#38bdf8")

        explanation = row.get("primary_reason", row.get("explanation", "Inspection priority assigned by multi-evidence decision engine."))
        sev_val = float(row.get("current_severity", 0.0))
        sev_band = row.get("current_severity_band", row.get("severity_band", "MODERATE"))
        rel_val = float(row.get("reliability_score", 0.0))
        rel_band = row.get("reliability_band", "MODERATE")
        temp_state = row.get("temporal_state", row.get("temporal_trend", "NO_TEMPORAL_CONTEXT"))
        dom_scen = row.get("largest_forecast_scenario", row.get("dominant_scenario", "NORMAL"))
        scen_delta = float(row.get("largest_forecast_delta", row.get("forecast_delta_90d", 0.0)))
        prim_reason = row.get("primary_reason", "Perception evidence synthesis")
        sec_reason = row.get("secondary_reason", "Multi-day trajectory tracking")

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
                <em>"{explanation}"</em>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 0.85rem;">
                <div><span style="color: #94a3b8;">Current Severity:</span><br><strong style="color: #38bdf8;">{sev_val:.4f} ({sev_band})</strong></div>
                <div><span style="color: #94a3b8;">Reliability Score:</span><br><strong style="color: #a855f7;">{rel_val:.4f} ({rel_band})</strong></div>
                <div><span style="color: #94a3b8;">Temporal Evidence:</span><br><strong style="color: #34d399;">{temp_state}</strong></div>
                <div><span style="color: #94a3b8;">Peak 90d Scenario:</span><br><strong style="color: #f59e0b;">{dom_scen} ({scen_delta:+.4f})</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"**Primary Decision Driver**: {prim_reason}")
        with col_r2:
            st.markdown(f"**Secondary Contributing Factor**: {sec_reason}")

    # Full Decisions Database Table
    with st.expander("View Full Road Health Decisions Table (All 40 Captures)"):
        st.dataframe(df_decisions, use_container_width=True)
