"""Domain Awareness & Perception Reliability Component for RoadSentinel Dashboard V2."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import (
    CHINA_RELIABILITY_PREDICTIONS_PATH,
    WORKSPACE_ROOT,
    load_canonical_metrics,
    load_canonical_phase1b_metrics,
    load_confidence_calibration_analysis,
    load_domain_gate_table,
    load_domain_threshold_sweep,
    load_headline_result_cards,
    load_per_sample_domain_distances,
    load_reliability_ablation_table,
    load_risk_coverage_table,
)


def plot_risk_coverage_curve(df_rc: pd.DataFrame) -> plt.Figure:
    """Plot dynamic Risk-Coverage selective prediction curve for Target T0 and T1."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    if not df_rc.empty:
        t1 = df_rc[df_rc["target"] == "Target_T1"].sort_values("retained_coverage_pct")
        t0 = df_rc[df_rc["target"] == "Target_T0"].sort_values("retained_coverage_pct")

        if not t1.empty:
            ax.plot(
                t1["retained_coverage_pct"],
                t1["accepted_failure_rate_pct"],
                marker="o",
                color="#38bdf8",
                linewidth=2.2,
                label="Target T1: Strict Detection Quality (F1 ≥ 0.50)",
            )
            # Highlight 80% and 50% coverage points
            pt80 = t1[t1["retained_coverage_pct"] == 80]
            if not pt80.empty:
                y80 = float(pt80.iloc[0]["accepted_failure_rate_pct"])
                ax.plot(80, y80, marker="s", markersize=8, color="#fbbf24")
                ax.annotate(
                    f"80% Cov: {y80:.2f}%\n(Audited Percentile)",
                    xy=(80, y80),
                    xytext=(80, y80 + 2.5),
                    ha="center",
                    color="#fbbf24",
                    fontweight="bold",
                    fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1),
                )

            pt50 = t1[t1["retained_coverage_pct"] == 50]
            if not pt50.empty:
                y50 = float(pt50.iloc[0]["accepted_failure_rate_pct"])
                ax.plot(50, y50, marker="s", markersize=8, color="#34d399")
                ax.annotate(
                    f"50% Cov: {y50:.2f}%\n(Audited Percentile)",
                    xy=(50, y50),
                    xytext=(50, y50 + 2.5),
                    ha="center",
                    color="#34d399",
                    fontweight="bold",
                    fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="#34d399", lw=1),
                )

        if not t0.empty:
            ax.plot(
                t0["retained_coverage_pct"],
                t0["accepted_failure_rate_pct"],
                marker="^",
                color="#a855f7",
                linewidth=1.8,
                linestyle="--",
                label="Target T0: Zero Defect Failure (F1 > 0)",
            )

    ax.set_title("Risk–Coverage Curve (Selective Prediction on In-Domain China Validation)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Retained Inspection Coverage (%)", color="#94a3b8", fontsize=9.5)
    ax.set_ylabel("Accepted Failure Rate (%)", color="#94a3b8", fontsize=9.5)
    ax.tick_params(colors="#cbd5e1", labelsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
    ax.set_xlim(5, 105)
    ax.set_ylim(0, 22)

    leg = ax.legend(loc="upper left", frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=8)
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    plt.tight_layout()
    return fig


def plot_calibration_diagram(df_cal: pd.DataFrame) -> plt.Figure:
    """Plot detection-level reliability/calibration diagram."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Bins from YOLO_CONFIDENCE_FAILURE_ANALYSIS
    bins_data = [
        {"bin": "[0.2, 0.4)", "mean_conf": 0.2989, "accuracy": 38.87, "count": 265},
        {"bin": "[0.4, 0.6)", "mean_conf": 0.4952, "accuracy": 64.06, "count": 256},
        {"bin": "[0.6, 0.8)", "mean_conf": 0.6974, "accuracy": 83.33, "count": 252},
        {"bin": "[0.8, 1.0]", "mean_conf": 0.8710, "accuracy": 96.04, "count": 101},
    ]

    confs = [d["mean_conf"] * 100.0 for d in bins_data]
    accs = [d["accuracy"] for d in bins_data]
    labels = [f"{d['bin']}\nN={d['count']}" for d in bins_data]
    x = np.arange(len(bins_data))
    width = 0.35

    ax.bar(x - width/2, confs, width, label="Mean Confidence (%)", color="#38bdf8", alpha=0.75, edgecolor="#0284c7")
    ax.bar(x + width/2, accs, width, label="Empirical Accuracy (%)", color="#10b981", alpha=0.75, edgecolor="#059669")

    # Perfect calibration line
    ax.axhline(50, color="#64748b", linestyle=":", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("Percentage (%)", color="#94a3b8", fontsize=9.5)
    ax.set_title("Detection Calibration Diagram (YOLOv8n In-Domain, N=874)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#cbd5e1", labelsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
    ax.set_ylim(0, 110)

    # Annotate summary scores
    ax.text(
        0.03, 0.92,
        "AUROC: 0.7741\nBrier Loss: 0.1921\nECE: 0.1075 (10.75%)",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#fbbf24",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#fbbf24", alpha=0.85),
    )

    leg = ax.legend(loc="lower right", frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=8)
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    plt.tight_layout()
    return fig


def plot_confidence_histogram() -> plt.Figure:
    """Plot confidence distribution across the in-domain validation set."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    conf_vals = []
    if CHINA_RELIABILITY_PREDICTIONS_PATH.exists():
        try:
            df = pd.read_csv(CHINA_RELIABILITY_PREDICTIONS_PATH)
            if "mean_confidence" in df.columns:
                conf_vals = df["mean_confidence"].dropna().tolist()
        except Exception:
            pass

    if conf_vals:
        ax.hist(conf_vals, bins=25, range=(0.0, 1.0), color="#818cf8", alpha=0.7, edgecolor="#4f46e5")
        ax.axvline(np.mean(conf_vals), color="#fbbf24", linestyle="--", linewidth=1.8, label=f"Mean Image Conf ({np.mean(conf_vals):.3f})")
    else:
        # Fallback dummy
        ax.hist([0.5], bins=10, color="#818cf8")

    ax.set_title("Per-Image Confidence Distribution (China Drone, N=480)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Detection Confidence per Image", color="#94a3b8", fontsize=9.5)
    ax.set_ylabel("Image Count", color="#94a3b8", fontsize=9.5)
    ax.tick_params(colors="#cbd5e1", labelsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")

    leg = ax.legend(loc="upper left", frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=8)
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    plt.tight_layout()
    return fig


def render_reliability_analysis():
    st.markdown("## Domain Awareness & Perception Reliability")
    st.markdown(
        "Evaluate the two-tier perception safety framework: **DINOv2 Macro Domain Gating** at the input boundary "
        "and **YOLO Confidence-Based Reliability Estimation** at the sample prediction boundary."
    )

    canon = load_canonical_metrics()
    df_gate = load_domain_gate_table()
    df_ablation = load_reliability_ablation_table()
    df_rc = load_risk_coverage_table()
    df_cal = load_confidence_calibration_analysis()
    headline = load_headline_result_cards()

    # Headline Summary Row
    h1, h2, h3, h4, h5 = st.columns(5)
    with h1:
        st.metric("Detection AUROC", "0.7741", help="AUROC for discriminating TP from FP based on detector confidence alone")
    with h2:
        st.metric("Brier Loss Score", "0.1921", help="Mean squared error between confidence and binary correctness label")
    with h3:
        st.metric("Calibration ECE", "0.1075", delta="10.75% error", help="Expected Calibration Error across 10 uniform bins")
    with h4:
        st.metric("80% Cov Accepted Error", "9.38%", delta="Target T1 (36/384)", help="Pure percentile rejection on Target T1 (Strict F1 >= 0.50)")
    with h5:
        st.metric("50% Cov Accepted Error", "6.25%", delta="Target T1 (15/240)", help="Pure percentile rejection on Target T1 (Strict F1 >= 0.50)")

    st.markdown("""
    <div class="callout-box warn">
        <strong>AUDITED SELECTIVE REJECTION RECONCILIATION:</strong><br>
        The historical figures of <strong>8.85%</strong> (at 80% coverage) and <strong>2.08%</strong> (at 50% coverage) 
        originated in Phase 12 from earlier fixed probability threshold filters (p ≥ 0.7629 and p ≥ 0.9850). 
        Under audited <strong>pure rank-ordered percentile selective prediction</strong>, the accepted failure rates 
        under Target T1 are strictly <strong>9.38% at 80% coverage</strong> (36 failures / 384 accepted images) 
        and <strong>6.25% at 50% coverage</strong> (15 failures / 240 accepted images).
        The older 2.08% value is officially deprecated as a pure selective prediction metric.
    </div>
    """, unsafe_allow_html=True)

    # Two-Tier Concept Tabs
    tab_overview, tab_domain_gate, tab_sample_rel, tab_risk_cov = st.tabs([
        "Two-Tier Framework",
        "1. DINOv2 Domain Gating",
        "2. Sample Reliability Ablation",
        "3. Risk-Coverage & Calibration",
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
                    • <b>Domain AUROC:</b> <b>1.0000</b> (Zero distribution overlap on evaluated benchmark).<br>
                    • <b>Separation Margin:</b> <b>+0.1158</b> cosine distance (min India 0.6405 - max China 0.5247).<br>
                    • <b>India Detection Rate:</b> <b>100.0%</b> (300/300 escalated at p99=0.4491).<br>
                    • <b>China False Warning Rate:</b> <b>1.46%</b> (7/480 images).
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div style="background: #1e293b; padding: 18px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <h4 style="color: #60a5fa; margin-top: 0;">Tier 2: Sample-Level Reliability</h4>
                <p style="color: #cbd5e1; font-size: 13px;">
                    • <b>Mechanism:</b> Model B Calibrated YOLO Confidence Features.<br>
                    • <b>Detection AUROC:</b> <b>0.7741</b> (Brier loss 0.1921, ECE 0.1075).<br>
                    • <b>Target T1 Failure AUROC:</b> <b>0.8166</b> across 5-fold cross-validation.<br>
                    • <b>80% Coverage Failure Rate:</b> <b>9.38%</b> (down from 17.92% baseline).<br>
                    • <b>50% Coverage Failure Rate:</b> <b>6.25%</b> (isolating 82.56% of baseline errors).
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
            display_cols = [c for c in ["model", "model_name", "target", "dataset", "AUROC", "AUROC_95CI_low", "AUROC_95CI_high", "AUPRC", "failure_F1"] if c in df_ablation.columns]
            st.dataframe(df_ablation[display_cols], use_container_width=True)

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
        st.markdown("### Risk-Coverage & Calibration Analysis")
        st.markdown(
            "By rejecting the lowest-reliability observations using pure percentile sorting, the accepted failure rate "
            "is systematically suppressed below the baseline rate."
        )

        # Dynamic Risk-Coverage Plot
        fig_rc = plot_risk_coverage_curve(df_rc)
        st.pyplot(fig_rc)
        plt.close(fig_rc)

        # Calibration Diagram and Confidence Histogram
        c_diag, c_hist = st.columns(2)
        with c_diag:
            fig_cal = plot_calibration_diagram(df_cal)
            st.pyplot(fig_cal)
            plt.close(fig_cal)
        with c_hist:
            fig_hist = plot_confidence_histogram()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        # Full Risk-Coverage Table
        if not df_rc.empty:
            st.markdown("##### Audited Selective Prediction Risk-Coverage Table (Target T0 & T1)")
            display_rc_cols = [
                "target", "rejection_pct", "retained_coverage_pct", "accepted_samples", "accepted_failures",
                "accepted_failure_rate_pct", "error_reduction_pct", "min_reliability_threshold", "target_description"
            ]
            avail_rc = [c for c in display_rc_cols if c in df_rc.columns]
            st.dataframe(df_rc[avail_rc], hide_index=True, use_container_width=True)

