"""System Overview Component for RoadSentinel Dashboard V2."""

from __future__ import annotations

import streamlit as st
from data_loader import (
    WORKSPACE_ROOT,
    load_canonical_metrics,
    load_headline_result_cards,
)


def render_overview():
    canon = load_canonical_metrics()
    headline_cards = load_headline_result_cards()

    st.markdown("""
    <div class="main-header" style="background: linear-gradient(135deg, #0f172a, #1e293b); padding: 24px; border-radius: 12px; border-left: 6px solid #2563eb; margin-bottom: 24px;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 28px;">RoadSentinel: Perception, Reliability & Deterioration Intelligence</h1>
        <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 15px;">
            Examiner-Facing Interactive Platform integrating Supervised Damage Detection (YOLOv8n),
            Foundation Macro Domain Gating (DINOv2), Sample-Level Reliability Estimation, Multi-Day Synthetic Temporal Tracking,
            Phase 1C Longitudinal Deterioration Forecasting (XGBoost vs Persistence), and the Decoupled Road-Health Decision Engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Status Badges
    st.markdown("### System Pipeline Status")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div style="background: #1e293b; border: 1px solid #3b82f6; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #60a5fa;">PERCEPTION</b><br><span style="color: #94a3b8; font-size: 12px;">YOLOv8n Frozen (F1 0.7104)</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="background: #1e293b; border: 1px solid #8b5cf6; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #a78bfa;">DOMAIN GATE</b><br><span style="color: #94a3b8; font-size: 12px;">DINOv2 AUROC 1.0000</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="background: #1e293b; border: 1px solid #10b981; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #34d399;">TEMPORAL & FC</b><br><span style="color: #94a3b8; font-size: 12px;">Phase 1C Audited (113 Pairs)</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div style="background: #1e293b; border: 1px solid #ef4444; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #f87171;">DECISION</b><br><span style="color: #94a3b8; font-size: 12px;">5-Tier Routing (0 Unsafe)</span></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div style="background: #1e293b; border: 1px solid #64748b; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #94a3b8;">EDGE PI 5</b><br><span style="color: #fbbf24; font-size: 12px;">Pending Physical Profiling</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================================
    # Headline Audited Result Cards
    # =========================================================================
    st.markdown("### Audited Headline Results")
    st.caption("Values below are dynamically ingested from audited machine-readable JSON/CSV artifacts. Hover over cards for source artifact paths and sampling strata.")

    # Row 1: 4 cards
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    with r1_c1:
        c = headline_cards["yolo_china_f1"]
        st.markdown(f"""
        <div class="metric-card" title="Source: {c['source']} | {c['sample_stratum']}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{c['title']}</span>
                <span class="status-pill {c['status'].lower()}">{c['status']}</span>
            </div>
            <div class="metric-value" style="color: #60a5fa;">{c['value']}</div>
            <div class="metric-delta neutral">{c['delta']}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 6px;">Source: <code>{c['source']}</code></div>
        </div>
        """, unsafe_allow_html=True)

    with r1_c2:
        c = headline_cards["yolo_india_f1"]
        st.markdown(f"""
        <div class="metric-card" title="Source: {c['source']} | {c['sample_stratum']}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{c['title']}</span>
                <span class="status-pill {c['status'].lower()}">{c['status']}</span>
            </div>
            <div class="metric-value" style="color: #f87171;">{c['value']}</div>
            <div class="metric-delta negative">{c['delta']}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 6px;">Source: <code>{c['source']}</code></div>
        </div>
        """, unsafe_allow_html=True)

    with r1_c3:
        c = headline_cards["india_domain_escalation"]
        st.markdown(f"""
        <div class="metric-card" title="Source: {c['source']} | {c['sample_stratum']}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{c['title']}</span>
                <span class="status-pill {c['status'].lower()}">{c['status']}</span>
            </div>
            <div class="metric-value" style="color: #a78bfa;">{c['value']}</div>
            <div class="metric-delta positive">{c['delta']}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 6px;">Source: <code>{c['source']}</code></div>
        </div>
        """, unsafe_allow_html=True)

    with r1_c4:
        c = headline_cards["china_warning_rate"]
        st.markdown(f"""
        <div class="metric-card" title="Source: {c['source']} | {c['sample_stratum']}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{c['title']}</span>
                <span class="status-pill {c['status'].lower()}">{c['status']}</span>
            </div>
            <div class="metric-value" style="color: #34d399;">{c['value']}</div>
            <div class="metric-delta positive">{c['delta']}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 6px;">Source: <code>{c['source']}</code></div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: 3 cards
    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        c = headline_cards["reliability_accepted_failure_rate"]
        st.markdown(f"""
        <div class="metric-card" title="Source: {c['source']} | {c['sample_stratum']}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{c['title']}</span>
                <span class="status-pill {c['status'].lower()}">{c['status']}</span>
            </div>
            <div class="metric-value" style="font-size: 1.45rem; color: #38bdf8;">{c['value']}</div>
            <div class="metric-delta neutral">{c['delta']}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 6px;">Source: <code>{c['source']}</code></div>
        </div>
        """, unsafe_allow_html=True)

    with r2_c2:
        c = headline_cards["m4_forecast_skill"]
        st.markdown(f"""
        <div class="metric-card" title="Source: {c['source']} | {c['sample_stratum']}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{c['title']}</span>
                <span class="status-pill {c['status'].lower()}">{c['status']}</span>
            </div>
            <div class="metric-value" style="color: #c084fc;">{c['value']}</div>
            <div class="metric-delta neutral">{c['delta']}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 6px;">Source: <code>{c['source']}</code></div>
        </div>
        """, unsafe_allow_html=True)

    with r2_c3:
        c = headline_cards["raspberry_pi_profiling"]
        st.markdown(f"""
        <div class="metric-card" title="Source: {c['source']} | {c['sample_stratum']}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{c['title']}</span>
                <span class="status-pill {c['status'].lower()}">{c['status']}</span>
            </div>
            <div class="metric-value" style="font-size: 1.45rem; color: #fbbf24;">{c['value']}</div>
            <div class="metric-delta neutral">{c['delta']}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 6px;">Source: <code>{c['source']}</code></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Canonical Architecture Flow
    st.markdown("### Canonical Multi-Stream Technical Architecture")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  INCOMING ROAD CAPTURE + METADATA                                │
    └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             │                                               │
                             ▼                                               ▼
             ┌───────────────────────────────┐               ┌───────────────────────────────┐
             │      TIER 1: DOMAIN GATE      │               │     TIER 2: PERCEPTION CORE   │
             │   DINOv2 ViT-S/14 Embedding   │               │   • YOLOv8n Damage Detector   │
             │   (Threshold p99 = 0.4491)    │               │   • Matched IoU = 0.8007      │
             └───────────────┬───────────────┘               └───────────────┬───────────────┘
                             │                                               │
               [ d_kNN > 0.4491: SHIFT ]                     [ Current Model Severity: 0-1 ]
                             │                                               │
                             ▼                                               ▼
             ┌───────────────────────────────┐               ┌───────────────────────────────┐
             │       DOMAIN_ESCALATION       │               │   TIER 3: RELIABILITY FILTER  │
             │   Quarantine Visual Shift     │               │   Model B Calibrated Conf     │
             │   300/300 India Flagged       │               │   Risk-Coverage Selective Rej │
             └───────────────────────────────┘               └───────────────┬───────────────┘
                                                                             │
                                                     ┌───────────────────────┴───────────────────────┐
                                                     │                                               │
                                                     ▼                                               ▼
                                     ┌───────────────────────────────┐               ┌───────────────────────────────┐
                                     │   TIER 4: SYNTHETIC TEMPORAL  │               │   TIER 5: RESIDUAL FORECAST   │
                                     │   • CARLA Simulator Captures  │               │   • M0 Persistence Default    │
                                     │   • 48 Tracks (45.83% Persist)│               │   • M4 Residual XGBoost       │
                                     │   • NOT_OBSERVED != REPAIRED  │               │   • +5.27% Point Skill (95% CI) │
                                     └───────────────┬───────────────┘               └───────────────┬───────────────┘
                                                     │                                               │
                                                     └───────────────────────┬───────────────────────┘
                                                                             │
                                                                             ▼
                                                             ┌───────────────────────────────┐
                                                             │   TIER 6: DECISION ENGINE     │
                                                             │   • PRIORITY_REVIEW           │
                                                             │   • MONITOR                   │
                                                             │   • AUTOMATED_ACCEPT          │
                                                             │   • REINSPECT                 │
                                                             │   0 Unsafe Automated Accepts  │
                                                             └───────────────────────────────┘
    ```
    """)

    st.markdown("---")

    # Reconciled Research Findings
    st.markdown("### Examiner Research Findings Panel (Audited)")
    findings = [
        ("1. In-Domain Detector Dominance", "YOLOv8n achieves high in-domain distress localization accuracy on China UAV surveys (F1 = 0.7104, Mean Matched IoU = 0.8007) compared to zero-shot DINOv2+SAM2 (F1 = 0.0267). Perceptual near-duplicate filtering yields F1 = 0.7022, confirming robust generalization."),
        ("2. Cross-Domain Stress Test & Transfer Collapse", "Under an independent cross-domain stress test on Indian vehicle dashcams, YOLO F1 collapses by -96.9% (0.7104 to 0.0218, with 293/300 T1 failures). This catastrophic degradation is confounded by UAV top-down vs. forward-facing dashcam viewpoint, optics, and environmental differences."),
        ("3. Foundation Model as Macro Domain Gate", "DINOv2 patch embeddings perfectly separated the evaluated China-Drone and India-dashcam benchmark domains (AUROC 1.0000, separation margin +0.1158, 300/300 escalations at p99, 7/480 familiar China warnings). This demonstrates separation on this evaluated benchmark, not universal OOD detection."),
        ("4. Reliability & Selective Prediction", "Model B confidence estimation achieves detection-correctness AUROC 0.7741 (Brier loss 0.1921, ECE 0.1075). Pure percentile selective rejection achieves an accepted failure rate of 9.38% at 80% coverage and 6.25% at 50% coverage under Target T1. (Historical figures 8.85% and 2.08% reflect older fixed-threshold filters)."),
        ("5. CARLA/Unreal Synthetic Temporal Evaluation", "Evaluated on 40 captures across 8 sequences in CARLA/Unreal simulation. Tracking tracks 48 unique distresses, with 22 persistent tracks (45.83% persistence rate) and 33 adjacent transitions (14 area increases, 19 area decreases). Strictly enforces semantic rule NOT_OBSERVED != REPAIRED."),
        ("6. Phase 1C Longitudinal Deterioration Forecasting", "On 113 FHWA LTPP transition pairs across 23 highway sections, M4 Combined XGBoost achieves outer MAE 0.0551 (+5.27% point-estimate skill over M0 Persistence MAE 0.0582). The paired site-clustered 95% CI [-0.0106, +0.0043] crosses zero; universal statistical superiority over persistence is not established."),
        ("7. Evidence-Based Decision Governance", "Zero unsafe automated accepts were observed on the evaluated India benchmark under the full policy. Safe escalation routes uncalibrated frames to human experts, but escalation represents safe operational routing, not successful defect detection."),
    ]

    for title, desc in findings:
        st.markdown(f"**{title}**: {desc}")

    st.markdown("---")

    # =========================================================================
    # Master System Limitations Panel (Mandatory Audited Disclaimers)
    # =========================================================================
    st.markdown("### Master System Limitations & Scientific Scope")
    st.markdown("""
    <div class="callout-box warn">
        <h4 style="color: #fbbf24; margin-top: 0; margin-bottom: 8px;">Mandatory Methodological & Scientific Limitations</h4>
        <ul style="color: #e2e8f0; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 18px;">
            <li><strong>Cross-Domain Confounding:</strong> The China UAV to India dashcam evaluation is confounded by UAV-versus-dashcam viewpoint, mounting height, camera optics, and sensor differences.</li>
            <li><strong>Domain Gate Scope:</strong> DINOv2 gate separation (AUROC 1.0000, margin +0.1158) is benchmark-specific to China UAV vs India Dashcam and does not represent universal out-of-distribution detection across all unseen road environments.</li>
            <li><strong>Foundation Anomaly Segmentation:</strong> Direct DINOv2 + SAM2 defect-accuracy superiority has not been established over task-specific supervised detectors on organic road distress.</li>
            <li><strong>Forecast Scenario Variables:</strong> Future rainfall, temperature, traffic, and wet exposure must be declared scenarios or externally supplied forecasts; they are not dynamically predicted by the computer vision model.</li>
            <li><strong>Non-Causal Projections:</strong> Scenario deterioration projections represent modelled statistical associations from historical survey records, not causal civil-engineering or counterfactual guarantees.</li>
            <li><strong>Forecasting Sample Scope:</strong> The longitudinal dataset contains 23 highway sections and 113 transition pairs; rapid 30–90-day deterioration forecasting is not validated.</li>
            <li><strong>Rehabilitation Auditing:</strong> Pavement transitions crossing major structural rehabilitation or reconstruction events still require explicit maintenance work order auditing.</li>
            <li><strong>Routing Threshold Validation:</strong> The 365-day forecast routing boundary requires predefined or independent validation before deployment.</li>
            <li><strong>Temporal Simulation Data:</strong> All 40 multi-day temporal captures are CARLA / Unreal Engine synthetic simulator captures with injected distresses and configured weather; they demonstrate tracking functionality, not real-world pavement deterioration.</li>
            <li><strong>Edge Hardware Profiling:</strong> Raspberry Pi 5 performance, latency, and power consumption remain pending physical profiling on target edge hardware.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

