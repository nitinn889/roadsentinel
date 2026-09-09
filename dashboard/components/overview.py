"""System Overview Component for RoadSentinel Dashboard V2."""

from __future__ import annotations

import streamlit as st
from data_loader import load_canonical_metrics, WORKSPACE_ROOT


def render_overview():
    canon = load_canonical_metrics()

    st.markdown("""
    <div class="main-header" style="background: linear-gradient(135deg, #0f172a, #1e293b); padding: 24px; border-radius: 12px; border-left: 6px solid #2563eb; margin-bottom: 24px;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 28px;">RoadSentinel: Perception, Reliability & Deterioration Intelligence</h1>
        <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 15px;">
            Examiner-Facing Interactive Demonstration integrating Supervised Damage Detection (YOLOv8n),
            Foundation Macro Domain Gating (DINOv2), Sample-Level Reliability Estimation, Multi-Day Temporal Tracking,
            Scenario-Conditioned Forecasting (XGBoost), and the Decoupled Road-Health Decision Engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Status Badges
    st.markdown("### System Pipeline Status")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div style="background: #1e293b; border: 1px solid #3b82f6; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #60a5fa;">PERCEPTION</b><br><span style="color: #94a3b8; font-size: 12px;">YOLOv8n Frozen</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="background: #1e293b; border: 1px solid #8b5cf6; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #a78bfa;">DOMAIN GATE</b><br><span style="color: #94a3b8; font-size: 12px;">DINOv2 AUROC 1.0</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="background: #1e293b; border: 1px solid #10b981; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #34d399;">TEMPORAL & FC</b><br><span style="color: #94a3b8; font-size: 12px;">Goal 1 & 2 Complete</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div style="background: #1e293b; border: 1px solid #ef4444; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #f87171;">DECISION</b><br><span style="color: #94a3b8; font-size: 12px;">Phase 13 Complete</span></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div style="background: #1e293b; border: 1px solid #64748b; border-radius: 6px; padding: 10px; text-align: center;"><b style="color: #94a3b8;">EDGE PI 5</b><br><span style="color: #64748b; font-size: 12px;">Future Phase Pending</span></div>', unsafe_allow_html=True)

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
             │   (Threshold p99 = 0.4491)    │               │   • DINO/SAM2 Surface Feature │
             └───────────────┬───────────────┘               └───────────────┬───────────────┘
                             │                                               │
               [ d_kNN > 0.4491: SHIFT ]                     [ Current Model Severity: 0-1 ]
                             │                                               │
                             ▼                                               ▼
             ┌───────────────────────────────┐               ┌───────────────────────────────┐
             │       DOMAIN_ESCALATION       │               │   TIER 3: RELIABILITY FILTER  │
             │   Quarantine Visual Shift     │               │   Model B Calibrated Conf     │
             └───────────────────────────────┘               └───────────────┬───────────────┘
                                                                             │
                                                     ┌───────────────────────┴───────────────────────┐
                                                     │                                               │
                                                     ▼                                               ▼
                                     ┌───────────────────────────────┐               ┌───────────────────────────────┐
                                     │   TIER 4: TEMPORAL TRACKING   │               │   TIER 5: SCENARIO FORECAST   │
                                     │   • Greedy Track Matching     │               │   • XGBoost Model V2 (LTPP)   │
                                     │   • Area Delta Quantification │               │   • 90-Day Climate Scenarios  │
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
                                                             └───────────────────────────────┘
    ```
    """)

    # Headline Research Metrics
    st.markdown("### Core Empirical Anchors at a Glance")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        st.markdown("""
        <div style="background: #1e293b; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: bold;">IN-DOMAIN BENCHMARK F1</div>
            <div style="color: #60a5fa; font-size: 26px; font-weight: bold; margin: 4px 0;">0.7104</div>
            <div style="color: #10b981; font-size: 12px;">YOLOv8n (vs DINO/SAM 0.0267)</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        st.markdown("""
        <div style="background: #1e293b; padding: 16px; border-radius: 8px; border-left: 4px solid #8b5cf6;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: bold;">DOMAIN GATE ACCURACY</div>
            <div style="color: #a78bfa; font-size: 26px; font-weight: bold; margin: 4px 0;">1.0000</div>
            <div style="color: #10b981; font-size: 12px;">AUROC (100% India Shift Detection)</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol3:
        st.markdown("""
        <div style="background: #1e293b; padding: 16px; border-radius: 8px; border-left: 4px solid #10b981;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: bold;">SELECTIVE ERROR REDUCTION</div>
            <div style="color: #34d399; font-size: 26px; font-weight: bold; margin: 4px 0;">88.4%</div>
            <div style="color: #10b981; font-size: 12px;">In-Domain Risk-Coverage Control</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol4:
        st.markdown("""
        <div style="background: #1e293b; padding: 16px; border-radius: 8px; border-left: 4px solid #ef4444;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: bold;">CROSS-DOMAIN QUARANTINE</div>
            <div style="color: #f87171; font-size: 26px; font-weight: bold; margin: 4px 0;">100.0%</div>
            <div style="color: #10b981; font-size: 12px;">0/293 Silent Failures Accepted</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Research Findings Summary Panel
    st.markdown("### Examiner Research Findings Panel")
    findings = [
        ("1. In-Domain Detector Dominance", "YOLOv8n dominates direct road damage detection in-domain (F1 = 0.7104 vs 0.0267) with 72.7x lower inference latency (3.62 ms vs 263.15 ms)."),
        ("2. Cross-Domain Transfer Collapse", "Both supervised and foundation models degrade drastically under zero-shot domain transfer (YOLO F1 drops -96.8% from 0.7104 to 0.0226 on Indian dashcams)."),
        ("3. Foundation Model as Macro Domain Gate", "DINOv2 foundation embeddings provide flawless macro domain shift detection (AUROC 1.0000, separating China from India with a +0.1158 cosine distance margin)."),
        ("4. Sample-Level Reliability Selective Prediction", "YOLO confidence calibration enables selective prediction, dropping accepted inspection error from 17.9% down to 2.1% (88.4% error reduction)."),
        ("5. Model-Observed Temporal Analytics", "Greedy hierarchical matching successfully tracks persistent defect regions and quantifies area increase/decrease across multi-day surveillance."),
        ("6. Scenario-Conditioned Forecasting", "XGBoost Model V2 projects future pavement severity under 5 climate and traffic scenarios, identifying Wet Exposure as the primary degradation catalyst."),
        ("7. Evidence-Based Decision Governance", "The Decision Engine combines all 4 decoupled evidence streams into actionable review tiers without uncalibrated score scaling."),
    ]

    for title, desc in findings:
        st.markdown(f"**{title}**: {desc}")
