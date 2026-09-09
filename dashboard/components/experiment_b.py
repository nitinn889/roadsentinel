"""Controlled Experiment B Specification Component for RoadSentinel Dashboard."""

from __future__ import annotations

import streamlit as st


def render_experiment_b():
    st.markdown("## Controlled Experiment B: Deterioration → Repair → Re-Deterioration")
    st.markdown(
        "Scientific specification and synthetic simulation protocol for the planned controlled pavement "
        "lifecycle experiment (**SEG_005** and **SEG_006**)."
    )

    # Mandatory Planned Status Callout
    st.markdown("""
    <div style="background: rgba(234, 179, 8, 0.12); border: 2px solid #eab308; border-radius: 14px; padding: 22px 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #facc15; font-family: 'Outfit', sans-serif;">
                CONTROLLED EXPERIMENT B — STATUS: PLANNED FOR NEXT RESEARCH REVIEW
            </div>
            <span class="status-pill pending">PLANNED / DEFERRED</span>
        </div>
        <div style="font-size: 0.95rem; color: #fef08a; margin-top: 10px; line-height: 1.5;">
            <strong>Scientific Integrity Statement:</strong> Experiment B physical image rendering is operated exclusively by the Team Lead in Unreal Engine 5.
            To preserve research validity, no fabricated or simulated placeholder images are rendered here. The complete experimental design and quantitative verification criteria are frozen below.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Experimental Design & Objectives
    st.markdown("### 1. Research Objectives & Motivation")
    st.markdown("""
    While **Experiment A** evaluated multi-state sequences with varying camera geometry, lighting, and weather shifts, 
    **Experiment B** provides a strictly controlled benchmark with:
    1. **Fixed Camera Geometry**: 100% Top-down nadir drone survey (constant 30m altitude, zero perspective skew).
    2. **Constant Environmental Lighting**: Fixed Clear Noon (70° solar elevation, zero overcast/shadow modulation).
    3. **Zero Weather Confounds**: Completely dry pavement across all 10 inspection states.
    4. **Formal Pavement Intervention**: Explicit Day 06 repair operation evaluating post-repair tracking and re-deterioration dynamics.
    """)

    # 2. Protocol Timeline Table
    st.markdown("### 2. Standardized 10-Day Lifecycle Protocol")
    
    protocol_data = [
        {"Phase": "Phase 1: Controlled Deterioration", "Days": "Day 01 – Day 05", "Pavement Evolution": "Pristine asphalt progresses steadily through micro-crack inception to severe fatigue cracking.", "Target Severity": "0.0000 → 0.7500", "Expected Perception": "Continuous defect expansion (AREA_INCREASED)"},
        {"Phase": "Phase 2: Repair Intervention", "Days": "Day 06", "Pavement Evolution": "Potholes and fissures patched with fresh asphalt overlay / sealant.", "Target Severity": "0.7500 → 0.0500", "Expected Perception": "Defect disappearance (NOT_OBSERVED) & Repair class detection"},
        {"Phase": "Phase 3: Re-Deterioration", "Days": "Day 07 – Day 10", "Pavement Evolution": "Reflective cracking propagates along repaired patch boundaries under repeated traffic stress.", "Target Severity": "0.0500 → 0.5500", "Expected Perception": "New defect tracks along patch margins (NEW_DEFECT)"},
    ]
    st.dataframe(protocol_data, hide_index=True, use_container_width=True)

    st.markdown("---")

    # 3. Segments Definition & Test Conditions
    st.markdown("### 3. Segment Definitions")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-weight: 700; color: #38bdf8; font-size: 1.1rem;">SEG_005: Flexible Bituminous Highway</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 8px;">
                - <strong>Pavement Type</strong>: Hot Mix Asphalt (HMA) Flexible Pavement.<br>
                - <strong>Initial Defect</strong>: Longitudinal centerline seam opening.<br>
                - <strong>Repair Method</strong>: Mastic crack sealing & localized bituminous patch.<br>
                - <strong>Target Evaluation</strong>: Sealant contrast false-positive resistance and seam reopening tracking.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-weight: 700; color: #a855f7; font-size: 1.1rem;">SEG_006: Composite Urban Pavement</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 8px;">
                - <strong>Pavement Type</strong>: Asphalt overlay over concrete base.<br>
                - <strong>Initial Defect</strong>: High-density alligator cracking lattice (D20) and pothole (D40).<br>
                - <strong>Repair Method</strong>: Full-depth rectangular patch restoration.<br>
                - <strong>Target Evaluation</strong>: Bipartite tracking boundary stability across high-contrast rectangular repair edges.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Success Criteria
    st.markdown("### 4. Quantitative Acceptance Criteria")
    st.markdown("""
    When captured and evaluated in the next phase, Experiment B will be verified against:
    - **Monotonic Deterioration Fidelity**: Strict positive rank correlation ($r_s \\ge 0.95$) between inspection day and measured severity across Days 01–05.
    - **Repair Recognition Rate**: Immediate detection drop ($\\Delta S \\le -0.60$) on Day 06 with 100% suppression of pre-repair defect tracks.
    - **Post-Repair Tracking Continuity**: High-precision boundary association on Days 07–10 without spurious track fragmentation.
    """)
