"""Temporal Road Monitoring & Model-Observed Change Component for RoadSentinel Dashboard."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
from data_loader import WORKSPACE_ROOT, load_canonical_metrics, load_temporal_sequences_table

PLOTS_DIR = WORKSPACE_ROOT / "integration/experiment_a/temporal/plots"


def render_temporal_monitoring():
    st.markdown("## Temporal Road Monitoring & Model-Observed Change")
    st.markdown(
        "Explore multi-day defect tracking across repeated camera inspections in Experiment A. "
        "Evaluates temporal correspondence, track persistence, and distinguishes systematic "
        "sensor responses from environmental confounds."
    )

    metrics = load_canonical_metrics()
    seq_table = load_temporal_sequences_table()

    # 1. Scientific Disclaimer & Tracking Algorithm
    st.markdown("""
    <div class="callout-box warn">
        <strong>CRITICAL SCIENTIFIC WORDING & TRACKING SPECIFICATION:</strong><br>
        • <strong>Strict Phrasing</strong>: Multi-day metrics represent <strong>MODEL-OBSERVED TEMPORAL CHANGE</strong> across simulated road states and environmental conditions, NOT certified ground-truth physical pavement deterioration.<br>
        • <strong>Hierarchical Tracking Engine</strong>: Greedy one-to-one bipartite matching executed in order:
          <code>Mask IoU ≥ 0.50</code> → <code>BBox IoU ≥ 0.30</code> → <code>Centroid Distance ≤ 75 px + Area Ratio ≤ 3.0×</code> (No Hungarian assignment).
    </div>
    """, unsafe_allow_html=True)

    # 2. Canonical Event Taxonomy Counts
    st.markdown("### 1. Canonical Temporal Event Totals (All 8 Subsequences)")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.metric("NEW_DEFECT", "48", help="A defect observed on Day T with no spatial correspondence on Day T-1.")
    with e2:
        st.metric("TOTAL_MATCHED", "33", help="Defect track maintained across consecutive observations (14 increased, 19 decreased).")
    with e3:
        st.metric("AREA_INCREASED", "14", help="Matched region exhibiting >15% observed bounding area growth (canonical count: 14).")
    with e4:
        st.metric("AREA_DECREASED", "19", help="Matched region exhibiting >15% observed area contraction (canonical count: 19).")
    with e5:
        st.metric("NOT_OBSERVED", "34", help="Region tracked previously but unobserved on Day T. NEVER marked as REPAIRED.")

    st.markdown("---")

    # 3. Validated Sequences Selector (All 8 Subsequences)
    st.markdown("### 2. Evaluated Same-Camera Sequences (8 Sequences, 33 Transitions)")
    
    sequences_dict = {
        "SEG_004_D01_D05": {
            "title": "SEG_004 D01–D05 — Progression Sequence (Top-Down Drone, Δ: +0.7302)",
            "renderer": render_seg004_progression_view,
        },
        "SEG_003_D01_D07": {
            "title": "SEG_003 D01–D07 — Stability Control Test (Fixed Overlook, CV: 0.67%)",
            "renderer": render_seg003_control_view,
        },
        "SEG_001_D03_D10": {
            "title": "SEG_001 D03–D10 — Environmental Modulation (Overcast / Rain / Sunset Glare)",
            "renderer": render_seg001_confound_view,
        },
        "SEG_004_D06_D10": {
            "title": "SEG_004 D06–D10 — Extreme Weather & Sunset Flare Confound",
            "renderer": render_seg004_weather_confound_view,
        },
        "SEG_002_D01_D02": {
            "title": "SEG_002 D01–D02 — Highway Overlook Baseline Pair",
            "renderer": lambda: render_generic_sequence_view("SEG_002_D01_D02", "Highway Overlook", 2, "0.2444", "0.2444", "+0.0000", "Stable shoulder tracking across initial clear days."),
        },
        "SEG_002_D04_D05": {
            "title": "SEG_002 D04–D05 — Moderate Overcast Shift Pair",
            "renderer": lambda: render_generic_sequence_view("SEG_002_D04_D05", "Highway Overlook", 2, "0.2444", "0.2444", "+0.0000", "Stable tracking through moderate diffuse cloud cover."),
        },
        "SEG_002_D06_D07": {
            "title": "SEG_002 D06–D07 — Late Afternoon Low-Angle Pair",
            "renderer": lambda: render_generic_sequence_view("SEG_002_D06_D07", "Highway Overlook", 2, "0.2444", "0.2444", "+0.0000", "Shoulder aggregate track retained under 35° sun elevation."),
        },
        "SEG_003_D08_D09": {
            "title": "SEG_003 D08–D09 — Overcast Diffuse Lighting Pair",
            "renderer": lambda: render_generic_sequence_view("SEG_003_D08_D09", "Overlook Vantage", 2, "0.2444", "0.2444", "+0.0000", "Invariant control verification following initial 7-day run."),
        },
    }

    selected_key = st.selectbox(
        "Select Evaluated Temporal Sequence:",
        list(sequences_dict.keys()),
        format_func=lambda k: sequences_dict[k]["title"],
        index=0,
    )

    st.markdown("---")

    # Render selected sequence view
    sequences_dict[selected_key]["renderer"]()

    st.markdown("---")

    # 4. Temporal Ineligible Captures Table (Section 15)
    st.markdown("### 3. Temporal Tracking Ineligibility Accounting")
    st.markdown(
        "To preserve geometric tracking validity, camera transitions with changing viewpoint geometries or isolated non-contiguous "
        "states are excluded from bipartite matching. All 40 physical captures remain 100% accessible in single-image view."
    )

    ineligible_data = [
        {"Capture": "SEG_001 Day 01", "Condition": "Nadir Overhead Drone", "Ineligibility Reason": "VIEWPOINT_CHANGE", "Detailed Note": "Camera switch between Day 02 and Day 03 (switched from overhead nadir to oblique roadside view)."},
        {"Capture": "SEG_001 Day 02", "Condition": "Nadir Overhead Drone", "Ineligibility Reason": "VIEWPOINT_CHANGE", "Detailed Note": "Camera perspective changed before Day 03 sequence inception."},
        {"Capture": "SEG_002 Day 03", "Condition": "Overlook Variant", "Ineligibility Reason": "VIEWPOINT_CHANGE", "Detailed Note": "Intermediate pan/tilt geometry offset between Day 02 and Day 04 pairs."},
        {"Capture": "SEG_002 Days 08, 09, 10", "Condition": "Overlook Low Sun", "Ineligibility Reason": "INSUFFICIENT_CONTIGUOUS_STATES", "Detailed Note": "Non-consecutive single-frame captures with varying illumination angles."},
        {"Capture": "SEG_003 Day 10", "Condition": "Overcast Heavy", "Ineligibility Reason": "INSUFFICIENT_CONTIGUOUS_STATES", "Detailed Note": "Isolated terminal state following the D08-D09 pair."},
    ]
    st.dataframe(pd.DataFrame(ineligible_data), hide_index=True, use_container_width=True)

    st.markdown("---")

    # 5. Publication Figures & Sequences Summary
    st.markdown("### 4. Temporal Analytics Research Figures")
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        p_event = PLOTS_DIR / "fig6_temporal_event_distribution.png"
        if p_event.exists():
            st.image(str(p_event), caption="Figure 6: Temporal Event Transition Distribution (Phase 4)", use_container_width=True)
    with col_fig2:
        p_track = PLOTS_DIR / "fig5_track_length_distribution.png"
        if p_track.exists():
            st.image(str(p_track), caption="Figure 5: Track Lifespan Retention Distribution", use_container_width=True)

    if not seq_table.empty:
        with st.expander("View Full Sequences Metadata Table (All 8 Subsequences)"):
            st.dataframe(seq_table, use_container_width=True)


def render_seg004_progression_view():
    st.markdown("### SEG_004 D01–D05: Model-Observed Progression Sequence")

    col_info, col_plot = st.columns([1, 1.2])
    with col_info:
        st.markdown("""
        **Experimental Condition**:
        - **Pavement Evolution**: Model-observed severity progression across simulated road states under a fixed camera viewpoint.
        - **Camera Geometry**: Fixed Overhead Drone Survey (Top-Down Nadir, 30m)
        - **Inspection Horizon**: 5 Consecutive States (Days 01–05)
        """)

        st.markdown("#### Measured Severity Progression")
        prog_df = pd.DataFrame({
            "Day": ["Day 01", "Day 02", "Day 03", "Day 04", "Day 05"],
            "Measured Severity": [0.0000, 0.0000, 0.2564, 0.6542, 0.7302],
            "Observed State": ["State D01 (0 defects)", "State D02 (0 defects)", "State D03 (1 defect)", "State D04 (2 defects)", "State D05 (2 defects)"]
        })
        st.dataframe(prog_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="callout-box">
            <strong>Scientific Nuance:</strong><br>
            The sequence displays a <strong>model-observed severity progression across simulated road states under a fixed camera viewpoint</strong> (<strong>Δ = +0.7302</strong>).<br>
            <em>Do not describe this as an unconfounded physical deterioration measurement:</em> while the drone camera angle is fixed top-down, subtle ambient illumination variations exist between states.
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        plot_p = PLOTS_DIR / "fig2_seg004_drone_progression.png"
        if plot_p.exists():
            st.image(str(plot_p), caption="Figure 2: SEG_004 Drone Nadir Severity & Defect Progression", use_container_width=True)


def render_seg003_control_view():
    st.markdown("### SEG_003 D01–D07: Model Response Stability Control")
    
    col_info, col_plot = st.columns([1, 1.2])
    with col_info:
        st.markdown("""
        **Experimental Condition**:
        - **Pavement State**: Invariant Pristine (Grade A)
        - **Camera Geometry**: Fixed Highway Curve Vantage Overlook
        - **Lighting & Weather**: Constant Clear Noon (70° Sun Elevation)
        - **Inspection Horizon**: 7 Consecutive Days (Days 01–07)
        """)
        
        st.markdown("#### Quantitative Invariance Metrics")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Mean Severity", "0.2444", delta="Range: 0.2417 – 0.2472")
            st.metric("Defect Count", "1.00", delta="Exactly 1 region every day")
        with m2:
            st.metric("Coefficient of Var. (CV)", "0.67%", delta="Near-Zero Variance")
            st.metric("Longest Track", "7 States", delta="100% Track Retention")

        st.markdown("""
        <div class="callout-box success">
            <strong>Critical Research Interpretation:</strong><br>
            The near-zero coefficient of variation (<strong>CV = 0.67%</strong>) demonstrates that when physical pavement and lighting remain constant, the DINOv2+SAM2 perception pipeline exhibits exceptional repeatability.<br><br>
            <strong>Examiner Note:</strong> The persistent detected region is a <em>systematic false-positive candidate</em> (road shoulder aggregate contrast under grazing perspective), not confirmed physical damage.
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        plot_p = PLOTS_DIR / "fig1_seg003_stability_control.png"
        if plot_p.exists():
            st.image(str(plot_p), caption="Figure 1: SEG_003 Multi-State Severity & Defect Count Invariance", use_container_width=True)


def render_seg001_confound_view():
    st.markdown("### SEG_001 D03–D10: Environmental Modulation Sequence")

    st.markdown("""
    <div class="callout-box warn">
        <strong>EXPLICIT ENVIRONMENTAL CONFOUND WARNING:</strong><br>
        <em>"Observed severity changes in this sequence reflect both underlying road-state variation and perception sensitivity to illumination and moisture conditions."</em>
    </div>
    """, unsafe_allow_html=True)

    col_info, col_plot = st.columns([1, 1.2])
    with col_info:
        st.markdown("""
        **Sequence Characteristics**:
        - **Initial State (Day 03)**: Severity = 0.0000
        - **Peak State (Day 06–08)**: Cloud Cover / Overcast inflates pavement contrast -> Peak Severity = **0.7285** (12 unique tracks).
        - **Final State (Day 10)**: Low-angle Sunset lighting triggers RoadMarkingSuppressor over-filtering -> Severity collapses to **0.0000**.
        - **Key Lesson**: Optical perception models are sensitive to environmental contrast changes. Temporal filtering and weather normalization are essential for operational deployment.
        """)

    with col_plot:
        plot_p = PLOTS_DIR / "fig3_seg001_environmental_response.png"
        if plot_p.exists():
            st.image(str(plot_p), caption="Figure 3: Environmental Sensitivity & False Suppression under Weather Shifts", use_container_width=True)


def render_seg004_weather_confound_view():
    st.markdown("### SEG_004 D06–D10: Weather Glare & Rain Artifacts")
    st.markdown("""
    <div class="callout-box danger">
        <strong>Severe Weather Confound Notice:</strong><br>
        Heavy Rain on Days 07–08 produced water reflections and specular highlights on asphalt, inflating detected defect regions to 12 clusters. On Days 09–10, extreme sunset glare caused full road-corridor suppression.
    </div>
    """, unsafe_allow_html=True)

    plot_p = PLOTS_DIR / "seg004_d06_d10_temporal_response.png"
    if plot_p.exists():
        st.image(str(plot_p), caption="SEG_004 D06–D10 Extreme Environmental Response Curve")


def render_generic_sequence_view(seq_id: str, camera: str, num_states: int, s_init: str, s_final: str, delta: str, note: str):
    st.markdown(f"### {seq_id}: Subsequence Pair Analysis")
    st.markdown(f"**Camera Geometry**: `{camera}` | **Observed States**: `{num_states}` consecutive frames")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Initial Severity", s_init)
    with c2:
        st.metric("Final Severity", s_final)
    with c3:
        st.metric("Observed Severity Delta", delta)

    st.markdown(f"**Analysis Note**: {note}")
