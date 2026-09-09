"""Temporal Road Monitoring Component for RoadSentinel Dashboard."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
from data_loader import WORKSPACE_ROOT, load_temporal_sequences_table

PLOTS_DIR = WORKSPACE_ROOT / "integration/experiment_a/temporal/plots"


def render_temporal_monitoring():
    st.markdown("## Temporal Road Monitoring & Multi-Day Analytics")
    st.markdown(
        "Explore multi-day defect tracking across repeated camera inspections. "
        "Evaluates temporal correspondence, track persistence, and distinguishes systematic "
        "sensor responses from environmental confounds."
    )

    seq_table = load_temporal_sequences_table()

    # Sequence Selector
    st.markdown("### Select Temporal Inspection Sequence")
    sequences = [
        "SEG_003_D01_D07 — Stability Control Test (CV: 0.67%)",
        "SEG_004_D01_D05 — Monotonic Progression Sequence (Δ: +0.7302)",
        "SEG_001_D03_D10 — Environmental Confound Sequence (Overcast/Rain)",
        "SEG_004_D06_D10 — Extreme Weather & Sunset Glare Confound",
    ]
    selected_seq_label = st.selectbox("Sequence Selection", sequences, index=0)
    selected_seq_id = selected_seq_label.split(" — ")[0].strip()

    st.markdown("---")

    # Sequence Views
    if selected_seq_id == "SEG_003_D01_D07":
        render_seg003_control_view()
    elif selected_seq_id == "SEG_004_D01_D05":
        render_seg004_progression_view()
    elif selected_seq_id == "SEG_001_D03_D10":
        render_seg001_confound_view()
    else:
        render_seg004_weather_confound_view()

    st.markdown("---")

    # Temporal Event Taxonomy Section
    st.markdown("### Temporal Event Taxonomy & Change Analysis")
    st.markdown(
        "Across all 8 temporal subsequences in Experiment A, the bipartite tracking engine classified "
        "pairwise temporal transitions into 5 formal event types:"
    )

    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.metric("NEW_DEFECT", "48", help="A defect observed on Day T with no spatial correspondence on Day T-1.")
    with e2:
        st.metric("MATCHED_EXISTING", "33", help="Defect track maintained across consecutive observations.")
    with e3:
        st.metric("AREA_INCREASED", "21", help="Matched region exhibiting >15% observed bounding area growth.")
    with e4:
        st.metric("AREA_DECREASED", "12", help="Matched region exhibiting >15% observed area contraction.")
    with e5:
        st.metric("NOT_OBSERVED", "34", help="Region tracked previously but unobserved on Day T. NEVER marked as REPAIRED.")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        p_event = PLOTS_DIR / "fig6_temporal_event_distribution.png"
        if p_event.exists():
            st.image(str(p_event), caption="Figure 6: Temporal Event Transition Distribution (Phase 4)")
    with col_fig2:
        p_track = PLOTS_DIR / "fig5_track_length_distribution.png"
        if p_track.exists():
            st.image(str(p_track), caption="Figure 5: Track Length Lifespan Distribution")

    # Sequences Table
    if not seq_table.empty:
        with st.expander("View All 8 Evaluated Temporal Sequences Table"):
            st.dataframe(seq_table[["sequence_id", "camera_preset", "num_states", "initial_severity", "final_severity", "severity_change", "unique_tracks", "evidence_grade"]], use_container_width=True)


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
            st.metric("Coefficient of Var. (CV)", "0.67%", delta="Near-Zero Variance", delta_color="normal")
            st.metric("Longest Track", "7 States", delta="100% Track Retention")

        st.markdown("""
        <div class="callout-box success">
            <strong>Critical Research Interpretation:</strong><br>
            The near-zero coefficient of variation (<strong>CV = 0.67%</strong>) proves that when physical pavement and lighting remain constant, the DINOv2+SAM2 perception pipeline exhibits exceptional repeatability.<br><br>
            <strong>Examiner Note:</strong> The persistent detected region is a <em>systematic false-positive candidate</em> (road shoulder aggregate contrast under grazing perspective), not confirmed physical damage.
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        plot_p = PLOTS_DIR / "fig1_seg003_stability_control.png"
        if plot_p.exists():
            st.image(str(plot_p), caption="Figure 1: SEG_003 Multi-State Severity & Defect Count Invariance", use_container_width=True)


def render_seg004_progression_view():
    st.markdown("### SEG_004 D01–D05: Model-Observed Progression Sequence")

    col_info, col_plot = st.columns([1, 1.2])
    with col_info:
        st.markdown("""
        **Experimental Condition**:
        - **Pavement Evolution**: Unblemished baseline transitioning to severe road breakdown
        - **Camera Geometry**: Fixed Overhead Drone Survey (Top-Down Nadir)
        - **Inspection Horizon**: 5 Consecutive States (Days 01–05)
        """)

        st.markdown("#### Measured Severity Progression")
        prog_df = pd.DataFrame({
            "Day": ["Day 01", "Day 02", "Day 03", "Day 04", "Day 05"],
            "Measured Severity": [0.0000, 0.0000, 0.2564, 0.6542, 0.7302],
            "Observed State": ["Baseline (0 defects)", "Baseline (0 defects)", "Initial Crack (1 defect)", "Branching Cracks (2 defects)", "Critical Breakdown (2 defects)"]
        })
        st.dataframe(prog_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="callout-box">
            <strong>Scientific Nuance:</strong><br>
            The sequence displays a clean monotonic severity growth (<strong>Δ = +0.7302</strong>) tracking physical road degradation.<br>
            <em>Do not describe this as a laboratory-perfect deterioration measurement:</em> while the drone camera angle is fixed top-down, subtle ambient illumination variations exist between states.
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        plot_p = PLOTS_DIR / "fig2_seg004_drone_progression.png"
        if plot_p.exists():
            st.image(str(plot_p), caption="Figure 2: SEG_004 Drone Nadir Severity & Defect Progression", use_container_width=True)


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
