"""Temporal Road Monitoring & Repeated Inspection Component for RoadSentinel Dashboard V2.

Provides:
- Dynamic discovery of stored physical segment observations from env/output/temporal_segments/
- Zero image-upload requirement (reads directly from repository disk)
- VIEW A: Full Segment History (all stored observations, metadata sidecars, and ineligibility reasons)
- VIEW B: Validated Temporal Sequences (geometric tracking, change quantification, and event taxonomy)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from data_loader import (
    WORKSPACE_ROOT,
    discover_segment_observations,
    discover_temporal_segments,
    load_canonical_metrics,
    load_sequence_daily_summary,
    load_sequence_events,
    load_sequence_progression,
    load_temporal_sequences_table,
)

PLOTS_DIR = WORKSPACE_ROOT / "integration/experiment_a/temporal/plots"

VALIDATED_SEQUENCES_MAP = {
    "SEG_001": [
        {
            "id": "SEG_001_D03_D10",
            "title": "SEG_001 D03–D10 — Environmental Modulation Sequence (Overcast / Rain / Sunset Glare)",
            "states": "Days 03 to 10 (8 states)",
            "camera": "🌄 Highway Curve Vantage Overlook",
            "key_takeaway": "Severity rises under overcast contrast (Day 06–08) and collapses under low-angle sunset lighting (Day 10) due to RoadMarkingSuppressor over-filtering.",
        }
    ],
    "SEG_002": [
        {
            "id": "SEG_002_D01_D02",
            "title": "SEG_002 D01–D02 — Highway Overlook Baseline Pair",
            "states": "Days 01 to 02 (2 states)",
            "camera": "🌄 Highway Curve Vantage Overlook",
            "key_takeaway": "Stable shoulder aggregate tracking across initial clear noon observations (Severity Δ: +0.0343).",
        },
        {
            "id": "SEG_002_D04_D05",
            "title": "SEG_002 D04–D05 — Moderate Overcast Shift Pair",
            "states": "Days 04 to 05 (2 states)",
            "camera": "🌄 Highway Curve Vantage Overlook",
            "key_takeaway": "Diffuse cloud cover inflates pavement contrast, increasing candidate defect area.",
        },
        {
            "id": "SEG_002_D06_D07",
            "title": "SEG_002 D06–D07 — Late Afternoon Low-Angle Pair",
            "states": "Days 06 to 07 (2 states)",
            "camera": "🌄 Highway Curve Vantage Overlook",
            "key_takeaway": "Shoulder tracking retained under grazing 35° sun angle.",
        },
    ],
    "SEG_003": [
        {
            "id": "SEG_003_D01_D07",
            "title": "SEG_003 D01–D07 — Model Response Stability Control Test (CV: 0.67%)",
            "states": "Days 01 to 07 (7 states)",
            "camera": "🌄 Highway Curve Vantage Overlook",
            "key_takeaway": "Near-zero coefficient of variation (CV = 0.67%) across 7 consecutive days confirms outstanding perception repeatability under fixed lighting. The single tracked region is a systematic shoulder false-positive candidate.",
        },
        {
            "id": "SEG_003_D08_D09",
            "title": "SEG_003 D08–D09 — Overcast Diffuse Lighting Pair",
            "states": "Days 08 to 09 (2 states)",
            "camera": "🌄 Highway Curve Vantage Overlook",
            "key_takeaway": "Invariant control verification following initial 7-day run under diffuse overcast conditions.",
        },
    ],
    "SEG_004": [
        {
            "id": "SEG_004_D01_D05",
            "title": "SEG_004 D01–D05 — Model-Observed Progression Sequence (Top-Down Drone, Δ: +0.7302)",
            "states": "Days 01 to 05 (5 states)",
            "camera": "🔭 Overhead Drone Survey (SAM 2 Top-Down)",
            "key_takeaway": "Model-observed severity progression across simulated road states under a fixed camera viewpoint (0.0000 → 0.7302, Δ = +0.7302). Tracking captures continuous defect area expansion.",
        },
        {
            "id": "SEG_004_D06_D10",
            "title": "SEG_004 D06–D10 — Extreme Weather & Sunset Flare Confound",
            "states": "Days 06 to 10 (5 states)",
            "camera": "🔭 Overhead Drone Survey (SAM 2 Top-Down)",
            "key_takeaway": "Rain puddles on Days 07–08 inflate defect count to 12 clusters; sunset glare on Days 09–10 triggers full road corridor suppression.",
        },
    ],
}


def render_temporal_monitoring():
    st.markdown("## Repeated Inspection & Temporal Road Monitoring")
    st.markdown(
        "Evaluate multi-day road surveillance across repeated camera inspections stored in "
        "`env/output/temporal_segments/`. No image uploads required — observations are discovered "
        "dynamically from the project filesystem and sidecar metadata."
    )

    canon = load_canonical_metrics()

    # 1. Scientific Disclaimer & Tracking Engine Callout
    st.markdown("""
    <div class="callout-box warn">
        <strong>CRITICAL SCIENTIFIC WORDING & TRACKING SPECIFICATION:</strong><br>
        • <strong>Strict Phrasing</strong>: Multi-day metrics represent <strong>MODEL-OBSERVED TEMPORAL CHANGE</strong> across simulated road states and environmental conditions, NOT certified ground-truth physical pavement deterioration.<br>
        • <strong>Hierarchical Tracking Engine</strong>: Greedy one-to-one bipartite matching executed in strict order:
          <code>Mask IoU ≥ 0.50</code> → <code>BBox IoU ≥ 0.30</code> → <code>Centroid Distance ≤ 75 px + Area Ratio ≤ 3.0×</code> (No Hungarian assignment).
    </div>
    """, unsafe_allow_html=True)

    # 2. Dynamic Segment Selector & Execution Mode
    segments = discover_temporal_segments()
    if not segments:
        st.error("No temporal segment directories discovered in env/output/temporal_segments/.")
        return

    ctrl_col1, ctrl_col2 = st.columns([2, 2])
    with ctrl_col1:
        selected_segment = st.selectbox(
            "Select Road Segment (Dynamically Discovered)",
            segments,
            index=segments.index("SEG_004") if "SEG_004" in segments else 0,
            help="Discovered dynamically from env/output/temporal_segments/. SEG_005/SEG_006 will appear automatically when captured."
        )

    with ctrl_col2:
        exec_mode = st.radio(
            "Execution & Evaluation Mode",
            ["Verified Demo Mode (Precomputed)", "Live Prototype Mode (Rerun YOLO on Stored Images)"],
            index=0,
            horizontal=True,
            help="Verified Demo Mode loads frozen Phase-2/3/4/11 outputs instantly. Live Mode runs YOLO directly on the stored segment files without re-uploading."
        )

    # Discover observations for the chosen segment
    observations = discover_segment_observations(selected_segment)

    st.markdown("---")

    # 3. Two Primary Views Tab Navigation
    tab_view_a, tab_view_b, tab_events = st.tabs([
        "VIEW A: Full Segment History (All Observations)",
        "VIEW B: Validated Temporal Sequences",
        "Canonical Event Taxonomy & Research Figures"
    ])

    # ==========================================
    # VIEW A: FULL SEGMENT HISTORY
    # ==========================================
    with tab_view_a:
        st.markdown(f"### VIEW A: Full Observation History for `{selected_segment}`")
        st.markdown(
            f"Displays all **{len(observations)} physical observations** stored in `env/output/temporal_segments/{selected_segment}/`. "
            "Includes all capture metadata sidecars, pavement distress metrics, and explicit temporal eligibility reasons."
        )

        if not observations:
            st.warning(f"No stored day captures found for {selected_segment}.")
        else:
            # Summary Table
            summary_rows = []
            for obs in observations:
                summary_rows.append({
                    "Day": obs["day_label"],
                    "Metadata": obs["metadata_status"],
                    "Camera Preset": obs["camera_preset"],
                    "Lighting": obs["lighting_preset"],
                    "Pavement State": obs["road_health_state"],
                    "Moisture": obs["moisture_state"],
                    "Severity": f"{obs['current_severity']:.4f}",
                    "Defects": obs["defect_count"],
                    "Area Ratio": f"{obs['defect_area_ratio']:.6f}",
                    "Anomaly Score": f"{obs['surface_anomaly_score']:.4f}",
                    "Temporal Status": obs["temporal_status"],
                    "Ineligibility Reason": obs["exclusion_reason"] if obs["temporal_status"] != "ELIGIBLE" else "None (Eligible)",
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Physical Observation Cards & Sidecar Inspector")

            # Day Selector for detailed view
            day_labels = [obs["day_label"] for obs in observations]
            selected_day_label = st.selectbox(
                "Select Day Observation to Inspect:",
                day_labels,
                index=min(4, len(day_labels) - 1),
                key="view_a_day_select"
            )
            obs_match = next((o for o in observations if o["day_label"] == selected_day_label), observations[0])

            col_img, col_info = st.columns([1.2, 1])
            with col_img:
                img_path = Path(obs_match["image_path"])
                if img_path.exists():
                    if "Live" in exec_mode:
                        try:
                            import cv2
                            from ultralytics import YOLO
                            yolo_weights = WORKSPACE_ROOT / "yolo/weights/best.pt"
                            model = YOLO(str(yolo_weights))
                            res = model(str(img_path), conf=0.25, imgsz=512, verbose=False)[0]
                            canvas = cv2.imread(str(img_path))
                            for box in res.boxes:
                                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                                conf = float(box.conf.item())
                                cname = {0: "D00", 1: "D10", 2: "D20", 3: "D40", 4: "Repair"}.get(int(box.cls.item()), "Defect")
                                cv2.rectangle(canvas, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 140, 255), 2)
                                cv2.putText(canvas, f"{cname} {conf:.2f}", (xyxy[0], max(15, xyxy[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
                            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"⚡ Live YOLOv8n GPU Inference on {selected_segment} {selected_day_label}", use_container_width=True)
                        except Exception as e:
                            st.image(str(img_path), caption=f"{selected_segment} — {selected_day_label} (Raw Capture)", use_container_width=True)
                    else:
                        st.image(str(img_path), caption=f"{selected_segment} — {selected_day_label} ({obs_match['camera_preset']})", use_container_width=True)
                else:
                    st.warning(f"Image asset missing on disk at `{img_path}`.")

            with col_info:
                # Eligibility Badge
                if obs_match["temporal_status"] == "ELIGIBLE":
                    st.markdown("""
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                        <span style="color: #34d399; font-weight: bold; font-size: 14px;">TEMPORAL ELIGIBLE</span>
                        <div style="color: #cbd5e1; font-size: 12px; margin-top: 4px;">Compatible fixed camera geometry and continuous multi-state progression.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(234, 179, 8, 0.15); border: 1px solid #eab308; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                        <span style="color: #facc15; font-weight: bold; font-size: 14px;">TEMPORAL MATCH NOT APPLICABLE</span>
                        <div style="color: #fef08a; font-size: 12px; margin-top: 4px;"><b>Reason:</b> {obs_match['exclusion_reason']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Metadata Sidecar Details
                if obs_match["has_metadata"]:
                    st.markdown(f"""
                    **Metadata Sidecar Status**: `VALID`  
                    - **Camera Geometry**: {obs_match['camera_preset']}  
                    - **Lighting / Solar Angle**: {obs_match['lighting_preset']}  
                    - **Pavement Evolution State**: {obs_match['road_health_state']}  
                    - **Moisture State**: {obs_match['moisture_state']}  
                    """)
                else:
                    st.markdown("""
                    <div class="callout-box danger">
                        <strong>METADATA MISSING (Diagnostic Case):</strong><br>
                        Simulation metadata sidecar was intentionally omitted for this capture (diagnostic verification).
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                # Perception & Deterioration Metrics
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Current Severity", f"{obs_match['current_severity']:.4f}")
                    st.metric("Defect Count", obs_match["defect_count"])
                with m2:
                    st.metric("Defect Area Ratio", f"{obs_match['defect_area_ratio']:.6f}")
                    st.metric("Surface Anomaly", f"{obs_match['surface_anomaly_score']:.4f}")

    # ==========================================
    # VIEW B: VALIDATED TEMPORAL SEQUENCES
    # ==========================================
    with tab_view_b:
        st.markdown(f"### VIEW B: Validated Compatible Sequences for `{selected_segment}`")
        st.markdown(
            "To preserve geometric tracking integrity, bipartite correspondence is executed **only on validated sequences** "
            "with constant camera geometry. Incompatible transitions (viewpoint switches, isolated frames) are safely excluded."
        )

        val_seqs = VALIDATED_SEQUENCES_MAP.get(selected_segment, [])
        if not val_seqs:
            st.info(f"No validated multi-state temporal sequences defined yet for {selected_segment}.")
        else:
            seq_options = [s["title"] for s in val_seqs]
            selected_seq_title = st.selectbox("Select Validated Sequence:", seq_options, index=0)
            selected_seq_obj = next(s for s in val_seqs if s["title"] == selected_seq_title)
            selected_seq_id = selected_seq_obj["id"]

            st.markdown(f"""
            <div style="background: #1e293b; padding: 14px 18px; border-radius: 8px; border-left: 4px solid #38bdf8; margin: 12px 0 18px 0;">
                <b style="color: #38bdf8;">Sequence Scope:</b> {selected_seq_obj['states']} | <b>Camera:</b> {selected_seq_obj['camera']}<br>
                <span style="color: #cbd5e1; font-size: 13px; margin-top: 4px; display: block;"><b>Research Takeaway:</b> {selected_seq_obj['key_takeaway']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Load sequence outputs
            df_seq_daily = load_sequence_daily_summary(selected_seq_id)
            seq_events = load_sequence_events(selected_seq_id)
            seq_prog = load_sequence_progression(selected_seq_id)

            if not df_seq_daily.empty:
                st.markdown("#### MODEL-OBSERVED TEMPORAL CHANGE: Quantitative Trajectories")

                # Metric Line Charts
                chart_col1, chart_col2, chart_col3 = st.columns(3)
                with chart_col1:
                    st.markdown("**Current Severity vs Day**")
                    st.line_chart(df_seq_daily.set_index("day")["current_severity"], use_container_width=True)
                with chart_col2:
                    st.markdown("**Defect Area Ratio vs Day**")
                    st.line_chart(df_seq_daily.set_index("day")["defect_area_ratio"], use_container_width=True)
                with chart_col3:
                    st.markdown("**Candidate Defect Count vs Day**")
                    st.line_chart(df_seq_daily.set_index("day")["defect_count"], use_container_width=True)

                # Day-to-Day Change Table
                st.markdown("#### Day-to-Day Transition & Delta Table")
                disp_cols = [
                    "day", "current_severity", "current_severity_delta",
                    "defect_count", "defect_count_delta",
                    "defect_area_ratio", "defect_area_ratio_delta",
                    "surface_anomaly_score", "surface_anomaly_score_delta"
                ]
                available_disp = [c for c in disp_cols if c in df_seq_daily.columns]
                st.dataframe(df_seq_daily[available_disp], hide_index=True, use_container_width=True)

            # Events for this specific sequence
            if seq_events:
                st.markdown(f"#### Sequence Track Events ({len(seq_events)} transitions)")
                df_ev = pd.DataFrame(seq_events)
                st.dataframe(df_ev, hide_index=True, use_container_width=True)

            # Sequence figure if available
            p_seq_fig = PLOTS_DIR / f"fig2_{selected_segment.lower()}_drone_progression.png"
            if not p_seq_fig.exists():
                p_seq_fig = PLOTS_DIR / f"fig1_{selected_segment.lower()}_stability_control.png"
            if not p_seq_fig.exists():
                p_seq_fig = PLOTS_DIR / f"fig3_{selected_segment.lower()}_environmental_response.png"

            if p_seq_fig.exists():
                st.image(str(p_seq_fig), caption=f"Publication Figure for {selected_seq_id}", use_container_width=True)

    # ==========================================
    # CANONICAL EVENT TAXONOMY & FIGURES
    # ==========================================
    with tab_events:
        st.markdown("### Canonical Event Taxonomy & All 8 Sequences Overview")
        st.markdown(
            "Across all 8 temporal subsequences in Experiment A (40 captures, 33 matched transitions, 48 unique tracks), "
            "the bipartite tracking engine classified pairwise temporal transitions into 5 formal event types:"
        )

        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            st.metric("NEW_DEFECT", "48", help="A defect observed on Day T with no spatial correspondence on Day T-1.")
        with e2:
            st.metric("MATCHED_EXISTING", "33", help="Defect track maintained across consecutive observations (14 increased, 19 decreased).")
        with e3:
            st.metric("AREA_INCREASED", "14", help="Matched region exhibiting >15% observed bounding area growth (canonical count: 14).")
        with e4:
            st.metric("AREA_DECREASED", "19", help="Matched region exhibiting >15% observed area contraction (canonical count: 19).")
        with e5:
            st.metric("NOT_OBSERVED", "34", help="Region tracked previously but unobserved on Day T. NEVER marked as REPAIRED.")

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            p_event = PLOTS_DIR / "fig6_temporal_event_distribution.png"
            if p_event.exists():
                st.image(str(p_event), caption="Figure 6: Temporal Event Transition Distribution across all 8 Sequences", use_container_width=True)
        with col_fig2:
            p_track = PLOTS_DIR / "fig5_track_length_distribution.png"
            if p_track.exists():
                st.image(str(p_track), caption="Figure 5: Track Lifespan Retention Distribution (Longest track: 7 states)", use_container_width=True)

        # Full Sequences Table
        seq_table = load_temporal_sequences_table()
        if not seq_table.empty:
            with st.expander("View Full Sequences Metadata Table (All 8 Subsequences)"):
                st.dataframe(seq_table, use_container_width=True)
