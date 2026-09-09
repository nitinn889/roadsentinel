"""RoadSentinel Driver User Dashboard Application.

Comprehensive interactive user/driver demonstration platform connecting:
- 2.4 km continuous simulated road corridor (SEG_001 through SEG_006)
- Haversine distance, route interpolation, and 50m road geofence
- Real-time car driving simulation (0-100 km/h) with manual GPS test mode
- Model-derived 400m look-ahead road health score
- Real-time hazard detection, ETA to hazard, and non-directive driver alerts
- PyDeck map visualization and 6-segment road strip status
- CITIZEN ROAD-DAMAGE REPORTING + AI VERIFICATION (YOLO, DINOv2, SAM2, Reliability, Severity)
- Persistent local storage of citizen reports (RS-CR-XXXX)
- Offline-ready, zero Unreal/CARLA dependency, and 100% repository data consumption

Launch command:
    streamlit run driver_dashboard/app.py --server.port 8502
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from PIL import Image
import pydeck as pdk
import streamlit as st

# Ensure driver_dashboard directory is in sys.path
DRIVER_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DRIVER_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DASHBOARD_DIR))

WORKSPACE_ROOT = DRIVER_DASHBOARD_DIR.parent

from config import (
    ALERT_LEVEL_ADVISORY,
    ALERT_LEVEL_URGENT,
    ALERT_LEVEL_WARNING,
    DEFAULT_SPEED_KMH,
    DISCLAIMER_TEXT,
    LOOKAHEAD_DISTANCE_M,
    MISSING_DATA_SEGMENTS,
    ROAD_GEOFENCE_RADIUS_M,
    SEGMENT_BOUNDARIES,
    SEGMENT_NAMES,
    START_LATITUDE,
    START_LONGITUDE,
    TOTAL_ROAD_LENGTH_M,
)
from data_loader import get_segment_day_info, load_primary_results_data
from gps import check_geofence, haversine_distance
from hazard_engine import evaluate_alert_status, get_hazards_for_day, get_nearest_hazard_ahead
from route import generate_deterministic_route, get_gps_at_distance, get_segment_at_distance
from simulation import SimulationState, compute_400m_lookahead_health, compute_road_health_score

from citizen_reporting import (
    CitizenReportingPipeline,
    analyze_citizen_report,
    save_citizen_report,
    load_all_reports,
    STATUS_VERIFIED,
    STATUS_MANUAL_REVIEW,
    STATUS_RETAKE,
    REPORTS_DIR,
    RELIABILITY_HIGH_THRESH,
    RELIABILITY_LOW_THRESH,
)


@st.cache_resource(show_spinner="Loading RoadSentinel AI Perception Models (YOLO, DINOv2, SAM2)...")
def get_ai_pipeline() -> CitizenReportingPipeline:
    """Load and cache YOLO, DINOv2, and SAM2 models once in memory."""
    return CitizenReportingPipeline()


def safe_render_image(image_source: Any, caption: str = "") -> None:
    """Safely render an image in Streamlit, catching corrupted or invalid formats without crashing."""
    if image_source is None:
        st.caption("No image data available.")
        return

    try:
        if isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if not p.exists():
                st.warning(f"⚠️ Image file not found on disk: `{p.name}`")
                return
            with Image.open(p) as img:
                img.load()
                st.image(img, caption=caption if caption else None, use_container_width=True)
        elif isinstance(image_source, (bytes, bytearray)):
            if len(image_source) == 0:
                st.caption("Empty image data.")
                return
            with Image.open(io.BytesIO(image_source)) as img:
                img.load()
                st.image(img, caption=caption if caption else None, use_container_width=True)
        else:
            st.image(image_source, caption=caption if caption else None, use_container_width=True)
    except Exception as err:
        st.warning(f"⚠️ Image preview unavailable ({type(err).__name__}: {err})")


def init_session_state():
    """Initialize persistent Streamlit session state for driver simulation and citizen reporting."""
    if "sim_state" not in st.session_state:
        st.session_state.sim_state = SimulationState(day=5, speed_kmh=DEFAULT_SPEED_KMH)
    if "mode" not in st.session_state:
        st.session_state.mode = "Simulated Drive"
    if "manual_lat" not in st.session_state:
        st.session_state.manual_lat = START_LATITUDE
    if "manual_lon" not in st.session_state:
        st.session_state.manual_lon = START_LONGITUDE
    if "manual_speed" not in st.session_state:
        st.session_state.manual_speed = 40.0
    if "last_tick_time" not in st.session_state:
        st.session_state.last_tick_time = time.time()

    # Citizen Reporting Session State
    if "citizen_analysis" not in st.session_state:
        st.session_state.citizen_analysis = None
    if "citizen_image_bytes" not in st.session_state:
        st.session_state.citizen_image_bytes = None
    if "citizen_image_name" not in st.session_state:
        st.session_state.citizen_image_name = None
    if "citizen_saved_id" not in st.session_state:
        st.session_state.citizen_saved_id = None


def render_header():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a, #1e293b); padding: 20px 24px; border-radius: 12px; border-left: 6px solid #38bdf8; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="color: #f8fafc; margin: 0; font-size: 26px; font-weight: 800; font-family: 'Outfit', sans-serif;">
                    🛣️ RoadSentinel Driver Dashboard
                </h1>
                <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 14px;">
                    Real-time driver road-health intelligence, hazard warning, 400m look-ahead simulation, and citizen damage reporting.
                </p>
            </div>
            <div style="background: #0284c7; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px;">
                PROTOTYPE USER MODE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_citizen_reporting_tab(pipeline: CitizenReportingPipeline, curr_lat: float, curr_lon: float):
    st.markdown("### 📸 Citizen Road-Damage Reporting + AI Verification")
    st.markdown(
        "Upload a road or pothole photo to automatically evaluate damage using the existing RoadSentinel pipeline. "
        "The system runs YOLO defect detection, DINOv2 visual domain familiarity check, SAM2 defect segmentation, "
        "model-derived severity calculation, perception reliability scoring, and route corridor geofencing."
    )

    col_input, col_meta = st.columns([1.5, 1])

    with col_input:
        st.markdown("#### 1. Select or Upload Road Photo")
        st.caption("Quick Test Presets (Existing Repo Datasets):")
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            if st.button("🕳️ Pothole Sample", use_container_width=True):
                sample_p = WORKSPACE_ROOT / "RoadSentinel_datasets/mwpd/Multi-Weather Pothole Detection (MWPD)/MWPD/test/images/img-455_jpg.rf.b365528f28600d6182bee8a0eb821978.jpg"
                if not sample_p.exists():
                    sample_p = WORKSPACE_ROOT / "benchmark/common_candidate/images/0454.png"
                if sample_p.exists():
                    st.session_state.citizen_image_bytes = sample_p.read_bytes()
                    st.session_state.citizen_image_name = sample_p.name
                    st.session_state.citizen_analysis = None
                    st.session_state.citizen_saved_id = None
        with p_col2:
            if st.button("🩹 Defect/Repair Sample", use_container_width=True):
                sample_p = WORKSPACE_ROOT / "benchmark/common_candidate/images/China_Drone_000104.jpg"
                if sample_p.exists():
                    st.session_state.citizen_image_bytes = sample_p.read_bytes()
                    st.session_state.citizen_image_name = sample_p.name
                    st.session_state.citizen_analysis = None
                    st.session_state.citizen_saved_id = None
        with p_col3:
            if st.button("🛣️ Pristine Road Baseline", use_container_width=True):
                sample_p = WORKSPACE_ROOT / "benchmark/common_candidate/images/original_healthy.jpg"
                if sample_p.exists():
                    st.session_state.citizen_image_bytes = sample_p.read_bytes()
                    st.session_state.citizen_image_name = sample_p.name
                    st.session_state.citizen_analysis = None
                    st.session_state.citizen_saved_id = None

        uploaded_file = st.file_uploader(
            "Or Upload Photo from Device",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="citizen_file_uploader",
        )
        if uploaded_file is not None:
            st.session_state.citizen_image_bytes = uploaded_file.getvalue()
            st.session_state.citizen_image_name = uploaded_file.name
            st.session_state.citizen_analysis = None
            st.session_state.citizen_saved_id = None

        if st.session_state.citizen_image_name:
            st.info(f"Selected Image: **{st.session_state.citizen_image_name}** ({len(st.session_state.citizen_image_bytes) // 1024} KB)")

    with col_meta:
        st.markdown("#### 2. Optional GPS Location")
        st.caption("Provide coordinates to map report against the 2.4 km RoadSentinel corridor:")

        gps_toggle = st.checkbox("Include GPS Location", value=True, key="citizen_gps_toggle")

        lat_val = None
        lon_val = None
        if gps_toggle:
            c_lat, c_lon = st.columns(2)
            with c_lat:
                lat_input = st.number_input("Latitude", value=float(curr_lat), format="%.6f", key="citizen_lat_in")
            with c_lon:
                lon_input = st.number_input("Longitude", value=float(curr_lon), format="%.6f", key="citizen_lon_in")

            lat_val = lat_input
            lon_val = lon_input
        else:
            st.info("GPS coordinates omitted (report will be marked as LOCATION NOT PROVIDED).")

    # Action buttons
    act_col1, act_col2 = st.columns([1, 1])
    with act_col1:
        analyze_btn = st.button("🔍 Analyze & Verify Report", type="primary", use_container_width=True)
    with act_col2:
        save_btn = st.button(
            "💾 Submit & Store Report Locally",
            use_container_width=True,
            disabled=(st.session_state.citizen_analysis is None),
        )

    if analyze_btn:
        if st.session_state.citizen_image_bytes is None:
            st.error("Please upload or select a road photo before analyzing.")
        else:
            with st.spinner("Executing RoadSentinel AI Pipeline (YOLO + DINOv2 + SAM2)..."):
                analysis_res = analyze_citizen_report(
                    file_bytes=st.session_state.citizen_image_bytes,
                    pipeline=pipeline,
                    latitude=lat_val,
                    longitude=lon_val,
                )
                st.session_state.citizen_analysis = analysis_res
                st.session_state.citizen_saved_id = None
            st.rerun()

    if save_btn and st.session_state.citizen_analysis is not None:
        report_id = save_citizen_report(
            raw_image_bytes=st.session_state.citizen_image_bytes,
            analysis_result=st.session_state.citizen_analysis,
        )
        st.session_state.citizen_saved_id = report_id
        st.success(f"🎉 Report submitted and stored locally as **{report_id}** in `driver_dashboard/data/citizen_reports/`!")

    # Display Results Card
    if st.session_state.citizen_analysis is not None:
        res = st.session_state.citizen_analysis
        st.markdown("---")
        st.markdown("### 📊 AI Verification Results Card")

        # Status Banner
        status = res["status"]
        if status == STATUS_VERIFIED:
            st.success(f"### 🟢 {status}\n\n**Decision Reason:** {res['decision_reason']}")
        elif status == STATUS_MANUAL_REVIEW:
            st.warning(f"### 🟡 {status}\n\n**Decision Reason:** {res['decision_reason']}")
        else:
            st.error(f"### 🔴 {status}\n\n**Decision Reason:** {res['decision_reason']}")

        # Images Row
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown(f"**Original Uploaded Image** (`{st.session_state.citizen_image_name or 'photo'}`)")
            if st.session_state.citizen_image_bytes:
                safe_render_image(st.session_state.citizen_image_bytes)
        with img_col2:
            st.markdown("**AI Perception Overlay (YOLO Bounding Box + SAM2 Defect Mask)**")
            if res.get("annotated_image_bytes"):
                safe_render_image(res["annotated_image_bytes"])
            else:
                st.caption("No defect overlay generated.")

        # Metrics KPI Grid
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Defect Type", res["defect_type"])
        with m2:
            conf_str = f"{res['yolo_confidence']*100:.1f}%" if res["defect_detected"] else "0.0%"
            st.metric("YOLO Confidence", conf_str)
        with m3:
            st.metric("Perception Reliability", f"{res['reliability_level']} ({res['reliability_score']:.2f})")
        with m4:
            st.metric("Domain Familiarity", f"{res['domain_familiarity']} (d={res['domain_distance']:.4f})")
        with m5:
            st.metric("Model-Derived Severity", f"{res['model_severity']:.2f} / 1.0", help="MODEL-DERIVED SEVERITY — NOT PCI")
            st.caption("<span style='font-size:10px; color:#94a3b8;'>MODEL-DERIVED SEVERITY — NOT PCI</span>", unsafe_allow_html=True)

        # GPS & Corridor Information Card
        gps = res["gps_info"]
        st.markdown(f"""
        <div style="background: #1e293b; padding: 14px 18px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #38bdf8;">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div><strong>📍 Location:</strong> <code>{gps['gps_display']}</code></div>
                <div><strong>🛣️ Geofence Corridor:</strong> <code>{gps['geofence_status']}</code></div>
                <div><strong>📍 Nearest Road Segment:</strong> <code>{gps['nearest_segment']}</code></div>
                <div><strong>🎯 Distance to Route:</strong> <code>{gps['distance_to_corridor_m'] if gps['distance_to_corridor_m'] is not None else 'N/A'}m</code></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Diagnostics & Evidence Breakdown
        with st.expander("🔍 AI Pipeline Diagnostics & Evidence Breakdown"):
            st.markdown(f"""
            - **SAM2 Segmentation Mask**: `{'Generated' if res['has_sam2_mask'] else 'No defect mask generated'}` (Area Ratio: `{res['mask_area_ratio']:.6f}`)
            - **DINOv2 Domain Gate**: `{res.get('domain_description', 'N/A')}`
            - **Detections Count**: `{len(res.get('detections', []))}`
            - **Raw Detections**: `{res.get('detections', [])}`
            - **Reliability Logistic Score**: `{res['reliability_score']:.4f}` (Thresholds: Low < {RELIABILITY_LOW_THRESH}, High >= {RELIABILITY_HIGH_THRESH})
            """)

        st.caption("Notice: Road damage analysis is generated by research AI models and does not substitute for official civil-engineering inspection.")


def render_citizen_reports_history_tab():
    st.markdown("### 📋 Stored Local Citizen Reports Archive")
    st.markdown("All verified citizen submissions and AI evaluations stored locally in `driver_dashboard/data/citizen_reports/`.")

    reports = load_all_reports()
    if not reports:
        st.info("No citizen reports stored yet. Go to the '📸 Citizen Report Road Damage' tab to analyze and submit a report.")
        return

    # Filter by status
    statuses = ["ALL", STATUS_VERIFIED, STATUS_MANUAL_REVIEW, STATUS_RETAKE]
    selected_filter = st.selectbox("Filter Reports by Verification Status", statuses)

    filtered_reports = reports if selected_filter == "ALL" else [r for r in reports if r.get("status") == selected_filter]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Reports", len(reports))
    with c2:
        st.metric("Verified Reports", len([r for r in reports if r.get("status") == STATUS_VERIFIED]))
    with c3:
        st.metric("Manual Review Required", len([r for r in reports if r.get("status") == STATUS_MANUAL_REVIEW]))
    with c4:
        st.metric("Retake / No Damage", len([r for r in reports if r.get("status") == STATUS_RETAKE]))

    st.markdown("---")

    # Reports summary table
    table_data = []
    for r in reversed(filtered_reports):
        table_data.append({
            "Report ID": r.get("report_id"),
            "Timestamp": r.get("timestamp", "")[:19].replace("T", " "),
            "Status": r.get("status"),
            "Defect Type": r.get("defect_type"),
            "Confidence": f"{r.get('yolo_confidence', 0.0)*100:.1f}%",
            "Reliability": r.get("reliability_level"),
            "Severity": f"{r.get('model_severity', 0.0):.2f}",
            "Nearest Segment": r.get("nearest_segment"),
            "GPS": r.get("gps_display"),
        })
    st.dataframe(table_data, use_container_width=True)

    # Report Detail Viewer
    st.markdown("#### 🔎 Inspect Stored Report")
    report_ids = [r["report_id"] for r in reversed(filtered_reports)]
    if report_ids:
        chosen_id = st.selectbox("Select Report to Inspect", report_ids)
        chosen_record = next((r for r in reports if r["report_id"] == chosen_id), None)
        if chosen_record:
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                raw_path = REPORTS_DIR / chosen_record.get("raw_image_path", "")
                safe_render_image(raw_path, caption=f"Raw Upload ({chosen_id})")
            with v_col2:
                ann_path = REPORTS_DIR / chosen_record.get("annotated_image_path", "")
                safe_render_image(ann_path, caption=f"AI Annotated Overlay ({chosen_id})")

            with st.expander("📄 Full JSON Metadata Record"):
                st.json(chosen_record)


def main():
    st.set_page_config(
        page_title="RoadSentinel Driver Dashboard",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    sim = st.session_state.sim_state
    ai_pipeline = get_ai_pipeline()

    # Data & Route Initialization
    waypoints = generate_deterministic_route()
    primary_data = load_primary_results_data()

    # Calculate tick delta if simulation is running
    now = time.time()
    dt = now - st.session_state.last_tick_time
    st.session_state.last_tick_time = now
    if sim.is_running:
        sim.update(min(0.5, dt))  # Clamp dt to prevent large leaps

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎮 Drive Controls")

        mode = st.radio(
            "Operating Mode",
            ["Simulated Drive", "Manual GPS Test Mode"],
            index=0 if st.session_state.mode == "Simulated Drive" else 1,
            key="operating_mode_radio"
        )
        st.session_state.mode = mode

        st.markdown("---")

        # Inspection Day Selector
        day_num = st.selectbox(
            "Select Inspection Day",
            options=list(range(1, 11)),
            index=sim.day - 1,
            format_func=lambda d: f"Day {d:02d} Inspection State",
        )
        if day_num != sim.day:
            sim.set_day(day_num)

        st.markdown("---")

        if mode == "Simulated Drive":
            st.markdown("#### Vehicle Drive Simulation")

            # Start / Pause / Reset Controls
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                if st.button("▶ START", use_container_width=True):
                    sim.start()
            with b_col2:
                if st.button("⏸ PAUSE", use_container_width=True):
                    sim.pause()
            with b_col3:
                if st.button("🔄 RESET", use_container_width=True):
                    sim.reset()

            # Speed Slider
            speed_val = st.slider(
                "Vehicle Speed (km/h)",
                min_value=0.0,
                max_value=100.0,
                value=float(sim.speed_kmh),
                step=5.0,
            )
            sim.set_speed(speed_val)

            # Manual Position Slider
            dist_val = st.slider(
                "Position on Route (m)",
                min_value=0.0,
                max_value=float(TOTAL_ROAD_LENGTH_M),
                value=float(sim.distance_m),
                step=10.0,
            )
            if abs(dist_val - sim.distance_m) > 1.0 and not sim.is_running:
                sim.set_position(dist_val)

        else:
            st.markdown("#### Manual GPS Test Mode")
            st.caption("Enter arbitrary coordinates to test Haversine distance, geofence, and hazards ahead.")

            m_lat = st.number_input(
                "Latitude",
                value=float(st.session_state.manual_lat),
                format="%.6f",
                step=0.001,
            )
            m_lon = st.number_input(
                "Longitude",
                value=float(st.session_state.manual_lon),
                format="%.6f",
                step=0.001,
            )
            m_spd = st.number_input(
                "Speed (km/h)",
                value=float(st.session_state.manual_speed),
                min_value=0.0,
                max_value=100.0,
                step=5.0,
            )
            st.session_state.manual_lat = m_lat
            st.session_state.manual_lon = m_lon
            st.session_state.manual_speed = m_spd

            # Corridor preset buttons
            st.markdown("Quick Corridor Presets:")
            p1, p2 = st.columns(2)
            with p1:
                if st.button("Inside Route", use_container_width=True):
                    st.session_state.manual_lat = START_LATITUDE + 0.001
                    st.session_state.manual_lon = START_LONGITUDE + 0.001
                    st.rerun()
            with p2:
                if st.button("Outside Route", use_container_width=True):
                    st.session_state.manual_lat = START_LATITUDE + 0.01
                    st.session_state.manual_lon = START_LONGITUDE + 0.01
                    st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Corridor Status")
        st.caption("Active Day 01–10: SEG_001–SEG_004 verified. SEG_005–SEG_006 pending capture.")

    # Determine Active Coordinates & State
    if mode == "Simulated Drive":
        curr_dist_m = sim.distance_m
        curr_speed_kmh = sim.speed_kmh
        curr_lat, curr_lon = get_gps_at_distance(curr_dist_m, waypoints)
        curr_seg = get_segment_at_distance(curr_dist_m)
        geofence_res = check_geofence(curr_lat, curr_lon, waypoints)
    else:
        curr_lat = st.session_state.manual_lat
        curr_lon = st.session_state.manual_lon
        curr_speed_kmh = st.session_state.manual_speed
        geofence_res = check_geofence(curr_lat, curr_lon, waypoints)

        if geofence_res["is_on_route"] and geofence_res["nearest_waypoint"]:
            nearest_wp = geofence_res["nearest_waypoint"]
            curr_dist_m = nearest_wp["distance_m"]
            curr_seg = nearest_wp["segment_id"]
        else:
            curr_dist_m = 0.0
            curr_seg = "OFF_ROUTE"

    # Compute Hazards for Current Day
    hazards_for_day = get_hazards_for_day(sim.day)

    # Evaluate Geofence & Hazards Ahead
    if geofence_res["is_on_route"]:
        nearest_hazard = get_nearest_hazard_ahead(curr_dist_m, curr_lat, curr_lon, hazards_for_day)
        if nearest_hazard:
            speed_mps = (curr_speed_kmh / 3.6) if curr_speed_kmh > 0 else 0.0
            dist_ahead = nearest_hazard.get("distance_ahead_m", 0.0)
            nearest_hazard["eta_seconds"] = round(dist_ahead / speed_mps, 1) if speed_mps > 0 else None

        lookahead_info = compute_400m_lookahead_health(curr_dist_m, sim.day, primary_data)
        alert_obj, sim.fired_alerts = evaluate_alert_status(nearest_hazard, curr_speed_kmh, sim.fired_alerts)
        if alert_obj and alert_obj["is_new"]:
            sim.alert_history.append({
                "time": time.strftime("%H:%M:%S"),
                "segment": curr_seg,
                "distance_ahead_m": nearest_hazard["distance_ahead_m"],
                "hazard_type": nearest_hazard["hazard_type"],
                "level": alert_obj["level"],
                "message": alert_obj["message"],
            })
    else:
        nearest_hazard = None
        lookahead_info = {"health_score": None, "status": "OUTSIDE MONITORED ROUTE", "data_pending": True}
        alert_obj = None

    # Render Header Banner
    render_header()

    # Top Level Navigation Tabs
    tab_corridor, tab_citizen, tab_archive = st.tabs([
        "🚗 2.4 km Road Corridor & Simulation",
        "📸 Citizen Report Road Damage",
        "📋 Stored Citizen Reports Log",
    ])

    # -------------------------------------------------------------
    # TAB 1: 2.4 KM ROAD CORRIDOR & DRIVER SIMULATION
    # -------------------------------------------------------------
    with tab_corridor:
        # Top Driver KPI Dashboard Bar
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:
            st.markdown(f"""
            <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid #3b82f6;">
                <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">CURRENT SPEED</div>
                <div style="color: #f8fafc; font-size: 26px; font-weight: 800; margin: 2px 0;">{curr_speed_kmh:.0f} <span style="font-size: 14px; color: #94a3b8;">km/h</span></div>
                <div style="color: #60a5fa; font-size: 11px;">{curr_speed_kmh / 3.6:.1f} m/s</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            score_val = f"{lookahead_info['health_score']:.0f}" if lookahead_info['health_score'] is not None else "N/A"
            score_color = "#34d399" if lookahead_info['health_score'] and lookahead_info['health_score'] >= 75 else ("#facc15" if lookahead_info['health_score'] and lookahead_info['health_score'] >= 50 else "#f87171")
            st.markdown(f"""
            <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid {score_color};">
                <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">ROAD HEALTH AHEAD</div>
                <div style="color: {score_color}; font-size: 26px; font-weight: 800; margin: 2px 0;">{score_val} <span style="font-size: 14px; color: #94a3b8;">/ 100</span></div>
                <div style="color: #94a3b8; font-size: 11px;">400m Look-Ahead Window</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            if nearest_hazard:
                haz_txt = f"{nearest_hazard['hazard_type']} — {nearest_hazard['distance_ahead_m']:.0f}m"
                eta_val = nearest_hazard.get("eta_seconds")
                if eta_val is None and curr_speed_kmh > 0:
                    speed_mps = curr_speed_kmh / 3.6
                    eta_val = round(nearest_hazard["distance_ahead_m"] / speed_mps, 1) if speed_mps > 0 else None
                eta_txt = f"ETA: ~{eta_val:.1f}s" if eta_val is not None else "ETA: Stationary"
                haz_color = "#f87171" if nearest_hazard["distance_ahead_m"] <= 30 else ("#facc15" if nearest_hazard["distance_ahead_m"] <= 80 else "#38bdf8")
            else:
                haz_txt = "No Hazards Ahead"
                eta_txt = "Route Clear"
                haz_color = "#34d399"

            st.markdown(f"""
            <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid {haz_color};">
                <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">NEAREST HAZARD</div>
                <div style="color: #f8fafc; font-size: 18px; font-weight: 700; margin: 6px 0 2px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{haz_txt}</div>
                <div style="color: {haz_color}; font-size: 11px; font-weight: bold;">{eta_txt}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid #a855f7;">
                <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">CURRENT SEGMENT</div>
                <div style="color: #f8fafc; font-size: 26px; font-weight: 800; margin: 2px 0;">{curr_seg}</div>
                <div style="color: #c084fc; font-size: 11px;">Day {sim.day:02d} State</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi5:
            dist_remain_km = max(0.0, (TOTAL_ROAD_LENGTH_M - curr_dist_m) / 1000.0)
            st.markdown(f"""
            <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid #14b8a6;">
                <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">DISTANCE REMAINING</div>
                <div style="color: #f8fafc; font-size: 26px; font-weight: 800; margin: 2px 0;">{dist_remain_km:.2f} <span style="font-size: 14px; color: #94a3b8;">km</span></div>
                <div style="color: #2dd4bf; font-size: 11px;">{curr_dist_m:.0f} / {TOTAL_ROAD_LENGTH_M:.0f} m</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Active Alert Banner
        if alert_obj:
            level = alert_obj["level"]
            msg = alert_obj["message"]
            if level == ALERT_LEVEL_URGENT:
                bg_col = "#7f1d1d"
                border_col = "#ef4444"
                icon = "🚨 URGENT HAZARD"
            elif level == ALERT_LEVEL_WARNING:
                bg_col = "#78350f"
                border_col = "#f59e0b"
                icon = "⚠️ DRIVER WARNING"
            else:
                bg_col = "#0c4a6e"
                border_col = "#0284c7"
                icon = "ℹ️ ADVISORY"

            st.markdown(f"""
            <div style="background: {bg_col}; border: 2px solid {border_col}; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-weight: 800; color: white; font-size: 14px; margin-right: 12px;">{icon}</span>
                    <span style="color: #f8fafc; font-size: 16px; font-weight: 600;">{msg}</span>
                </div>
                <div style="color: #cbd5e1; font-size: 12px; font-family: monospace;">
                    {alert_obj.get('distance_ahead_m', 0):.0f}m ahead
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Geofence Warning if outside corridor
        if not geofence_res["is_on_route"]:
            st.warning(f"⚠️ OUTSIDE MONITORED ROUTE corridor ({geofence_res['distance_m']:.1f}m from nearest waypoint). Hazard alerts and health look-ahead are paused outside the 50m road corridor.")

        # PyDeck Interactive Route Map
        st.markdown("#### 🗺️ Continuous 2.4 km Road Corridor Map")

        route_coords = [[wp["longitude"], wp["latitude"]] for wp in waypoints]
        path_data = [{"path": route_coords, "name": "RoadSentinel Corridor"}]

        # Segment Boundary Points for Map
        boundary_pts = []
        for s_idx, (seg_name, (b_start, b_end)) in enumerate(SEGMENT_BOUNDARIES.items()):
            b_lat, b_lon = get_gps_at_distance(b_start, waypoints)
            boundary_pts.append({
                "latitude": b_lat,
                "longitude": b_lon,
                "name": f"{seg_name} ({b_start:.0f}m)",
            })

        vehicle_data = [{
            "latitude": curr_lat,
            "longitude": curr_lon,
            "speed": curr_speed_kmh,
            "segment": curr_seg,
        }]

        hazard_data = []
        for h in hazards_for_day:
            is_nearest = nearest_hazard and (h["hazard_id"] == nearest_hazard["hazard_id"])
            hazard_data.append({
                "latitude": h["latitude"],
                "longitude": h["longitude"],
                "hazard_id": h["hazard_id"],
                "hazard_type": h["hazard_type"],
                "is_nearest": is_nearest,
                "color": [239, 68, 68, 255] if is_nearest else [245, 158, 11, 200],
                "radius": 15 if is_nearest else 10,
            })

        view_state = pdk.ViewState(
            latitude=curr_lat,
            longitude=curr_lon,
            zoom=14.5,
            pitch=35,
        )

        layers = [
            pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_color=[59, 130, 246, 200],
                width_min_pixels=5,
                rounded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=vehicle_data,
                get_position=["longitude", "latitude"],
                get_color=[250, 204, 21, 255],
                get_radius=20,
                radius_min_pixels=8,
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=hazard_data,
                get_position=["longitude", "latitude"],
                get_color="color",
                get_radius="radius",
                radius_min_pixels=6,
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=boundary_pts,
                get_position=["longitude", "latitude"],
                get_color=[148, 163, 184, 180],
                get_radius=8,
                radius_min_pixels=4,
                pickable=True,
            ),
        ]

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{hazard_type}\n{hazard_id}"},
            map_style="mapbox://styles/mapbox/dark-v10",
        )
        st.pydeck_chart(deck, use_container_width=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # 6-Segment Road Strip Visualization
        st.markdown("#### 🛣️ 6-Segment Status Strip (400 m each — 2.4 km total)")
        strip_cols = st.columns(6)

        for idx, seg_name in enumerate(SEGMENT_NAMES):
            b_start, b_end = SEGMENT_BOUNDARIES[seg_name]
            is_active_seg = (curr_seg == seg_name)

            seg_hazards = [h for h in hazards_for_day if h["segment_id"] == seg_name]
            seg_info = get_segment_day_info(seg_name, sim.day, primary_data)

            with strip_cols[idx]:
                if seg_info.get("data_pending", False):
                    border_color = "#64748b"
                    bg_color = "#1e293b"
                    health_display = "PENDING"
                    haz_count_display = "Inspection Pending"
                else:
                    sev = seg_info["current_severity"]
                    score = compute_road_health_score(sev) if sev is not None else 100.0
                    score_col = "#34d399" if score >= 75 else ("#facc15" if score >= 50 else "#f87171")
                    border_color = "#38bdf8" if is_active_seg else score_col
                    bg_color = "#0f233a" if is_active_seg else "#1e293b"
                    health_display = f"{score:.0f} / 100"
                    haz_count_display = f"{len(seg_hazards)} Hazards"

                active_badge = "<span style='color:#38bdf8; font-weight:bold; font-size:10px;'>● ACTIVE</span>" if is_active_seg else ""

                st.markdown(f"""
                <div style="background: {bg_color}; border: 2px solid {border_color}; padding: 10px; border-radius: 6px; text-align: center;">
                    <div style="color: #cbd5e1; font-weight: 800; font-size: 13px;">{seg_name} {active_badge}</div>
                    <div style="color: #94a3b8; font-size: 11px;">{b_start:.0f}–{b_end:.0f}m</div>
                    <div style="font-size: 16px; font-weight: bold; margin: 4px 0; color: {border_color};">{health_display}</div>
                    <div style="color: #94a3b8; font-size: 10px;">{haz_count_display}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Day-over-Day Comparison Expander
        with st.expander("📈 MODEL-OBSERVED DAY-OVER-DAY COMPARISON"):
            if sim.day > 1:
                prev_day = sim.day - 1
                st.caption(f"Comparing Day {sim.day:02d} with Previous Day {prev_day:02d} (Model-Observed Change, not measured physical deterioration)")
                cmp_rows = []
                for s_name in SEGMENT_NAMES:
                    curr_s = get_segment_day_info(s_name, sim.day, primary_data)
                    prev_s = get_segment_day_info(s_name, prev_day, primary_data)

                    if curr_s.get("data_pending", False) or prev_s.get("data_pending", False):
                        change_txt = "DATA PENDING"
                        c_score_txt = "N/A"
                        p_score_txt = "N/A"
                    else:
                        c_score = compute_road_health_score(curr_s["current_severity"])
                        p_score = compute_road_health_score(prev_s["current_severity"])
                        delta = c_score - p_score
                        change_txt = f"{delta:+.1f}"
                        c_score_txt = f"{c_score:.1f}"
                        p_score_txt = f"{p_score:.1f}"

                    cmp_rows.append({
                        "Segment": s_name,
                        f"Day {prev_day:02d} Health": p_score_txt,
                        f"Day {sim.day:02d} Health": c_score_txt,
                        "Model-Observed Change": change_txt,
                    })
                st.dataframe(cmp_rows, hide_index=True, use_container_width=True)
            else:
                st.caption("Day 01 is baseline inspection state; day-over-day delta begins on Day 02.")

        # Local Alert History Log Expander
        with st.expander("📜 SIMULATION-LOCAL ALERT HISTORY"):
            if sim.alert_history:
                st.dataframe(sim.alert_history, hide_index=True, use_container_width=True)
            else:
                st.caption("No alerts triggered in current simulation session yet.")

        # Technical Details Expander
        with st.expander("⚙️ TECHNICAL DETAILS & RESEARCH METADATA"):
            st.markdown(f"""
            - **Simulated GPS Location**: `{curr_lat:.6f}, {curr_lon:.6f}`
            - **Geofence Status**: `{geofence_res['status']}` (Nearest Waypoint Distance: `{geofence_res['distance_m']}m`)
            - **Look-Ahead Window**: `{LOOKAHEAD_DISTANCE_M}m` (Weighted Severity: `{lookahead_info.get('weighted_severity', 'N/A')}`)
            - **Active Hazards Count**: `{len(hazards_for_day)}` hazards registered for Day {sim.day:02d}
            - **Data Source Files**: `integration/primary_goals/ROADSENTINEL_PRIMARY_RESULTS.csv`, `decision_engine/ROAD_HEALTH_DECISIONS.csv`
            """)

        st.markdown("---")
        st.caption(DISCLAIMER_TEXT)

    # -------------------------------------------------------------
    # TAB 2: CITIZEN REPORT ROAD DAMAGE + AI VERIFICATION
    # -------------------------------------------------------------
    with tab_citizen:
        render_citizen_reporting_tab(ai_pipeline, curr_lat, curr_lon)

    # -------------------------------------------------------------
    # TAB 3: SUBMITTED CITIZEN REPORTS LOG
    # -------------------------------------------------------------
    with tab_archive:
        render_citizen_reports_history_tab()

    # Rerun simulation tick if running
    if sim.is_running and mode == "Simulated Drive":
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()
