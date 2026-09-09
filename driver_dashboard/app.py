"""RoadSentinel Driver User Dashboard Application.

Comprehensive interactive user/driver demonstration platform connecting:
- 2.4 km continuous simulated road corridor (SEG_001 through SEG_006)
- Haversine distance, route interpolation, and 50m road geofence
- Real-time car driving simulation (0-100 km/h) with manual GPS test mode
- Model-derived 400m look-ahead road health score
- Real-time hazard detection, ETA to hazard, and non-directive driver alerts
- PyDeck map visualization and 6-segment road strip status
- Offline-ready, zero Unreal/CARLA dependency, and 100% repository data consumption

Launch command:
    streamlit run driver_dashboard/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pydeck as pdk
import streamlit as st

# Ensure driver_dashboard directory is in sys.path
DRIVER_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DRIVER_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DASHBOARD_DIR))

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


def init_session_state():
    """Initialize persistent Streamlit session state for driver simulation."""
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


def render_header():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a, #1e293b); padding: 20px 24px; border-radius: 12px; border-left: 6px solid #38bdf8; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="color: #f8fafc; margin: 0; font-size: 26px; font-weight: 800; font-family: 'Outfit', sans-serif;">
                    🛣️ RoadSentinel Driver Dashboard
                </h1>
                <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 14px;">
                    Real-time driver road-health intelligence, hazard warning, and 400m look-ahead simulation (2.4 km corridor).
                </p>
            </div>
            <div style="background: #0284c7; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px;">
                PROTOTYPE USER MODE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="RoadSentinel Driver Dashboard",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    sim = st.session_state.sim_state

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
                "Road Distance (meters)",
                min_value=0.0,
                max_value=TOTAL_ROAD_LENGTH_M,
                value=float(sim.distance_m),
                step=10.0,
            )
            if not sim.is_running and abs(dist_val - sim.distance_m) > 1e-3:
                sim.distance_m = dist_val

        else:
            st.markdown("#### Manual GPS Test Inputs")
            st.caption("Enter arbitrary lat/lon coordinates to test geofence and hazard alerts.")
            manual_lat = st.number_input("Latitude", value=st.session_state.manual_lat, format="%.6f")
            manual_lon = st.number_input("Longitude", value=st.session_state.manual_lon, format="%.6f")
            manual_speed = st.number_input("Speed (km/h)", value=st.session_state.manual_speed, min_value=0.0, max_value=120.0)
            
            st.session_state.manual_lat = manual_lat
            st.session_state.manual_lon = manual_lon
            st.session_state.manual_speed = manual_speed

        st.markdown("---")
        st.caption("RoadSentinel v2.0 Driver Experience\n2.4 km Monitored Corridor (SEG_001–006)")

    # Main UI Header
    render_header()

    # Determine Current Vehicle Coordinates & Geofence
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
            h_type = nearest_hazard["hazard_type"]
            h_dist = nearest_hazard["distance_ahead_m"]
            speed_mps = curr_speed_kmh / 3.6
            eta_sec = round(h_dist / speed_mps, 1) if speed_mps > 0 else "N/A"
            eta_str = f"~{eta_sec}s" if isinstance(eta_sec, (int, float)) else "STOPPED"
            st.markdown(f"""
            <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid #ef4444;">
                <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">NEAREST HAZARD AHEAD</div>
                <div style="color: #f87171; font-size: 20px; font-weight: 800; margin: 2px 0;">{h_type} — {h_dist:.0f}m</div>
                <div style="color: #fca5a5; font-size: 11px;">ETA: {eta_str}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid #10b981;">
                <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">NEAREST HAZARD AHEAD</div>
                <div style="color: #34d399; font-size: 20px; font-weight: 800; margin: 2px 0;">ROAD CLEAR</div>
                <div style="color: #94a3b8; font-size: 11px;">No Hazards Ahead</div>
            </div>
            """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid #8b5cf6;">
            <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">CURRENT CORRIDOR SEGMENT</div>
            <div style="color: #c084fc; font-size: 24px; font-weight: 800; margin: 2px 0;">{curr_seg}</div>
            <div style="color: #94a3b8; font-size: 11px;">{curr_dist_m:.0f} m / {TOTAL_ROAD_LENGTH_M:.0f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        dist_rem_m = max(0.0, TOTAL_ROAD_LENGTH_M - curr_dist_m)
        st.markdown(f"""
        <div style="background: #1e293b; padding: 14px; border-radius: 8px; text-align: center; border-bottom: 4px solid #0284c7;">
            <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase;">DISTANCE REMAINING</div>
            <div style="color: #38bdf8; font-size: 24px; font-weight: 800; margin: 2px 0;">{dist_rem_m / 1000.0:.2f} <span style="font-size: 14px; color: #94a3b8;">km</span></div>
            <div style="color: #94a3b8; font-size: 11px;">Total Corridor: 2.4 km</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Active Driver Alert Banner
    if alert_obj:
        level = alert_obj["level"]
        bg_color = "#450a0a" if level == ALERT_LEVEL_URGENT else ("#422006" if level == ALERT_LEVEL_WARNING else "#0f172a")
        border_color = "#ef4444" if level == ALERT_LEVEL_URGENT else ("#f59e0b" if level == ALERT_LEVEL_WARNING else "#3b82f6")
        text_color = "#f87171" if level == ALERT_LEVEL_URGENT else ("#fbbf24" if level == ALERT_LEVEL_WARNING else "#60a5fa")

        st.markdown(f"""
        <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 10px; padding: 18px 24px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 12px; font-weight: bold; color: {text_color}; text-transform: uppercase; letter-spacing: 0.05em;">
                    🚨 ACTIVE DRIVER ALERT ({level})
                </div>
                <div style="font-size: 18px; font-weight: 800; color: #f8fafc; margin-top: 4px;">
                    {alert_obj['message']}
                </div>
            </div>
            <div style="font-size: 13px; color: #cbd5e1; background: rgba(255,255,255,0.08); padding: 6px 12px; border-radius: 6px;">
                Hazard ID: <code>{alert_obj['hazard_id']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif not geofence_res["is_on_route"]:
        st.markdown("""
        <div style="background: #1e1b4b; border: 2px solid #6366f1; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; color: #c7d2fe;">
            <b>📍 OUTSIDE MONITORED ROADSENTINEL ROUTE:</b> Vehicle GPS coordinate is beyond 50m geofence radius. Hazard warning alerts and look-ahead calculations are safely suspended.
        </div>
        """, unsafe_allow_html=True)

    # PyDeck Route & Hazard Map
    st.markdown("### 🗺️ Live 2.4 km Monitored Corridor Map")
    
    # Format PyDeck Layers
    route_path_coords = [[wp["longitude"], wp["latitude"]] for wp in waypoints]
    
    # Layer 1: Route Line
    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": route_path_coords, "name": "RoadSentinel 2.4km Route"}],
        get_path="path",
        get_color=[56, 189, 248, 200],
        width_scale=5,
        width_min_pixels=4,
    )

    # Layer 2: Moving Vehicle Point
    vehicle_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"position": [curr_lon, curr_lat], "name": "Vehicle"}],
        get_position="position",
        get_color=[239, 68, 68, 255] if alert_obj else [16, 185, 129, 255],
        get_radius=18,
        radius_min_pixels=8,
        pickable=True,
    )

    # Layer 3: Active Hazards
    hazard_data = [
        {
            "position": [h["longitude"], h["latitude"]],
            "name": f"{h['hazard_type']} ({h['segment_id']})",
            "type": h["hazard_type"],
            "severity": h["severity"],
        }
        for h in hazards_for_day
    ]

    hazard_layer = pdk.Layer(
        "ScatterplotLayer",
        data=hazard_data,
        get_position="position",
        get_color=[245, 158, 11, 220],
        get_radius=14,
        radius_min_pixels=6,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=curr_lat,
        longitude=curr_lon,
        zoom=14.5,
        pitch=30,
    )

    deck = pdk.Deck(
        layers=[path_layer, hazard_layer, vehicle_layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}"},
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.markdown("---")

    # Road Strip Component
    st.markdown("### 🛣️ Continuous 2.4 km Road Strip (6 Segments)")
    strip_cols = st.columns(6)

    for i, seg_id in enumerate(SEGMENT_NAMES):
        seg_start, seg_end = SEGMENT_BOUNDARIES[seg_id]
        seg_info = get_segment_day_info(seg_id, sim.day)
        is_active_seg = (curr_seg == seg_id)
        
        # Determine strip block color
        if not seg_info["has_data"]:
            bg = "#1e293b"
            border = "#64748b"
            h_score_str = "PENDING"
            def_cnt_str = "—"
        else:
            sev = seg_info["current_severity"]
            h_score = compute_road_health_score(sev)
            h_score_str = f"{h_score:.0f}/100"
            def_cnt_str = f"{seg_info['defect_count']} hazards"
            border = "#10b981" if h_score >= 75 else ("#f59e0b" if h_score >= 50 else "#ef4444")
            bg = "#0f172a" if not is_active_seg else "#1e293b"

        active_border = f"border: 3px solid #38bdf8;" if is_active_seg else f"border: 1px solid {border};"

        with strip_cols[i]:
            st.markdown(f"""
            <div style="background: {bg}; {active_border} border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-weight: 800; font-size: 13px; color: {'#38bdf8' if is_active_seg else '#f8fafc'};">{seg_id}</div>
                <div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">{int(seg_start)}–{int(seg_end)} m</div>
                <div style="font-size: 18px; font-weight: bold; color: #f8fafc; margin: 6px 0;">{h_score_str}</div>
                <div style="font-size: 10px; color: #94a3b8;">{def_cnt_str}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Multi-Day Comparison Expander
    with st.expander("📊 Compare Road Health With Previous Inspection Day"):
        if sim.day > 1:
            prev_day = sim.day - 1
            prev_data = primary_data.get(prev_day, {})
            curr_data = primary_data.get(sim.day, {})
            
            cmp_rows = []
            for s_id in ["SEG_001", "SEG_002", "SEG_003", "SEG_004"]:
                c_info = curr_data.get(s_id, {})
                p_info = prev_data.get(s_id, {})
                if c_info.get("has_data") and p_info.get("has_data"):
                    c_sev = c_info["current_severity"]
                    p_sev = p_info["current_severity"]
                    c_health = compute_road_health_score(c_sev)
                    p_health = compute_road_health_score(p_sev)
                    delta_health = round(c_health - p_health, 1)
                    cmp_rows.append({
                        "Segment": s_id,
                        f"Day {prev_day:02d} Health": f"{p_health:.0f} / 100",
                        f"Day {sim.day:02d} Health": f"{c_health:.0f} / 100",
                        "MODEL-OBSERVED CHANGE": f"{delta_health:+.1f} pts",
                    })
            if cmp_rows:
                st.dataframe(cmp_rows, hide_index=True, use_container_width=True)
                st.caption("Note: MODEL-OBSERVED CHANGE quantifies perception output deltas across inspections, not measured physical asphalt deterioration.")
        else:
            st.info("Day 01 is the initial baseline inspection. Select Day 02 to Day 10 to view day-over-day changes.")

    # Alert History Log Expander
    with st.expander("📋 Simulation Alert History Log"):
        if sim.alert_history:
            st.dataframe(sim.alert_history, hide_index=True, use_container_width=True)
        else:
            st.caption("No alerts triggered in current simulation session yet.")

    # Technical Details Expander (Collapsible)
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

    # Rerun simulation tick if running
    if sim.is_running and mode == "Simulated Drive":
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()
