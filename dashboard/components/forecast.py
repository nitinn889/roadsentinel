"""Future Condition Forecast Component for RoadSentinel Dashboard."""

from __future__ import annotations

import numpy as np
import streamlit as st
from data_loader import load_xgboost_forecast_model

PRESET_VALUES = {
    "NORMAL": {
        "rainfall_level": 2.7376,
        "traffic_level": 346.0,
        "temperature": 6.3965,
        "water_exposure": 0.5663,
        "description": "Historical median environmental and traffic loading conditions."
    },
    "HEAVY_RAIN": {
        "rainfall_level": 3.7690,
        "traffic_level": 346.0,
        "temperature": 6.3965,
        "water_exposure": 0.5663,
        "description": "Upper 90th-percentile precipitation with normal traffic."
    },
    "HEAVY_TRAFFIC": {
        "rainfall_level": 2.7376,
        "traffic_level": 784.0,
        "temperature": 6.3965,
        "water_exposure": 0.5663,
        "description": "Upper 90th-percentile heavy vehicle axle repetitions."
    },
    "HIGH_HEAT": {
        "rainfall_level": 2.7376,
        "traffic_level": 346.0,
        "temperature": 21.4915,
        "water_exposure": 0.5663,
        "description": "Elevated pavement surface temperature accelerating thermal binder fatigue."
    },
    "WET_EXPOSURE": {
        "rainfall_level": 2.7376,
        "traffic_level": 346.0,
        "temperature": 6.3965,
        "water_exposure": 0.6876,
        "description": "High moisture retention and subbase saturation exposure."
    }
}


def render_future_forecast():
    st.markdown("## Future Condition Forecast (XGBoost Model V2)")
    st.markdown(
        "Generate scenario-conditioned road deterioration projections using the frozen "
        "**XGBoost Model V2** trained on LTPP multi-year pavement data with monotone deterioration constraints."
    )

    model, config = load_xgboost_forecast_model()
    if model is None or config is None:
        st.error("XGBoost Model V2 or configuration file not found in `xgboost/model/`. Please check directory structure.")
        return

    # 1. Input Section
    st.markdown("### 1. Pavement Baseline & Scenario Selection")

    col_base, col_preset = st.columns([1, 1])

    # Initial severity from previous page or manual slider
    default_sev = float(st.session_state.get("selected_current_severity", 0.25))
    selected_img_id = st.session_state.get("selected_image_id", "Manual Input")

    with col_base:
        st.markdown(f"**Selected Road Origin**: `{selected_img_id}`")
        current_sev = st.slider(
            "Current Measured Severity Index ($S_0$)",
            min_value=0.0,
            max_value=0.90,
            value=min(0.90, max(0.0, default_sev)),
            step=0.01,
            help="Starting surface damage severity index estimated from perception output [0.0 - 0.90]."
        )

    with col_preset:
        preset_choice = st.selectbox(
            "Scenario Preset",
            ["CUSTOM"] + list(PRESET_VALUES.keys()),
            index=1,
            help="Choose an observational quantile preset or select CUSTOM to set individual variables."
        )
        if preset_choice != "CUSTOM":
            st.caption(f"**Preset Info**: {PRESET_VALUES[preset_choice]['description']}")

    # Scenario parameters
    st.markdown("### 2. Environmental & Operational Stressors")

    if preset_choice != "CUSTOM":
        p_vals = PRESET_VALUES[preset_choice]
        rain_val = p_vals["rainfall_level"]
        traffic_val = p_vals["traffic_level"]
        temp_val = p_vals["temperature"]
        water_val = p_vals["water_exposure"]
    else:
        rain_val = 2.74
        traffic_val = 346.0
        temp_val = 6.4
        water_val = 0.57

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        rainfall = st.slider("Rainfall Level (mm/day)", 0.72, 5.14, float(rain_val), 0.05, disabled=(preset_choice != "CUSTOM"))
    with c2:
        traffic = st.slider("Traffic Level (AADT)", 58.0, 1257.0, float(traffic_val), 10.0, disabled=(preset_choice != "CUSTOM"))
    with c3:
        temperature = st.slider("Temperature (°C)", -4.4, 24.7, float(temp_val), 0.5, disabled=(preset_choice != "CUSTOM"))
    with c4:
        water_exp = st.slider("Water Exposure Ratio", 0.26, 0.78, float(water_val), 0.02, disabled=(preset_choice != "CUSTOM"))
    with c5:
        days_ahead = st.select_slider("Forecast Horizon", options=[30, 60, 90], value=60, help="Days into the future (trained interface horizons)")

    # Execute Inference
    feature_order = config.get("feature_order", [
        "current_severity", "rainfall_level", "traffic_level", "temperature", "water_exposure", "days_ahead"
    ])
    feature_dict = {
        "current_severity": float(current_sev),
        "rainfall_level": float(rainfall),
        "traffic_level": float(traffic),
        "temperature": float(temperature),
        "water_exposure": float(water_exp),
        "days_ahead": float(days_ahead)
    }

    feature_vector = np.asarray([[feature_dict[name] for name in feature_order]], dtype=float)
    raw_pred = float(model.predict(feature_vector)[0])
    future_sev = min(1.0, max(0.0, raw_pred))
    delta_sev = future_sev - current_sev

    # 3. Forecast Output Card
    st.markdown("---")
    st.markdown("### 3. Model-Based Forecast Output")

    st.markdown(f"""
    <div style="background: rgba(18, 24, 38, 0.85); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 14px; padding: 22px 28px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <span class="status-pill complete">STATUS: MODEL-BASED FORECAST</span>
            <span style="color: #94a3b8; font-size: 0.9rem;">Scenario: <strong>{preset_choice}</strong> | Horizon: <strong>+{days_ahead} Days</strong></span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center;">
            <div>
                <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Baseline Severity</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #f8fafc; font-family: 'Outfit', sans-serif;">{current_sev:.4f}</div>
            </div>
            <div>
                <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Projected Future Severity</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: {'#f43f5e' if delta_sev > 0.1 else '#fbbf24'}; font-family: 'Outfit', sans-serif;">{future_sev:.4f}</div>
            </div>
            <div>
                <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Estimated Deterioration (Δ)</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: {'#f43f5e' if delta_sev > 0 else '#34d399'}; font-family: 'Outfit', sans-serif;">{delta_sev:+.4f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Progression Bar visualization
    st.progress(float(future_sev), text=f"Estimated Pavement Degradation: {future_sev*100:.1f}% of Critical Limit")

    # Mandatory Disclaimer
    st.markdown("""
    <div class="callout-box warn">
        <strong>Mandatory Scientific Disclaimer:</strong><br>
        <em>"The forecast is scenario-conditioned and derived from observational training data; it should not be interpreted as a deterministic physical deterioration prediction."</em>
        <br>
        Pavement degradation is subject to micro-climatic events, subsurface drainage variations, unobserved axle weights, and maintenance interventions not captured in 2D survey photography.
    </div>
    """, unsafe_allow_html=True)

    # Technical Details
    with st.expander("Technical Model Parameters & Monotone Constraints"):
        st.json({
            "model_id": config.get("model_id"),
            "training_samples": config.get("training_row_count", 89),
            "feature_order": feature_order,
            "inputs_evaluated": feature_dict,
            "monotone_constraints": "(1, 1, 1, 1, 1, 1) [Enforces non-negative deterioration per stressor]",
            "target_scale": "LTPP IRI mapped to [0, 1] bounds",
        })
