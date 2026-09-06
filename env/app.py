#!/usr/bin/env python3
"""
env/app.py
----------
RoadSentinel - Phase 4: Dynamic Web Parameter Dashboard & Flask Blueprint Extension.
Hosts the real-time aerial survey dashboard, MJPEG live video feed, telemetry streams,
and interactive simulation parameter controls.

Registered Blueprint:
  rs_dashboard (prefix: /rs/)
Routes:
  GET  /rs/                   - Live dashboard (renders templates/rs/dashboard.html or SSE stream)
  GET  /rs/control            - Parameter control form (templates/rs/control.html)
  POST /rs/control            - Pydantic-validated simulation configuration override
  GET  /rs/stream             - MJPEG stream of preview frames (up to 10 fps)
  GET  /rs/annotations/latest - Latest ground-truth annotation JSON
  GET  /rs/status             - CARLA connectivity, weather preset, drone telemetry, disk usage guard
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any

from flask import Flask, Blueprint, request, Response, jsonify, render_template, redirect, url_for
from pydantic import BaseModel, Field, ValidationError

# Configure workspace paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Paths
OUTPUT_DIR = WORKSPACE_ROOT / "env" / "output" / "images"
FRAMES_DIR = OUTPUT_DIR / "frames"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"
PREVIEW_DIR = OUTPUT_DIR / "preview"
STATIC_DIR = WORKSPACE_ROOT / "static"
TEMPLATES_DIR = WORKSPACE_ROOT / "templates"
PRESETS_FILE = WORKSPACE_ROOT / "env" / "config" / "rs_weather_presets.json"

# Import Phase 3 & Phase 4 modules
try:
    from env.scripts.rs_phase3_weather_lighting import (
        DynamicWeatherManager,
        load_presets,
        DEFAULT_PRESETS,
    )
except ImportError:
    DynamicWeatherManager = None
    load_presets = lambda: {}
    DEFAULT_PRESETS = {}

try:
    from env.scripts.rs_phase4_drone_sensor import (
        DroneSensorPipeline,
        get_sensor_pipeline,
        check_disk_space_gb,
        check_disk_space_guard,
    )
except ImportError:
    DroneSensorPipeline = None
    get_sensor_pipeline = lambda: None
    check_disk_space_gb = lambda p: 999.0
    check_disk_space_guard = lambda p: (True, "OK")


# ==============================================================================
# Pydantic Configuration Model
# ==============================================================================

VALID_PRESETS = [
    "ClearNoon",
    "OvercastDay",
    "HeavyRain",
    "LightDrizzle",
    "DuskGoldenHour",
    "NightClear",
    "NightFoggy",
    "NightRain",
]


class RS_SimConfig(BaseModel):
    weather_preset: Optional[str] = Field(
        default=None,
        description="Dynamic weather preset from Phase 3 catalogue"
    )
    defect_density: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=10.0,
        description="Procedural defect density per 100 m² (0.5 to 10.0)"
    )
    defect_types: Optional[List[str]] = Field(
        default=None,
        description="Active defect catalogue types filter"
    )
    water_ior: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=2.0,
        description="Water surface refractive index (1.0 to 2.0)"
    )
    altitude_m: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=80.0,
        description="Camera survey altitude AGL in meters (10 to 80)"
    )
    flight_mode: Optional[Literal["hover", "transect", "survey_grid"]] = Field(
        default=None,
        description="Drone autonomous trajectory controller mode"
    )
    is_capturing: Optional[bool] = Field(
        default=None,
        description="Frame capture enable/disable toggle"
    )
    transition_time_s: Optional[float] = Field(
        default=10.0,
        ge=0.1,
        le=120.0,
        description="Weather interpolation duration in seconds"
    )


# ==============================================================================
# Global Simulation State & Managers
# ==============================================================================

_weather_manager: Optional[Any] = None
_active_water_ior: float = 1.333
_active_defect_density: float = 3.0
_active_defect_types: List[str] = [
    "PotholeWet",
    "PotholeDry",
    "LongitudinalCrack",
    "AlligatorCrack",
]


def get_weather_mgr():
    global _weather_manager
    if _weather_manager is None and DynamicWeatherManager is not None:
        try:
            _weather_manager = DynamicWeatherManager()
        except Exception as e:
            print(f"[RoadSentinel Server] Weather manager init error: {e}")
    return _weather_manager


# ==============================================================================
# Flask Blueprint Definition
# ==============================================================================

rs_dashboard = Blueprint(
    "rs_dashboard",
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)


@rs_dashboard.route("/", methods=["GET"])
def index():
    """
    Live dashboard view:
    - If Accept: text/event-stream or ?sse=1, streams server-sent telemetry events.
    - Otherwise, renders live dashboard HTML template.
    """
    pipeline = get_sensor_pipeline()
    wm = get_weather_mgr()
    active_preset = wm.active_preset.get("name", "ClearNoon") if wm else "ClearNoon"

    # Handle Server-Sent Events (SSE)
    if "text/event-stream" in request.headers.get("Accept", "") or request.args.get("sse") == "1":
        def sse_event_stream():
            while True:
                pose = (
                    pipeline.flight_ctrl.current_x,
                    pipeline.flight_ctrl.current_y,
                    pipeline.flight_ctrl.altitude_m,
                    pipeline.flight_ctrl.current_yaw_deg,
                ) if pipeline else (0.0, 0.0, 25.0, 0.0)
                
                # Fetch latest annotation if available
                latest_defects = 0
                latest_frame_id = ""
                ann_path = pipeline.last_annotation_path if pipeline else None
                if ann_path and Path(ann_path).exists():
                    try:
                        with open(ann_path, "r", encoding="utf-8") as f:
                            ann = json.load(f)
                            latest_defects = ann.get("total_defects_in_frame", 0)
                            latest_frame_id = ann.get("frame_id", "")
                    except Exception:
                        pass

                payload = {
                    "active_preset": active_preset,
                    "defect_count": latest_defects,
                    "flight_mode": pipeline.flight_ctrl.flight_mode if pipeline else "hover",
                    "drone_position": {
                        "x": round(pose[0], 2),
                        "y": round(pose[1], 2),
                        "z": round(pose[2], 2),
                        "yaw": round(pose[3], 1),
                    },
                    "frame_id": latest_frame_id,
                    "total_frames": pipeline.total_frames_captured if pipeline else 0,
                    "is_capturing": pipeline.is_capturing if pipeline else False,
                    "timestamp": int(time.time() * 1000),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(1.0)

        return Response(sse_event_stream(), mimetype="text/event-stream")

    # Render standard dashboard page
    return render_template(
        "rs/dashboard.html",
        active_preset=active_preset,
        flight_mode=pipeline.flight_ctrl.flight_mode if pipeline else "hover",
        altitude_m=pipeline.flight_ctrl.altitude_m if pipeline else 25.0,
        frame_count=pipeline.total_frames_captured if pipeline else 0,
    )


@rs_dashboard.route("/control", methods=["GET"])
def control_page():
    """Renders the interactive parameter control form."""
    pipeline = get_sensor_pipeline()
    wm = get_weather_mgr()
    active_preset = wm.active_preset.get("name", "ClearNoon") if wm else "ClearNoon"
    presets_data = wm.presets if wm else {}

    return render_template(
        "rs/control.html",
        active_preset=active_preset,
        presets=presets_data,
        flight_mode=pipeline.flight_ctrl.flight_mode if pipeline else "hover",
        altitude_m=pipeline.flight_ctrl.altitude_m if pipeline else 25.0,
        defect_density=_active_defect_density,
        water_ior=_active_water_ior,
        is_capturing=pipeline.is_capturing if pipeline else False,
    )


@rs_dashboard.route("/control", methods=["POST"])
def apply_control():
    """
    Accepts JSON body, validates via Pydantic RS_SimConfig, and applies
    parameters to the running simulation and sensor pipeline.
    """
    global _active_water_ior, _active_defect_density, _active_defect_types

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"status": "error", "message": "Invalid JSON body"}), 400

    try:
        config = RS_SimConfig.model_validate(body)
    except ValidationError as err:
        return jsonify({"status": "error", "errors": err.errors()}), 400

    pipeline = get_sensor_pipeline()
    wm = get_weather_mgr()
    updated_fields = {}

    # 1. Weather preset override
    if config.weather_preset is not None:
        if config.weather_preset in VALID_PRESETS:
            if wm:
                duration = config.transition_time_s if config.transition_time_s is not None else 10.0
                wm.set_preset(config.weather_preset, transition_time_s=duration)
            updated_fields["weather_preset"] = config.weather_preset
        else:
            return jsonify({
                "status": "error",
                "message": f"Invalid weather preset: '{config.weather_preset}'. Must be one of {VALID_PRESETS}"
            }), 400

    # 2. Drone Altitude
    if config.altitude_m is not None and pipeline:
        pipeline.flight_ctrl.altitude_m = config.altitude_m
        pipeline.cam_cfg.altitude_m = config.altitude_m
        updated_fields["altitude_m"] = config.altitude_m

    # 3. Flight Mode
    if config.flight_mode is not None and pipeline:
        pipeline.flight_ctrl.flight_mode = config.flight_mode
        updated_fields["flight_mode"] = config.flight_mode

    # 4. Water IOR
    if config.water_ior is not None:
        _active_water_ior = config.water_ior
        updated_fields["water_ior"] = config.water_ior
        # If Unreal Engine is available in active process, update material parameter
        try:
            import unreal
            all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
            for actor in all_actors:
                if actor and actor.actor_has_tag(unreal.Name("RS_Defect")):
                    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
                    if comp:
                        for idx in range(comp.get_num_materials()):
                            mid = comp.get_material(idx)
                            if isinstance(mid, unreal.MaterialInstanceDynamic):
                                mid.set_scalar_parameter_value("RS_WaterIOR", _active_water_ior)
        except Exception:
            pass

    # 5. Defect Density & Type Filter (triggers procedural re-scatter on submit)
    rescat_needed = False
    if config.defect_density is not None and config.defect_density != _active_defect_density:
        _active_defect_density = config.defect_density
        updated_fields["defect_density"] = config.defect_density
        rescat_needed = True

    if config.defect_types is not None:
        _active_defect_types = config.defect_types
        updated_fields["defect_types"] = config.defect_types
        rescat_needed = True

    if rescat_needed:
        # Trigger procedural re-scatter logic
        try:
            from RoadSentinelSim.Content.Python.rs_phase2_surface_defects import (
                DefectConfig,
                scatter_defects,
            )
            type_weights = {
                "PotholeWet": 0.35 if "PotholeWet" in _active_defect_types else 0.0,
                "PotholeDry": 0.25 if "PotholeDry" in _active_defect_types else 0.0,
                "LongitudinalCrack": 0.20 if "LongitudinalCrack" in _active_defect_types else 0.0,
                "AlligatorCrack": 0.20 if "AlligatorCrack" in _active_defect_types else 0.0,
            }
            tot = sum(type_weights.values()) or 1.0
            norm_weights = {k: v / tot for k, v in type_weights.items()}

            dcfg = DefectConfig(
                defect_density=_active_defect_density,
                defect_weights=norm_weights,
            )
            scatter_defects(dcfg)
            if pipeline:
                pipeline.load_manifest()
            updated_fields["rescattered"] = True
        except Exception as e:
            print(f"[RoadSentinel Server] Re-scatter execution warning: {e}")
            updated_fields["rescattered"] = False

    # 6. Capture Toggle
    if config.is_capturing is not None and pipeline:
        pipeline.is_capturing = config.is_capturing
        updated_fields["is_capturing"] = config.is_capturing

    return jsonify({
        "status": "success",
        "message": "Simulation configuration updated successfully",
        "updated": updated_fields,
        "config": config.model_dump(),
    })


@rs_dashboard.route("/stream", methods=["GET"])
def mjpeg_stream():
    """
    MJPEG stream of preview frames at up to 10 fps.
    Yields multipart boundary chunks with JPEG payloads.
    """
    pipeline = get_sensor_pipeline()
    wm = get_weather_mgr()

    def generate_frames():
        while True:
            t0 = time.time()
            active_preset = wm.active_preset.get("name", "ClearNoon") if wm else "ClearNoon"

            # Check if capturing or generate frame on demand
            frame_bytes = None
            if pipeline:
                if pipeline.is_capturing:
                    pipeline.flight_ctrl.update(0.1)
                    res = pipeline.capture_frame(weather_preset=active_preset)
                    if res and Path(res["preview_path"]).exists():
                        with open(res["preview_path"], "rb") as f:
                            frame_bytes = f.read()

                # If not actively capturing, serve latest preview if available
                if frame_bytes is None and pipeline.last_preview_path and Path(pipeline.last_preview_path).exists():
                    try:
                        with open(pipeline.last_preview_path, "rb") as f:
                            frame_bytes = f.read()
                    except Exception:
                        frame_bytes = None

            # Fallback: locate latest file in PREVIEW_DIR
            if frame_bytes is None and PREVIEW_DIR.exists():
                preview_files = sorted(PREVIEW_DIR.glob("*_preview.jpg"), key=os.path.getmtime, reverse=True)
                if preview_files:
                    try:
                        with open(preview_files[0], "rb") as f:
                            frame_bytes = f.read()
                    except Exception:
                        frame_bytes = None

            # Secondary fallback: static preview fallback
            if frame_bytes is None:
                fallback_path = STATIC_DIR / "rs" / "preview_fallback.jpg"
                if fallback_path.exists():
                    with open(fallback_path, "rb") as f:
                        frame_bytes = f.read()

            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

            # Cap frame rate at 10 fps (100 ms per frame)
            elapsed = time.time() - t0
            sleep_time = max(0.01, 0.10 - elapsed)
            time.sleep(sleep_time)

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@rs_dashboard.route("/annotations/latest", methods=["GET"])
def latest_annotation():
    """Returns the most recent ground-truth annotation JSON."""
    pipeline = get_sensor_pipeline()

    # 1. Try pipeline reference
    if pipeline and pipeline.last_annotation_path and Path(pipeline.last_annotation_path).exists():
        try:
            with open(pipeline.last_annotation_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            pass

    # 2. Try searching directory
    if ANNOTATIONS_DIR.exists():
        ann_files = sorted(ANNOTATIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
        if ann_files:
            try:
                with open(ann_files[0], "r", encoding="utf-8") as f:
                    return jsonify(json.load(f))
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

    # 3. Fallback: generate a single frame to produce an initial annotation
    if pipeline:
        wm = get_weather_mgr()
        active_preset = wm.active_preset.get("name", "ClearNoon") if wm else "ClearNoon"
        res = pipeline.capture_frame(weather_preset=active_preset)
        if res and Path(res["annotation_path"]).exists():
            with open(res["annotation_path"], "r", encoding="utf-8") as f:
                return jsonify(json.load(f))

    return jsonify({"status": "empty", "message": "No annotations generated yet", "defects": []}), 200


@rs_dashboard.route("/status", methods=["GET"])
def simulation_status():
    """
    Returns live simulation telemetry, CARLA server reachability, active weather preset,
    drone position, frame count, and disk usage guard (with warning if < 2 GB).
    """
    pipeline = get_sensor_pipeline()
    wm = get_weather_mgr()

    # Check CARLA reachability
    carla_reachable = False
    if wm and hasattr(wm, "carla_controller"):
        carla_reachable = wm.carla_controller.is_connected
        if not carla_reachable:
            # Quick check if client can ping CARLA
            try:
                carla_reachable = wm.carla_controller.connect()
            except Exception:
                carla_reachable = False

    # Drone pose
    drone_pose = {
        "x": round(pipeline.flight_ctrl.current_x, 2) if pipeline else 0.0,
        "y": round(pipeline.flight_ctrl.current_y, 2) if pipeline else 0.0,
        "z": round(pipeline.flight_ctrl.altitude_m, 2) if pipeline else 25.0,
        "yaw": round(pipeline.flight_ctrl.current_yaw_deg, 2) if pipeline else 0.0,
    }

    # Disk usage guard (check output images dir)
    try:
        usage = shutil.disk_usage(str(OUTPUT_DIR if OUTPUT_DIR.exists() else WORKSPACE_ROOT))
        total_gb = round(usage.total / (1024.0 ** 3), 2)
        used_gb = round(usage.used / (1024.0 ** 3), 2)
        free_gb = round(usage.free / (1024.0 ** 3), 2)
    except Exception:
        total_gb, used_gb, free_gb = 100.0, 50.0, 50.0

    disk_warning = free_gb < 2.0
    disk_warning_msg = (
        f"CRITICAL WARNING: Available disk space {free_gb:.2f} GB is below safety threshold (2.0 GB)!"
        if disk_warning else None
    )

    status_data = {
        "status": "operational",
        "carla_server_reachable": carla_reachable,
        "active_preset": wm.active_preset.get("name", "ClearNoon") if wm else "ClearNoon",
        "flight_mode": pipeline.flight_ctrl.flight_mode if pipeline else "hover",
        "drone_position": drone_pose,
        "frame_count": pipeline.total_frames_captured if pipeline else 0,
        "is_capturing": pipeline.is_capturing if pipeline else False,
        "water_ior": _active_water_ior,
        "defect_density": _active_defect_density,
        "disk_usage": {
            "output_directory": str(OUTPUT_DIR),
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "warning": disk_warning,
            "warning_msg": disk_warning_msg,
        },
        "timestamp_ms": int(time.time() * 1000),
    }

    return jsonify(status_data)


# ==============================================================================
# Flask Application Factory & Server Initialization
# ==============================================================================

def create_app() -> Flask:
    """Creates and configures the Flask application, registering the rs_dashboard Blueprint."""
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )

    # Register the required RoadSentinel dashboard blueprint under prefix /rs/
    app.register_blueprint(rs_dashboard, url_prefix="/rs")

    # Root route: redirect to /rs/
    @app.route("/")
    def root_redirect():
        return redirect("/rs/")

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n========================================================")
    print(f" RoadSentinel Aerial Rig - Web Dashboard & Control Panel")
    print(f" Live Stream & Telemetry: http://localhost:{port}/rs/")
    print(f" Mission Controls:        http://localhost:{port}/rs/control")
    print(f" Status API:              http://localhost:{port}/rs/status")
    print(f"========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
