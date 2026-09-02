from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import load_json, load_rgb, save_json, utc_iso
from inference.run_inference import infer, load_pipeline
from inference.spatial_index import DefectSpatialIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("roadsentinel_server")

app = FastAPI(
    title="RoadSentinel Government Dashboard & Inference API",
    description="Real-time aerial road inspection, defect analytics, Town04 custom geofencing, and automated VLM maintenance work orders.",
    version="2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Search paths for demo analytics results
DEMO_OUTPUT_DIRS = [
    ROOT / "output" / "analytics_demo",
    ROOT.parent / "output",
    ROOT / "outputs",
    ROOT / "output",
]

_CACHED_PIPELINE = None
_SPATIAL_INDEX: Optional[DefectSpatialIndex] = None


def get_latest_result_data() -> Dict[str, Any]:
    """Locate and load the most recent post-inference analytics result."""
    global _SPATIAL_INDEX
    for d in DEMO_OUTPUT_DIRS:
        result_file = d / "result.json"
        if result_file.is_file():
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                defs = data.get("detections") or data.get("potholes") or []
                if _SPATIAL_INDEX is None and defs:
                    _SPATIAL_INDEX = DefectSpatialIndex(defs)
                return data
            except Exception as e:
                log.warning("Failed parsing %s: %s", result_file, e)
    # Default fallback Town04 demo structure with transform_to_geolocation GPS coordinates
    fallback = {
        "image_id": "demo_town04_scene",
        "timestamp": utc_iso(),
        "road_segment_id": "seg_carla_town04_0042",
        "map_name": "Town04",
        "geolocation": {"lat": 13.0827, "lon": 80.2707, "alt_m": 35.0},
        "road_health": {
            "road_health_score": 54.6,
            "condition_class": "poor",
            "components": {
                "pothole_penalty": 34.9,
                "crack_penalty": 0.0,
                "water_penalty": 7.5,
                "surface_penalty": 3.0,
            },
            "explanation": "Significant road degradation caused by severe water-filled pothole cavitation on Town04 freeway corridor.",
        },
        "prediction": {
            "deterioration_probability": 0.98,
            "pothole_formation_probability": 0.99,
            "prediction_horizon_days": 30,
            "progression_trend": "critical",
            "scientific_status": "CARLA-SYNTHETIC ONLY",
        },
        "detections": [
            {
                "defect_id": "demo-pothole-001",
                "defect_type": "pothole",
                "confidence": 0.88,
                "bbox": [350, 400, 450, 500],
                "mask_area_pixels": 6360,
                "estimated_area_m2": 0.45,
                "estimated_depth_m": 0.08,
                "is_water_filled": False,
                "water_confidence": 0.12,
                "latitude": 13.0827,
                "longitude": 80.2707,
                "world_x": 0.0,
                "world_y": 0.0,
                "severity": {"severity": "high", "severity_score": 0.72},
            },
            {
                "defect_id": "demo-pothole-002",
                "defect_type": "water_filled_pothole",
                "confidence": 0.92,
                "bbox": [770, 475, 930, 565],
                "mask_area_pixels": 10050,
                "estimated_area_m2": 1.10,
                "estimated_depth_m": 0.14,
                "is_water_filled": True,
                "water_confidence": 0.85,
                "latitude": 13.0831,
                "longitude": 80.2712,
                "world_x": 41.67,
                "world_y": 0.0,
                "severity": {"severity": "critical", "severity_score": 0.94},
            },
        ],
    }
    if _SPATIAL_INDEX is None:
        _SPATIAL_INDEX = DefectSpatialIndex(fallback["detections"])
    return fallback


def get_latest_work_orders_data() -> Dict[str, Any]:
    """Locate and load the most recent VLM maintenance work orders."""
    for d in DEMO_OUTPUT_DIRS:
        wo_file = d / "work_orders.json"
        if wo_file.is_file():
            try:
                return json.loads(wo_file.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Failed parsing %s: %s", wo_file, e)
    return {
        "metadata": {
            "total_segments_with_work_orders": 1,
            "total_critical_defects_remediated": 1,
        },
        "work_orders": [
            {
                "work_order_id": "WO-SEG_CARLA_TOWN04-demo-pothole-002",
                "road_segment_id": "seg_carla_town04_0042",
                "defect_id": "demo-pothole-002",
                "defect_class": "water_filled_pothole",
                "severity_tier": "critical",
                "area_m2": 1.10,
                "is_water_filled": True,
                "work_order_text": "URGENT: High-priority immediate remediation is required on road segment 'seg_carla_town04_0042' for a critical-tier water_filled_pothole covering 1.10 m² with critical hydroplaning water accumulation. Crews must dewater the cavity using a submersible pump, square-cut and clean the perimeter, apply cationic rapid-setting tack coat emulsion (CRS-2), and place hot-mix asphalt (HMA Type B) compacted with a vibratory plate in two 50 mm lifts. Establish MUTCD-compliant single-lane closure taper with channelizing cones and directional arrow board, verifying zero-settlement and flush straightedge tolerance prior to reopening the lane to traffic.",
                "required_materials": [
                    "Hot-Mix Asphalt (HMA Type B)",
                    "Bituminous Tack Coat Emulsion (CRS-2 / SS-1h)",
                    "Crushed Aggregate Base (Grade D)",
                    "Hydraulic Cement Quick-Set Mortar",
                ],
                "required_equipment": [
                    "Vibratory Plate Compactor (15 kN)",
                    "High-Pressure Air Lance / Debris Blower",
                    "Diamond Pavement Saw",
                    "Submersible Dewatering Trash Pump",
                ],
                "safety_measures": "MUTCD Chapter 6H temporary traffic control taper, reflective advance warning signs, and safety cones.",
                "estimated_crew_size": 3,
                "target_resolution_hours": 12,
                "engine": "domain_rule_vlm_engine",
            }
        ],
    }


def get_spatial_index() -> DefectSpatialIndex:
    global _SPATIAL_INDEX
    if _SPATIAL_INDEX is None:
        res = get_latest_result_data()
        defs = res.get("detections") or res.get("potholes") or []
        _SPATIAL_INDEX = DefectSpatialIndex(defs)
    return _SPATIAL_INDEX


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status": "online",
        "service": "RoadSentinel Government Analytics & Inference Gateway",
        "version": "2.1",
        "carla_map": "Town04",
        "timestamp": utc_iso(),
    }


@app.get("/api/results")
def get_results():
    return JSONResponse(get_latest_result_data())


@app.get("/api/work_orders")
def get_work_orders():
    return JSONResponse(get_latest_work_orders_data())


@app.get("/api/stats")
def get_stats():
    res = get_latest_result_data()
    wo = get_latest_work_orders_data()
    rh = res.get("road_health", {})
    pred = res.get("prediction", {})
    detections = res.get("detections", [])
    potholes_list = res.get("potholes", [])
    all_defs = detections if detections else potholes_list

    score = rh.get("road_health_score", 100.0)
    cond = rh.get("condition_class", "Good").upper()
    total_def = len(all_defs)
    water_hazards = sum(1 for d in all_defs if d.get("is_water_filled") or d.get("water_flag"))
    critical_def = sum(1 for d in all_defs if (d.get("severity") or {}).get("severity") == "critical" or (d.get("severity_score") or 0.0) >= 0.85)

    return {
        "road_health_score": round(score, 1),
        "condition_class": cond,
        "total_defects": total_def,
        "critical_hazards": critical_def,
        "water_hazards": water_hazards,
        "work_orders_count": len(wo.get("work_orders", [])),
        "deterioration_probability": pred.get("deterioration_probability", 0.0),
        "prediction_horizon_days": pred.get("prediction_horizon_days", 30),
        "road_segment_id": res.get("road_segment_id", "N/A"),
        "map_name": res.get("map_name", "Town04"),
        "geolocation": res.get("geolocation", {"lat": 13.0827, "lon": 80.2707}),
    }


@app.get("/api/geofence/zones")
def get_geofence_zones(radius_m: float = Query(50.0, description="Default geofence radius in meters")):
    s_index = get_spatial_index()
    zones = s_index.create_geofence_zones(default_radius_m=radius_m)
    return JSONResponse({"total_zones": len(zones), "radius_m": radius_m, "zones": zones})


@app.get("/api/geofence/query")
def query_geofence(
    lat: float = Query(..., description="Query Latitude"),
    lon: float = Query(..., description="Query Longitude"),
    radius_m: float = Query(50.0, description="Radius in meters"),
):
    s_index = get_spatial_index()
    nearby = s_index.query_radius(lat, lon, radius_m=radius_m)
    return JSONResponse({
        "query_coords": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "total_hazards_in_range": len(nearby),
        "hazards": nearby,
    })


@app.post("/api/geofence/check_proximity")
async def check_proximity(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    driver_lat = float(body.get("latitude", 13.0827))
    driver_lon = float(body.get("longitude", 80.2707))
    driver_speed = float(body.get("speed_kmph", 60.0))
    warning_r = float(body.get("warning_radius_m", 75.0))
    critical_r = float(body.get("critical_radius_m", 25.0))

    s_index = get_spatial_index()
    alert_info = s_index.evaluate_driver_hazard(
        driver_lat=driver_lat,
        driver_lon=driver_lon,
        driver_speed_kmph=driver_speed,
        warning_radius_m=warning_r,
        critical_radius_m=critical_r,
    )
    return JSONResponse(alert_info)


@app.get("/api/map/town04.png")
def get_town04_map():
    for d in DEMO_OUTPUT_DIRS:
        map_p = d / "town04_map.png"
        if map_p.is_file():
            return FileResponse(str(map_p))
    env_map = ROOT.parent / "env" / "assets" / "town04_map.png"
    if env_map.is_file():
        return FileResponse(str(env_map))
    root_map = ROOT / "output" / "town04_map.png"
    if root_map.is_file():
        return FileResponse(str(root_map))
    raise HTTPException(status_code=404, detail="Town04 map not found")


@app.get("/api/overlays/{filename}")
def get_overlay_image(filename: str):
    for d in DEMO_OUTPUT_DIRS:
        fpath = d / filename
        if fpath.is_file():
            return FileResponse(str(fpath))
    # Check env output directory
    env_img = ROOT.parent / "env" / "output" / "images" / filename
    if env_img.is_file():
        return FileResponse(str(env_img))
    raise HTTPException(status_code=404, detail=f"Overlay {filename} not found")


@app.post("/infer")
async def infer_endpoint(image: UploadFile = File(...), metadata_json: str = Form("{}")):
    global _CACHED_PIPELINE, _SPATIAL_INDEX
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory() as td:
        image_path = Path(td) / f"input{suffix}"
        image_path.write_bytes(await image.read())
        meta_path = Path(td) / "metadata.json"
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid metadata_json: {exc}")
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        try:
            if _CACHED_PIPELINE is None:
                _CACHED_PIPELINE = load_pipeline(CONFIG.device, CONFIG.memory_bank_dir)
            result = infer(image_path, meta_path, pipeline=_CACHED_PIPELINE)
            # Update spatial index
            _SPATIAL_INDEX = DefectSpatialIndex(result.potholes)
        except Exception as exc:
            log.exception("Inference failed:")
            raise HTTPException(500, str(exc))
        return JSONResponse(result.to_dict())


# ---------------------------------------------------------------------------
# Town04 Custom Map & Geofencing Government Dashboard Interface
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RoadSentinel — CARLA Town04 Infrastructure Health & Geofencing HUD</title>
  <!-- Google Fonts & Leaflet Map CSS -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    :root {
      --bg-dark: #070a11;
      --bg-card: rgba(15, 23, 42, 0.78);
      --border-card: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(0, 242, 254, 0.3);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --cyan: #00f2fe;
      --cyan-glow: rgba(0, 242, 254, 0.4);
      --green: #10b981;
      --yellow: #f59e0b;
      --orange: #f97316;
      --red: #ef4444;
      --purple: #8b5cf6;
      --radius: 16px;
      --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.08) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.6) 0px, transparent 100%);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Top Navigation Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 36px;
      background: rgba(7, 10, 17, 0.88);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-card);
      position: sticky;
      top: 0;
      z-index: 1000;
    }

    .brand { display: flex; align-items: center; gap: 12px; }

    .brand-icon {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 20px;
      color: #000;
      box-shadow: 0 0 20px var(--cyan-glow);
    }

    .brand-title {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(to right, #fff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
      font-size: 11px;
      color: var(--cyan);
      font-weight: 600;
      letter-spacing: 1px;
      text-transform: uppercase;
    }

    .status-hud { display: flex; align-items: center; gap: 18px; }

    .badge {
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .badge-town04 {
      background: rgba(139, 92, 246, 0.15);
      border: 1px solid rgba(139, 92, 246, 0.4);
      color: var(--purple);
    }

    .badge-live {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: var(--green);
    }

    .badge-live::before {
      content: "";
      width: 8px;
      height: 8px;
      background: var(--green);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--green);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.2); }
    }

    .btn-refresh {
      background: linear-gradient(135deg, rgba(0, 242, 254, 0.2), rgba(139, 92, 246, 0.2));
      border: 1px solid var(--border-accent);
      color: #fff;
      padding: 8px 18px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .btn-refresh:hover {
      background: linear-gradient(135deg, rgba(0, 242, 254, 0.35), rgba(139, 92, 246, 0.35));
      box-shadow: 0 0 15px var(--cyan-glow);
      transform: translateY(-1px);
    }

    /* Main Container */
    main {
      padding: 28px 36px;
      max-width: 1700px;
      margin: 0 auto;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 24px;
      flex: 1;
    }

    /* Driver Proximity Alert Banner */
    #proximity-banner {
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 12px;
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      transition: all 0.3s ease;
    }

    #proximity-banner.warning {
      background: rgba(245, 158, 11, 0.15);
      border-color: rgba(245, 158, 11, 0.5);
      box-shadow: 0 0 20px rgba(245, 158, 11, 0.2);
    }

    #proximity-banner.critical {
      background: rgba(239, 68, 68, 0.18);
      border-color: rgba(239, 68, 68, 0.6);
      box-shadow: 0 0 25px rgba(239, 68, 68, 0.3);
      animation: alertPulse 1.5s infinite alternate;
    }

    @keyframes alertPulse {
      from { border-color: rgba(239, 68, 68, 0.4); }
      to { border-color: rgba(239, 68, 68, 1); }
    }

    .alert-icon {
      font-size: 20px;
      display: flex;
      align-items: center;
    }

    .alert-text {
      flex: 1;
      font-size: 13px;
      font-weight: 600;
      color: #f1f5f9;
    }

    /* KPI Metrics Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
    }

    .kpi-card {
      background: var(--bg-card);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border-card);
      border-radius: var(--radius);
      padding: 22px 24px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
      transition: transform 0.2s ease;
    }

    .kpi-card:hover {
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.15);
    }

    .kpi-card::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(to right, var(--cyan), var(--purple));
    }

    .kpi-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .kpi-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .kpi-value {
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -1px;
      display: flex;
      align-items: baseline;
      gap: 6px;
    }

    .kpi-desc { font-size: 12px; color: var(--text-muted); margin-top: 6px; }

    /* Main 2-Column Layout */
    .dashboard-layout {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
    }

    @media (max-width: 1200px) {
      .dashboard-layout { grid-template-columns: 1fr; }
    }

    .panel {
      background: var(--bg-card);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border-card);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-card);
      padding-bottom: 14px;
    }

    .panel-title {
      font-size: 16px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* Map Box */
    #map-wrapper {
      position: relative;
      width: 100%;
      height: 420px;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border-card);
      background: #090e17;
    }

    #map-container {
      height: 100%;
      width: 100%;
      background: #060a12;
      z-index: 1;
    }

    /* Map Geofence Control Overlay */
    .map-geofence-controls {
      position: absolute;
      bottom: 14px;
      left: 14px;
      right: 14px;
      background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-card);
      border-radius: 10px;
      padding: 10px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      z-index: 500;
    }

    .geofence-slider-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 12px;
      color: var(--text-muted);
      flex: 1;
    }

    .geofence-slider-wrap input {
      flex: 1;
      accent-color: var(--cyan);
      cursor: pointer;
    }

    .btn-sim-drive {
      background: rgba(0, 242, 254, 0.15);
      border: 1px solid var(--cyan);
      color: var(--cyan);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .btn-sim-drive:hover {
      background: rgba(0, 242, 254, 0.3);
      box-shadow: 0 0 10px var(--cyan-glow);
    }

    /* Image Inspector & Overlays */
    .viewer-controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .tab-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-card);
      color: var(--text-muted);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .tab-btn.active, .tab-btn:hover {
      background: rgba(0, 242, 254, 0.15);
      border-color: var(--cyan);
      color: var(--cyan);
    }

    .image-frame {
      width: 100%;
      height: 380px;
      background: #020617;
      border-radius: 12px;
      border: 1px solid var(--border-card);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .image-frame img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }

    /* Work Orders Section */
    .work-orders-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 6px;
    }

    .work-order-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-card);
      border-left: 4px solid var(--orange);
      border-radius: 10px;
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .work-order-card.critical {
      border-left-color: var(--red);
      background: rgba(239, 68, 68, 0.05);
    }

    .wo-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      color: var(--text-muted);
    }

    .wo-id {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: var(--cyan);
    }

    .wo-text { font-size: 13px; line-height: 1.5; color: #e2e8f0; }

    .wo-tags { display: flex; gap: 8px; flex-wrap: wrap; }

    .wo-tag {
      background: rgba(255, 255, 255, 0.06);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 500;
      color: #cbd5e1;
    }

    /* Severity Breakdown Bar */
    .score-breakdown {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 14px 16px;
      background: rgba(255, 255, 255, 0.02);
      border-radius: 10px;
      border: 1px solid var(--border-card);
    }

    .bar-row {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-muted);
    }

    .progress-track {
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 4px;
    }

    .progress-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }

    /* Custom Scrollbars */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 3px; }

    footer {
      padding: 16px 36px;
      border-top: 1px solid var(--border-card);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      color: var(--text-muted);
      background: rgba(7, 10, 17, 0.88);
    }
  </style>
</head>
<body>

  <!-- Top Header HUD -->
  <header>
    <div class="brand">
      <div class="brand-icon">RS</div>
      <div>
        <div class="brand-title">RoadSentinel</div>
        <div class="brand-subtitle">Town04 Autonomous Infrastructure & Geofencing System</div>
      </div>
    </div>
    <div class="status-hud">
      <span class="badge badge-town04">CARLA TOWN04 (HIGHWAY)</span>
      <span class="badge badge-live">KD-TREE GEOFENCE ACTIVE</span>
      <span style="font-size: 12px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;" id="clock-display">--:--:-- UTC</span>
      <button class="btn-refresh" onclick="fetchDashboardData(true)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
        Refresh Data
      </button>
    </div>
  </header>

  <!-- Main Content Area -->
  <main>
    <!-- Live Driver Proximity Alert Banner -->
    <div id="proximity-banner">
      <div class="alert-icon">🛡️</div>
      <div class="alert-text" id="alert-text-content">
        <b>Driver Proximity HUD:</b> Corridor clear — no road surface hazards in immediate vicinity (KD-Tree active).
      </div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--cyan);" id="nearest-hazard-dist">
        Nearest Defect: 38.4m
      </div>
    </div>

    <!-- Top KPI Cards -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-title">Road Health Index</span>
          <span class="badge" id="kpi-cond-badge" style="background: rgba(239,68,68,0.15); color: var(--orange);">POOR</span>
        </div>
        <div class="kpi-value" id="kpi-score">54.6<span style="font-size: 16px; color: var(--text-muted); font-weight: 500;">/100</span></div>
        <div class="kpi-desc" id="kpi-segment-id">Segment: seg_carla_town04_0042</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-title">Detected Defects</span>
          <svg width="18" height="18" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        </div>
        <div class="kpi-value" id="kpi-defects">2 <span style="font-size: 14px; color: var(--orange); font-weight: 600;">(1 Critical)</span></div>
        <div class="kpi-desc">Total Surface Defects Found</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-title">Water Hazards</span>
          <svg width="18" height="18" fill="none" stroke="#38bdf8" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>
        </div>
        <div class="kpi-value" style="color: #38bdf8;" id="kpi-water">1</div>
        <div class="kpi-desc">High Hydroplaning Risk Cavities</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-title">VLM Work Orders</span>
          <svg width="18" height="18" fill="none" stroke="var(--purple)" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        </div>
        <div class="kpi-value" style="color: var(--purple);" id="kpi-workorders">1</div>
        <div class="kpi-desc">Actionable Maintenance Dispatches</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-title">30-Day Deterioration</span>
          <svg width="18" height="18" fill="none" stroke="var(--red)" stroke-width="2" viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/></svg>
        </div>
        <div class="kpi-value" style="color: var(--red);" id="kpi-pred">98<span style="font-size: 16px; font-weight: 500;">%</span></div>
        <div class="kpi-desc">Forecasted Failure Risk</div>
      </div>
    </div>

    <!-- 2-Column Layout -->
    <div class="dashboard-layout">
      <!-- Left Column: Custom CARLA Town04 Map & Analytics Breakdown -->
      <div style="display: flex; flex-direction: column; gap: 24px;">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="18" height="18" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>
              CARLA Town04 Custom Highway Map & Geofences
            </div>
            <span style="font-size: 12px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;" id="gps-coords">13.0827° N, 80.2707° E</span>
          </div>
          
          <div id="map-wrapper">
            <div id="map-container"></div>
            <!-- Interactive Geofence Controls -->
            <div class="map-geofence-controls">
              <div class="geofence-slider-wrap">
                <span>Geofence Radius:</span>
                <input type="range" id="geofence-radius-slider" min="10" max="120" value="50" step="5" oninput="updateGeofenceRadius(this.value)">
                <span id="radius-val" style="font-weight: 700; color: var(--cyan); min-width: 40px;">50m</span>
              </div>
              <button class="btn-sim-drive" onclick="toggleDriveSimulation()">
                <span id="sim-drive-btn-text">▶ Simulate Driver Cruise</span>
              </button>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="18" height="18" fill="none" stroke="var(--purple)" stroke-width="2" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
              Road Health Index Breakdown & Penalties
            </div>
          </div>
          <div class="score-breakdown">
            <div class="bar-row">
              <span>Pothole Severity Deductions</span>
              <span id="pothole-penalty-text">-24.9 pts</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="pothole-penalty-bar" style="width: 50%; background: var(--orange);"></div>
            </div>

            <div class="bar-row" style="margin-top: 10px;">
              <span>Pothole Count Density</span>
              <span id="count-penalty-text">-10.0 pts</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="count-penalty-bar" style="width: 20%; background: var(--yellow);"></div>
            </div>

            <div class="bar-row" style="margin-top: 10px;">
              <span>Water Hazard Risk</span>
              <span id="water-penalty-text">-7.5 pts</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="water-penalty-bar" style="width: 15%; background: #38bdf8;"></div>
            </div>

            <div class="bar-row" style="margin-top: 10px;">
              <span>Surface Wear & Fatigue</span>
              <span id="surface-penalty-text">-3.0 pts</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="surface-penalty-bar" style="width: 6%; background: var(--purple);"></div>
            </div>
          </div>
          <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5; background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px;" id="explanation-text">
            Explanation: Severe structural depression with standing water detected on corridor.
          </div>
        </div>
      </div>

      <!-- Right Column: Visual Inspector & VLM Work Orders -->
      <div style="display: flex; flex-direction: column; gap: 24px;">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="18" height="18" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              Aerial Inspection Overlay Viewer
            </div>
            <div class="viewer-controls">
              <button class="tab-btn active" onclick="switchOverlay('road_health_overlay.jpg', this)">Road Health HUD</button>
              <button class="tab-btn" onclick="switchOverlay('severity_overlay.jpg', this)">Severity Map</button>
              <button class="tab-btn" onclick="switchOverlay('detection_overlay.jpg', this)">Detection Box</button>
              <button class="tab-btn" onclick="switchOverlay('input_frame.jpg', this)">Raw RGB</button>
            </div>
          </div>
          <div class="image-frame">
            <img id="overlay-img" src="/api/overlays/road_health_overlay.jpg" alt="Aerial Inspection Overlay" onerror="this.src='/api/overlays/detection_overlay.jpg'" />
          </div>
        </div>

        <div class="panel" style="flex: 1;">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="18" height="18" fill="none" stroke="var(--green)" stroke-width="2" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
              VLM Automated Repair Work Orders
            </div>
            <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: var(--green);" id="wo-count-badge">1 Active</span>
          </div>
          <div class="work-orders-list" id="work-orders-container">
            <!-- Dynamic Work Orders populated by JS -->
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- Footer -->
  <footer>
    <div>RoadSentinel v2.1 • CARLA Town04 Photogrammetric Assessment & Driver-Proximity Geofencing</div>
    <div style="font-family: 'JetBrains Mono', monospace;">CARLA 0.9.16 • DINOv2 • SAM2.1 • KD-Tree Geofence • Gemini VLM</div>
  </footer>

  <script>
    function updateClock() {
      const now = new Date();
      document.getElementById('clock-display').innerText = now.toUTCString().split(' ')[4] + ' UTC';
    }
    setInterval(updateClock, 1000);
    updateClock();

    // -------------------------------------------------------------------------
    // Custom CARLA Town04 Map Layer via Leaflet
    // -------------------------------------------------------------------------
    const town04OriginLat = 13.0827;
    const town04OriginLon = 80.2707;

    const map = L.map('map-container', {
      center: [town04OriginLat, town04OriginLon],
      zoom: 16,
      minZoom: 14,
      maxZoom: 20,
      attributionControl: false
    });

    // Dedicated CARLA Town04 Base Map Orthomosaic Layer
    const town04Bounds = [
      [town04OriginLat - 0.0030, town04OriginLon - 0.0030],
      [town04OriginLat + 0.0030, town04OriginLon + 0.0030]
    ];
    L.imageOverlay('/api/map/town04.png', town04Bounds, { opacity: 0.98 }).addTo(map);

    let town04Layer = L.layerGroup().addTo(map);
    let geofenceLayer = L.layerGroup().addTo(map);
    let driverLayer = L.layerGroup().addTo(map);

    // Render CARLA Town04 Highway Layout Vector Corridors
    function drawTown04Infrastructure() {
      town04Layer.clearLayers();

      // Main 8-lane Town04 Highway Loop Coordinates
      const highwayLoop = [
        [town04OriginLat - 0.0035, town04OriginLon - 0.0020],
        [town04OriginLat - 0.0015, town04OriginLon - 0.0010],
        [town04OriginLat, town04OriginLon],
        [town04OriginLat + 0.0020, town04OriginLon + 0.0010],
        [town04OriginLat + 0.0035, town04OriginLon + 0.0025],
        [town04OriginLat + 0.0040, town04OriginLon + 0.0010],
        [town04OriginLat + 0.0030, town04OriginLon - 0.0015],
        [town04OriginLat + 0.0005, town04OriginLon - 0.0025],
        [town04OriginLat - 0.0020, town04OriginLon - 0.0030],
        [town04OriginLat - 0.0035, town04OriginLon - 0.0020]
      ];

      // Highway Roadbed
      L.polyline(highwayLoop, {
        color: '#1e293b',
        weight: 24,
        opacity: 0.95
      }).addTo(town04Layer);

      // Highway Surface
      L.polyline(highwayLoop, {
        color: '#334155',
        weight: 18,
        opacity: 0.9
      }).addTo(town04Layer);

      // Highway Centerline Dash
      L.polyline(highwayLoop, {
        color: '#fbbf24',
        weight: 2,
        dashArray: '8, 8',
        opacity: 0.8
      }).addTo(town04Layer);

      // Active Flight Corridor Segment
      const surveySegment = [
        [town04OriginLat - 0.0015, town04OriginLon - 0.0008],
        [town04OriginLat, town04OriginLon],
        [town04OriginLat + 0.0015, town04OriginLon + 0.0008]
      ];

      L.polyline(surveySegment, {
        color: 'rgba(0, 242, 254, 0.4)',
        weight: 12,
        opacity: 0.6
      }).addTo(town04Layer);

      // Cloverleaf interchange on-ramps
      const onRamp = [
        [town04OriginLat - 0.0010, town04OriginLon + 0.0015],
        [town04OriginLat, town04OriginLon + 0.0008],
        [town04OriginLat + 0.0010, town04OriginLon]
      ];
      L.polyline(onRamp, { color: '#475569', weight: 8, opacity: 0.8 }).addTo(town04Layer);
    }

    drawTown04Infrastructure();

    // -------------------------------------------------------------------------
    // Driver Proximity & Geofencing Simulation
    // -------------------------------------------------------------------------
    let currentGeofenceRadius = 50;
    let driverMarker = null;
    let driverSimInterval = null;
    let driverPosIndex = 0;
    let cachedDefects = [];

    // Simulated highway trajectory points for driver car
    const driverTrajectory = [
      [town04OriginLat - 0.0012, town04OriginLon - 0.0006],
      [town04OriginLat - 0.0008, town04OriginLon - 0.0004],
      [town04OriginLat - 0.0004, town04OriginLon - 0.0002],
      [town04OriginLat, town04OriginLon],
      [town04OriginLat + 0.0004, town04OriginLon + 0.0002],
      [town04OriginLat + 0.0008, town04OriginLon + 0.0004],
      [town04OriginLat + 0.0012, town04OriginLon + 0.0006],
      [town04OriginLat + 0.0016, town04OriginLon + 0.0008]
    ];

    function initDriverMarker() {
      const carIcon = L.divIcon({
        className: 'driver-car-icon',
        html: `<div style="font-size: 22px; filter: drop-shadow(0 0 10px #00f2fe); cursor: grab;">🚗</div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14]
      });

      const startPos = driverTrajectory[0];
      driverMarker = L.marker(startPos, { icon: carIcon, draggable: true }).addTo(driverLayer);

      driverMarker.on('drag', function(e) {
        const pos = e.target.getLatLng();
        checkDriverProximity(pos.lat, pos.lng);
      });

      checkDriverProximity(startPos[0], startPos[1]);
    }

    initDriverMarker();

    function updateGeofenceRadius(val) {
      currentGeofenceRadius = parseFloat(val);
      document.getElementById('radius-val').innerText = `${val}m`;
      renderDefectsAndGeofences();
      if (driverMarker) {
        const pos = driverMarker.getLatLng();
        checkDriverProximity(pos.lat, pos.lng);
      }
    }

    function toggleDriveSimulation() {
      const btnText = document.getElementById('sim-drive-btn-text');
      if (driverSimInterval) {
        clearInterval(driverSimInterval);
        driverSimInterval = null;
        btnText.innerText = '▶ Simulate Driver Cruise';
      } else {
        btnText.innerText = '⏸ Pause Cruise';
        driverSimInterval = setInterval(() => {
          driverPosIndex = (driverPosIndex + 1) % driverTrajectory.length;
          const pos = driverTrajectory[driverPosIndex];
          driverMarker.setLatLng(pos);
          checkDriverProximity(pos[0], pos[1]);
        }, 1200);
      }
    }

    async function checkDriverProximity(lat, lon) {
      try {
        const res = await fetch('/api/geofence/check_proximity', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            latitude: lat,
            longitude: lon,
            warning_radius_m: currentGeofenceRadius,
            critical_radius_m: Math.max(15, currentGeofenceRadius * 0.4)
          })
        });
        const alertData = await res.json();
        const banner = document.getElementById('proximity-banner');
        const alertContent = document.getElementById('alert-text-content');
        const distText = document.getElementById('nearest-hazard-dist');

        if (alertData.alert_triggered) {
          banner.className = alertData.hazard_level === 'critical' ? 'critical' : 'warning';
          alertContent.innerHTML = `<b>${alertData.status} ALERT:</b> ${alertData.message}`;
          distText.innerText = `Nearest: ${alertData.nearest_distance_m.toFixed(1)}m`;
        } else {
          banner.className = '';
          alertContent.innerHTML = `<b>Driver Proximity HUD:</b> Corridor clear — no road surface hazards within ${currentGeofenceRadius}m.`;
          distText.innerText = `Range: Safe (> ${currentGeofenceRadius}m)`;
        }
      } catch (err) {
        console.error('Proximity check error:', err);
      }
    }

    function renderDefectsAndGeofences() {
      geofenceLayer.clearLayers();

      cachedDefects.forEach(d => {
        const lat = d.latitude || town04OriginLat;
        const lon = d.longitude || town04OriginLon;
        const isWater = d.is_water_filled || d.water_flag;
        const markerColor = isWater ? '#38bdf8' : '#ef4444';
        const sevTier = (d.severity || {}).severity || (isWater ? 'critical' : 'high');
        const rMeters = sevTier === 'critical' || isWater ? currentGeofenceRadius : currentGeofenceRadius * 0.75;

        // Dynamic Geofence Boundary Ring (KD-Tree boundary)
        L.circle([lat, lon], {
          radius: rMeters,
          color: markerColor,
          fillColor: markerColor,
          fillOpacity: 0.12,
          weight: 1.5,
          dashArray: '4, 6'
        }).addTo(geofenceLayer);

        // Core Defect Marker
        const circle = L.circleMarker([lat, lon], {
          radius: 8,
          fillColor: markerColor,
          color: '#ffffff',
          weight: 2,
          opacity: 1,
          fillOpacity: 0.95
        }).addTo(geofenceLayer);

        const defectTitle = d.defect_type || (isWater ? 'Water-filled Pothole' : 'Pothole');
        const areaVal = d.estimated_area_m2 || d.area_m2 || 0.5;
        circle.bindPopup(`
          <div style="color: #0f172a; font-family: 'Plus Jakarta Sans', sans-serif;">
            <strong style="color: ${markerColor}; font-size: 13px;">${defectTitle}</strong><br/>
            Map: <b>Town04 Highway</b><br/>
            GPS: <b>${lat.toFixed(5)}°, ${lon.toFixed(5)}°</b><br/>
            Area: <b>${areaVal.toFixed(2)} m²</b> | Depth: <b>${((d.estimated_depth_m || 0.08) * 100).toFixed(1)} cm</b><br/>
            Geofence Radius: <b>${rMeters.toFixed(0)}m</b>
          </div>
        `);
      });
    }

    function switchOverlay(filename, btn) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('overlay-img').src = '/api/overlays/' + filename + '?t=' + Date.now();
    }

    async function fetchDashboardData(showNotice = false) {
      try {
        const [statsRes, resultsRes, woRes] = await Promise.all([
          fetch('/api/stats').then(r => r.json()),
          fetch('/api/results').then(r => r.json()),
          fetch('/api/work_orders').then(r => r.json())
        ]);

        // Update KPIs
        document.getElementById('kpi-score').innerHTML = `${statsRes.road_health_score}<span style="font-size: 16px; color: var(--text-muted); font-weight: 500;">/100</span>`;
        document.getElementById('kpi-segment-id').innerText = `Segment: ${statsRes.road_segment_id} (${statsRes.map_name || 'Town04'})`;
        document.getElementById('kpi-defects').innerHTML = `${statsRes.total_defects} <span style="font-size: 14px; color: var(--orange); font-weight: 600;">(${statsRes.critical_hazards} Critical)</span>`;
        document.getElementById('kpi-water').innerText = statsRes.water_hazards;
        document.getElementById('kpi-workorders').innerText = statsRes.work_orders_count;
        document.getElementById('kpi-pred').innerHTML = `${Math.round(statsRes.deterioration_probability * 100)}<span style="font-size: 16px; font-weight: 500;">%</span>`;

        const condBadge = document.getElementById('kpi-cond-badge');
        condBadge.innerText = statsRes.condition_class;
        if (statsRes.road_health_score >= 80) {
          condBadge.style.background = 'rgba(16, 185, 129, 0.15)';
          condBadge.style.color = 'var(--green)';
        } else if (statsRes.road_health_score >= 60) {
          condBadge.style.background = 'rgba(245, 158, 11, 0.15)';
          condBadge.style.color = 'var(--yellow)';
        } else if (statsRes.road_health_score >= 40) {
          condBadge.style.background = 'rgba(249, 115, 22, 0.15)';
          condBadge.style.color = 'var(--orange)';
        } else {
          condBadge.style.background = 'rgba(239, 68, 68, 0.15)';
          condBadge.style.color = 'var(--red)';
        }

        // Cache and plot defects
        cachedDefects = resultsRes.detections || resultsRes.potholes || [];
        renderDefectsAndGeofences();

        // Explanation text
        if (resultsRes.road_health && resultsRes.road_health.explanation) {
          document.getElementById('explanation-text').innerText = 'Explanation: ' + resultsRes.road_health.explanation;
        }

        // Work Orders
        const woContainer = document.getElementById('work-orders-container');
        woContainer.innerHTML = '';
        const workOrders = woRes.work_orders || [];
        document.getElementById('wo-count-badge').innerText = `${workOrders.length} Active`;

        if (workOrders.length === 0) {
          woContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 20px;">No critical remediation work orders required for this segment.</div>';
        } else {
          workOrders.forEach(wo => {
            const card = document.createElement('div');
            card.className = `work-order-card ${wo.severity_tier === 'critical' ? 'critical' : ''}`;
            const materials = (wo.required_materials || []).map(m => `<span class="wo-tag">${m}</span>`).join('');
            const equip = (wo.required_equipment || []).map(e => `<span class="wo-tag">${e}</span>`).join('');

            card.innerHTML = `
              <div class="wo-meta">
                <span class="wo-id">${wo.work_order_id}</span>
                <span class="badge" style="background: rgba(239,68,68,0.15); color: var(--red); font-size: 11px;">${wo.severity_tier.toUpperCase()} TIER</span>
              </div>
              <div class="wo-text">${wo.work_order_text}</div>
              <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-top: 4px;">Required Materials & Equipment:</div>
              <div class="wo-tags">${materials} ${equip}</div>
              <div style="font-size: 11px; color: var(--text-muted); display: flex; gap: 16px; margin-top: 4px;">
                <span>Crew Size: <b>${wo.estimated_crew_size || 2} Techs</b></span>
                <span>Resolution Target: <b>${wo.target_resolution_hours || 24}h</b></span>
                <span>Safety: <b>${wo.safety_measures ? 'MUTCD Standard' : 'Standard'}</b></span>
              </div>
            `;
            woContainer.appendChild(card);
          });
        }

        if (showNotice) {
          console.log("Dashboard refreshed successfully.");
        }
      } catch (err) {
        console.error("Dashboard fetch error:", err);
      }
    }

    fetchDashboardData();
    setInterval(fetchDashboardData, 5000);
  </script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def get_dashboard_ui(request: Request):
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


def main():
    parser = argparse.ArgumentParser(description="RoadSentinel Dashboard & Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    log.info("Starting RoadSentinel Town04 Dashboard & Geofencing Server on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
