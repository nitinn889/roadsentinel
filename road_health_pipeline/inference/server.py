from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from PIL import Image, ImageDraw, ImageFont
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
RESULTS_DIR = WORKSPACE_ROOT / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import load_json, load_rgb, save_json, utc_iso
from inference.run_inference import infer, load_pipeline
from inference.spatial_index import DefectSpatialIndex, haversine_distance_m

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("roadsentinel_server")

app = FastAPI(
    title="RoadSentinel Government Dashboard & Inference API",
    description="Real-time aerial road inspection, defect analytics, spatial deduplication, and automated VLM maintenance work orders.",
    version="3.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Search Paths for 2D Results, CARLA Outputs, and Imagery
# ---------------------------------------------------------------------------

DEMO_OUTPUT_DIRS = [
    RESULTS_DIR,
    ROOT / "results",
    WORKSPACE_ROOT / "results",
    ROOT / "output" / "analytics_demo",
    WORKSPACE_ROOT / "output",
    ROOT / "outputs",
    ROOT / "output",
]

DRONE_IMAGE_DIRS = [
    RESULTS_DIR,
    ROOT / "results",
    WORKSPACE_ROOT / "results",
    WORKSPACE_ROOT / "env" / "output" / "images",
    WORKSPACE_ROOT / "env" / "output",
    WORKSPACE_ROOT / "RoadSentinel_datasets",
    WORKSPACE_ROOT / "dataset",
    WORKSPACE_ROOT / "output" / "images",
    WORKSPACE_ROOT / "output",
    Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output/images"),
]

DRONE_METADATA_DIRS = [
    WORKSPACE_ROOT / "env" / "output",
    WORKSPACE_ROOT / "output",
    RESULTS_DIR,
    ROOT.parent / "env" / "output",
    Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output"),
]

_CACHED_PIPELINE = None
_SPATIAL_INDEX: Optional[DefectSpatialIndex] = None
_LAST_LOADED_FILE: Optional[str] = None
_LAST_LOADED_MTIME: float = 0.0
_CACHED_RESULT_DATA: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 2D Results Parser & Normalizer
# ---------------------------------------------------------------------------

def parse_2d_detection_results(raw_json: Dict[str, Any], src_path: Path) -> Dict[str, Any]:
    """Normalize 2D detection results (from results/detection_results.json) into dashboard schema."""
    meta = raw_json.get("metadata", {})
    img_results = raw_json.get("results", [])

    total_images = meta.get("total_images_processed", len(img_results))
    total_defects = meta.get("total_defects_detected", sum(len(r.get("detections", [])) for r in img_results))
    elapsed_time = meta.get("elapsed_time_s", 0.0)
    conf_thresh = meta.get("confidence_threshold", 0.15)
    min_area_px = meta.get("min_area_px", 50)
    gen_time = meta.get("generated_at", utc_iso())

    all_detections: List[Dict[str, Any]] = []
    water_count = 0
    critical_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0

    total_counter = 0
    for img_idx, r in enumerate(img_results):
        img_name = r.get("image_name", f"image_{img_idx+1:04d}.jpg")
        img_path = r.get("image_path", "")
        overlay_path = r.get("overlay_path", "")
        dets = r.get("detections", [])

        for det_idx, d in enumerate(dets):
            total_counter += 1
            conf = float(d.get("confidence", 0.0))
            anomaly = float(d.get("anomaly_score", 0.5))
            area_px = int(d.get("mask_area_px", d.get("mask_area_pixels", 100)))
            bbox = d.get("bbox_xyxy", [0, 0, 0, 0])
            raw_type = str(d.get("defect_type", "pothole"))
            is_water = "water" in raw_type.lower() or bool(d.get("water_flag", False))

            # Calculate normalized severity score (0.0 to 1.0)
            sev_score = round(max(0.20, min(0.98, anomaly * 0.60 + conf * 0.40 + (0.15 if is_water else 0.0))), 2)
            if sev_score >= 0.85:
                sev_tier = "critical"
                critical_count += 1
            elif sev_score >= 0.65:
                sev_tier = "high"
                high_count += 1
            elif sev_score >= 0.35:
                sev_tier = "medium"
                medium_count += 1
            else:
                sev_tier = "low"
                low_count += 1

            if is_water:
                water_count += 1

            # Estimate nominal area in m² (~0.00035 m² per pixel for street-level resolution)
            area_m2 = round(area_px * 0.00035, 2)
            depth_m = 0.14 if is_water else (0.10 if sev_score >= 0.80 else 0.06)

            # Synthetic GPS offsets for geospatial plotting on Map View
            lat = 13.08270 + (img_idx * 0.00045) + ((det_idx % 6) * 0.00009)
            lon = 80.27070 + (img_idx * 0.00045) + ((det_idx // 6) * 0.00009)

            pid = f"2d-def-{img_idx+1:02d}-{det_idx+1:03d}"

            det_record = {
                "index": total_counter,
                "pothole_id": pid,
                "defect_id": pid,
                "defect_type": raw_type,
                "defect_class": raw_type,
                "confidence": conf,
                "pothole_confidence": conf,
                "anomaly_score": anomaly,
                "bbox_xyxy": bbox,
                "mask_area_pixels": area_px,
                "mask_area_px": area_px,
                "area_m2": area_m2,
                "estimated_depth_m": depth_m,
                "is_water_filled": is_water,
                "water_flag": is_water,
                "water_confidence": 0.88 if is_water else 0.10,
                "severity_score": sev_score,
                "severity_tier": sev_tier,
                "severity_breakdown": {
                    "area": round(min(1.0, area_m2 / 2.0), 3),
                    "depth": round(min(1.0, depth_m / 0.15), 3),
                    "anomaly": anomaly,
                    "confidence": conf,
                    "water": 0.88 if is_water else 0.0,
                },
                "source_image": img_name,
                "image_path": img_path,
                "overlay_path": overlay_path,
                "latitude": lat,
                "longitude": lon,
            }
            all_detections.append(det_record)

    # Road Health calculation
    pothole_pen = min(40.0, total_defects * 2.0)
    sev_pen = min(30.0, (critical_count * 4.0 + high_count * 2.0))
    water_pen = min(20.0, water_count * 8.0)
    surf_pen = 4.0
    health_score = max(10.0, round(100.0 - (pothole_pen + sev_pen + water_pen + surf_pen), 1))
    cond_class = "good" if health_score >= 80 else "fair" if health_score >= 60 else "poor" if health_score >= 40 else "critical"

    # Prediction forecast
    det_prob = min(0.99, round(0.35 + (total_defects * 0.03), 2))
    pot_prob = min(0.99, round(0.40 + (total_defects * 0.025), 2))
    trend = "critical" if total_defects >= 15 else "accelerating" if total_defects >= 5 else "moderate"

    # Generate work orders
    generated_work_orders = []
    for d in all_detections:
        if d["severity_tier"] in ("critical", "high"):
            cid = d["pothole_id"]
            seg_id = "SEG-2D-LOCAL-SURVEY"
            is_w = d["is_water_filled"]
            urgency = "URGENT IMMEDIATE" if d["severity_tier"] == "critical" else "SCHEDULED HIGH-PRIORITY"
            water_note = "Significant water accumulation present; cavity must be dewatered prior to tack application." if is_w else "No active ponding observed."
            
            wo = {
                "work_order_id": f"WO-2D-{cid.upper()}",
                "road_segment_id": seg_id,
                "pothole_id": cid,
                "defect_class": d["defect_class"],
                "severity_tier": d["severity_tier"],
                "severity_score": d["severity_score"],
                "area_m2": d["area_m2"],
                "estimated_depth_m": d["estimated_depth_m"],
                "is_water_filled": is_w,
                "water_hazard": is_w,
                "work_order_text": (
                    f"{urgency}: Road maintenance dispatch for {d['defect_class']} on image '{d['source_image']}'. "
                    f"Bounding box: {d['bbox_xyxy']} (Area: {d['mask_area_px']} px / {d['area_m2']:.2f} m², Severity: {d['severity_score']*100:.0f}%, Confidence: {d['confidence']*100:.0f}%). "
                    f"{water_note} Deploy repair crew with hot-mix asphalt (HMA Type B) and vibratory plate compactor. "
                    f"Establish MUTCD-compliant single-lane traffic control."
                ),
                "required_materials": [
                    "Hot-Mix Asphalt (HMA Type B)" if d["severity_tier"] == "critical" else "Polymer-Modified Asphalt",
                    "Bituminous Tack Coat Emulsion (CRS-2 / SS-1h)",
                    "Crushed Aggregate Base (Grade D)",
                ] + (["Submersible Pump Sorbent Pack", "Hydraulic Quick-Set Cement"] if is_w else []),
                "required_equipment": [
                    "Vibratory Plate Compactor (15 kN)",
                    "High-Pressure Air Lance",
                    "Diamond Pavement Saw",
                ] + (["Submersible Dewatering Trash Pump"] if is_w else []),
                "safety_measures": "MUTCD Chapter 6H temporary traffic control taper.",
                "estimated_crew_size": 3 if d["severity_tier"] == "critical" or is_w else 2,
                "target_resolution_hours": 12 if d["severity_tier"] == "critical" else 24,
                "latitude": d["latitude"],
                "longitude": d["longitude"],
                "engine": "vlm_maintenance_engine_2d",
            }
            generated_work_orders.append(wo)

    return {
        "is_2d_mode": True,
        "source_file": str(src_path),
        "image_id": img_results[0].get("image_name", "2d_imagery") if img_results else "2d_imagery",
        "timestamp": gen_time,
        "road_segment_id": "SEG-2D-LOCAL-SURVEY",
        "map_name": "2D Local Imagery",
        "geolocation": {"lat": 13.0827, "lon": 80.2707, "alt_m": 0.0},
        "metadata": {
            "total_images_processed": total_images,
            "total_defects_detected": total_defects,
            "elapsed_time_s": elapsed_time,
            "confidence_threshold": conf_thresh,
            "min_area_px": min_area_px,
            "generated_at": gen_time,
            "test_mode_2d": True,
        },
        "road_health": {
            "road_health_score": health_score,
            "condition_class": cond_class,
            "components": {
                "pothole_penalty": round(pothole_pen, 1),
                "severity_penalty": round(sev_pen, 1),
                "water_penalty": round(water_pen, 1),
                "surface_penalty": round(surf_pen, 1),
            },
            "explanation": f"2D vision inspection detected {total_defects} crater/pothole anomaly defect(s) across {total_images} photo(s) in {elapsed_time:.2f}s.",
        },
        "prediction": {
            "deterioration_probability": det_prob,
            "pothole_formation_probability": pot_prob,
            "prediction_horizon_days": 30,
            "progression_trend": trend,
            "scientific_status": "2D ZERO-SHOT VISION INFERENCE",
        },
        "detections": all_detections,
        "raw_2d_results": img_results,
        "work_orders": generated_work_orders,
    }


def get_latest_result_data() -> Dict[str, Any]:
    """Locate and load the most recent post-inference analytics result (2D or 3D)."""
    global _SPATIAL_INDEX, _LAST_LOADED_FILE, _LAST_LOADED_MTIME, _CACHED_RESULT_DATA

    # 1. First priority: Check for results/detection_results.json (2D Test Mode)
    for d in DEMO_OUTPUT_DIRS:
        det_file = d / "detection_results.json"
        if det_file.is_file():
            try:
                mtime = det_file.stat().st_mtime
                if _CACHED_RESULT_DATA is None or _LAST_LOADED_FILE != str(det_file) or mtime != _LAST_LOADED_MTIME:
                    raw_data = json.loads(det_file.read_text(encoding="utf-8"))
                    parsed = parse_2d_detection_results(raw_data, det_file)
                    _CACHED_RESULT_DATA = parsed
                    _LAST_LOADED_FILE = str(det_file)
                    _LAST_LOADED_MTIME = mtime
                    defs = parsed.get("detections", [])
                    if defs:
                        _SPATIAL_INDEX = DefectSpatialIndex(defs)
                    # Seed deduplicator with 2D detections and work orders
                    get_deduplicator().seed_from_result(defs, parsed.get("work_orders", []))
                return _CACHED_RESULT_DATA
            except Exception as e:
                log.warning("Failed parsing 2D detection_results.json from %s: %s", det_file, e)

    # 2. Second priority: Check for result.json (3D CARLA Mode)
    for d in DEMO_OUTPUT_DIRS:
        result_file = d / "result.json"
        if result_file.is_file():
            try:
                mtime = result_file.stat().st_mtime
                if _CACHED_RESULT_DATA is None or _LAST_LOADED_FILE != str(result_file) or mtime != _LAST_LOADED_MTIME:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                    _CACHED_RESULT_DATA = data
                    _LAST_LOADED_FILE = str(result_file)
                    _LAST_LOADED_MTIME = mtime
                    defs = data.get("detections") or data.get("potholes") or []
                    if defs:
                        _SPATIAL_INDEX = DefectSpatialIndex(defs)
                    wos = get_latest_work_orders_data().get("work_orders", [])
                    get_deduplicator().seed_from_result(defs, wos)
                return _CACHED_RESULT_DATA
            except Exception as e:
                log.warning("Failed parsing %s: %s", result_file, e)

    # 3. Default fallback demo structure
    fallback = {
        "image_id": "demo_2d_scene",
        "timestamp": utc_iso(),
        "road_segment_id": "SEG-2D-LOCAL-SURVEY",
        "map_name": "Local 2D Imagery",
        "geolocation": {"lat": 13.0827, "lon": 80.2707, "alt_m": 0.0},
        "metadata": {
            "total_images_processed": 1,
            "total_defects_detected": 20,
            "elapsed_time_s": 1.24,
            "confidence_threshold": 0.15,
            "min_area_px": 50,
            "generated_at": utc_iso(),
            "test_mode_2d": True,
        },
        "road_health": {
            "road_health_score": 38.5,
            "condition_class": "poor",
            "components": {
                "pothole_penalty": 35.0,
                "severity_penalty": 20.0,
                "water_penalty": 0.0,
                "surface_penalty": 6.5,
            },
            "explanation": "Significant pavement degradation with multiple high-severity crater anomalies identified.",
        },
        "prediction": {
            "deterioration_probability": 0.95,
            "pothole_formation_probability": 0.98,
            "prediction_horizon_days": 30,
            "progression_trend": "critical",
            "scientific_status": "2D ZERO-SHOT INFERENCE",
        },
        "detections": [],
        "work_orders": [],
    }
    if _SPATIAL_INDEX is None:
        _SPATIAL_INDEX = DefectSpatialIndex([])
    return fallback


def get_latest_work_orders_data() -> Dict[str, Any]:
    """Locate and load the most recent VLM maintenance work orders."""
    res_data = get_latest_result_data()
    if res_data.get("is_2d_mode") and res_data.get("work_orders"):
        return {
            "metadata": {
                "total_segments_with_work_orders": 1,
                "total_critical_defects_remediated": len(res_data.get("work_orders", [])),
            },
            "work_orders": res_data.get("work_orders", []),
        }

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
            "total_critical_defects_remediated": 0,
        },
        "work_orders": [],
    }


def get_drone_images_dir() -> Optional[Path]:
    """Find the directory containing inspection images."""
    for d in DRONE_IMAGE_DIRS:
        if d.is_dir():
            return d
    return None


def get_drone_metadata() -> Dict[str, Dict[str, Any]]:
    """Load telemetry and metadata from metadata.csv if available."""
    import csv
    meta: Dict[str, Dict[str, Any]] = {}
    for d in DRONE_METADATA_DIRS:
        csv_p = d / "metadata.csv"
        if csv_p.is_file():
            try:
                with open(csv_p, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_name = row.get("image_name", "")
                        if not img_name:
                            continue
                        meta[img_name] = {
                            "image_name": img_name,
                            "sim_time_s": float(row.get("sim_time_s", 0.0) or 0.0),
                            "local_x_m": float(row.get("local_x_m", 0.0) or 0.0),
                            "local_y_m": float(row.get("local_y_m", 0.0) or 0.0),
                            "altitude_m": float(row.get("altitude_m", 100.0) or 100.0),
                            "latitude": float(row.get("latitude", 13.0827) or 13.0827),
                            "longitude": float(row.get("longitude", 80.2744) or 80.2744),
                            "yaw_deg": float(row.get("yaw_deg", -82.83) or -82.83),
                            "pitch_deg": float(row.get("pitch_deg", -90.0) or -90.0),
                            "roll_deg": float(row.get("roll_deg", 0.0) or 0.0),
                            "gsd_cm_per_px": float(row.get("gsd_cm_per_px", 6.014) or 6.014),
                        }
                if meta:
                    break
            except Exception as e:
                log.warning("Failed parsing metadata.csv from %s: %s", csv_p, e)
    return meta


def get_all_drone_images_data() -> List[Dict[str, Any]]:
    """Collect all processed images with associated metadata, detections, and overlay references."""
    res_data = get_latest_result_data()

    # 1. Handle 2D Test Mode Results directly from detection_results.json
    if res_data.get("is_2d_mode") and res_data.get("raw_2d_results"):
        raw_list = res_data["raw_2d_results"]
        meta_dict = res_data.get("metadata", {})
        images_data: List[Dict[str, Any]] = []

        for idx, r in enumerate(raw_list):
            fname = r.get("image_name", f"image_{idx+1:04d}.jpg")
            img_path = r.get("image_path", "")
            overlay_p = r.get("overlay_path", "")
            dets = r.get("detections", [])
            shape = r.get("image_shape", [0, 0, 3])

            max_sev = max([float(d.get("severity_score", 0.0)) for d in dets], default=0.0)
            if max_sev == 0.0 and dets:
                max_sev = max([float(d.get("confidence", 0.0)) for d in dets], default=0.0)
            total_area_px = sum([int(d.get("mask_area_px", d.get("mask_area_pixels", 0))) for d in dets])
            total_area_m2 = round(total_area_px * 0.00035, 2)
            has_water = any("water" in str(d.get("defect_type", "")).lower() for d in dets)

            # Format normalized detections
            norm_dets = []
            for d_idx, d in enumerate(dets):
                conf = float(d.get("confidence", 0.0))
                area_px = int(d.get("mask_area_px", d.get("mask_area_pixels", 0)))
                sev = float(d.get("severity_score", 0.0)) or round(conf * 0.9, 2)
                norm_dets.append({
                    "index": d_idx + 1,
                    "pothole_id": f"2d-def-{idx+1:02d}-{d_idx+1:03d}",
                    "bbox_xyxy": d.get("bbox_xyxy", []),
                    "confidence": conf,
                    "pothole_confidence": conf,
                    "mask_area_pixels": area_px,
                    "mask_area_px": area_px,
                    "area_m2": round(area_px * 0.00035, 2),
                    "estimated_depth_m": 0.12 if has_water else 0.08,
                    "defect_type": d.get("defect_type", "pothole"),
                    "severity_score": sev,
                    "is_water_filled": "water" in str(d.get("defect_type", "")).lower(),
                    "source_image": fname,
                })

            images_data.append({
                "index": idx,
                "filename": fname,
                "image_path": str(img_path),
                "image_url": f"/api/drone_image/{fname}",
                "annotated_url": f"/api/annotated_image/{fname}",
                "overlay_path": str(overlay_p),
                "metadata": {
                    "image_name": fname,
                    "mode": "2D Test Mode (Zero-Shot SAM2 + DINOv2)",
                    "altitude_m": "N/A (2D Image)",
                    "sim_time_s": 0.0,
                    "latitude": 13.0827,
                    "longitude": 80.2744,
                    "gsd_cm_per_px": "N/A",
                    "resolution": f"{shape[1]}x{shape[0]}" if len(shape) >= 2 else "Unknown",
                    "elapsed_time_s": meta_dict.get("elapsed_time_s", 1.24),
                    "confidence_threshold": meta_dict.get("confidence_threshold", 0.15),
                    "min_area_px": meta_dict.get("min_area_px", 50),
                    "test_mode_2d": True,
                },
                "detections": norm_dets,
                "pothole_count": len(norm_dets),
                "max_severity_score": round(max_sev, 2),
                "total_area_px": total_area_px,
                "total_area_m2": total_area_m2,
                "max_depth_m": 0.12 if has_water else 0.08,
                "has_water_hazard": has_water,
                "status": f"{len(norm_dets)} Defect{'s' if len(norm_dets) != 1 else ''} Detected (2D Mode)",
            })
        return images_data

    # 2. CARLA 3D Fallback Mode
    img_dir = get_drone_images_dir()
    if not img_dir:
        return []

    all_dets = res_data.get("detections") or res_data.get("potholes") or []
    dets_by_img: Dict[str, List[Dict[str, Any]]] = {}
    for d in all_dets:
        src = d.get("source_image") or Path(d.get("image_path", "")).name
        if src:
            dets_by_img.setdefault(src, []).append(d)

    meta_by_img = get_drone_metadata()

    raw_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith(".")],
        key=lambda x: x.name,
    )

    images_data = []
    for idx, p in enumerate(raw_files):
        fname = p.name
        img_dets = dets_by_img.get(fname, [])
        m = meta_by_img.get(fname, {})

        max_sev = max([float(d.get("severity_score") or 0.0) for d in img_dets], default=0.0)
        total_area = sum([float(d.get("area_m2") or 0.0) for d in img_dets])
        max_depth = max([float(d.get("estimated_depth_m") or 0.0) for d in img_dets], default=0.0)
        has_water = any(bool(d.get("is_water_filled") or d.get("water_flag")) for d in img_dets)

        images_data.append({
            "index": idx,
            "filename": fname,
            "image_path": str(p),
            "image_url": f"/api/drone_image/{fname}",
            "annotated_url": f"/api/annotated_image/{fname}",
            "metadata": m,
            "detections": img_dets,
            "pothole_count": len(img_dets),
            "max_severity_score": round(max_sev, 2),
            "total_area_m2": round(total_area, 2),
            "max_depth_m": round(max_depth, 3),
            "has_water_hazard": has_water,
            "status": "Clean / Nominal" if len(img_dets) == 0 else f"{len(img_dets)} Defect{'s' if len(img_dets) > 1 else ''} Detected",
        })
    return images_data


def annotate_full_drone_image(
    image_path: Path,
    detections: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Burn bounding boxes, defect callout badges, and telemetry HUD onto the photo."""
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Telemetry HUD banner at top-left
        fname = image_path.name
        det_cnt = len(detections)
        is_2d = meta and meta.get("test_mode_2d")

        if is_2d:
            hud_text = f"ROADSENTINEL 2D SURVEY  |  {fname}  |  Res: {w}x{h}  |  DINOv2+SAM2  |  Defects: {det_cnt}"
        else:
            lat_str = f"{meta['latitude']:.5f}° N" if meta and meta.get("latitude") else "13.08° N"
            lon_str = f"{meta['longitude']:.5f}° E" if meta and meta.get("longitude") else "80.27° E"
            alt_str = f"Alt: {meta.get('altitude_m', 100.0):.1f}m" if meta else "Alt: 100.0m"
            time_str = f"T+{meta.get('sim_time_s', 0.0):.1f}s" if meta and 'sim_time_s' in meta else ""
            hud_text = f"ROADSENTINEL SURVEY  |  {fname}  |  {lat_str}, {lon_str}  |  {alt_str}  |  {time_str}  |  Defects: {det_cnt}"

        banner_w = min(w - 20, len(hud_text) * 8 + 30)
        draw.rectangle([10, 10, 10 + banner_w, 38], fill=(7, 10, 17), outline=(0, 242, 254), width=1)
        draw.text((20, 16), hud_text, fill=(240, 249, 255))

        for i, det in enumerate(detections):
            box = det.get("bbox_xyxy")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            sev = float(det.get("severity_score") or 0.0)
            conf = float(det.get("pothole_confidence") or det.get("confidence") or 0.0)
            is_water = bool(det.get("is_water_filled") or det.get("water_flag"))
            area_px = int(det.get("mask_area_px") or det.get("mask_area_pixels") or ((x2 - x1) * (y2 - y1)))
            def_type = (det.get("defect_type") or "Pothole").replace("_", " ").title()

            color = (56, 189, 248) if is_water else (239, 68, 68) if sev >= 0.85 else (249, 115, 22) if sev >= 0.65 else (245, 158, 11)

            # High-visibility bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Technical corner brackets
            c_len = min(14, max(4, (x2 - x1) // 3), max(4, (y2 - y1) // 3))
            draw.rectangle([x1 - 2, y1 - 2, x1 + c_len, y1 + 3], fill=color)
            draw.rectangle([x1 - 2, y1 - 2, x1 + 3, y1 + c_len], fill=color)
            draw.rectangle([x2 - c_len, y2 - 3, x2 + 2, y2 + 2], fill=color)
            draw.rectangle([x2 - 3, y2 - c_len, x2 + 2, y2 + 2], fill=color)

            # Informative label badge
            water_tag = " [WATER]" if is_water else ""
            tag = f"#{i+1} {def_type} | Conf: {conf*100:.0f}% | {area_px}px{water_tag}"
            tag_w = len(tag) * 8 + 14
            tag_h = 20
            ty = max(45, y1 - tag_h - 4)
            tx = max(10, min(w - tag_w - 10, x1))

            draw.rectangle([tx, ty, tx + tag_w, ty + tag_h], fill=(15, 23, 42), outline=color, width=1)
            draw.rectangle([tx, ty, tx + 4, ty + tag_h], fill=color)
            draw.text((tx + 8, ty + 3), tag, fill=(255, 255, 255))

        out_buf = io.BytesIO()
        img.save(out_buf, format="JPEG", quality=90)
        return out_buf.getvalue()
    except Exception as exc:
        log.warning("annotate_full_drone_image failed: %s", exc)
        return image_path.read_bytes()


# ---------------------------------------------------------------------------
# Spatial Deduplication Engine
# ---------------------------------------------------------------------------

DEDUP_RADIUS_M = 3.0


class SpatialDeduplicator:
    """Groups incoming defect detections by proximity and prevents duplicate work orders."""

    def __init__(self):
        self._clusters: Dict[str, Dict[str, Any]] = {}
        self._patch_images: Dict[str, bytes] = {}
        self._all_bboxes: Dict[str, List[List[Union[int, float]]]] = {}
        self._cluster_sizes: Dict[str, int] = {}
        self._work_orders: Dict[str, Dict[str, Any]] = {}
        self._observations: Dict[str, List[Dict[str, Any]]] = {}

    def _generate_work_order(self, cluster_id: str, detection: Dict[str, Any]) -> Dict[str, Any]:
        sev_score = float(detection.get("severity_score") or 0.70)
        sev_tier = "critical" if sev_score >= 0.85 else "high" if sev_score >= 0.65 else "medium"
        is_water = bool(detection.get("is_water_filled") or detection.get("water_flag"))
        def_class = detection.get("defect_type") or detection.get("defect_class") or ("water_filled_pothole" if is_water else "pothole")
        area = float(detection.get("area_m2") or 0.5)
        depth = float(detection.get("estimated_depth_m") or 0.08)
        seg_id = detection.get("road_segment_id") or "seg_carla_town04_0042"
        lat = float(detection.get("latitude") or detection.get("lat") or 13.0827)
        lon = float(detection.get("longitude") or detection.get("lon") or 80.2707)

        urgency = "URGENT IMMEDIATE" if sev_tier == "critical" else "SCHEDULED HIGH-PRIORITY"
        water_note = "Significant water pooling present; cavity must be dewatered prior to tack application." if is_water else "No active ponding observed."

        clean_cid = str(cluster_id).replace("carla-", "").replace("road_", "R").replace("_", "-")
        return {
            "work_order_id": f"WO-{seg_id.upper()[:10]}-{clean_cid}",
            "road_segment_id": seg_id,
            "pothole_id": cluster_id,
            "defect_class": def_class,
            "severity_tier": sev_tier,
            "severity_score": round(sev_score, 2),
            "area_m2": round(area, 2),
            "estimated_depth_m": round(depth, 3),
            "is_water_filled": is_water,
            "water_hazard": is_water,
            "work_order_text": (
                f"{urgency}: Road repair required at GPS ({lat:.5f}°, {lon:.5f}°) on segment '{seg_id}'. "
                f"Defect is a {def_class} (Area: {area:.2f} m², Depth: {depth*100:.1f} cm, Severity: {sev_score*100:.0f}%). "
                f"{water_note} Deploy asphalt crew with hot-mix asphalt (HMA Type B) and vibratory plate compactor. "
                f"Establish MUTCD-compliant single-lane traffic control."
            ),
            "required_materials": [
                "Hot-Mix Asphalt (HMA Type B)",
                "Bituminous Tack Coat Emulsion (CRS-2)",
                "Crushed Aggregate Base (Grade D)",
            ] + (["Submersible Pump Sorbent Pack", "Hydraulic Quick-Set Cement"] if is_water else ["Joint Sealant Elastomer"]),
            "required_equipment": [
                "Vibratory Plate Compactor (15 kN)",
                "Diamond Pavement Saw",
                "High-Pressure Air Lance",
            ] + (["Submersible Dewatering Trash Pump"] if is_water else ["Infrared Pavement Heater"]),
            "safety_measures": "MUTCD Chapter 6H temporary traffic control taper.",
            "estimated_crew_size": 3 if sev_tier == "critical" else 2,
            "target_resolution_hours": 12 if sev_tier == "critical" else 24,
            "latitude": lat,
            "longitude": lon,
            "engine": "domain_rule_vlm_engine",
        }

    def seed_from_result(self, detections: List[Dict[str, Any]], work_orders: Optional[List[Dict[str, Any]]] = None) -> None:
        """Pre-populate from existing detections and work orders."""
        self._clusters.clear()
        self._work_orders.clear()
        for det in detections:
            cid = det.get("pothole_id") or det.get("defect_id") or str(uuid.uuid4())
            d = dict(det)
            d["all_bboxes"] = list(det.get("all_bboxes") or ([det["bbox_xyxy"]] if det.get("bbox_xyxy") else []))
            d["merged_count"] = det.get("merged_count", len(d["all_bboxes"]))
            self._clusters[cid] = d
            self._all_bboxes[cid] = list(d["all_bboxes"])
            self._cluster_sizes[cid] = d["merged_count"]
            self._observations[cid] = [d]

        if work_orders:
            for wo in work_orders:
                pid = wo.get("pothole_id") or wo.get("work_order_id")
                if pid:
                    self._work_orders[pid] = dict(wo)

        # Guarantee work orders exist for all clusters
        for cid, cluster_det in self._clusters.items():
            if cid not in self._work_orders:
                self._work_orders[cid] = self._generate_work_order(cid, cluster_det)

    def ingest(self, detection: Dict[str, Any], image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        lat = float(detection.get("latitude") or detection.get("lat") or 0.0)
        lon = float(detection.get("longitude") or detection.get("lon") or 0.0)
        incoming_bbox = detection.get("bbox_xyxy")
        incoming_conf = float(detection.get("pothole_confidence") or detection.get("confidence") or 0.0)

        for cid, canonical in self._clusters.items():
            c_lat = float(canonical.get("latitude") or canonical.get("lat") or 0.0)
            c_lon = float(canonical.get("longitude") or canonical.get("lon") or 0.0)
            dist = haversine_distance_m(lat, lon, c_lat, c_lon)
            if dist <= DEDUP_RADIUS_M:
                merged_bboxes = self._all_bboxes.setdefault(cid, [])
                if incoming_bbox:
                    merged_bboxes.append(incoming_bbox)
                self._cluster_sizes[cid] = self._cluster_sizes.get(cid, 1) + 1
                self._observations.setdefault(cid, []).append(dict(detection))

                stored_conf = float(canonical.get("pothole_confidence") or canonical.get("confidence") or 0.0)
                if incoming_conf > stored_conf:
                    prev_id = canonical.get("pothole_id") or cid
                    self._clusters[cid] = dict(detection)
                    self._clusters[cid]["pothole_id"] = prev_id
                    if image_bytes:
                        self._patch_images[cid] = image_bytes
                elif image_bytes and cid not in self._patch_images:
                    self._patch_images[cid] = image_bytes

                self._clusters[cid]["all_bboxes"] = merged_bboxes
                self._clusters[cid]["merged_count"] = len(merged_bboxes)
                return {
                    "status": "deduplicated",
                    "cluster_id": cid,
                    "distance_m": round(dist, 2),
                    "canonical": self._clusters[cid],
                    "total_merged_bboxes": len(merged_bboxes),
                    "total_work_orders": len(self._work_orders),
                }

        cid = detection.get("pothole_id") or detection.get("defect_id") or str(uuid.uuid4())
        while cid in self._clusters:
            cid = cid + "_" + uuid.uuid4().hex[:4]

        detection["all_bboxes"] = [incoming_bbox] if incoming_bbox else []
        detection["merged_count"] = 1
        self._observations[cid] = [dict(detection)]
        self._clusters[cid] = dict(detection)
        self._all_bboxes[cid] = list(detection["all_bboxes"])
        self._cluster_sizes[cid] = 1
        if image_bytes:
            self._patch_images[cid] = image_bytes

        # Create exactly 1 work order per unique defect cluster
        self._work_orders[cid] = self._generate_work_order(cid, detection)

        return {
            "status": "new",
            "cluster_id": cid,
            "canonical": self._clusters[cid],
            "total_merged_bboxes": len(self._all_bboxes[cid]),
            "total_work_orders": len(self._work_orders),
        }

    def get_all_detections(self) -> List[Dict[str, Any]]:
        return list(self._clusters.values())

    def get_patch_image(self, cluster_id: str) -> Optional[bytes]:
        return self._patch_images.get(cluster_id)

    def get_all_work_orders(self) -> List[Dict[str, Any]]:
        return list(self._work_orders.values())

    def cluster_count(self) -> int:
        return len(self._clusters)


_DEDUPLICATOR = SpatialDeduplicator()


def get_deduplicator() -> SpatialDeduplicator:
    return _DEDUPLICATOR


def get_spatial_index() -> DefectSpatialIndex:
    global _SPATIAL_INDEX
    if _SPATIAL_INDEX is None:
        res = get_latest_result_data()
        defs = res.get("detections") or res.get("potholes") or []
        _SPATIAL_INDEX = DefectSpatialIndex(defs)
    return _SPATIAL_INDEX


# ---------------------------------------------------------------------------
# API Routes — Health & Telemetry Data
# ---------------------------------------------------------------------------

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status": "online",
        "service": "RoadSentinel Government Analytics & Inference Gateway",
        "version": "3.2",
        "dedup_radius_m": DEDUP_RADIUS_M,
        "active_clusters": get_deduplicator().cluster_count(),
        "timestamp": utc_iso(),
    }


@app.get("/api/results")
def get_results():
    """Returns analytics result merged with live deduplicated detections."""
    data = get_latest_result_data()
    dedup = get_deduplicator()
    live = dedup.get_all_detections()
    if live:
        data = dict(data)
        data["detections"] = live
    return JSONResponse(data)


@app.get("/api/work_orders")
def get_work_orders():
    dedup = get_deduplicator()
    wos = dedup.get_all_work_orders()
    if not wos:
        wos = get_latest_work_orders_data().get("work_orders", [])
    return JSONResponse({"work_orders": wos, "total": len(wos)})


@app.get("/api/stats")
def get_stats():
    res = get_latest_result_data()
    meta = res.get("metadata", {})
    rh = res.get("road_health", {})
    pred = res.get("prediction", {})
    dedup = get_deduplicator()
    all_defs = dedup.get_all_detections() or res.get("detections") or res.get("potholes") or []
    wos = dedup.get_all_work_orders()
    wo_count = len(wos) if wos else len(res.get("work_orders", []))

    score = rh.get("road_health_score", 100.0)
    cond = rh.get("condition_class", "Good").upper()
    total_def = meta.get("total_defects_detected", len(all_defs))
    total_imgs = meta.get("total_images_processed", 1)
    elapsed_time = meta.get("elapsed_time_s", 1.24)
    conf_thresh = meta.get("confidence_threshold", 0.15)
    min_area_px = meta.get("min_area_px", 50)
    is_2d = res.get("is_2d_mode", True)

    water_hazards = sum(
        1 for d in all_defs if d.get("is_water_filled") or d.get("water_flag")
    )
    critical_def = sum(
        1 for d in all_defs
        if (d.get("severity_score") or 0.0) >= 0.85
        or (d.get("severity_tier") == "critical")
    )
    high_def = sum(
        1 for d in all_defs
        if 0.65 <= (d.get("severity_score") or 0.0) < 0.85
        or (d.get("severity_tier") == "high")
    )

    return {
        "road_health_score": round(score, 1),
        "condition_class": cond,
        "total_defects": total_def,
        "total_images_processed": total_imgs,
        "elapsed_time_s": elapsed_time,
        "confidence_threshold": conf_thresh,
        "min_area_px": min_area_px,
        "test_mode_2d": is_2d,
        "active_clusters": dedup.cluster_count() or total_def,
        "critical_hazards": critical_def,
        "high_hazards": high_def,
        "water_hazards": water_hazards,
        "work_orders_count": wo_count,
        "deterioration_probability": pred.get("deterioration_probability", 0.0),
        "prediction_horizon_days": pred.get("prediction_horizon_days", 30),
        "road_segment_id": res.get("road_segment_id", "SEG-2D-LOCAL-SURVEY"),
        "map_name": res.get("map_name", "2D Local Imagery"),
        "geolocation": res.get("geolocation", {"lat": 13.0827, "lon": 80.2707}),
        "dedup_radius_m": DEDUP_RADIUS_M,
    }


@app.post("/api/ingest")
async def ingest_detection(request: Request):
    """Accept detection payload and apply spatial deduplication."""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, f"Invalid JSON body: {exc}")

    lat = body.get("latitude") or body.get("lat")
    lon = body.get("longitude") or body.get("lon")
    if lat is None or lon is None:
        raise HTTPException(400, "latitude and longitude are required fields")

    image_bytes: Optional[bytes] = None
    b64_img = body.pop("_image_b64", None)
    if b64_img:
        try:
            image_bytes = base64.b64decode(b64_img)
        except Exception:
            pass

    if "timestamp" not in body:
        body["timestamp"] = utc_iso()

    dedup = get_deduplicator()
    result = dedup.ingest(body, image_bytes=image_bytes)

    global _SPATIAL_INDEX
    _SPATIAL_INDEX = DefectSpatialIndex(dedup.get_all_detections())

    return JSONResponse({
        "ok": True,
        "dedup_status": result["status"],
        "cluster_id": result["cluster_id"],
        "distance_m": result.get("distance_m"),
        "total_clusters": dedup.cluster_count(),
        "total_work_orders": len(dedup.get_all_work_orders()),
        "canonical": result["canonical"],
    })


@app.get("/api/drone_images")
def get_drone_images():
    """Return list of processed inspection images with metadata and detections."""
    images = get_all_drone_images_data()
    total_dets = sum(img["pothole_count"] for img in images)
    res_data = get_latest_result_data()
    meta = res_data.get("metadata", {})
    return JSONResponse({
        "total_images": len(images),
        "total_detections": total_dets,
        "test_mode_2d": res_data.get("is_2d_mode", True),
        "elapsed_time_s": meta.get("elapsed_time_s", 1.24),
        "confidence_threshold": meta.get("confidence_threshold", 0.15),
        "min_area_px": meta.get("min_area_px", 50),
        "images_dir": str(RESULTS_DIR),
        "images": images,
    })


@app.api_route("/api/drone_image/{filename}", methods=["GET", "HEAD"])
def get_drone_image_file(filename: str):
    """Serve original raw image file."""
    # Search in 2D image paths recorded in results
    res_data = get_latest_result_data()
    for r in res_data.get("raw_2d_results", []):
        if r.get("image_name") == filename:
            p = Path(r.get("image_path", ""))
            if p.is_file():
                return FileResponse(str(p), media_type="image/jpeg")

    # Search in known image directories
    for d in DRONE_IMAGE_DIRS:
        p = d / filename
        if p.is_file():
            return FileResponse(str(p), media_type="image/jpeg")
        # Check subdirectories
        for sub in ("images", "rgb", "test/images", "valid/images", "train/images"):
            sub_p = d / sub / filename
            if sub_p.is_file():
                return FileResponse(str(sub_p), media_type="image/jpeg")

    raise HTTPException(404, f"Image '{filename}' not found")


@app.api_route("/api/annotated_image/{filename}", methods=["GET", "HEAD"])
def get_annotated_drone_image(filename: str):
    """Serve the SAM2/OpenCV segmentation overlay image directly from results/."""
    stem = Path(filename).stem

    # Check for direct pre-rendered overlay files in results/
    for overlay_candidate in (
        RESULTS_DIR / f"{stem}_detection_overlay.jpg",
        RESULTS_DIR / f"{stem}_overlay.jpg",
        RESULTS_DIR / filename,
        ROOT / "results" / f"{stem}_detection_overlay.jpg",
    ):
        if overlay_candidate.is_file():
            return FileResponse(str(overlay_candidate), media_type="image/jpeg")

    # Check detection results metadata for overlay path
    res_data = get_latest_result_data()
    for r in res_data.get("raw_2d_results", []):
        if r.get("image_name") == filename:
            op = r.get("overlay_path")
            if op and Path(op).is_file():
                return FileResponse(str(op), media_type="image/jpeg")

    # Check indexed files in results/
    for f in RESULTS_DIR.glob("*_detection_overlay.jpg"):
        if f.is_file():
            return FileResponse(str(f), media_type="image/jpeg")

    # Fallback to dynamic overlay burning
    for d in DRONE_IMAGE_DIRS:
        cand = d / filename
        if cand.is_file():
            img_dets = [
                det for det in res_data.get("detections", [])
                if det.get("source_image") == filename
            ]
            annotated_bytes = annotate_full_drone_image(cand, img_dets, meta=res_data.get("metadata"))
            return Response(content=annotated_bytes, media_type="image/jpeg")

    raise HTTPException(404, f"Overlay for '{filename}' not found")


@app.api_route("/api/overlays/{filename}", methods=["GET", "HEAD"])
def get_overlay_image(filename: str):
    for d in DEMO_OUTPUT_DIRS:
        fpath = d / filename
        if fpath.is_file():
            return FileResponse(str(fpath))
    raise HTTPException(404, f"Overlay {filename} not found")


@app.api_route("/api/patch_image/{defect_id}", methods=["GET", "HEAD"])
def get_patch_image(defect_id: str):
    """Return patch crop or image for a specific defect ID."""
    res_data = get_latest_result_data()
    for det in res_data.get("detections", []):
        pid = det.get("pothole_id") or det.get("defect_id")
        if pid == defect_id:
            op = det.get("overlay_path")
            if op and Path(op).is_file():
                return FileResponse(str(op), media_type="image/jpeg")
            src_name = det.get("source_image")
            if src_name:
                for d in DRONE_IMAGE_DIRS:
                    cand = d / src_name
                    if cand.is_file():
                        return FileResponse(str(cand), media_type="image/jpeg")

    for f in RESULTS_DIR.glob("*_detection_overlay.jpg"):
        if f.is_file():
            return FileResponse(str(f), media_type="image/jpeg")

    raise HTTPException(404, f"Patch for {defect_id} not found")


@app.get("/api/geofence/zones")
def get_geofence_zones(radius_m: float = Query(50.0)):
    s_index = get_spatial_index()
    zones = s_index.create_geofence_zones(default_radius_m=radius_m)
    return JSONResponse({"total_zones": len(zones), "radius_m": radius_m, "zones": zones})


@app.get("/api/geofence/query")
def query_geofence(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_m: float = Query(50.0),
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
            _SPATIAL_INDEX = DefectSpatialIndex(result.potholes)
        except Exception as exc:
            log.exception("Inference failed:")
            raise HTTPException(500, str(exc))
        return JSONResponse(result.to_dict())


# ---------------------------------------------------------------------------
# Interactive Web Dashboard UI
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RoadSentinel — Government Infrastructure Intelligence & 2D Defect Analytics</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    :root {
      --bg-dark: #070a11;
      --bg-card: rgba(15, 23, 42, 0.85);
      --border-card: rgba(255,255,255,0.08);
      --border-accent: rgba(0,242,254,0.35);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --cyan: #00f2fe;
      --cyan-glow: rgba(0,242,254,0.4);
      --green: #10b981;
      --yellow: #f59e0b;
      --orange: #f97316;
      --red: #ef4444;
      --purple: #8b5cf6;
      --blue: #3b82f6;
      --radius: 16px;
      --shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-dark);
      background-image:
        radial-gradient(at 0% 0%, rgba(0,242,254,0.08) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(139,92,246,0.08) 0px, transparent 50%);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 32px;
      background: rgba(7,10,17,0.94);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-card);
      position: sticky;
      top: 0;
      z-index: 1000;
      gap: 16px;
      flex-wrap: wrap;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon {
      width: 40px; height: 40px;
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 800; font-size: 19px; color: #000;
      box-shadow: 0 0 20px var(--cyan-glow);
      flex-shrink: 0;
    }
    .brand-title { font-size: 20px; font-weight: 800; letter-spacing: -0.5px;
      background: linear-gradient(to right, #fff, #cbd5e1);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .brand-subtitle { font-size: 11px; color: var(--cyan); font-weight: 600;
      letter-spacing: 1px; text-transform: uppercase; }
    .header-right { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    
    .badge {
      padding: 5px 12px; border-radius: 999px; font-size: 11px; font-weight: 700;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .badge-live { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: var(--green); }
    .badge-live::before {
      content: ""; width: 7px; height: 7px; background: var(--green);
      border-radius: 50%; box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite;
    }
    .badge-mode {
      background: rgba(0,242,254,0.12); border: 1px solid var(--border-accent);
      color: var(--cyan); font-family: 'JetBrains Mono', monospace; font-size: 11px;
    }
    .badge-time {
      background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.35);
      color: var(--purple); font-family: 'JetBrains Mono', monospace; font-size: 11px;
    }

    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.3)} }

    .btn-refresh {
      background: linear-gradient(135deg, rgba(0,242,254,0.2), rgba(139,92,246,0.2));
      border: 1px solid var(--border-accent); color: #fff;
      padding: 8px 16px; border-radius: 10px; font-weight: 700; font-size: 12px;
      cursor: pointer; transition: all .2s ease; display: flex; align-items: center; gap: 6px;
    }
    .btn-refresh:hover { background: linear-gradient(135deg,rgba(0,242,254,.35),rgba(139,92,246,.35)); box-shadow: 0 0 15px var(--cyan-glow); }

    .btn-upload {
      background: linear-gradient(135deg, rgba(0,242,254,0.22), rgba(16,185,129,0.22));
      border: 1px solid var(--cyan); color: var(--cyan);
      padding: 8px 16px; border-radius: 10px; font-weight: 700; font-size: 12px;
      cursor: pointer; transition: all .2s ease; display: flex; align-items: center; gap: 6px;
    }
    .btn-upload:hover { background: rgba(0,242,254,.35); box-shadow: 0 0 15px var(--cyan-glow); color: #fff; }

    /* ── Tab Nav ── */
    .tab-nav {
      display: flex; gap: 4px;
      padding: 0 32px;
      background: rgba(7,10,17,0.88);
      border-bottom: 1px solid var(--border-card);
      position: sticky; top: 68px; z-index: 900;
    }
    .tab-nav-btn {
      padding: 13px 22px; font-size: 13px; font-weight: 600;
      color: var(--text-muted); background: none; border: none;
      border-bottom: 2px solid transparent; cursor: pointer;
      transition: all .2s; white-space: nowrap;
      display: flex; align-items: center; gap: 8px;
    }
    .tab-nav-btn:hover { color: #fff; }
    .tab-nav-btn.active { color: var(--cyan); border-bottom-color: var(--cyan); }

    /* ── Main Layout ── */
    main { padding: 26px 32px; max-width: 1700px; margin: 0 auto; width: 100%; flex: 1; }
    .tab-panel { display: none; flex-direction: column; gap: 22px; animation: fadein .25s ease; }
    .tab-panel.active { display: flex; }
    @keyframes fadein { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:none} }

    /* ── KPI Grid ── */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px,1fr)); gap: 16px; }
    .kpi-card {
      background: var(--bg-card); backdrop-filter: blur(14px);
      border: 1px solid var(--border-card); border-radius: var(--radius);
      padding: 20px 22px; box-shadow: var(--shadow); position: relative; overflow: hidden;
      transition: transform .2s;
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,.15); }
    .kpi-card::before {
      content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(to right, var(--cyan), var(--purple));
    }
    .kpi-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .kpi-title { font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; }
    .kpi-value { font-size: 32px; font-weight: 800; letter-spacing: -1px; display: flex; align-items: baseline; gap: 5px; }
    .kpi-desc { font-size: 11px; color: var(--text-muted); margin-top: 5px; }

    /* ── Panel Card ── */
    .panel {
      background: var(--bg-card); backdrop-filter: blur(14px);
      border: 1px solid var(--border-card); border-radius: var(--radius);
      padding: 22px; box-shadow: var(--shadow);
      display: flex; flex-direction: column; gap: 16px;
    }
    .panel-header {
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--border-card); padding-bottom: 12px;
      flex-wrap: wrap; gap: 10px;
    }
    .panel-title { font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 9px; }

    .two-col { display: grid; grid-template-columns: 1.1fr .9fr; gap: 22px; }
    @media (max-width: 1100px) { .two-col { grid-template-columns: 1fr; } }

    /* ── Survey Toolbar ── */
    .survey-toolbar {
      display: flex; align-items: center; justify-content: space-between;
      background: rgba(15,23,42,.92); backdrop-filter: blur(12px);
      border: 1px solid var(--border-card); border-radius: var(--radius);
      padding: 12px 20px; gap: 14px; flex-wrap: wrap;
    }
    .survey-nav-group { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .btn-survey-nav {
      background: rgba(255,255,255,0.06); border: 1px solid var(--border-card);
      color: #fff; padding: 7px 15px; border-radius: 9px; font-weight: 600; font-size: 13px;
      cursor: pointer; transition: all .2s; display: flex; align-items: center; gap: 6px;
    }
    .btn-survey-nav:hover { background: rgba(0,242,254,0.18); border-color: var(--cyan); color: var(--cyan); }
    .btn-survey-nav:disabled { opacity: 0.35; cursor: not-allowed; border-color: transparent; }
    .select-survey-photo {
      background: #090e17; border: 1px solid var(--border-card); color: #f1f5f9;
      padding: 7px 14px; border-radius: 9px; font-size: 13px; font-family: 'JetBrains Mono', monospace;
      outline: none; cursor: pointer; transition: border .2s;
    }
    .select-survey-photo:focus { border-color: var(--cyan); }
    .survey-toggle-group { display: flex; background: rgba(0,0,0,0.4); border-radius: 9px; padding: 3px; border: 1px solid var(--border-card); }
    .btn-toggle-view {
      background: none; border: none; color: var(--text-muted); font-size: 12px; font-weight: 700;
      padding: 6px 14px; border-radius: 7px; cursor: pointer; transition: all .2s;
    }
    .btn-toggle-view.active {
      background: linear-gradient(135deg, rgba(0,242,254,0.25), rgba(139,92,246,0.25));
      color: #fff; border: 1px solid var(--border-accent); box-shadow: 0 0 12px var(--cyan-glow);
    }

    /* ── Image Viewer & Inspector ── */
    .survey-inspector-grid {
      display: grid; grid-template-columns: 1.25fr 0.75fr; gap: 22px;
    }
    @media (max-width: 1200px) { .survey-inspector-grid { grid-template-columns: 1fr; } }

    .survey-photo-viewer {
      background: #020617; border: 1px solid var(--border-card); border-radius: var(--radius);
      overflow: hidden; position: relative; display: flex; flex-direction: column; min-height: 480px;
      box-shadow: var(--shadow);
    }
    .survey-photo-header {
      padding: 12px 18px; background: rgba(7,10,17,0.9); border-bottom: 1px solid var(--border-card);
      display: flex; justify-content: space-between; align-items: center; font-size: 12px;
    }
    .survey-photo-canvas-wrap {
      flex: 1; display: flex; align-items: center; justify-content: center; position: relative;
      background: radial-gradient(circle at center, #0f172a 0%, #020617 100%);
      padding: 14px; overflow: hidden; min-height: 420px;
    }
    .survey-photo-canvas-wrap img {
      max-width: 100%; max-height: 580px; object-fit: contain; border-radius: 8px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.6); transition: opacity .2s ease;
    }
    .photo-overlay-tag {
      position: absolute; bottom: 20px; left: 20px; background: rgba(7,10,17,0.9);
      backdrop-filter: blur(10px); border: 1px solid var(--border-accent);
      padding: 6px 14px; border-radius: 8px; font-size: 11px; font-family: 'JetBrains Mono', monospace;
      color: var(--cyan);
    }

    .survey-telemetry-panel {
      display: flex; flex-direction: column; gap: 16px;
    }
    .telemetry-card {
      background: var(--bg-card); border: 1px solid var(--border-card); border-radius: var(--radius);
      padding: 18px 20px; box-shadow: var(--shadow);
    }
    .telemetry-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; margin-top: 12px;
    }
    .telemetry-item { display: flex; flex-direction: column; gap: 2px; }
    .telemetry-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    .telemetry-val { font-size: 13px; color: #f8fafc; font-family: 'JetBrains Mono', monospace; font-weight: 700; }

    .defect-list-wrap {
      display: flex; flex-direction: column; gap: 10px; max-height: 380px; overflow-y: auto; padding-right: 4px;
    }
    .defect-item-card {
      background: rgba(255,255,255,0.03); border: 1px solid var(--border-card);
      border-radius: 10px; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px;
      transition: all .2s ease;
    }
    .defect-item-card:hover {
      background: rgba(255,255,255,0.06); border-color: rgba(0,242,254,0.4);
    }
    .defect-item-header {
      display: flex; justify-content: space-between; align-items: center;
    }

    /* ── Thumbnail Gallery Strip ── */
    .thumbnail-gallery-wrap {
      background: var(--bg-card); border: 1px solid var(--border-card); border-radius: var(--radius);
      padding: 16px 20px; box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 12px;
    }
    .thumbnail-gallery-strip {
      display: flex; gap: 14px; overflow-x: auto; padding-bottom: 8px; scroll-behavior: smooth;
    }
    .thumb-card {
      flex: 0 0 160px; background: rgba(0,0,0,0.4); border: 2px solid transparent;
      border-radius: 10px; overflow: hidden; cursor: pointer; transition: all .2s ease;
      display: flex; flex-direction: column; position: relative;
    }
    .thumb-card:hover { transform: translateY(-3px); border-color: rgba(0,242,254,0.5); }
    .thumb-card.active { border-color: var(--cyan); box-shadow: 0 0 16px var(--cyan-glow); }
    .thumb-img { width: 100%; height: 95px; object-fit: cover; background: #060a12; }
    .thumb-info {
      padding: 6px 8px; font-size: 11px; display: flex; justify-content: space-between; align-items: center;
      background: rgba(7,10,17,0.85);
    }
    .thumb-name { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #cbd5e1; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100px; }
    .thumb-badge {
      font-size: 9px; font-weight: 800; padding: 2px 6px; border-radius: 999px;
    }
    .thumb-badge-clean { background: rgba(16,185,129,0.2); color: var(--green); }
    .thumb-badge-defect { background: rgba(249,115,22,0.2); color: var(--orange); }

    /* ── Patch Inspection & Work Orders ── */
    .patch-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px; }
    .patch-card {
      background: rgba(255,255,255,.03); border: 1px solid var(--border-card);
      border-radius: 14px; overflow: hidden; display: flex; flex-direction: column;
    }
    .patch-img-wrap { position: relative; width: 100%; background: #020617; min-height: 200px;
      display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .patch-img-wrap img { max-width: 100%; max-height: 260px; object-fit: contain; display: block; }
    .patch-meta { padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; }
    .patch-id { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--cyan); font-weight: 700; }
    .patch-defect-badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .5px; }

    .wo-table-wrap { overflow-x: auto; }
    .wo-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .wo-table th { text-align: left; color: var(--text-muted); font-size: 11px; font-weight: 700;
      padding: 10px 12px; text-transform: uppercase; border-bottom: 1px solid var(--border-card);
      white-space: nowrap; }
    .wo-table td { padding: 11px 12px; border-bottom: 1px solid rgba(255,255,255,.04); vertical-align: top; }
    .wo-table tr:hover td { background: rgba(255,255,255,.02); }
    .wo-id-cell { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--cyan); font-weight: 700; }
    .sev-badge { display: inline-block; padding: 3px 9px; border-radius: 6px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .sev-critical { background: rgba(239,68,68,.18); color: var(--red); border: 1px solid rgba(239,68,68,.35); }
    .sev-high { background: rgba(249,115,22,.15); color: var(--orange); border: 1px solid rgba(249,115,22,.35); }
    .sev-medium { background: rgba(245,158,11,.15); color: var(--yellow); border: 1px solid rgba(245,158,11,.35); }
    .sev-low { background: rgba(16,185,129,.12); color: var(--green); border: 1px solid rgba(16,185,129,.3); }
    .wo-text-cell { max-width: 340px; font-size: 12px; line-height: 1.55; color: #cbd5e1; }
    .materials-list { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
    .mat-tag { background: rgba(255,255,255,.06); padding: 3px 8px; border-radius: 5px; font-size: 10px; color: #94a3b8; }

    .score-breakdown { display: flex; flex-direction: column; gap: 8px;
      padding: 13px 15px; background: rgba(255,255,255,.02); border-radius: 10px; border: 1px solid var(--border-card); }
    .bar-row { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted); }
    .progress-track { height: 7px; background: rgba(255,255,255,.08); border-radius: 999px; overflow: hidden; margin-top: 4px; }
    .progress-fill { height: 100%; border-radius: 999px; transition: width .6s ease; }

    #map-container { height: 430px; border-radius: 12px; overflow: hidden;
      border: 1px solid var(--border-card); background: #060a12; }
    .map-geofence-controls {
      background: rgba(15,23,42,.9); backdrop-filter: blur(12px);
      border: 1px solid var(--border-card); border-radius: 10px;
      padding: 10px 16px; display: flex; align-items: center; justify-content: space-between; gap: 14px;
    }
    .geofence-slider-wrap { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-muted); flex: 1; }
    .geofence-slider-wrap input { flex: 1; accent-color: var(--cyan); cursor: pointer; }
    .btn-sim { background: rgba(0,242,254,.15); border: 1px solid var(--cyan); color: var(--cyan);
      padding: 5px 12px; border-radius: 8px; font-size: 11px; font-weight: 700; cursor: pointer; }

    #proximity-banner {
      background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.3);
      border-radius: 10px; padding: 12px 18px; display: flex; align-items: center;
      justify-content: space-between; gap: 14px;
    }
    #proximity-banner.warning { background: rgba(245,158,11,.13); border-color: rgba(245,158,11,.5); }
    #proximity-banner.critical { background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.6); }

    footer {
      padding: 14px 32px; border-top: 1px solid var(--border-card);
      display: flex; justify-content: space-between; align-items: center;
      font-size: 11px; color: var(--text-muted); background: rgba(7,10,17,.92); flex-wrap: wrap; gap: 8px;
    }
    .empty-state { text-align: center; padding: 35px; color: var(--text-muted); font-size: 13px; }
  </style>
</head>
<body>

<!-- ── Header ── -->
<header>
  <div class="brand">
    <div class="brand-icon">RS</div>
    <div>
      <div class="brand-title">RoadSentinel</div>
      <div class="brand-subtitle">Government Infrastructure &amp; 2D Vision Intelligence</div>
    </div>
  </div>
  <div class="header-right">
    <span class="badge badge-live">LIVE 2D SYNC</span>
    <span class="badge badge-mode" id="header-mode-badge">2D TEST MODE</span>
    <span class="badge badge-time" id="header-time-badge">⚡ 1.24s</span>
    <span style="font-size:11px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;" id="clock-display">--:--:-- UTC</span>
    <button class="btn-refresh" onclick="fetchAll(true)">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
      Refresh
    </button>
  </div>
</header>

<!-- ── Tab Navigation ── -->
<nav class="tab-nav">
  <button class="tab-nav-btn active" onclick="switchTab('survey')" id="tab-btn-survey">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
    2D Detection &amp; Survey Overlays (<span id="tab-photo-count">1</span>)
  </button>
  <button class="tab-nav-btn" onclick="switchTab('map')" id="tab-btn-map">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>
    GIS Map View
  </button>
  <button class="tab-nav-btn" onclick="switchTab('patches')" id="tab-btn-patches">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
    Defect Patch Analytics
  </button>
  <button class="tab-nav-btn" onclick="switchTab('workorders')" id="tab-btn-workorders">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
    Maintenance Work Orders
  </button>
</nav>

<!-- ── Main Content ── -->
<main>

  <!-- ════════════ TAB 1: 2D SURVEY & DEFECT OVERLAYS ════════════ -->
  <div class="tab-panel active" id="tab-survey">

    <!-- KPI Strip -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Surveyed 2D Photos</span><span class="badge badge-live">results/</span></div>
        <div class="kpi-value" id="kpi-survey-photos">1</div>
        <div class="kpi-desc">Parsed from <code>detection_results.json</code></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Total Defects Found</span></div>
        <div class="kpi-value" style="color:var(--orange);" id="kpi-survey-dets">20</div>
        <div class="kpi-desc" id="kpi-speed-sub">⚡ Elapsed: 1.24s (0.06s/defect)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Current Photo Defects</span></div>
        <div class="kpi-value" style="color:var(--cyan);" id="kpi-current-photo-dets">20</div>
        <div class="kpi-desc" id="kpi-current-photo-status">DINOv2 Conf ≥ 0.15 | Area ≥ 50px</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Road Health Index</span><span class="badge" id="kpi-survey-cond-badge" style="background:rgba(239,68,68,.15);color:var(--orange);">POOR</span></div>
        <div class="kpi-value" id="kpi-survey-score">38.5<span style="font-size:15px;color:var(--text-muted);font-weight:500;">/100</span></div>
        <div class="kpi-desc" id="kpi-survey-seg">Segment: SEG-2D-LOCAL-SURVEY</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Work Orders Active</span></div>
        <div class="kpi-value" style="color:var(--purple);" id="kpi-survey-wo">16</div>
        <div class="kpi-desc">Critical &amp; High Priority Dispatches</div>
      </div>
    </div>

    <!-- Toolbar -->
    <div class="survey-toolbar">
      <div class="survey-nav-group">
        <button class="btn-survey-nav" id="btn-prev-photo" onclick="prevSurveyPhoto()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg>
          Previous Photo
        </button>
        <select class="select-survey-photo" id="select-survey-photo" onchange="onSelectSurveyPhoto(this.value)">
          <option value="0">Loading survey photos…</option>
        </select>
        <button class="btn-survey-nav" id="btn-next-photo" onclick="nextSurveyPhoto()">
          Next Photo
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
        </button>
        <span style="font-size:12px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;" id="survey-counter-text">Photo 1 of 1</span>
      </div>

      <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
        <span id="current-photo-defect-badge" class="badge" style="background:rgba(249,115,22,.15);color:var(--orange);border:1px solid rgba(249,115,22,.35);">
          Loading…
        </span>
        <div class="survey-toggle-group">
          <button class="btn-toggle-view active" id="btn-toggle-annotated" onclick="setSurveyViewMode('annotated')">
            🎯 SAM2 Overlay &amp; BBoxes
          </button>
          <button class="btn-toggle-view" id="btn-toggle-raw" onclick="setSurveyViewMode('raw')">
            🖼 Original 2D Photo
          </button>
        </div>
      </div>
    </div>

    <!-- Inspector Main View (Image Viewer + Telemetry & Assessment) -->
    <div class="survey-inspector-grid">
      <!-- Left: High-Resolution Photo Viewer -->
      <div class="survey-photo-viewer">
        <div class="survey-photo-header">
          <div style="display:flex;align-items:center;gap:8px;">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#fff;" id="survey-viewer-filename">Loading…</span>
          </div>
          <span style="font-size:11px;color:var(--text-muted);" id="survey-viewer-mode-label">Mode: SAM2 Segmentation &amp; BBox Overlay (Hotkeys: ← / → / O)</span>
        </div>
        <div class="survey-photo-canvas-wrap">
          <img id="survey-main-image" src="" alt="Detection Overlay" onerror="this.alt='Loading overlay image...'" />
          <div class="photo-overlay-tag" id="survey-photo-tag">Photo 1 • SAM2 Alpha Overlay</div>
        </div>
      </div>

      <!-- Right: Telemetry & Assessment Details Panel -->
      <div class="survey-telemetry-panel">
        <!-- Telemetry Card -->
        <div class="telemetry-card">
          <div class="panel-header" style="border-bottom:1px solid var(--border-card);padding-bottom:10px;">
            <div class="panel-title" style="font-size:14px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
              2D Vision Telemetry &amp; Parameters
            </div>
            <span style="font-size:11px;color:var(--cyan);font-family:'JetBrains Mono',monospace;" id="telemetry-simtime">TEST_MODE_2D</span>
          </div>
          <div class="telemetry-grid">
            <div class="telemetry-item">
              <span class="telemetry-label">Image Name</span>
              <span class="telemetry-val" id="tel-filename">—</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Resolution</span>
              <span class="telemetry-val" id="tel-resolution">—</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">DINOv2 Conf Thresh</span>
              <span class="telemetry-val" style="color:var(--cyan);" id="tel-conf-thresh">≥ 0.15</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">SAM2 Min Area</span>
              <span class="telemetry-val" style="color:var(--purple);" id="tel-min-area">≥ 50 px</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Inference Time</span>
              <span class="telemetry-val" id="tel-elapsed">1.24s</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Backbone</span>
              <span class="telemetry-val">DINOv2 + SAM2.1</span>
            </div>
          </div>
        </div>

        <!-- Defect Summary Card -->
        <div class="telemetry-card">
          <div class="panel-header" style="border-bottom:1px solid var(--border-card);padding-bottom:10px;">
            <div class="panel-title" style="font-size:14px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--purple)" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
              Photo Defect Summary
            </div>
            <span class="badge" id="survey-assess-badge" style="background:rgba(239,68,68,.18);color:var(--red);">High Severity</span>
          </div>
          <div class="telemetry-grid" style="grid-template-columns:1fr 1fr 1fr;margin-top:10px;">
            <div class="telemetry-item">
              <span class="telemetry-label">Defects</span>
              <span class="telemetry-val" id="assess-potholes-cnt">0</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Max Severity</span>
              <span class="telemetry-val" id="assess-max-sev">0%</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Damaged Area</span>
              <span class="telemetry-val" id="assess-total-area">0 px</span>
            </div>
          </div>
          <div style="margin-top:12px;font-size:12px;" id="assess-water-note">
            <span style="color:var(--text-muted);">Pavement maintenance recommended.</span>
          </div>
        </div>

        <!-- Itemized Defect List -->
        <div class="telemetry-card" style="flex:1;">
          <div class="panel-header" style="border-bottom:1px solid var(--border-card);padding-bottom:10px;">
            <div class="panel-title" style="font-size:14px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--orange)" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              Detected Potholes in This Photo (<span id="current-photo-defect-items-cnt">0</span>)
            </div>
          </div>
          <div class="defect-list-wrap" id="current-photo-defect-list">
            <div class="empty-state" style="padding:20px 0;">
              <div class="icon" style="font-size:24px;margin-bottom:6px;">✓</div>
              No defects detected.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Thumbnail Strip -->
    <div class="thumbnail-gallery-wrap">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:13px;font-weight:700;display:flex;align-items:center;gap:7px;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
          Captured Survey Gallery (Click thumbnail to inspect)
        </span>
        <span style="font-size:11px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;">Source: results/detection_results.json</span>
      </div>
      <div class="thumbnail-gallery-strip" id="survey-thumbnail-strip"></div>
    </div>

  </div>

  <!-- ════════════ TAB 2: MAP VIEW ════════════ -->
  <div class="tab-panel" id="tab-map">
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Road Health Index</span><span class="badge" id="kpi-cond-badge" style="background:rgba(239,68,68,.15);color:var(--orange);">POOR</span></div>
        <div class="kpi-value" id="kpi-score">—<span style="font-size:15px;color:var(--text-muted);font-weight:500;">/100</span></div>
        <div class="kpi-desc" id="kpi-segment-id">Segment: —</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Detected Defects</span></div>
        <div class="kpi-value" id="kpi-defects">—</div>
        <div class="kpi-desc">Active localized clusters</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Water Hazards</span></div>
        <div class="kpi-value" style="color:#38bdf8;" id="kpi-water">0</div>
        <div class="kpi-desc">Hydroplaning Risk</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Work Orders</span></div>
        <div class="kpi-value" style="color:var(--purple);" id="kpi-workorders">—</div>
        <div class="kpi-desc">Actionable dispatches</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">30-Day Deterioration Risk</span></div>
        <div class="kpi-value" style="color:var(--red);" id="kpi-pred">—<span style="font-size:15px;font-weight:500;">%</span></div>
        <div class="kpi-desc">Forecasted failure probability</div>
      </div>
    </div>

    <div id="proximity-banner">
      <div style="font-size:18px;">🛡️</div>
      <div class="alert-text" id="alert-text-content"><b>Driver Proximity HUD:</b> Geofencing active.</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan);" id="nearest-hazard-dist">Corridor Safe</div>
    </div>

    <div class="two-col">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <svg width="16" height="16" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>
            Georeferenced Defect Map
          </div>
          <span style="font-size:11px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;" id="gps-coords">13.0827° N, 80.2707° E</span>
        </div>
        <div id="map-container"></div>
        <div class="map-geofence-controls">
          <div class="geofence-slider-wrap">
            <span>Geofence Radius:</span>
            <input type="range" id="geofence-radius-slider" min="10" max="120" value="50" step="5" oninput="updateGeofenceRadius(this.value)">
            <span id="radius-val" style="font-weight:700;color:var(--cyan);min-width:38px;">50m</span>
          </div>
          <button class="btn-sim" onclick="toggleDriveSimulation()"><span id="sim-btn-text">▶ Simulate Driver</span></button>
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:18px;">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="16" height="16" fill="none" stroke="var(--purple)" stroke-width="2" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
              Road Health Index Breakdown
            </div>
          </div>
          <div class="score-breakdown">
            <div class="bar-row"><span>Pothole Count Penalty</span><span id="cnt-penalty-txt">—</span></div>
            <div class="progress-track"><div class="progress-fill" id="cnt-penalty-bar" style="width:0%;background:var(--yellow);"></div></div>
            <div class="bar-row" style="margin-top:8px;"><span>Severity Deduction</span><span id="sev-penalty-txt">—</span></div>
            <div class="progress-track"><div class="progress-fill" id="sev-penalty-bar" style="width:0%;background:var(--orange);"></div></div>
            <div class="bar-row" style="margin-top:8px;"><span>Water Hazard Deduction</span><span id="water-penalty-txt">—</span></div>
            <div class="progress-track"><div class="progress-fill" id="water-penalty-bar" style="width:0%;background:#38bdf8;"></div></div>
          </div>
          <div style="font-size:12px;color:#cbd5e1;line-height:1.5;background:rgba(255,255,255,.02);padding:12px;border-radius:8px;" id="explanation-text">
            —
          </div>
        </div>

        <div class="panel" style="flex:1;">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="16" height="16" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/></svg>
              30-Day Deterioration Forecast
            </div>
            <span style="font-size:10px;color:var(--text-muted);">2D ZERO-SHOT INFERENCE</span>
          </div>
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:13px;color:var(--text-muted);">Deterioration Probability</span>
              <span style="font-size:20px;font-weight:800;color:var(--red);" id="detp-score">—%</span>
            </div>
            <div class="progress-track"><div class="progress-fill" id="detp-bar" style="width:0%;background:var(--red);"></div></div>
            <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-muted);">
              <span>Horizon: <b id="horizon-days" style="color:#e2e8f0;">30</b> days</span>
              <span id="prog-direction" style="font-weight:700;color:var(--red);">CRITICAL</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ════════════ TAB 3: PATCH INSPECTION ════════════ -->
  <div class="tab-panel" id="tab-patches">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="16" height="16" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          Patch Inspection — Bounding Boxes, Mask Area &amp; Confidence
        </div>
        <span style="font-size:12px;color:var(--text-muted);" id="patch-count-badge">Loading…</span>
      </div>
    </div>
    <div class="patch-grid" id="patch-grid">
      <div class="empty-state">Loading patches…</div>
    </div>
  </div>

  <!-- ════════════ TAB 4: WORK ORDERS ════════════ -->
  <div class="tab-panel" id="tab-workorders">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="16" height="16" fill="none" stroke="var(--green)" stroke-width="2" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          Maintenance &amp; Repair Work Orders
        </div>
        <span class="badge" style="background:rgba(16,185,129,.13);color:var(--green);" id="wo-total-badge">— Active</span>
      </div>
    </div>
    <div class="panel" style="padding:0;overflow:hidden;">
      <div class="wo-table-wrap">
        <table class="wo-table" id="wo-table">
          <thead>
            <tr>
              <th>Work Order ID</th>
              <th>Defect Class</th>
              <th>Severity</th>
              <th>Area</th>
              <th>Crew</th>
              <th>Target (h)</th>
              <th>Repair Description</th>
              <th>Required Materials</th>
            </tr>
          </thead>
          <tbody id="wo-tbody">
            <tr><td colspan="8" class="empty-state">Loading work orders…</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

</main>

<!-- ── Footer ── -->
<footer>
  <div>RoadSentinel v3.2 • 2D Pothole Detection &amp; SAM2/DINOv2 Pipeline Sync</div>
  <div style="font-family:'JetBrains Mono',monospace;">PyTorch • DINOv2 • SAM2.1 • FastAPI</div>
</footer>

<script>
function tick() {
  const n = new Date();
  document.getElementById('clock-display').innerText = n.toUTCString().split(' ')[4] + ' UTC';
}
setInterval(tick, 1000); tick();

function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-nav-btn').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById('tab-' + name);
  const btn = document.getElementById('tab-btn-' + name);
  if (panel) panel.classList.add('active');
  if (btn) btn.classList.add('active');
  if (name === 'map') {
    setTimeout(() => {
      map.invalidateSize();
      if (cachedDefects.length > 0) renderDefects(cachedDefects);
    }, 150);
  }
}

let surveyImages = [];
let currentSurveyIdx = 0;
let currentViewMode = 'annotated';
let cachedDefects = [];

async function fetchDroneSurveyImages() {
  try {
    const res = await fetch('/api/drone_images');
    const data = await res.json();
    surveyImages = data.images || [];

    document.getElementById('kpi-survey-photos').innerText = surveyImages.length;
    document.getElementById('kpi-survey-dets').innerText = data.total_detections || 0;
    document.getElementById('tab-photo-count').innerText = surveyImages.length;
    document.getElementById('header-time-badge').innerText = `⚡ ${data.elapsed_time_s || 1.24}s`;
    document.getElementById('kpi-speed-sub').innerText = `⚡ Elapsed: ${data.elapsed_time_s || 1.24}s (${(data.elapsed_time_s / Math.max(1, data.total_detections)).toFixed(2)}s/defect)`;

    renderSurveyDropdown();
    renderThumbnailGallery();
    if (surveyImages.length > 0) {
      if (currentSurveyIdx >= surveyImages.length) currentSurveyIdx = 0;
      displaySurveyPhoto(currentSurveyIdx);
    }
  } catch (err) {
    console.error('Failed fetching survey images:', err);
  }
}

function renderSurveyDropdown() {
  const sel = document.getElementById('select-survey-photo');
  sel.innerHTML = '';
  surveyImages.forEach((img, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    const defCount = img.pothole_count;
    const defLabel = defCount === 0 ? 'Nominal' : `${defCount} defect${defCount !== 1 ? 's' : ''}`;
    opt.innerText = `Photo #${idx + 1}: ${img.filename} (${defLabel})`;
    sel.appendChild(opt);
  });
}

function renderThumbnailGallery() {
  const strip = document.getElementById('survey-thumbnail-strip');
  strip.innerHTML = '';
  surveyImages.forEach((img, idx) => {
    const card = document.createElement('div');
    card.className = 'thumb-card' + (idx === currentSurveyIdx ? ' active' : '');
    card.id = `thumb-card-${idx}`;
    card.onclick = () => {
      currentSurveyIdx = idx;
      displaySurveyPhoto(idx);
    };

    const isClean = img.pothole_count === 0;
    const badgeCls = isClean ? 'thumb-badge-clean' : 'thumb-badge-defect';
    const badgeText = isClean ? 'CLEAN' : `${img.pothole_count} DEF`;

    card.innerHTML = `
      <img class="thumb-img" src="${img.annotated_url || img.image_url}" alt="${img.filename}" loading="lazy" />
      <div class="thumb-info">
        <span class="thumb-name">${img.filename}</span>
        <span class="thumb-badge ${badgeCls}">${badgeText}</span>
      </div>
    `;
    strip.appendChild(card);
  });
}

function displaySurveyPhoto(idx) {
  if (!surveyImages || surveyImages.length === 0 || idx < 0 || idx >= surveyImages.length) return;
  currentSurveyIdx = idx;
  const imgData = surveyImages[idx];

  const sel = document.getElementById('select-survey-photo');
  if (sel) sel.value = idx;
  document.getElementById('btn-prev-photo').disabled = (idx === 0);
  document.getElementById('btn-next-photo').disabled = (idx === surveyImages.length - 1);
  document.getElementById('survey-counter-text').innerText = `Photo ${idx + 1} of ${surveyImages.length}`;

  document.querySelectorAll('.thumb-card').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  const imgEl = document.getElementById('survey-main-image');
  imgEl.src = currentViewMode === 'annotated' ? imgData.annotated_url : imgData.image_url;
  document.getElementById('survey-viewer-filename').innerText = imgData.filename;
  document.getElementById('survey-photo-tag').innerText = `Photo ${idx + 1} of ${surveyImages.length} • ${imgData.filename}`;

  const defBadge = document.getElementById('current-photo-defect-badge');
  if (imgData.pothole_count === 0) {
    defBadge.className = 'badge';
    defBadge.style.background = 'rgba(16,185,129,0.15)';
    defBadge.style.color = 'var(--green)';
    defBadge.style.borderColor = 'rgba(16,185,129,0.35)';
    defBadge.innerHTML = '✓ Nominal Surface (0 Defects)';
  } else {
    defBadge.className = 'badge';
    defBadge.style.background = 'rgba(239,68,68,0.18)';
    defBadge.style.color = 'var(--red)';
    defBadge.style.borderColor = 'rgba(239,68,68,0.4)';
    defBadge.innerHTML = `⚠ ${imgData.pothole_count} Crater/Pothole Defect${imgData.pothole_count > 1 ? 's' : ''} Identified • Max Sev: ${(imgData.max_severity_score*100).toFixed(0)}%`;
  }

  document.getElementById('kpi-current-photo-dets').innerText = imgData.pothole_count;
  document.getElementById('kpi-current-photo-status').innerText = `DINOv2 Conf ≥ 0.15 | Area ≥ 50px`;

  const m = imgData.metadata || {};
  document.getElementById('tel-filename').innerText = imgData.filename;
  document.getElementById('tel-resolution').innerText = m.resolution || '691x385';
  document.getElementById('tel-conf-thresh').innerText = `≥ ${m.confidence_threshold || 0.15}`;
  document.getElementById('tel-min-area').innerText = `≥ ${m.min_area_px || 50} px`;
  document.getElementById('tel-elapsed').innerText = `${m.elapsed_time_s || 1.24}s`;

  document.getElementById('assess-potholes-cnt').innerText = imgData.pothole_count;
  document.getElementById('assess-max-sev').innerText = `${(imgData.max_severity_score*100).toFixed(0)}%`;
  document.getElementById('assess-total-area').innerText = `${imgData.total_area_px || 0} px (${imgData.total_area_m2 || 0} m²)`;

  const defList = document.getElementById('current-photo-defect-list');
  document.getElementById('current-photo-defect-items-cnt').innerText = imgData.pothole_count;
  defList.innerHTML = '';

  if (imgData.pothole_count === 0) {
    defList.innerHTML = `<div class="empty-state">✓ Nominal Asphalt Surface</div>`;
  } else {
    imgData.detections.forEach((det, i) => {
      const pid = det.pothole_id || `Defect #${i+1}`;
      const defType = (det.defect_type || 'pothole').replace(/_/g, ' ').toUpperCase();
      const sev = parseFloat(det.severity_score || 0);
      const conf = parseFloat(det.pothole_confidence || det.confidence || 0);
      const areaPx = det.mask_area_px || det.mask_area_pixels || 0;
      const box = det.bbox_xyxy || [];
      const boxStr = box.length === 4 ? `[${box.join(', ')}]` : '—';
      const col = sev >= 0.80 ? 'var(--red)' : sev >= 0.60 ? 'var(--orange)' : 'var(--yellow)';

      const item = document.createElement('div');
      item.className = 'defect-item-card';
      item.innerHTML = `
        <div class="defect-item-header">
          <span style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;color:var(--cyan);">${pid}</span>
          <span class="patch-defect-badge" style="background:${col}22;color:${col};border:1px solid ${col}44;">${defType}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-size:11px;margin-top:2px;">
          <div><span style="color:var(--text-muted);">Confidence:</span> <b style="color:#f8fafc;">${(conf*100).toFixed(0)}%</b></div>
          <div><span style="color:var(--text-muted);">Mask Area:</span> <b style="color:#f8fafc;">${areaPx} px</b></div>
          <div><span style="color:var(--text-muted);">Severity:</span> <b style="color:${col};">${(sev*100).toFixed(0)}%</b></div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:11px;">
          <div style="flex:1;" class="progress-track"><div class="progress-fill" style="width:${Math.round(sev*100)}%;background:${col};"></div></div>
        </div>
        <div style="font-size:10px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;">
          Bounding Box: ${boxStr}
        </div>
      `;
      defList.appendChild(item);
    });
  }
}

function prevSurveyPhoto() {
  if (currentSurveyIdx > 0) displaySurveyPhoto(currentSurveyIdx - 1);
}

function nextSurveyPhoto() {
  if (currentSurveyIdx < surveyImages.length - 1) displaySurveyPhoto(currentSurveyIdx + 1);
}

function onSelectSurveyPhoto(val) {
  displaySurveyPhoto(parseInt(val, 10));
}

function setSurveyViewMode(mode) {
  currentViewMode = mode;
  document.getElementById('btn-toggle-annotated').classList.toggle('active', mode === 'annotated');
  document.getElementById('btn-toggle-raw').classList.toggle('active', mode === 'raw');
  document.getElementById('survey-viewer-mode-label').innerText =
    mode === 'annotated' ? 'Mode: SAM2 Segmentation & BBox Overlay (Hotkeys: ← / → / O)' : 'Mode: Original Clean 2D Photo (Hotkeys: ← / → / O)';
  displaySurveyPhoto(currentSurveyIdx);
}

window.addEventListener('keydown', (e) => {
  if (document.activeElement && ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowLeft') prevSurveyPhoto();
  else if (e.key === 'ArrowRight') nextSurveyPhoto();
  else if (e.key === 'o' || e.key === 'O' || e.key === 'a' || e.key === 'A') {
    setSurveyViewMode(currentViewMode === 'annotated' ? 'raw' : 'annotated');
  }
});

/* Map setup */
const DEFAULT_LAT = 13.0827, DEFAULT_LON = 80.2707;
const map = L.map('map-container', { center: [DEFAULT_LAT, DEFAULT_LON], zoom: 17, attributionControl: false });
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 20 }).addTo(map);
let defectLayer = L.layerGroup().addTo(map);
let currentRadius = 50;

function renderDefects(defects) {
  defectLayer.clearLayers();
  if (!defects || defects.length === 0) return;
  defects.forEach(d => {
    const lat = parseFloat(d.latitude || DEFAULT_LAT);
    const lon = parseFloat(d.longitude || DEFAULT_LON);
    const sev = parseFloat(d.severity_score || 0.7);
    const col = sev >= 0.80 ? '#ef4444' : sev >= 0.60 ? '#f97316' : '#f59e0b';
    L.circleMarker([lat, lon], { radius: 8, fillColor: col, color: '#fff', weight: 2, fillOpacity: 0.9 }).addTo(defectLayer)
      .bindPopup(`<b>${d.defect_type || 'Pothole'}</b><br/>Conf: ${((d.confidence||0)*100).toFixed(0)}%<br/>Area: ${d.mask_area_px || 0}px`);
  });
}

function updateGeofenceRadius(val) {
  currentRadius = parseFloat(val);
  document.getElementById('radius-val').innerText = val + 'm';
}

function toggleDriveSimulation() {
  alert('Driver Simulation HUD Active: Corridor monitored for road defects.');
}

function renderPatchTab(defects) {
  const grid = document.getElementById('patch-grid');
  grid.innerHTML = '';
  document.getElementById('patch-count-badge').innerText = `${defects.length} Defect Patches`;
  if (defects.length === 0) {
    grid.innerHTML = '<div class="empty-state">No defect patches available.</div>';
    return;
  }
  defects.forEach(d => {
    const pid = d.pothole_id || 'Defect';
    const conf = ((d.confidence||0)*100).toFixed(0);
    const sev = ((d.severity_score||0)*100).toFixed(0);
    const area = d.mask_area_px || d.mask_area_pixels || 0;
    const card = document.createElement('div');
    card.className = 'patch-card';
    card.innerHTML = `
      <div class="patch-img-wrap">
        <img src="/api/patch_image/${encodeURIComponent(pid)}" alt="${pid}" onerror="this.src='/api/annotated_image/${encodeURIComponent(d.source_image||'')}'" />
      </div>
      <div class="patch-meta">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span class="patch-id">${pid}</span>
          <span class="patch-defect-badge" style="background:rgba(249,115,22,0.2);color:var(--orange);">${d.defect_type || 'POTHOLE'}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px;">
          <div><span style="color:var(--text-muted);">Conf:</span> <b>${conf}%</b></div>
          <div><span style="color:var(--text-muted);">Area:</span> <b>${area} px</b></div>
          <div><span style="color:var(--text-muted);">Severity:</span> <b>${sev}%</b></div>
          <div><span style="color:var(--text-muted);">BBox:</span> <b>[${(d.bbox_xyxy||[]).join(', ')}]</b></div>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

function renderWorkOrdersTab(workOrders) {
  const tbody = document.getElementById('wo-tbody');
  tbody.innerHTML = '';
  document.getElementById('wo-total-badge').innerText = `${workOrders.length} Active`;
  if (workOrders.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No active work orders.</td></tr>';
    return;
  }
  workOrders.forEach(wo => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="wo-id-cell">${wo.work_order_id}</td>
      <td>${wo.defect_class}</td>
      <td><span class="sev-badge sev-${wo.severity_tier}">${wo.severity_tier}</span></td>
      <td>${wo.area_m2} m²</td>
      <td>${wo.estimated_crew_size} techs</td>
      <td>${wo.target_resolution_hours}h</td>
      <td class="wo-text-cell">${wo.work_order_text}</td>
      <td><div class="materials-list">${(wo.required_materials||[]).map(m=>`<span class="mat-tag">${m}</span>`).join('')}</div></td>
    `;
    tbody.appendChild(tr);
  });
}

async function fetchAll(showMsg = false) {
  try {
    const [statsRes, resultsRes, woRes] = await Promise.all([
      fetch('/api/stats').then(r => r.json()),
      fetch('/api/results').then(r => r.json()),
      fetch('/api/work_orders').then(r => r.json()),
    ]);

    document.getElementById('kpi-score').innerHTML = `${statsRes.road_health_score}<span style="font-size:15px;color:var(--text-muted);font-weight:500;">/100</span>`;
    document.getElementById('kpi-survey-score').innerHTML = `${statsRes.road_health_score}<span style="font-size:15px;color:var(--text-muted);font-weight:500;">/100</span>`;
    document.getElementById('kpi-defects').innerText = `${statsRes.total_defects}`;
    document.getElementById('kpi-survey-dets').innerText = `${statsRes.total_defects}`;
    document.getElementById('kpi-water').innerText = `${statsRes.water_hazards || 0}`;
    document.getElementById('kpi-workorders').innerText = `${statsRes.work_orders_count}`;
    document.getElementById('kpi-survey-wo').innerText = `${statsRes.work_orders_count}`;
    document.getElementById('kpi-pred').innerHTML = `${Math.round((statsRes.deterioration_probability||0)*100)}<span style="font-size:15px;font-weight:500;">%</span>`;

    const cb = document.getElementById('kpi-cond-badge');
    const scb = document.getElementById('kpi-survey-cond-badge');
    if (cb) cb.innerText = statsRes.condition_class;
    if (scb) scb.innerText = statsRes.condition_class;

    const rh = resultsRes.road_health || {};
    const comp = rh.components || {};
    document.getElementById('cnt-penalty-txt').innerText = `-${comp.pothole_penalty || 0} pts`;
    document.getElementById('cnt-penalty-bar').style.width = `${Math.min(100, (comp.pothole_penalty||0)/40*100)}%`;
    document.getElementById('sev-penalty-txt').innerText = `-${comp.severity_penalty || 0} pts`;
    document.getElementById('sev-penalty-bar').style.width = `${Math.min(100, (comp.severity_penalty||0)/30*100)}%`;
    document.getElementById('water-penalty-txt').innerText = `-${comp.water_penalty || 0} pts`;
    document.getElementById('water-penalty-bar').style.width = `${Math.min(100, (comp.water_penalty||0)/20*100)}%`;
    if (rh.explanation) document.getElementById('explanation-text').innerText = rh.explanation;

    cachedDefects = resultsRes.detections || [];
    renderDefects(cachedDefects);
    await fetchDroneSurveyImages();
    renderPatchTab(cachedDefects);
    renderWorkOrdersTab(woRes.work_orders || []);

    if (showMsg) console.log('Dashboard refreshed from results/detection_results.json');
  } catch (err) {
    console.error('Fetch error:', err);
  }
}

fetchAll();
setInterval(fetchAll, 3500);
</script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def get_dashboard_ui(request: Request):
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


def main():
    parser = argparse.ArgumentParser(description="RoadSentinel Dashboard & Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    log.info(
        "Starting RoadSentinel Dashboard v3.2 on http://%s:%d  |  2D Sync Active",
        args.host, args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
