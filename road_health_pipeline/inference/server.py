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
    version="3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Demo output search paths
# ---------------------------------------------------------------------------

DEMO_OUTPUT_DIRS = [
    ROOT / "output" / "analytics_demo",
    ROOT.parent / "output",
    ROOT / "outputs",
    ROOT / "output",
]

_CACHED_PIPELINE = None
_SPATIAL_INDEX: Optional[DefectSpatialIndex] = None


# ---------------------------------------------------------------------------
# Spatial Deduplication Engine
# ---------------------------------------------------------------------------

DEDUP_RADIUS_M = 3.0  # Two detections within this radius = same physical defect


def annotate_patch_image(
    img_bytes: bytes,
    bboxes: List[List[Union[int, float]]],
    label: str = "Pothole",
    is_water: bool = False,
    severity: float = 0.0,
    confidence: float = 0.0,
    merged_count: int = 1,
) -> bytes:
    """Burn bounding boxes and defect metrics onto image bytes for visual display."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        color = (56, 189, 248) if is_water else (239, 68, 68) if severity >= 0.85 else (249, 115, 22) if severity >= 0.65 else (245, 158, 11)

        for i, box in enumerate(bboxes):
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            box_col = color if i == 0 else (168, 85, 247)  # Primary box in theme color, overlap in purple
            draw.rectangle([x1, y1, x2, y2], outline=box_col, width=3)

            tag = f"{label} #{i+1} (conf:{confidence*100:.0f}%)" if i > 0 else f"{label} (sev:{severity*100:.0f}%)"
            lw = min(w - x1, len(tag) * 8 + 8)
            ty = max(0, y1 - 16)
            draw.rectangle([x1, ty, x1 + lw, ty + 16], fill=box_col)
            draw.text((x1 + 3, ty + 1), tag, fill=(255, 255, 255))

        if merged_count > 1:
            badge = f"DEDUP: {merged_count} photos merged (<3m)"
            draw.rectangle([6, 6, min(w - 6, 6 + len(badge) * 8 + 8), 24], fill=(15, 23, 42))
            draw.text((10, 8), badge, fill=(56, 189, 248))

        out_buf = io.BytesIO()
        img.save(out_buf, format="JPEG", quality=90)
        return out_buf.getvalue()
    except Exception as exc:
        log.warning("annotate_patch_image failed: %s", exc)
        return img_bytes


class SpatialDeduplicator:
    """Groups incoming pothole/defect detections by GPS proximity.

    If two detections fall within ``DEDUP_RADIUS_M`` (3.0m) metres of each other
    they are considered the *same* physical defect. When a duplicate arrives:
      - The bounding box data is merged into the cluster.
      - The detection record and photo with the HIGHEST confidence score is selected
        as canonical.
      - The work order is NOT duplicated. Exactly one work order exists per physical defect.
    """

    def __init__(self):
        self._clusters: Dict[str, Dict[str, Any]] = {}
        self._patch_images: Dict[str, bytes] = {}
        self._all_bboxes: Dict[str, List[List[Union[int, float]]]] = {}
        self._cluster_sizes: Dict[str, int] = {}
        self._work_orders: Dict[str, Dict[str, Any]] = {}

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

        return {
            "work_order_id": f"WO-{seg_id.upper()[:14]}-{cluster_id[-8:]}",
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

    # ------------------------------------------------------------------
    def seed_from_result(self, detections: List[Dict[str, Any]], work_orders: Optional[List[Dict[str, Any]]] = None) -> None:
        """Pre-populate from existing on-disk detections and work orders."""
        for det in detections:
            lat = det.get("latitude") or det.get("lat")
            lon = det.get("longitude") or det.get("lon")
            if lat is None or lon is None:
                continue
            cid = det.get("pothole_id") or det.get("defect_id") or str(uuid.uuid4())
            if cid not in self._clusters:
                d = dict(det)
                d["all_bboxes"] = [det["bbox_xyxy"]] if det.get("bbox_xyxy") else []
                d["merged_count"] = 1
                self._clusters[cid] = d
                self._all_bboxes[cid] = list(d["all_bboxes"])
                self._cluster_sizes[cid] = 1

        if work_orders:
            for wo in work_orders:
                pid = wo.get("pothole_id") or wo.get("work_order_id")
                if pid:
                    self._work_orders[pid] = dict(wo)

    # ------------------------------------------------------------------
    def ingest(
        self,
        detection: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Ingest a new detection, applying 3-meter spatial deduplication.

        Returns:
            dict with 'status' ("new"|"deduplicated"), 'cluster_id', 'canonical', 'total_work_orders'
        """
        lat = float(detection.get("latitude") or detection.get("lat") or 0.0)
        lon = float(detection.get("longitude") or detection.get("lon") or 0.0)
        incoming_conf = float(detection.get("pothole_confidence") or detection.get("confidence") or 0.0)
        incoming_bbox = detection.get("bbox_xyxy")

        # Search existing clusters for a nearby match (within 3 meters)
        for cid, canonical in self._clusters.items():
            c_lat = float(canonical.get("latitude") or canonical.get("lat") or 0.0)
            c_lon = float(canonical.get("longitude") or canonical.get("lon") or 0.0)
            dist = haversine_distance_m(lat, lon, c_lat, c_lon)
            if dist <= DEDUP_RADIUS_M:
                # ── DUPLICATE PHYSICAL DEFECT ──
                merged_bboxes = self._all_bboxes.setdefault(cid, [])
                if incoming_bbox:
                    merged_bboxes.append(incoming_bbox)
                self._cluster_sizes[cid] = self._cluster_sizes.get(cid, 1) + 1

                stored_conf = float(canonical.get("pothole_confidence") or canonical.get("confidence") or 0.0)

                # Select the image & record with the highest confidence score
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

                # DO NOT DUPLICATE THE WORK ORDER:
                # If a work order exists, preserve it. If incoming has higher severity, update in-place.
                if cid in self._work_orders:
                    wo = self._work_orders[cid]
                    new_sev = float(self._clusters[cid].get("severity_score") or 0.0)
                    if new_sev > float(wo.get("severity_score") or 0.0):
                        wo["severity_score"] = round(new_sev, 2)
                        wo["severity_tier"] = "critical" if new_sev >= 0.85 else "high"
                elif float(self._clusters[cid].get("severity_score") or 0.0) >= 0.65:
                    self._work_orders[cid] = self._generate_work_order(cid, self._clusters[cid])

                log.info(
                    "DEDUP: detection at (%.5f, %.5f) merged into cluster %s (dist=%.2fm, conf=%.2f vs %.2f, total_bboxes=%d). Work order preserved.",
                    lat, lon, cid, dist, incoming_conf, stored_conf, len(merged_bboxes),
                )
                return {
                    "status": "deduplicated",
                    "cluster_id": cid,
                    "distance_m": round(dist, 2),
                    "canonical": self._clusters[cid],
                    "total_merged_bboxes": len(merged_bboxes),
                    "total_work_orders": len(self._work_orders),
                }

        # ── NEW CLUSTER (> 3.0m) ──
        cid = detection.get("pothole_id") or detection.get("defect_id") or str(uuid.uuid4())
        while cid in self._clusters:
            cid = cid + "_" + uuid.uuid4().hex[:4]

        detection["all_bboxes"] = [incoming_bbox] if incoming_bbox else []
        detection["merged_count"] = 1
        self._clusters[cid] = dict(detection)
        self._all_bboxes[cid] = list(detection["all_bboxes"])
        self._cluster_sizes[cid] = 1
        if image_bytes:
            self._patch_images[cid] = image_bytes

        # Create 1 work order if severity warrants it
        sev_score = float(detection.get("severity_score") or 0.0)
        if sev_score >= 0.60 or detection.get("severity_tier") in ("high", "critical"):
            self._work_orders[cid] = self._generate_work_order(cid, detection)

        log.info("DEDUP: new cluster %s at (%.5f, %.5f). Work orders count: %d", cid, lat, lon, len(self._work_orders))
        return {
            "status": "new",
            "cluster_id": cid,
            "canonical": self._clusters[cid],
            "total_merged_bboxes": len(self._all_bboxes[cid]),
            "total_work_orders": len(self._work_orders),
        }

    # ------------------------------------------------------------------
    def get_all_detections(self) -> List[Dict[str, Any]]:
        return list(self._clusters.values())

    def get_patch_image(self, cluster_id: str) -> Optional[bytes]:
        return self._patch_images.get(cluster_id)

    def get_all_work_orders(self) -> List[Dict[str, Any]]:
        return list(self._work_orders.values())

    def cluster_count(self) -> int:
        return len(self._clusters)


_DEDUPLICATOR = SpatialDeduplicator()
_DEDUP_SEEDED = False


def get_deduplicator() -> SpatialDeduplicator:
    """Return the singleton deduplicator, seeding it from disk on first call."""
    global _DEDUP_SEEDED
    if not _DEDUP_SEEDED:
        data = get_latest_result_data()
        defs = data.get("detections") or data.get("potholes") or []
        wos = get_latest_work_orders_data().get("work_orders", [])
        _DEDUPLICATOR.seed_from_result(defs, wos)
        _DEDUP_SEEDED = True
    return _DEDUPLICATOR


# ---------------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------------

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

    # Default fallback demo structure
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
            "explanation": "Significant road degradation caused by severe water-filled pothole cavitation.",
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
                "pothole_id": "demo-pothole-001",
                "defect_type": "pothole",
                "confidence": 0.88,
                "pothole_confidence": 0.88,
                "bbox_xyxy": [350, 400, 450, 500],
                "mask_area_pixels": 6360,
                "area_m2": 0.45,
                "estimated_depth_m": 0.08,
                "is_water_filled": False,
                "water_flag": False,
                "water_confidence": 0.12,
                "latitude": 13.0827,
                "longitude": 80.2707,
                "severity_score": 0.72,
                "severity_breakdown": {
                    "area": 0.225, "depth": 0.533, "shape": 0.85,
                    "water": 0.12, "surrounding_damage": 0.3, "confidence": 0.88,
                },
                "source_image": "input_frame.jpg",
                "defect_id": "demo-pothole-001",
            },
            {
                "pothole_id": "demo-pothole-002",
                "defect_type": "water_filled_pothole",
                "confidence": 0.92,
                "pothole_confidence": 0.92,
                "bbox_xyxy": [770, 475, 930, 565],
                "mask_area_pixels": 10050,
                "area_m2": 1.10,
                "estimated_depth_m": 0.14,
                "is_water_filled": True,
                "water_flag": True,
                "water_confidence": 0.85,
                "latitude": 13.0831,
                "longitude": 80.2712,
                "severity_score": 0.94,
                "severity_breakdown": {
                    "area": 0.55, "depth": 0.933, "shape": 0.72,
                    "water": 0.85, "surrounding_damage": 0.4, "confidence": 0.92,
                },
                "source_image": "input_frame.jpg",
                "defect_id": "demo-pothole-002",
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
                "pothole_id": "demo-pothole-002",
                "defect_class": "water_filled_pothole",
                "severity_tier": "critical",
                "severity_score": 0.94,
                "area_m2": 1.10,
                "estimated_depth_m": 0.14,
                "is_water_filled": True,
                "work_order_text": (
                    "URGENT: High-priority immediate remediation is required on road segment "
                    "'seg_carla_town04_0042' for a critical-tier water_filled_pothole covering 1.10 m² "
                    "with critical hydroplaning water accumulation. Crews must dewater the cavity using a "
                    "submersible pump, square-cut and clean the perimeter, apply cationic rapid-setting tack "
                    "coat emulsion (CRS-2), and place hot-mix asphalt (HMA Type B) compacted with a vibratory "
                    "plate in two 50 mm lifts. Establish MUTCD-compliant single-lane closure."
                ),
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
                "safety_measures": "MUTCD Chapter 6H temporary traffic control taper.",
                "estimated_crew_size": 3,
                "target_resolution_hours": 12,
                "latitude": 13.0831,
                "longitude": 80.2712,
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
# API Routes — Health & Data
# ---------------------------------------------------------------------------

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status": "online",
        "service": "RoadSentinel Government Analytics & Inference Gateway",
        "version": "3.0",
        "dedup_radius_m": DEDUP_RADIUS_M,
        "active_clusters": get_deduplicator().cluster_count(),
        "timestamp": utc_iso(),
    }


@app.get("/api/results")
def get_results():
    """Returns on-disk analytics result merged with live deduplicated detections."""
    data = get_latest_result_data()
    dedup = get_deduplicator()
    live = dedup.get_all_detections()
    # Overlay live deduplicated detections if any were ingested via /api/ingest
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
    wo = get_latest_work_orders_data()
    rh = res.get("road_health", {})
    pred = res.get("prediction", {})
    dedup = get_deduplicator()
    all_defs = dedup.get_all_detections() or res.get("detections") or res.get("potholes") or []
    wos = dedup.get_all_work_orders()
    wo_count = len(wos) if wos else len(wo.get("work_orders", []))

    score = rh.get("road_health_score", 100.0)
    cond = rh.get("condition_class", "Good").upper()
    total_def = len(all_defs)
    water_hazards = sum(
        1 for d in all_defs if d.get("is_water_filled") or d.get("water_flag")
    )
    critical_def = sum(
        1 for d in all_defs
        if (d.get("severity_score") or 0.0) >= 0.85
        or (d.get("severity") or {}).get("severity") == "critical"
    )

    return {
        "road_health_score": round(score, 1),
        "condition_class": cond,
        "total_defects": total_def,
        "active_clusters": dedup.cluster_count(),
        "critical_hazards": critical_def,
        "water_hazards": water_hazards,
        "work_orders_count": wo_count,
        "deterioration_probability": pred.get("deterioration_probability", 0.0),
        "prediction_horizon_days": pred.get("prediction_horizon_days", 30),
        "road_segment_id": res.get("road_segment_id", "N/A"),
        "map_name": res.get("map_name", "Town04"),
        "geolocation": res.get("geolocation", {"lat": 13.0827, "lon": 80.2707}),
        "dedup_radius_m": DEDUP_RADIUS_M,
    }


# ---------------------------------------------------------------------------
# /api/ingest — accepts mock or real detection payloads with 3m dedup
# ---------------------------------------------------------------------------

@app.post("/api/ingest")
async def ingest_detection(request: Request):
    """Accept a detection payload and apply 3-meter spatial deduplication.

    Expected JSON body:
    {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "pothole_id": "...",          // optional
        "bbox_xyxy": [x1,y1,x2,y2],  // optional
        "area_m2": 0.45,              // optional
        "estimated_depth_m": 0.08,    // optional
        "pothole_confidence": 0.88,   // optional
        "severity_score": 0.72,       // optional
        "defect_type": "pothole",     // optional
        "water_flag": false,          // optional
        "source_image": "...",        // optional
        "timestamp": "...",           // optional
        "_image_b64": "..."           // optional: base64-encoded patch image
    }
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, f"Invalid JSON body: {exc}")

    lat = body.get("latitude") or body.get("lat")
    lon = body.get("longitude") or body.get("lon")
    if lat is None or lon is None:
        raise HTTPException(400, "latitude and longitude are required fields")

    # Decode optional base64 patch image
    image_bytes: Optional[bytes] = None
    b64_img = body.pop("_image_b64", None)
    if b64_img:
        try:
            image_bytes = base64.b64decode(b64_img)
        except Exception:
            log.warning("Failed to decode _image_b64 field — ignoring")

    if "timestamp" not in body:
        body["timestamp"] = utc_iso()

    dedup = get_deduplicator()
    result = dedup.ingest(body, image_bytes=image_bytes)

    # Invalidate spatial index so geofence queries use fresh data
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


# ---------------------------------------------------------------------------
# /api/patch_image/{defect_id} — serve annotated patch image with bboxes
# ---------------------------------------------------------------------------

@app.get("/api/patch_image/{defect_id}")
def get_patch_image(defect_id: str):
    """Return the patch image with visually overlaid bounding boxes for a defect cluster."""
    dedup = get_deduplicator()

    # Search deduplicator clusters
    for cid, det in dedup._clusters.items():
        pid = det.get("pothole_id") or det.get("defect_id") or cid
        if pid == defect_id or cid == defect_id:
            raw_bytes = dedup.get_patch_image(cid)
            bboxes = det.get("all_bboxes") or ([det.get("bbox_xyxy")] if det.get("bbox_xyxy") else [])
            is_water = bool(det.get("is_water_filled") or det.get("water_flag"))
            sev = float(det.get("severity_score") or 0.0)
            conf = float(det.get("pothole_confidence") or det.get("confidence") or 0.0)
            def_type = (det.get("defect_type") or "Pothole").replace("_", " ").title()
            merged_count = det.get("merged_count", len(bboxes))

            if raw_bytes:
                annotated = annotate_patch_image(
                    raw_bytes,
                    bboxes,
                    label=def_type,
                    is_water=is_water,
                    severity=sev,
                    confidence=conf,
                    merged_count=merged_count,
                )
                return Response(content=annotated, media_type="image/jpeg")

            # Fallback: crop from demo output detection_overlay or input_frame
            bbox = det.get("bbox_xyxy")
            for d in DEMO_OUTPUT_DIRS:
                for fname in ("input_frame.jpg", "detection_overlay.jpg"):
                    fpath = d / fname
                    if fpath.is_file():
                        try:
                            from PIL import Image
                            full_img = Image.open(fpath).convert("RGB")
                            iw, ih = full_img.size
                            if bbox and len(bbox) == 4:
                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                pad = 35
                                x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
                                x2p, y2p = min(iw, x2 + pad), min(ih, y2 + pad)
                                crop = full_img.crop((x1p, y1p, x2p, y2p))
                                crop_bboxes = []
                                for b in bboxes:
                                    if b and len(b) == 4:
                                        crop_bboxes.append([b[0] - x1p, b[1] - y1p, b[2] - x1p, b[3] - y1p])
                                c_buf = io.BytesIO()
                                crop.save(c_buf, format="JPEG", quality=90)
                                annotated = annotate_patch_image(
                                    c_buf.getvalue(),
                                    crop_bboxes,
                                    label=def_type,
                                    is_water=is_water,
                                    severity=sev,
                                    confidence=conf,
                                    merged_count=merged_count,
                                )
                                return Response(content=annotated, media_type="image/jpeg")
                        except Exception as exc:
                            log.warning("Crop fallback failed for %s: %s", defect_id, exc)

    # Last resort: serve the full input frame
    for d in DEMO_OUTPUT_DIRS:
        for fname in ("input_frame.jpg", "detection_overlay.jpg"):
            fpath = d / fname
            if fpath.is_file():
                return FileResponse(str(fpath))

    raise HTTPException(404, f"No patch image found for defect '{defect_id}'")


# ---------------------------------------------------------------------------
# Existing routes — Geofencing / Overlays / Inference
# ---------------------------------------------------------------------------

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


@app.get("/api/map/town04.png")
def get_town04_map():
    for d in DEMO_OUTPUT_DIRS:
        map_p = d / "town04_map.png"
        if map_p.is_file():
            return FileResponse(str(map_p))
    env_map = ROOT.parent / "env" / "assets" / "town04_map.png"
    if env_map.is_file():
        return FileResponse(str(env_map))
    raise HTTPException(status_code=404, detail="Town04 map not found")


@app.get("/api/overlays/{filename}")
def get_overlay_image(filename: str):
    for d in DEMO_OUTPUT_DIRS:
        fpath = d / filename
        if fpath.is_file():
            return FileResponse(str(fpath))
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
            _SPATIAL_INDEX = DefectSpatialIndex(result.potholes)
        except Exception as exc:
            log.exception("Inference failed:")
            raise HTTPException(500, str(exc))
        return JSONResponse(result.to_dict())


# ---------------------------------------------------------------------------
# Three-Tab Government Dashboard UI
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RoadSentinel — Government Infrastructure Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    :root {
      --bg-dark: #070a11;
      --bg-card: rgba(15, 23, 42, 0.82);
      --border-card: rgba(255,255,255,0.08);
      --border-accent: rgba(0,242,254,0.3);
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

    /* ── Header ── */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 32px;
      background: rgba(7,10,17,0.92);
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
      width: 38px; height: 38px;
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 800; font-size: 18px; color: #000;
      box-shadow: 0 0 20px var(--cyan-glow);
      flex-shrink: 0;
    }
    .brand-title { font-size: 19px; font-weight: 800; letter-spacing: -0.5px;
      background: linear-gradient(to right, #fff, #cbd5e1);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .brand-subtitle { font-size: 10px; color: var(--cyan); font-weight: 600;
      letter-spacing: 1px; text-transform: uppercase; }
    .header-right { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .badge {
      padding: 5px 12px; border-radius: 999px; font-size: 11px; font-weight: 600;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .badge-live { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: var(--green); }
    .badge-live::before {
      content: ""; width: 7px; height: 7px; background: var(--green);
      border-radius: 50%; box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.3)} }
    .btn-refresh {
      background: linear-gradient(135deg, rgba(0,242,254,0.18), rgba(139,92,246,0.18));
      border: 1px solid var(--border-accent); color: #fff;
      padding: 7px 16px; border-radius: 10px; font-weight: 600; font-size: 12px;
      cursor: pointer; transition: all .2s ease; display: flex; align-items: center; gap: 6px;
    }
    .btn-refresh:hover { background: linear-gradient(135deg,rgba(0,242,254,.3),rgba(139,92,246,.3)); box-shadow: 0 0 15px var(--cyan-glow); }

    .btn-upload {
      background: linear-gradient(135deg, rgba(0,242,254,0.22), rgba(16,185,129,0.22));
      border: 1px solid var(--cyan); color: var(--cyan);
      padding: 7px 16px; border-radius: 10px; font-weight: 700; font-size: 12px;
      cursor: pointer; transition: all .2s ease; display: flex; align-items: center; gap: 6px;
    }
    .btn-upload:hover { background: rgba(0,242,254,.35); box-shadow: 0 0 15px var(--cyan-glow); color: #fff; }

    /* ── Tab Nav ── */
    .tab-nav {
      display: flex; gap: 4px;
      padding: 0 32px;
      background: rgba(7,10,17,0.85);
      border-bottom: 1px solid var(--border-card);
      position: sticky; top: 67px; z-index: 900;
    }
    .tab-nav-btn {
      padding: 12px 22px; font-size: 13px; font-weight: 600;
      color: var(--text-muted); background: none; border: none;
      border-bottom: 2px solid transparent; cursor: pointer;
      transition: all .2s; white-space: nowrap;
      display: flex; align-items: center; gap: 7px;
    }
    .tab-nav-btn:hover { color: #fff; }
    .tab-nav-btn.active { color: var(--cyan); border-bottom-color: var(--cyan); }

    /* ── Main / Tab Panels ── */
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
    .kpi-title { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; }
    .kpi-value { font-size: 30px; font-weight: 800; letter-spacing: -1px; display: flex; align-items: baseline; gap: 5px; }
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
    }
    .panel-title { font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 9px; }

    /* ── Layout helpers ── */
    .two-col { display: grid; grid-template-columns: 1.1fr .9fr; gap: 22px; }
    @media (max-width: 1100px) { .two-col { grid-template-columns: 1fr; } }

    /* ── Map ── */
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
      padding: 5px 12px; border-radius: 8px; font-size: 11px; font-weight: 700; cursor: pointer; transition: all .2s; }
    .btn-sim:hover { background: rgba(0,242,254,.3); }

    /* ── Proximity banner ── */
    #proximity-banner {
      background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.3);
      border-radius: 10px; padding: 12px 18px; display: flex; align-items: center;
      justify-content: space-between; gap: 14px; transition: all .3s;
    }
    #proximity-banner.warning { background: rgba(245,158,11,.13); border-color: rgba(245,158,11,.5); }
    #proximity-banner.critical { background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.6); animation: alertPulse 1.5s infinite alternate; }
    @keyframes alertPulse { from{border-color:rgba(239,68,68,.4)} to{border-color:rgba(239,68,68,1)} }
    .alert-text { flex: 1; font-size: 13px; font-weight: 600; color: #f1f5f9; }

    /* ── Score Breakdown ── */
    .score-breakdown { display: flex; flex-direction: column; gap: 8px;
      padding: 13px 15px; background: rgba(255,255,255,.02); border-radius: 10px; border: 1px solid var(--border-card); }
    .bar-row { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted); }
    .progress-track { height: 7px; background: rgba(255,255,255,.08); border-radius: 999px; overflow: hidden; margin-top: 4px; }
    .progress-fill { height: 100%; border-radius: 999px; transition: width .6s ease; }

    /* ── Patch Inspection ── */
    .patch-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px; }
    .patch-card {
      background: rgba(255,255,255,.03); border: 1px solid var(--border-card);
      border-radius: 14px; overflow: hidden; display: flex; flex-direction: column;
    }
    .patch-img-wrap { position: relative; width: 100%; background: #020617; min-height: 200px;
      display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .patch-img-wrap canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
    .patch-img-wrap img { max-width: 100%; max-height: 260px; object-fit: contain; display: block; }
    .patch-meta { padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; }
    .patch-id { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--cyan); font-weight: 700; }
    .patch-defect-badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .5px; }
    .metrics-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .metrics-table th { text-align: left; color: var(--text-muted); font-size: 11px;
      font-weight: 600; padding: 5px 0; text-transform: uppercase; border-bottom: 1px solid var(--border-card); }
    .metrics-table td { padding: 6px 0; color: #e2e8f0; font-weight: 500; }
    .metrics-table td.val { font-family: 'JetBrains Mono', monospace; color: var(--cyan); font-weight: 700; }
    .sev-bar-wrap { display: flex; align-items: center; gap: 8px; }
    .sev-label { font-size: 11px; font-weight: 700; min-width: 52px; }

    /* ── Work Orders Table ── */
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
    .wo-expand-btn { background: none; border: none; color: var(--cyan); cursor: pointer; font-size: 11px; font-weight: 600; padding: 0; margin-top: 4px; }
    .materials-list { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
    .mat-tag { background: rgba(255,255,255,.06); padding: 3px 8px; border-radius: 5px; font-size: 10px; color: #94a3b8; }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,.2); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.15); border-radius: 3px; }

    /* ── Footer ── */
    footer {
      padding: 14px 32px; border-top: 1px solid var(--border-card);
      display: flex; justify-content: space-between; align-items: center;
      font-size: 11px; color: var(--text-muted); background: rgba(7,10,17,.88); flex-wrap: wrap; gap: 8px;
    }

    /* ── Empty state ── */
    .empty-state { text-align: center; padding: 40px; color: var(--text-muted); font-size: 14px; }
    .empty-state .icon { font-size: 36px; margin-bottom: 12px; }

    /* ── Dedup notice ── */
    .dedup-badge {
      background: rgba(59,130,246,.12); border: 1px solid rgba(59,130,246,.3);
      color: var(--blue); font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 6px;
    }
  </style>
</head>
<body>

<!-- ── Header ── -->
<header>
  <div class="brand">
    <div class="brand-icon">RS</div>
    <div>
      <div class="brand-title">RoadSentinel</div>
      <div class="brand-subtitle">Government Infrastructure Intelligence Platform</div>
    </div>
  </div>
  <div class="header-right">
    <span class="badge badge-live">LIVE</span>
    <span style="font-size:11px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;" id="clock-display">--:--:-- UTC</span>
    <span class="dedup-badge" id="dedup-badge">⊕ Dedup: <span id="dedup-count">—</span> clusters</span>
    <button class="btn-upload" onclick="document.getElementById('mock-file-input').click()">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
      Upload Photo (Mock Ingest)
    </button>
    <input type="file" id="mock-file-input" accept="image/*" style="display:none;" onchange="handleMockFileUpload(event)" />
    <button class="btn-refresh" onclick="fetchAll(true)">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
      Refresh
    </button>
  </div>
</header>

<!-- ── Tab Navigation ── -->
<nav class="tab-nav">
  <button class="tab-nav-btn active" onclick="switchTab('map')" id="tab-btn-map">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>
    Map View
  </button>
  <button class="tab-nav-btn" onclick="switchTab('patches')" id="tab-btn-patches">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
    Patch Inspection
  </button>
  <button class="tab-nav-btn" onclick="switchTab('workorders')" id="tab-btn-workorders">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
    Work Orders
  </button>
</nav>

<!-- ── Main Content ── -->
<main>

  <!-- ════════════ TAB 1: MAP VIEW ════════════ -->
  <div class="tab-panel active" id="tab-map">

    <!-- KPI Strip -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Road Health Index</span><span class="badge" id="kpi-cond-badge" style="background:rgba(239,68,68,.15);color:var(--orange);">POOR</span></div>
        <div class="kpi-value" id="kpi-score">—<span style="font-size:15px;color:var(--text-muted);font-weight:500;">/100</span></div>
        <div class="kpi-desc" id="kpi-segment-id">Segment: —</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Detected Defects</span></div>
        <div class="kpi-value" id="kpi-defects">—</div>
        <div class="kpi-desc">Active deduplicated clusters</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Water Hazards</span></div>
        <div class="kpi-value" style="color:#38bdf8;" id="kpi-water">—</div>
        <div class="kpi-desc">High hydroplaning risk</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Work Orders</span></div>
        <div class="kpi-value" style="color:var(--purple);" id="kpi-workorders">—</div>
        <div class="kpi-desc">Actionable maintenance dispatches</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">30-Day Deterioration Risk</span></div>
        <div class="kpi-value" style="color:var(--red);" id="kpi-pred">—<span style="font-size:15px;font-weight:500;">%</span></div>
        <div class="kpi-desc">Forecasted failure probability</div>
      </div>
    </div>

    <!-- Proximity Banner -->
    <div id="proximity-banner">
      <div style="font-size:18px;">🛡️</div>
      <div class="alert-text" id="alert-text-content"><b>Driver Proximity HUD:</b> Corridor clear — KD-Tree active.</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan);" id="nearest-hazard-dist">Nearest: —</div>
    </div>

    <!-- Map + Score Breakdown -->
    <div class="two-col">
      <div style="display:flex;flex-direction:column;gap:18px;">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="16" height="16" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>
              GPS Defect Map
            </div>
            <span style="font-size:11px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;" id="gps-coords">—° N, —° E</span>
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
      </div>

      <div style="display:flex;flex-direction:column;gap:18px;">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="16" height="16" fill="none" stroke="var(--purple)" stroke-width="2" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
              Road Health Breakdown
            </div>
          </div>
          <div class="score-breakdown">
            <div class="bar-row"><span>Pothole Severity Penalty</span><span id="sev-penalty-txt">—</span></div>
            <div class="progress-track"><div class="progress-fill" id="sev-penalty-bar" style="width:0%;background:var(--orange);"></div></div>
            <div class="bar-row" style="margin-top:8px;"><span>Pothole Count Penalty</span><span id="cnt-penalty-txt">—</span></div>
            <div class="progress-track"><div class="progress-fill" id="cnt-penalty-bar" style="width:0%;background:var(--yellow);"></div></div>
            <div class="bar-row" style="margin-top:8px;"><span>Water Hazard Penalty</span><span id="water-penalty-txt">—</span></div>
            <div class="progress-track"><div class="progress-fill" id="water-penalty-bar" style="width:0%;background:#38bdf8;"></div></div>
            <div class="bar-row" style="margin-top:8px;"><span>Surface Wear Penalty</span><span id="surf-penalty-txt">—</span></div>
            <div class="progress-track"><div class="progress-fill" id="surf-penalty-bar" style="width:0%;background:var(--purple);"></div></div>
          </div>
          <div style="font-size:12px;color:#cbd5e1;line-height:1.5;background:rgba(255,255,255,.02);padding:12px;border-radius:8px;" id="explanation-text">
            —
          </div>
        </div>

        <div class="panel" style="flex:1;">
          <div class="panel-header">
            <div class="panel-title">
              <svg width="16" height="16" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/></svg>
              Deterioration Prediction
            </div>
            <span style="font-size:10px;color:var(--text-muted);">CARLA-SYNTHETIC ONLY</span>
          </div>
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:13px;color:var(--text-muted);">Deterioration Probability</span>
              <span style="font-size:20px;font-weight:800;color:var(--red);" id="detp-score">—%</span>
            </div>
            <div class="progress-track"><div class="progress-fill" id="detp-bar" style="width:0%;background:var(--red);"></div></div>
            <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-muted);">
              <span>Horizon: <b id="horizon-days" style="color:#e2e8f0;">—</b> days</span>
              <span id="prog-direction" style="font-weight:700;color:var(--red);">—</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ════════════ TAB 2: PATCH INSPECTION ════════════ -->
  <div class="tab-panel" id="tab-patches">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="16" height="16" fill="none" stroke="var(--cyan)" stroke-width="2" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          Patch Inspection — Defect Bounding Boxes &amp; Metrics
        </div>
        <span style="font-size:12px;color:var(--text-muted);" id="patch-count-badge">Loading…</span>
      </div>
      <p style="font-size:12px;color:var(--text-muted);line-height:1.6;">
        Each patch shows the road image with the detected pothole's bounding box overlaid. The metrics table displays
        depth, area, and severity score computed by the analytics pipeline.
        Bounding boxes are drawn from the <code>bbox_xyxy</code> field of each <code>PotholeRecord</code>.
      </p>
    </div>
    <div class="patch-grid" id="patch-grid">
      <div class="empty-state"><div class="icon">🔍</div>Loading patch data…</div>
    </div>
  </div>

  <!-- ════════════ TAB 3: WORK ORDERS ════════════ -->
  <div class="tab-panel" id="tab-workorders">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="16" height="16" fill="none" stroke="var(--green)" stroke-width="2" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          Repair Work Orders
        </div>
        <span class="badge" style="background:rgba(16,185,129,.13);color:var(--green);" id="wo-total-badge">— Active</span>
      </div>
      <p style="font-size:12px;color:var(--text-muted);line-height:1.6;">
        Auto-generated maintenance dispatches from the VLM work order engine.
        Only <b>critical</b> and <b>high</b> severity defects trigger work orders.
        Each work order is unique per deduplicated cluster — no duplicate dispatch.
      </p>
    </div>
    <div class="panel" style="padding:0;overflow:hidden;">
      <div class="wo-table-wrap">
        <table class="wo-table" id="wo-table">
          <thead>
            <tr>
              <th>Work Order ID</th>
              <th>Segment</th>
              <th>Defect Class</th>
              <th>Severity</th>
              <th>Area (m²)</th>
              <th>Depth (cm)</th>
              <th>Water</th>
              <th>Crew</th>
              <th>Target (h)</th>
              <th>Description</th>
              <th>Materials</th>
            </tr>
          </thead>
          <tbody id="wo-tbody">
            <tr><td colspan="11" class="empty-state">Loading work orders…</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

</main>

<!-- ── Footer ── -->
<footer>
  <div>RoadSentinel v3.0 • Multi-Tab Dashboard • Spatial Dedup (3m radius) • feature/dashboard-rebuild</div>
  <div style="font-family:'JetBrains Mono',monospace;">DINOv2 • SAM2.1 • KD-Tree • Gemini VLM</div>
</footer>

<script>
/* ── Clock ── */
function tick() {
  const n = new Date();
  document.getElementById('clock-display').innerText = n.toUTCString().split(' ')[4] + ' UTC';
}
setInterval(tick, 1000); tick();

/* ── Tab switching ── */
function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('tab-btn-' + name).classList.add('active');
  if (name === 'map') {
    setTimeout(() => {
      map.invalidateSize();
      if (cachedDefects.length > 0) renderDefects(cachedDefects);
    }, 150);
  }
}

/* ═══════════════════════════════════════
   MAP — Leaflet initialisation
═══════════════════════════════════════ */
const DEFAULT_LAT = 13.0827, DEFAULT_LON = 80.2707;
const map = L.map('map-container', { center: [DEFAULT_LAT, DEFAULT_LON], zoom: 17, attributionControl: false });

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 20,
  attribution: '© OpenStreetMap'
}).addTo(map);

let defectLayer = L.layerGroup().addTo(map);
let driverLayer = L.layerGroup().addTo(map);
let currentRadius = 50;
let driverMarker = null, driverSimInterval = null, driverPosIdx = 0;
let cachedDefects = [];

/* Render defect markers from real GPS data */
function renderDefects(defects) {
  defectLayer.clearLayers();
  if (!defects || defects.length === 0) return;

  defects.forEach(d => {
    const lat = parseFloat(d.latitude || d.lat || DEFAULT_LAT);
    const lon = parseFloat(d.longitude || d.lon || DEFAULT_LON);
    const isWater = !!(d.is_water_filled || d.water_flag);
    const sev = parseFloat(d.severity_score || 0);
    const defType = (d.defect_type || 'pothole').replace(/_/g, ' ');
    const area = parseFloat(d.area_m2 || d.estimated_area_m2 || 0).toFixed(2);
    const depth = parseFloat(d.estimated_depth_m || 0);
    const pid = d.pothole_id || d.defect_id || '?';

    const markerColor = isWater ? '#38bdf8' : (sev >= 0.85 ? '#ef4444' : sev >= 0.65 ? '#f97316' : '#f59e0b');

    /* Geofence ring */
    L.circle([lat, lon], {
      radius: currentRadius,
      color: markerColor, fillColor: markerColor,
      fillOpacity: 0.1, weight: 1.5, dashArray: '4,6'
    }).addTo(defectLayer);

    /* Core marker */
    const cm = L.circleMarker([lat, lon], {
      radius: 9, fillColor: markerColor, color: '#fff',
      weight: 2, opacity: 1, fillOpacity: 0.95
    }).addTo(defectLayer);

    cm.bindPopup(`
      <div style="font-family:'Plus Jakarta Sans',sans-serif;min-width:200px;">
        <strong style="color:${markerColor};font-size:13px;">${defType}</strong><br/>
        <span style="font-size:11px;color:#6b7280;">ID: ${pid}</span><br/><br/>
        <b>GPS:</b> ${lat.toFixed(5)}°, ${lon.toFixed(5)}°<br/>
        <b>Area:</b> ${area} m² &nbsp;|&nbsp; <b>Depth:</b> ${(depth*100).toFixed(1)} cm<br/>
        <b>Severity:</b> ${(sev*100).toFixed(0)}%
        ${isWater ? '<br/><span style="color:#38bdf8;font-weight:700;">⚠ Water Hazard</span>' : ''}
      </div>
    `);
  });

  /* Fit map to markers */
  if (defects.length > 0) {
    const lats = defects.map(d => parseFloat(d.latitude || d.lat || DEFAULT_LAT));
    const lons = defects.map(d => parseFloat(d.longitude || d.lon || DEFAULT_LON));
    const bounds = [[Math.min(...lats)-0.001, Math.min(...lons)-0.001],
                    [Math.max(...lats)+0.001, Math.max(...lons)+0.001]];
    map.fitBounds(bounds, { maxZoom: 17, padding: [40, 40] });

    const cLat = (Math.min(...lats)+Math.max(...lats))/2;
    const cLon = (Math.min(...lons)+Math.max(...lons))/2;
    document.getElementById('gps-coords').innerText = `${cLat.toFixed(5)}° N, ${cLon.toFixed(5)}° E`;
  }
}

function updateGeofenceRadius(val) {
  currentRadius = parseFloat(val);
  document.getElementById('radius-val').innerText = val + 'm';
  renderDefects(cachedDefects);
  if (driverMarker) {
    const pos = driverMarker.getLatLng();
    checkDriverProximity(pos.lat, pos.lng);
  }
}

/* Driver simulation */
function toggleDriveSimulation() {
  const btn = document.getElementById('sim-btn-text');
  if (driverSimInterval) {
    clearInterval(driverSimInterval); driverSimInterval = null;
    btn.innerText = '▶ Simulate Driver';
  } else {
    btn.innerText = '⏸ Pause Cruise';
    if (!driverMarker && cachedDefects.length > 0) {
      const startLat = parseFloat(cachedDefects[0].latitude || DEFAULT_LAT) - 0.001;
      const startLon = parseFloat(cachedDefects[0].longitude || DEFAULT_LON) - 0.001;
      const carIcon = L.divIcon({ className:'', html:'<div style="font-size:22px;filter:drop-shadow(0 0 8px #00f2fe);">🚗</div>', iconSize:[28,28], iconAnchor:[14,14] });
      driverMarker = L.marker([startLat, startLon], { icon: carIcon, draggable: true }).addTo(driverLayer);
      driverMarker.on('drag', e => checkDriverProximity(e.target.getLatLng().lat, e.target.getLatLng().lng));
    }
    driverSimInterval = setInterval(() => {
      if (!driverMarker || cachedDefects.length === 0) return;
      driverPosIdx = (driverPosIdx + 1) % 12;
      const ang = (driverPosIdx / 12) * 2 * Math.PI;
      const r = 0.0015;
      const cLat = parseFloat(cachedDefects[0].latitude || DEFAULT_LAT);
      const cLon = parseFloat(cachedDefects[0].longitude || DEFAULT_LON);
      const lat = cLat + r * Math.sin(ang);
      const lon = cLon + r * Math.cos(ang);
      driverMarker.setLatLng([lat, lon]);
      checkDriverProximity(lat, lon);
    }, 1100);
  }
}

async function checkDriverProximity(lat, lon) {
  try {
    const res = await fetch('/api/geofence/check_proximity', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ latitude: lat, longitude: lon, warning_radius_m: currentRadius, critical_radius_m: Math.max(15, currentRadius*0.4) })
    });
    const d = await res.json();
    const banner = document.getElementById('proximity-banner');
    const txt = document.getElementById('alert-text-content');
    const distEl = document.getElementById('nearest-hazard-dist');
    if (d.alert_triggered) {
      banner.className = d.hazard_level === 'critical' ? 'critical' : 'warning';
      txt.innerHTML = `<b>${d.status} ALERT:</b> ${d.message}`;
      distEl.innerText = `Nearest: ${d.nearest_distance_m.toFixed(1)}m`;
    } else {
      banner.className = '';
      txt.innerHTML = `<b>Driver Proximity HUD:</b> Corridor clear — no hazards within ${currentRadius}m.`;
      distEl.innerText = `Range: Safe (>${currentRadius}m)`;
    }
  } catch(_) {}
}

/* ═══════════════════════════════════════
   PATCH INSPECTION TAB
═══════════════════════════════════════ */
function sevColor(score) {
  if (score >= 0.85) return 'var(--red)';
  if (score >= 0.65) return 'var(--orange)';
  if (score >= 0.35) return 'var(--yellow)';
  return 'var(--green)';
}
function sevLabel(score) {
  if (score >= 0.85) return 'Critical';
  if (score >= 0.65) return 'High';
  if (score >= 0.35) return 'Medium';
  return 'Low';
}

function buildPatchCard(d) {
  const pid = d.pothole_id || d.defect_id || '?';
  const defType = (d.defect_type || 'pothole').replace(/_/g,' ');
  const sev = parseFloat(d.severity_score || 0);
  const conf = parseFloat(d.pothole_confidence || d.confidence || 0);
  const area = d.area_m2 != null ? parseFloat(d.area_m2).toFixed(2) : '—';
  const depth = d.estimated_depth_m != null ? (parseFloat(d.estimated_depth_m)*100).toFixed(1) + ' cm' : '—';
  const isWater = !!(d.is_water_filled || d.water_flag);
  const bboxColor = isWater ? '#38bdf8' : sevColor(sev);
  const sb = d.severity_breakdown || {};
  const mergedCount = d.merged_count || (d.all_bboxes ? d.all_bboxes.length : 1);
  const dedupBadge = mergedCount > 1
    ? `<span class="dedup-badge" style="background:rgba(59,130,246,.2);color:#60a5fa;border:1px solid rgba(59,130,246,.4);">⊕ ${mergedCount} Photos Merged (&lt;3m)</span>`
    : `<span style="font-size:10px;color:var(--text-muted);font-weight:600;">1 Photo Ingested</span>`;

  const card = document.createElement('div');
  card.className = 'patch-card';
  card.innerHTML = `
    <div class="patch-img-wrap" id="wrap-${pid}">
      <img id="img-${pid}"
           src="/api/patch_image/${encodeURIComponent(pid)}"
           alt="Patch ${pid}"
           style="max-height:260px;"
           onerror="this.src='/api/overlays/input_frame.jpg'" />
      <canvas id="canvas-${pid}"></canvas>
    </div>
    <div class="patch-meta">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span class="patch-id">${pid}</span>
        <span class="patch-defect-badge" style="background:${bboxColor}22;color:${bboxColor};border:1px solid ${bboxColor}44;">${defType}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px;">
        ${dedupBadge}
        <span style="font-size:11px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;">Conf: ${(conf*100).toFixed(0)}%</span>
      </div>
      <table class="metrics-table">
        <thead><tr><th>Metric</th><th>Value</th><th>Score</th></tr></thead>
        <tbody>
          <tr>
            <td>Depth</td>
            <td class="val">${depth}</td>
            <td>
              <div class="sev-bar-wrap">
                <div style="flex:1;" class="progress-track"><div class="progress-fill" style="width:${Math.round((parseFloat(sb.depth||0))*100)}%;background:var(--cyan);"></div></div>
                <span style="font-size:10px;color:var(--text-muted);min-width:28px;">${((parseFloat(sb.depth||0))*100).toFixed(0)}%</span>
              </div>
            </td>
          </tr>
          <tr>
            <td>Area</td>
            <td class="val">${area} m²</td>
            <td>
              <div class="sev-bar-wrap">
                <div style="flex:1;" class="progress-track"><div class="progress-fill" style="width:${Math.round((parseFloat(sb.area||0))*100)}%;background:var(--purple);"></div></div>
                <span style="font-size:10px;color:var(--text-muted);min-width:28px;">${((parseFloat(sb.area||0))*100).toFixed(0)}%</span>
              </div>
            </td>
          </tr>
          <tr>
            <td>Severity Score</td>
            <td class="val" style="color:${sevColor(sev)};">${(sev*100).toFixed(1)}%</td>
            <td>
              <div class="sev-bar-wrap">
                <span class="sev-label" style="color:${sevColor(sev)};">${sevLabel(sev)}</span>
                <div style="flex:1;" class="progress-track"><div class="progress-fill" style="width:${Math.round(sev*100)}%;background:${sevColor(sev)};"></div></div>
              </div>
            </td>
          </tr>
          ${isWater ? `<tr><td colspan="3" style="color:#38bdf8;font-size:11px;font-weight:700;">⚠ Water-filled cavity — hydroplaning risk</td></tr>` : ''}
        </tbody>
      </table>
      <div style="display:flex;gap:12px;font-size:11px;color:var(--text-muted);">
        <span>GPS: <b style="color:#e2e8f0;">${parseFloat(d.latitude||0).toFixed(5)}°, ${parseFloat(d.longitude||0).toFixed(5)}°</b></span>
      </div>
    </div>
  `;
  return card;
}

/* Offline File Upload Handler for local testing */
async function handleMockFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async function() {
    const b64 = reader.result.split(',')[1];
    const baseLat = cachedDefects.length > 0 ? parseFloat(cachedDefects[0].latitude || 13.0827) : 13.0827;
    const baseLon = cachedDefects.length > 0 ? parseFloat(cachedDefects[0].longitude || 80.2707) : 80.2707;

    const isOverlap = confirm("Simulate spatial overlap (<3m)?\n\n[OK] = Place within 1.2m of existing defect to test 3m Deduplication.\n[Cancel] = Place as a new defect cluster 35m away.");
    const lat = isOverlap ? (baseLat + 0.000010) : (baseLat + (Math.random() * 0.0006 - 0.0003));
    const lon = isOverlap ? (baseLon + 0.000010) : (baseLon + (Math.random() * 0.0006 - 0.0003));

    const area = parseFloat((0.4 + Math.random() * 1.2).toFixed(2));
    const depth = parseFloat((0.05 + Math.random() * 0.12).toFixed(3));
    const sev = parseFloat((Math.min(0.96, Math.max(0.60, area/2.0 + depth/0.25))).toFixed(2));
    const conf = parseFloat((0.80 + Math.random() * 0.18).toFixed(2));
    const isWater = Math.random() < 0.4;

    const payload = {
      pothole_id: 'mock-upload-' + Math.random().toString(36).substr(2, 6),
      latitude: lat,
      longitude: lon,
      bbox_xyxy: [100, 120, 380, 320],
      area_m2: area,
      estimated_depth_m: depth,
      severity_score: sev,
      confidence: conf,
      pothole_confidence: conf,
      defect_type: isWater ? 'water_filled_pothole' : 'pothole',
      is_water_filled: isWater,
      water_flag: isWater,
      _image_b64: b64,
    };

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.ok) {
        alert(`Ingestion Successful!\n\nStatus: ${data.dedup_status.toUpperCase()}\nCluster ID: ${data.cluster_id}\nDistance: ${data.distance_m != null ? data.distance_m + 'm' : 'N/A'}\nActive Clusters: ${data.total_clusters}\nActive Work Orders: ${data.total_work_orders}`);
        await fetchAll();
        switchTab('patches');
      }
    } catch(err) {
      alert('Upload failed: ' + err.message);
    }
  };
  reader.readAsDataURL(file);
  event.target.value = '';
}

function renderPatchTab(defects) {
  const grid = document.getElementById('patch-grid');
  const badge = document.getElementById('patch-count-badge');
  grid.innerHTML = '';
  if (!defects || defects.length === 0) {
    grid.innerHTML = '<div class="empty-state"><div class="icon">🔍</div>No defects found. Use /api/ingest or run the pipeline.</div>';
    badge.innerText = '0 patches';
    return;
  }
  badge.innerText = `${defects.length} patch${defects.length !== 1 ? 'es' : ''}`;
  defects.forEach(d => grid.appendChild(buildPatchCard(d)));
}

/* ═══════════════════════════════════════
   WORK ORDERS TABLE TAB
═══════════════════════════════════════ */
function renderWorkOrdersTab(workOrders) {
  const tbody = document.getElementById('wo-tbody');
  const badge = document.getElementById('wo-total-badge');
  tbody.innerHTML = '';
  badge.innerText = `${workOrders.length} Active`;

  if (workOrders.length === 0) {
    tbody.innerHTML = '<tr><td colspan="11" class="empty-state">No critical/high work orders. All segments clear.</td></tr>';
    return;
  }

  workOrders.forEach(wo => {
    const sev = (wo.severity_tier || 'low').toLowerCase();
    const sevCls = 'sev-' + sev;
    const area = wo.area_m2 != null ? parseFloat(wo.area_m2).toFixed(2) : '—';
    const depth = wo.estimated_depth_m != null ? (parseFloat(wo.estimated_depth_m)*100).toFixed(1) : '—';
    const mats = (wo.required_materials || []).map(m => `<span class="mat-tag">${m}</span>`).join('');
    const fullText = wo.work_order_text || '—';
    const preview = fullText.length > 120 ? fullText.slice(0, 120) + '…' : fullText;
    const tid = 'wo-txt-' + (wo.work_order_id || Math.random()).replace(/[^a-z0-9]/gi, '');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="wo-id-cell">${wo.work_order_id || '—'}</td>
      <td style="font-size:12px;color:#94a3b8;">${wo.road_segment_id || '—'}</td>
      <td style="font-size:12px;">${(wo.defect_class || '—').replace(/_/g,' ')}</td>
      <td><span class="sev-badge ${sevCls}">${sev}</span></td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px;">${area}</td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px;">${depth}</td>
      <td style="font-size:12px;">${wo.water_hazard || wo.is_water_filled ? '💧 Yes' : 'No'}</td>
      <td style="font-size:12px;">${wo.estimated_crew_size || '—'} tech${(wo.estimated_crew_size||1)!==1?'s':''}</td>
      <td style="font-size:12px;">${wo.target_resolution_hours || '—'}h</td>
      <td class="wo-text-cell">
        <span id="${tid}">${preview}</span>
        ${fullText.length > 120 ? `<br/><button class="wo-expand-btn" onclick="expandWo('${tid}', \`${fullText.replace(/`/g,"'")}\`)">Read more ↓</button>` : ''}
      </td>
      <td><div class="materials-list">${mats || '<span class="mat-tag">—</span>'}</div></td>
    `;
    tbody.appendChild(tr);
  });
}

function expandWo(id, fullText) {
  const el = document.getElementById(id);
  if (el) { el.innerText = fullText; el.nextElementSibling && el.nextElementSibling.remove(); }
}

/* ═══════════════════════════════════════
   Master Fetch
═══════════════════════════════════════ */
async function fetchAll(showRefreshMsg = false) {
  try {
    const [statsRes, resultsRes, woRes] = await Promise.all([
      fetch('/api/stats').then(r => r.json()),
      fetch('/api/results').then(r => r.json()),
      fetch('/api/work_orders').then(r => r.json()),
    ]);

    /* KPIs */
    document.getElementById('kpi-score').innerHTML =
      `${statsRes.road_health_score}<span style="font-size:15px;color:var(--text-muted);font-weight:500;">/100</span>`;
    document.getElementById('kpi-segment-id').innerText =
      `Segment: ${statsRes.road_segment_id} (${statsRes.map_name || 'Town04'})`;
    document.getElementById('kpi-defects').innerText =
      `${statsRes.total_defects} (${statsRes.critical_hazards} crit.)`;
    document.getElementById('kpi-water').innerText = statsRes.water_hazards;
    document.getElementById('kpi-workorders').innerText = statsRes.work_orders_count;
    document.getElementById('kpi-pred').innerHTML =
      `${Math.round((statsRes.deterioration_probability||0)*100)}<span style="font-size:15px;font-weight:500;">%</span>`;
    document.getElementById('dedup-count').innerText = statsRes.active_clusters || '—';

    /* Condition badge */
    const cb = document.getElementById('kpi-cond-badge');
    cb.innerText = statsRes.condition_class;
    const s = statsRes.road_health_score;
    if (s >= 80) { cb.style.background='rgba(16,185,129,.15)'; cb.style.color='var(--green)'; }
    else if (s >= 60) { cb.style.background='rgba(245,158,11,.15)'; cb.style.color='var(--yellow)'; }
    else if (s >= 40) { cb.style.background='rgba(249,115,22,.15)'; cb.style.color='var(--orange)'; }
    else { cb.style.background='rgba(239,68,68,.15)'; cb.style.color='var(--red)'; }

    /* Score breakdown */
    const rh = resultsRes.road_health || {};
    const comp = rh.components || {};
    const getP = (k1, k2) => parseFloat(comp[k1] || comp[k2] || 0);
    const pSev = getP('pothole_severity_penalty', 'pothole_penalty');
    const pCnt = getP('pothole_count_penalty', 'count_penalty');
    const pWat = getP('water_hazard_penalty', 'water_penalty');
    const pSurf = getP('surface_roughness_penalty', 'surface_penalty');
    const setBar = (idTxt, idBar, val) => {
      document.getElementById(idTxt).innerText = `-${val.toFixed(1)} pts`;
      document.getElementById(idBar).style.width = Math.min(100, val/30*100) + '%';
    };
    setBar('sev-penalty-txt','sev-penalty-bar', pSev);
    setBar('cnt-penalty-txt','cnt-penalty-bar', pCnt);
    setBar('water-penalty-txt','water-penalty-bar', pWat);
    setBar('surf-penalty-txt','surf-penalty-bar', pSurf);
    if (rh.explanation) document.getElementById('explanation-text').innerText = 'Explanation: ' + rh.explanation;

    /* Prediction */
    const pred = resultsRes.prediction || {};
    const dp = parseFloat(pred.deterioration_probability || 0);
    document.getElementById('detp-score').innerText = (dp*100).toFixed(0) + '%';
    document.getElementById('detp-bar').style.width = (dp*100) + '%';
    document.getElementById('horizon-days').innerText = pred.prediction_horizon_days || pred.horizon_days || 30;
    document.getElementById('prog-direction').innerText = (pred.progression_direction || pred.progression_trend || '—').toUpperCase();

    /* Map */
    cachedDefects = resultsRes.detections || resultsRes.potholes || [];
    renderDefects(cachedDefects);

    /* Patch Inspection */
    renderPatchTab(cachedDefects);

    /* Work Orders */
    renderWorkOrdersTab(woRes.work_orders || []);

    if (showRefreshMsg) console.log('[RoadSentinel] Dashboard refreshed.');
  } catch (err) {
    console.error('[RoadSentinel] Fetch error:', err);
  }
}

fetchAll();
setInterval(fetchAll, 6000);
</script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def get_dashboard_ui(request: Request):
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RoadSentinel Dashboard & Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    log.info(
        "Starting RoadSentinel Dashboard v3.0 on http://%s:%d  |  Dedup radius: %.1fm",
        args.host, args.port, DEDUP_RADIUS_M,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
