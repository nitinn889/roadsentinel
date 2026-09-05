#!/usr/bin/env python3
"""RoadSentinel Real CARLA Image Ingestion & Analytics Pipeline.

Ingests actual CARLA-generated drone photos from env/output/images (and metadata.csv),
runs real neural inference (DINOv2 + SAM2 + defect classification + physical area/depth/severity),
projects defect ground positions to WGS-84 GPS coordinates, applies Marion's 3-meter spatial deduplication,
generates repair work orders, and exports result.json and work_orders.json for the Government Dashboard.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG, Config
from common.schemas import (
    CandidateRegion,
    DefectMeasurement,
    DefectType,
    InferenceResult,
    PotholeRecord,
    PredictionResult,
    RoadHealthResult,
    SeverityBreakdown,
    Telemetry,
)
from common.io_utils import load_rgb, save_json, utc_iso
from analytics.severity import calculate_defect_severity, classify_severity_label
from analytics.road_health import calculate_road_health_score
from analytics.prediction import RoadDeteriorationPredictor, SegmentObservation
from inference.run_inference import load_pipeline, infer, generate_visual_overlays, water_heuristic, PipelineComponents
from inference.gps_localizer import GPSLocalizer, telemetry_from_dict
from inference.area_estimator import estimate_area_m2
from inference.server import SpatialDeduplicator, DEDUP_RADIUS_M

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("carla_pipeline")


def calculate_defect_ground_gps(
    bbox: list[int],
    img_w: int,
    img_h: int,
    telemetry: Telemetry,
    localizer: GPSLocalizer,
    horizontal_fov_deg: float = 60.0,
) -> tuple[float, float, float, float]:
    """Project bounding box centroid to metric ground coordinates and global GPS."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    alt_m = max(1.0, float(telemetry.altitude_m or CONFIG.altitude_m or 100.0))
    hfov = math.radians(horizontal_fov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (img_h / max(1.0, float(img_w))))

    # Offset from optical center in meters (nadir projection)
    dx_m = (cx - img_w / 2.0) * (2.0 * alt_m * math.tan(hfov / 2.0) / img_w)
    dy_m = (cy - img_h / 2.0) * (2.0 * alt_m * math.tan(vfov / 2.0) / img_h)

    world_x = float(telemetry.world_x or 0.0)
    world_y = float(telemetry.world_y or 0.0)
    yaw_deg = float(telemetry.heading_deg or 0.0)
    rad_yaw = math.radians(yaw_deg)
    cos_y = math.cos(rad_yaw)
    sin_y = math.sin(rad_yaw)

    # In camera frame: dx_m is right (+x_cam), dy_m is down (+y_cam, so forward is -dy_m)
    fwd_m = -dy_m
    right_m = dx_m

    # Rotate into CARLA world coordinates
    def_world_x = world_x + (fwd_m * cos_y - right_m * sin_y)
    def_world_y = world_y + (fwd_m * sin_y + right_m * cos_y)

    def_lat, def_lon = localizer.carla_world_to_gps(def_world_x, def_world_y)
    return def_lat, def_lon, def_world_x, def_world_y


def crop_patch_image(
    image_path: Path,
    bbox: list[int],
    padding: int = 40,
) -> tuple[bytes, list[int]]:
    """Crop defect bounding box from the image with padding and return JPEG bytes and local bbox."""
    with Image.open(image_path) as pil_img:
        rgb_img = pil_img.convert("RGB")
        w, h = rgb_img.size
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(w, x2 + padding)
        y2p = min(h, y2 + padding)

        crop = rgb_img.crop((x1p, y1p, x2p, y2p))
        local_bbox = [x1 - x1p, y1 - y1p, x2 - x1p, y2 - y1p]

        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), local_bbox


def post_detection_to_server(server_url: str, payload: dict) -> Optional[dict]:
    """Post detection payload to active dashboard server /api/ingest endpoint."""
    try:
        url = server_url.rstrip("/") + "/api/ingest"
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def run_carla_pipeline(
    image_input_dir: Path,
    output_dir: Path,
    server_url: Optional[str] = "http://localhost:8000",
    device: str = CONFIG.device,
    memory_bank_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    stride: int = 1,
) -> Path:
    """Execute real ML detection and spatial deduplication on CARLA captured images."""
    image_input_dir = Path(image_input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate images directory
    img_dir = image_input_dir / "images" if (image_input_dir / "images").is_dir() else image_input_dir
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    log.info("Found %d CARLA image(s) in %s", len(images), img_dir)

    # 2. Parse telemetry from metadata.csv if available
    metadata_csv = image_input_dir / "metadata.csv"
    if not metadata_csv.is_file() and (image_input_dir.parent / "metadata.csv").is_file():
        metadata_csv = image_input_dir.parent / "metadata.csv"

    meta_rows: dict[str, dict] = {}
    if metadata_csv.is_file():
        log.info("Loading telemetry from: %s", metadata_csv)
        with open(metadata_csv, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta_rows[row["image_name"]] = row
        if meta_rows:
            images = [img for img in images if img.name in meta_rows]
            log.info("Filtered to %d image(s) matching active telemetry in %s", len(images), metadata_csv.name)

    # 3. Initialize models & deduplicator
    if memory_bank_dir is None:
        cand_mb = ROOT / "output" / "real_memory_bank"
        memory_bank_dir = cand_mb if cand_mb.is_dir() else CONFIG.memory_bank_dir

    log.info("Loading RoadSentinel models on device '%s' (memory bank: %s)...", device, memory_bank_dir)
    pipeline = load_pipeline(device=device, memory_bank_dir=memory_bank_dir)
    dedup = SpatialDeduplicator()
    localizer = GPSLocalizer(CONFIG.carla_origin_lat, CONFIG.carla_origin_lon)

    # Filter/stride image selection if specified
    selected_images = images[::stride]
    if max_images and len(selected_images) > max_images:
        selected_images = selected_images[:max_images]

    log.info("Processing %d image(s) with DINOv2 patch features & SAM2 segmentation...", len(selected_images))

    raw_detections_count = 0
    primary_overlay_img: Optional[np.ndarray] = None
    primary_overlay_records: list[PotholeRecord] = []
    primary_road_health = None
    best_defect_conf = 0.0

    # 4. Image Processing Loop
    for idx, img_path in enumerate(selected_images):
        row = meta_rows.get(img_path.name, {})
        x_m = float(row.get("local_x_m", 0.0))
        y_m = float(row.get("local_y_m", 0.0))
        alt_m = float(row.get("altitude_m", CONFIG.altitude_m or 30.0))
        raw_lat = float(row.get("latitude", 0.0))
        raw_lon = float(row.get("longitude", 0.0))

        # Georeference to reference datum if raw lat/lon is near equator / simulation origin
        if abs(raw_lat) < 1.0 and (x_m != 0.0 or y_m != 0.0):
            frame_lat, frame_lon = localizer.carla_world_to_gps(x_m, y_m)
        elif abs(raw_lat) >= 1.0:
            frame_lat, frame_lon = raw_lat, raw_lon
        else:
            frame_lat, frame_lon = CONFIG.carla_origin_lat, CONFIG.carla_origin_lon

        yaw_val = float(row.get("yaw_deg", 0.0))
        telem = Telemetry(
            timestamp=str(row.get("sim_time_s", str(idx))),
            latitude=frame_lat,
            longitude=frame_lon,
            altitude_m=alt_m,
            heading_deg=yaw_val,
            world_x=x_m,
            world_y=y_m,
            frame_id=img_path.stem,
        )

        res = infer(img_path, pipeline=pipeline)

        if res.detections:
            rgb_arr = load_rgb(img_path)
            h, w = rgb_arr.shape[:2]

            for d in res.detections:
                raw_detections_count += 1
                def_lat, def_lon, _, _ = calculate_defect_ground_gps(
                    d.bbox, w, h, telem, localizer, horizontal_fov_deg=60.0
                )

                patch_bytes, local_bbox = crop_patch_image(img_path, d.bbox, padding=40)
                cid_name = f"carla-{img_path.stem}-{d.defect_id}"

                det_payload = {
                    "pothole_id": cid_name,
                    "defect_id": cid_name,
                    "road_segment_id": "seg_carla_town04_0042",
                    "latitude": round(def_lat, 6),
                    "longitude": round(def_lon, 6),
                    "bbox_xyxy": d.bbox,
                    "local_bbox_xyxy": local_bbox,
                    "area_m2": round(d.estimated_area_m2 or 0.45, 2),
                    "estimated_depth_m": round(d.estimated_depth_m or 0.08, 3),
                    "severity_score": round(d.severity.severity_score if hasattr(d.severity, "severity_score") else 0.70, 2),
                    "confidence": round(d.confidence, 2),
                    "pothole_confidence": round(d.confidence, 2),
                    "defect_type": d.defect_type,
                    "is_water_filled": d.is_water_filled,
                    "water_flag": d.is_water_filled,
                    "source_image": img_path.name,
                    "image_path": str(img_path.resolve()),
                    "_image_b64": base64.b64encode(patch_bytes).decode("ascii"),
                    "severity_breakdown": {
                        "area": round(float(d.severity.severity_components.get("area") or 0.35) if hasattr(d.severity, "severity_components") else 0.35, 2),
                        "depth": round(float(d.severity.severity_components.get("depth") or 0.30) if hasattr(d.severity, "severity_components") else 0.30, 2),
                        "water": round(float(d.severity.severity_components.get("water") or 0.20) if hasattr(d.severity, "severity_components") else 0.20, 2),
                        "confidence": round(float(d.confidence or 0.8), 2),
                    },
                }

                # Ingest into Spatial Deduplicator
                ingest_res = dedup.ingest(det_payload, image_bytes=patch_bytes)
                cid = ingest_res["cluster_id"]
                status = ingest_res["status"].upper()
                dist_str = f"({ingest_res.get('distance_m')}m to canonical)" if status == "DEDUPLICATED" else "(new physical site)"
                log.info("  [%s] %s -> %s at (%.5f, %.5f) %s", status, img_path.name, d.defect_type, def_lat, def_lon, dist_str)

                # If server is running, forward to /api/ingest
                if server_url:
                    post_detection_to_server(server_url, det_payload)

            if d.confidence > best_defect_conf:
                best_defect_conf = d.confidence
                primary_overlay_img = rgb_arr
                primary_overlay_records = res.potholes
                primary_road_health = res.road_health

    log.info("Inference complete. Raw defect detections: %d -> Unique clusters after 3m dedup: %d",
             raw_detections_count, dedup.cluster_count())

    # 5. Extract canonical deduplicated detections and work orders
    canonical_clusters = dedup.get_all_detections()
    work_orders = dedup.get_all_work_orders()

    # Save canonical patch images
    for c in canonical_clusters:
        cid = c.get("pothole_id") or c.get("defect_id")
        img_bytes = dedup.get_patch_image(cid)
        if img_bytes:
            (patches_dir / f"{cid}.jpg").write_bytes(img_bytes)

    # 6. Aggregate Road Health & Prediction
    pothole_records: list[PotholeRecord] = []
    for c in canonical_clusters:
        pothole_records.append(PotholeRecord(
            pothole_id=c.get("pothole_id") or c.get("defect_id"),
            timestamp=utc_iso(),
            latitude=float(c.get("latitude") or CONFIG.carla_origin_lat),
            longitude=float(c.get("longitude") or CONFIG.carla_origin_lon),
            altitude_m=float(CONFIG.altitude_m or 30.0),
            area_m2=float(c.get("area_m2") or 0.5),
            estimated_depth_m=float(c.get("estimated_depth_m") or 0.08),
            anomaly_score=float(c.get("confidence") or 0.8),
            pothole_confidence=float(c.get("pothole_confidence") or c.get("confidence") or 0.8),
            severity_score=float(c.get("severity_score") or 0.7),
            water_flag=bool(c.get("is_water_filled") or c.get("water_flag")),
            water_confidence=0.85 if c.get("is_water_filled") else 0.1,
            source_image=str(c.get("source_image") or "carla_frame.jpg"),
            mask_area_px=int((c.get("area_m2") or 0.5) * 1000),
            bbox_xyxy=c.get("bbox_xyxy") or [100, 100, 200, 200],
            defect_type=c.get("defect_type") or "pothole",
            road_segment_id=c.get("road_segment_id") or "seg_carla_town04_0042",
            severity_breakdown=c.get("severity_breakdown") or {},
        ))

    road_health = calculate_road_health_score(pothole_records, total_crack_area_m2=0.0)

    # 30-day temporal prediction
    predictor = RoadDeteriorationPredictor()
    obs = SegmentObservation(
        timestamp=utc_iso(),
        road_health_score=road_health.road_health_score,
        pothole_count=len(canonical_clusters),
        total_defects=len(canonical_clusters),
        damaged_area_m2=float(sum(float(c.get("area_m2") or 0.0) for c in canonical_clusters)),
        max_severity=float(max((float(c.get("severity_score") or 0.0) for c in canonical_clusters), default=0.0)),
        avg_severity=float(np.mean([float(c.get("severity_score") or 0.0) for c in canonical_clusters])) if canonical_clusters else 0.0,
        has_water_hazard=any(bool(c.get("is_water_filled")) for c in canonical_clusters),
    )
    prediction = predictor.predict([obs], road_segment_id="seg_carla_town04_0042")

    # 7. Generate diagnostic overlays
    if primary_overlay_img is not None:
        overlays = generate_visual_overlays(primary_overlay_img, primary_overlay_records, road_health)
        cv2.imwrite(str(output_dir / "detection_overlay.jpg"), overlays["detection_overlay"])
        cv2.imwrite(str(output_dir / "severity_overlay.jpg"), overlays["severity_overlay"])
        cv2.imwrite(str(output_dir / "road_health_overlay.jpg"), overlays["road_health_overlay"])
        cv2.imwrite(str(output_dir / "input_frame.jpg"), cv2.cvtColor(primary_overlay_img, cv2.COLOR_RGB2BGR))
    elif selected_images:
        # Fallback to saving first image as input_frame.jpg
        first_img = load_rgb(selected_images[0])
        cv2.imwrite(str(output_dir / "input_frame.jpg"), cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR))

    # 8. Export result.json
    center_lat = float(np.mean([c["latitude"] for c in canonical_clusters])) if canonical_clusters else CONFIG.carla_origin_lat
    center_lon = float(np.mean([c["longitude"] for c in canonical_clusters])) if canonical_clusters else CONFIG.carla_origin_lon

    result_data = {
        "image_id": "carla_town04_flight_session",
        "timestamp": utc_iso(),
        "road_segment_id": "seg_carla_town04_0042",
        "map_name": "Town04",
        "geolocation": {"lat": round(center_lat, 6), "lon": round(center_lon, 6), "alt_m": 30.0},
        "road_health": road_health.to_dict(),
        "prediction": prediction.to_dict(),
        "detections": canonical_clusters,
        "potholes": [p.to_dict() for p in pothole_records],
        "metadata": {
            "total_carla_images_scanned": len(selected_images),
            "raw_detections_count": raw_detections_count,
            "deduplicated_clusters_count": len(canonical_clusters),
            "dedup_radius_m": DEDUP_RADIUS_M,
            "work_orders_count": len(work_orders),
        }
    }

    result_json_path = output_dir / "result.json"
    save_json(result_data, result_json_path)
    log.info("Saved result.json to: %s", result_json_path)

    # 9. Export work_orders.json
    work_orders_data = {
        "metadata": {
            "generated_at": utc_iso(),
            "generator": "RoadSentinel-SpatialDedup-VLM",
            "total_segments_with_work_orders": 1 if work_orders else 0,
            "total_critical_defects_remediated": len(work_orders),
        },
        "road_segment_id": "seg_carla_town04_0042",
        "work_orders": work_orders,
    }
    work_orders_path = output_dir / "work_orders.json"
    save_json(work_orders_data, work_orders_path)
    log.info("Saved work_orders.json (%d orders) to: %s", len(work_orders), work_orders_path)

    return result_json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Process real CARLA images into RoadSentinel analytics with 3m dedup")
    parser.add_argument("--input-dir", type=Path, default=ROOT.parent / "env" / "output",
                        help="Path to folder containing CARLA images and metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "analytics_demo",
                        help="Path to save result.json and work_orders.json")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="Dashboard server URL to forward detections")
    parser.add_argument("--device", type=str, default=CONFIG.device,
                        help="Computation device (cuda/cpu)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images processed (for fast debug)")
    args = parser.parse_args()

    run_carla_pipeline(
        image_input_dir=args.input_dir,
        output_dir=args.output_dir,
        server_url=args.server,
        device=args.device,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
