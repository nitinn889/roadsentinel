"""Citizen Road-Damage Reporting and AI Verification Engine.

This module provides the core AI perception, domain verification, multi-factor severity,
reliability calibration, GPS corridor mapping, and local report persistence for citizen reports.

Pipeline:
1. Image validation
2. YOLO road-defect detection (Pothole D40 vs Other Road Defects D00/D10/D20/Repair)
3. DINOv2 domain-familiarity check (cosine kNN distance to training reference embeddings)
4. SAM2 instance segmentation on detected defects
5. Multi-factor model-derived severity calculation
6. Reliability scoring (logistic confidence calibration + domain gate)
7. Final report outcome decision (VERIFIED REPORT, NEEDS MANUAL REVIEW, RETAKE / NO VERIFIED DAMAGE)
8. Persistent local storage in driver_dashboard/data/citizen_reports/
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO

from config import (
    ROAD_GEOFENCE_RADIUS_M,
    START_LATITUDE,
    START_LONGITUDE,
    TOTAL_ROAD_LENGTH_M,
)
from gps import check_geofence, haversine_distance
from route import generate_deterministic_route, get_segment_at_distance

log = logging.getLogger("citizen_reporting")

# Directory Paths
DRIVER_DASHBOARD_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = DRIVER_DASHBOARD_DIR.parent
REPORTS_DIR = DRIVER_DASHBOARD_DIR / "data" / "citizen_reports"
REPORTS_MANIFEST_JSON = REPORTS_DIR / "reports_manifest.json"
REPORTS_MANIFEST_CSV = REPORTS_DIR / "reports_manifest.csv"

# Model Artifact Paths
YOLO_WEIGHTS_PATH = WORKSPACE_ROOT / "yolo" / "weights" / "best.pt"
DINO_REFERENCE_PATH = WORKSPACE_ROOT / "cross_domain" / "results" / "train_domain_reference_embeddings.npz"
SAM2_CHECKPOINT_PATH = WORKSPACE_ROOT / "road_health_pipeline" / "checkpoints" / "sam2.1_hiera_small.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Frozen Domain Thresholds (from Phase 8 / Phase 10 / Phase 13)
CHINA_TRAIN_P95_KNN = 0.3804
CHINA_TRAIN_P99_KNN = 0.4491
RELIABILITY_HIGH_THRESH = 0.85
RELIABILITY_LOW_THRESH = 0.60

# Decision States
STATUS_VERIFIED = "VERIFIED REPORT"
STATUS_MANUAL_REVIEW = "NEEDS MANUAL REVIEW"
STATUS_RETAKE = "RETAKE / NO VERIFIED DAMAGE"

# Class Mapping for YOLO
CLASS_POTHOLE = "Pothole"
CLASS_ROAD_DEFECT = "Road Defect"

YOLO_CLASS_NAME_MAP = {
    "D40": CLASS_POTHOLE,
    "D00": CLASS_ROAD_DEFECT,  # Longitudinal Crack
    "D10": CLASS_ROAD_DEFECT,  # Transverse Crack
    "D20": CLASS_ROAD_DEFECT,  # Alligator Crack
    "Repair": "Repair / Patch",
}

CLASS_COLORS = {
    CLASS_POTHOLE: (0, 0, 255),       # Red (BGR)
    CLASS_ROAD_DEFECT: (0, 165, 255),  # Orange (BGR)
    "Repair / Patch": (0, 255, 0),    # Green (BGR)
}
DEFAULT_BOX_COLOR = (255, 255, 0)     # Cyan


class CitizenReportingPipeline:
    """Singleton/Cached container managing AI inference models."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model: Optional[YOLO] = None
        self.dino_model: Optional[Any] = None
        self.dino_transform: Optional[Any] = None
        self.dino_ref_embeddings: Optional[np.ndarray] = None
        self.sam2_predictor: Optional[Any] = None
        self.sam2_available: bool = False

        self._load_models()

    def _load_models(self) -> None:
        """Load YOLO, DINOv2, and SAM2 models safely."""
        # 1. Load YOLO
        if YOLO_WEIGHTS_PATH.exists():
            try:
                log.info("Loading YOLO weights from %s", YOLO_WEIGHTS_PATH)
                self.yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
            except Exception as e:
                log.error("Failed to load YOLO model: %s", e)
                self.yolo_model = None
        else:
            log.warning("YOLO weights not found at %s", YOLO_WEIGHTS_PATH)

        # 2. Load DINOv2 & Reference Embeddings
        if DINO_REFERENCE_PATH.exists():
            try:
                raw_refs = np.load(DINO_REFERENCE_PATH)["embeds"]
                self.dino_ref_embeddings = raw_refs / (np.linalg.norm(raw_refs, axis=-1, keepdims=True) + 1e-12)
                log.info("Loaded %d DINO reference embeddings", len(self.dino_ref_embeddings))
            except Exception as e:
                log.error("Failed to load DINO reference embeddings: %s", e)
                self.dino_ref_embeddings = None

        try:
            self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.dino_model.to(self.device)
            self.dino_model.eval()
            self.dino_transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            log.info("DINOv2 ViT-S/14 initialized on %s", self.device)
        except Exception as e:
            log.warning("Could not load DINOv2 model via torch hub: %s. Using heuristic fallback.", e)
            self.dino_model = None

        # 3. Load SAM2
        if SAM2_CHECKPOINT_PATH.exists():
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                sam2_model = build_sam2(SAM2_CONFIG_NAME, str(SAM2_CHECKPOINT_PATH), device=self.device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                self.sam2_available = True
                log.info("SAM2 2.1 Hiera-Small initialized on %s", self.device)
            except Exception as e:
                log.warning("SAM2 initialization skipped: %s", e)
                self.sam2_predictor = None
                self.sam2_available = False


def validate_image_bytes(file_bytes: bytes) -> Tuple[bool, Optional[np.ndarray], str]:
    """Validate uploaded file bytes as a valid decodable RGB image.

    Returns:
        (is_valid, cv2_bgr_image, error_or_success_message)
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, None, "Uploaded file is empty."

    try:
        # Check image using PIL first
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()

        # Decode using OpenCV for inference
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            return False, None, "Could not decode image content. File may be corrupted or unsupported format."

        h, w = img_bgr.shape[:2]
        if h < 64 or w < 64:
            return False, None, f"Image dimensions ({w}x{h}) are too small for reliable defect analysis."

        return True, img_bgr, "Image format validated successfully."
    except Exception as e:
        return False, None, f"Image validation failed: {str(e)}"


def compute_dino_familiarity(
    pipeline: CitizenReportingPipeline,
    img_rgb: Image.Image,
) -> Tuple[float, str, str]:
    """Compute DINOv2 kNN cosine distance to training reference embeddings.

    Returns:
        (knn_distance, familiarity_level, description)
    """
    if pipeline.dino_model is None or pipeline.dino_ref_embeddings is None:
        # Fallback estimation if torch hub or reference weights not loaded
        return 0.3500, "MODERATE", "DINO reference model unavailable; default baseline familiarity applied."

    try:
        tensor = pipeline.dino_transform(img_rgb).unsqueeze(0).to(pipeline.device)
        with torch.no_grad():
            feat = pipeline.dino_model(tensor).cpu().numpy()[0]

        feat_norm = feat / (np.linalg.norm(feat) + 1e-12)
        sims = np.dot(pipeline.dino_ref_embeddings, feat_norm)
        dists = 1.0 - sims
        knn_dist = float(np.mean(np.sort(dists)[:20]))
        knn_dist = round(knn_dist, 4)

        if knn_dist <= CHINA_TRAIN_P95_KNN:
            familiarity = "IN_DOMAIN"
            desc = f"High visual domain familiarity (d={knn_dist:.4f} <= {CHINA_TRAIN_P95_KNN:.4f})."
        elif knn_dist <= CHINA_TRAIN_P99_KNN:
            familiarity = "MODERATE"
            desc = f"Moderate visual domain familiarity (d={knn_dist:.4f} <= {CHINA_TRAIN_P99_KNN:.4f})."
        else:
            familiarity = "UNFAMILIAR_DOMAIN"
            desc = f"Domain shift detected (d={knn_dist:.4f} > {CHINA_TRAIN_P99_KNN:.4f}). Out-of-distribution visual cues."

        return knn_dist, familiarity, desc
    except Exception as e:
        log.error("DINO feature extraction error: %s", e)
        return 0.4000, "MODERATE", f"Feature distance computation failed: {str(e)}"


def run_sam2_defect_mask(
    pipeline: CitizenReportingPipeline,
    img_rgb_np: np.ndarray,
    box_xyxy: List[float],
) -> Tuple[Optional[np.ndarray], float]:
    """Run SAM2 prompt on the primary detected defect bounding box.

    Returns:
        (binary_mask_bool, mask_area_ratio)
    """
    if not pipeline.sam2_available or pipeline.sam2_predictor is None:
        return None, 0.0

    try:
        pipeline.sam2_predictor.set_image(img_rgb_np)
        box_np = np.array([box_xyxy], dtype=np.float32)

        masks, scores, _ = pipeline.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np,
            multimask_output=True,
        )
        if len(masks) == 0:
            return None, 0.0

        # Select highest-scoring mask
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)
        h, w = img_rgb_np.shape[:2]
        area_ratio = float(np.sum(best_mask) / (h * w))
        return best_mask, round(area_ratio, 6)
    except Exception as e:
        log.warning("SAM2 mask prediction failed: %s", e)
        return None, 0.0


def calculate_model_severity(
    max_conf: float,
    defect_type: str,
    area_ratio: float,
    pred_count: int,
) -> Tuple[float, str]:
    """Calculate transparent multi-factor model-derived severity in [0.0, 1.0].

    Formula:
    - Base confidence contribution (weight 0.45)
    - Defect type weight: Pothole = 1.0, Road Defect = 0.70, Repair = 0.40 (weight 0.30)
    - Area ratio scaled relative to 0.05 nominal frame area (weight 0.15)
    - Multiplicity count boost (weight 0.10)
    """
    if pred_count == 0 or max_conf <= 0.0:
        return 0.0, "None"

    # Type weight
    if defect_type == CLASS_POTHOLE:
        type_w = 1.0
    elif defect_type == CLASS_ROAD_DEFECT:
        type_w = 0.70
    else:
        type_w = 0.40

    norm_area = min(1.0, area_ratio / 0.04) if area_ratio > 0 else min(1.0, max_conf * 0.20)
    norm_count = min(1.0, pred_count / 3.0)

    score = 0.45 * max_conf + 0.30 * type_w + 0.15 * norm_area + 0.10 * norm_count
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.70:
        band = "HIGH"
    elif score >= 0.35:
        band = "MODERATE"
    else:
        band = "LOW"

    return round(score, 4), band


def evaluate_report_gps(lat: Optional[float], lon: Optional[float]) -> Dict[str, Any]:
    """Validate GPS coordinates and determine proximity to the 2.4 km RoadSentinel corridor."""
    if lat is None or lon is None:
        return {
            "has_gps": False,
            "latitude": None,
            "longitude": None,
            "gps_display": "LOCATION NOT PROVIDED",
            "is_on_route": False,
            "geofence_status": "NO_GPS_SUPPLIED",
            "nearest_segment": "N/A",
            "distance_to_corridor_m": None,
            "nearest_route_distance_m": None,
        }

    # Validate coordinate range
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {
            "has_gps": False,
            "latitude": lat,
            "longitude": lon,
            "gps_display": f"INVALID COORDINATES ({lat}, {lon})",
            "is_on_route": False,
            "geofence_status": "INVALID_COORDINATES",
            "nearest_segment": "N/A",
            "distance_to_corridor_m": None,
            "nearest_route_distance_m": None,
        }

    waypoints = generate_deterministic_route()
    geofence_res = check_geofence(lat, lon, waypoints, radius_m=ROAD_GEOFENCE_RADIUS_M)
    nearest_wp = geofence_res.get("nearest_waypoint") or {}
    nearest_dist_m = nearest_wp.get("distance_m", 0.0)
    nearest_seg = nearest_wp.get("segment_id", get_segment_at_distance(nearest_dist_m))
    dist_to_corridor_m = geofence_res.get("distance_m", 0.0)
    is_on_route = geofence_res.get("is_on_route", False)

    return {
        "has_gps": True,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "gps_display": f"{lat:.6f}, {lon:.6f}",
        "is_on_route": is_on_route,
        "geofence_status": geofence_res["status"],
        "nearest_segment": nearest_seg,
        "distance_to_corridor_m": round(dist_to_corridor_m, 2),
        "nearest_route_distance_m": round(nearest_dist_m, 1),
    }


def draw_report_annotations(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    sam2_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw bounding boxes, labels, and translucent SAM2 mask on the image."""
    canvas = img_bgr.copy()

    # Draw SAM2 mask if present
    if sam2_mask is not None and sam2_mask.shape[:2] == canvas.shape[:2]:
        overlay = canvas.copy()
        overlay[sam2_mask] = (0, 0, 220)  # Red tint for pothole/defect
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    # Draw YOLO detections
    for det in detections:
        box = [int(v) for v in det["bbox"]]
        x1, y1, x2, y2 = box
        label_text = f"{det['user_label']} ({det['confidence']*100:.1f}%)"
        color = CLASS_COLORS.get(det["user_label"], DEFAULT_BOX_COLOR)

        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - bl - 4)), (x1 + tw + 6, max(th + bl + 4, y1)), color, -1)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 3, max(y1 - 4, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255) if sum(color) < 400 else (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def analyze_citizen_report(
    file_bytes: bytes,
    pipeline: CitizenReportingPipeline,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Execute complete Citizen Report AI Verification Pipeline on an uploaded photo."""
    # Step 1: Validate Image
    is_valid, img_bgr, val_msg = validate_image_bytes(file_bytes)
    if not is_valid or img_bgr is None:
        return {
            "status": STATUS_RETAKE,
            "decision_reason": f"Input validation failure: {val_msg}",
            "is_valid": False,
            "defect_detected": False,
            "defect_type": "None",
            "yolo_confidence": 0.0,
            "reliability_score": 0.0,
            "reliability_level": "LOW",
            "domain_distance": 1.0,
            "domain_familiarity": "UNFAMILIAR_DOMAIN",
            "model_severity": 0.0,
            "severity_band": "None",
            "detections": [],
            "has_sam2_mask": False,
            "mask_area_ratio": 0.0,
            "gps_info": evaluate_report_gps(latitude, longitude),
            "annotated_image_bytes": None,
        }

    h, w = img_bgr.shape[:2]
    img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # Step 2: DINOv2 Domain Familiarity Check
    dino_dist, familiarity, dino_desc = compute_dino_familiarity(pipeline, img_rgb)

    # Step 3: YOLO Road Defect Detector
    detections: List[Dict[str, Any]] = []
    max_conf = 0.0
    primary_defect_type = "None"
    best_box = None

    if pipeline.yolo_model is not None:
        try:
            results = pipeline.yolo_model(img_bgr, conf=conf_threshold, verbose=False)[0]
            for b in results.boxes:
                cls_id = int(b.cls[0])
                raw_cls = pipeline.yolo_model.names[cls_id]
                user_label = YOLO_CLASS_NAME_MAP.get(raw_cls, CLASS_ROAD_DEFECT)
                conf = float(b.conf[0])
                bbox = [float(v) for v in b.xyxy[0].tolist()]

                det_info = {
                    "raw_class": raw_cls,
                    "user_label": user_label,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in bbox],
                }
                detections.append(det_info)

                if conf > max_conf:
                    max_conf = conf
                    primary_defect_type = user_label
                    best_box = bbox
        except Exception as e:
            log.error("YOLO inference failed: %s", e)

    defect_detected = (len(detections) > 0)

    # Step 4: SAM2 Defect Mask Refinement
    sam2_mask = None
    mask_area_ratio = 0.0
    if defect_detected and best_box is not None:
        sam2_mask, mask_area_ratio = run_sam2_defect_mask(pipeline, np.array(img_rgb), best_box)

    # Step 5: Reliability Scoring (Logistic Confidence Calibration + Domain Gate)
    if defect_detected:
        rel_score = float(1.0 / (1.0 + np.exp(-4.5 * (max_conf - 0.40))))
        # Penalize reliability if domain is unfamiliar
        if familiarity == "UNFAMILIAR_DOMAIN":
            rel_score *= 0.65
        elif familiarity == "MODERATE":
            rel_score *= 0.90
    else:
        # 0 detections: If in-domain road, high reliability clean baseline; if unfamiliar, low reliability
        if familiarity == "IN_DOMAIN":
            rel_score = 0.88
        elif familiarity == "MODERATE":
            rel_score = 0.70
        else:
            rel_score = 0.35

    rel_score = round(max(0.01, min(0.99, rel_score)), 4)
    if rel_score >= RELIABILITY_HIGH_THRESH:
        rel_level = "HIGH"
    elif rel_score >= RELIABILITY_LOW_THRESH:
        rel_level = "MEDIUM"
    else:
        rel_level = "LOW"

    # Step 6: Model-Derived Severity (NOT PCI)
    model_sev, sev_band = calculate_model_severity(
        max_conf=max_conf,
        defect_type=primary_defect_type,
        area_ratio=mask_area_ratio,
        pred_count=len(detections),
    )

    # Step 7: Final Citizen Report Outcome Decision
    # 1. VERIFIED REPORT:
    #    - Defect detected with strong confidence (>= 0.38)
    #    - Reliability not LOW
    #    - Domain familiar or moderate
    # 2. NEEDS MANUAL REVIEW:
    #    - Possible defect detected, but lower confidence or low reliability or unfamiliar domain
    # 3. RETAKE / NO VERIFIED DAMAGE:
    #    - 0 defects detected or unusable image
    if not defect_detected:
        status = STATUS_RETAKE
        decision_reason = "No confirmed pothole or road damage detected in the uploaded image."
    elif familiarity == "UNFAMILIAR_DOMAIN" and max_conf < 0.60:
        status = STATUS_MANUAL_REVIEW
        decision_reason = f"Possible defect identified, but image visual domain differs from training ODD (d={dino_dist:.4f}). Routed for manual verification."
    elif rel_level == "LOW":
        status = STATUS_MANUAL_REVIEW
        decision_reason = f"Perception reliability is LOW ({rel_score:.2f}) with detection confidence {max_conf*100:.1f}%. Requires manual review."
    elif max_conf >= 0.35 and rel_level in ["HIGH", "MEDIUM"]:
        status = STATUS_VERIFIED
        decision_reason = f"Verified {primary_defect_type} with {max_conf*100:.1f}% confidence and {rel_level} perception reliability."
    else:
        status = STATUS_MANUAL_REVIEW
        decision_reason = f"Marginal confidence ({max_conf*100:.1f}%); recommended for manual review."

    # Step 8: GPS Evaluation
    gps_info = evaluate_report_gps(latitude, longitude)

    # Step 9: Render Annotated Visual
    annotated_bgr = draw_report_annotations(img_bgr, detections, sam2_mask)
    is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
    annotated_bytes = buffer.tobytes() if is_success else None

    return {
        "status": status,
        "decision_reason": decision_reason,
        "is_valid": True,
        "defect_detected": defect_detected,
        "defect_type": primary_defect_type,
        "yolo_confidence": round(max_conf, 4),
        "reliability_score": rel_score,
        "reliability_level": rel_level,
        "domain_distance": dino_dist,
        "domain_familiarity": familiarity,
        "domain_description": dino_desc,
        "model_severity": model_sev,
        "severity_band": sev_band,
        "detections": detections,
        "has_sam2_mask": (sam2_mask is not None),
        "mask_area_ratio": mask_area_ratio,
        "gps_info": gps_info,
        "annotated_image_bytes": annotated_bytes,
    }


def save_citizen_report(
    raw_image_bytes: bytes,
    analysis_result: Dict[str, Any],
    reports_dir: Optional[Path] = None,
) -> str:
    """Persist citizen report, raw image, annotated overlay, and manifest record locally.

    Returns:
        Generated unique report ID, e.g. "RS-CR-0001"
    """
    target_dir = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_json_path = target_dir / "reports_manifest.json"

    # Determine unique sequential ID
    existing_reports = load_all_reports(reports_dir=target_dir)
    seq = len(existing_reports) + 1
    report_id = f"RS-CR-{seq:04d}"

    timestamp_iso = datetime.now(timezone.utc).isoformat()

    raw_img_filename = f"{report_id}_raw.jpg"
    annotated_img_filename = f"{report_id}_annotated.jpg"
    metadata_filename = f"{report_id}.json"

    # Save raw image
    (target_dir / raw_img_filename).write_bytes(raw_image_bytes)

    # Save annotated overlay if present, else copy raw
    annotated_bytes = analysis_result.get("annotated_image_bytes")
    if annotated_bytes:
        (target_dir / annotated_img_filename).write_bytes(annotated_bytes)
    else:
        (target_dir / annotated_img_filename).write_bytes(raw_image_bytes)

    # Construct metadata record
    record = {
        "report_id": report_id,
        "timestamp": timestamp_iso,
        "status": analysis_result["status"],
        "decision_reason": analysis_result["decision_reason"],
        "defect_detected": analysis_result["defect_detected"],
        "defect_type": analysis_result["defect_type"],
        "yolo_confidence": analysis_result["yolo_confidence"],
        "reliability_level": analysis_result["reliability_level"],
        "reliability_score": analysis_result["reliability_score"],
        "domain_familiarity": analysis_result["domain_familiarity"],
        "domain_distance": analysis_result["domain_distance"],
        "model_severity": analysis_result["model_severity"],
        "severity_label": "MODEL-DERIVED SEVERITY — NOT PCI",
        "has_sam2_mask": analysis_result["has_sam2_mask"],
        "mask_area_ratio": analysis_result["mask_area_ratio"],
        "latitude": analysis_result["gps_info"]["latitude"],
        "longitude": analysis_result["gps_info"]["longitude"],
        "gps_display": analysis_result["gps_info"]["gps_display"],
        "geofence_status": analysis_result["gps_info"]["geofence_status"],
        "nearest_segment": analysis_result["gps_info"]["nearest_segment"],
        "raw_image_path": str(raw_img_filename),
        "annotated_image_path": str(annotated_img_filename),
        "detections_count": len(analysis_result.get("detections", [])),
    }

    # Save individual JSON
    with open(target_dir / metadata_filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # Append/Update summary manifest JSON
    existing_reports.append(record)
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_reports, f, indent=2)

    # Update summary CSV
    update_manifest_csv(existing_reports, reports_dir=target_dir)

    log.info("Saved citizen report %s to %s", report_id, target_dir)
    return report_id


def load_all_reports(reports_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load all stored citizen reports from the local manifest."""
    target_json = (Path(reports_dir) if reports_dir is not None else REPORTS_DIR) / "reports_manifest.json"
    if not target_json.exists():
        return []
    try:
        with open(target_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error("Failed to read reports manifest: %s", e)
        return []


def update_manifest_csv(reports: List[Dict[str, Any]], reports_dir: Optional[Path] = None) -> None:
    """Save reports summary manifest as CSV."""
    if not reports:
        return

    fieldnames = [
        "report_id",
        "timestamp",
        "status",
        "defect_type",
        "yolo_confidence",
        "reliability_level",
        "domain_familiarity",
        "model_severity",
        "gps_display",
        "geofence_status",
        "nearest_segment",
        "decision_reason",
    ]

    target_csv = (Path(reports_dir) if reports_dir is not None else REPORTS_DIR) / "reports_manifest.csv"
    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in reports:
            writer.writerow(r)

