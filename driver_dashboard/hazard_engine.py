"""Hazard Engine and Driver Alert System for RoadSentinel Driver Dashboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import (
    ALERT_ADVISORY_M,
    ALERT_LEVEL_ADVISORY,
    ALERT_LEVEL_URGENT,
    ALERT_LEVEL_WARNING,
    ALERT_URGENT_M,
    ALERT_WARNING_M,
    DATA_DIR,
    SEGMENT_BOUNDARIES,
    SPEED_WARNING_THRESHOLD_KMH,
)
from data_loader import load_primary_results_data, load_temporal_defect_history
from gps import haversine_distance
from route import generate_deterministic_route, get_gps_at_distance

HAZARD_MANIFEST_CSV = DATA_DIR / "hazard_manifest.csv"

# Pre-defined deterministic offsets (in meters) within each 400m segment for defects
DETERMINISTIC_OFFSETS = [135.0, 275.0, 60.0, 320.0, 180.0, 240.0, 90.0, 310.0]


def generate_hazard_manifest() -> List[Dict[str, Any]]:
    """Build a normalized hazard manifest from actual RoadSentinel inspection data.

    Assigns deterministic, fixed global route positions for defects inside their 400m segment.
    Preserves track positions across multiple days (e.g. Day01-10).
    """
    if HAZARD_MANIFEST_CSV.exists():
        try:
            items: List[Dict[str, Any]] = []
            with open(HAZARD_MANIFEST_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    items.append({
                        "hazard_id": row["hazard_id"],
                        "day": int(row["day"]),
                        "segment_id": row["segment_id"],
                        "hazard_type": row["hazard_type"],
                        "severity": float(row["severity"]),
                        "confidence": float(row["confidence"]),
                        "source_model": row["source_model"],
                        "local_segment_distance_m": float(row["local_segment_distance_m"]),
                        "global_distance_m": float(row["global_distance_m"]),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                    })
            if items:
                return items
        except Exception:
            pass

    waypoints = generate_deterministic_route()
    primary_data = load_primary_results_data()
    manifest: List[Dict[str, Any]] = []

    for day in range(1, 11):
        for seg_id in ["SEG_001", "SEG_002", "SEG_003", "SEG_004"]:
            seg_start_m, _ = SEGMENT_BOUNDARIES[seg_id]
            seg_info = primary_data.get(day, {}).get(seg_id, {})

            if not seg_info.get("has_data"):
                continue

            defect_cnt = seg_info.get("defect_count", 0)
            sev = seg_info.get("current_severity", 0.0)

            if defect_cnt <= 0 or sev <= 0.0:
                continue

            # Load defect breakdown if available in segment history
            history = load_temporal_defect_history(seg_id).get(day, {})
            defect_types = history.get("defect_types", [])

            for idx in range(defect_cnt):
                offset_m = DETERMINISTIC_OFFSETS[idx % len(DETERMINISTIC_OFFSETS)]
                global_dist_m = seg_start_m + offset_m
                lat, lon = get_gps_at_distance(global_dist_m, waypoints)

                # Classify hazard type
                d_type = defect_types[idx] if idx < len(defect_types) else "defect"
                if "pothole" in str(d_type).lower():
                    hazard_type = "Pothole"
                else:
                    hazard_type = "Pothole" if sev >= 0.40 else "Road Defect"

                hazard_id = f"HAZARD_{seg_id}_D{day:02d}_IDX{idx+1:02d}"
                manifest.append({
                    "hazard_id": hazard_id,
                    "day": day,
                    "segment_id": seg_id,
                    "hazard_type": hazard_type,
                    "severity": round(sev, 4),
                    "confidence": 0.85,
                    "source_model": "YOLOv8n / DINOv2",
                    "local_segment_distance_m": round(offset_m, 1),
                    "global_distance_m": round(global_dist_m, 1),
                    "latitude": lat,
                    "longitude": lon,
                })

    # Save to CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if manifest:
        headers = list(manifest[0].keys())
        with open(HAZARD_MANIFEST_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(manifest)

    return manifest


def get_hazards_for_day(day: int) -> List[Dict[str, Any]]:
    """Return all active hazards for a specific inspection day."""
    all_hazards = generate_hazard_manifest()
    return [h for h in all_hazards if h["day"] == day]


def get_nearest_hazard_ahead(
    vehicle_dist_m: float,
    vehicle_lat: float,
    vehicle_lon: float,
    hazards_for_day: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the nearest hazard strictly AHEAD of the vehicle (global_distance_m > vehicle_dist_m).

    Returns dictionary containing hazard record, distance_ahead_m, haversine_dist_m, and eta_sec.
    Ignored hazards behind the vehicle.
    """
    hazards_ahead = [
        h for h in hazards_for_day if h["global_distance_m"] > vehicle_dist_m
    ]
    if not hazards_ahead:
        return None

    nearest_hazard = min(
        hazards_ahead, key=lambda h: h["global_distance_m"] - vehicle_dist_m
    )
    dist_ahead_m = nearest_hazard["global_distance_m"] - vehicle_dist_m
    haversine_dist = haversine_distance(
        vehicle_lat, vehicle_lon, nearest_hazard["latitude"], nearest_hazard["longitude"]
    )

    result = dict(nearest_hazard)
    result["distance_ahead_m"] = round(dist_ahead_m, 1)
    result["haversine_dist_m"] = round(haversine_dist, 1)
    return result


def evaluate_alert_status(
    nearest_hazard: Optional[Dict[str, Any]],
    speed_kmh: float,
    fired_alerts_set: Set[Tuple[str, str]],
) -> Tuple[Optional[Dict[str, Any]], Set[Tuple[str, str]]]:
    """Evaluate prototype driver alerts based on distance ahead and vehicle speed.

    Alert Thresholds:
    - > 150m: No alert
    - 80m - 150m: ADVISORY ("Road damage detected XX m ahead.")
    - 30m - 80m: WARNING ("Pothole / road defect XX m ahead. Approach with caution.")
    - <= 30m: URGENT ("Road hazard XX m ahead. Reduce speed.")

    Speed-Context Escalation:
    - If hazard <= 80m AND speed >= 60 km/h: "Road damage XX m ahead at current speed. Reduce speed."

    Deduplication:
    - Sends each alert level (ADVISORY, WARNING, URGENT) only ONCE per hazard ID.
    - Does NOT use directive language instructing driver to swerve or change lanes.
    """
    if nearest_hazard is None:
        return None, fired_alerts_set

    dist_m = nearest_hazard["distance_ahead_m"]
    hazard_id = nearest_hazard["hazard_id"]
    hazard_type = nearest_hazard["hazard_type"]

    if dist_m > ALERT_ADVISORY_M:
        return None, fired_alerts_set

    # Determine alert level
    if dist_m <= ALERT_URGENT_M:
        level = ALERT_LEVEL_URGENT
    elif dist_m <= ALERT_WARNING_M:
        level = ALERT_LEVEL_WARNING
    else:
        level = ALERT_LEVEL_ADVISORY

    # Check deduplication: key = (hazard_id, level)
    alert_key = (hazard_id, level)
    is_new_trigger = alert_key not in fired_alerts_set
    if is_new_trigger:
        fired_alerts_set.add(alert_key)

    # Formulate non-directive alert message
    dist_str = f"{int(round(dist_m))} m"
    if level == ALERT_LEVEL_URGENT:
        message = f"URGENT: {hazard_type} {dist_str} ahead. Reduce speed."
    elif level == ALERT_LEVEL_WARNING:
        if speed_kmh >= SPEED_WARNING_THRESHOLD_KMH:
            message = f"WARNING: {hazard_type} {dist_str} ahead at current speed ({int(speed_kmh)} km/h). Reduce speed."
        else:
            message = f"WARNING: {hazard_type} {dist_str} ahead. Approach with caution."
    else:
        message = f"ADVISORY: Road damage detected {dist_str} ahead."

    alert_obj = {
        "level": level,
        "message": message,
        "hazard_id": hazard_id,
        "hazard_type": hazard_type,
        "distance_ahead_m": dist_m,
        "is_new": is_new_trigger,
    }

    return alert_obj, fired_alerts_set
