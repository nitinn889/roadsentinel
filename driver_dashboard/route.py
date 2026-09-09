"""Deterministic Route Generation and GPS Interpolation for RoadSentinel Driver Dashboard."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import (
    DATA_DIR,
    EARTH_RADIUS_M,
    SEGMENT_BOUNDARIES,
    SEGMENT_LENGTH_M,
    START_LATITUDE,
    START_LONGITUDE,
    TOTAL_ROAD_LENGTH_M,
    WAYPOINT_INTERVAL_M,
)
from gps import haversine_distance


def get_segment_at_distance(distance_m: float) -> str:
    """Map global road distance in meters to segment ID (SEG_001 to SEG_006).

    Handles boundary transitions strictly:
    - [0, 400) -> SEG_001
    - [400, 800) -> SEG_002
    - [800, 1200) -> SEG_003
    - [1200, 1600) -> SEG_004
    - [1600, 2000) -> SEG_005
    - [2000, 2400] -> SEG_006
    """
    d = max(0.0, min(TOTAL_ROAD_LENGTH_M, distance_m))
    if d >= 2000.0:
        return "SEG_006"
    if d >= 1600.0:
        return "SEG_005"
    if d >= 1200.0:
        return "SEG_004"
    if d >= 800.0:
        return "SEG_003"
    if d >= 400.0:
        return "SEG_002"
    return "SEG_001"


def compute_destination_point(
    lat: float, lon: float, distance_m: float, bearing_deg: float
) -> Tuple[float, float]:
    """Compute destination point given distance and bearing on Earth sphere."""
    r = EARTH_RADIUS_M
    delta = distance_m / r
    theta = math.radians(bearing_deg)

    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    sin_phi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    phi2 = math.asin(sin_phi2)

    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * sin_phi2
    lambda2 = lambda1 + math.atan2(y, x)

    lat2 = math.degrees(phi2)
    lon2 = math.degrees(lambda2)
    return round(lat2, 6), round(lon2, 6)


def generate_deterministic_route() -> List[Dict[str, Any]]:
    """Generate 2.4 km deterministic route with waypoints every ~10m.

    Road trajectory starts at (8.8932, 76.6141) and follows a gentle curve
    (bearing transitions smoothly from 45.0 deg to 55.0 deg).
    """
    route_file = DATA_DIR / "road_route.json"
    if route_file.exists():
        try:
            data = json.loads(route_file.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) >= 200:
                return data
        except Exception:
            pass

    waypoints: List[Dict[str, Any]] = []
    curr_lat = START_LATITUDE
    curr_lon = START_LONGITUDE

    num_steps = int(round(TOTAL_ROAD_LENGTH_M / WAYPOINT_INTERVAL_M)) + 1  # 241 points

    for i in range(num_steps):
        dist = min(TOTAL_ROAD_LENGTH_M, i * WAYPOINT_INTERVAL_M)
        # Gentle curve: bearing shifts smoothly across 2.4 km
        bearing = 45.0 + (dist / TOTAL_ROAD_LENGTH_M) * 10.0

        if i == 0:
            lat, lon = START_LATITUDE, START_LONGITUDE
        else:
            lat, lon = compute_destination_point(
                START_LATITUDE, START_LONGITUDE, dist, bearing
            )

        segment_id = get_segment_at_distance(dist)
        waypoints.append({
            "distance_m": round(dist, 1),
            "latitude": lat,
            "longitude": lon,
            "segment_id": segment_id,
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    route_file.write_text(json.dumps(waypoints, indent=2), encoding="utf-8")
    return waypoints


def get_gps_at_distance(
    distance_m: float, waypoints: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """Interpolate (latitude, longitude) at any global road distance d in [0, 2400]."""
    if not waypoints:
        return START_LATITUDE, START_LONGITUDE

    d = max(0.0, min(TOTAL_ROAD_LENGTH_M, distance_m))

    if d <= waypoints[0]["distance_m"]:
        return waypoints[0]["latitude"], waypoints[0]["longitude"]
    if d >= waypoints[-1]["distance_m"]:
        return waypoints[-1]["latitude"], waypoints[-1]["longitude"]

    # Linear interpolation between nearest bounding waypoints
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        if wp1["distance_m"] <= d <= wp2["distance_m"]:
            span = wp2["distance_m"] - wp1["distance_m"]
            if span <= 1e-6:
                return wp1["latitude"], wp1["longitude"]
            frac = (d - wp1["distance_m"]) / span
            lat = wp1["latitude"] + frac * (wp2["latitude"] - wp1["latitude"])
            lon = wp1["longitude"] + frac * (wp2["longitude"] - wp1["longitude"])
            return round(lat, 6), round(lon, 6)

    return waypoints[-1]["latitude"], waypoints[-1]["longitude"]
