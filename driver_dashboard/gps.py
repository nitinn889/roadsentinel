"""GPS and Haversine Distance Calculations for RoadSentinel Driver Dashboard."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
from config import EARTH_RADIUS_M, ROAD_GEOFENCE_RADIUS_M


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth in meters.

    Uses the Haversine formula with R = 6,371,000 meters.
    Calculates internally using radians.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )

    # Clamp 'a' to [0.0, 1.0] to prevent domain errors in sqrt due to floating-point rounding
    a = max(0.0, min(1.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def check_geofence(
    user_lat: float,
    user_lon: float,
    waypoints: List[Dict[str, Any]],
    radius_m: float = ROAD_GEOFENCE_RADIUS_M,
) -> Dict[str, Any]:
    """Determine whether the simulated user is inside the monitored RoadSentinel corridor.

    Finds the nearest route waypoint using Haversine distance.
    If distance <= radius_m (default 50.0m), returns status ON_ROADSENTINEL_ROUTE;
    otherwise OUTSIDE_MONITORED_ROUTE.
    """
    if not waypoints:
        return {
            "status": "OUTSIDE_MONITORED_ROUTE",
            "distance_m": float("inf"),
            "nearest_waypoint": None,
            "is_on_route": False,
        }

    min_dist = float("inf")
    nearest_wp = None

    for wp in waypoints:
        dist = haversine_distance(user_lat, user_lon, wp["latitude"], wp["longitude"])
        if dist < min_dist:
            min_dist = dist
            nearest_wp = wp

    is_on_route = min_dist <= radius_m
    status = "ON_ROADSENTINEL_ROUTE" if is_on_route else "OUTSIDE_MONITORED_ROUTE"

    return {
        "status": status,
        "distance_m": round(min_dist, 2),
        "nearest_waypoint": nearest_wp,
        "is_on_route": is_on_route,
    }
