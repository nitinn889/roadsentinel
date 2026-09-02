"""Spatial Indexing & Geofencing Engine for RoadSentinel.

Builds a 2D KD-Tree spatial index over detected road defects (potholes, cracks, water hazards)
using georeferenced Latitude and Longitude (or local Cartesian metric coordinates).
Enables sub-millisecond dynamic radius queries, proximity alerts, and geofence boundary checks
for autonomous and driver-assistance early-warning systems.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from scipy.spatial import cKDTree as KDTreeImpl
except ImportError:
    try:
        from sklearn.neighbors import KDTree as KDTreeImpl
    except ImportError:
        KDTreeImpl = None

METERS_PER_DEG_LAT = 111_320.0


def latlon_to_metric(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """Projects (lat, lon) onto tangent Cartesian plane in meters relative to reference."""
    dy_m = (lat - ref_lat) * METERS_PER_DEG_LAT
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    dx_m = (lon - ref_lon) * meters_per_deg_lon
    return dx_m, dy_m


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute precise great-circle distance between two GPS coordinates in meters."""
    R = 6_371_000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class DefectSpatialIndex:
    """2D KD-Tree Spatial Index for Georeferenced Road Defects & Dynamic Geofencing."""

    def __init__(self, defects: Optional[List[Dict[str, Any]]] = None, ref_lat: float = 13.0827, ref_lon: float = 80.2707):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.defects: List[Dict[str, Any]] = []
        self.metric_coords: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self.kdtree = None

        if defects:
            self.build_index(defects)

    def build_index(self, defects: List[Union[Dict[str, Any], Any]]):
        """Construct KD-Tree over list of defect records or DefectMeasurement objects."""
        self.defects = []
        metric_points = []

        # Find reference center from defects if possible
        valid_lats = []
        valid_lons = []
        for d in defects:
            d_dict = d if isinstance(d, dict) else (d.to_dict() if hasattr(d, "to_dict") else vars(d))
            lat = d_dict.get("latitude")
            lon = d_dict.get("longitude")
            if lat is not None and lon is not None and np.isfinite(lat) and np.isfinite(lon):
                valid_lats.append(float(lat))
                valid_lons.append(float(lon))

        if valid_lats:
            self.ref_lat = float(np.mean(valid_lats))
            self.ref_lon = float(np.mean(valid_lons))

        for d in defects:
            d_dict = d if isinstance(d, dict) else (d.to_dict() if hasattr(d, "to_dict") else vars(d))
            lat = d_dict.get("latitude")
            lon = d_dict.get("longitude")
            world_x = d_dict.get("world_x") or d_dict.get("x_m")
            world_y = d_dict.get("world_y") or d_dict.get("y_m")

            if lat is not None and lon is not None:
                dx, dy = latlon_to_metric(float(lat), float(lon), self.ref_lat, self.ref_lon)
            elif world_x is not None and world_y is not None:
                dx, dy = float(world_x), float(world_y)
            else:
                continue

            self.defects.append(d_dict)
            metric_points.append([dx, dy])

        if metric_points:
            self.metric_coords = np.array(metric_points, dtype=np.float64)
            if KDTreeImpl is not None:
                try:
                    self.kdtree = KDTreeImpl(self.metric_coords)
                except Exception:
                    self.kdtree = None
        else:
            self.metric_coords = np.empty((0, 2), dtype=np.float64)
            self.kdtree = None

    def query_radius(self, lat: float, lon: float, radius_m: float) -> List[Dict[str, Any]]:
        """Query all defects within `radius_m` meters of (lat, lon)."""
        if len(self.defects) == 0:
            return []

        qx, qy = latlon_to_metric(lat, lon, self.ref_lat, self.ref_lon)
        results = []

        if self.kdtree is not None and hasattr(self.kdtree, "query_ball_point"):
            indices = self.kdtree.query_ball_point([qx, qy], r=radius_m)
            for idx in indices:
                d = dict(self.defects[idx])
                dist = float(np.linalg.norm(self.metric_coords[idx] - np.array([qx, qy])))
                d["distance_m"] = round(dist, 2)
                results.append(d)
        else:
            # Fallback linear distance search
            diff = self.metric_coords - np.array([qx, qy])
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            for idx, dist in enumerate(dists):
                if dist <= radius_m:
                    d = dict(self.defects[idx])
                    d["distance_m"] = round(float(dist), 2)
                    results.append(d)

        results.sort(key=lambda x: x.get("distance_m", 0.0))
        return results

    def query_nearest(self, lat: float, lon: float, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """Query k nearest defects to (lat, lon) and their distances in meters."""
        if len(self.defects) == 0:
            return []

        qx, qy = latlon_to_metric(lat, lon, self.ref_lat, self.ref_lon)
        k_clamped = min(k, len(self.defects))

        if self.kdtree is not None and hasattr(self.kdtree, "query"):
            dists, indices = self.kdtree.query([qx, qy], k=k_clamped)
            if k_clamped == 1:
                dists = [dists]
                indices = [indices]
            results = []
            for d, idx in zip(dists, indices):
                results.append((dict(self.defects[int(idx)]), float(d)))
            return results
        else:
            diff = self.metric_coords - np.array([qx, qy])
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            sorted_indices = np.argsort(dists)[:k_clamped]
            return [(dict(self.defects[idx]), float(dists[idx])) for idx in sorted_indices]

    def create_geofence_zones(self, default_radius_m: float = 50.0) -> List[Dict[str, Any]]:
        """Generate geofencing circular hazard zones for each indexed defect."""
        zones = []
        for d in self.defects:
            is_water = bool(d.get("is_water_filled") or d.get("water_flag"))
            sev = d.get("severity") or {}
            sev_tier = sev.get("severity") if isinstance(sev, dict) else "medium"
            if sev_tier == "critical" or is_water:
                radius = default_radius_m
                level = "critical"
            elif sev_tier == "high":
                radius = default_radius_m * 0.75
                level = "high"
            else:
                radius = default_radius_m * 0.50
                level = "warning"

            zones.append({
                "geofence_id": f"GEO_{d.get('defect_id', 'DEF')}",
                "defect_id": d.get("defect_id"),
                "defect_type": d.get("defect_type", "pothole"),
                "latitude": d.get("latitude"),
                "longitude": d.get("longitude"),
                "radius_m": round(radius, 1),
                "hazard_level": level,
                "is_water_filled": is_water,
                "speed_advisory_kmph": 30 if level == "critical" else 50,
            })
        return zones

    def evaluate_driver_hazard(
        self,
        driver_lat: float,
        driver_lon: float,
        driver_speed_kmph: float = 60.0,
        warning_radius_m: float = 75.0,
        critical_radius_m: float = 25.0,
    ) -> Dict[str, Any]:
        """Evaluate proximity of a driver vehicle to indexed defects and generate safety alerts."""
        nearby = self.query_radius(driver_lat, driver_lon, radius_m=warning_radius_m)
        if not nearby:
            return {
                "alert_triggered": False,
                "status": "CLEAR",
                "nearest_distance_m": None,
                "hazard_level": "none",
                "message": "Corridor clear — no road surface hazards in immediate vicinity.",
                "speed_advisory_kmph": None,
            }

        closest = nearby[0]
        dist_m = closest["distance_m"]
        is_water = bool(closest.get("is_water_filled") or closest.get("water_flag"))
        defect_class = closest.get("defect_type", "pothole").replace("_", " ").title()

        if dist_m <= critical_radius_m:
            hazard_level = "CRITICAL"
            speed_advisory = 30
            msg = f"CRITICAL HAZARD: {defect_class} {dist_m:.0f}m ahead! Reduce speed immediately to {speed_advisory} km/h."
        else:
            hazard_level = "WARNING"
            speed_advisory = 50
            msg = f"CAUTION: Road defect ({defect_class}) detected {dist_m:.0f}m ahead. Maintain lane vigilance."

        return {
            "alert_triggered": True,
            "status": hazard_level,
            "hazard_level": hazard_level.lower(),
            "nearest_distance_m": dist_m,
            "defect_id": closest.get("defect_id"),
            "defect_type": closest.get("defect_type"),
            "is_water_filled": is_water,
            "speed_advisory_kmph": speed_advisory,
            "message": msg,
            "total_defects_in_range": len(nearby),
        }
