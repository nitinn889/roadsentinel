"""geo_utils.py
------------
Converts CARLA local Cartesian coordinates into real-world geographic coordinates (Latitude, Longitude, Altitude).
Supports CARLA's native OpenDRIVE map georeferencing via `map.transform_to_geolocation(location)`
as well as planar projection math for offline / standalone execution.
"""

from __future__ import annotations

import math
from typing import Any, Tuple
import config

METERS_PER_DEG_LAT = 111_320.0


def carla_transform_to_geolocation(carla_map: Any, location: Any) -> Tuple[float, float, float]:
    """Convert CARLA Location (X, Y, Z in meters) to real-world georeferenced (lat, lon, alt)
    using the standardized project datum projection (REFERENCE_LAT, REFERENCE_LON).
    """
    x_m = getattr(location, "x", float(location[0] if isinstance(location, (list, tuple)) else 0.0))
    y_m = getattr(location, "y", float(location[1] if isinstance(location, (list, tuple)) else 0.0))
    z_m = getattr(location, "z", float(location[2] if isinstance(location, (list, tuple)) else config.ALTITUDE_M))
    lat, lon = local_xy_to_latlon(x_m, y_m)
    return lat, lon, z_m


def local_xy_to_latlon(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert local Cartesian meters relative to CARLA reference origin into (lat, lon) in decimal degrees."""
    lat = config.REFERENCE_LAT + (y_m / METERS_PER_DEG_LAT)
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(config.REFERENCE_LAT))
    lon = config.REFERENCE_LON + (x_m / max(1e-6, meters_per_deg_lon))
    return lat, lon


def latlon_to_local_xy(lat: float, lon: float) -> Tuple[float, float]:
    """Convert (lat, lon) back into local Cartesian metric coordinates relative to reference datum."""
    dy_m = (lat - config.REFERENCE_LAT) * METERS_PER_DEG_LAT
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(config.REFERENCE_LAT))
    dx_m = (lon - config.REFERENCE_LON) * meters_per_deg_lon
    return dx_m, dy_m
