"""
geo_utils.py
------------
Converts CARLA's local (x, y) metre coordinates into a synthetic
lat/lon so downstream photogrammetry tools (OpenDroneMap, etc.) that
expect geotagged imagery have something to ingest.

This is a small-angle equirectangular approximation - perfectly fine over
the few-hundred-metre extent of a single flight, not intended for anything
beyond simulation testing.
"""

import math
import config

METERS_PER_DEG_LAT = 111_320.0


def local_xy_to_latlon(x_m: float, y_m: float):
    """
    x_m, y_m: CARLA world-space offsets in metres relative to the flight's
    origin (i.e. pass carla_location.x - origin.x, carla_location.y - origin.y).
    Returns (lat, lon) in decimal degrees.
    """
    lat = config.REFERENCE_LAT + (y_m / METERS_PER_DEG_LAT)
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(config.REFERENCE_LAT))
    lon = config.REFERENCE_LON + (x_m / meters_per_deg_lon)
    return lat, lon
