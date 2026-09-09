"""Configuration and Constants for RoadSentinel Driver Dashboard."""

from __future__ import annotations

from pathlib import Path

# Repository Paths
DRIVER_DASHBOARD_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = DRIVER_DASHBOARD_DIR.parent
DATA_DIR = DRIVER_DASHBOARD_DIR / "data"

# Road Geometry Constants
SEGMENT_NAMES = ["SEG_001", "SEG_002", "SEG_003", "SEG_004", "SEG_005", "SEG_006"]
SEGMENT_LENGTH_M = 400.0
TOTAL_ROAD_LENGTH_M = len(SEGMENT_NAMES) * SEGMENT_LENGTH_M  # 2400.0 m (2.4 km)

# Segment Boundaries (Global Distance in Meters)
SEGMENT_BOUNDARIES = {
    "SEG_001": (0.0, 400.0),
    "SEG_002": (400.0, 800.0),
    "SEG_003": (800.0, 1200.0),
    "SEG_004": (1200.0, 1600.0),
    "SEG_005": (1600.0, 2000.0),
    "SEG_006": (2000.0, 2400.0),
}

# Missing Inspection Data Segments (Pending Inspection)
MISSING_DATA_SEGMENTS = {"SEG_005", "SEG_006"}

# GPS & Navigation Parameters
START_LATITUDE = 8.8932
START_LONGITUDE = 76.6141
EARTH_RADIUS_M = 6371000.0  # WGS-84 mean radius
WAYPOINT_INTERVAL_M = 10.0  # 241 waypoints from 0 to 2400 m
ROAD_GEOFENCE_RADIUS_M = 50.0  # Radius in meters for ON_ROADSENTINEL_ROUTE

# Car Simulation Parameters
DEFAULT_SPEED_KMH = 40.0
MIN_SPEED_KMH = 0.0
MAX_SPEED_KMH = 100.0
LOOKAHEAD_DISTANCE_M = 400.0  # 400m look-ahead window

# Hazard Alert Prototype Distance Thresholds
ALERT_ADVISORY_M = 150.0
ALERT_WARNING_M = 80.0
ALERT_URGENT_M = 30.0
SPEED_WARNING_THRESHOLD_KMH = 60.0

# Alert Level Names
ALERT_LEVEL_ADVISORY = "ADVISORY"
ALERT_LEVEL_WARNING = "WARNING"
ALERT_LEVEL_URGENT = "URGENT"

# Disclaimer Text
DISCLAIMER_TEXT = (
    "RoadSentinel Driver Dashboard is a research prototype. "
    "Road-health scores and hazard locations shown in simulation are model-derived "
    "and are not a substitute for official navigation or road-safety information."
)
