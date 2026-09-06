"""
config.py
---------
Single source of truth for all tunable parameters of the RoadSentinel
CARLA drone-simulation rig. Change values here rather than hunting through
the rest of the codebase.

IMPORTANT - the physics of aerial-survey photogrammetry couples several of
these numbers together (speed, altitude, camera FOV/resolution, overlap %
and capture interval are NOT independent - see overlap_calculator.py for
the derivation). If you change ALTITUDE_M, CAMERA_FOV_DEG or the image
resolution, the actual capture interval used at runtime will shift
automatically - it is computed live in overlap_calculator.py, not hardcoded.
"""

# ---------------------------------------------------------------------------
# MAP / ENVIRONMENT
# ---------------------------------------------------------------------------
# Town06 is the CARLA town best suited to "National Highway"-style testing:
# it has long, straight, multi-lane highway stretches with entrance/exit
# ramps (plus a small urban section, covering the general "flies over a
# city" brief too). Town07 is a fully rural, single-lane-country-road map -
# swap to that if you want an even more rural look and don't need the
# highway geometry.
CARLA_MAP = "Town04"
ALT_RURAL_MAP = "Town07"  # fallback / alternative, see README

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT_S = 15.0

# Simulation runs in SYNCHRONOUS mode so that "seconds" in this script mean
# simulated seconds, not wall-clock seconds - this keeps speed, distance and
# capture-interval math exact regardless of how fast/slow your machine can
# render frames.
FIXED_DELTA_SECONDS = 0.05  # 20 simulation ticks per simulated second

# ---------------------------------------------------------------------------
# DRONE FLIGHT
# ---------------------------------------------------------------------------
# The "drone" is implemented as a free-floating (kinematic, non-physics)
# camera actor - see README for why this is the standard approach in CARLA,
# which has no native multirotor flight model. You drive it with the
# keyboard; it is not attached to any vehicle or parent actor.

SPEED_KMPH = 30.0  # hard cap, constant - see drone_controller.py
SPEED_MPS = SPEED_KMPH / 3.6

YAW_RATE_DEG_S = 45.0       # turning rate for Q/E
CLIMB_RATE_MPS = 3.0        # ascend/descend rate for R/F
ALTITUDE_MIN_M = 15.0
ALTITUDE_MAX_M = 150.0

ALTITUDE_M = 100.0  # Standardized 100 m aerial survey altitude

# ---------------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------------
# Calibrated to 60° (equivalent to ~24mm full-frame survey lens on enterprise
# inspection drones like DJI Matrice 300 / Zenmuse P1). At 100m altitude, 60° FOV
# yields a ground swath width of ~115m and ~16.7 px/m GSD (~6.0 cm/px).
# A 1.5m pothole spans ~25 pixels—well above the 14x14 patch embedding threshold
# of DINOv2 and SAM2 prompt requirements. The previous 100° FOV spanned >238m
# ground width, shrinking potholes below 8 pixels and causing them to vanish.
CAMERA_FOV_DEG = 60.0
IMAGE_WIDTH = 1920          # High-Resolution Aerial Photogrammetry (1080p)
IMAGE_HEIGHT = 1080
CAMERA_PITCH_DEG = -90.0    # nadir (straight down) - correct for orthomosaic-style capture

# ---------------------------------------------------------------------------
# PHOTOGRAMMETRY CAPTURE
# ---------------------------------------------------------------------------
FORWARD_OVERLAP = 0.70      # 70% overlap, point 7
# Requested interval was "6-7 s"; because interval is mathematically
# determined by SPEED_MPS, ALTITUDE_M, CAMERA_FOV_DEG, resolution and
# FORWARD_OVERLAP, it is computed at runtime rather than fixed here.
# With 30 km/h, 100m altitude and 60° FOV, the forward capture interval is ~4.1s.

# ---------------------------------------------------------------------------
# SYNTHETIC GEOREFERENCE (for ODM/COLMAP-style geo.txt output)
# ---------------------------------------------------------------------------
# Aligned with RoadSentinel Town04 reference datum for geospatial consistency
REFERENCE_LAT = 13.0827
REFERENCE_LON = 80.2707

# ---------------------------------------------------------------------------
# ROAD-HEALTH SCENARIO PROFILES
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ScenarioConfig:
    name: str
    description: str
    defects_per_corridor: Tuple[int, int]   # min, max defects per corridor
    pothole_size_range_m: Tuple[float, float] # (min_diameter, max_diameter)
    pothole_depth_range_m: Tuple[float, float] # (min_depth, max_depth)
    water_filled_ratio: float                # probability [0.0, 1.0] of pothole containing water
    cluster_probability: float               # probability of defect clustering
    crack_density: float                     # [0.0, 1.0]
    patch_density: float                     # [0.0, 1.0] (repaired patches)
    wear_level: float                        # [0.0, 1.0] general surface wear

SCENARIOS: Dict[str, ScenarioConfig] = {
    "healthy": ScenarioConfig(
        name="healthy",
        description="Pristine or near-pristine road with minimal minor blemishes and clean asphalt",
        defects_per_corridor=(0, 2),
        pothole_size_range_m=(0.25, 0.50),
        pothole_depth_range_m=(0.015, 0.035),
        water_filled_ratio=0.0,
        cluster_probability=0.05,
        crack_density=0.05,
        patch_density=0.05,
        wear_level=0.10,
    ),
    "moderate": ScenarioConfig(
        name="moderate",
        description="Moderate road deterioration with small/medium potholes, patches and isolated cracks",
        defects_per_corridor=(5, 9),
        pothole_size_range_m=(0.40, 1.10),
        pothole_depth_range_m=(0.040, 0.085),
        water_filled_ratio=0.25,
        cluster_probability=0.30,
        crack_density=0.30,
        patch_density=0.25,
        wear_level=0.35,
    ),
    "poor": ScenarioConfig(
        name="poor",
        description="Substantial degradation with multiple large potholes, cracks and standing water",
        defects_per_corridor=(10, 18),
        pothole_size_range_m=(0.60, 1.80),
        pothole_depth_range_m=(0.065, 0.150),
        water_filled_ratio=0.45,
        cluster_probability=0.50,
        crack_density=0.60,
        patch_density=0.45,
        wear_level=0.60,
    ),
    "critical": ScenarioConfig(
        name="critical",
        description="Severe failure with high density of deep cavitations, clusters, water hazards and broken pavement",
        defects_per_corridor=(18, 28),
        pothole_size_range_m=(0.90, 2.40),
        pothole_depth_range_m=(0.095, 0.220),
        water_filled_ratio=0.65,
        cluster_probability=0.75,
        crack_density=0.85,
        patch_density=0.65,
        wear_level=0.85,
    ),
}

# ---------------------------------------------------------------------------
# WEATHER & LIGHTING PRESETS
# ---------------------------------------------------------------------------
# Calibrated for aerial nadir survey: Screen-space reflections (SSR) of sky clouds
# are eliminated by capping cloudiness/wetness interactions, preventing blocky
# white reflection blotches from appearing over asphalt while preserving an authentic
# damp road sheen.
WEATHER_PRESETS: Dict[str, Dict[str, float]] = {
    "clear": {
        "cloudiness": 0.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 0.0,
        "fog_density": 0.0,
        "sun_altitude_angle": 75.0,
        "sun_azimuth_angle": 180.0,
    },
    "overcast": {
        "cloudiness": 65.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 5.0,
        "fog_density": 2.0,
        "sun_altitude_angle": 55.0,
        "sun_azimuth_angle": 180.0,
    },
    "early_morning": {
        "cloudiness": 5.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 5.0,
        "fog_density": 5.0,
        "sun_altitude_angle": 22.0,
        "sun_azimuth_angle": 75.0,
    },
    "late_afternoon": {
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 0.0,
        "fog_density": 0.0,
        "sun_altitude_angle": 20.0,
        "sun_azimuth_angle": 255.0,
    },
    "sunset": {
        "cloudiness": 15.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 0.0,
        "fog_density": 0.0,
        "sun_altitude_angle": 12.0,
        "sun_azimuth_angle": 270.0,
    },
    "low_light": {
        "cloudiness": 40.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 5.0,
        "fog_density": 8.0,
        "sun_altitude_angle": 3.0,
        "sun_azimuth_angle": 280.0,
    },
    "wet": {
        "cloudiness": 15.0,
        "precipitation": 0.0,
        "precipitation_deposits": 10.0,
        "wetness": 20.0,
        "fog_density": 2.0,
        "sun_altitude_angle": 60.0,
        "sun_azimuth_angle": 180.0,
    },
    "rain": {
        "cloudiness": 45.0,
        "precipitation": 40.0,
        "precipitation_deposits": 15.0,
        "wetness": 28.0,
        "fog_density": 5.0,
        "sun_altitude_angle": 50.0,
        "sun_azimuth_angle": 180.0,
    },
    "post_rain": {
        "cloudiness": 18.0,
        "precipitation": 0.0,
        "precipitation_deposits": 12.0,
        "wetness": 18.0,
        "fog_density": 0.0,
        "sun_altitude_angle": 65.0,
        "sun_azimuth_angle": 180.0,
    },
}
# Helpful aliases for user convenience
WEATHER_PRESETS["post-rain"] = WEATHER_PRESETS["post_rain"]
WEATHER_PRESETS["early morning"] = WEATHER_PRESETS["early_morning"]
WEATHER_PRESETS["low light"] = WEATHER_PRESETS["low_light"]

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
import os
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
IMAGES_SUBDIR = "images"
IMAGE_FORMAT = "jpg"
JPEG_QUALITY = 95

# Straight-road-segment finder (for the N/P "control the road" keys)
ROAD_WAYPOINT_SPACING_M = 25.0
ROAD_SEGMENT_LOOKAHEAD_M = 150.0
ROAD_SEGMENT_MAX_HEADING_CHANGE_DEG = 8.0  # how "straight" counts as straight

