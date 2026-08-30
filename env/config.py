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

ALTITUDE_M = 35.0

# ---------------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------------
CAMERA_FOV_DEG = 100.0      # horizontal FOV, CARLA sensor convention
IMAGE_WIDTH = 640          # 4K - "clear photos"
IMAGE_HEIGHT = 360
CAMERA_PITCH_DEG = -90.0    # nadir (straight down) - correct for orthomosaic-style capture

# ---------------------------------------------------------------------------
# PHOTOGRAMMETRY CAPTURE
# ---------------------------------------------------------------------------
FORWARD_OVERLAP = 0.70      # 70% overlap, point 7
# Requested interval was "6-7 s"; because interval is mathematically
# determined by SPEED_MPS, ALTITUDE_M, CAMERA_FOV_DEG, resolution and
# FORWARD_OVERLAP, it is computed at runtime rather than fixed here (you
# were explicit that this number can move). With the defaults above it
# lands at ~4.8 s - printed to console on startup. Push ALTITUDE_M up
# toward ~135 m if you need it closer to 6-7 s exactly; see README.

# ---------------------------------------------------------------------------
# SYNTHETIC GEOREFERENCE (for ODM/COLMAP-style geo.txt output)
# ---------------------------------------------------------------------------
# CARLA world coordinates are local meters with an arbitrary origin, not
# real GPS. For the output to be consumable by a photogrammetry pipeline
# that expects geotags (OpenDroneMap's geo.txt, etc.) we synthesize a
# lat/lon by treating this reference point as the world origin (0,0) and
# offsetting by the drone's local x/y in meters. Purely a simulation
# convenience - not a real GPS fix.
REFERENCE_LAT = 11.0168   # Coimbatore, TN - arbitrary but documented origin
REFERENCE_LON = 76.9558

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
