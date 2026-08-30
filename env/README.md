# RoadSentinel - CARLA Drone Simulation Rig

Keyboard-controlled drone camera flying a CARLA highway, capturing
geo-tagged, overlap-controlled aerial imagery for the RoadSentinel
pothole-detection pipeline (DINOv2 anomaly detection + SAM2 + photogrammetry).

For full step-by-step run instructions (starting the CARLA server, activating
your venv, etc.) see the separate **demo instructions** package - this file
just covers what's in this codebase.

## Files
| File | Purpose |
|---|---|
| `config.py` | All tunable parameters - start here |
| `drone_sim.py` | Main script - run this |
| `drone_controller.py` | Keyboard -> movement |
| `overlap_calculator.py` | Footprint / GSD / capture-interval math |
| `road_utils.py` | Finds straight road segments for the N/P keys |
| `geo_utils.py` | Local metres -> synthetic lat/lon |
| `metadata_writer.py` | Writes `output/images/`, `geo.txt`, `metadata.csv`, `capture_log.json` |

## Design notes / assumptions made

- **No native drone model in CARLA.** CARLA doesn't simulate multirotor
  flight dynamics. The standard workaround (used here) is a free-floating,
  non-physics camera actor whose `carla.Transform` you set directly each
  tick - this is exactly what a downward-facing survey drone's camera feed
  looks like, without needing a rigid-body flight model you don't need for
  a computer-vision dataset.
- **Map**: `Town06` by default - it has long straight highway stretches
  (closest CARLA analogue to a National Highway) plus a small urban area.
  `Town07` (fully rural, single-lane country roads) is set as
  `ALT_RURAL_MAP` in `config.py` if you want a more rural look instead -
  just swap `CARLA_MAP` to it.
- **Speed** is a hard constant 30 km/h whenever a movement key is held
  (not a ceiling on a variable speed) - this matters because the overlap
  math assumes constant forward speed.
- **Altitude** defaults to 100 m (mirrors the common 120 m/400 ft
  real-world UAV ceiling). At this altitude/FOV/resolution the ~70%
  overlap requirement works out to a capture interval of roughly **4.8 s**
  (not exactly the requested 6-7 s - speed, altitude, FOV, resolution and
  overlap aren't independent; see `overlap_calculator.py`). Raise
  `ALTITUDE_M` toward ~135 m in `config.py` if you need it inside 6-7 s
  exactly; the interval is computed live either way, never hardcoded.
- **"Control the road" (point 6)** is implemented as N/P keys that jump
  the drone to the next/previous auto-detected straight road segment,
  since CARLA's road network isn't itself editable at runtime.
- **Output format**: `output/images/*.jpg` + `output/geo.txt` (OpenDroneMap
  geolocation-file format: `EPSG:4326` header, then `image lon lat alt`
  rows) + `output/metadata.csv` (full 6-DOF pose, timestamp, GSD per
  photo) + `output/capture_log.json` (run summary). Lat/lon are synthetic
  (CARLA has no real GPS) - see `geo_utils.py`.

## Quick controls reference
```
W / S       forward / backward
A / D       strafe left / right
Q / E       yaw left / right
R / F       ascend / descend
N / P       next / previous straight road segment
SPACE       pause / resume auto-capture
C           force an immediate photo
ESC         quit and finalize output
```
