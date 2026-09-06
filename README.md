# RoadSentinel — UAV Aerial Road-Inspection Simulation & AI Pipeline

RoadSentinel is an end-to-end aerial road-health inspection platform combining a photorealistic synthetic CARLA drone simulation testbed, foundational vision models (DINOv2 + SAM 2), 3-meter spatial deduplication, deterioration forecasting, and automated vision-language maintenance work order dispatch.

---

## 🌟 Key Capabilities

1. **Pre-CARLA Interactive TUI Launcher (`run_simulation.py`)**:
   - Visual terminal UI with dropdown-style selectors for Scenario, Weather, Altitude, Speed, Water condition, Procedural Seed, Duration, and Rendering mode.
   - Appears **before** CARLA is opened.
   - Automatically detects running CARLA instances or starts the `carlasim/carla:0.9.16` Docker container with GPU acceleration and socket readiness polling on port 2000.

2. **Photorealistic 3D Road Depressions**:
   - Replaced all flat rectangular tile meshes with procedural 3D depression geometry:
     - 6 physical shape families: `elongated_longitudinal`, `elongated_transverse`, `irregular_natural`, `jagged`, `compound_cluster`, `partially_connected`.
     - Sloped concave bowl depth transitions from surface asphalt down into rough aggregate cavity floor.
     - Naturalistic crumbling spall perimeter and meandering asphalt fatigue cracks.
     - No rectangular borders or fake props sitting on top of the road.

3. **Water-Filled Potholes (Primary Research Focus)**:
   - True depressed water geometry following the internal cavity contours.
   - Variable water coverage ($0.25 \to 1.0$), water depth, and turbidity (clear dark transparent road water showing rock bed vs. murky silt runoff).
   - Sun-aligned specular glint and capillary damp asphalt halos darkening around the pothole perimeter.

4. **Realistic Road Surface Degradation**:
   - Highway lane-aligned asphalt resurfacing patches with dark bitumen sealant borders.
   - Longitudinal tire-wear bands along vehicle wheel tracks.
   - Oil drip stains and localized asphalt aggregate variation.
   - Screen-space reflection (SSR) cloud blotch artifacts eliminated.

5. **Calibrated 100 m Drone Inspection Rig**:
   - Nadir 1920×1080 downward camera at 100 m altitude with calibrated 60° horizontal FOV (~24mm full-frame survey lens, GSD 6.0 cm/px).
   - Survey flight at 30 km/h with ~70% sequential image overlap.

6. **Precision Ground Truth & Quantitative Evaluation**:
   - Ground truth catalog exported to `ground_truth.json` with exact metric coordinates, dimensions, depth, water coverage, turbidity, and WGS-84 coordinates.
   - Strict ground-truth isolation: ML inference models predict independently from RGB imagery and telemetry without cheating.
   - Evaluation benchmarking via `env/evaluate_simulation.py`.

---

## 🚀 Quickstart Workflow

### 1. Interactive Simulation Launch (Preferred Human Workflow)

Run the single interactive launcher command:

```bash
python run_simulation.py
```

1. The interactive terminal menu opens before CARLA starts:
   - Navigate with **Up / Down** arrows
   - Cycle options with **Left / Right** arrows or **Space**
   - Press **Enter** on `[ START SIMULATION ]`
2. CARLA 0.9.16 Docker container is automatically detected or launched.
3. The drone autonomously surveys the highway corridor, capturing high-resolution photos and recording telemetry.
4. Final images and `ground_truth.json` are written to `env/output/`.

---

### 2. Automated & Batch CLI Execution

For automated benchmarks, headless CI runs, or scriptable experiments, pass CLI flags to bypass the TUI:

```bash
# 100 m POOR + POST_RAIN Acceptance Test Flight
python run_simulation.py \
  --non-interactive \
  --scenario poor \
  --weather post_rain \
  --altitude 100.0 \
  --speed 30.0 \
  --seed 42 \
  --duration 60.0 \
  --headless
```

Available CLI Options:
- `--scenario`: `healthy`, `moderate`, `poor`, `critical`
- `--weather`: `post_rain`, `clear`, `overcast`, `wet`, `rain`, `low_light`, `sunset`, `early_morning`
- `--altitude`: Drone survey altitude in meters (default `100.0`)
- `--speed`: Survey flight speed in km/h (default `30.0`)
- `--water`: `Automatic by Scenario`, `Mostly Dry`, `Mixed`, `Mostly Water-Filled`
- `--seed`: Deterministic procedural seed integer (e.g. `42`) or `random`
- `--duration`: Flight time in seconds (or `0` for continuous manual flight)
- `--headless`: Run without opening GUI window
- `--standalone`: Force standalone procedural flight engine without CARLA server

To generate multi-axis scenario matrices across scenarios, weathers, altitudes, and seeds:
```bash
./.venv/bin/python env/generate_road_scenarios.py --scenarios poor critical --weathers clear post_rain --seeds 42 101
```

---

### 3. Run RoadSentinel ML & Analytics Pipeline

Process the captured aerial images with DINOv2 feature extraction, SAM 2 segmentation, depth estimation, 3-meter spatial deduplication, and VLM maintenance work orders:

```bash
./.venv/bin/python orchestrator.py --input-dir ./env/output --no-server
```

Output files produced:
- `road_health_pipeline/output/analytics_demo/result.json` (detections, 3m spatial clusters, severity metrics)
- `road_health_pipeline/output/analytics_demo/work_orders.json` (municipal repair work orders)

---

### 4. Evaluate Against CARLA Ground Truth

Compare predicted detections and physical estimates against known CARLA ground truth:

```bash
./.venv/bin/python env/evaluate_simulation.py \
  --ground-truth ./env/output/ground_truth.json \
  --predictions road_health_pipeline/output/analytics_demo/result.json \
  --corridor 0
```

Evaluates:
- Detection Precision, Recall, and F1-Score
- 3-meter spatial localization error (MAE & RMSE)
- Area estimation error (MAE in m²)
- Depth estimation error (MAE in cm)
- Water hazard classification F1 and confusion matrix
- Macro road-health scenario concordance

---

## 🧪 Running the Test Suite

Run the full automated test suite (26 unit and regression tests):

```bash
./.venv/bin/python road_health_pipeline/tests/run_tests.py
```

Test coverage includes:
- Interactive configuration parsing and validation bounds
- Scenario and weather selection matrices
- Seed and altitude parameter propagation
- Procedural seed bitwise reproducibility
- Representation of all 6 pothole shape families
- Water state diversity (coverage fractions, turbidity, wet halos)
- Ground truth JSON schema integrity
- CARLA Docker socket probing and readiness detection
- 3-meter spatial deduplication and work order preservation
- End-to-end pipeline analytics and JSON schema validation
