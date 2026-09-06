# RoadSentinel Simulation Testbed v3.0 — Implementation Walkthrough

## Summary of Accomplishments

We transformed the RoadSentinel CARLA simulation testbed from a basic setup with square tile props and screen-space reflection blotches into a **photorealistic, physically plausible synthetic aerial road-health testing environment** tailored for 100 m drone surveys.

---

## 1. Visual Comparison & Artifacts

### 3D Road Depression & Water Representation

The previous square brown tile props (`static.prop.brokentile*` and `static.prop.dirtdebris*`) have been completely replaced with procedural 3D depressions featuring concave bowl depth gradients, rough sub-base aggregate cavity beds, crumbling spall lips, and capillary wet halos:

![Photorealistic 3D Pothole Geometry & Water States](/home/nitin-nandakumar/.gemini/antigravity-ide/brain/a0625195-d828-45c5-9e6e-dee14b7fb148/potholes_v2_comparison.jpg)
*Figure 1: (Left to Right) 1. Irregular natural dry pothole with depth gradient and meandering cracks; 2. Elongated longitudinal rutting pothole; 3. Water-filled pothole with capillary wet halo and submerged aggregate bed; 4. Jagged crumbling pothole with high turbidity silt water and perimeter fractures.*

---

### Road Surface Features & Resurfacing Patches

Road variation is physically localized rather than uniform noise. Resurfacing patches align with highway lane markings, accompanied by wheel-track wear bands and oil stains:

![Realistic Road Surface Degradation & Resurfacing Patches](/home/nitin-nandakumar/.gemini/antigravity-ide/brain/a0625195-d828-45c5-9e6e-dee14b7fb148/road_material_features.jpg)
*Figure 2: Highway lane wear tracks, bitumen resurfacing patch with dark crack sealant border, and vehicle oil drips.*

---

### High-Altitude Aerial Survey Overlap (100 m, 1920×1080, 30 km/h)

Sequential inspection frames demonstrate ~70% forward overlap along Town04, with zero cloud reflection SSR blotches on the asphalt:

````carousel
![Frame 2 at 100m Altitude](/home/nitin-nandakumar/.gemini/antigravity-ide/brain/a0625195-d828-45c5-9e6e-dee14b7fb148/aerial_inspection_frame_02.jpg)
<!-- slide -->
![Frame 3 at 100m Altitude (70% Forward Overlap)](/home/nitin-nandakumar/.gemini/antigravity-ide/brain/a0625195-d828-45c5-9e6e-dee14b7fb148/aerial_inspection_frame_03.jpg)
````

---

## 2. Files Modified and Added

### Files Added:
- [`run_simulation.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/run_simulation.py): Pre-CARLA interactive curses TUI launcher, Docker container detection/startup, socket readiness polling on port 2000, and session runner.
- [`road_health_pipeline/tests/test_interactive_sim.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/tests/test_interactive_sim.py): 9 unit tests covering interactive configuration, validation bounds, 6 shape families, water states, and seed reproducibility.

### Files Modified:
- [`env/config.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/config.py): Standardized survey altitude to 100.0 m, 60° calibrated FOV, tuned weather presets to eliminate SSR cloud blotches, and enforced monotonic defect distributions.
- [`env/road_injector.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/road_injector.py): Implemented 6 distinct physical shape families, removed all CARLA debris actors, added concave bowl depth rendering, turbidity blending, and lane-aligned resurfacing patches.
- [`env/drone_sim.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/drone_sim.py): Added `--fov` support, forward `--auto-fly` motion integration, contiguous memory guarantees for OpenCV, and auto-cleanup of stale session imagery.
- [`env/drone_controller.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/drone_controller.py): Enabled configurable survey `speed_mps` propagation in kinematic flight updates.
- [`env/geo_utils.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/geo_utils.py): Standardized GPS georeferencing to project reference datum (`REFERENCE_LAT = 13.0827, REFERENCE_LON = 80.2707`).
- [`env/evaluate_simulation.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/evaluate_simulation.py): Added `--corridor` filtering for localized survey flight evaluation.
- [`road_health_pipeline/inference/carla_pipeline.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/carla_pipeline.py): Updated defect ground GPS projection with camera yaw rotation and 60° survey lens geometry.
- [`road_health_pipeline/tests/run_tests.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/tests/run_tests.py): Integrated launcher and simulation test suites (expanded from 16 to 26 tests).
- [`README.md`](file:///home/nitin-nandakumar/Downloads/roadsentinel/README.md) and [`env/README.md`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/README.md): Documented the new interactive workflow, CLI arguments, and pipeline commands.

---

## 3. Exact Commands

### Interactive Simulation Launch
```bash
python run_simulation.py
```

### 100 m POOR + POST_RAIN Test
```bash
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

### Run RoadSentinel ML Pipeline
```bash
./.venv/bin/python orchestrator.py --input-dir ./env/output --no-server
```

### Run Evaluation Against Ground Truth
```bash
./.venv/bin/python env/evaluate_simulation.py \
  --ground-truth ./env/output/ground_truth.json \
  --predictions road_health_pipeline/output/analytics_demo/result.json \
  --corridor 0
```

---

## 4. Test Results

Executed the automated test suite:
```bash
./.venv/bin/python road_health_pipeline/tests/run_tests.py
```

```text
=================================================================
      RUNNING ROADSENTINEL ANALYTICS & PREDICTION TEST SUITE     
=================================================================
 [PASS] Area Estimation & Ray-tracing
 [PASS] Depth Metric Calculations
 [PASS] Severity Calculation (With & Without Depth)
 [PASS] Road Health Scoring & Deductions
 [PASS] Segment Aggregation & Defect Traceability
 [PASS] Spatial Index KD-Tree & Geofencing
 [PASS] Spatial Deduplication (3m Radius & Work Order Preservation)
 [PASS] Spatial Deduplication (Distinct Clusters >3m)
 [PASS] CARLA Ground Truth Validation
 [PASS] Temporal Prediction Interface
 [PASS] CARLA Temporal Dataset & Split Integrity
 [PASS] Procedural Defect Diversity & Non-Uniformity
 [PASS] Road-Health Scenario Monotonicity
 [PASS] Procedural Seed Reproducibility
 [PASS] Ground Truth JSON Schema Integrity
 [PASS] 3-Meter Spatial Matching & Error Metrics
 [PASS] Launcher Interactive Configuration Parsing
 [PASS] Launcher Configuration Validation Bounds
 [PASS] Scenario & Weather Selection Matrix
 [PASS] Seed & Altitude Parameter Propagation
 [PASS] Procedural Seed Reproducibility Verification
 [PASS] 6 Pothole Geometry Shape Families Diversity
 [PASS] Water State Diversity (Coverage, Turbidity, Halos)
 [PASS] Ground Truth Correctness & Schema Integrity
 [PASS] CARLA Server Readiness & Socket Probing
 [PASS] End-to-End Pipeline & JSON Schema Validation
=================================================================
 Test Summary: 26 passed, 0 failed out of 26 total.
=================================================================
```

---

## 5. Technical Implementation Details

### How Pothole Geometry Was Improved
1. **Removed all square props**: Eliminated `static.prop.brokentile*` and `static.prop.dirtdebris*` actors from CARLA world.
2. **6 Procedural Shape Families**:
   - `elongated_longitudinal`: Wheel-track rutting with aspect ratios $2.0 \to 3.8$, oriented along traffic flow ($\sim 0^\circ$).
   - `elongated_transverse`: Thermal contraction joint fractures spanning across lanes ($\sim 90^\circ$).
   - `irregular_natural`: Multi-harmonic fractal lobe boundaries with organic perimeter notches.
   - `jagged`: Sharp polygonal crumble sectors with edge irregularity.
   - `compound_cluster`: Merged multi-depression complexes with variable sub-lobes.
   - `partially_connected`: Twin dumbbell depressions with a raised sub-grade bridge.
3. **Continuous Depth & Shading Model**:
   - 2D Euclidean distance transform maps the concave bowl depth from rim ($0.0$) to center ($1.0$).
   - Directional sunlight shading ($gx \cdot s_x + gy \cdot s_y$) casts interior slope shadows on the leeward wall while highlighting the windward rim.
   - Smooth sloped transition drops from the surrounding asphalt color down into a dark aggregate gravel bed.
   - Spalling asphalt lip with micro-fracture noise and meandering fatigue cracks.

### How Water-Filled Potholes Are Represented
1. **Internal Water Level**: Potholes maintain a water coverage fraction ($0.25 \to 1.0$). If coverage $< 1.0$, water pools in the deepest section of the bowl, leaving an exposed damp stone shelf along the upper slopes.
2. **Turbidity & Depth Transparency**:
   - Clear water ($turbidity \to 0$): Low opacity ($0.35$), dark reflective pool allowing the rocky gravel bottom to remain visible.
   - Muddy water ($turbidity \to 1$): High opacity ($0.95$), turbid silt-brown coloration obscuring the bottom bed.
3. **Sky Glint & Capillary Damp Halo**:
   - Sun-aligned specular glint on the water surface.
   - Capillary damp halo extending outside the depression ($0.18 \to 0.45$ m), realistically darkening the wet porous asphalt surrounding the water pool.

### How the Pre-CARLA Launcher Operates
1. **Interactive TUI**: Implemented using `curses` with ANSI fallback. The menu renders **before** CARLA opens, allowing users to configure scenario, weather, altitude, speed, water ratio, seed, duration, and GUI/headless mode using arrow keys.
2. **Container Detection & Readiness**:
   - Checks if CARLA RPC is already active on port `2000` using TCP socket probing.
   - If inactive, inspects Docker for container `carla_sim`; starts existing or launches `carlasim/carla:0.9.16` with GPU runtime and OpenGL support.
   - Polls port `2000` with progress dots and a 35-second timeout.
3. **Session Orchestration**: Validates all parameters, writes `session_metadata.json`, launches `drone_sim.py`, and prints a clean session summary.
