# RoadSentinel — Pipeline Files Summary

This document provides a concise architectural summary of all files utilized and modified across the RoadSentinel end-to-end aerial road health monitoring, Town04 geofencing, and deterioration prediction pipeline.

---

## 1. Master Orchestration & Execution

### [`orchestrator.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/orchestrator.py)
* Coordinates all six stages of the RoadSentinel pipeline across CARLA simulation, ML inference, and dashboard serving.
* Manages sub-environments by dispatching simulation flight to `carla_env` (Python 3.10) and neural processing to `.venv` (PyTorch).
* Explicitly initializes CARLA Town04, introduces generated road defects, and executes a 15-second manual drone flight session.
* Launches and monitors background services with clean process teardown, port conflict resolution, and automatic browser dashboard launching.

### [`run_demo.sh`](file:///home/nitin-nandakumar/Downloads/roadsentinel/run_demo.sh)
* Provides a one-command CLI entry point with colored status logging and environment validation.
* Verifies active virtual environments, Docker availability, and necessary dependencies before triggering execution.
* Forwards command-line flags (such as duration, map selection, and headless mode) directly to `orchestrator.py`.
* Manages clean terminal teardown and displays direct links to the running dashboard interface upon completion.

---

## 2. CARLA Simulation & Aerial Image Capture Layer (`env/`)

### [`env/drone_sim.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/drone_sim.py)
* Connects to CARLA, deploys the Town04 map, and executes road defect / pothole injection before spawning the drone actor.
* Spawns the downward-facing (nadir) drone camera actor in CARLA Town04 and opens the interactive preview window for 15s manual flight.
* Converts local Cartesian XYZ coordinates to standard WGS-84 Latitude and Longitude using CARLA's `transform_to_geolocation()`.
* Enforces a 15-second demo duration with live telemetry HUD, graceful actor destruction, and metadata finalization.

### [`env/config.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/config.py)
* Defines global simulation parameters including CARLA map selection (Town04), server port, and synchronous tick rate.
* Specifies aerial flight constraints such as constant horizontal speed (30 km/h) and nominal survey altitude (35–50 m).
* Configures camera sensor properties including nadir pitch (-90°), horizontal field of view (100°), and resolution (640×360).
* Establishes synthetic reference geolocations (latitude/longitude) and output directory paths for captured frames.

### [`env/drone_controller.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/drone_controller.py)
* Translates user keyboard events (W/A/S/D/Q/E/R/F) and automated commands into 6-DOF coordinate updates for the drone actor.
* Clamps horizontal velocity to an exact constant speed regardless of diagonal flight vectors to preserve overlap math.
* Manages yaw rotation, altitude ascent/descent boundaries, manual capture triggers, and road segment jumping.
* Signals simulation exit and triggers clean actor teardown upon receiving termination signals.

### [`env/metadata_writer.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/metadata_writer.py)
* Writes sequentially captured road RGB frames to disk with standardized indexing and JPEG compression.
* Generates OpenDroneMap/COLMAP-compatible `geo.txt` containing real-world OpenDRIVE latitude, longitude, and altitude.
* Exports detailed `metadata.csv` recording full 6-DOF camera pose, ground sampling distance (GSD), and simulation timestamps.
* Produces `capture_log.json` containing run-level performance summaries and simulation parameters at shutdown.

### [`env/overlap_calculator.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/overlap_calculator.py)
* Derives ground footprint dimensions (length and width in meters) based on altitude, sensor resolution, and camera FOV.
* Calculates Ground Sampling Distance (GSD in cm/pixel) to ensure road defect textures meet resolution requirements.
* Mathematically determines required time intervals between shutter releases to achieve the targeted 70% forward overlap.
* Generates startup validation reports summarizing flight physics and capture geometry.

### [`env/road_utils.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/road_utils.py)
* Inspects the OpenDRIVE map topology in CARLA to locate continuous straight highway road corridors.
* Evaluates road heading variations over lookahead distances to filter out sharp turns and intersections.
* Returns ordered waypoint sequences defining safe, unobstructed flight paths for aerial inspection.
* Provides reference coordinates for spawning drones directly above centered lane dividers.

### [`env/road_injector.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/road_injector.py)
* Procedurally spawns synthetic road defect meshes (potholes, cracks, and water puddles) on CARLA road surfaces.
* Positions defects within the drone camera's visible flight corridor based on road waypoint coordinates.
* Generates ground-truth bounding boxes, metric depths, and surface areas for downstream pipeline validation.
* Cleans up spawned defect actors from the CARLA world upon simulation shutdown.

### [`env/textures.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/textures.py)
* Generates procedural texture maps and surface materials for asphalt cracking and pothole depth rendering.
* Provides color and reflectivity profiles distinguishing dry asphalt from water-filled cavities.
* Supplies synthetic visual patterns for evaluating computer vision segmentation under diverse lighting.
* Enhances the visual fidelity of CARLA road anomalies during high-altitude drone capture.

### [`env/geo_utils.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/env/geo_utils.py)
* Converts CARLA local Cartesian coordinates into real-world geographic coordinates using `carla_transform_to_geolocation()`.
* Implements inverse planar projection formulas converting GPS coordinates back into local metric offsets.
* Anchors coordinates to standardized Town04 reference datums for spatial indexing and geofencing.
* Ensures exported camera poses match standard GIS and OpenDroneMap coordinate conventions.

---

## 3. Data Schemas & Shared Infrastructure (`road_health_pipeline/common/`)

### [`road_health_pipeline/common/schemas.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/common/schemas.py)
* Defines typed dataclass contracts for all stages (`Telemetry`, `CandidateRegion`, `SegmentationResult`, `DefectMeasurement`).
* Encapsulates explainable multi-factor severity breakdowns and 0–100 segment road health scores.
* Provides the unified `InferenceResult` schema supporting defect traceability, telemetry, and temporal predictions.
* Implements robust JSON serialization methods (`to_dict()`, `to_json()`) across all analytical data models.

### [`road_health_pipeline/common/io_utils.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/common/io_utils.py)
* Handles RGB and BGR image loading with color conversion and boundary validation.
* Provides standardized JSON file reading and writing utilities with UTF-8 encoding.
* Generates ISO-8601 UTC timestamp strings for synchronized telemetry logging.
* Ensures safe directory creation and file path normalization across the workspace.

### [`road_health_pipeline/common/geometry.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/common/geometry.py)
* Implements metric pinhole ray-tracing formulas connecting pixel coordinates to real-world ground distances.
* Calculates ground sampling distance and pixel surface area based on flight altitude and camera FOV.
* Computes bounding box intersections, aspect ratios, and mask circularity metrics.
* Ensures mathematical separation between RGB-estimated dimensions and sensor ground-truth depth.

---

## 4. Feature Extraction, Segmentation & Detection (`road_health_pipeline/inference/`)

### [`road_health_pipeline/config.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/config.py)
* Central configuration declaring neural network models, patch dimensions, and memory bank parameters.
* Sets configurable weights and normalization constants for defect severity and road health index calculations.
* Defines thresholds for FAISS k-NN anomaly detection, SAM2 road masking, and shadow suppression.
* Configures geographic clustering grid sizes (50 m) and default temporal prediction horizons (30 days).

### [`road_health_pipeline/inference/sam2_mask.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/sam2_mask.py)
* Wraps the Meta Segment Anything 2 (SAM2.1 Hiera) model for automated zero-shot road surface segmentation.
* Generates road region-of-interest (ROI) masks to isolate drivable pavement from roadside terrain and sky.
* Refines rough bounding-box anomaly prompts into pixel-accurate defect contour masks.
* Manages GPU/CPU tensor placement and image resizing for real-time inference.

### [`road_health_pipeline/inference/dinov2_embed.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/dinov2_embed.py)
* Loads the DINOv2 vision transformer (`dinov2_vits14`) to extract dense visual patch embeddings ($14\times14$ px).
* Filters patch tokens to extract embeddings exclusively from valid road mask regions.
* Performs L2 vector normalization to support inner-product cosine similarity searching.
* Retains 2D spatial grid coordinates for mapping feature vectors back to image pixels.

### [`road_health_pipeline/inference/anomaly_detector.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/anomaly_detector.py)
* Uses a FAISS IndexFlatIP index populated with patch embeddings of known undamaged road surfaces.
* Queries nearest-neighbor distances for each image patch to compute localized anomaly scores.
* Reconstructs 2D continuous anomaly heatmaps across the road surface at native image resolution.
* Applies percentile-based thresholding to distinguish anomalies from normal pavement variations.

### [`road_health_pipeline/inference/pothole_localizer.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/pothole_localizer.py)
* Performs morphological connected-component analysis on thresholded anomaly heatmaps to extract candidate regions.
* Computes bounding boxes, mask areas, centroid coordinates, and circularity metrics for each candidate.
* Suppresses shadow artifacts by evaluating local color variance and lightness gradients.
* Formats candidate detections into structured `CandidateRegion` objects for downstream measurement.

---

## 5. Analytics, Spatial Indexing & Prediction

### [`road_health_pipeline/inference/spatial_index.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/spatial_index.py)
* Implements a 2D KD-Tree spatial index over detected road defects using georeferenced Latitude and Longitude.
* Enables sub-millisecond dynamic radius queries (`query_radius`) and nearest-hazard lookups (`query_nearest`).
* Generates multi-tiered geofencing hazard zones with configurable warning and critical boundary rings.
* Evaluates real-time driver proximity to generate automated early-warning alerts and speed advisories.

### [`road_health_pipeline/inference/run_inference.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/run_inference.py)
* Acts as the primary inference engine connecting DINOv2/SAM2 feature extraction with the analytics layer.
* Exposes both single-image functional interfaces (`infer()`) and reusable batch pipelines (`RoadSentinelPipeline`).
* Calculates physical measurements, severity breakdowns, road health scores, and temporal predictions for each frame.
* Generates visual overlay artifacts (`detection_overlay.jpg`, `severity_overlay.jpg`, `road_health_overlay.jpg`).

### [`road_health_pipeline/inference/defect_classifier.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/defect_classifier.py)
* Classifies detected anomalies into specific defect categories (potholes, cracks, surface wear, water-filled potholes).
* Analyzes HSV color distribution, intensity, and Laplacian texture variance to detect standing water hazards.
* Evaluates shape aspect ratios and perimeter-to-area proportions to differentiate elongated cracks from potholes.
* Returns fine-grained defect labels and water detection confidence scores.

### [`road_health_pipeline/inference/area_estimator.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/area_estimator.py)
* Converts 2D pixel mask areas into physical metric surface areas ($m^2$) using pinhole camera geometry.
* Incorporates drone flight altitude and horizontal field of view to determine metric ground sampling distance.
* Handles invalid or missing altitude values by returning safe `None` outputs rather than uncalibrated estimates.
* Provides the physical foundation for defect severity and road maintenance cost calculations.

### [`road_health_pipeline/inference/depth_estimator.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/depth_estimator.py)
* Provides the interface for monocular depth estimation and metric depth sensor data ingestion.
* Implements `NullDepthEstimator` to ensure safe, transparent execution when no dedicated depth sensor is present.
* Extracts depth percentiles and median cavity depths within defect mask boundaries when depth maps are supplied.
* Keeps metric depth estimates scientifically decoupled from ground-truth simulation depth.

### [`road_health_pipeline/inference/severity_estimator.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/severity_estimator.py)
* Evaluates defect severity on a transparent 0–100 scale based on physical area, depth, water, and surrounding damage.
* Dynamically renormalizes available weights when metric depth is unavailable to maintain scientific honesty.
* Classifies continuous severity scores into standard municipal tiers: Low, Medium, High, and Critical.
* Produces explainable component breakdowns detailing the exact numerical contribution of each factor.

### [`road_health_pipeline/inference/road_health_scorer.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/road_health_scorer.py)
* Computes a standardized 0–100 road health score for surveyed road segments (100 = perfect, 0 = impassable).
* Deducts configurable penalty points based on pothole density, peak defect severity, crack ratios, and water hazards.
* Categorizes road segments into actionable condition classes: Good ($\ge 80$), Fair ($\ge 60$), Poor ($\ge 40$), and Critical ($< 40$).
* Generates clear textual explanations outlining the primary factors driving road health degradation.

### [`road_health_pipeline/inference/segment_aggregator.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/segment_aggregator.py)
* Groups localized defect measurements into unified road segments using spatial geospatial binning (50 m grid).
* Computes segment-level summary statistics including total defect counts, aggregate damaged area, and max severity.
* Maintains full bidirectional traceability between aggregated segment records and individual defect detections.
* Formats aggregated summaries into `RoadSegmentAggregate` records ready for database storage and dashboard rendering.

### [`road_health_pipeline/inference/gps_localizer.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/gps_localizer.py)
* Maps local camera coordinates and image pixel offsets into global WGS-84 geographic coordinates.
* Attaches verified latitude and longitude values to incoming drone frame telemetry.
* Provides fallback coordinate synthesis for simulated flights anchored to configured datum origins.
* Enables precise geographic positioning of detected road defects for municipal maintenance routing.

### [`road_health_pipeline/inference/visualizer.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/visualizer.py)
* Renders bounding boxes and defect labels on RGB frames for visual detection verification.
* Produces color-coded severity overlays (green $\rightarrow$ yellow $\rightarrow$ orange $\rightarrow$ red) illustrating defect criticality.
* Generates head-up display (HUD) banners overlaying road health scores and condition classes onto road imagery.
* Saves rendered diagnostic images to disk for inspection gallery rendering in the dashboard.

### [`road_health_pipeline/prediction/progression_model.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/prediction/progression_model.py)
* Forecasts road health degradation and pothole formation probabilities over configurable horizons (e.g., 30 days).
* Fits trend lines across historical inspection records to estimate health decline rates ($\Delta \text{score} / \text{day}$).
* Flags emerging defect hazards when surface cracking and wear exceed formation thresholds.
* Tags all synthetic predictions with scientific status indicators (`CARLA-SYNTHETIC ONLY`).

### [`road_health_pipeline/prediction/carla_temporal_dataset.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/prediction/carla_temporal_dataset.py)
* Simulates multi-epoch road deterioration sequences in CARLA (healthy $\rightarrow$ micro-cracks $\rightarrow$ wear $\rightarrow$ pothole).
* Maintains consistent segment identifiers across consecutive inspection epochs to prevent data leakage.
* Generates verified progression labels for training and validating temporal prediction algorithms.
* Provides reproducible synthetic benchmarks for evaluating deterioration forecasters.

---

## 6. VLM Work Orders & Town04 Dashboard Serving

### [`road_health_pipeline/vlm_work_order_gen.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/vlm_work_order_gen.py)
* Ingests post-inference `result.json` and filters for high and critical severity road segment defects.
* Crops high-resolution visual bounding-box defect regions from corresponding drone frames.
* Queries Vision-Language Models (Google Gemini 2.0 Flash or built-in offline domain-rule engine) to generate work orders.
* Exports structured municipal repair orders specifying repair actions, required materials, equipment, crew sizes, and MUTCD traffic safety measures into `work_orders.json`.

### [`road_health_pipeline/inference/server.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/inference/server.py)
* FastAPI web server providing REST API endpoints (`/infer`, `/api/results`, `/api/work_orders`, `/api/geofence/query`, `/api/geofence/check_proximity`).
* Hosts the custom CARLA Town04 government dashboard with glassmorphic dark theme and interactive map layers.
* Renders dynamically scaled defect markers, geofence boundary rings, and visual overlay galleries.
* Features an interactive driver-proximity simulator with real-time early-warning HUD alerts and speed advisories.

### [`road_health_pipeline/scripts/run_analytics_e2e.py`](file:///home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline/scripts/run_analytics_e2e.py)
* Executes the complete post-inference analytics workflow on synthetic or captured road imagery.
* Generates sample defect scenes, computes health scores, builds KD-Tree spatial index, and runs temporal prediction.
* Evaluates predictions against ground truth and outputs visual overlays and `result.json` for end-to-end testing.
* Serves as the standalone automated integration test for the analytics and scoring modules.
