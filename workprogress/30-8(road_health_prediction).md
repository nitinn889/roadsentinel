# RoadSentinel — Marion Task
## Road-Health Scoring, FAISS Threshold Tuning & Deterioration Prediction

You are an AI coding agent working on the RoadSentinel project. Nitin has successfully deployed the DINOv2 + SAM2 inference pipeline, generated the real-road memory bank, and implemented the batch evaluation script (`run_full_eval.sh`) for the 5GB dataset. Your task is to handle the **analytics layer after the trained DINOv2 + SAM2 pipeline** and optimize the anomaly detection thresholds.

Your goal is to turn detected road defects into measurable properties, a defensible road-health/severity score, and a foundation for deterioration/pothole prediction. **Do not modify or push `main`.** The final dashboard will be handled by another teammate; user alerts are out of scope.

---

## 1. Project Context

Repository:

```text
[https://github.com/nitinn889/roadsentinel](https://github.com/nitinn889/roadsentinel)
```

Local project:

```text
~/Downloads/roadsentinel/
```

RoadSentinel flow:

```text
CARLA road
  ↓
Drone RGB photographs
  ↓
DINOv2 + SAM2 (Memory Bank & FAISS)
  ↓
Defect detections/masks
  ↓
Area / geometry / depth
  ↓
Severity + Road Health Score
  ↓
Future deterioration / pothole prediction
  ↓
Geotagged record
  ↓
Government dashboard
```

The project already has a CARLA drone simulation, a pothole-containing road environment, RGB capture, and trained/working DINOv2 + SAM2 components. Read the actual repository before changing anything.

---

## 2. Git Safety

```bash
cd ~/Downloads/roadsentinel
git fetch origin
git checkout marion-sam2-dinov2
git pull origin marion-sam2-dinov2
git checkout -b marion-road-health-analytics
```

Verify:

```bash
git branch --show-current
git status
git remote -v
```

Expected branch:

```text
marion-road-health-analytics
```

Never push or merge into `main`.

---

## 3. Audit First

Inspect at minimum:

```text
road_health_pipeline/config.py
road_health_pipeline/common/
road_health_pipeline/inference/area_estimator.py
road_health_pipeline/inference/depth_estimator.py
road_health_pipeline/inference/gps_localizer.py
road_health_pipeline/inference/run_inference.py
road_health_pipeline/evaluation/depth_metrics.py
run_full_eval.sh
output/real_memory_bank/
```

Also inspect the current DINOv2/SAM2 output schema and CARLA metadata. Reuse existing interfaces. Do not rewrite working code merely for style.

---

## 4. Main Objective

Implement:

```text
SAM2 mask (Post-FAISS Tuning)
   ↓
Defect measurements
   ↓
Severity
   ↓
Road-health score
   ↓
Temporal change
   ↓
Prediction interface
```

Each result must remain traceable to an individual defect and road segment.

---

## 5. FAISS Tuning & False Positive Reduction

Nitin's initial segmentation benchmark on `pothole600_test` revealed a significant gap between the Mean IoU (~20.4%) and Median IoU (~2.3%). The pipeline is currently struggling with false positives (shadows, patches) and incomplete bounds.
* You must adjust the FAISS distance threshold to improve DINOv2's anomaly detection rate.
* Filter out non-defect anomalies before they reach the SAM2 prompting stage.

---

## 6. Defect Measurements

Support a structured record containing where available:

```text
defect_type
confidence
bbox
mask_area_pixels
estimated_area_m2
estimated_depth_m
is_water_filled
water_confidence
crack_or_damage_extent
road_segment_id
timestamp
latitude
longitude
```

Use `null` when a value cannot honestly be calculated. Distinguish, where possible:

```text
pothole
water-filled pothole
crack/deteriorated region
unknown road anomaly
```

Do not classify every DINOv2 anomaly as a pothole.

---

## 7. Pothole Severity

Create a transparent, configurable severity model using measurable features such as:

```text
area
estimated depth
shape/extent
water-filled status
surrounding damage/cracking
confidence
```

Keep thresholds/weights in configuration. Do not present arbitrary weights as scientifically validated.

Provide a breakdown such as:

```json
{
  "severity": "high",
  "severity_score": null,
  "severity_components": {
    "area": null,
    "depth": null,
    "water": null,
    "surrounding_damage": null
  }
}
```

---

## 8. Road Health Score

Implement a configurable **0–100 segment-level score**:

```text
100 = healthy
0   = severely hazardous
```

Potential components:

```text
pothole density
pothole severity
crack/damage extent
surface deterioration
water hazards
confidence
```

The score must be:
* transparent
* reproducible
* configurable
* explainable

Example structure:

```json
{
  "road_health_score": null,
  "condition_class": null,
  "components": {
    "pothole_penalty": null,
    "crack_penalty": null,
    "water_penalty": null,
    "surface_penalty": null
  }
}
```

Do not claim validation until evaluated against appropriate data.

---

## 9. Road-Segment Aggregation

Create/reuse a stable `road_segment_id`. Aggregate detections by segment and retain:

```text
total potholes
total damaged area
average/max severity
water hazards
crack/damage indicators
health score
inspection timestamp
GPS/geospatial information
```

Individual detections must remain traceable from the segment result.

---

## 10. CARLA Ground Truth

CARLA is a controlled evaluation environment. Do **not** feed CARLA depth into the normal RGB-only inference path. Use CARLA ground truth to compare:

```text
true pothole size
true depth
true location
true water status
```

Where available calculate:

```text
area error
location error
depth MAE
Depth RMSE
severity agreement
water classification performance
```

Only calculate metrics supported by actual ground truth.

---

## 11. Depth / Geometry Evaluation

Inspect:

```text
inference/depth_estimator.py
evaluation/depth_metrics.py
common/geometry.py
```

Ensure units are explicit, masks are handled correctly, invalid pixels are handled, and RGB-estimated depth remains separate from ground-truth depth. Use existing camera/altitude/calibration metadata from CARLA where appropriate.

---

## 12. Prediction — Scientific Constraint

RoadSentinel eventually needs **deterioration and pothole-formation prediction**. Ordinary still-image pothole datasets do not automatically provide prediction labels. **Do not fabricate progression labels.**

Use CARLA for controlled temporal prototypes by repeatedly observing the same road segment, for example:

```text
t1: healthy/minor crack
t2: increased wear/cracking
t3: depression
t4: pothole
```

Every sequence needs a stable road-segment ID and known progression. Clearly label this as **CARLA synthetic temporal evaluation**.

---

## 13. Prediction Interface

Create a modular interface accepting:

```text
road_segment_id
historical health scores
defect history
time intervals
severity evolution
defect growth
spatial context
```

Possible output:

```json
{
  "deterioration_probability": null,
  "pothole_formation_probability": null,
  "prediction_horizon_days": null
}
```

Start with an interpretable baseline such as logistic regression, random forest, or gradient boosting when real/synthetic sequence data is available. Do not jump to a complex temporal neural model unless justified.

---

## 14. Prediction Evaluation

For CARLA sequences, evaluate where valid:

```text
precision
recall
F1
ROC-AUC
calibration
```

Also verify whether predicted deterioration direction agrees with simulated progression. Keep complete road segments/sequences separated between train and test to avoid temporal leakage.

---

## 15. Stable Output Schema

Extend the existing schema rather than creating an incompatible one. It should conceptually support:

```json
{
  "image_id": "...",
  "timestamp": "...",
  "road_segment_id": "...",
  "geolocation": {"lat": null, "lon": null},
  "detections": [],
  "road_health": {
    "score": null,
    "condition": null
  },
  "prediction": {
    "deterioration_probability": null,
    "pothole_formation_probability": null,
    "horizon_days": null
  }
}
```

Use actual values only when computed.

---

## 16. Testing

Add tests for:

* area estimation
* depth evaluation
* severity calculation
* road-health scoring
* road-segment aggregation
* CARLA ground-truth comparison
* temporal sequence handling
* prediction interface
* final JSON schema

Include cases for:

```text
no defects
one pothole
multiple potholes
water-filled pothole
missing depth
missing GPS
low-confidence detection
incomplete temporal sequence
```

---

## 17. End-to-End Test

Actually execute:

```text
RGB image
 ↓
DINOv2 (Tuned FAISS)
 ↓
SAM2
 ↓
defect mask
 ↓
area
 ↓
depth
 ↓
severity
 ↓
road-health score
 ↓
prediction interface
 ↓
structured result
```

Generate where practical:

```text
outputs/
├── detection_overlay.jpg
├── severity_overlay.jpg
├── road_health_overlay.jpg
└── result.json
```

Do not generate fake numerical values.

---

## 18. Scientific Status

Every feature must be identified as one of:

```text
IMPLEMENTED
PROTOTYPE
REQUIRES REAL DATA
CARLA-SYNTHETIC ONLY
```

Do not claim CARLA prediction experiments prove real-world performance.

---

## 19. Documentation

Update the relevant README with:

* score formulation
* inputs/outputs
* CARLA evaluation
* prediction architecture
* configuration
* limitations

Do not commit datasets, model weights, secrets, or huge generated artifacts.

---

## 20. Git Commit and Push

Before committing:

```bash
git status
git diff
git branch --show-current
```

Verify:

```text
marion-road-health-analytics
```

Commit:

```bash
git add road_health_pipeline/
git commit -m "Optimize FAISS thresholds, add road health scoring and prediction"
```

Push only:

```bash
git push -u origin marion-road-health-analytics
```

Never push or merge into `main`.

---

## 21. Final Handoff

Provide:

```text
Branch:
Commit:
GitHub branch URL:
```

Implementation status:

```text
FAISS Tuning & Metrics:
Area estimation:
Depth evaluation:
Severity:
Road-health score:
Segment aggregation:
CARLA ground-truth evaluation:
Temporal sequence generation:
Prediction baseline:
Prediction evaluation:
End-to-end pipeline:
```

For each use:

```text
PASS
FAIL
PROTOTYPE
REQUIRES REAL DATA
CARLA-SYNTHETIC ONLY
```

Report actual measured results and runtime. Clearly list remaining needs such as calibration, real temporal data, additional validation, or dashboard integration.

## Definition of Done

The trained DINOv2 + SAM2 output has been tuned for accurate masking and converted into a **working, traceable and explainable road-health layer**, with CARLA-based evaluation and a clean temporal prediction interface for future real-world validation.