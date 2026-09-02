# RoadSentinel — SAM2 + DINOv2 Development Task

## Purpose

This document contains the development instructions for the **SAM2 + DINOv2 pipeline** of RoadSentinel.

RoadSentinel has expanded from simple pothole detection toward:

> **road-defect identification + road-health/severity assessment + depth estimation + future deterioration/pothole prediction**

The operational system is intended to use **RGB drone imagery**. CARLA depth and external depth/disparity data are for evaluation, not primary inference.

### Critical instructions

* **Do not commit or push anything to `main`.**
* Work only on your own branch.
* I will review your branch and decide what is merged.
* The complete datasets are available locally, but do not assume they should be copied into GitHub.
* Use existing data and small subsets for development/testing where appropriate.
* Do not claim accuracy without real evaluation.
* Clearly distinguish **identification, segmentation, scoring, depth validation, and prediction**.
* Do not invent prediction labels where temporal data does not exist.

---

# 1. Repository

GitHub:

https://github.com/nitinn889/roadsentinel

Clone:

```bash
git clone https://github.com/nitinn889/roadsentinel.git
cd roadsentinel
```

If already cloned:

```bash
cd roadsentinel
git checkout main
git pull origin main
```

Verify:

```bash
git status
git remote -v
git branch --show-current
```

---

# 2. Development Branch

Do not modify `main`.

```bash
git checkout -b marion-sam2-dinov2
git branch --show-current
```

Expected:

```text
marion-sam2-dinov2
```

Never push or merge into `main`.

---

# 3. Current Pipeline Structure

```text
road_health_pipeline/
├── README.md
├── config.py
│
├── carla_sim/
│   ├── __init__.py
│   ├── drone_controller.py
│   └── rgb_depth_capture.py
│
├── common/
│   ├── __init__.py
│   ├── geometry.py
│   ├── io_utils.py
│   └── schemas.py
│
├── evaluation/
│   ├── __init__.py
│   └── depth_metrics.py
│
├── inference/
│   ├── __init__.py
│   ├── anomaly_detector.py
│   ├── area_estimator.py
│   ├── depth_estimator.py
│   ├── dinov2_embed.py
│   ├── gps_localizer.py
│   ├── pothole_localizer.py
│   ├── run_inference.py
│   ├── sam2_mask.py
│   └── server.py
│
├── memory_bank/
│   ├── __init__.py
│   ├── build_memory_bank.py
│   ├── coreset.py
│   └── validate_memory_bank.py
│
├── pi_edge/
│   ├── __init__.py
│   ├── edge_processor.py
│   ├── telemetry.py
│   └── uploader.py
│
├── requirements-gpu.txt
└── requirements-pi.txt
```

---

# 4. Intended RoadSentinel Architecture

The current target is:

```text
RGB drone image
        ↓
DINOv2 representation
        ↓
Healthy-road memory bank / anomaly detection
        ↓
Candidate road-damage regions
        ↓
SAM2 segmentation
        ↓
Defect characterization
        ↓
Area estimation
        ↓
RGB depth-estimation interface
        ↓
Severity / road-health interface
        ↓
GPS localization
```

Future extension:

```text
Survey t1
   ↓
Road-health state
   ↓
Survey t2
   ↓
Change/deterioration
   ↓
Future risk
   ↓
Pothole formation prediction
```

Prediction must remain a **future-stage interface** until suitable temporal/progression data exists.

---

# 5. Current RoadSentinel Datasets

The datasets currently available for this project are:

```text
RoadSentinel_datasets/
├── mwpd/
├── pi5_smoketest_subset/
├── pothole_600/
├── pothole_mix/
├── qr4change/
├── rdd2022/
├── rdd2022_full/
└── water_filled_potholes/
```

The existing development subset inside `rdd2022/` must remain untouched.

### Dataset roles

| Dataset                 | Primary role                                  |
| ----------------------- | --------------------------------------------- |
| `pothole_mix`           | Pothole segmentation / SAM2                   |
| `pothole_600`           | Segmentation + depth/disparity validation     |
| `water_filled_potholes` | Dry vs water-filled pothole handling          |
| `rdd2022_full`          | Large-scale road-damage data                  |
| `rdd2022`               | Existing development/reference subset         |
| `mwpd`                  | Weather/lighting/domain variation             |
| `qr4change`             | Healthy/Indian-road representation for DINOv2 |
| `pi5_smoketest_subset`  | Raspberry Pi preprocessing/inference testing  |

Important:

* RDD2022 bounding boxes are **not segmentation masks**.
* Pothole-600 depth/disparity is **ground truth/validation data**, not an RGB inference input.
* QR4Change should not be treated as a pothole-segmentation dataset.
* Water-filled data should be explicitly identified as such where annotations permit.
* Do not assume every dataset has severity labels.
* Do not fabricate temporal progression from these datasets.

---

# 6. Dataset-Aware Pipeline Design

The pipeline must not assume one dataset provides everything.

Use the datasets according to their actual annotations:

```text
SAM2 segmentation
    → pothole_mix
    → pothole_600
    → other compatible data if available

DINOv2 healthy-road representation
    → qr4change
    → healthy RDD images
    → other healthy/negative road images

Road-damage detection
    → rdd2022 / rdd2022_full

Water-aware evaluation
    → water_filled_potholes

Depth validation
    → pothole_600
    → CARLA ground truth

Weather/lighting robustness
    → mwpd

Raspberry Pi testing
    → pi5_smoketest_subset
```

Maintain separate evaluation paths rather than blindly merging incompatible annotations.

---

# 7. Audit Existing Pipeline

Before modifying anything, read:

```text
road_health_pipeline/README.md
road_health_pipeline/config.py

road_health_pipeline/inference/dinov2_embed.py
road_health_pipeline/inference/sam2_mask.py
road_health_pipeline/inference/anomaly_detector.py

road_health_pipeline/memory_bank/build_memory_bank.py
road_health_pipeline/memory_bank/coreset.py
road_health_pipeline/memory_bank/validate_memory_bank.py

road_health_pipeline/inference/pothole_localizer.py
road_health_pipeline/inference/area_estimator.py
road_health_pipeline/inference/depth_estimator.py
road_health_pipeline/inference/run_inference.py
```

Identify:

* broken imports
* hard-coded paths
* incorrect assumptions
* tensor-shape mismatches
* model-loading problems
* weak interfaces
* missing error handling
* incorrect DINOv2/SAM2 integration

Do not rewrite everything merely for style.

---

# 8. DINOv2

Primary file:

```text
road_health_pipeline/inference/dinov2_embed.py
```

Support:

* CUDA
* CPU fallback
* configurable model
* configurable preprocessing
* reusable model loading
* inference-only execution

Use patch tokens where appropriate.

Document:

* model
* architecture
* feature dimension
* patch size
* token count
* tensor shape
* normalization
* spatial correspondence

Patch-level features should be preserved for localized road anomaly detection.

---

# 9. DINOv2 Healthy-Road Memory Bank

Improve:

```text
memory_bank/build_memory_bank.py
memory_bank/coreset.py
memory_bank/validate_memory_bank.py
```

Desired flow:

```text
Healthy-road images
        ↓
DINOv2 patch features
        ↓
Normalization
        ↓
Feature pool
        ↓
Coreset selection
        ↓
Healthy-road memory bank
```

Use the existing 25 healthy China_Drone images only as a **prototype**.

The architecture should eventually support a larger, more diverse healthy-road collection including Indian roads, different asphalt appearances, weather and UAV viewpoints.

---

# 10. DINOv2 Anomaly Detection

Improve:

```text
inference/anomaly_detector.py
```

Desired flow:

```text
Query patch
    ↓
DINOv2 feature
    ↓
Nearest healthy feature(s)
    ↓
Distance
    ↓
Anomaly score
```

Prefer:

```text
patch scores
    ↓
2D spatial anomaly heatmap
```

Thresholds must be configurable.

Do not interpret every anomaly as a pothole. Possible anomalies include:

* road markings
* shadows
* repairs
* cracks
* stains
* debris
* puddles
* lighting changes

Maintain the distinction between:

```text
anomaly
road defect
pothole
```

---

# 11. Candidate Region Generation

Implement:

```text
DINOv2 anomaly map
        ↓
Upsampling
        ↓
Thresholding
        ↓
Connected regions
        ↓
Candidate box / points
        ↓
SAM2 prompt
```

The candidate represents a **potential road-damage region**, not automatically a pothole.

---

# 12. SAM2

Primary file:

```text
inference/sam2_mask.py
```

Desired flow:

```text
RGB image
    ↓
DINOv2 candidate
    ↓
Box / point prompt
    ↓
SAM2
    ↓
Defect mask
```

Investigate:

* box prompts
* positive points
* negative points
* combined prompts
* multiple masks
* confidence/stability
* area filtering
* background leakage
* mask ranking

Do not blindly select the first returned mask.

Return a structured result containing at least:

```text
mask
confidence
bbox
area
defect_type
```

---

# 13. DINOv2 + SAM2 Integration

This is the highest-priority part of the task.

Implement and verify:

```text
RGB IMAGE
   ↓
DINOv2
   ↓
Patch features
   ↓
Healthy memory bank
   ↓
Anomaly heatmap
   ↓
Candidate road-damage region
   ↓
SAM2 prompt
   ↓
SAM2 segmentation
   ↓
Defect mask
   ↓
Area estimation
   ↓
Depth interface
   ↓
Severity / road-health interface
```

The integration should work as one coherent pipeline rather than disconnected modules.

---

# 14. Road-Health / Severity Interface

Create a structured interface that can eventually combine:

```text
pothole count
pothole area
estimated depth
crack/damage extent
surface wear where measurable
water hazard
confidence
```

Example:

```json
{
  "road_health_score": null,
  "severity": null,
  "components": {
    "pothole_area": null,
    "depth": null,
    "damage_extent": null,
    "water_hazard": null
  }
}
```

Use `null` when measurements are unavailable.

Do not invent scores.

The current datasets may not provide every component needed for a scientifically complete road-health score. Design the interface now; full scoring comes after proper evaluation data is available.

---

# 15. Water-Aware Handling

Preserve, where supported:

```text
is_water_filled
water_confidence
water_related_risk
```

The water-filled dataset should be used to test whether dry and water-covered potholes are handled differently.

Do not hard-code a severity multiplier without an explicit project formulation.

---

# 16. Depth Estimation

Primary file:

```text
inference/depth_estimator.py
```

RoadSentinel operational inference remains **RGB-only**.

CARLA depth and Pothole-600 disparity/depth should be treated as ground truth/validation.

Clearly distinguish:

```text
RGB-estimated depth
```

from:

```text
ground-truth depth/disparity
```

Do not feed ground-truth depth into the normal RGB inference path.

---

# 17. Prediction Interface

RoadSentinel eventually aims to predict deterioration and pothole formation.

Create the interface now, but do not claim it is trained.

It should eventually accept:

```text
road segment
+
historical observations
+
defect history
+
time
+
severity evolution
```

Possible future output:

```json
{
  "deterioration_probability": null,
  "pothole_formation_probability": null,
  "prediction_horizon_days": null
}
```

The current listed datasets do **not automatically provide temporal progression**, so prediction should remain:

```text
REQUIRES TEMPORAL DATA
```

until appropriate data is obtained.

---

# 18. Development Fixtures

Use the existing development images plus synthetic fixtures where necessary.

Create:

```text
tests/
└── fixtures/
    ├── healthy_mock/
    ├── pothole_mock/
    ├── water_pothole_mock/
    └── deteriorated_road_mock/
```

Use these to test software flow, not to report real-world accuracy.

---

# 19. Automated Tests

### DINOv2

```text
image → embedding
image → patch tokens
```

Verify shape, dimensions and finite values.

### Memory Bank

```text
healthy images → memory bank
```

Verify non-empty output, expected shape and reproducibility.

### Anomaly Detection

Verify:

* anomaly-map dimensions
* numeric validity
* spatial correspondence
* configurable thresholds

### SAM2

Verify:

* prompt handling
* mask dimensions
* valid values
* confidence
* empty-result handling

### Road-health interface

Verify:

* missing values
* stable schema
* correct field types

### Prediction interface

Verify:

* temporal-record schema
* missing temporal data handling
* no fake predictions

### Full Integration

```text
image
 ↓
DINOv2
 ↓
anomaly
 ↓
candidate
 ↓
SAM2
 ↓
mask
 ↓
area
 ↓
depth interface
 ↓
severity interface
```

---

# 20. Full End-to-End Test

Actually execute the complete pipeline.

Do not only test imports.

Every component must be labelled:

```text
IMPLEMENTED
PLACEHOLDER
REQUIRES REAL DATA
REQUIRES TEMPORAL DATA
```

---

# 21. Output Artifacts

Generate:

```text
outputs/
├── original.jpg
├── dinov2_anomaly_heatmap.jpg
├── candidate_regions.jpg
├── sam2_mask.png
├── road_health_overlay.jpg
└── result.json
```

Example:

```json
{
  "image": "example.jpg",
  "detections": [
    {
      "defect_type": "unknown",
      "bbox": [x1, y1, x2, y2],
      "confidence": null,
      "anomaly_score": null,
      "mask_area_pixels": null,
      "depth_m": null,
      "is_water_filled": null,
      "severity": null
    }
  ],
  "road_health_score": null,
  "deterioration_probability": null,
  "pothole_formation_probability": null
}
```

Do not invent values.

---

# 22. Dataset Validation / Evaluation

When testing with the real datasets:

* respect their original annotation formats
* keep segmentation and bounding-box evaluation separate
* keep depth evaluation separate
* keep water-aware evaluation separate
* document UAV vs ground-level domain differences
* avoid train/test leakage
* preserve dataset-specific splits where provided

Do not merge incompatible labels into one training target without an explicit conversion strategy.

---

# 23. Performance Testing

Record:

```text
Device:
GPU:
CPU:
RAM:
VRAM:
DINOv2 load time:
DINOv2 inference time:
SAM2 load time:
SAM2 inference time:
Total latency:
Peak RAM:
Peak VRAM:
```

Test CUDA and CPU fallback where practical.

Identify components that are likely too expensive for Raspberry Pi 5.

---

# 24. Raspberry Pi 5

Use:

```text
pi5_smoketest_subset/
```

for smoke testing.

Focus on:

* image loading
* preprocessing
* postprocessing
* schema compatibility
* memory use
* CPU latency
* packaging

Do not prematurely redesign the main GPU pipeline for the Pi.

---

# 25. Configuration and Error Handling

Avoid machine-specific hard-coded paths.

Support:

```text
--image
--memory-bank
--device
--sam2-checkpoint
--dinov2-model
--output
--dataset
```

Provide clear errors/fallbacks for:

* missing images
* missing checkpoints
* invalid dimensions
* missing memory bank
* invalid configuration
* CUDA failure
* tensor mismatches
* invalid annotations
* unavailable depth ground truth
* unavailable temporal data

---

# 26. Scientific Validation

### DINOv2

Check:

* patch-token correctness
* normalization
* memory-bank construction
* distance metric
* spatial anomaly map
* threshold configuration
* healthy-data diversity
* UAV vs ground-level domain differences

### SAM2

Check:

* prompt generation
* candidate-box correctness
* mask selection
* confidence usage
* background leakage
* tiny/huge mask filtering

### Integration

Verify:

```text
DINOv2 → candidate generation → SAM2
```

works end-to-end.

Fix algorithmic problems, not just syntax errors.

---

# 27. Dataset Documentation

Update:

```text
datasets/
├── README.md
├── DATASET_SOURCES.md
└── DATASET_MATRIX.md
```

Document for each current dataset:

```text
source
license
annotation type
image domain
UAV/ground-level
segmentation availability
depth availability
water labels
severity labels
temporal information
intended RoadSentinel use
limitations
```

Do not claim temporal prediction capability from the current datasets unless verified.

---

# 28. Pipeline README

Update:

```text
road_health_pipeline/README.md
```

Document:

1. Installation
2. GPU requirements
3. DINOv2 setup
4. SAM2 setup
5. checkpoints
6. memory-bank creation
7. inference
8. tests
9. dataset evaluation
10. road-health interface
11. prediction interface
12. outputs
13. configuration
14. limitations

---

# 29. What You Must NOT Claim

Do not claim:

* high detection accuracy
* high segmentation IoU
* high recall
* reliable depth accuracy
* reliable severity prediction
* reliable road-health scoring
* reliable future pothole prediction

unless measured on suitable held-out data.

Report separately:

```text
DINOv2: PASS/FAIL
Patch tokens: PASS/FAIL
Memory bank: PASS/FAIL
Anomaly detection: PASS/FAIL
SAM2: PASS/FAIL
DINOv2 + SAM2: PASS/FAIL
Area estimation: PASS/FAIL
Depth interface: PASS/FAIL
Road-health interface: PASS/FAIL
Prediction interface: PASS/FAIL
End-to-end: PASS/FAIL
Real-data validation: PASS/FAIL/NOT AVAILABLE
Temporal validation: PASS/FAIL/NOT AVAILABLE
```

---

# 30. Git Safety

Before committing:

```bash
git status
git branch --show-current
```

The branch must be:

```text
marion-sam2-dinov2
```

Add only relevant code/documentation:

```bash
git add road_health_pipeline/
git add datasets/
```

Review:

```bash
git diff --cached
```

Do not commit:

```text
RoadSentinel_datasets/
.venv
env
model weights
large archives
API keys
passwords
tokens
massive generated outputs
```

Commit:

```bash
git commit -m "Expand RoadSentinel SAM2 DINOv2 pipeline"
```

Push only:

```bash
git push -u origin marion-sam2-dinov2
```

Never push or merge into `main`.

---

# 31. Final Handoff

Provide:

### Git

```text
Branch:
GitHub branch URL:
Commit hash:
git diff origin/main...HEAD --stat:
```

### Pipeline

```text
DINOv2:
Patch tokens:
Memory bank:
Coreset:
Anomaly detection:
Candidate generation:
SAM2:
DINOv2 + SAM2:
Area estimation:
Depth interface:
Road-health interface:
Prediction interface:
End-to-end:
```

### Performance

```text
Device:
GPU:
RAM:
VRAM:
DINOv2 latency:
SAM2 latency:
Total latency:
```

### Dataset compatibility

```text
Pothole Mix:
Pothole-600:
Water/Dry:
RDD2022:
RDD2022 China_Drone:
MWPD:
QR4Change:
Pi5 subset:
```

Use:

```text
TESTED
NOT TESTED
NOT AVAILABLE
NOT APPLICABLE
```

### Outputs

Provide paths for:

```text
anomaly heatmap
candidate visualization
SAM2 mask
road-health visualization
final JSON
```

Include representative outputs where possible.

### Limitations

Clearly identify:

* unavailable annotations
* components requiring real data
* camera-calibration requirements
* unvalidated depth
* unvalidated severity/health scoring
* unavailable temporal data
* unvalidated prediction
* mocked/placeholder components

---

# 32. Final Definition of Done

The SAM2 + DINOv2 portion is complete only when it is:

* modular
* configurable
* reproducible
* tested
* executable end-to-end
* documented
* compatible with the current multi-dataset ecosystem
* committed to a separate branch
* pushed to GitHub
* accompanied by actual test results and output artifacts

### Current target

```text
RGB image
   ↓
DINOv2
   ↓
Patch features
   ↓
Healthy memory bank
   ↓
Anomaly heatmap
   ↓
Candidate road-damage region
   ↓
SAM2
   ↓
Defect mask
   ↓
Area
   ↓
Depth interface
   ↓
Severity / road-health interface
```

### Future extension

```text
Survey t1
   ↓
Road-health state
   ↓
Survey t2
   ↓
Change estimation
   ↓
Temporal model
   ↓
Deterioration probability
   ↓
Pothole formation risk
```

The current datasets support strong development of the **identification/segmentation/scoring/depth-validation side**. A scientifically validated prediction system requires appropriate temporal/progression data later.

Do not merge into `main`.

Push only to:

```text
marion-sam2-dinov2
```

I will review the branch and merge accepted work later.

