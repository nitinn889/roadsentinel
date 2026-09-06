# RoadSentinel — SAM2 + DINOv2 Development Task

## Purpose

This document contains the complete development instructions for continuing the RoadSentinel project.

The task is specifically focused on the **SAM2 + DINOv2 pipeline**.

You are working from the RoadSentinel pipeline that you originally provided. The pipeline has since been reorganized and adapted into the current RoadSentinel repository.

### Critical instructions

- **Do not commit or push anything to `main`.**
- Work only on your own branch.
- I will review your branch and decide what should be merged into `main`.
- You do **not** need access to the final RoadSentinel datasets yet.
- Use the existing small development images, synthetic/mock images, or generated test fixtures where necessary.
- The goal is to make the **SAM2 + DINOv2 pipeline executable, modular, testable, and ready for proper dataset evaluation later.**
- Do not make unsupported claims about accuracy without real evaluation data.

---

# 1. RoadSentinel Repository

GitHub repository:

https://github.com/nitinn889/roadsentinel

Clone it:

```bash
git clone https://github.com/nitinn889/roadsentinel.git
cd roadsentinel
```

If you already have the repository:

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

The repository should be connected to:

```text
https://github.com/nitinn889/roadsentinel.git
```

---

# 2. Create Your Development Branch

Do **not** modify `main` directly.

Create:

```bash
git checkout -b marion-sam2-dinov2
```

Verify:

```bash
git branch --show-current
```

Expected:

```text
marion-sam2-dinov2
```

All development must happen on this branch.

Do not run:

```bash
git push origin main
```

Do not merge the branch into `main`.

I will review the branch and handle the merge.

---

# 3. Current RoadSentinel Pipeline Structure

The current pipeline has been organized as:

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

# 4. What I Changed From Your Original Files

The original RoadSentinel implementation that you provided was not discarded. I reorganized and adapted it into the current repository structure.

## 4.1 Modularized the pipeline

The functionality was separated into:

```text
carla_sim/
common/
evaluation/
inference/
memory_bank/
pi_edge/
```

This makes the system easier to maintain and eventually deploy across the development machine, CARLA simulation and Raspberry Pi.

---

## 4.2 Organized the inference modules

The inference package now contains:

```text
inference/
├── anomaly_detector.py
├── area_estimator.py
├── depth_estimator.py
├── dinov2_embed.py
├── gps_localizer.py
├── pothole_localizer.py
├── run_inference.py
├── sam2_mask.py
└── server.py
```

The intended architecture is:

```text
RGB image
    ↓
DINOv2 feature extraction
    ↓
Healthy-road comparison / anomaly detection
    ↓
Candidate pothole region
    ↓
SAM2 segmentation
    ↓
Pothole mask
    ↓
Area estimation
    ↓
Depth estimation
    ↓
Severity interface
    ↓
GPS localization
    ↓
Final RoadSentinel output
```

---

## 4.3 Organized the DINOv2 memory-bank implementation

The memory-bank components are:

```text
memory_bank/
├── build_memory_bank.py
├── coreset.py
└── validate_memory_bank.py
```

The intended logic is:

```text
Healthy-road images
        ↓
DINOv2 features
        ↓
Patch-level representations
        ↓
Feature pool
        ↓
Coreset selection
        ↓
Healthy-road memory bank
        ↓
Query image
        ↓
DINOv2 features
        ↓
Distance/similarity to memory bank
        ↓
Anomaly score
```

A previous development experiment used 25 healthy China_Drone images for an initial memory-bank test.

That is only a prototype and must not be considered the final training/evaluation setup.

---

## 4.4 Organized the CARLA components

CARLA-related code is now under:

```text
carla_sim/
├── drone_controller.py
└── rgb_depth_capture.py
```

The eventual intended architecture is:

```text
CARLA
 ↓
Simulated drone
 ↓
RGB camera
 ↓
RoadSentinel inference pipeline
```

Important:

The simulated depth camera should be treated as **ground truth for evaluation**, not as an input to the real RoadSentinel RGB-only inference pipeline.

---

## 4.5 Added Raspberry Pi edge modules

The current edge structure is:

```text
pi_edge/
├── edge_processor.py
├── telemetry.py
└── uploader.py
```

These are intended for later Raspberry Pi 5 deployment.

Do not spend most of this task optimizing Raspberry Pi execution. The immediate priority is making the **DINOv2 + SAM2 pipeline correct and testable**.

---

# 5. First Task — Audit the Existing Pipeline

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

Understand the current data flow before making changes.

Identify:

- broken imports
- hard-coded paths
- incorrect assumptions
- tensor-shape mismatches
- model-loading problems
- unused code
- weak interfaces between modules
- missing error handling
- places where DINOv2 and SAM2 are not properly connected

Do not rewrite everything just for style.

Make changes where they improve correctness, robustness, modularity or testability.

---

# 6. Main Task — Improve DINOv2

Primary file:

```text
road_health_pipeline/inference/dinov2_embed.py
```

## 6.1 Robust model loading

The implementation should support:

- CUDA
- CPU fallback
- configurable DINOv2 model
- inference-only execution
- configurable device
- configurable preprocessing
- model reuse across multiple images

Avoid loading the model separately for every image.

Desired pattern:

```text
Application starts
        ↓
Load DINOv2 once
        ↓
Process many images
        ↓
Reuse model
```

---

# 7. DINOv2 Patch Tokens

RoadSentinel should not depend only on a global image embedding.

Investigate and use DINOv2 patch tokens where appropriate.

Desired conceptual representation:

```text
RGB image
   ↓
DINOv2
   ↓
patch tokens
   ↓
spatial feature grid
```

Maintain spatial correspondence between patch tokens and image coordinates.

Document:

- DINOv2 model name
- model architecture
- feature dimension
- patch size
- number of patch tokens
- tensor shape
- normalization
- preprocessing

For a patch-based ViT setup, explicitly document how token indices correspond to image regions.

---

# 8. DINOv2 Healthy-Road Memory Bank

Improve:

```text
road_health_pipeline/memory_bank/build_memory_bank.py
road_health_pipeline/memory_bank/coreset.py
road_health_pipeline/memory_bank/validate_memory_bank.py
```

The memory bank should represent healthy road appearance.

Desired pipeline:

```text
Healthy-road images
        ↓
DINOv2 patch features
        ↓
Feature normalization
        ↓
Feature pool
        ↓
Coreset selection
        ↓
Compact memory bank
```

The implementation should be:

- reproducible
- deterministic where appropriate
- configurable
- memory-efficient
- independent of hard-coded local paths

Do not assume that 25 images are enough for the final system.

---

# 9. DINOv2 Anomaly Detection

Improve:

```text
road_health_pipeline/inference/anomaly_detector.py
```

The query image should be compared against the healthy-road memory bank.

Investigate appropriate distance/similarity metrics.

Desired logic:

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

Prefer a spatial anomaly map when patch tokens are available:

```text
RGB image
    ↓
DINOv2 patch grid
    ↓
Anomaly score per patch
    ↓
2D anomaly heatmap
```

Thresholds must not be hard-coded without explanation.

Make them configurable.

---

# 10. Anomaly Map → Candidate Region

Investigate how the DINOv2 anomaly map can generate candidate pothole regions.

Desired pipeline:

```text
DINOv2 patch anomaly map
        ↓
Upsampling
        ↓
Thresholding
        ↓
Connected regions
        ↓
Candidate bounding box / points
        ↓
SAM2 prompt
```

Important:

Do not treat every anomaly as a pothole.

RoadSentinel can encounter anomalies such as:

- road markings
- shadows
- repaired asphalt
- cracks
- stains
- debris
- lighting changes

The code should distinguish:

```text
anomaly candidate
```

from:

```text
confirmed pothole
```

SAM2 should be used as part of the localization/segmentation process.

---

# 11. Main Task — Improve SAM2

Primary file:

```text
road_health_pipeline/inference/sam2_mask.py
```

The SAM2 component should receive a candidate region and produce a segmentation mask.

Desired conceptual flow:

```text
RGB image
    ↓
DINOv2 anomaly candidate
    ↓
Candidate box / point
    ↓
SAM2
    ↓
Pothole mask
```

---

# 12. SAM2 Prompting

Implement/support appropriate SAM2 prompting.

Investigate:

- bounding-box prompting
- positive-point prompting
- negative-point prompting
- combined point + box prompts where useful

The initial preferred approach is:

```text
DINOv2 anomaly region
        ↓
Candidate bounding box
        ↓
SAM2 box prompt
        ↓
Pothole segmentation
```

However, test whether point prompts or combinations improve segmentation quality.

Document the reasoning.

---

# 13. SAM2 Mask Selection

Do not blindly return the first SAM2 mask.

Investigate:

- multiple candidate masks
- confidence scores
- mask stability
- area filtering
- connected components
- background leakage
- candidate ranking

The SAM2 module should ideally return a structured result such as:

```text
mask
confidence
bounding box
area
```

Use a clear schema instead of arbitrary dictionaries scattered through the pipeline.

---

# 14. DINOv2 + SAM2 Integration

This is the most important part of the task.

DINOv2 and SAM2 should be integrated into one coherent pipeline.

Target architecture:

```text
                 ┌──────────────┐
                 │   RGB IMAGE  │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │    DINOv2    │
                 └──────┬───────┘
                        ↓
                Patch-level features
                        ↓
                 Healthy memory bank
                        ↓
                   Anomaly map
                        ↓
                 Candidate regions
                        ↓
                  Box / point prompt
                        ↓
                 ┌──────────────┐
                 │     SAM2     │
                 └──────┬───────┘
                        ↓
                 Pothole segmentation
                        ↓
            ┌───────────┴───────────┐
            ↓                       ↓
      Area estimation         Depth estimation
            ↓                       ↓
            └───────────┬───────────┘
                        ↓
                  Severity interface
```

The code should support this complete path.

---

# 15. No Final Dataset Required Yet

You currently do not need the complete public datasets.

Create or use synthetic/mock road images to test the pipeline.

Create:

```text
tests/
└── fixtures/
    ├── healthy_mock/
    └── pothole_mock/
```

Synthetic images can contain:

- road-like backgrounds
- simulated potholes
- dark irregular regions
- shadows
- repaired road regions
- road markings
- texture variation

The purpose is not to report model accuracy.

The purpose is to verify that the full software pipeline executes correctly.

---

# 16. Automated Tests

Create tests for the major components.

## DINOv2

Test:

```text
image → embedding
image → patch tokens
```

Verify:

- correct tensor shapes
- expected dimensionality
- finite values
- no NaN/Inf

## Memory Bank

Test:

```text
healthy images
    ↓
memory bank
```

Verify:

- non-empty memory bank
- expected shape
- reproducibility
- no NaN/Inf

## Anomaly Detection

Test:

```text
query features
    ↓
anomaly scores
```

Verify:

- expected shape
- valid numeric output
- spatial correspondence

## SAM2

Test:

```text
image + prompt
    ↓
mask
```

Verify:

- mask dimensions
- valid mask values
- confidence availability
- no unexpected empty result

## Full Integration

Test:

```text
image
 ↓
DINOv2
 ↓
anomaly map
 ↓
candidate region
 ↓
SAM2
 ↓
mask
```

---

# 17. Full End-to-End Pipeline Test

This is REQUIRED.

The pipeline must actually run from image input to final output.

At minimum:

```text
Input RGB image
      ↓
DINOv2
      ↓
Feature extraction
      ↓
Healthy memory-bank comparison
      ↓
Anomaly detection
      ↓
Candidate localization
      ↓
SAM2
      ↓
Segmentation mask
      ↓
Area estimation
      ↓
Depth-estimation interface
      ↓
Final structured result
```

Do not only test imports.

Actually execute the pipeline.

If a component cannot yet provide scientifically valid values because final data/calibration is unavailable, it should still execute using clearly documented placeholder/mock values.

Every component should be labelled as one of:

```text
IMPLEMENTED
```

```text
PLACEHOLDER
```

```text
REQUIRES REAL DATA
```

Do not hide missing functionality.

---

# 18. Output Artifacts

The test should generate useful outputs such as:

```text
outputs/
├── original.jpg
├── dinov2_anomaly_heatmap.jpg
├── candidate_regions.jpg
├── sam2_mask.png
└── result.json
```

The JSON should follow a stable schema.

Example:

```json
{
  "image": "example.jpg",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0,
      "anomaly_score": 0.0,
      "mask_area_pixels": 0,
      "depth_m": null,
      "severity": null
    }
  ]
}
```

Do not invent depth or severity values.

Use `null` when they cannot currently be calculated reliably.

---

# 19. Performance Testing

Run the complete pipeline on multiple test images.

Record:

```text
Device:
GPU:
CPU:
GPU VRAM:
RAM:
DINOv2 model loading time:
DINOv2 inference time:
SAM2 model loading time:
SAM2 inference time:
Total image latency:
Peak RAM:
Peak VRAM:
```

If CUDA is available, test CUDA.

If practical, test CPU fallback as well.

Identify which components are likely to be too computationally expensive for Raspberry Pi deployment.

Do not prematurely force large models onto the Pi.

---

# 20. Configuration and Portability

Avoid hard-coded paths such as:

```text
/home/nitin-nandakumar/...
```

Use:

```text
config.py
```

and/or command-line parameters.

Support parameters such as:

```text
--image
--memory-bank
--device
--sam2-checkpoint
--dinov2-model
--output
```

The pipeline should be portable to another machine.

---

# 21. Error Handling

Add clear errors for:

- missing images
- missing model checkpoints
- invalid image dimensions
- unavailable memory bank
- missing configuration
- unsupported device
- CUDA unavailable
- incorrect tensor shapes

Where appropriate, automatically fall back to CPU.

Avoid unnecessarily obscure traceback-only failures.

---

# 22. Scientific/Algorithmic Validation

Do not only make the code execute.

Review whether the algorithm makes sense.

## DINOv2

Check:

- Are patch tokens being extracted correctly?
- Are embeddings normalized?
- Is the memory bank built at the correct feature level?
- Is the distance metric appropriate?
- Is the anomaly map spatially meaningful?
- Are thresholds configurable?
- Are healthy-road examples representative?

## SAM2

Check:

- Are prompts generated correctly?
- Is the candidate box correct?
- Are masks selected correctly?
- Are confidence scores used?
- Is background leakage controlled?
- Are obviously tiny/huge masks filtered?

## Integration

Check whether:

```text
DINOv2 → candidate detection → SAM2
```

actually works end-to-end.

If you identify a flaw in the existing implementation, fix it and document why.

---

# 23. Update the Pipeline README

Update:

```text
road_health_pipeline/README.md
```

Document:

1. Installation
2. GPU requirements
3. DINOv2 model setup
4. SAM2 model setup
5. Model checkpoint requirements
6. Memory-bank creation
7. Running inference
8. Running tests
9. Expected outputs
10. Configuration options
11. Known limitations

Do not put:

- datasets
- model checkpoints
- API keys
- credentials
- huge generated artifacts

into GitHub.

---

# 24. What You Must NOT Claim

Because final datasets are not available yet, do not claim:

- high detection accuracy
- high segmentation IoU
- high recall
- reliable depth accuracy
- reliable severity prediction

unless these are actually measured on appropriate test data.

For now, report:

```text
Pipeline execution: PASS/FAIL
DINOv2: PASS/FAIL
Memory bank: PASS/FAIL
Anomaly detection: PASS/FAIL
SAM2: PASS/FAIL
DINOv2 + SAM2 integration: PASS/FAIL
End-to-end pipeline: PASS/FAIL
Real-data validation: NOT AVAILABLE
```

---

# 25. Git Safety Rules

Before committing:

```bash
git status
git branch --show-current
```

The branch must be:

```text
marion-sam2-dinov2
```

Add only the relevant project files:

```bash
git add road_health_pipeline/
```

Review:

```bash
git diff --cached
```

Do not commit:

```text
datasets
.venv
env
model weights
large archives
API keys
passwords
tokens
generated massive outputs
```

Commit:

```bash
git commit -m "Improve SAM2 and DINOv2 RoadSentinel pipeline"
```

Push ONLY your branch:

```bash
git push -u origin marion-sam2-dinov2
```

Do not push to `main`.

Do not merge the branch.

---

# 26. Final Handoff Requirements

When you finish, provide me with all of the following.

## Git branch

```text
marion-sam2-dinov2
```

## GitHub branch URL

Provide the actual branch URL.

## Commit hash

Run:

```bash
git rev-parse HEAD
```

and report the result.

## File change summary

Run:

```bash
git diff origin/main...HEAD --stat
```

and provide the result.

## Pipeline test results

Report:

```text
DINOv2: PASS/FAIL
Patch tokens: PASS/FAIL
Memory bank: PASS/FAIL
Coreset: PASS/FAIL
Anomaly detection: PASS/FAIL
Candidate generation: PASS/FAIL
SAM2: PASS/FAIL
DINOv2 + SAM2 integration: PASS/FAIL
End-to-end pipeline: PASS/FAIL
```

## Performance results

Report:

```text
Device:
GPU:
RAM:
VRAM:
DINOv2 latency:
SAM2 latency:
Total latency:
```

## Output files

Provide paths for:

```text
anomaly heatmap
candidate visualization
SAM2 mask
final JSON result
```

Include representative output images where possible.

## Known limitations

Clearly identify:

- missing datasets
- components requiring real data
- components requiring camera calibration
- unvalidated depth estimation
- unvalidated severity estimation
- anything currently mocked or placeholder

---

# 27. Final Definition of Done

This task is complete only when the SAM2 + DINOv2 portion of RoadSentinel is:

- modular
- configurable
- reproducible
- tested
- executable end-to-end
- documented
- integrated rather than being a collection of disconnected modules
- committed to a separate Git branch
- pushed to GitHub
- accompanied by actual test results and output artifacts

The desired final pipeline is:

```text
RGB image
   ↓
DINOv2
   ↓
Patch features
   ↓
Healthy-road memory bank
   ↓
Anomaly heatmap
   ↓
Candidate pothole region
   ↓
SAM2 prompt
   ↓
SAM2 segmentation
   ↓
Pothole mask
   ↓
Area estimation
   ↓
Depth-estimation interface
   ↓
Severity-estimation interface
   ↓
Structured RoadSentinel result
```

Do not merge into `main`.

Push only to:

```text
marion-sam2-dinov2
```

I will review the branch and merge the accepted work later.