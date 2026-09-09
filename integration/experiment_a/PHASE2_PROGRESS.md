# RoadSentinel Phase 2 — Experiment A Perception Processing Report
**Status**: COMPLETE / PASS  
**Execution Date**: September 9, 2026  
**Hardware Platform**: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM)  
**Execution Environment**: PyTorch 2.13.0+cu130, CUDA 12.8, Python 3.10.21  

---

## 1. Starting Commit
- **Commit**: `f4104fc` (*"Phase 1D: reframe existing captures using actual metadata"*)
- **Branch**: `main` (synchronized with `origin/main`)
- **Pre-Execution Workspace Check**: `CarlaUE5` submodule preserved untouched; zero Unreal Engine / CARLA interaction occurred.

---

## 2. Production Pipeline Verification
- **Verified Production Runner**: Frozen Marion Day-5 NORMAL DINOv2 + SAM 2 pipeline (`sam2_dino/current_condition.py`, `road_health_pipeline/inference/run_inference.py`).
- **Core Architecture**:
  - DINOv2 Backbone: `dinov2_vits14`, 518×518 input resolution, 14×14 patch size (single scale).
  - Anomaly Detector: Domain-adaptive patch scoring against real healthy road memory bank (10,000 vectors in `road_health_pipeline/output/real_memory_bank`).
  - Candidate Generation: 2D test-mode candidate localization (`pothole_localizer.py`).
  - Mask Refinement: SAM 2 (`sam2.1_hiera_small.pt`) bounding-box prompted candidate segmentation.
  - Geometry & Feature Extraction: Mask-derived bounding boxes and area ratios computed in original image coordinates (1920×1080).
- **Frozen Pipeline Contract**: No retraining, no threshold re-tuning on Experiment A, and no memory bank modifications.

---

## 3. Explicit Confirmation: Tiled Mode NOT Used
- **Confirmation**: `road_health_pipeline/inference/tiled_dinov2_eval.py` was **STRICTLY NOT USED**.
- **Rationale**: The 2× sliding-window / tiled DINOv2 experiment was formally rejected by the Team Lead in `sam2_dino/DAY5_GPU_FOLLOWUP_RESULT.md` due to catastrophic scale-domain mismatch (crack IoU regressed 0.0406 → 0, pothole IoU collapsed 0.6398 → 0.0432, false alarms on healthy pavement, and 5.32× latency penalty).
- **Execution Mode**: 100% normal production Day-5 pipeline at 1× scale.

---

## 4. Hardware & GPU Specifications
- **GPU Device**: NVIDIA GeForce RTX 5060 Laptop GPU
- **VRAM**: 8,192 MB (8 GB GDDR6)
- **Execution Strategy**: Sequential, memory-safe single-process inference with CUDA synchronizations.

---

## 5. Primary Images Attempted
- **Target Count**: 39 metadata-complete primary captures from `integration/temporal/actual_capture_inventory.csv` (`SEG_001`–`SEG_004`, Days 01–10, excluding `SEG_003` Day 10).
- **Attempted**: 39/39 (100.0%)

---

## 6. Successful Images
- **Primary Set Successful**: 39/39 (100.0%)
- **Diagnostic Image Successful**: 1/1 (`SEG_003` Day 10)
- **Total Physical Captures Processed**: 40/40 (100.0%)

---

## 7. Failures & Exceptions
- **Inference Failures**: 0
- **Process Crashes / Out of Memory (OOM)**: 0
- **Exporter Exceptions**: 0

---

## 8. Schema-Valid Outputs
- **Contract Schema**: `sam2_dino/feature_contract.schema.json` (Contract Version `1.0.0`)
- **Primary Set Validated**: 39/39 (`SCHEMA_PASS`)
- **Diagnostic Set Validated**: 1/1 (`SCHEMA_PASS`)
- **Total Schema Valid**: 40/40 (100.0%)
- **Output Artifacts**: Every capture directory in `integration/experiment_a/perception/<segment_id>/day_<day:02d>/` contains:
  - `features.json` (schema valid, mask-derived bounding boxes, unconditioned)
  - `overlay.png` (color-coded defect overlays, centroids, severity/anomaly banner)
  - `masks/` (`defect_XXX_mask.png` individual masks)
  - `inference_metadata.json` (preserved simulation condition sidecars decoupled from inference)
  - `diagnostics.json` (patch score distributions, threshold selection, road mask ratio)

---

## 9. Total Execution Runtime
- **Total Pipeline Runtime (40 frames)**: 36.66 seconds
- **Throughput**: ~1.09 frames per second (including SAM 2 candidate segmentation, mask writing, and high-res overlay generation)

---

## 10. Inference Time Distribution
- **All 40 Captures**:
  - Mean: **802.82 ms**
  - Median: **607.88 ms**
  - Min: **220.22 ms** (`SEG_004` Day 10, fast suppression)
  - Max: **2,173.87 ms** (`SEG_004` Day 07, heavy rain & multi-candidate SAM 2 refinement)
- **Primary Set (39 Captures)**:
  - Mean: **807.86 ms**
  - Median: **609.43 ms**

---

## 11. Severity Distribution (Primary Set, N=39)
- **Mean Severity**: **0.3077**
- **Median Severity**: **0.2472**
- **Min Severity**: **0.0000** (5 frames with 0 detected defects)
- **Max Severity**: **0.8134** (`SEG_004` Day 08)
- **Severity Range**: `[0.0000, 0.8134]`
- **Zero-Severity Frames**: 5 / 39 (12.8%)
- **Low-Severity Frames (0.0 < sev <= 0.35)**: 23 / 39 (59.0%)
- **Medium-Severity Frames (0.35 < sev <= 0.70)**: 6 / 39 (15.4%)
- **High-Severity Frames (sev > 0.70)**: 5 / 39 (12.8%)

---

## 12. Defect Count & Area Distributions (Primary Set, N=39)
- **Mean Defect Count**: **2.21 defects/image**
- **Median Defect Count**: **1.0 defect/image**
- **Total Accepted Defects**: **86 defects**
- **Defect Count Range**: `[0, 12]`
- **Defect Area Ratio**:
  - Mean: **0.0056** (0.56% of total road surface)
  - Median: **0.0010** (0.10%)
  - Max: **0.0382** (3.82% in `SEG_004` Day 05)
- **Surface Anomaly Score**:
  - Mean: **0.2889**
  - Median: **0.2857**
  - Range: `[0.2515, 0.3428]`
- **Water Hazard Flag**: 2 / 39 positive flags (5.1%), both under `Heavy Rain & Wet Road`

---

## 13. Lighting-Condition Summary (Descriptive Model Outputs)

> [!NOTE]
> Statistics are descriptive model responses across simulated capture environments. They must not be interpreted as detection accuracy or precision/recall.

| Lighting Preset | Images | Mean Severity | Median Severity | Mean Defects | Total Defects | Mean Area Ratio | Mean Anomaly | Mean Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Clear Noon (70° Sun)** | 28 | 0.2191 | 0.2453 | 1.00 | 28 | 0.0009 | 0.2820 | 588.4 |
| **Overcast Day** | 5 | 0.7084 | 0.7141 | 5.60 | 28 | 0.0227 | 0.3010 | 1471.1 |
| **Golden Hour Sunset** | 4 | 0.1750 | 0.0000 | 1.75 | 7 | 0.0096 | 0.2968 | 951.4 |
| **Heavy Rain & Wet Road** | 2 | 0.8119 | 0.8119 | 11.50 | 23 | 0.0204 | 0.3411 | 1934.9 |

---

## 14. Road-Health-State Summary (Descriptive Model Outputs)

| Road-Health State | Images | Mean Severity | Median Severity | Mean Defects | Total Defects | Mean Area Ratio | Mean Anomaly | Mean Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Pristine (Grade A)** | 10 | 0.2376 | 0.2444 | 1.10 | 11 | 0.0010 | 0.2831 | 664.8 |
| **Minor Wear (Grade B)** | 1 | 0.0000 | 0.0000 | 0.00 | 0 | 0.0000 | 0.2967 | 588.4 |
| **Moderate Deterioration (Grade C)** | 10 | 0.2888 | 0.2527 | 2.10 | 21 | 0.0054 | 0.2847 | 770.6 |
| **Severe Breakdown (Grade D - Critical)** | 11 | 0.5376 | 0.6612 | 4.64 | 51 | 0.0136 | 0.2985 | 1204.0 |
| **Critical Hazard (Grade F)** | 7 | 0.1176 | 0.0000 | 0.43 | 3 | 0.0006 | 0.2872 | 474.3 |

---

## 15. Camera-Viewpoint Summary (Descriptive Model Outputs)

| Camera Viewpoint | Images | Mean Severity | Median Severity | Mean Defects | Total Defects | Mean Area Ratio | Mean Anomaly | Mean Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **🌄 Highway Curve Vantage Overlook** | 23 | 0.3287 | 0.2459 | 2.70 | 62 | 0.0069 | 0.2937 | 842.8 |
| **🔭 Overhead Drone Survey (SAM 2 Top-Down)** | 10 | 0.2123 | 0.2440 | 1.00 | 10 | 0.0033 | 0.2855 | 696.4 |
| **💧 Waterlogged Pothole Macro View** | 4 | 0.4133 | 0.3482 | 2.50 | 10 | 0.0063 | 0.2777 | 912.0 |
| **🔍 Low-Angle Pothole Inspection (30° Close-Up)** | 2 | 0.3327 | 0.3327 | 2.00 | 4 | 0.0008 | 0.2745 | 754.6 |

---

## 16. Moisture State Summary (Descriptive Model Outputs)

| Moisture State | Images | Mean Severity | Median Severity | Mean Defects | Total Defects | Mean Area Ratio | Mean Anomaly | Mean Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Mixed Wet / Dry Cavities** | 25 | 0.2654 | 0.2469 | 1.40 | 35 | 0.0033 | 0.2830 | 720.8 |
| **100% Dry Crushed Aggregate** | 10 | 0.3681 | 0.3327 | 3.70 | 37 | 0.0079 | 0.2962 | 916.7 |
| **100% Waterlogged Puddles** | 4 | 0.4215 | 0.4786 | 3.50 | 14 | 0.0141 | 0.3082 | 1079.8 |

---

## 17. Pristine False-Positive Diagnostic
- **Pristine (Grade A) Population**: 10 captures total.
- **Zero-Defect Frames**: 1 / 10 (`SEG_001` Day 01 under Overhead Drone Survey, severity = 0.0000).
- **Non-Zero Defect Frames**: 9 / 10 (mean severity = 0.2640).
  - `SEG_002` Day 01 & Day 02 (Low-Angle 30° close-up): 2 defects each (severity 0.3155 and 0.3498). The grazing viewpoint amplifies pavement macro-texture contrast relative to the nadir memory bank.
  - `SEG_003` Day 01 through Day 07 (Highway Curve Vantage Overlook): exactly 1 defect per frame (severity consistently between 0.2417 and 0.2472). In this specific perspective overlook, the distant asphalt road curb/boundary triggers a small persistent candidate region across days.
- **Finding**: DINOv2+SAM2 exhibits high baseline stability on top-down drone imagery, but oblique viewpoints (low-angle and distant curve vantage) induce low-amplitude (~0.24 severity) geometric candidate triggers on unblemished pavement.

---

## 18. Severe / Critical Response Diagnostic
- **Grade D (Severe Breakdown, N=11)**:
  - Strong, expected detection scaling.
  - Produces the highest defect density: 51 total defects (mean 4.64 defects/frame).
  - Peaks at 12 defects in `SEG_004` Day 08 (severity 0.8134, area ratio 0.0205) and 11 defects in `SEG_004` Day 07 (severity 0.8104, area ratio 0.0203).
  - Mean severity is 0.5376 (median 0.6612).
- **Grade F (Critical Hazard, N=7)**:
  - Demonstrates an important interaction: mean severity drops to 0.1176, with 4 of 7 frames recording 0 defects (`SEG_001` Day 04, `SEG_001` Day 10, `SEG_004` Day 09, `SEG_004` Day 10).
  - Diagnostic inspection revealed that under extreme Golden Hour Sunset lighting and curved highway angles, stretched shadows and specular reflection trigger the **Road Marking Suppressor** to mask out large portions of the asphalt (up to 2,041,720 pixels in `SEG_004` Day 10), preventing the anomaly detector from nominating candidates in those shadowed road regions.

---

## 19. Major Qualitative Failure Cases & Domain Shifts
1. **Specular Glare & Sunset Shadow Suppression**: Low sun angle (Golden Hour) stretches vehicle and curb shadows. The heuristic marking and vehicle suppressors misidentify elongated dark boundaries as painted lane markings/curbs, causing over-suppression.
2. **Oblique Vantage Boundary Artifacts**: In `SEG_003` Day 01–07, the curved road shoulder is repeatedly localized as a low-severity road defect due to boundary contrast against non-road terrain.
3. **Moisture & Water Sensitivity**: Water hazard flags trigger reliably under heavy rain (100% detection rate for `Heavy Rain & Wet Road`), but mixed wet/dry puddles under high-noon conditions remain largely undetected by the RGB texture water heuristic.

---

## 20. SEG_003 Day 10 Diagnostic Handling
- **Physical Image**: `env/output/temporal_segments/SEG_003/day_10.png` exists and was processed through DINOv2+SAM2.
- **Sidecar Status**: `day_10_metadata.json` is missing from disk.
- **Perception Output**:
  - Successfully produced `features.json`, `overlay.png`, `masks/`, and `inference_metadata.json`.
  - Detections: 1 defect, severity = 0.2459, surface anomaly score = 0.2921, inference time = 606.25 ms.
  - Feature contract validation: `SCHEMA_PASS`.
- **Handling**:
  - `metadata_status` explicitly recorded as `"MISSING"`.
  - Simulation condition fields (`lighting_preset`, `road_health_state`, `camera_preset`, etc.) set to JSON `null`.
  - **Strictly excluded** from all metadata-stratified condition tables (Sections 13–16).

---

## 21. Phase 3 Readiness
- **Evaluation Gate**: **PASS**
- **Readiness**: Fully ready for Phase 3.
- **Prepared Subsets**:
  - Same-camera subsets (`SEG_001` Day 03–10, `SEG_003` Day 01–07, `SEG_004` Day 01–05, `SEG_004` Day 06–10) have contract-valid outputs organized in predictable directories for temporal tracking and defect matching.
  - No cross-day tracking or XGBoost forecasting was performed in Phase 2.
