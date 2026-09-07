# RoadSentinel — YOLO Day 1 Corrective Progress Report

Date: 2026-09-07  
Role: Team Lead  
Component: Road-Defect Detection (YOLO Baseline)  
Status: **VERIFIED BASELINE OPERATIONAL**

---

## 1. Local Datasets Found

A filesystem audit across `RoadSentinel_datasets/`, `datasets/`, `ml/data/`, and `~/Downloads/` identified seven local road defect datasets:
1. **RDD2022_full (China_Drone)**: 2,401 512×512 UAV nadir images with PASCAL VOC XML annotations.
2. **RDD2022_full (India)**: 9,665 vehicle dashcam images (7,706 with XML annotations, 1,959 unannotated test images).
3. **MWPD (Multi-Weather Pothole Detection)**: 2,752 images with YOLO `.txt` labels across train/valid/test splits.
4. **Annotated Water-Filled and Dry Potholes**: 713 images with both Pascal VOC XML and YOLO TXT annotations.
5. **Pothole-600**: 600 stereo RGB image pairs with PNG pixel segmentation masks and disparity maps.
6. **Pothole Mix (SHREC2022)**: 4,340 image/mask pairs for semantic segmentation.
7. **QR4Change**: 2,966 smartphone images (image-level binary classification only).

Full machine-readable inventory recorded at: `yolo/local_dataset_inventory.json`.

---

## 2. Exact Dataset Selected

- **Selected**: **RDD2022 (China_Drone subset)** (`RoadSentinel_datasets/rdd2022_full/China_Drone/China_Drone`).
- **Rationale**: Direct UAV/drone perspective matching RoadSentinel's aerial inspection mission, uniform 512×512 resolution, and multi-defect coverage (cracks, potholes, patches).

---

## 3. Internet Download Status

- **Required**: **NO**.
- Zero external datasets were downloaded. All training and validation data were sourced from local pre-existing checkouts.

---

## 4. Annotation Format & Conversion

- **Source Format**: PASCAL VOC XML bounding boxes (`<xmin>`, `<ymin>`, `<xmax>`, `<ymax>`).
- **Target Format**: Normalized YOLO text format (`<class_id> <x_center> <y_center> <width> <height>`).
- **Converter**: Deterministic Python conversion script: `yolo/src/prepare_rdd2022_yolo.py`. Image files are symlinked to prevent redundant disk usage.

---

## 5. Verified Class Mapping

Preserved from official RDD2022 taxonomy (`yolo/configs/class_mapping.json`):
- `0`: `D00` — Longitudinal Crack
- `1`: `D10` — Transverse Crack
- `2`: `D20` — Alligator (Fatigue) Crack
- `3`: `D40` — Pothole
- `4`: `Repair` — Repaired / Patched Road Surface

Rare/non-standard tags (`Block crack`, `D43`, `D44`, etc.) are filtered out.

---

## 6. Dataset Split & Leakage Prevention

- **Split Strategy**: Seeded deterministic 80/20 split (`seed=42`).
- **Total Images**: 2,401
  - **Train Set**: 1,921 images
  - **Validation Set**: 480 images
- **Anti-Leakage Guarantee**:
  - `China_Drone_001267.jpg` (used as the fixed RDD test benchmark in Marion Days 2–3) was strictly reserved in the validation set and excluded from training.
- **Instance Counts**:
  - Train: 1,160 D00, 1,007 D10, 235 D20, 71 D40, 622 Repair
  - Val: 266 D00, 256 D10, 58 D20, 15 D40, 147 Repair

---

## 7. Model Selection & Environment

- **Ultralytics Version**: `8.4.143`
- **Starting Pretrained Checkpoint**: `yolov8n.pt` (6.5 MB, COCO 80-class pretrained initialization).
- **Distinction**:
  - Pretrained initialization: `yolov8n.pt` (COCO objects: person, car, etc.)
  - Fine-tuned weights: `yolo/weights/best.pt` (Road defects: D00, D10, D20, D40, Repair).

---

## 8. Training Execution

- **Script**: `yolo/src/train_yolo.py`
- **Command**:
  ```bash
  ./.venv/bin/python yolo/src/train_yolo.py --epochs 3 --batch 32 --device 0
  ```
- **Configuration**:
  - Epochs: 3
  - Image size: 512
  - Batch size: 32
  - Device: `CUDA:0` (NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM)
  - Wall Duration: **30.33 seconds**
- **Status**: **COMPLETED**.

---

## 9. Local Weight Artifacts

- **Fine-Tuned Best Weights**: `yolo/weights/best.pt` (6,227,562 bytes)
- **Last Epoch Weights**: `yolo/weights/last.pt` (6,227,562 bytes)
- **Weight Verification**:
  Loaded with `ultralytics.YOLO`. Confirmed active class dictionary:
  `{0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40', 4: 'Repair'}`.

---

## 10. Baseline Validation Metrics (3 Epochs)

Engineering holdout metrics on 480 validation images (742 defect instances):

| Metric | Overall | D00 (Long. Crack) | D10 (Trans. Crack) | D20 (Alligator) | D40 (Pothole) | Repair (Patch) |
|---|---:|---:|---:|---:|---:|---:|
| **Precision** | **0.3895** | 0.4480 | 0.3410 | 0.5410 | 0.2940 | 0.3240 |
| **Recall** | **0.2935** | 0.2820 | 0.5230 | 0.0172 | 0.0667 | 0.5780 |
| **mAP50** | **0.2628** | 0.3320 | 0.3740 | 0.0607 | 0.0771 | 0.4700 |
| **mAP50-95** | **0.1407** | 0.1440 | 0.1670 | 0.0271 | 0.0429 | 0.3240 |

*Note: These are preliminary Day-1 smoke training baselines on 3 epochs, not final research performance.*

---

## 11. Inference Runner & Verification

- **Script**: `yolo/run_yolo.py`
- **CLI Help**: Verified functional (`python yolo/run_yolo.py --help`).
- **Inference Command**:
  ```bash
  ./.venv/bin/python yolo/run_yolo.py \
      --input yolo/data/smoke_test_images \
      --output yolo/outputs/inference \
      --conf 0.25
  ```
- **Output Artifacts**:
  - Structured Summary: `yolo/outputs/inference/results.json`
  - Annotated Images: `yolo/outputs/inference/annotated/`
- **Functionality Smoke Results**:
  1. `China_Drone_001267.jpg`: Detected **`D00` (Longitudinal Crack)** at `[434, 1, 466, 213]` (conf 0.355). Matches ground truth XML box `[439, 19, 463, 180]`.
  2. `India_005086.jpg`: Detected **`D10` (Transverse Crack)** at `[358, 86, 695, 301]` (conf 0.251).
  3. `original_healthy.jpg`: **0 detections** (True Negative).
  4. `0454.png`: 0 detections at 0.25 threshold.

---

## 12. Files Created

1. `yolo/local_dataset_inventory.json`: Local dataset audit.
2. `yolo/configs/class_mapping.json`: RDD2022 class taxonomy.
3. `yolo/configs/data.yaml`: YOLO training dataset configuration.
4. `yolo/src/prepare_rdd2022_yolo.py`: VOC XML to YOLO converter.
5. `yolo/src/train_yolo.py`: Baseline training harness.
6. `yolo/run_yolo.py`: Standalone CLI runner emitting JSON and visual overlays.
7. `yolo/weights/best.pt` & `yolo/weights/last.pt`: Fine-tuned road-defect weights (local only, excluded by `.gitignore`).
8. `yolo/outputs/training/rdd2022_baseline/`: Metrics CSV, plots, and training summary JSON.
9. `yolo/outputs/inference/results.json`: Smoke inference detections.
10. `yolo/DAY1_CORRECTIVE_PROGRESS.md`: This report.

---

## 13. Limitations & Unverified Items

1. **Short Training**: Only 3 epochs were run for baseline verification; higher epochs (30–50) are required for full convergence.
2. **Pothole (D40) Sample Scarcity in China_Drone**: China_Drone has only 86 pothole instances; integrating `RoadSentinel_datasets/rdd2022_full/India` or `water_filled_potholes` is recommended for Day 2 to boost pothole recall.
3. **Adverse Weather Robustness**: Not verified today; Multi-Weather Pothole Detection (MWPD) evaluation is slated for Day 2.

---

## 14. Recommended Day 2 Priorities

1. **Multi-Dataset Expansion**: Incorporate `RDD2022 India` (3,187 potholes) into the training pipeline to balance D40 representation.
2. **Extended Training**: Scale from 3 epochs to 30–50 epochs with cosine LR scheduling.
3. **Interface Alignment**: Standardize bounding box and class format to feed downstream into the RoadSentinel analytics and dashboard pipeline.
