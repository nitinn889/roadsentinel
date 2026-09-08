# RoadSentinel Team Lead — GPU Follow-Up Execution Report
**Task**: Marion Day 5 Thin-Crack Resolution GPU Experiment  
**Executor**: Team Lead  
**Hardware / GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)  
**Environment**: PyTorch 2.13.0+cu130, CUDA 12.8, Python 3.10.21  
**Date**: September 8, 2026  
**Final Decision**: **REJECT** (Keep Day 5 production pipeline unchanged)  

---

## 1. Executive Summary

As requested in `sam2_dino/DAY5_GPU_FOLLOWUP.md`, the Team Lead executed the GPU-intensive 2× tiled / sliding-window DINOv2 experiment on `China_Drone_001267.jpg`, `0454.png`, `original_healthy.jpg`, and the common benchmark subset.

### Decision: **REJECT**
The 2× tiled inference experiment is **REJECTED** for the production RoadSentinel perception pipeline because:
1. **GT Box 2 remains completely unrecovered (0 px overlap)**: The lack of anomaly signal in GT Box 2 is not caused by patch grid sampling density, but by extreme low-contrast in the downsampled 512×512 RDD2022 dataset (~7 gray levels difference from background asphalt, which is below the natural asphalt noise floor).
2. **GT Box 1 severely regressed (882 px → 0 px)**: At 2× magnification, natural coarse asphalt aggregate is magnified, increasing background road anomaly distance to the 1× memory bank. This inflates the 92nd percentile road threshold from 0.3515 to 0.4876, extinguishing the genuine GT Box 1 crack signal entirely.
3. **Pothole localization collapsed (IoU 0.6398 → 0.0432)**: The strong Day 5 single-defect pothole was shattered into 8 disjoint fragments with a 93% drop in IoU.
4. **Healthy road produced false alarms (0 → 1 defect)**: Magnified pavement texture violated the zero-false-positive healthy invariant.
5. **Inference time increased by 5.32×** (146 ms → 779 ms).

The current Day-5 baseline (commit `25f5fa9`) remains the official, verified production perception pipeline. The tiled experiment is preserved separately in `road_health_pipeline/inference/tiled_dinov2_eval.py` for scientific reproducibility.

---

## 2. Technical Architecture & Spatial Resolution

| Parameter | Normal Production Mode (Day 5) | Tiled Experimental Mode |
|---|---|---|
| Input Image Handling | Full image (512×512) resized to 518×518 | Overlapping 256×256 crops resized to 518×518 |
| Effective Zoom Factor | 1.0× (native scale) | 2.0× (digital magnification) |
| Patch Size in Model Space | 14×14 px | 14×14 px |
| **Effective Patch Size on Road** | **13.84 × 13.84 px** | **6.92 × 6.92 px** |
| Tile Layout | Single frame | 3×3 grid (9 tiles), 50% stride (128 px) |
| Tile Seam Blending | N/A | 2D Hann window weighted accumulation |
| Memory Bank | Real memory bank (10,000 vectors) | Real memory bank (10,000 vectors) |
| Normalization Policy | 92nd percentile positive normalized scores | 92nd percentile positive normalized scores |

---

## 3. Anomaly Score Distributions & Patch Evidence

### China_Drone_001267.jpg Patch Diagnostics

| Region / Ground Truth | Normal Day 5 Anomaly (1×) | Tiled Anomaly (2×) | Impact |
|---|---|---|---|
| Road Background Median | Raw: 0.2660, Scale: 0.1328 | Raw: 0.2811, Scale: 0.1742 | Background noise variance widened |
| **Selected Threshold (p92)** | **0.3515** | **0.4876** | **Threshold inflated by +38.7%** |
| **GT Box 1** `[439, 19, 463, 180]` | Max norm: 0.5041 (1 patch >= thresh) | Max norm: 0.1470 (0 patches >= thresh) | **Signal extinguished by higher threshold** |
| **GT Box 2** `[434, 353, 456, 474]` | Max norm: 0.3780 (diluted in 14px patch) | Max norm: 0.3586 (0 patches >= thresh) | **Still below the 0.4876 threshold** |
| Background Road Max Norm | 0.5947 | 1.0000 | Magnified gravel aggregate triggers high anomaly |

### Why Tiling Failed: The Scale-Domain Mismatch Mechanism
- The frozen healthy-road memory bank was constructed from un-magnified (1×) roadway images.
- When an asphalt patch is digitally zoomed 2×, individual gravel stones and tar pores become visually prominent, sharp structures.
- DINOv2 self-attention tokens perceive this magnified aggregate as out-of-distribution compared to 1× healthy pavement, elevating raw background distance from 0.26 to ~0.33.
- Because background road variance increases, the 92nd percentile threshold rises from 0.3515 to 0.4876.
- The genuine crack (a subtle dark line) has lower texture contrast than the magnified gravel, causing the crack signal to fall beneath the elevated road threshold.

---

## 4. End-to-End Metrics: Normal Day-5 vs Tiled Experiment

### A. Crack Localization (`China_Drone_001267.jpg`)

| Metric | Normal Day 5 Baseline | Tiled 2× Experiment | Outcome / Delta |
|---|---|---|---|
| Accepted Defects | 3 | 4 | Noisy candidate clusters |
| **GT Box 1 Overlap** | **882 px (22.8%)** | **0 px (0.0%)** | **Severe regression (-882 px)** |
| **GT Box 2 Overlap** | **0 px (0.0%)** | **0 px (0.0%)** | **No recovery (0 px)** |
| **Total GT Overlap** | **882 px** | **0 px** | **Complete signal loss** |
| **IoU** | **0.0406** | **0.0000** | **Regressed to 0.0** |
| **Recall** | **0.1352** | **0.0000** | **Regressed to 0.0** |
| **Precision** | **0.0548** | **0.0000** | **Regressed to 0.0** |

### B. Pothole Localization (`0454.png`)

| Metric | Normal Day 5 Baseline | Tiled 2× Experiment | Outcome / Delta |
|---|---|---|---|
| **Fragments** | **1** | **8** | **Severe over-fragmentation (+7)** |
| **GT Overlap** | **10,052 px** | **1,130 px** | **-8,922 px (-88.8% loss)** |
| **IoU** | **0.6398** | **0.0432** | **Catastrophic collapse (-93.2%)** |
| **Recall** | **0.6694** | **0.0753** | **Collapsed (-88.8%)** |
| **Precision** | **0.9354** | **0.0922** | **Collapsed (-90.1%)** |

### C. Healthy Pavement Specificity (`original_healthy.jpg`)

| Metric | Normal Day 5 Baseline | Tiled 2× Experiment | Outcome / Requirement |
|---|---|---|---|
| **Accepted Defects** | **0** | **1** | **FAILED (False positive generated)** |
| Current Severity | 0.0000 | 0.4812 | Unacceptable on healthy road |
| Road Mask Ratio | 0.7133 | 0.7133 | Stable |
| Threshold | 0.8025 | 1.0000 | Clipped to maximum |

### D. Small Common-Benchmark Regression (3 Images)

| Sample ID | Condition | Normal Day 5 Defects | Tiled Defects | Behavior |
|---|---|---|---|---|
| `China_Drone_000104` | shadow | 2 | 1 | Reduced sensitivity |
| `China_Drone_000122` | shadow | 3 | 5 | False alarm fragmentation |
| `China_Drone_000175` | low_light | 2 | 2 | High threshold (0.6642) |

---

## 5. Computational & Hardware Performance Cost

Measured on local NVIDIA GeForce RTX 5060 Laptop GPU:

| Metric | Normal Day 5 Mode | Tiled 2× Mode | Cost Multiplier |
|---|---|---|---|
| `China_Drone_001267` Time | 207.0 ms | 663.1 ms | **3.20× slower** |
| `0454` Time | 130.2 ms | 1059.7 ms | **8.14× slower** |
| `original_healthy` Time | 102.1 ms | 614.9 ms | **6.02× slower** |
| **Mean Inference Time** | **146.4 ms / frame** | **779.2 ms / frame** | **5.32× slower** |
| **Peak GPU VRAM** | **639.2 MB** | **639.2 MB** | 1.00× (batch=1 sequential tiles) |

---

## 6. Test Suite & Integrity Verification

All 157 pipeline and integration tests were executed under PyTorch 2.13.0+cu130:
- `road_health_pipeline/tests/`: 145 passed
- `sam2_dino/test_day4_fixes.py`: 6 passed
- `sam2_dino/test_day5_fixes.py`: 4 passed
- `sam2_dino/test_contract.py`: 2 passed
- **Total: 157 passed, 0 failed, 0 skipped** (20.94s).

---

## 7. Strategic Recommendations for Marion Day 6

1. **Retain Day-5 Baseline as Production Default**:
   The Day-5 perception code (`pothole_localizer.py` with pre-consolidation and adaptive elongation context) remains the verified baseline. Pothole IoU (0.6398) and healthy specificity (0 FP) must be protected.
2. **Acknowledge the Fundamental Dataset Resolution Bound**:
   In RDD2022 `China_Drone`, the original raw images were downsampled by the dataset creators to 512×512 before publishing. A 1–2 px crack in GT Box 2 contains only ~7 gray levels of contrast across a 22 px bounding box. Digital upscaling (whether bilinear, bicubic, or tiled DINO) cannot synthesize optical photons that do not exist in the source JPEG.
3. **Focus Day 6 on Legitimate Fusion / Dual-Branch Perception**:
   Rather than forcing DINOv2 to act as a sub-pixel edge detector, Marion Day 6 should explore complementary feature fusion:
   - YOLO specializes in high-frequency bounding-box edge detection (already detecting D00 on both cracks).
   - DINOv2+SAM2 excels at unsupervised morphological crater/pothole segmentation (IoU 0.6398).
   - Dual-branch cross-verification yields the best of both worlds without breaking healthy road specificity.
