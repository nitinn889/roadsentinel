# RoadSentinel Phase 1B: Architectural Limitations, Boundaries & Open Constraints

**Document**: `reports/phase1b/LIMITATIONS.md`  
**Execution Phase**: Prompt 1B — Experimental Validity, Leakage Audit & Robustness Validation  
**Date**: September 10, 2026  
**Auditor**: Pair-Programming Agent (Antigravity IDE)  
**Status**: **OFFICIAL REGULATORY & SCIENTIFIC CAVEAT REGISTRY**  

---

## 1. Overview & Purpose

This document establishes the scientific boundaries, non-claims, and open engineering constraints of the RoadSentinel pipeline following the Phase 1B audit. In accordance with Prompt 1B rules, RoadSentinel prioritizes scientific conservatism and negative empirical preservation over inflated capability claims.

---

## 2. Key Scientific & Operational Limitations

### 2.1 Non-Universal Nature of DINOv2 Domain Separation
- **Observed Result**: The DINOv2 domain gate achieves AUROC $= 1.0000$ with a $+0.1158$ separation margin between China UAV frames ($N=480$) and India Dashcam frames ($N=300$).
- **Scientific Limitation**: This separation holds **strictly on the evaluated benchmark pair**. It is driven by composite multi-factor physical shifts:
  1. *Platform & Viewpoint*: Aerial UAV (steep downward pitch) vs. automotive dashcam (horizontal line of sight with horizon/sky).
  2. *Internal Windshield Artifacts*: Vehicle hood reflections, wiper arcs, and tinted glass glare present in India frames.
  3. *Surrounding Context*: Built environment, roadside clutter, and vegetation differ radically between China and India.
- **Prohibited Claim**: DINOv2 must **NEVER** be described as an "infallible universal out-of-distribution detector" or a generic detector of unseen road environments. Unseen aerial surveys with identical camera geometry may produce much narrower separation margins.

### 2.2 Uncalibrated Physical Depth & Volumetric Estimation
- **Observed Result**: SAM2 provides geometric mask contours and bounding pixel areas.
- **Scientific Limitation**: Pure monocular RGB images do not contain metric depth information. Without calibrated stereo cameras, LiDAR point clouds, or physical ground-truth pothole depth gauges, the pipeline **cannot estimate pothole cavity depth or volumetric asphalt loss**.
- **Prohibited Claim**: RoadSentinel does not claim calibrated metric depth or physical volume measurements.

### 2.3 ASTM D6433 Pavement Condition Index (PCI) Non-Compliance
- **Observed Result**: RoadSentinel computes a normalized distress severity index ($[0, 1]$) based on detected defect count, bounding box area, and SAM2 mask coverage.
- **Civil Engineering Limitation**: Standard ASTM D6433 pavement condition indexing requires rigorous distress severity classification (Low, Medium, High severity per distress type) over 100-meter sample units, combined with deduct-value curves and deduct-value adjustments.
- **Prohibited Claim**: RoadSentinel's severity index must **NEVER** be claimed as compliant with or equivalent to ASTM D6433 PCI ratings.

### 2.4 Statistical Regression Projections vs. Causal Deterioration
- **Observed Result**: The XGBoost scenario forecasting model predicts 30–90 day future IRI roughness deltas, reporting `WET_EXPOSURE = +0.0526` as the highest deterioration condition.
- **Statistical Limitation**: This model is a supervised empirical regression trained on historical FHWA LTPP section pairs. Environmental variables (`water_exposure`, `rainfall_level`, `traffic_level`) are descriptive covariates.
- **Prohibited Claim**: RoadSentinel does **NOT** claim causal deterioration modeling. The $+0.0526$ delta is a modelled statistical association, not a proven mechanical cause of structural subgrade failure.

### 2.5 Temporal Disappearance Semantics (`NOT_OBSERVED != REPAIRED`)
- **Observed Result**: In multi-day tracking sequences (e.g. `SEG_004`), 16 defect candidates disappear on subsequent days due to sunset grazing glare and camera contrast over-suppression.
- **Semantic Rule**: The pipeline enforces `NOT_OBSERVED != REPAIRED`.
- **Operational Limitation**: In autonomous visual inspection, a defect may disappear due to water pooling, lighting shifts, occluding debris, or sensor grazing angles. 
- **Prohibited Action**: Disappeared defects must **NEVER** be classified as repaired without confirmed municipal maintenance dispatch records.

### 2.6 Hardware Deployment Status (Raspberry Pi 5)
- **Observed Result**: Full computational profiling has been completed on the workstation GPU (NVIDIA GeForce RTX 5060 Laptop GPU, 8GB VRAM).
- **Deployment Limitation**: Physical profiling on embedded hardware (Raspberry Pi 5 with AI Kit / Coral NPU) remains pending hardware availability and real-time quantization testing.
- **Prohibited Claim**: No claims of Raspberry Pi 5 FPS, power wattage, or thermal stability may be reported until physical hardware benchmarking is completed.

### 2.7 Video Burst-Frame Similarity in Training/Validation Splits
- **Observed Result**: Perceptual hashing (pHash) identified **56 candidate near-duplicate pairs** between China Train ($N=1,921$) and China Val ($N=480$) with Hamming distance $\le 5$.
- **Data Limitation**: Because RDD2022 drone surveys were collected from continuous flight video passes along highway segments, sequential video frames in training flights can visually resemble validation flights passing over adjacent roadway verges.
- **Audit Action**: The 56 pairs are documented in [`artifacts/phase1b/leakage_pairs.csv`](../../artifacts/phase1b/leakage_pairs.csv) for open scientific transparency.

---

## 3. Summary of Open Research Blockers

| Research Component | Current Status | Primary Blocker | Planned Resolution |
|---|---|---|---|
| **Cross-Domain India Perception** | F1 = 0.0218 (Severe collapse) | Camera angle & windshield glare mismatch | Unsupervised domain adaptation / viewpoint normalization |
| **Physical Depth Estimation** | 2D pixel area only | Lack of metric 3D ground truth | Stereo depth integration or LiDAR sensor fusion |
| **Edge Hardware Profiling** | GPU benchmarks verified | Physical Raspberry Pi 5 hardware pending | TensorRT / ONNX INT8 deployment on edge SBC |
| **Causal Pavement Mechanics** | Statistical LTPP regression | Correlative observational data | Integration with mechanistic-empirical pavement design (MEPDG) |

---

## 4. Scientific Integrity Commitment

All conclusions drawn across RoadSentinel reports are constrained strictly to the evidence supported by the audited artifacts in [`artifacts/phase1b/`](../../artifacts/phase1b/). Negative empirical findings (e.g., standalone YOLO cross-domain collapse, failure of confidence-only filtering without domain gating, and persistence dominance over naive baselines) are fully preserved.
