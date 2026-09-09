# RoadSentinel Perception Research Summary
**Final Perception Freeze & Scientific Interpretation**

- **Project Phase**: Phase 6
- **Status**: **COMPLETE & FROZEN** (`PERCEPTION_PIPELINES_FROZEN = TRUE`)
- **Evaluation Target**: 480 RDD2022 China_Drone Validation Images (742 GT Boxes)
- **Target Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM)
- **Date**: September 9, 2026

---

## 1. Headline Findings & Core Comparison

On the RDD2022 China_Drone common validation benchmark, the supervised **YOLOv8n** detector substantially outperformed the frozen **DINOv2–SAM2** anomaly pipeline in binary road-defect localization and computational efficiency:

| Metric | Supervised YOLOv8n | Zero-Shot DINOv2 + SAM2 | Comparison / Delta |
|---|---:|---:|---|
| **True Positives (TP)** | **574** | 25 | +549 TP (YOLO) |
| **False Positives (FP)** | **300** | 1,104 | +804 FP (DINO/SAM) |
| **False Negatives (FN)** | **168** | 717 | -549 FN (YOLO) |
| **Precision** | **0.6568** | 0.0221 | **+0.6346 (YOLO higher)** |
| **Recall** | **0.7736** | 0.0337 | **+0.7399 (YOLO higher)** |
| **F1 Score** | **0.7104** | 0.0267 | **+0.6837 (YOLO higher)** |
| **Mean Matched BBox IoU** | **0.8007** | 0.6533 | **+0.1474 (YOLO tighter)** |
| **Median Matched BBox IoU** | **0.8154** | 0.6480 | **+0.1674 (YOLO tighter)** |
| **Mean Inference Latency** | **3.62 ms** | 263.15 ms | **YOLO 72.7× faster** |
| **Throughput (FPS)** | **276.3 FPS** | 3.8 FPS | **YOLO 72.7× higher** |

> [!IMPORTANT]
> **Approved Research Conclusion**:
> "On the RDD2022 China-Drone common validation benchmark, the supervised YOLOv8n detector substantially outperformed the frozen DINOv2–SAM2 anomaly pipeline in binary road-defect localization and computational efficiency."
> 
> "This benchmark favors the supervised detector because it is evaluated within the same dataset/domain family used for its training."
> 
> "DINOv2–SAM2 remains valuable for pixel-level generic region segmentation, temporal geometry, anomaly analysis, and qualitative zero-shot investigation, but the current healthy-reference anomaly formulation exhibits poor precision on natural RDD asphalt textures."

---

## 2. Scientific Characterization of the Performance Gap

The dramatic performance disparity (F1 = 0.7104 vs. F1 = 0.0267) does not indicate that DINOv2 or SAM2 are flawed foundation models. Rather, **the specific anomaly-driven DINOv2–SAM2 formulation is poorly calibrated to the in-domain supervised RDD box-localization task.**

### Core Contributing Mechanisms:
1. **Supervised Inductive Bias**: YOLOv8n was trained for 25 epochs directly on 1,921 RDD2022 China_Drone images, optimizing anchor-free regression heads to fit CRDDC bounding-box conventions.
2. **Domain Alignment**: The validation set belongs to the exact same camera, sensor, flight altitude, and geographic domain as YOLO's training set, heavily favoring the supervised model.
3. **Semantic Class Priors**: YOLO explicitly learned discriminative spatial and textural signatures for longitudinal cracks, transverse cracks, and patch boundaries. DINOv2+SAM2 possesses no defect-specific training.
4. **Patch Token Dilution**: DINOv2 ViT-S/14 operates on a 37×37 patch grid over 518×518 input. Each patch corresponds to an effective 13.8 × 13.8 pixel area on the road surface. For narrow cracks (1–3 pixels wide), the crack occupies less than 5% of the patch area. The resulting token embedding is dominated by background asphalt, diluting the anomaly distance beneath the road threshold.
5. **Asphalt Noise Floor**: Natural asphalt exhibits significant aggregate variance (crushed stone, bitumen gloss, gravel discoloration). These natural variations frequently exceed the distance threshold to the 10,000-vector clean pavement memory bank, inducing severe false alarm rates.
6. **Mask vs. Box Geometric Asymmetry**: SAM2 segments tight, irregular defect contours. When converted to rectangular bounding boxes, fragmented or partial masks produce bounding boxes that overlap rectangular ground-truth boxes poorly, failing the strict 0.50 IoU cutoff.
7. **Downstream Prompt Failure**: SAM2 is a prompted segmenter. When DINOv2 candidate generation produces a spurious or offset centroid prompt, SAM2 segments the local texture at that prompt, amplifying false positives rather than filtering them.

---

## 3. DINOv2 + SAM2 Failure Mode Analysis

### 3.1 False Positive Characterization (FP = 1,104)
Quantitative analysis of the 1,104 false positive bounding boxes reveals:
- **Small Noise Fragments (< 500 px²)**: 15 boxes (1.4%).
- **Medium Texture Patches (500 – 5,000 px²)**: 560 boxes (50.7%).
- **Broad Anomaly Regions (≥ 5,000 px²)**: 529 boxes (47.9%).
- **Near Image Boundary (within 20 px of border)**: 307 boxes (27.8%).

#### Qualitative Failure Categories:
1. **Asphalt Aggregate & Surface Grain**: Coarse gravel variations trigger high anomaly cosine distances against the smooth memory bank (e.g. `China_Drone_000072`).
2. **Roadside Soil & Shoulder Transitions**: Unpaved shoulder boundaries where the asphalt edge meets vegetation or dirt trigger prominent anomaly clusters.
3. **Residual Lane Stripe Shadows**: Faint or degraded lane marking edges that bypass the road marking suppressor.
4. **Benign Oil Stains & Discolorations**: Water drying marks or vehicle fluid drops that appear statistically anomalous but represent zero structural damage.

### 3.2 False Negative Breakdown (FN = 717)
Out of 717 ground-truth defect boxes missed by DINOv2+SAM2 at IoU ≥ 0.50:
- **Detection Failures (0 px overlap / IoU ≤ 0.01)**: **389 boxes (54.3% of misses / 52.4% of all GT)**. The patch token anomaly score was completely sub-threshold; no candidate was generated or prompted to SAM2.
- **Localization / IoU Failures (0.01 < IoU < 0.50)**: **328 boxes (45.7% of misses / 44.2% of all GT)**. A candidate was detected and segmented, but the resulting mask-derived box failed the 0.50 IoU threshold due to partial crack coverage, mask fragmentation, or geometric offset.

---

## 4. Secondary IoU Sensitivity Interpretation (IoU ≥ 0.25)

When the IoU threshold is relaxed to 0.25:
- **DINOv2+SAM2 True Positives roughly triple**: 25 TP → **74 TP** (+49 additional matches, Recall 3.4% → 10.0%, F1 0.0267 → 0.0791).
- **YOLOv8n True Positives increase moderately**: 574 TP → **610 TP** (+36 matches, Recall 77.4% → 82.2%, F1 0.7104 → 0.7550).

### Interpretation:
The 49 additional DINO/SAM matches confirm that **spatial anomaly signal is present in many missed cases**, but the resulting segmentation masks capture only a portion of the elongated crack. When enclosed in a rectangular box, the partial mask achieves an IoU between 0.25 and 0.49. Relaxing the threshold demonstrates spatial awareness, but IoU 0.50 remains the primary scientific standard.

---

## 5. Supervised YOLOv8n Error & Semantic Analysis

While YOLOv8n achieved strong aggregate metrics (P = 0.6568, R = 0.7736, F1 = 0.7104), it is not without failure modes (FP = 300, FN = 168):

| Class | GT Count | Pred Count | TP | FP | FN | Precision | Recall | F1 | Mean IoU | Support & Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **D00** (Long.) | 266 | 351 | 208 | 143 | 58 | 0.5926 | 0.7820 | 0.6742 | 0.7710 | Strong baseline; 143 FPs on road seams |
| **D10** (Trans.) | 256 | 282 | 199 | 83 | 57 | 0.7057 | 0.7773 | **0.7398** | 0.7856 | **Strongest crack class** |
| **D20** (Alligator) | 58 | 50 | 23 | 27 | 35 | 0.4600 | 0.3966 | **0.4259** | 0.7631 | **Weakest meaningful class** (60.3% missed) |
| **D40** (Pothole) | 15 | 10 | 7 | 3 | 8 | 0.7000 | 0.4667 | 0.5600 | 0.8206 | **LOW SUPPORT — DO NOT DRAW STRONG CONCLUSION** |
| **Repair** (Patch) | 147 | 181 | 119 | 62 | 28 | 0.6575 | 0.8095 | **0.7256** | 0.8858 | Highest box tightness (IoU 0.886) |

### Key YOLO Error Drivers:
1. **D20 Alligator Crack Confusion**: Complex, fine mesh cracking is easily confused with coarse pavement aggregate, resulting in low recall (39.7%) and F1 (0.4259).
2. **Roadside False Positives**: Soil borders, curb transitions, and construction joints triggered 143 false positive D00 predictions (e.g. `China_Drone_000479`).
3. **Pothole Sample Limitation**: With only 15 instances in the validation set, pothole metrics cannot support robust generalization claims.

---

## 6. Cross-Domain Qualitative Analysis & Domain Shift

Evaluation on fixed out-of-domain diagnostic reference samples provides key insights into architecture generalization:

| Reference Sample | Viewpoint / Domain | YOLOv8n Response | DINOv2 + SAM2 Response | Scientific Finding |
|---|---|---|---|---|
| `India_005086.jpg` | **Forward Dashcam** (RDD2022 India) | **0 detections (Complete Failure)** | **2 plausible defect regions** | **Severe Domain Shift**: Under forward-facing dashcam perspective, YOLO completely collapsed due to perspective distortion, while DINOv2+SAM2 generated two plausible zero-shot defect regions without retraining. |
| `0454.png` | Ground-level (Pothole-600) | 1 detection (D40 pothole) | 1 detection (segmented pothole) | Both systems successfully localized the cavity despite ground-level perspective. |
| `original_healthy.jpg` | UAV / Nadir (Pristine Pavement) | **0 detections** | **0 detections** | **Zero False-Alarm Invariant Holds**: Neither system produced false positives on clean benchmark pavement. |

> [!WARNING]
> **Claim Boundary**: We do **not** claim DINOv2+SAM2 generalizes better overall based on a single image. We report strictly: under the forward-facing viewpoint shift of `India_005086`, YOLO produced zero detections, whereas DINOv2+SAM2 produced two plausible candidate regions.

---

## 7. Paper-Ready Research Figures

Seven publication-quality figures were generated in `benchmark/final_comparison/figures/`:
1. [`fig1_precision_recall_f1.png`](file:///home/nitin-nandakumar/Downloads/roadsentinel/benchmark/final_comparison/figures/fig1_precision_recall_f1.png): Direct cross-model P/R/F1 bar chart.
2. [`fig2_latency_throughput_log.png`](file:///home/nitin-nandakumar/Downloads/roadsentinel/benchmark/final_comparison/figures/fig2_latency_throughput_log.png): Log-scale latency and FPS throughput comparison.
3. [`fig3_matched_bbox_iou.png`](file:///home/nitin-nandakumar/Downloads/roadsentinel/benchmark/final_comparison/figures/fig3_matched_bbox_iou.png): Mean and median matched bounding-box IoU on true positives.
4. [`fig4_yolo_class_wise_metrics.png`](file:///home/nitin-nandakumar/Downloads/roadsentinel/benchmark/final_comparison/figures/fig4_yolo_class_wise_metrics.png): Semantic breakdown across D00, D10, D20, D40, Repair.
5. [`fig5_dino_failure_mode_breakdown.png`](file:///home/nitin-nandakumar/Downloads/roadsentinel/benchmark/final_comparison/figures/fig5_dino_failure_mode_breakdown.png): Quantitative split of DINO/SAM false negatives (detection vs localization) and false positive area distribution.
6. [`fig6_iou_sensitivity_comparison.png`](file:///home/nitin-nandakumar/Downloads/roadsentinel/benchmark/final_comparison/figures/fig6_iou_sensitivity_comparison.png): Impact of IoU 0.50 vs 0.25 threshold relaxation.
7. [`fig7_qualitative_panel_montage.png`](file:///home/nitin-nandakumar/Downloads/roadsentinel/benchmark/final_comparison/figures/fig7_qualitative_panel_montage.png): 4-panel visual montage demonstrating representative qualitative cases.

---

## 8. Dashboard Handoff Readiness

All perception benchmark artifacts, tables, manifests, figures, and panels are consolidated in `integration/dashboard_assets/perception/` for consumption by the Phase 7 interactive dashboard.
