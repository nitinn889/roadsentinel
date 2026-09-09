# RoadSentinel Phase 12 Methodology: Reliability Validation & Full Signal Ablation

## 1. Overview & Research Objective

The goal of Phase 12 is to definitively validate whether the RoadSentinel reliability framework genuinely identifies poor perception assessments and to determine which signals contribute useful predictive information.

Phase 8 established an initial in-domain proof of concept under a lenient target criterion ($T_0: \text{F1} > 0$). Phase 12 expands this inquiry into a rigorous multi-target, multi-signal scientific ablation across both in-domain validation data (RDD2022 China_Drone, $N=480$) and independent cross-domain benchmark data (RDD2022 India Dashcam, $N=300$).

---

## 2. Multi-Target Perceptual Definitions ($T_0$ to $T_4$)

Perception failure is evaluated across five distinct ground-truth target definitions:

| Target | Formal Definition | Perceptual Objective | In-Domain China Class Balance ($N=480$) | Cross-Domain India Class Balance ($N=300$) |
|---|---|---|---|---|
| **$T_0$** | $\text{Image-level F1} > 0$ | Lenient Baseline (Phase 8 Target) | 423 Success (88.1%) / 57 Failure (11.9%) | 8 Success (2.7%) / 292 Failure (97.3%) |
| **$T_1$** | $\text{Image-level F1} \ge 0.50$ | Balanced Detection Quality | 394 Success (82.1%) / 86 Failure (17.9%) | 7 Success (2.3%) / 293 Failure (97.7%) |
| **$T_2$** | $\text{Image-level Recall} \ge 0.50$ | Moderate Defect Capture | 410 Success (85.4%) / 70 Failure (14.6%) | 8 Success (2.7%) / 292 Failure (97.3%) |
| **$T_3$** | $\text{Image-level Recall} \ge 0.75$ | High Defect Capture | 346 Success (72.1%) / 134 Failure (27.9%) | 4 Success (1.3%) / 296 Failure (98.7%) |
| **$T_4$** | $\text{Recall} == 1.0 \text{ at IoU} \ge 0.50 \text{ for GT} > 0$ | Complete Ground-Truth Recall | 343 Success (71.5%) / 137 Failure (28.5%) | 4 Success (1.3%) / 296 Failure (98.7%) |

---

## 3. Signal Ablation Models (Models A to H)

Eight feature configurations are systematically trained and evaluated:

1. **Model A (Max Confidence Only)**: `[max_confidence]`
2. **Model B (YOLO Confidence Features)**: `[max_confidence, mean_confidence, std_confidence, yolo_pred_count, num_low_conf, num_high_conf]`
3. **Model C (DINO OOD Only)**: `[dino_ood_score, dino_knn_distance, dino_centroid_distance]`
4. **Model D (Image Quality Only)**: `[brightness, contrast, sharpness, edge_density]`
5. **Model E (Confidence + DINO OOD)**: Model B + Model C
6. **Model F (Confidence + Image Quality)**: Model B + Model D
7. **Model G (DINO OOD + Image Quality)**: Model C + Model D
8. **Model H (Full Combined)**: Model B + Model C + Model D (All 13 features)

---

## 4. Modeling & Validation Protocol

- **Classifier**: Standardized $L_2$-penalized Logistic Regression (`C=1.0, solver='lbfgs'`) and Random Forest (`n_estimators=100, max_depth=5`) for non-linear comparison.
- **In-Domain Validation**: 5-Fold Stratified Cross-Validation on China_Drone ($N=480$). Feature scaling (`StandardScaler`) is fitted strictly on training folds to prevent data leakage. Out-of-fold probability predictions are collected across all 480 frames.
- **Cross-Domain Transfer**: Models fitted on the complete China_Drone development set are applied directly to the 300 India frames. Zero India ground-truth labels are used during training or threshold calibration.
- **Statistical Uncertainty**: 95% Bootstrap Confidence Intervals are computed via $B=1000$ resamples for all AUROC and AUPRC metrics.

---

## 5. Domain Gating vs. Sample-Level Reliability

A primary conceptual contribution of Phase 12 is the formal separation of:
- **Macro Domain Gating** (DINOv2 foundation embeddings): Evaluating whether an input image belongs to the operational design domain (ODD) of the detector.
- **Sample-Level Failure Prediction** (YOLO confidence metrics): Evaluating whether a specific bounding box inference is likely correct given that the image is within-domain.

The raw DINO $k$-NN distance $d_{k\text{NN}}$ is evaluated as a standalone binary domain classifier between China ($y=0$) and India ($y=1$).
