# RoadSentinel — Phase 8 Progress Report: Reliability & Out-of-Distribution (OOD) Module

**Status**: COMPLETED & VERIFIED  
**Date**: September 2026  
**Execution Platform**: Linux x86_64, NVIDIA GeForce RTX 5060 Laptop GPU, PyTorch 2.13.0+cu130  
**Simulation Policy Compliance**: 100% compliant. Zero interaction with Unreal Engine, CARLA, or Interactive Studio. Zero YOLO retraining or perception retuning.

---

## 1. Executive Summary

Phase 8 builds and evaluates a principled **Reliability & Out-of-Distribution (OOD) Estimation Module** for RoadSentinel. The module provides a calibrated meta-assessment of YOLOv8n road-damage predictions without modifying the frozen perception detector or benchmark ground truth.

### Key Empirical Deliverables:
1. **Deterministic Target**: Evaluated on the frozen 480-image validation benchmark ($N=423$ successes, $N=57$ failures under IoU $\ge 0.50$).
2. **Training-Domain Reference**: Extracted 384-dimensional DINOv2 CLS embeddings across all 1,921 China_Drone training images to establish a non-parametric $k$-NN reference.
3. **Statistical Hypothesis Testing**: Confirmed that YOLO failures have statistically significant higher DINOv2 OOD scores ($p = 0.01351$, rank-biserial $r = 0.200$).
4. **Predictive Modeling**: Compared YOLO-confidence baseline (AUROC = 0.8649, AUPRC = 0.7274), DINOv2 OOD baseline (AUROC = 0.5971, AUPRC = 0.1540), and Combined model (AUROC = 0.8549, Balanced Acc = 0.7999, Failure F1 = 0.7143).
5. **Risk-Coverage Selective Prediction**: Demonstrated that rejecting the bottom 10–20% lowest reliability predictions drops accepted error from $11.87\%$ down to $4.69\%$ (**60.5% error reduction**).
6. **Operational Risk Bands**: High Band ($P \ge 0.85$, $84.8\%$ of data) achieves a **$95.09\%$ success rate**; Low Band ($P < 0.60$, $7.5\%$ of data) isolates an **$88.89\%$ failure rate**.
7. **Cross-Domain Diagnostics**: DINOv2 flagged maximum OOD ($1.0000$) on the forward-facing dashcam domain shift (`India_005086`), successfully alerting the system where YOLO produced zero detections.

---

## 2. Quantitative Model Comparison (5-Fold Stratified CV)

| Model Name | Features Evaluated | AUROC | AUPRC | Balanced Acc | Failure F1 | Brier Score | ECE |
|---|---|---|---|---|---|---|---|
| **YOLO Confidence Baseline** | max_conf, mean_conf, std_conf, pred_count, num_low_conf, num_high_conf | **0.8649** | **0.7274** | 0.7719 | 0.7045 | **0.0497** | **0.0156** |
| **DINOv2 OOD Baseline** | dino_ood_score, dino_knn_dist, dino_centroid_dist | 0.5971 | 0.1540 | 0.5833 | 0.2500 | 0.1039 | 0.0064 |
| **Combined Reliability Model** | Confidence + DINO OOD + Brightness, Contrast, Sharpness, Edge Density | 0.8549 | 0.7170 | **0.7999** | **0.7143** | 0.0511 | 0.0308 |

---

## 3. Risk-Coverage & Selective Prediction Table

| Coverage (%) | Accepted Images | Reliability Threshold | Accepted Failure Rate (%) | Error Reduction vs Baseline | Mean Accepted YOLO F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **100.0%** | 480 | 0.0000 | 11.87% | **0.0%** (Baseline) | 0.7161 |
| **95.0%** | 456 | 0.4412 | 7.24% | **+39.1%** | 0.7538 |
| **90.0%** | 432 | 0.7629 | 5.09% | **+57.1%** | 0.7697 |
| **85.0%** | 408 | 0.8847 | 4.90% | **+58.7%** | 0.7684 |
| **80.0%** | 384 | 0.9443 | 4.69% | **+60.5%** | 0.7663 |
| **70.0%** | 336 | 0.9850 | 4.76% | **+59.9%** | 0.7613 |
| **60.0%** | 288 | 0.9943 | 4.17% | **+64.9%** | 0.7667 |
| **50.0%** | 240 | 0.9981 | 2.92% | **+75.4%** | 0.7677 |

---

## 4. Operational Reliability Bands Table

| Band Name | Range ($P_{\text{reliable}}$) | Sample Count | Share (%) | Actual Success Rate (%) | Actual Failure Rate (%) | Mean YOLO F1 | Action Policy |
|---|---|---|---|---|---|---|---|
| **HIGH** | $\ge 0.85$ | 407 | 84.8% | **95.09%** | 4.91% | 0.7687 | `AUTOMATED_ACCEPT` |
| **MEDIUM** | $0.60 - 0.85$ | 37 | 7.7% | **86.49%** | 13.51% | 0.7433 | `SECONDARY_INSPECTION` |
| **LOW** | $< 0.60$ | 36 | 7.5% | **11.11%** | **88.89%** | 0.0944 | `ESCALATE_MANUAL_REVIEW` |

---

## 5. Visual Artifacts & Figures Summary

All 7 publication figures were generated at 300 DPI and saved to [`reliability/figures/`](file:///home/nitin-nandakumar/Downloads/roadsentinel/reliability/figures/):
1. `fig1_ood_success_failure.png`: Statistical distribution boxplot and jittered points comparing DINOv2 OOD scores for YOLO successes vs failures ($p = 0.0135$).
2. `fig2_roc_comparison.png`: ROC curves for Confidence-only, DINO OOD-only, and Combined models.
3. `fig3_pr_comparison.png`: Precision-Recall curves illustrating failure detection under 11.9% class imbalance.
4. `fig4_risk_coverage.png`: Dual-axis risk-coverage curve showing error rate collapse from 11.87% to 2.92%.
5. `fig5_reliability_calibration.png`: Probability calibration curves and reliability bin reliability.
6. `fig6_feature_importance.png`: Permutation feature importance ranking over 100 repeats.
7. `fig7_case_studies.png`: Visual diagnostic panel showing True Warning, High Trust, Overconfident Failure, and Underconfident Success case studies.

---

## 6. Critical Decision Gate Assessment

### Decision: **PARTIALLY SUPPORTED**

1. **Reliability-Aware Rejection**: **STRONGLY SUPPORTED**. Selectively filtering predictions by reliability score reduces YOLO failure rates by over $60\%$ with minimal data loss.
2. **In-Domain vs Cross-Domain Value of DINOv2 OOD**:
   - *In-Domain*: YOLO confidence features dominate failure prediction because the detector's internal probability distribution already reflects local crack ambiguity within the training distribution.
   - *Cross-Domain*: DINOv2 OOD is vital. When camera perspective shifts (e.g. `India_005086` dashcam), YOLO produces 0 detections with zero confidence signal; DINOv2 flags maximum OOD ($1.0000$), successfully alerting the inspection system.

---

## 7. Phase 9 Readiness Assessment

- **Readiness**: YES.
- All Phase 8 code, figures, datasets, and metadata are verified and stored under `reliability/`.
- Lightweight dashboard handoff assets are prepared in `integration/dashboard_assets/reliability/`.
- Ready for Phase 9 (Controlled Experiment B Capture & Processing).
