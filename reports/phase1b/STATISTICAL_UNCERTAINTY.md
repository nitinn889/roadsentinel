# RoadSentinel Phase 1B: Experiment 3 — Statistical Uncertainty Report

**Audit Document**: `reports/phase1b/STATISTICAL_UNCERTAINTY.md`  
**Execution Date**: September 10, 2026  
**Auditor**: Pair-Programming Agent (Antigravity IDE)  
**Resampling Protocol**: Empirical Percentile Bootstrap ($B = 2,000$ iterations, Seed $= 42$, 95% Confidence Interval)  

---

## 1. Executive Summary

This report establishes non-parametric 95% bootstrap confidence intervals for all core RoadSentinel perceptual, domain-gating, reliability, and longitudinal deterioration metrics. To prevent distortion from clustered or sequential observations, resampling units strictly mirror the data generation process:
- **Perception metrics** are resampled at the **image level** ($N=480$ China UAV, $N=300$ India Dashcam). Fine-grained flight sequence IDs are unverified/unavailable in RDD2022.
- **Matched IoU** is resampled at the **detection pair level** across valid true positives ($N=574$).
- **Domain Gate metrics** use stratified image-level resampling ($480$ familiar, $300$ out-of-domain).
- **Risk-Coverage metrics** resample complete validation evaluation frames ($N=480$) with re-ranking inside each bootstrap iteration.
- **XGBoost forecasting metrics** use **site-level cluster bootstrap**, resampling the 6 disjoint test sites with replacement ($N=24$ rows clustered across 6 sites).

---

## 2. Canonical 95% Confidence Intervals

| Metric Identifier | Point Estimate | 95% Confidence Interval | Sample Size (Unit) | Status | Methodological Notes |
|---|---|---|---|---|---|
| `china_yolo_precision` | `0.6568` | `[0.6238, 0.6920]` | 480 (image) | VALID | Image-level resampling (N=480) |
| `china_yolo_recall` | `0.7736` | `[0.7395, 0.8068]` | 480 (image) | VALID | Image-level resampling (N=480) |
| `china_yolo_f1` | `0.7104` | `[0.6818, 0.7388]` | 480 (image) | VALID | Image-level resampling (N=480) |
| `china_yolo_matched_iou` | `0.8007` | `[0.7911, 0.8100]` | 574 (matched_defect_box) | VALID | Detection-level resampling over TPs (N=574) |
| `india_yolo_precision` | `0.0964` | `[0.0385, 0.1667]` | 300 (image) | VALID | Image-level resampling (N=300) |
| `india_yolo_recall` | `0.0123` | `[0.0045, 0.0214]` | 300 (image) | VALID | Image-level resampling (N=300) |
| `india_yolo_f1` | `0.0218` | `[0.0081, 0.0374]` | 300 (image) | VALID | Image-level resampling (N=300) |
| `india_yolo_matched_iou` | `0.6841` | `[UNAVAILABLE_INSUFFICIENT_SAMPLE, UNAVAILABLE_INSUFFICIENT_SAMPLE]` | 8 (matched_defect_box) | INSUFFICIENT_SAMPLE | Only 8 TP matches exist in 300 India images; bootstrap CI is statistically degenerate and marked UNAVAILABLE_INSUFFICIENT_SAMPLE per Prompt 1B instructions |
| `domain_gate_auroc` | `1.0000` | `[1.0000, 1.0000]` | 780 (image_pair) | VALID | Stratified resampling across China (480) & India (300) |
| `domain_gate_china_false_warning_pct` | `1.4583` | `[0.6250, 2.5000]` | 480 (image) | VALID | China in-domain threshold p99=0.4491 (7/480 flagged) |
| `domain_gate_india_shift_detection_pct` | `100.0000` | `[100.0000, 100.0000]` | 300 (image) | VALID | India cross-domain detection at p99=0.4491 (300/300 detected) |
| `risk_coverage_target_t0_80pct_cov_error_pct` | `4.1667` | `[2.3438, 6.5104]` | 384 (accepted_image) | VALID | Accepted failure rate at 80pct_cov (384 accepted images) |
| `risk_coverage_target_t0_50pct_cov_error_pct` | `2.9167` | `[0.8333, 5.0000]` | 240 (accepted_image) | VALID | Accepted failure rate at 50pct_cov (240 accepted images) |
| `risk_coverage_target_t1_80pct_cov_error_pct` | `9.3750` | `[6.5104, 12.5000]` | 384 (accepted_image) | VALID | Accepted failure rate at 80pct_cov (384 accepted images) |
| `risk_coverage_target_t1_50pct_cov_error_pct` | `6.2500` | `[3.3333, 9.5833]` | 240 (accepted_image) | VALID | Accepted failure rate at 50pct_cov (240 accepted images) |
| `xgboost_test_r2` | `0.8055` | `[0.2120, 0.8943]` | 6 (site_cluster) | VALID | Site-level cluster bootstrap (2000 iterations, resampling 6 held-out test sites with replacement) |
| `xgboost_test_mae` | `0.0924` | `[0.0703, 0.1159]` | 6 (site_cluster) | VALID | Site-level cluster bootstrap (2000 iterations, resampling 6 held-out test sites with replacement) |
| `xgboost_test_rmse` | `0.1144` | `[0.0807, 0.1406]` | 6 (site_cluster) | VALID | Site-level cluster bootstrap (2000 iterations, resampling 6 held-out test sites with replacement) |

---

## 3. Detailed Interpretations & Methodological Notes

### 3.1 In-Domain China Perception Robustness
- **F1 Score**: $0.7104$, 95% CI: `[0.6818, 0.7388]`. Standalone YOLO achieves consistent in-domain performance above $0.68$ under familiar aerial flight conditions. Note: Fine-grained UAV flight IDs are unavailable in RDD2022, so image-level resampling is used; perceptual grouping sensitivity is evaluated in [`group_sensitive_metrics.csv`](../../artifacts/phase1b/group_sensitive_metrics.csv).
- **Matched IoU**: $0.8007$, 95% CI: `[0.7911, 0.8100]`. Spatial overlap among true positives clusters tightly around 0.80.

### 3.2 Cross-Domain India Collapse & Low-Sample Floor
- **F1 Score**: $0.0218$, 95% CI: `[0.0081, 0.0374]`. Upper bound remains below $0.040$, confirming cross-domain failure regardless of sampling variation.
- **Matched IoU**: Marked **`UNAVAILABLE_INSUFFICIENT_SAMPLE`** because only 8 true positive detections exist across 300 India images.

### 3.3 DINOv2 Domain Gate Benchmark Separation
- **AUROC**: $1.0000$, 95% CI: `[1.0000, 1.0000]`. Perfect empirical separation holds across all 2,000 bootstrap iterations on this evaluated UAV vs. Dashcam benchmark.
- **False Warning Rate**: $1.46\%$, 95% CI: `[0.6250%, 2.5000%]`. In-domain false quarantine is strictly bounded below $2.5\%$.

### 3.4 Risk-Coverage Selective Error Reduction
- Under Target $T_1$, accepted failure drops from $17.92\%$ (at 100% coverage) to:
  - $80\%$ Coverage: Point $9.38\%$, 95% CI `[6.5104%, 12.5000%]`
  - $50\%$ Coverage: Point $6.25\%$, 95% CI `[3.3333%, 9.5833%]`
- Uncertainty was estimated by complete frame resampling ($N=480$) with re-ranking in each bootstrap round.

### 3.5 XGBoost Forecasting Generalization & Site-Level Uncertainty
- **Held-Out Test $R^2$**: $0.8055$, Site-Level Cluster 95% CI: `[0.2120, 0.8943]`.
- **Held-Out Test MAE**: $0.0924$, Site-Level Cluster 95% CI: `[0.0703, 0.1159]`.
- **Site-Level Cluster Protocol**: By resampling the 6 test highway sections rather than independent rows, the cluster bootstrap properly reflects between-site variance and produces wider, scientifically honest bounds.
- **Critical Baseline Finding**: On this test set, naive Persistence ($\text{MAE} = 0.0754$) and OLS Linear Regression ($\text{MAE} = 0.0832$) outperform XGBoost ($\text{MAE} = 0.0924$). See [`forecast_baseline_comparison.csv`](../../artifacts/phase1b/forecast_baseline_comparison.csv).
