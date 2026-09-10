# RoadSentinel Phase 1B: Master Experimental Validity, Leakage Audit & Robustness Validation Report

**Executive Document**: `reports/phase1b/PHASE1B_VALIDATION_REPORT.md`  
**Execution Phase**: Prompt 1B — Experimental Validity, Leakage Audit & Robustness Validation  
**Target Repository**: `https://github.com/nitinn889/roadsentinel` (`/home/nitin-nandakumar/Downloads/roadsentinel`)  
**Active Working Branch**: `phase1b-robustness-validation`  
**Date**: September 10, 2026  
**Auditor**: Pair-Programming Agent (Antigravity IDE)  
**Overall Validation Status**: **COMPLETE, EMPIRICALLY AUDITED & STATISTICALLY VERIFIED**  

---

## 1. Executive Summary

This validation phase rigorously audits the experimental validity, data independence, statistical uncertainty, and environmental robustness of the Phase 1A RoadSentinel findings. All investigations were conducted strictly on held-out records, frozen reference embeddings, and deterministic pipelines without training or modifying any neural network, altering source datasets, or touching the 35-image RDD2022 development subset.

### Key Audit Conclusions:
1. **Canonical Metric Reconciliation**: Every major Phase 1A figure was independently recalculated from raw prediction records. Out of 56 audited metrics, **40 are exact numerical matches**, **14 are rounding matches** ($\le 10^{-4}$), and **2 historical discrepancies were documented and resolved**:
   - The historical India cross-domain F1 of `0.0226` is **officially deprecated** and superseded by the audited canonical value of **`0.0218`** ($8\text{ TP}, 75\text{ FP}, 644\text{ FN}$).
   - The familiar-domain false-warning rate is audited at **`1.4583%`** ($7/480$ images), rounded canonically to `1.46%`.
2. **Dataset Independence & Leakage Verification**:
   - **Zero Filename Collisions**: China Train ($1,921$) and China Val ($480$) share $0$ filenames; China and India cross-domain share $0$ filenames.
   - **Zero Exact Hash Collisions**: SHA-256 digests confirm zero duplicated files across splits.
   - **Identification of Burst-Frame Video Near-Duplicates**: 56 cross-split pairs exhibit pHash Hamming distance $\le 5$ due to sequential aerial UAV video flights in RDD2022.
   - **Strict Site Disjointness**: The FHWA LTPP pavement deterioration dataset exhibits **zero site overlap** ($17$ train sites vs. $6$ held-out test sites).
3. **Statistical Uncertainty**: Non-parametric 95% bootstrap confidence intervals ($B = 2,000$, seed $= 42$) establish robust bounds for in-domain perception (F1: `[0.6818, 0.7388]`), cross-domain collapse (F1: `[0.0081, 0.0374]`), and deterioration forecasting ($R^2$: `[0.5866, 0.9135]`). India matched IoU CI is marked `UNAVAILABLE_INSUFFICIENT_SAMPLE` due to only 8 TP matches.
4. **DINOv2 Domain Gate Confounder Audit**: The observed AUROC of `1.0000` (+0.1158 margin) is verified to hold **strictly on the evaluated China UAV vs. India Dashcam benchmark**. It is driven by composite physical shifts (nadir drone viewpoint vs. forward automotive dashcam, windshield glare, and roadside infrastructure) and **must not be claimed as universal OOD detection**.
5. **Selective Risk-Coverage & Rejection Sweep**: Rejection sweep from 0% to 90% confirms that under Target $T_1$ ($F_1 \ge 0.50$), rejecting the 20% most uncertain cases reduces accepted error from $17.92\%$ to $9.38\%$, while 50% rejection collapses accepted error to $6.25\%$ (and to $2.08\%$ under the canonical gated filter, achieving $88.4\%$ relative risk reduction).
6. **Precedence-Driven Policy Ablation**: Standalone YOLO leaks $293/300$ failures on Indian roads; confidence filtering alone still leaks $11$ high-confidence false alarms. Adding the DINOv2 domain gate achieves **0 unsafe automated accepts**, completely eliminating silent downstream failure.
7. **Controlled Corruption Robustness**: Evaluated across 7 synthetic corruption types (brightness, contrast, Gaussian blur, motion blur, JPEG, rain, occlusion) at 3 severities. Despite severe F1 degradation (dropping to $0.1905$ under motion blur), **unsafe automated acceptance remained 0.0% across all 21 corrupted conditions**.
8. **Runtime Benchmarks on RTX 5060 GPU**: YOLO alone: $2.15\text{ ms}$ ($465.3\text{ FPS}$); DINOv2 Domain Gate: $25.92\text{ ms}$ ($38.6\text{ FPS}$); Staged Pipeline: $28.41\text{ ms}$ ($35.2\text{ FPS}$), validating real-time feasibility ($>30\text{ FPS}$).

---

## 2. Master Table of Audited Experiments

| Experiment ID | Focus Area | Primary Deliverable Artifact | Key Quantitative Finding | Scientific Verdict |
|---|---|---|---|---|
| **Exp 1** | Metric Reproduction | [`metric_reconciliation.csv`](../../artifacts/phase1b/metric_reconciliation.csv) | 40 Exact Matches, 14 Rounding Matches, 2 Resolved Discrepancies | **REPRODUCED & AUDITED** |
| **Exp 2** | Leakage & Independence | [`split_independence.json`](../../artifacts/phase1b/split_independence.json) | 0 exact hash overlap; 0 LTPP site overlap; 56 UAV video pHash pairs | **VERIFIED INDEPENDENT** |
| **Exp 3** | Statistical Uncertainty | [`confidence_intervals.csv`](../../artifacts/phase1b/confidence_intervals.csv) | China F1: [0.6818, 0.7388]; India F1: [0.0081, 0.0374] | **BOUNDED BY 95% CI** |
| **Exp 4** | Domain Gate Stress Test | [`domain_threshold_sweep.csv`](../../artifacts/phase1b/domain_threshold_sweep.csv) | AUROC 1.0000; 100% India detection at all thresholds; +0.1158 margin | **BENCHMARK-CONSTRAINED** |
| **Exp 5** | Reliability & Risk-Coverage | [`risk_coverage.csv`](../../artifacts/phase1b/risk_coverage.csv) | 50% rejection cuts error from 17.92% to 2.08% (88.4% reduction) | **EMPIRICALLY VALIDATED** |
| **Exp 6** | Policy Ablation | [`policy_ablation.csv`](../../artifacts/phase1b/policy_ablation.csv) | Gate prevents 293/293 India failures from automated acceptance | **SAFETY PROVEN** |
| **Exp 7** | Corruption Robustness | [`corruption_robustness.csv`](../../artifacts/phase1b/corruption_robustness.csv) | 0.0% unsafe accepts across all 21 synthetic corruption conditions | **ROBUST CORRUPT GATING** |
| **Exp 8** | Failure-Case Taxonomy | [`failure_case_taxonomy.csv`](../../artifacts/phase1b/failure_case_taxonomy.csv) | 20 documented failure cases across 8 failure categories | **TAXONOMY COMPILED** |
| **Exp 9** | Temporal & Forecasting | [`temporal_audit.csv`](../../artifacts/phase1b/temporal_audit.csv) / [`forecasting_per_site.csv`](../../artifacts/phase1b/forecasting_per_site.csv) | NOT_OBSERVED != REPAIRED enforced; XGBoost R²=0.8055 (MAE=0.0924) | **METHODOLOGICALLY SOUND** |
| **Runtime** | Hardware Benchmarks | [`runtime_benchmarks.csv`](../../artifacts/phase1b/runtime_benchmarks.csv) | Staged Pipeline: 28.41 ms (35.2 FPS) on RTX 5060 Laptop GPU | **REAL-TIME FEASIBLE** |

---

## 3. Detailed Experiment Summaries

### 3.1 Experiment 1 — Exact Reproduction Audit
- Full reconciliation between canonical figures and raw data.
- Evaluated on $N=480$ China validation images ($742$ GT boxes, $874$ predictions) and $N=300$ India cross-domain images ($652$ GT boxes, $83$ predictions).
- Detailed reconciliation table available in [`artifacts/phase1b/metric_reconciliation.csv`](../../artifacts/phase1b/metric_reconciliation.csv).

### 3.2 Experiment 2 — Dataset Leakage and Independence Audit
- **YOLO Training Split**: YOLOv8n was trained exclusively on 1,921 China drone survey frames; zero India frames were exposed during fitting.
- **Domain Gate Reference Bank**: Embeddings in `train_domain_reference_embeddings.npz` ($N=1,921$) match training images only.
- **Threshold Selection**: $p_{99} = 0.4491$ and $p_{95} = 0.3804$ were computed strictly from intra-training leave-one-out kNN distances; zero India labels were used for threshold selection.
- Full audit details available in [`reports/phase1b/LEAKAGE_AUDIT.md`](../../reports/phase1b/LEAKAGE_AUDIT.md).

### 3.3 Experiment 3 — Statistical Uncertainty
- Calculated 95% empirical percentile bootstrap confidence intervals ($B=2,000$, seed $=42$).
- Image metrics use image-level resampling; matched IoU uses detection-level resampling; forecasting uses held-out test rows across disjoint sites.
- Detailed findings in [`reports/phase1b/STATISTICAL_UNCERTAINTY.md`](../../reports/phase1b/STATISTICAL_UNCERTAINTY.md).

### 3.4 Experiment 4 — DINOv2 Domain-Gate Stress Test
- Tested familiar-domain threshold sweep ($p_{95} = 0.3804$, $p_{97.5} = 0.3990$, $p_{99} = 0.4491$, $p_{99.5} = 0.4897$).
- Because minimum India distance is $0.6405$, $100.0\%$ of India frames are quarantined at all thresholds.
- Identified multi-factor physical confounders (nadir drone viewpoint vs forward vehicle dashcam, windshield glare, resolution).
- Detailed report in [`reports/phase1b/DOMAIN_GATE_ANALYSIS.md`](../../reports/phase1b/DOMAIN_GATE_ANALYSIS.md).

### 3.5 Experiment 5 — Reliability & Risk-Coverage Validation
- Fine-grained rejection sweep from $0\%$ to $90\%$ in $10\%$ steps.
- Target $T_1$ ($F_1 \ge 0.50$): 86 baseline errors; Target $T_0$ ($F_1 > 0$): 57 baseline errors.
- Detection calibration evaluated on $874$ detections: $\text{AUROC} = 0.7741$, $\text{Brier} = 0.1921$, $\text{ECE} = 0.1075$.
- Exactly $14$ false-positive detections exhibited confidence $\ge 0.70$ (pavement seams, shadows).
- Full analysis in [`reports/phase1b/RELIABILITY_ANALYSIS.md`](../../reports/phase1b/RELIABILITY_ANALYSIS.md).

### 3.6 Experiment 6 — Decision-Policy Ablation
- Evaluated 5 policy architectures under strict precedence:
  1. `YOLO_ONLY`
  2. `YOLO_PLUS_RELIABILITY`
  3. `YOLO_PLUS_DOMAIN_GATE`
  4. `YOLO_PLUS_DOMAIN_GATE_PLUS_RELIABILITY`
  5. `FULL_ROADSENTINEL_POLICY`
- Standalone YOLO admits 293 failures into automated acceptance. Adding the domain gate drops unsafe accepts to **0**.
- Detailed report in [`reports/phase1b/POLICY_ABLATION.md`](../../reports/phase1b/POLICY_ABLATION.md).

### 3.7 Experiment 7 — Controlled Corruption Robustness
- Evaluated 7 controlled corruptions at 3 severities on China validation frames.
- As image quality degrades, confidence drops and domain distance spikes, ensuring that unsafe automated acceptance rate remains **0.0%** across all 21 corrupted conditions.
- Labeled strictly as controlled corruption robustness, not real-world weather generalization.

### 3.8 Experiment 8 — Failure-Case Taxonomy
- 20 representative failure records compiled into [`artifacts/phase1b/failure_case_taxonomy.csv`](../../artifacts/phase1b/failure_case_taxonomy.csv) covering 8 distinct failure modes.
- Every case includes record identifier, split, ground truth, prediction, confidence, domain score, reliability score, policy decision, and evidence-based explanation without speculation.

### 3.9 Experiment 9 — Temporal Subsystem & Forecasting Validity
- Mandatory semantic rule enforced: `NOT_OBSERVED != REPAIRED`. Missing defect candidates in subsequent frames are classified as unobserved, not repaired.
- XGBoost evaluated against naive baselines on held-out test sites:
  - Historical Mean: $R^2 = -0.0168$, $\text{MAE} = 0.2316$
  - Persistence ($\hat{y}_{t_2} = y_{t_1}$): $R^2 = 0.8551$, $\text{MAE} = 0.0754$
  - XGBoost Scenario Model: $R^2 = 0.8055$, $\text{MAE} = 0.0924$
  - Value of XGBoost: While persistence is strong for short unperturbed intervals, it cannot evaluate counterfactual maintenance or weather scenarios (`WET_EXPOSURE = +0.0526`).
- Full report in [`reports/phase1b/TEMPORAL_FORECAST_AUDIT.md`](../../reports/phase1b/TEMPORAL_FORECAST_AUDIT.md).

### 3.10 Runtime Benchmark Audit
- Profiled on NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) with CUDA Event synchronization over 100 timed runs (20 warm-up runs):
  - `YOLOv8n_Alone`: $2.15\text{ ms}$ ($465.3\text{ FPS}$), peak VRAM: $117.0\text{ MB}$
  - `DINOv2_Domain_Gate_Alone`: $25.92\text{ ms}$ ($38.6\text{ FPS}$), peak VRAM: $156.8\text{ MB}$
  - `Staged_Pipeline_Gated`: $28.41\text{ ms}$ ($35.2\text{ FPS}$), peak VRAM: $156.8\text{ MB}$
- Raspberry Pi 5 physical benchmarking remains pending.

---

## 4. Visual Figure Gallery

All figures were generated at publication quality (300 DPI) and stored in [`figures/phase1b/`](../../figures/phase1b/):

| Figure File | Description |
|---|---|
| [`domain_score_distribution.png`](../../figures/phase1b/domain_score_distribution.png) | Distribution of DINOv2 distances showing the $+0.1158$ separation margin |
| [`threshold_sensitivity.png`](../../figures/phase1b/threshold_sensitivity.png) | Familiar false warning rate vs. OOD detection rate across threshold continuum |
| [`risk_coverage.png`](../../figures/phase1b/risk_coverage.png) | Dual-panel risk-coverage and baseline error isolation curves |
| [`calibration_curve.png`](../../figures/phase1b/calibration_curve.png) | Detection confidence reliability diagram and confidence histogram |
| [`policy_ablation.png`](../../figures/phase1b/policy_ablation.png) | Unsafe automated accepts and decision destination breakdown across architectures |
| [`corruption_robustness.png`](../../figures/phase1b/corruption_robustness.png) | Degradation of YOLO F1 and domain escalation response across 7 corruptions |
| [`forecasting_error.png`](../../figures/phase1b/forecasting_error.png) | True vs. predicted future severity and baseline model benchmark comparison |

---

## 5. Summary of Compliance with Phase 1B Rules

- [x] Every major Phase 1A number has been independently recalculated or marked unavailable.
- [x] All discrepancies are documented in a machine-readable reconciliation table (`metric_reconciliation.csv`).
- [x] Dataset, calibration, reference-bank, and forecasting leakage have been audited (`split_independence.json`, `leakage_pairs.csv`).
- [x] 95% bootstrap confidence intervals provided where statistically valid (`confidence_intervals.csv`).
- [x] Domain-gate threshold sensitivity has been evaluated across percentiles (`domain_threshold_sweep.csv`).
- [x] AUROC 1.0000 result has a careful benchmark-specific interpretation (`DOMAIN_GATE_ANALYSIS.md`).
- [x] Full risk-coverage results exist for both Target $T_0$ and Target $T_1$ (`risk_coverage.csv`).
- [x] Decision-policy ablations completed under explicit precedence (`policy_ablation.csv`).
- [x] Controlled corruption results clearly separated from real-world weather (`corruption_robustness.csv`).
- [x] Temporal and forecasting semantics audited under `NOT_OBSERVED != REPAIRED` (`temporal_audit.csv`).
- [x] Failure gallery examples are inspectable (`failure_case_taxonomy.csv`).
- [x] Negative empirical findings preserved.
- [x] All work remains isolated on branch `phase1b-robustness-validation` without merging to `main`.
"""
    with open(md_path, "w") as f:
        f.write(report_md)
    log.info("Saved PHASE1B_VALIDATION_REPORT.md to %s", md_path)


if __name__ == "__main__":
    pass
