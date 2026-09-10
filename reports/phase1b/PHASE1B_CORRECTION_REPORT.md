# RoadSentinel Phase 1B-R: Comprehensive Correction and Revalidation Report

**Document**: `reports/phase1b/PHASE1B_CORRECTION_REPORT.md`  
**Phase**: Prompt 1B-R — Correct and Revalidate Phase 1B  
**Target Repository**: `https://github.com/nitinn889/roadsentinel` (`/home/nitin-nandakumar/Downloads/roadsentinel`)  
**Active Working Branch**: `phase1b-robustness-validation`  
**Date**: September 10, 2026  
**Auditor**: Pair-Programming Agent (Antigravity IDE)  
**Status**: **OFFICIAL PASS & AUDIT RECONCILIATION COMPLETE**  

---

## 1. Executive Summary

This correction report addresses all contradictions, historical discrepancies, unsupported claims, and syntax issues identified in Prompt 1B-R regarding the Phase 1B robustness audit. Rather than merely rewriting narrative prose, every correction has been grounded in **executable analysis of raw data artifacts**, reproducible scripts in `integration/phase1b/`, and machine-readable data products.

All negative empirical findings—specifically the fact that simple physical persistence and OLS linear regression strictly outperform the XGBoost tree model on the held-out pavement progression test set—are fully preserved and documented.

---

## 2. Master Correction Reconciliation Matrix

The table below summarizes each required correction, referencing the supporting artifact and stating whether the scientific conclusion changed:

| Item # | Focus Area | Old Claim / Value | Corrected Claim / Value | Raw Supporting Artifact | Code / Script Used | Reason for Correction | Scientific Conclusion Changed? |
|:---:|---|---|---|---|---|---|:---:|
| **1** | Syntactic Hygiene | Trailing Python boilerplate (`with open(md_path... if __name__ == '__main__': pass`) in `PHASE1B_VALIDATION_REPORT.md` | Clean Markdown document terminating properly at the compliance checklist | `reports/phase1b/PHASE1B_VALIDATION_REPORT.md` | `reports/phase1b/PHASE1B_VALIDATION_REPORT.md` | File generation script accidentally appended stringified execution wrapper into file text | **NO** (Syntactic fix) |
| **2** | Reliability Rejection | 8.85% (80% cov) and 2.08% (50% cov) attributed to pure Model B selective prediction | Pure Model B T1 rejection yields **9.38%** (80% cov) and **6.25%** (50% cov); 8.85% and 2.08% are deprecated historical fixed-threshold values | `artifacts/phase1b/risk_coverage.csv` | `integration/phase1b/audit_reliability.py` | Pure percentile sorting by Model B out-of-fold probability isolates 82.56% of errors leaving 6.25% accepted error; 2.08% belongs strictly to historical Phase 12 fixed-threshold / gated filter | **YES** (Pure selective error at 50% cov is 6.25%, not 2.08%) |
| **3** | Leakage Audit | LEAKAGE_AUDIT reported 0 near-duplicates for DINOv2 Ref Bank vs Eval, but 56 for China Train vs Val | The 56 pHash pairs connect 46 training images to 24 validation images across 16 connected components; group independence marked **UNVERIFIED**; sensitivity delta $\Delta\text{F1} = -0.0082$ | `artifacts/phase1b/group_sensitive_metrics.csv` | `integration/phase1b/revalidate_corrections.py` | Ref Bank consists of China Train embeddings; 56 perceptual pairs apply identically to the source data. Excluding affected images demonstrates robust in-domain F1 (0.7022 vs 0.7104, $\Delta\text{F1} = -0.0082$) | **NO** (Robust in-domain performance confirmed) |
| **4** | Domain Gate Thresholds | Canonical $p_{95}=0.3804$ vs exact $0.3503$; canonical $p_{99}=0.4491$ vs exact $0.4558$ | Documented exact mathematical formulation (LOO, 20-NN cosine distance, L2-norm, linear interpolation); canonical $p_{99}=0.4491$ selected purely on familiar train data | `artifacts/phase1b/domain_threshold_sweep.csv` | `cross_domain/results/train_domain_reference_embeddings.npz` | Resolved origin: 0.4491 from Phase 8 metadata, 0.4558 from frozen npz archive; 0.3804 was Phase 12 conservative warning threshold; 100% India quarantine holds under all | **NO** (Benchmark separation intact at all thresholds) |
| **5** | Domain Lighting Prose | Prose in DOMAIN_GATE_ANALYSIS stated mean distances were `0.1873 vs 0.1842`, while table showed `0.1771 vs 0.1912` | Prose aligned with table: Low Brightness mean $= 0.1771$, High Brightness mean $= 0.1912$ | `reports/phase1b/DOMAIN_GATE_ANALYSIS.md` | `reliability_validation/data/china_validation_ablation_master.csv` | Early draft cited preliminary unstratified averages rather than final audited stratum values | **NO** (Lighting invariance confirmed) |
| **6** | Statistical Uncertainty | Row-level bootstrap on $N=24$ test rows ($R^2$ 95% CI: `[0.5866, 0.9135]`); claimed unverified $p < 10^{-6}$ | Site-level cluster bootstrap resampling 6 test sites ($R^2$ 95% CI: `[0.2120, 0.8943]`, MAE: `[0.0703, 0.1159]`); removed $p < 10^{-6}$; noted lack of drone flight clustering | `artifacts/phase1b/confidence_intervals.csv` | `integration/phase1b/audit_uncertainty.py` | Row-level resampling ignores intra-site correlation, producing overconfident bounds; site-level cluster bootstrap reflects genuine between-site variance | **YES** (Honest, wider uncertainty bounds for forecasting) |
| **7** | Forecasting Interpretation | XGBoost presented as superior; claimed "30–90 day horizons" and counterfactual capability; current_severity stated as 92.6% | Persistence ($\text{MAE}=0.0754$) and OLS ($\text{MAE}=0.0832$) outperform XGBoost ($\text{MAE}=0.0924$); observed horizons span 240–720 days (mean 447.5 days); severity importance is 75.53% gain; non-causal interpretation | `artifacts/phase1b/forecast_baseline_comparison.csv` | `integration/phase1b/revalidate_corrections.py` | Historical LTPP survey pairs span 1–2 years, not 30–90 days; tree model overfits compared to physical persistence; 75.53% is exact gain in scenario_model_v2 | **YES** (Critical negative result preserved: Persistence dominates) |
| **8** | Temporal Provenance | Experiment A described as "total physical captures: 40" / real-world captures | Formally classified as CARLA / Unreal Engine synthetic simulator captures; preserved `NOT_OBSERVED != REPAIRED` | `reports/phase1b/TEMPORAL_FORECAST_AUDIT.md` | `env/scripts/rs_inspection_capture.py` | Captures in SEG_001–SEG_004 were collected in CARLA simulation with injected defects and configured weather, not physical aerial survey | **YES** (Cannot claim simulator progression as real pavement deterioration) |
| **9** | Runtime Latency | 2.15 ms in ablation table conflicted with 3.62 ms canonical latency | Explicit protocol separation: 2.15 ms is GPU inference-only forward pass (RTX 5060, batch=1, CUDA events); 3.62 ms is end-to-end benchmark (preprocessing + forward + NMS) | `artifacts/phase1b/runtime_benchmarks.csv` | `integration/phase1b/audit_runtime.py` | Numbers came from distinct benchmark protocols (pure tensor forward pass vs. complete deployment pipeline) | **NO** (Both numbers verified under explicit protocols) |
| **10** | Policy Language | Claimed "SAFETY PROVEN" and "guarantees zero unsafe leakage" | Replaced with objective standard: "zero unsafe automated accepts were observed on the evaluated benchmark"; distinguished safe routing from defect detection | `reports/phase1b/POLICY_ABLATION.md` | `reports/phase1b/POLICY_ABLATION.md` | Universal safety claims cannot be proven empirically on a finite 300-image benchmark; domain escalation is safe routing, not defect detection | **NO** (Maintains empirical rigor without unwarranted extrapolation) |
| **11** | Controlled Corruption Validation | Only aggregate condition-level averages reported in `corruption_robustness.csv` | Enriched with full frame-level breakdown: total frames (50), failure frames, auto-accepted failures (0), human reviews, escalations across all 22 conditions | `artifacts/phase1b/corruption_robustness.csv` | `integration/phase1b/revalidate_corrections.py` | Verifies that zero unsafe accepts was evaluated and confirmed on all 1,100 individual frame inferences, not merely aggregate condition rows | **NO** (Strengthens empirical foundation of robustness claim) |
| **12** | False Positive Enumeration | Heading stated "Exactly 14 detections" but table displayed only 10 rows | Enumerated all 14 high-confidence false positive detections in `RELIABILITY_ANALYSIS.md` and `failure_case_taxonomy.csv` | `artifacts/phase1b/failure_case_taxonomy.csv` | `integration/phase1b/audit_failure_taxonomy.py` | Previous report generator silently sliced `high_conf_fp[:10]`, leaving 4 detections omitted | **NO** (Completeness and transparency fix) |
| **13** | LaTeX & Typography | Rendered with broken sequences such as `k\text{NN}` $\rightarrow$ `k ext{NN}`, `t \rightarrow t+1` $\rightarrow$ `t ightarrow t+1` | Repaired all instances to valid LaTeX math mode: `$k\text{NN}$`, `$\text{IoU}$`, `$t \rightarrow t+1$` across all reports | `reports/phase1b/*.md` | Regex / Python raw-string escape fix | String escape collisions (`\t` converted to ASCII tab, `\r` converted to carriage return) during text output | **NO** (Typography and formatting fix) |

---

## 3. Detailed Audit of Corrections

### 3.1 Correction 1: Removal of Accidentally Appended Python Code
- **Old Claim / State**: `reports/phase1b/PHASE1B_VALIDATION_REPORT.md` had dangling Python file-writing boilerplate at the bottom:
  ```python
  """
      with open(md_path, "w") as f:
          f.write(report_md)
      log.info("Saved PHASE1B_VALIDATION_REPORT.md to %s", md_path)
  
  if __name__ == "__main__":
      pass
  ```
- **Corrected Claim / State**: Clean Markdown document ending at Section 4 ("Summary of Compliance with Phase 1B Rules").
- **Supporting Artifact**: `reports/phase1b/PHASE1B_VALIDATION_REPORT.md`.
- **Reason**: The automated markdown generator script leaked its execution wrapper into the text buffer.

---

### 3.2 Correction 2: Reconciling Reliability Results (9.38%, 8.85%, 6.25%, 2.08%)
- **Old Claim**: Canonical research metrics cited $8.85\%$ (at 80% coverage) and $2.08\%$ (at 50% coverage) as the performance of Model B selective prediction under Target $T_1$.
- **Corrected Claim**:
  1. **Pure Reliability Rejection (Model B on Target $T_1$, $F_1 \ge 0.50$)**:
     - At 80% Coverage (20% Rejection): Accepted error drops from $17.92\%$ to **$9.38\%$** ($36$ accepted failures / $384$ accepted images), achieving $47.66\%$ error reduction and isolating $58.14\%$ of baseline errors.
     - At 50% Coverage (50% Rejection): Accepted error drops to **$6.25\%$** ($15$ accepted failures / $240$ accepted images), achieving $65.12\%$ error reduction and isolating $82.56\%$ of baseline errors.
  2. **Historical Fixed-Threshold / Gated Baseline**:
     - The numbers **$8.85\%$** ($34/384$) and **$2.08\%$** ($5/240$) originated in Phase 12 from applying fixed probability thresholds ($p \ge 0.7629$ and $p \ge 0.9850$) calibrated on an earlier composite model.
     - **Deprecation**: $2.08\%$ is officially deprecated as a pure selective prediction metric. It must **never** be attributed to pure 50% reliability rejection, as the exact raw out-of-fold records reproduce $6.25\%$.
- **Supporting Artifact**: [`artifacts/phase1b/risk_coverage.csv`](../../artifacts/phase1b/risk_coverage.csv).
- **Scientific Impact**: **Changed**. Pure confidence-based selective prediction at 50% coverage reduces accepted failure rate to 6.25% (not 2.08%).

---

### 3.3 Correction 3: Dataset Leakage Audit & Group Independence Sensitivity
- **Old Claim**: `LEAKAGE_AUDIT.md` reported 56 pHash near-duplicates for China Train vs. China Val, but reported 0 for DINOv2 Reference Bank vs. Evaluated Data, creating a direct contradiction.
- **Corrected Claim**:
  - Because the DINOv2 Reference Bank is constructed from the $1,921$ China Train images, the 56 cross-split pHash pairs apply identically to the source data of the reference bank.
  - The 56 pairs connect **46 training images to 24 validation images** across **16 connected components**.
  - Because RDD2022 does not publish UAV flight trajectory logs, timestamps, or sequence IDs, sequence/flight independence is marked **`UNVERIFIED`**.
  - **Sensitivity Analysis ([`artifacts/phase1b/group_sensitive_metrics.csv`](../../artifacts/phase1b/group_sensitive_metrics.csv))**:
    - Full China Val ($N=480$): Precision $= 0.6568$, Recall $= 0.7736$, $\text{F1} = 0.7104$, Matched IoU $= 0.8007$.
    - Excluding 24 connected images ($N=456$): Precision $= 0.6516$, Recall $= 0.7614$, $\text{F1} = 0.7022$, Matched IoU $= 0.8008$.
    - Delta: $\Delta\text{F1} = -0.0082$ ($-1.15\%$), and mean matched IoU is virtually identical ($\Delta = +0.0001$).
- **Scientific Impact**: **Unchanged**. The presence of suspected burst-frame pairs does not meaningfully distort or artificially inflate in-domain detector performance.

---

### 3.4 Correction 4: Reconciling Domain Thresholds ($p_{95}, p_{99}$)
- **Old Claim**: Canonical $p_{95} = 0.3804$ and $p_{99} = 0.4491$ conflicted with exact reference bank percentiles $p_{95} = 0.3503$ and $p_{99} = 0.4558$.
- **Corrected Claim**:
  - **Distance Definition**: Cosine distance ($1.0 - \mathbf{u} \cdot \mathbf{v}$) on L2-normalized 384-dimensional DINOv2 ViT-S/14 CLS embeddings.
  - **Parameters**: $k = 20$ nearest neighbors; leave-one-out (diagonal set to $-1.0$) self-neighbor exclusion; linear quantile interpolation on $N=1,921$ training embeddings (`cross_domain/results/train_domain_reference_embeddings.npz`).
  - **Divergence Origin**:
    - $p_{99} = 0.4491$ originates from `domain_reference_metadata.json` (Phase 8 initial in-memory extraction).
    - $p_{99} = 0.4558$ is the exact percentile of the frozen npz archive serialized in Phase 10.
    - $p_{95} = 0.3804$ was introduced in Phase 12 as a conservative operational warning threshold, whereas exact intra-training 95th percentile is $0.3503$.
  - **Operational Selection**: Canonical $p_{99} = \mathbf{0.4491}$ is selected as the operational threshold. It was determined **purely on familiar China training data**; zero India observations or labels were used for threshold selection.
  - Under all evaluated thresholds ($0.3503$, $0.3804$, $0.4491$, $0.4558$), India shift detection rate is **100.0%** and familiar China false warning rate remains $\le 1.46\%$.
- **Supporting Artifact**: [`artifacts/phase1b/domain_threshold_sweep.csv`](../../artifacts/phase1b/domain_threshold_sweep.csv).
- **Scientific Impact**: **Unchanged**.

---

### 3.5 Correction 5: Domain Lighting Section Consistency
- **Old Claim**: In `reports/phase1b/DOMAIN_GATE_ANALYSIS.md`, the narrative text cited mean distances of `0.1873 vs 0.1842`, while the generated table displayed `0.1771 vs 0.1912`.
- **Corrected Claim**: Narrative prose aligned exactly with the table: Low Brightness mean $= 0.1771$ ($4/240$ false warnings, $1.67\%$), High Brightness mean $= 0.1912$ ($3/240$ false warnings, $1.25\%$).
- **Supporting Artifact**: `reports/phase1b/DOMAIN_GATE_ANALYSIS.md`.
- **Scientific Impact**: **Unchanged**. Confirms that the DINOv2 domain gate is invariant to exposure differences.

---

### 3.6 Correction 6: Statistical Uncertainty (Site-Level Cluster Bootstrap & No Unverified p-Values)
- **Old Claim**: XGBoost uncertainty was computed via row-level independent bootstrap on $N=24$ test rows ($R^2$ 95% CI: `[0.5866, 0.9135]`), and the report claimed "error reduction is statistically significant ($p < 10^{-6}$)".
- **Corrected Claim**:
  - **Site-Level Cluster Bootstrap**: Resampling the 6 held-out test highway sections with replacement ($B = 2,000$, seed $= 42$) yields $R^2$ 95% CI of **`[0.2120, 0.8943]`**, MAE 95% CI of **`[0.0703, 0.1159]`**, and RMSE 95% CI of **`[0.0807, 0.1406]`**.
  - **Removed $p < 10^{-6}$**: The unverified p-value claim was removed. Risk-coverage uncertainty is reported via non-parametric 95% bootstrap intervals with re-ranking across complete frames ($N=480$).
  - **Missing Clustering Metadata**: Explicitly noted in all reports that fine-grained UAV flight clustering metadata is unavailable in RDD2022.
- **Supporting Artifact**: [`artifacts/phase1b/confidence_intervals.csv`](../../artifacts/phase1b/confidence_intervals.csv).
- **Scientific Impact**: **Changed**. Site-level clustering reveals much wider, more scientifically honest uncertainty bounds on small sample test sites ($N=6$).

---

### 3.7 Correction 7: Forecasting Interpretation (Persistence Dominance, Horizons & Feature Importance)
- **Old Claim**: XGBoost was highlighted as the preferred model with "30–90 day forecasting horizons", claimed counterfactual capabilities, and current_severity importance was cited as $92.6\%$.
- **Corrected Claim**:
  - **Baselines Outperform XGBoost**: On the held-out test set ($N=24$), naive **Persistence** ($\hat{y}_{t_2} = y_{t_1}$, $R^2 = 0.8551$, $\text{MAE} = 0.0754$) and **OLS Linear Regression** ($R^2 = 0.8441$, $\text{MAE} = 0.0832$) strictly outperform **XGBoost** ($R^2 = 0.8055$, $\text{MAE} = 0.0924$).
  - **Paired Per-Site Breakdown**: Persistence achieves lower MAE than XGBoost on **4 out of 6 test sites**.
  - **Horizon Reality**: Observed LTPP test intervals span **240 to 720 days** (mean **447.5 days**, roughly 345–650 days per site). The "30–90 days" was only an arbitrary scenario evaluation setting.
  - **Feature Importance**: `current_severity` importance is **75.53%** (gain/attribute) / **48.13%** (weight) in `scenario_model_v2`, not 92.6%.
  - **Non-Causal Interpretation**: Feature importance is internal tree splitting behavior, not physical causality. No counterfactual or maintenance capabilities are claimed or validated.
- **Supporting Artifact**: [`artifacts/phase1b/forecast_baseline_comparison.csv`](../../artifacts/phase1b/forecast_baseline_comparison.csv).
- **Scientific Impact**: **Changed**. Critical negative finding preserved: naive physical persistence is superior to gradient-boosted trees on held-out LTPP highway sections.

---

### 3.8 Correction 8: Temporal Subsystem Provenance (CARLA Synthetic Captures)
- **Old Claim**: Experiment A captures were referred to as "total physical captures: 40" or real-world captures.
- **Corrected Claim**: The 40 captures across 8 sequences (`SEG_001`–`SEG_004`) are **CARLA / Unreal Engine synthetic simulator captures** generated via `env/scripts/rs_inspection_capture.py`. Simulator-configured progression is explicitly distinguished from real-world pavement deterioration.
- **Semantic Rule**: `NOT_OBSERVED != REPAIRED` is strictly enforced.
- **Supporting Artifact**: `reports/phase1b/TEMPORAL_FORECAST_AUDIT.md`.
- **Scientific Impact**: **Changed**. Simulator provenance is transparently declared.

---

### 3.9 Correction 9: Runtime Latency Protocol Reconciliation
- **Old Claim**: 2.15 ms in ablation table conflicted with 3.62 ms canonical latency.
- **Corrected Claim**: Explicit protocol separation:
  - **$2.15\text{ ms}$** ($465.3\text{ FPS}$): GPU inference-only forward pass (RTX 5060 Laptop GPU, batch=1, 512x512, CUDA Event synchronization, 20 warm-up runs).
  - **$3.62\text{ ms}$** ($276.2\text{ FPS}$): End-to-end benchmark from `benchmark_summary.json` (includes image loading/preprocessing, forward pass, and Ultralytics NMS postprocessing).
  - Real-time claims ($>30\text{ FPS}$) are strictly bounded to this tested workstation GPU. Edge deployment (Raspberry Pi 5) remains pending physical evaluation.
- **Supporting Artifact**: [`artifacts/phase1b/runtime_benchmarks.csv`](../../artifacts/phase1b/runtime_benchmarks.csv).
- **Scientific Impact**: **Unchanged**. Both metrics are verified under their stated protocols.

---

### 3.10 Correction 10: Policy Language Calibration
- **Old Claim**: Policy reports claimed "SAFETY PROVEN" and "guarantees zero unsafe leakage".
- **Corrected Claim**: Replaced with standard scientific framing: **"zero unsafe automated accepts were observed on the evaluated benchmark"**. Domain escalation is explicitly distinguished from correct pothole detection.
- **Supporting Artifact**: `reports/phase1b/POLICY_ABLATION.md`.
- **Scientific Impact**: **Unchanged**. Eliminates unwarranted universal extrapolation while maintaining exact empirical findings.

---

### 3.11 Correction 11: Controlled Corruption Frame-Level Validation
- **Old Claim**: Only aggregate condition-level rates were reported in `corruption_robustness.csv`.
- **Corrected Claim**: Full frame-level breakdown: 50 frames evaluated per condition ($1,100$ total evaluations across 22 conditions). Recorded total frames, failures, auto-accepted failures ($0$ across all conditions), human reviews, and domain escalations.
- **Supporting Artifact**: [`artifacts/phase1b/corruption_robustness.csv`](../../artifacts/phase1b/corruption_robustness.csv).
- **Scientific Impact**: **Unchanged**. Confirms that zero unsafe accepts was verified on every individual frame.

---

### 3.12 Correction 12: High-Confidence False Positive Enumeration
- **Old Claim**: Heading stated "Exactly 14 detections exhibited confidence $\ge 0.70$", but the table displayed only 10 rows.
- **Corrected Claim**: All 14 detections are fully listed with exact bounding boxes, confidence scores, and root-cause explanations in both `RELIABILITY_ANALYSIS.md` and `failure_case_taxonomy.csv`.
- **Supporting Artifact**: [`artifacts/phase1b/failure_case_taxonomy.csv`](../../artifacts/phase1b/failure_case_taxonomy.csv).
- **Scientific Impact**: **Unchanged**. Full transparency and completeness.

---

### 3.13 Correction 13: Repaired LaTeX Sequences & Typography
- **Old Claim**: Text generation converted LaTeX macros into tab characters and broken sequences (e.g., `k ext{NN}`, `	ext{IoU}`, `t ightarrow t+1`).
- **Corrected Claim**: Repaired to valid LaTeX math mode: `$k\text{NN}$`, `$\text{IoU}$`, `$t \rightarrow t+1$` across all 8 markdown reports in `reports/phase1b/`.
- **Supporting Artifact**: All markdown reports in `reports/phase1b/`.
- **Scientific Impact**: **Unchanged**. Professional rendering and typography restored.

---

## 4. Deliverable Verification Checklist

| Deliverable File | Status | Description |
|---|:---:|---|
| [`reports/phase1b/PHASE1B_CORRECTION_REPORT.md`](../../reports/phase1b/PHASE1B_CORRECTION_REPORT.md) | **CREATED** | This comprehensive master correction document |
| [`artifacts/phase1b/correction_reconciliation.csv`](../../artifacts/phase1b/correction_reconciliation.csv) | **CREATED** | Machine-readable table tracking old claim, corrected claim, artifact, code, reason, and scientific impact |
| [`artifacts/phase1b/group_sensitive_metrics.csv`](../../artifacts/phase1b/group_sensitive_metrics.csv) | **CREATED** | China YOLO performance evaluated on full, clean (excluding 24 connected frames), and suspected subsets |
| [`artifacts/phase1b/forecast_baseline_comparison.csv`](../../artifacts/phase1b/forecast_baseline_comparison.csv) | **CREATED** | Persistence, OLS, XGBoost, and Null comparison overall and per-site with observed horizons |
| [`artifacts/phase1b/corrected_canonical_metrics.json`](../../artifacts/phase1b/corrected_canonical_metrics.json) | **CREATED** | Master audited JSON incorporating all reconciled canonical figures |

---

## 5. Audit Conclusion

Phase 1B-R is **COMPLETE**. All 13 corrections have been executed, validated against raw records, and synchronized across reports, artifacts, and automated test suites. All negative results are strictly preserved. The repository remains on branch `phase1b-robustness-validation` without merging or pushing to `main`.
