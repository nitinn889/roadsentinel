# RoadSentinel Government Dashboard: Audited Metrics Integration Progress Report

**Date**: September 11, 2026  
**Target Environment**: RoadSentinel Examiner-Facing Research Dashboard (`dashboard/app.py`, Streamlit)  
**Status**: Fully Integrated, Verified, and Tested (31/31 automated tests passing, all 12 pages rendered cleanly with zero exceptions)

---

## 1. Executive Summary & Objective

The RoadSentinel Government Dashboard has been upgraded from legacy static estimates to a strictly audited, machine-readable metrics architecture. All perception benchmarks (Phase 1B), macro-domain gating (DINOv2 ViT-S/14), perception reliability & selective classification, safety policy ablation routing, synthetic temporal monitoring (CARLA/Unreal Engine), and Phase 1C long-term deterioration forecasting (FHWA LTPP) are now dynamically loaded from audited CSV and JSON artifacts.

### Key Architectural Tenets Enforced:
1. **Zero Hardcoded Research Metrics**: Every headline card, comparison table, confidence interval, and plot point is loaded from machine-readable files via a centralized cached loader layer (`dashboard/data_loader.py`).
2. **Strict Scientific Phrasing & Caveats**:
   - UAV vs dashcam comparison is explicitly marked as confounded by viewpoint and sensor geometry.
   - DINOv2 domain separation is restricted to the evaluated China-UAV vs India-dashcam benchmark; universal OOD claims are forbidden.
   - "Zero unsafe automated accepts were observed on the evaluated India benchmark" is contextualized with "Escalation is safe routing, not successful defect detection."
   - CARLA temporal tracking is prominently labeled as "CARLA/Unreal Synthetic Temporal Evaluation", demonstrating temporal-system functionality, not physical pavement deterioration, enforcing the rule `NOT_OBSERVED != REPAIRED`.
   - Phase 1C paired site-clustered 95% CI `[-0.0106, +0.0043]` is displayed with the mandatory interpretation: *"The interval crosses zero; universal statistical superiority over persistence is not established."*
   - Rapid 30–90-day forecasting is explicitly flagged as unvalidated due to the multi-year LTPP survey cadence (horizons: 31–728 days, mean 429.5 days).
3. **No Fabricated Data**: Missing physical testbed benchmarks (Raspberry Pi 5 on-device profiling) are explicitly tagged as `Pending physical profiling` and display `"Not yet measured"` with the expected artifact path.
4. **Historical Discrepancy Reconciliation**: The obsolete historical reliability figures `8.85%` and `2.08%` (which were derived from fixed confidence thresholds $p \ge 0.7629$ and $p \ge 0.9850$) are explicitly disassociated from pure percentile selective rejection. The audited pure percentile rejection failure rates under strict Target T1 detection quality are displayed: **9.38% at 80% coverage** and **6.25% at 50% coverage**.

---

## 2. Files Changed

| File Path | Description of Changes |
|---|---|
| `dashboard/data_loader.py` | Added cached loaders for Phase 1B and Phase 1C artifacts: `load_canonical_phase1b_metrics`, `load_risk_coverage_table`, `load_policy_ablation_table`, `load_domain_threshold_sweep`, `load_runtime_benchmarks`, `load_group_sensitive_metrics`, `load_phase1c_*` (model comparison, forecast skill, OOF predictions, feasibility summary, decision, paired site metrics), `load_per_sample_domain_distances`, `load_all_temporal_events_summary`, and `load_headline_result_cards`. Added generic `load_csv_artifact` and `load_json_artifact` handlers. |
| `dashboard/assets/style.css` | Added status pill classes (`.status-pill.measured`, `.status-pill.derived`) for visual transparency. |
| `dashboard/app.py` | Updated sidebar live pipeline status indicators to reflect audited Phase 1B/1C benchmarks and verified execution states. |
| `dashboard/components/overview.py` | Replaced legacy hardcoded metric cards with dynamic headline cards from `load_headline_result_cards()`, added all 10 mandatory limitation callouts, and integrated audited status indicators. |
| `dashboard/components/benchmark_view.py` | Updated perception benchmark tables to dynamically read from `artifacts/phase1b/group_sensitive_metrics.csv` and show duplicate-filtered China F1 (0.7022) alongside baseline (0.7104). |
| `dashboard/components/cross_domain_view.py` | Added detector comparison (China UAV vs India Dashcam), matched IoU metrics (0.8007 vs 0.7678), duplicate-filtered China F1 (0.7022), India T1 failures (293/300), and dark-themed matplotlib DINOv2 cosine distance distribution chart with operational threshold line (0.4491) and separation margin (+0.1158). Added viewpoint confounding warning. |
| `dashboard/components/reliability_view.py` | Integrated Risk–Coverage curve (80% cov → 9.38%, 50% cov → 6.25%), calibration/reliability diagram (AUROC 0.7741, Brier 0.1921, ECE 0.1075), confidence histogram, and historical 8.85%/2.08% reconciliation callout. |
| `dashboard/components/decision_view.py` | Replaced legacy mockups with audited Phase 1B policy ablation destinations stacked chart (automated accept, human review, domain escalation, unsafe accepts) across India, China, and Pooled strata. Added mandatory safety routing caution and "Zero unsafe automated accepts" statement. |
| `dashboard/components/temporal.py` | Labeled prominently as "CARLA/Unreal Synthetic Temporal Evaluation". Displayed summary metrics (40 captures, 8 sequences, 48 tracks, 22 persistent tracks, 45.83% persistence, 33 transitions, 14 area increases, 19 area decreases). Added persistence composition and transition direction plots, and enforced `NOT_OBSERVED != REPAIRED`. |
| `dashboard/components/forecast.py` | Fully integrated Phase 1C forecasting results: M0 vs M3 vs M4 comparison table, paired site-clustered 95% CI `[-0.0106, +0.0043]`, horizon and history subgroups (<365d: -3.77%, 365-550d: +9.12%, >550d: +10.15%, >=2 prior: +9.18%), out-of-fold empirical prediction behavior plots (observed vs predicted severity & delta, error vs horizon, error vs history depth), operational forecast router, and FHWA LTPP dataset provenance. |
| `dashboard/components/edge_deployment.py` | Integrated standardized runtime benchmark table identifying hardware host, processor, precision (FP32), resolution, batch size, warmup runs, iteration count, and latency (YOLO: 2.15 ms / 465.3 FPS; Staged: 28.41 ms / 35.2 FPS; Full Policy: 291.65 ms / 3.43 FPS). Marked Raspberry Pi 5 profiling as pending physical testbed. |
| `dashboard/test_dashboard_v2.py` | Expanded automated test suite to 31 comprehensive test cases covering headline cards, loaders, Phase 1B/1C numerical accuracy, missing artifact handling, and required caveat text assertions. |

---

## 3. Machine-Readable Artifacts Consumed by Component

| Dashboard Component | Source Artifact Paths | Loaded Entities & Metrics |
|---|---|---|
| **Overview (`overview.py`)** | `artifacts/phase1b/canonical_phase1b_metrics.json`<br>`cross_domain/tables/table_in_vs_cross_domain.csv`<br>`artifacts/phase1b/domain_threshold_sweep.csv`<br>`artifacts/phase1b/risk_coverage.csv`<br>`artifacts/phase1c/forecast_skill.json` | 7 Headline result cards (China UAV F1, India Dashcam F1, India Escalation, Familiar China Warning Rate, Reliability Accepted Failure Rate, M4 Forecast Skill, Raspberry Pi Status) + 10 Mandatory Limitations |
| **Detector Comparison (`cross_domain_view.py`)** | `cross_domain/tables/table_in_vs_cross_domain.csv`<br>`artifacts/phase1b/group_sensitive_metrics.csv`<br>`artifacts/phase1b/policy_ablation.csv` | China Precision 0.6568, Recall 0.7736, F1 0.7104; India Precision 0.0964, Recall 0.0123, F1 0.0218; Matched IoU 0.8007 vs 0.7678; Duplicate-filtered China F1 0.7022; India T1 failures 293/300 |
| **DINOv2 Domain Gate (`cross_domain_view.py`)** | `reliability/results/reliability_predictions.csv`<br>`cross_domain/cross_domain_reliability_dataset.csv`<br>`artifacts/phase1b/domain_threshold_sweep.csv` | 480 China distances, 300 India distances; AUROC 1.0000; Margin +0.1158; Threshold p99=0.4491; 300/300 India escalations; 7/480 China warnings |
| **Reliability & Calibration (`reliability_view.py`)** | `artifacts/phase1b/risk_coverage.csv`<br>`integration/research_strengthening/YOLO_CONFIDENCE_FAILURE_ANALYSIS.csv`<br>`reliability/results/reliability_predictions.csv` | Risk–coverage curve: 80% cov → 9.38% fail, 50% cov → 6.25% fail; Detection AUROC 0.7741; Brier 0.1921; ECE 0.1075; Confidence bins & per-image confidence distribution |
| **Policy Ablation (`decision_view.py`)** | `artifacts/phase1b/policy_ablation.csv` | Destinations across 5 architectures for India, China, and Pooled strata: automated accept, human review, domain escalation, unsafe automated accepts (0 on India for gated/full policies), downstream failures prevented |
| **Temporal Monitoring (`temporal.py`)** | `integration/experiment_a/temporal/tables/table_tracking_statistics.csv`<br>`integration/experiment_a/temporal/sequences/*/events.json` | 40 captures, 8 sequences, 48 unique tracks, 22 persistent tracks (45.83%), 33 transitions (14 increases, 19 decreases), 34 unobserved, 48 new defects |
| **Forecasting Comparison (`forecast.py`)** | `artifacts/phase1c/model_comparison.csv`<br>`artifacts/phase1c/forecast_skill.json`<br>`artifacts/phase1c/out_of_fold_predictions.csv`<br>`artifacts/phase1c/feasibility_audit_summary.json`<br>`artifacts/phase1c/final_model_decision.json` | M0 (MAE 0.0582, RMSE 0.0815, R² 0.9125); M3 (MAE 0.0563, Skill +3.26%); M4 (MAE 0.0551, RMSE 0.0766, R² 0.9229, Skill +5.27%); Paired 95% CI [-0.0106, +0.0043]; Horizons (<365d: -3.77% N=43, 365-550d: +9.12% N=35, >550d: +10.15% N=35, >=2 prior: +9.18% N=78); 113 OOF predictions; 113 LTPP pairs, 23 sections, 6 states |
| **Runtime Comparison (`edge_deployment.py`)** | `artifacts/phase1b/runtime_benchmarks.csv`<br>`artifacts/phase1b/policy_ablation.csv` | YOLO-only: 2.15 ms (465.3 FPS); Staged Pipeline: 28.41 ms (35.2 FPS); Full Policy: 291.65 ms (3.43 FPS); Raspberry Pi: Pending physical profiling |

---

## 4. Validation Values Loaded from Artifacts

| Evaluation Metric | Audited Target Value | Source Artifact | Value Loaded by UI | Verification Status |
|---|---|---|---|---|
| **YOLO China UAV F1** | `0.7104` | `canonical_phase1b_metrics.json` | `0.7104` | Verified Match |
| **YOLO India Dashcam F1** | `0.0218` | `table_in_vs_cross_domain.csv` | `0.0218` | Verified Match |
| **India Domain Escalation** | `300/300, 100%` | `domain_threshold_sweep.csv` | `300/300, 100%` | Verified Match |
| **Familiar China Warning Rate** | `7/480, ~1.46%` | `domain_threshold_sweep.csv` | `7/480, ~1.46%` | Verified Match |
| **Reliability Failure (80% cov)** | `9.38%` | `risk_coverage.csv` (Target T1) | `9.38%` | Verified Match |
| **Reliability Failure (50% cov)** | `6.25%` | `risk_coverage.csv` (Target T1) | `6.25%` | Verified Match |
| **M4 Forecast MAE Skill** | `+5.27%` (point estimate) | `forecast_skill.json` | `+5.27%` | Verified Match |
| **Raspberry Pi Profiling** | `Pending physical profiling` | `LIMITATIONS.md` | `Pending physical profiling` | Verified Match |
| **Mean Matched IoU (China)** | `0.8007` | `table_in_vs_cross_domain.csv` | `0.8007` | Verified Match |
| **Mean Matched IoU (India)** | `0.7678` | `table_in_vs_cross_domain.csv` | `0.7678` | Verified Match |
| **Duplicate-Filtered China F1** | `0.7022` | `group_sensitive_metrics.csv` | `0.7022` | Verified Match |
| **India T1 Baseline Failures** | `293/300` | `policy_ablation.csv` | `293/300` | Verified Match |
| **DINOv2 Gate AUROC** | `1.0000` | `domain_threshold_sweep.csv` | `1.0000` | Verified Match |
| **DINOv2 Separation Margin** | `+0.1158` | `canonical_phase1b_metrics.json` | `+0.1158` | Verified Match |
| **Detection-Correctness AUROC** | `0.7741` | `YOLO_CONFIDENCE_FAILURE_ANALYSIS.csv` | `0.7741` | Verified Match |
| **Brier Score** | `0.1921` | `YOLO_CONFIDENCE_FAILURE_ANALYSIS.csv` | `0.1921` | Verified Match |
| **Expected Calibration Error (ECE)**| `0.1075` | `YOLO_CONFIDENCE_FAILURE_ANALYSIS.csv` | `0.1075` | Verified Match |
| **India Policy Unsafe Accepts** | `0` (Zero) | `policy_ablation.csv` | `0` | Verified Match |
| **CARLA Synthetic Captures** | `40` | `table_tracking_statistics.csv` | `40` | Verified Match |
| **CARLA Synthetic Sequences** | `8` | `table_tracking_statistics.csv` | `8` | Verified Match |
| **CARLA Distress Tracks** | `48` | `table_tracking_statistics.csv` | `48` | Verified Match |
| **CARLA Persistent Tracks** | `22 (45.83%)` | `table_tracking_statistics.csv` | `22 (45.83%)` | Verified Match |
| **CARLA Matched Transitions** | `33` | `table_tracking_statistics.csv` | `33` | Verified Match |
| **CARLA Area Increases / Decreases**| `14 / 19` | `table_tracking_statistics.csv` | `14 / 19` | Verified Match |
| **M0 Persistence (MAE / RMSE / R²)**| `0.0582 / 0.0815 / 0.9125` | `model_comparison.csv` | `0.0582 / 0.0815 / 0.9125` | Verified Match |
| **M3 Temporal XGBoost (MAE / Skill)**| `0.0563 / +3.26%` | `model_comparison.csv` | `0.0563 / +3.26%` | Verified Match |
| **M4 Combined XGBoost (MAE / RMSE / R² / Skill)**| `0.0551 / 0.0766 / 0.9229 / +5.27%` | `model_comparison.csv` | `0.0551 / 0.0766 / 0.9229 / +5.27%` | Verified Match |
| **Paired 95% Clustered CI** | `[-0.0106, +0.0043]` | `forecast_skill.json` | `[-0.0106, +0.0043]` | Verified Match |
| **Horizon Subgroups** | `<365d: -3.77% (N=43)`<br>`365-550d: +9.12% (N=35)`<br>`>550d: +10.15% (N=35)`<br>`>=2 prior: +9.18% (N=78)` | `forecast_skill.json` | `<365d: -3.77% (N=43)`<br>`365-550d: +9.12% (N=35)`<br>`>550d: +10.15% (N=35)`<br>`>=2 prior: +9.18% (N=78)` | Verified Match |
| **LTPP Dataset Scale** | `113 pairs, 23 sections, 6 states, 31-728d` | `feasibility_audit_summary.json` | `113 pairs, 23 sections, 6 states, 31-728d` | Verified Match |

---

## 5. Automated Tests and Build Results

### Command 1: Pytest Suite Execution
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest dashboard/test_dashboard_v2.py -v
```
**Result**:
```
============================= test session starts ==============================
collected 31 items

dashboard/test_dashboard_v2.py::TestDashboardV2DataLoaders::test_all_40_images_accessible PASSED [  3%]
dashboard/test_dashboard_v2.py::TestDashboardV2DataLoaders::test_canonical_metrics_loaded PASSED [  6%]
dashboard/test_dashboard_v2.py::TestDashboardV2DataLoaders::test_decisions_table_loaded PASSED [  9%]
dashboard/test_dashboard_v2.py::TestDashboardV2DataLoaders::test_goal1_forecasts_loaded PASSED [ 12%]
dashboard/test_dashboard_v2.py::TestDashboardV2DataLoaders::test_primary_results_loaded PASSED [ 16%]
dashboard/test_dashboard_v2.py::TestAuditedHeadlineResultCards::test_all_seven_cards_present PASSED [ 19%]
dashboard/test_dashboard_v2.py::TestAuditedHeadlineResultCards::test_china_warning_rate PASSED [ 22%]
dashboard/test_dashboard_v2.py::TestAuditedHeadlineResultCards::test_india_domain_escalation PASSED [ 25%]
dashboard/test_dashboard_v2.py::TestAuditedHeadlineResultCards::test_m4_forecast_mae_skill PASSED [ 29%]
dashboard/test_dashboard_v2.py::TestAuditedHeadlineResultCards::test_raspberry_pi_pending PASSED [ 32%]
dashboard/test_dashboard_v2.py::TestAuditedHeadlineResultCards::test_reliability_accepted_failure_rates PASSED [ 35%]
dashboard/test_dashboard_v2.py::TestAuditedHeadlineResultCards::test_yolo_china_f1 PASSED [ 38%]
dashboard/test_dashboard_v2.py::TestAuditedHeadlineResultCards::test_yolo_india_f1 PASSED [ 41%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1BMetrics::test_carla_synthetic_temporal_summary PASSED [ 45%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1BMetrics::test_per_sample_domain_distances PASSED [ 48%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1BMetrics::test_policy_ablation_zero_unsafe_accepts_india PASSED [ 51%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1BMetrics::test_risk_coverage_values PASSED [ 54%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1CForecasting::test_forecast_dataset_metadata PASSED [ 58%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1CForecasting::test_horizon_subgroups PASSED [ 61%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1CForecasting::test_model_comparison_values PASSED [ 64%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1CForecasting::test_out_of_fold_predictions_integrity PASSED [ 67%]
dashboard/test_dashboard_v2.py::TestAuditedPhase1CForecasting::test_paired_confidence_interval_crosses_zero PASSED [ 70%]
dashboard/test_dashboard_v2.py::TestDashboardV2ComponentImports::test_component_imports PASSED [ 74%]
dashboard/test_dashboard_v2.py::TestGracefulMissingArtifactHandling::test_missing_csv_returns_empty_dataframe PASSED [ 77%]
dashboard/test_dashboard_v2.py::TestGracefulMissingArtifactHandling::test_missing_json_returns_empty_dict PASSED [ 80%]
dashboard/test_dashboard_v2.py::TestGracefulMissingArtifactHandling::test_raspberry_pi_status_is_pending_not_fabricated PASSED [ 83%]
dashboard/test_dashboard_v2.py::TestCaveatsAndLabels::test_domain_gate_scope_caveat PASSED [ 87%]
dashboard/test_dashboard_v2.py::TestCaveatsAndLabels::test_forecasting_zero_crossing_and_horizon_caveat PASSED [ 90%]
dashboard/test_dashboard_v2.py::TestCaveatsAndLabels::test_policy_zero_unsafe_accepts_india_statement PASSED [ 93%]
dashboard/test_dashboard_v2.py::TestCaveatsAndLabels::test_temporal_semantic_rule_and_label PASSED [ 96%]
dashboard/test_dashboard_v2.py::TestCaveatsAndLabels::test_viewpoint_confounding_caveat PASSED [100%]

============================== 31 passed in 1.15s ==============================
```

### Command 2: Streamlit AppTest Execution Across All 12 Pages
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -c '
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("dashboard/app.py", default_timeout=30)
at.run()
pages = at.sidebar.radio[0].options
for idx, page in enumerate(pages):
    at.sidebar.radio[0].set_value(page)
    at.run()
    assert not at.exception, f"Failed on {page}: {at.exception[0].value}"
    print(f"  [✓] Page {idx+1}: {page} - PASSED")
print("\n🎉 All 12 pages rendered cleanly with 0 exceptions!")
'
```
**Result**:
```
Testing 12 pages:
  [✓] Page 1: 1. System Overview - PASSED
  [✓] Page 2: 2. Single-Image Assessment - PASSED
  [✓] Page 3: 3. Domain & Reliability - PASSED
  [✓] Page 4: 4. XGBoost Future Forecast - PASSED
  [✓] Page 5: 5. Temporal Change Analysis - PASSED
  [✓] Page 6: 6. YOLO vs DINO/SAM Research - PASSED
  [✓] Page 7: 7. Cross-Domain Generalization - PASSED
  [✓] Page 8: 8. Decision & Inspection Priority - PASSED
  [✓] Page 9: 9. Failure Mode Taxonomy - PASSED
  [✓] Page 10: 10. Research Methodology - PASSED
  [✓] Page 11: 11. Controlled Experiment B [PLANNED] - PASSED
  [✓] Page 12: 12. Edge Deployment [PENDING] - PASSED

🎉 All 12 pages rendered cleanly with 0 exceptions!
```

---

## 6. Visual Verification Summary & Browser Diagnostics

1. **Streamlit Local Server**: Successfully launched and bound to `http://localhost:8501`. Verified responsive HTTP `200 OK` responses via curl and socket polling.
2. **Browser Subagent Status**:
   - The browser subagent encountered an environment-level Playwright driver CDN 404 error during automatic browser context initialization (`https://playwright.azureedge.net/builds/driver/playwright-1.57.0-linux.zip` returned 404).
   - As required by the system instructions, full DOM and execution verification was conducted programmatically using Streamlit's official testing framework (`streamlit.testing.v1.AppTest`), verifying that:
     - Navigation radio selections switch active component functions.
     - Single-image image paths, bounding boxes, and metadata sidecars load without errors.
     - Matplotlib figures (dark-themed distribution plots, stacked bar charts, risk-coverage curves, calibration diagrams, and OOF scatter/box plots) compile cleanly without deprecated or unsupported keyword arguments.
     - Tables serialize cleanly to PyArrow without type conflicts.

---

## 7. Discrepancy Analysis: Audited Target Values vs Source Artifacts

1. **Selective Prediction Failure Rates: Historical 8.85% & 2.08% vs Audited 9.38% & 6.25%**:
   - *Investigation*: The historical figures of `8.85%` and `2.08%` in early RoadSentinel research documents were calculated using **fixed model confidence thresholds** ($p \ge 0.7629$ and $p \ge 0.9850$).
   - *Audited Reality*: In `artifacts/phase1b/risk_coverage.csv`, under formal **pure percentile coverage rejection** on strict detection quality Target T1 ($F_1 \ge 0.50$):
     - At **80% coverage** (20% rejected), the accepted failure rate is **9.38%** (36 accepted failures / 384 accepted samples).
     - At **50% coverage** (50% rejected), the accepted failure rate is **6.25%** (15 accepted failures / 240 accepted samples).
   - *Resolution*: The dashboard prominently explains this distinction in `reliability_view.py` and headline cards, eliminating the misleading association of `8.85%` and `2.08%` with pure percentile coverage.

2. **YOLO Cross-Domain F1 Score: 0.0226 vs 0.0218**:
   - *Investigation*: Preliminary reports noted an approximate India F1 of 0.0226.
   - *Audited Reality*: `cross_domain/tables/table_in_vs_cross_domain.csv` and `artifacts/phase1b/canonical_phase1b_metrics.json` record the exact audited precision as **0.0964**, recall as **0.0123**, yielding an $F_1$ score of **0.0218** (an absolute drop of **-0.6886**, relative collapse of **-96.94%**).
   - *Resolution*: Audited value `0.0218` is strictly loaded.

3. **M4 Forecast Statistical Significance vs Persistence Baseline**:
   - *Investigation*: Point estimate skill for M4 Combined XGBoost is `+5.27%` over persistence ($MAE = 0.0551$ vs $0.0582$).
   - *Audited Reality*: In `artifacts/phase1c/forecast_skill.json`, the paired site-clustered 95% confidence interval for M4 minus persistence MAE is `[-0.0106, +0.0043]`.
   - *Resolution*: The dashboard displays this interval prominently and states: *"The interval crosses zero; universal statistical superiority over persistence is not established."*

---

## 8. Missing Artifacts & Pending Evaluations Inventory

1. **Raspberry Pi 5 Physical Profiling**:
   - *Status*: `Pending physical profiling`
   - *Expected Source*: `artifacts/edge/raspberry_pi5_profiling.json`
   - *Display Action*: Marked as "Not yet measured", displaying target specifications (Cortex-A76 quad-core @ 2.4 GHz, 8GB LPDDR4X) and pre-allocated evaluation schema without fabricated numbers.
2. **Rapid 30–90-Day High-Frequency Forecasting**:
   - *Status*: `Unvalidated / Not Supported`
   - *Reason*: Historical FHWA LTPP pavement inspection intervals range from 31 to 728 days with a mean of 429.5 days. Projections at 30, 60, and 90 days are synthetic scenario extrapolations, not empirically validated short-cadence survey intervals.
3. **Operational 365-Day Routing Boundary**:
   - *Status*: `Pending Holdout Validation Confirmation`
   - *Display Action*: Labeled as pending confirmation of whether the 365-day threshold was predefined a priori or discovered post hoc during exploratory data analysis.
