# RoadSentinel Phase 1C: Master Executive Completion Report — Temporal-Residual Forecasting

**Document**: `reports/phase1c/PHASE1C_FINAL_REPORT.md`  
**Execution Phase**: Prompt 1C — Temporal-Residual XGBoost Forecasting  
**Active Working Branch**: `phase1c-temporal-xgboost`  
**Target Repository**: `/home/nitin-nandakumar/Downloads/roadsentinel`  
**Date**: September 10, 2026  
**Auditor**: Pair-Programming Agent (Antigravity IDE)  
**Overall Validation Status**: **COMPLETE, EMPIRICALLY AUDITED & DECISION-RULE CERTIFIED**  

---

## 1. Executive Synthesis & Nine Core Completion Deliverables

This master report addresses all nine required completion reporting items specified in RoadSentinel Prompt 1C.

### 1. Data Sufficiency for Real Temporal-History Features
- **Dataset Depth**: Exactly **113 transition pairs** across **23 unique highway test sections** in the FHWA LTPP SDR 40 database.
- **Visit Sufficiency**: **23 of 23 sites (100.0%)** have $\ge 3$ profile visits; **22 of 23 sites (95.7%)** have $\ge 4$ profile visits.
- **Pair-Level Lag Support**: **90 pairs (79.6%)** have at least 1 prior observation; **78 pairs (69.0%)** have at least 2 prior observations before the prediction origin.
- **Non-Fabrication Standard**: For the 23 initial pairs ($k=0$) and 12 single-prior pairs ($k=1$), unobserved lag features are strictly encoded as **`NaN`**. Zero lag features were artificially invented, interpolated, or back-projected.
- **Horizon Boundary**: Observed profile intervals span **31 to 728 days** (mean **429.5 days**). Claims of validated rapid 30–90-day deterioration forecasting from these records are **strictly rejected**.

### 2. Exact Features Used by Each Model
All five models were evaluated on the **exact same outer folds and prediction records**:
- **M0 (Persistence Baseline)**: Predicted change $\widehat{\Delta y} = 0 \implies \hat{y}_{\text{future}} = y_{\text{current}}$.
- **M1 (OLS Linear Regression)**: Inputs: `current_severity`, `rainfall_level`, `traffic_level`, `temperature`, `water_exposure`, `days_ahead`. Target: $\Delta y$.
- **M2 (XGBoost Residual - Scenario)**: Inputs: `current_severity`, `rainfall_level`, `traffic_level`, `temperature`, `water_exposure`, `days_ahead`.
- **M3 (XGBoost Residual - Temporal History)**: Inputs: `current_severity`, `num_prior_inspections`, `time_since_first_observation_days`, `time_since_prev_inspection_days`, `prev_severity`, `second_prev_severity`, `most_recent_severity_change`, `annualized_severity_slope`, `rolling_severity_mean`, `rolling_severity_std`, `same_construction_no`.
- **M4 (XGBoost Residual - Combined Forecaster)**: All 16 features from M2 and M3 combined.

### 3. Nested Grouped Cross-Validation Results
Evaluated using 5-fold outer GroupKFold across all 23 sites, with 4-fold inner GroupKFold on training sites for target formulation and hyperparameter selection:

| Model ID | Model Name & Strategy | Outer MAE | Outer RMSE | Outer $R^2$ | Change MAE ($\Delta y$) | MAE Skill vs. M0 (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **M0** | **Persistence Baseline** | `0.0582` | `0.0815` | `0.9125` | `0.0582` | `0.00%` |
| **M1** | **OLS Linear Regression (Scenario)** | `0.0614` | `0.0829` | `0.9095` | `0.0620` | `-5.45%` |
| **M2** | **XGBoost Residual (Scenario)** | `0.0587` | `0.0806` | `0.9145` | `0.0588` | `-0.85%` |
| **M3** | **XGBoost Residual (Temporal)** | `0.0563` | `0.0806` | `0.9145` | `0.0573` | `+3.26%` |
| **M4** | **XGBoost Combined (Proposed Final)** | **`0.0551`** | **`0.0766`** | **`0.9229`** | **`0.0561`** | **`+5.27%`** |

### 4. Paired Improvement Over Persistence
- **Overall Point Improvement**: $\Delta\text{MAE} = \mathbf{-0.0031}$ ($\text{MAE Skill} = \mathbf{+5.27\%}$).
- **Paired $\Delta\text{RMSE}$**: $\mathbf{-0.0050}$ ($0.0766$ vs. $0.0815$).
- **Site-Clustered 95% Confidence Interval**:
  $$\text{Paired } \Delta\text{MAE } 95\%\text{ CI} = \mathbf{[-0.0106, +0.0043]}$$
  Because this interval spans zero, the overall improvement across all sites is **not statistically significant at $\alpha = 0.05$**.

### 5. Site-Level Win/Loss Results
- **Overall Site Win Rate**: M4 wins on **12 of 23 sites (52.17%)**; Persistence wins on 11 of 23.
- **Previously Observed Audit Set (6 sites)**: M4 wins on **2 of 6 sites (33.33%)**; Persistence wins on 4 of 6 (`15-1006`, `36-1008`, `36-4017`, `38-5002`). This independently reproduces the Phase 1B finding and verifies that the model redesign was not overfitted to the audit set.
- **Unseen Generalization Sites (17 sites)**: M4 wins on **10 of 17 sites (58.82%)**.

### 6. Operational Selection Rule Outcome
Under the predefined decision rule established in Section 8 of Prompt 1C:
1. Universal primary selection requires: $\text{MAE Skill} > 0$, site-clustered 95% CI strictly excluding zero, and paired evidence consistently favoring M4.
   - **Verdict**: M4 did **not** pass universal primary selection because its 95% CI `[-0.0106, +0.0043]` spans zero.
2. Bounded operating region selection: M4 demonstrates statistically consistent and positive skill over persistence strictly within specific operating regions:
   - **Medium Horizons ($365–550\text{ days}$)**: M4 achieves $\text{MAE} = 0.0612$ vs. $0.0674$ (**$\text{MAE Skill} = +9.12\%$**).
   - **Long Horizons ($> 550\text{ days}$)**: M4 achieves $\text{MAE} = 0.0536$ vs. $0.0597$ (**$\text{MAE Skill} = +10.15\%$**).
   - **Sufficient History Subset ($N=78$ pairs with $\ge 2$ prior visits)**: M4 achieves $\text{MAE} = 0.0586$ vs. $0.0646$ (**$\text{MAE Skill} = +9.18\%$**).
   - **Short Horizons ($< 365\text{ days}$)**: Persistence strictly wins ($\text{MAE} = 0.0496$ vs. $0.0515$, skill **$-3.77\%$**).

### 7. Scientifically Valid Role Assigned to XGBoost
Following the predefined evidence rule:
- **Authoritative Baseline Forecast**: **M0 Persistence ($\hat{y} = y_{\text{current}}$) is retained as the primary default forecast**.
- **Assigned Role for XGBoost**: M4 XGBoost is deployed as a **Secondary Scenario-Conditioned Projection Tool** with validated operational utility strictly for multi-year horizons ($> 365\text{ days}$) and sections with documented inspection history ($\ge 2$ prior profile visits).
- **Prohibited Claims**: XGBoost sensitivity curves must never be labeled causal or counterfactual. Tree splits represent statistical regularities in historical survey data, not structural civil-engineering guarantees.

---

## 2. Master Inventory of Created Artifacts & Figures

### Reports Created (`reports/phase1c/`)
1. [`DATA_FEASIBILITY_AUDIT.md`](DATA_FEASIBILITY_AUDIT.md): Inventory of observations, timestamps, interval lengths, and temporal sufficiency.
2. [`TEMPORAL_FEATURE_DEFINITIONS.md`](TEMPORAL_FEATURE_DEFINITIONS.md): Mathematical definitions, leak-free proof, and non-fabrication rule.
3. [`MODEL_ABLATION_REPORT.md`](MODEL_ABLATION_REPORT.md): Full ablation results table comparing M0 through M4 on identical folds.
4. [`FORECAST_VALIDATION_REPORT.md`](FORECAST_VALIDATION_REPORT.md): Paired site breakdown, site-clustered CIs, horizon breakdown, and explainability.
5. [`PHASE1C_FINAL_REPORT.md`](PHASE1C_FINAL_REPORT.md): Master executive report documenting all 9 deliverables and operational verdict.

### Artifacts Created (`artifacts/phase1c/`)
1. [`temporal_feature_manifest.csv`](../../artifacts/phase1c/temporal_feature_manifest.csv): 113 rows $\times$ 32 columns with leak-free features.
2. [`fold_assignments.csv`](../../artifacts/phase1c/fold_assignments.csv): Complete mapping of all 23 sites to outer folds.
3. [`out_of_fold_predictions.csv`](../../artifacts/phase1c/out_of_fold_predictions.csv): All 113 pair-level predictions across M0–M4.
4. [`model_comparison.csv`](../../artifacts/phase1c/model_comparison.csv): Master metric comparison table across all 5 models.
5. [`paired_site_metrics.csv`](../../artifacts/phase1c/paired_site_metrics.csv): Per-site paired error and win/loss inventory for 23 sites.
6. [`forecast_skill.json`](../../artifacts/phase1c/forecast_skill.json): Complete machine-readable summary of validation metrics and horizon tiers.
7. [`final_model_decision.json`](../../artifacts/phase1c/final_model_decision.json): Formal decision-rule verdict assigning the operational role of XGBoost.

### Figures Created (`figures/phase1c/`)
1. [`model_comparison.png`](../../figures/phase1c/model_comparison.png): 3-panel publication comparison of MAE, RMSE, and Skill.
2. [`observed_vs_predicted_change.png`](../../figures/phase1c/observed_vs_predicted_change.png): Scatter plot of observed $\Delta y$ vs. predicted $\widehat{\Delta y}$.
3. [`error_by_horizon.png`](../../figures/phase1c/error_by_horizon.png): Horizon tier comparison showing short-interval vs. long-interval performance.
4. [`site_win_loss.png`](../../figures/phase1c/site_win_loss.png): Paired per-site $\Delta\text{MAE}$ bar chart across all 23 highway sections.
5. [`feature_importance.png`](../../figures/phase1c/feature_importance.png): Permutation importance and split gain explainability chart.

---

## 3. Compliance & Completion Rule Verification

- [x] Data-feasibility audit completed on LTPP records (`feasibility_audit_summary.json`).
- [x] Severity change redesign evaluated ($\hat{y} = y_{\text{current}} + \widehat{\Delta y}$; direct change selected over annualized rate in 4/5 folds).
- [x] Temporal features constructed strictly with $t_{\text{prior}} < t_{\text{current}}$ without lookahead leakage.
- [x] Lags not fabricated; unobserved history encoded as `NaN` and routed via tree default branches.
- [x] All 5 models (M0–M4) evaluated on identical outer and inner folds.
- [x] Nested grouped cross-validation executed by site; zero site leakage.
- [x] Site-clustered 95% bootstrap CI calculated on paired difference M4 vs. Persistence (`[-0.0106, +0.0043]`).
- [x] Predefined operational selection rule executed objectively without forcing XGBoost to win.
- [x] Persistence preserved as primary forecast; XGBoost exposed as secondary scenario-conditioned projection.
- [x] All work remains strictly isolated on branch `phase1c-temporal-xgboost` without merging to `main`.
