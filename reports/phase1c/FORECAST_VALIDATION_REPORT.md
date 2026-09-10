# RoadSentinel Phase 1C: Longitudinal Forecast Validation & Uncertainty Report

**Document**: `reports/phase1c/FORECAST_VALIDATION_REPORT.md`  
**Execution Phase**: Prompt 1C — Temporal-Residual XGBoost Forecasting  
**Validation Scheme**: 5-Fold Nested Grouped Cross-Validation (Complete Sites as Groups)  
**Supporting Artifacts**:  
- [`artifacts/phase1c/paired_site_metrics.csv`](../../artifacts/phase1c/paired_site_metrics.csv)  
- [`artifacts/phase1c/forecast_skill.json`](../../artifacts/phase1c/forecast_skill.json)  
- [`artifacts/phase1c/final_model_decision.json`](../../artifacts/phase1c/final_model_decision.json)  
**Figures**:  
- [`figures/phase1c/observed_vs_predicted_change.png`](../../figures/phase1c/observed_vs_predicted_change.png)  
- [`figures/phase1c/error_by_horizon.png`](../../figures/phase1c/error_by_horizon.png)  
- [`figures/phase1c/site_win_loss.png`](../../figures/phase1c/site_win_loss.png)  
- [`figures/phase1c/feature_importance.png`](../../figures/phase1c/feature_importance.png)  
**Date**: September 10, 2026  
**Status**: **STATISTICALLY AUDITED & EMPIRICALLY BOUNDED**  

---

## 1. Executive Summary

This report provides the detailed validation analysis, site-clustered statistical uncertainty bounds, horizon-stratified breakdowns, and explainability audits for the final RoadSentinel M4 forecasting model ($\hat{y} = y_{\text{current}} + \widehat{\Delta y}_{\text{XGB}}$) relative to the M0 physical persistence baseline.

### Core Validation Findings:
1. **Paired Point Improvement Over Persistence**: Across all 113 evaluation pairs, M4 achieves an overall error reduction of $\Delta\text{MAE} = \mathbf{-0.0031}$ ($\text{MAE} = 0.0551$ vs. $0.0582$), representing an overall **MAE Skill of $+5.27\%$**.
2. **Site-Clustered Statistical Uncertainty**: Resampling the 23 complete highway sections with replacement ($B = 2,000$ iterations, seed $= 42$) yields a 95% confidence interval for the paired difference ($\text{MAE}_{\text{M4}} - \text{MAE}_{\text{M0}}$) of:
   $$\mathbf{[-0.0106, +0.0043]}$$
   **Crucial Scientific Conclusion**: Because this interval spans zero, M4 **cannot be claimed to possess statistically significant universal superiority** over physical persistence across all highway sections.
3. **Horizon-Stratified Divergence**:
   - **Short Horizons ($< 365\text{ days}$, $N=43$)**: Physical persistence **strictly outperforms M4** ($\text{MAE} = 0.0496$ vs. $0.0515$, skill **$-3.77\%$**).
   - **Medium Horizons ($365–550\text{ days}$, $N=35$)**: M4 **outperforms persistence** ($\text{MAE} = 0.0612$ vs. $0.0674$, skill **$+9.12\%$**).
   - **Long Horizons ($> 550\text{ days}$, $N=35$)**: M4 **outperforms persistence** ($\text{MAE} = 0.0536$ vs. $0.0597$, skill **$+10.15\%$**).
4. **Data-Depth Effect ($N=78$ Subset with $\ge 2$ Prior Inspections)**:
   - When sufficient longitudinal depth exists, M4 achieves $\text{MAE} = 0.0586$ vs. persistence $0.0646$ (skill **$+9.18\%$**).

---

## 2. Paired Site Win/Loss Breakdown

The table below catalogs paired performance across all 23 highway sections, contrasting M0, M1, and M4:

| Site ID | Generalization Stratum | Pair Count | Mean Horizon (Days) | M0 Persistence MAE | M1 OLS MAE | M4 XGB MAE | Paired $\Delta\text{MAE}$ (M4 - M0) | M4 Wins? | Best Model |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `02-1001` | Unseen Generalization | 4 | 628.0 | 0.0463 | 0.0450 | **0.0381** | **-0.0082** | **YES** | **M4_XGB** |
| `02-1002` | Audit Set (Phase 1B)* | 4 | 623.5 | 0.1471 | 0.1415 | **0.1264** | **-0.0207** | **YES** | **M4_XGB** |
| `02-1004` | Unseen Generalization | 3 | 682.7 | 0.1205 | 0.1416 | **0.1065** | **-0.0140** | **YES** | **M4_XGB** |
| `02-1008` | Unseen Generalization | 2 | 559.0 | 0.0694 | 0.0827 | **0.0664** | **-0.0030** | **YES** | **M4_XGB** |
| `02-6010` | Unseen Generalization | 3 | 613.3 | 0.0264 | 0.0305 | 0.0302 | +0.0038 | NO | M0_Persistence |
| `02-9035` | Unseen Generalization | 3 | 684.7 | 0.0863 | 0.0887 | **0.0673** | **-0.0190** | **YES** | **M4_XGB** |
| `15-1003` | Unseen Generalization | 3 | 655.7 | 0.0347 | 0.0381 | 0.0371 | +0.0024 | NO | M0_Persistence |
| `15-1006` | Audit Set (Phase 1B)* | 3 | 655.7 | **0.0683** | 0.1349 | 0.0768 | +0.0085 | NO | M0_Persistence |
| `15-1008` | Unseen Generalization | 3 | 649.0 | **0.0335** | 0.0463 | 0.0354 | +0.0019 | NO | M0_Persistence |
| `15-7080` | Unseen Generalization | 3 | 650.0 | **0.0163** | 0.0207 | 0.0198 | +0.0035 | NO | M0_Persistence |
| `22-3056` | Unseen Generalization | 3 | 609.0 | 0.0694 | 0.0722 | **0.0645** | **-0.0049** | **YES** | **M4_XGB** |
| `22-4001` | Unseen Generalization | 2 | 545.5 | 0.1065 | 0.1082 | **0.0984** | **-0.0081** | **YES** | **M4_XGB** |
| `36-1008` | Audit Set (Phase 1B)* | 6 | 345.3 | **0.0558** | 0.0592 | 0.0611 | +0.0053 | NO | M0_Persistence |
| `36-1011` | Unseen Generalization | 8 | 285.5 | **0.0247** | 0.0278 | 0.0263 | +0.0016 | NO | M0_Persistence |
| `36-1643` | Unseen Generalization | 8 | 370.1 | 0.0734 | 0.0709 | **0.0692** | **-0.0042** | **YES** | **M4_XGB** |
| `36-1644` | Unseen Generalization | 10 | 420.2 | **0.0354** | 0.0396 | 0.0372 | +0.0018 | NO | M0_Persistence |
| `36-4017` | Audit Set (Phase 1B)* | 5 | 357.2 | **0.0824** | 0.0872 | 0.0886 | +0.0062 | NO | M0_Persistence |
| `36-4018` | Unseen Generalization | 13 | 196.2 | 0.0784 | 0.0772 | **0.0738** | **-0.0046** | **YES** | **M4_XGB** |
| `38-2001` | Unseen Generalization | 5 | 355.4 | 0.0768 | 0.0784 | **0.0721** | **-0.0047** | **YES** | **M4_XGB** |
| `38-3005` | Unseen Generalization | 7 | 389.7 | 0.0832 | 0.0841 | **0.0784** | **-0.0048** | **YES** | **M4_XGB** |
| `38-3006` | Audit Set (Phase 1B)* | 2 | 346.5 | 0.0852 | 0.0912 | **0.0792** | **-0.0060** | **YES** | **M4_XGB** |
| `38-5002` | Audit Set (Phase 1B)* | 4 | 434.2 | **0.0247** | 0.0321 | 0.0289 | +0.0042 | NO | M0_Persistence |
| `44-7401` | Unseen Generalization | 9 | 362.0 | **0.0054** | 0.0078 | 0.0062 | +0.0008 | NO | M0_Persistence |

### Synthesis of Subgroup Win Rates:
- **Unseen Generalization Sites (17 sites)**: M4 wins on **10 of 17 sites (58.8%)**.
- **Audit Set Sites (6 sites)**: M4 wins on **2 of 6 sites (33.3%)**, with Persistence winning on 4 of 6 (`15-1006`, `36-1008`, `36-4017`, `38-5002`). This reproduces the empirical result discovered in Phase 1B and confirms that model redesign was not overfitted to the audit set.

---

## 3. Horizon-Stratified Performance Analysis

Inspection intervals vary significantly across highway jurisdictions. The table below partitions predictions into three operational duration tiers:

| Horizon Stratum | Duration Range (Days) | Evaluation Pairs | M0 Persistence MAE | M4 Combined XGB MAE | M4 Skill vs. M0 (%) | Winning Strategy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Short Horizon** | $< 365$ days | 43 | **`0.0496`** | `0.0515` | `-3.77%` | **M0 Persistence** |
| **Medium Horizon** | $365–550$ days | 35 | `0.0674` | **`0.0612`** | **`+9.12%`** | **M4 Combined XGB** |
| **Long Horizon** | $> 550$ days | 35 | `0.0597` | **`0.0536`** | **`+10.15%`** | **M4 Combined XGB** |

### Physical Interpretation:
- Over short intervals ($< 1\text{ year}$), highway pavement condition changes very little. Any tree-based delta estimate risks predicting small spurious changes, so physical persistence ($\Delta y = 0$) achieves the lowest error.
- Over multi-year intervals ($> 1\text{ year}$), cumulative vehicle passes, thermal cycling, and precipitation cause meaningful pavement degradation ($\Delta y > 0$). Here, M4 successfully captures the magnitude of change, reducing MAE by **$+9.12\%$** to **$+10.15\%$**.

---

## 4. Explainability & Feature Importance

To understand model decision weighting without conflating correlation with causation, permutation feature importance and split gain metrics were computed:

| Feature Name | Feature Category | Permutation Importance ($\Delta\text{MAE}$) | XGBoost Split Gain (%) | Epistemic Nature |
|---|---|:---:|:---:|---|
| `days_ahead` | Scenario / Horizon | **`+0.0042`** | **`28.4%`** | Exposure duration scaling |
| `current_severity` | Current Condition | **`+0.0031`** | **`22.1%`** | Baseline physical state |
| `annualized_severity_slope`| Temporal History | **`+0.0028`** | **`18.6%`** | Historical trajectory slope |
| `most_recent_severity_change`| Temporal History | **`+0.0019`** | **`11.2%`** | Immediate prior step |
| `traffic_level` | Scenario Exposure | **`+0.0014`** | **`8.5%`** | Cumulative load factor |
| `rainfall_level` | Scenario Exposure | **`+0.0009`** | **`4.8%`** | Moisture exposure factor |
| `temperature` | Scenario Exposure | **`+0.0007`** | **`3.9%`** | Thermal cycling factor |
| `rolling_severity_std` | Temporal History | **`+0.0005`** | **`2.5%`** | Historical volatility |

### Strict Epistemic Boundaries:
1. **Predictive Value $\ne$ Physical Causality**: Permutation importance identifies variables that help minimize quadratic regression error. It does not establish that artificially changing `rainfall_level` or `traffic_level` will cause the predicted condition delta in real pavement.
2. **Horizon Dominance**: `days_ahead` and `current_severity` account for over $50\%$ of tree gain, reinforcing that time elapsed and current distress are the primary statistical drivers of observed deterioration.
