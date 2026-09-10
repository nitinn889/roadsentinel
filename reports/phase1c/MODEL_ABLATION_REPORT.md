# RoadSentinel Phase 1C: Forecast Model Ablation Report

**Document**: `reports/phase1c/MODEL_ABLATION_REPORT.md`  
**Execution Phase**: Prompt 1C — Temporal-Residual XGBoost Forecasting  
**Validation Protocol**: 5-Fold Nested Grouped Cross-Validation (Grouped by `site_id`, $N=113$ pairs across 23 sites)  
**Supporting Artifact**: [`artifacts/phase1c/model_comparison.csv`](../../artifacts/phase1c/model_comparison.csv)  
**Figure**: [`figures/phase1c/model_comparison.png`](../../figures/phase1c/model_comparison.png)  
**Date**: September 10, 2026  
**Status**: **VALIDATED & REPRODUCIBLE**  

---

## 1. Executive Summary

This report documents the rigorous ablation analysis across five standardized deterioration forecasting models (M0 through M4). All models were trained, tuned, and evaluated on the **exact same outer and inner folds** using the residual formulation:
$$\hat{y}_{\text{future}} = y_{\text{current}} + \widehat{\Delta y}$$

### Key Findings of the Ablation:
1. **Scenario Features Alone Fail to Beat Persistence**: Adding environmental covariates (`rainfall_level`, `temperature`, `water_exposure`, `traffic_level`) to current condition without temporal history (M1 OLS, M2 XGBoost) results in **negative skill** relative to persistence ($\text{MAE Skill} = -5.45\%$ for M1; $-0.85\%$ for M2).
2. **Temporal History Provides Genuine Predictive Signal**: Adding strictly leak-free temporal-history features (M3 XGBoost Temporal) achieves **positive skill** over persistence ($\text{MAE Skill} = +3.26\%$, $\text{MAE} = 0.0563$ vs. $0.0582$), achieving a **$73.91\%$ site win rate** ($17/23$ sites) and positive Spearman rank correlation ($\rho = 0.2405$).
3. **M4 Combined Model Achieves Lowest Overall Error**: The proposed RoadSentinel forecaster combining current condition, temporal history, and scenario covariates (M4) achieves the lowest outer-fold error: $\text{MAE} = \mathbf{0.0551}$, $\text{RMSE} = \mathbf{0.0766}$, $R^2 = \mathbf{0.9229}$, and $\text{MAE Skill} = \mathbf{+5.27\%}$.

---

## 2. Master Model Ablation Matrix

The table below summarizes performance across all 113 out-of-fold predictions evaluated through 5-fold nested grouped cross-validation:

| Model ID | Model Architecture & Inputs | Target MAE (Severity) | Target RMSE | $R^2$ Score | Change MAE ($\Delta y$) | MAE Skill vs. M0 (%) | Paired $\Delta\text{MAE}$ | Paired $\Delta\text{RMSE}$ | Site Win Rate (%) | Direction Accuracy (%) | Spearman Rank Corr ($\rho$) | Outperforms Persistence? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0** | **Persistence Baseline**: $\widehat{\Delta y} = 0$ | **`0.0582`** | `0.0815` | `0.9125` | `0.0582` | `0.00%` | `0.0000` | `0.0000` | `0.00%` | `0.00%` | `0.0000` | *Baseline* |
| **M1** | **OLS Linear Regression**: Current condition + Scenario | `0.0614` | `0.0829` | `0.9095` | `0.0620` | `-5.45%` | `+0.0032` | `+0.0014` | `43.48%` | `58.04%` | `-0.0483` | **NO** |
| **M2** | **XGBoost Residual**: Current condition + Scenario | `0.0587` | `0.0806` | `0.9145` | `0.0588` | `-0.85%` | `+0.0005` | `-0.0009` | `52.17%` | `65.18%` | `-0.0094` | **NO** |
| **M3** | **XGBoost Residual**: Current condition + Temporal History | **`0.0563`** | `0.0806` | `0.9145` | `0.0573` | **`+3.26%`** | **`-0.0019`** | `-0.0009` | **`73.91%`** | `66.96%` | **`0.2405`** | **YES** |
| **M4** | **XGBoost Combined**: Current + Temporal + Scenario | **`0.0551`** | **`0.0766`** | **`0.9229`** | **`0.0561`** | **`+5.27%`** | **`-0.0031`** | **`-0.0050`** | `52.17%` | `65.18%` | **`0.2181`** | **YES** |

---

## 3. Target Formulation Selection (Inner Fold Tuning)

As mandated by Section 2 of Prompt 1C, the pipeline evaluated two target representations inside the 4-fold grouped inner cross-validation:
1. **Target A: Direct Severity Change**: $\Delta y = y_{\text{future}} - y_{\text{current}}$
2. **Target B: Annualized Severity Change Rate**: $r = \Delta y / (\text{observed\_days\_ahead} / 365.25)$, where $\widehat{\Delta y} = \hat{r} \cdot \Delta t_{\text{years}}$

### Inner CV Selection Results:
* **Fold 0**: Direct Change selected ($\text{MAE} = 0.0587$ vs. $0.0601$)
* **Fold 1**: Direct Change selected ($\text{MAE} = 0.0527$ vs. $0.0539$)
* **Fold 2**: Direct Change selected ($\text{MAE} = 0.0522$ vs. $0.0531$)
* **Fold 3**: Annualized Rate selected ($\text{MAE} = 0.0472$ vs. $0.0489$)
* **Fold 4**: Direct Change selected ($\text{MAE} = 0.0552$ vs. $0.0568$)

**Analytical Conclusion**: Direct severity change ($\Delta y$) demonstrated superior stability in 4 out of 5 outer folds. Annualized rate scaling can exaggerate errors over short intervals ($< 180\text{ days}$) where minor IRI measurement noise creates large artificial rate spikes ($r \propto 1/\Delta t$).

---

## 4. Deep-Dive Component Insights

### 4.1 Why Does M1 (OLS) Fail?
OLS linear regression on scenario features yields negative skill ($\text{MAE} = 0.0614$ vs. $0.0582$, skill $-5.45\%$). Without non-linear tree splits or regularization, linear coefficients fit spurious noise between environmental covariates and distress changes, pulling predictions away from physical inertia.

### 4.2 Why Does M3 (Temporal History) Dominate Site Win Rate?
M3 achieves the highest site win rate in the benchmark (**$73.91\%$**, winning 17 of 23 sites). Pavement distress progression is fundamentally an auto-correlated physical degradation process. Features capturing the previous deterioration slope (`annualized_severity_slope`), recent change (`most_recent_severity_change`), and historical dispersion (`rolling_severity_std`) allow the model to distinguish static, well-performing sections from actively accelerating deteriorations.

### 4.3 What Does M4 Add Over M3?
While M3 wins on more sites ($73.91\%$ vs. $52.17\%$), M4 achieves lower overall MAE ($0.0551$ vs. $0.0563$) and lower RMSE ($0.0766$ vs. $0.0806$). Scenario covariates (`rainfall_level`, `temperature`, `water_exposure`) provide useful bounding constraints on high-exposure highway sections, refining the magnitude of predicted deterioration when multi-year harsh climate conditions are present.
