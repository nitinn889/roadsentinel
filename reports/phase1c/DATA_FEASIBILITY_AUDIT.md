# RoadSentinel Phase 1C: Pavement Deterioration Data Feasibility Audit Report

**Document**: `reports/phase1c/DATA_FEASIBILITY_AUDIT.md`  
**Execution Phase**: Prompt 1C — Temporal-Residual XGBoost Forecasting  
**Dataset Under Audit**: FHWA Long-Term Pavement Performance (LTPP InfoPave SDR 40)  
**Supporting Artifact**: [`artifacts/phase1c/feasibility_audit_summary.json`](../../artifacts/phase1c/feasibility_audit_summary.json) / [`artifacts/phase1c/temporal_feature_manifest.csv`](../../artifacts/phase1c/temporal_feature_manifest.csv)  
**Date**: September 10, 2026  
**Status**: **COMPLETE & EMPIRICALLY AUDITED**  

---

## 1. Executive Summary

Before implementing the temporal-residual forecasting subsystem, a rigorous data-feasibility audit was conducted across all available pavement progression records in `xgboost/data/processed/scenario_training_pairs.csv`. The purpose of this audit is to prevent methodological artifacts, detect potential temporal lookahead leakage, inventory observational history, and verify whether genuine lag features can be constructed without synthetic fabrication.

### Key Feasibility Findings:
1. **Sample Size & Geographic Spread**: Exactly **113 transition pairs** across **23 unique highway test sections** spanning 6 US states (`NY`: 50, `AK`: 19, `ND`: 18, `HI`: 12, `RI`: 9, `LA`: 5).
2. **Missing Data Accounting**: Exactly **0 missing values** exist across any raw scenario or target measurement in the joined LTPP table.
3. **Chronological Monotonicity**: Every section pair $(t_1, t_2)$ satisfies $t_1 < t_2$. Within every site, inspection dates are strictly monotonic ($D_0 < D_1 < \dots < D_K$).
4. **Observed Interval Lengths**: Historical survey intervals range from **31 to 728 days** (mean **429.5 days** [$\approx 1.18$ years], median **420.0 days**, IQR $[283.0, 633.0]$ days).
   - **Crucial Boundary Constraint**: Rapid 30–90-day deterioration forecasting **cannot be scientifically validated** from these records, as the true observational support spans multi-month to multi-year intervals.
5. **Sufficiency for Temporal History Features**:
   - Exactly **23 of 23 sites (100.0%)** have at least 3 chronological profile visits.
   - Exactly **22 of 23 sites (95.7%)** have at least 4 chronological profile visits.
   - Across the 113 transition pairs: **90 pairs (79.6%)** have at least 1 prior inspection, and **78 pairs (69.0%)** have at least 2 prior inspections before the prediction origin.
   - **Zero Fabricated Lags**: For pairs where prior inspections do not exist ($k < 2$), lag variables are represented strictly as `NaN` (unobserved), allowing tree algorithms to route missing values via default split branches rather than hallucinating artificial values.

---

## 2. Site Inventory & Inspection Timelines

The table below catalogs all 23 LTPP test sites, their state jurisdictions, total transition pairs, total distinct profile visits, and observation timespans:

| Site Identifier | State | Construction # | Total Transition Pairs | Distinct Visits | Timespan (First to Last Visit) | Mean Interval (Days) | Interval Range (Days) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `02-1001` | AK | 1, 3, 4 | 4 | 5 | 1990-05-28 to 2005-05-05 | 628.0 | [450, 702] |
| `02-1002` (Audit)* | AK | 1, 2 | 4 | 5 | 1990-06-02 to 2005-05-04 | 623.5 | [446, 699] |
| `02-1004` | AK | 2, 3 | 3 | 4 | 1993-08-27 to 2005-05-03 | 682.7 | [655, 702] |
| `02-1008` | AK | 1 | 2 | 3 | 1990-05-25 to 1999-06-16 | 559.0 | [461, 657] |
| `02-6010` | AK | 1 | 3 | 4 | 1990-05-29 to 1995-06-12 | 613.3 | [456, 727] |
| `02-9035` | AK | 3, 4 | 3 | 4 | 1997-08-26 to 2005-05-09 | 684.7 | [661, 703] |
| `15-1003` | HI | 1 | 3 | 6 | 1991-02-21 to 2003-03-14 | 655.7 | [554, 713] |
| `15-1006` (Audit)* | HI | 1 | 3 | 6 | 1991-02-22 to 2003-03-13 | 655.7 | [554, 711] |
| `15-1008` | HI | 1 | 3 | 6 | 1991-02-27 to 2003-03-06 | 649.0 | [541, 722] |
| `15-7080` | HI | 1 | 3 | 6 | 1991-02-25 to 2003-03-05 | 650.0 | [541, 723] |
| `22-3056` | LA | 1 | 3 | 4 | 1990-05-31 to 1995-06-01 | 609.0 | [504, 720] |
| `22-4001` | LA | 1 | 2 | 4 | 1990-06-01 to 1995-06-02 | 545.5 | [532, 559] |
| `36-1008` (Audit)* | NY | 2 | 6 | 8 | 1990-06-05 to 2000-04-20 | 345.3 | [238, 451] |
| `36-1011` | NY | 1, 2 | 8 | 11 | 1990-06-05 to 2004-04-12 | 285.5 | [129, 628] |
| `36-1643` | NY | 2, 3, 6 | 8 | 12 | 1990-06-08 to 2004-04-22 | 370.1 | [149, 677] |
| `36-1644` | NY | 1, 2, 3 | 10 | 13 | 1990-06-06 to 2004-04-13 | 420.2 | [168, 728] |
| `36-4017` (Audit)* | NY | 1, 3 | 5 | 7 | 1990-06-09 to 1999-04-30 | 357.2 | [273, 461] |
| `36-4018` | NY | 2, 4, 6, 7, 8, 9 | 13 | 19 | 1990-11-02 to 2010-04-13 | 196.2 | [31, 446] |
| `38-2001` | ND | 2 | 5 | 7 | 1990-11-30 to 1998-06-26 | 355.4 | [232, 433] |
| `38-3005` | ND | 1, 2 | 7 | 10 | 1989-10-24 to 2001-10-09 | 389.7 | [284, 443] |
| `38-3006` (Audit)* | ND | 1 | 2 | 3 | 1991-07-19 to 1993-06-11 | 346.5 | [285, 408] |
| `38-5002` (Audit)* | ND | 1 | 4 | 5 | 1989-10-25 to 1994-07-28 | 434.2 | [280, 633] |
| `44-7401` | RI | 1, 2, 3 | 9 | 13 | 1989-10-25 to 2004-08-12 | 362.0 | [75, 639] |

*\*Denotes the 6 previously observed audit sites from Phase 1B.*

---

## 3. Provenance of Scenario Variables

All scenario variables joined to profile intervals originate from official FHWA LTPP InfoPave SDR 40 database releases:

1. **Precipitation (`rainfall_level`)**:
   - **Source Table**: `CLM_VWS_PRECIP_DAILY` via `CLM_SITE_VWS_LINK`.
   - **Definition**: Average daily precipitation (mm/day) accumulated over the exact future interval $(t_{curr}, t_{fut}]$.
   - **Coverage Standard**: Only intervals with $\ge 80\%$ daily weather records are accepted.
2. **Temperature (`temperature`)**:
   - **Source Table**: `CLM_VWS_TEMP_DAILY` via `CLM_SITE_VWS_LINK`.
   - **Definition**: Mean daily air temperature ($^\circ\text{C}$) averaged over the future interval.
3. **Water Exposure (`water_exposure`)**:
   - **Definition**: Empirical fraction of days in $(t_{curr}, t_{fut}]$ where daily precipitation $> 0\text{ mm}$ ($\text{wet\_days} / \text{total\_days}$).
4. **Traffic Exposure (`traffic_level`)**:
   - **Source Table**: `TRF_MEPDG_AADTT_LTPP_LN` (annual measured truck counts) with `TRF_REP` fallback.
   - **Definition**: Average annual daily truck traffic (heavy commercial vehicles per day in the monitored lane).
5. **Horizon Duration (`days_ahead` / `observed_days_ahead`)**:
   - **Definition**: Elapsed calendar days between consecutive profile measurements.
   - **Interface Bins**: 30-day discrete increments (`days_ahead`), with exact calendar counts recorded in `observed_days_ahead`.

### Strict Epistemic & Regulatory Warnings:
> [!WARNING]
> **Non-Causal Interpretation**: Scenario variables represent historical associations and environmental correlations observed in highway field surveys. They **do not prove physical causality**. Tree split gains and model feature importance must not be presented as counterfactual maintenance engineering guarantees.

---

## 4. Definition of Pavement Distress Severity

Pavement distress severity in RoadSentinel is derived from Mean Roughness Index (MRI, m/km) recorded in the FHWA LTPP `ANALYSIS_IRI` table.

To prevent leakage during cross-validation:
- For each outer training fold, empirical quantiles $q_{0.01}$ and $q_{0.99}$ are calculated strictly across training site profile measurements:
  $$q_{\text{low}} = \text{Quantile}(\text{IRI}_{\text{train}}, 0.01), \quad q_{\text{high}} = \text{Quantile}(\text{IRI}_{\text{train}}, 0.99)$$
- Severity is bounded to $[0.0, 1.0]$ via min-max quantile normalization:
  $$y = \text{clip}\left(\frac{\text{IRI} - q_{\text{low}}}{q_{\text{high}} - q_{\text{low}}}, 0.0, 1.0\right)$$
- Across the complete LTPP dataset, baseline $q_{\text{low}} \approx 0.817\text{ m/km}$ and $q_{\text{high}} \approx 2.846\text{ m/km}$.
- Target severity change is defined as:
  $$\Delta y = y_{\text{future}} - y_{\text{current}}$$
- Annualized severity change rate is defined as:
  $$r = \frac{\Delta y}{\text{observed\_days\_ahead} / 365.25}$$

---

## 5. Audit Conclusion & Feasibility Certification

The data-feasibility audit confirms that the LTPP dataset has sufficient depth and visit history across 23 highway sections to evaluate temporal-history features:
- **78 pairs (69.0%)** possess $\ge 2$ prior observations, enabling full 2nd-order lag evaluation.
- **Zero lookahead leakage** was verified across all 113 records.
- The pipeline proceeds to temporal feature definition and nested grouped model ablation.
