# RoadSentinel — Primary Goals Methodology
## Per-Image Road Health Forecasting (Goal 1) and Multi-Inspection Temporal Tracking (Goal 2)

---

## 1. System Overview & The Two Primary Research Goals

RoadSentinel is structured around two complementary operational capabilities:

```
               ┌─────────────────────────────────────────────────────────────┐
               │              SINGLE ROAD INSPECTION CAPTURE                 │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   DINOv2 + SAM2 PERCEPTION  │
                               │  (Frozen Day-5 NORMAL Model)│
                               └──────────────┬──────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
     ┌─────────────────────────────┐                     ┌─────────────────────────────┐
     │       PRIMARY GOAL 1        │                     │       PRIMARY GOAL 2        │
     │  Single-Image Health State  │                     │ Same-Road Repeated Tracking │
     │             +               │                     │             +               │
     │  XGBoost Scenario Forecast  │                     │   Temporal Change Analysis  │
     └─────────────────────────────┘                     └─────────────────────────────┘
```

### 1.1 Goal 1: Single-Image Current Assessment $\rightarrow$ Future Severity Prediction
- **Input**: Any single physical RGB road inspection capture (independent of camera geometry or trajectory continuity).
- **Processing**: DINOv2 + SAM2 feature extraction computes `current_severity`, `defect_count`, `defect_area_ratio`, and `surface_anomaly_score`.
- **Forecasting**: Frozen XGBoost Model V2 (`roadsentinel_xgb_v2_ltpp_scenario`) ingests the current severity alongside user-specified environmental parameters (`rainfall_level`, `traffic_level`, `temperature`, `water_exposure`, `days_ahead`) to generate deterministic future severity predictions.

### 1.2 Goal 2: Same-Road Repeated-Inspection Change Analysis
- **Input**: Sequences of repeated captures over the same road segment under a **fixed camera viewpoint**.
- **Processing**: Greedy one-to-one spatial correspondence matching across successive days using the frozen hierarchical matcher:
  1. Mask $\text{IoU} \ge 0.50$
  2. BBox $\text{IoU} \ge 0.30$
  3. Centroid Euclidean distance $\le 75\text{px}$ AND Area Ratio $\le 3.0\times$
- **Output**: Defect trajectory lifecycle tracking, observed daily deltas, and canonical events (`NEW_DEFECT`, `OBSERVED_AREA_INCREASED`, `OBSERVED_AREA_DECREASED`, `NOT_OBSERVED`).

---

## 2. Fundamental Distinction: Current vs Observed-Change vs Forecast

To preserve complete scientific integrity, RoadSentinel strictly demarcates three distinct quantities:

| Quantity | Operational Definition | Source of Truth | Example Notation |
|---|---|---|---|
| **CURRENT MODEL ASSESSMENT** | Point-in-time perception output from DINOv2+SAM2 on image $I_t$. | Frozen Vision Transformer + SAM 2 | $\text{Severity}(t) = 0.3498$ |
| **MODEL-OBSERVED TEMPORAL CHANGE** | Empirical difference observed between repeated images $I_{t_1} \rightarrow I_{t_2}$ under a fixed camera preset. | Ground-truth image sequence correspondence | $\Delta \text{Severity} = +0.0343$ |
| **MODEL-BASED FORECAST** | Machine-learned extrapolation of future condition conditioned on hypothetical environmental scenarios. | Frozen XGBoost Regressor | $\hat{\text{Severity}}(t + 90\text{d} \mid \text{HEAVY\_RAIN}) = 0.4851$ |

> [!CAUTION]
> **Anti-Fabrication Rule**: XGBoost is NEVER used to simulate or fake observed temporal transitions. Observed change is derived strictly from empirical perception on actual sequential captures.

---

## 3. Physical Dataset & Eligibility Criteria (Experiment A)

Experiment A consists of 40 physical images across segments `SEG_001` through `SEG_004`:

### 3.1 Master Image Participation Rule
- **Goal 1 Participation (100% Coverage)**: **All 40 physical captures** participate in single-image current assessment and XGBoost scenario forecasting. Images with non-standard viewpoints (e.g. `SEG_001` Day01 Drone, Day02 Macro) or missing simulation metadata (`SEG_003` Day10) are fully evaluated under Goal 1.
- **Goal 2 Participation (Controlled Same-Camera Subsets)**: Only images belonging to validated same-camera subsets are evaluated for geometric defect tracking. Ineligible captures are explicitly labeled with their exclusion reason:
  - `VIEWPOINT_CHANGE`: Isolated camera preset transition (e.g. `SEG_001` D01 Nadir vs D03 Overlook).
  - `INSUFFICIENT_CONTIGUOUS_STATES`: Single-day isolate without contiguous matching pair.
  - `MISSING_METADATA`: Incomplete simulation sidecar (`SEG_003` Day10).

### 3.2 Validated Temporal Subsequences (8 Sequences)
1. `SEG_001_D03_D10` (Overlook, 8 states): Environmental sensitivity study.
2. `SEG_002_D01_D02` (Low-Angle, 2 states): Pristine baseline pair.
3. `SEG_002_D04_D05` (Macro, 2 states): Waterlogged macro pair.
4. `SEG_002_D06_D07` (Overlook, 2 states): Moderate deterioration pair.
5. `SEG_003_D01_D07` (Overlook, 7 states): Stability control benchmark ($\text{CV} = 0.67\%$).
6. `SEG_003_D08_D09` (Overhead Drone, 2 states): Pairwise top-down tracking.
7. `SEG_004_D01_D05` (Overhead Drone, 5 states): Model-observed severity progression.
8. `SEG_004_D06_D10` (Overlook, 5 states): Model-observed severity progression.

---

## 4. XGBoost Scenario Forecasting Protocol (Goal 1)

### 4.1 Exact Feature Signature
The frozen XGBoost Regressor (`scenario_model_v2.json`, trained on LTPP pavement deterioration records) requires the following 6 features in exact order:
1. `current_severity` $\in [0.0, 1.0]$
2. `rainfall_level` (Annual precipitation intensity metric)
3. `traffic_level` (Equivalent single-axle load volume)
4. `temperature` (Mean ambient temperature in $^\circ\text{C}$)
5. `water_exposure` (Moisture exposure fraction)
6. `days_ahead` $\in \{30, 60, 90\}$

### 4.2 Standard Scenario Presets
- **`NORMAL`**: Baseline median climate and traffic loading ($\text{Rain}=2.738, \text{Traffic}=346.0, \text{Temp}=6.397^\circ\text{C}, \text{Water}=0.566$).
- **`HIGH_HEAT`**: Elevated summer thermal loading ($\text{Temp}=21.491^\circ\text{C}$).
- **`HEAVY_TRAFFIC`**: Elevated commercial freight loading ($\text{Traffic}=784.0$).
- **`HEAVY_RAIN`**: Elevated monsoon/storm precipitation ($\text{Rain}=3.769$).
- **`WET_EXPOSURE`**: Severe saturated subgrade conditions ($\text{Water}=0.688$).

---

## 5. Event Audit & Canonical Reconciliation (Goal 2)

An exhaustive re-audit of all 8 `temporal_events.json` files was conducted to establish the definitive canonical event totals:
- **`NEW_DEFECT`**: **48 events**
- **`NOT_OBSERVED`**: **34 events**
- **`OBSERVED_AREA_DECREASED`**: **19 events**
- **`OBSERVED_AREA_INCREASED`**: **14 events**
- **Total Matched Events**: $14 + 19 =$ **33 events**
- **Total Unique Defect Tracks**: **48 tracks**
