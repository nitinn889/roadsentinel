# RoadSentinel — Primary Goals Research Summary
## Comprehensive Evaluation of Goal 1 (Forecasting) and Goal 2 (Temporal Tracking) on Experiment A

---

## 1. Executive Summary

Phase 11 unifies and completes the two core operational pillars of RoadSentinel across the 40 physical captures of Experiment A (`SEG_001` through `SEG_004`):
1. **Goal 1 (Single-Image Health Assessment $\rightarrow$ Future Scenario Forecasting)**:
   - Evaluated on **100% of physical captures (40 / 40 images)**.
   - For every image, single-frame DINOv2+SAM2 current severity was coupled with the frozen XGBoost Model V2 across 5 environmental scenario presets (`NORMAL`, `HIGH_HEAT`, `HEAVY_TRAFFIC`, `HEAVY_RAIN`, `WET_EXPOSURE`) and 3 forecast horizons (30, 60, 90 days), producing 600 validated forecast records.
2. **Goal 2 (Multi-Inspection Same-Camera Temporal Tracking)**:
   - Evaluated across **8 verified same-camera sequences (33 temporal transitions)**.
   - Reconciled canonical temporal event totals: **48 unique tracks, 33 matched events (14 area increased / 19 area decreased), 48 new defects, 34 not observed**.
   - All 7 non-contiguous viewpoint transition captures (e.g. `SEG_001` Day01 Drone, Day02 Macro) are explicitly cataloged with their exact exclusion rationale rather than omitted.

---

## 2. Answers to Core Research Questions

### 1. Were all SEG001–004 images assessed?
**Yes, completely.** All 40 physical captures across `SEG_001`, `SEG_002`, `SEG_003`, and `SEG_004` are accounted for in the master inventory ([`ROADSENTINEL_PRIMARY_RESULTS.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/primary_goals/ROADSENTINEL_PRIMARY_RESULTS.csv)). Each image possesses a verified `current_severity`, `defect_count`, `defect_area_ratio`, and `surface_anomaly_score`.

### 2. Were future predictions generated?
**Yes.** All 40 images received complete XGBoost Model V2 scenario forecasts across all 5 standard presets and all 3 horizons (30d, 60d, 90d). For `SEG_003` Day10 (where simulation sidecar metadata was absent), forecasts were generated in diagnostic mode directly from the image perception output.

### 3. Which scenarios produced largest forecast changes?
Across all 40 captures, the scenarios ranked by mean 90-day predicted deterioration are:
1. **`WET_EXPOSURE`**: Mean 90d Severity = **0.4077** (Mean $\Delta = +0.2223$) $\rightarrow$ *Produces the highest rate of structural degradation due to subgrade moisture saturation.*
2. **`HEAVY_RAIN`**: Mean 90d Severity = **0.3705** (Mean $\Delta = +0.1851$)
3. **`HEAVY_TRAFFIC`**: Mean 90d Severity = **0.3541** (Mean $\Delta = +0.1687$)
4. **`NORMAL`**: Mean 90d Severity = **0.3375** (Mean $\Delta = +0.1521$)
5. **`HIGH_HEAT`**: Mean 90d Severity = **0.3375** (Mean $\Delta = +0.1521$)

### 4. Which sequences demonstrated stable responses?
- **`SEG_003_D01_D07` (Stability Control Benchmark)**: Evaluated over 7 consecutive days under invariant Overlook vantage and Clear Noon lighting on pristine asphalt.
  - Mean Severity: $0.2444 \pm 0.0016$ ($\text{CV} = 0.67\%$).
  - Defect Tracking: Exactly 1 persistent region (`DEFECT_001`) tracked continuously across all 7 inspections with zero false track splits.

### 5. Which sequences demonstrated progression?
- **`SEG_004_D01_D05` (Nadir Drone)**: Severity progressed from $0.0000 \rightarrow 0.4431$ across simulated state breakdown.
- **`SEG_004_D06_D10` (Overlook Vantage)**: Severity progressed from $0.4485 \rightarrow 0.6720$ ($\Delta = +0.2235$) across deteriorating road health states.

### 6. Which sequences were environmentally confounded?
- **`SEG_001_D03_D10` (Overlook Vantage)**: Highly sensitive to illumination and weather.
  - Day 03 (Clear Noon): Severity $= 0.0000$ (0 defects).
  - Day 06 (Overcast): Severity $= 0.7285$ (8 detected candidate regions).
  - Day 10 (Sunset): Severity $= 0.0000$ (0 defects).
  - Demonstrates that transient lighting shifts can modulate optical anomaly perception without underlying structural change.

### 7. Which images could not support temporal matching and why?
The following 7 physical images were ineligible for geometric defect tracking:
- `SEG_001` Day 01: `VIEWPOINT_CHANGE` (Nadir Drone view on Day01 vs Overlook on Day03–10).
- `SEG_001` Day 02: `VIEWPOINT_CHANGE` (Macro Pothole view on Day02 vs Overlook on Day03–10).
- `SEG_002` Day 03: `VIEWPOINT_CHANGE` (Nadir Drone isolate).
- `SEG_002` Day 08: `VIEWPOINT_CHANGE` (Nadir Drone isolate).
- `SEG_002` Day 09: `VIEWPOINT_CHANGE` (Macro Pothole isolate).
- `SEG_002` Day 10: `INSUFFICIENT_CONTIGUOUS_STATES` (Overlook single-day isolate).
- `SEG_003` Day 10: `MISSING_METADATA` (Unrecorded simulation sidecar).

### 8. What is the fundamental distinction between observed change and XGBoost forecast?
- **Observed Change**: Represents what the perception pipeline *actually detected* between two sequential physical captures under the same camera geometry.
- **XGBoost Forecast**: Represents a *machine-learned simulation* of how pavement would deteriorate over 30–90 days under external climate/traffic stress, starting from the current condition score.

### 9. What limitations remain?
1. **Experiment A Confounding**: Because the Team Lead intentionally varied simulation settings in Experiment A, longitudinal sequences reflect combined environmental and condition variations rather than pure monotonic deterioration (which is reserved for Experiment B).
2. **Horizon Interpolation**: XGBoost forecasts are discrete scenario models trained on historical LTPP pavement records and should be treated as planning estimates, not deterministic physics simulations.
