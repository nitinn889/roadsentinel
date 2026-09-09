# RoadSentinel — SEG_004 Primary Goal Evaluation Summary

## Segment Overview: SEG_004 (10 Physical Captures)

This report covers all 10 physical images captured for `SEG_004`, documenting the single-image current road-health assessment, 90-day XGBoost scenario forecasts (Goal 1), and same-camera temporal change tracking (Goal 2).

### Complete 10-Day Image Inventory & Results Table

| Day | Camera Preset | Lighting Preset | Health State | Current Severity | Defect Count | Defect Area | 90d NORMAL | 90d HEAVY RAIN | Goal 2 Status |
|---|---|---|---|---|---|---|---|---|---|
| Day 01 | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | **0.0000** | 0 | 0.000000 | **0.0345** | **0.0345** | ✅ Eligible (SEG_004_D01_D05) |
| Day 02 | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Golden Hour Sunset | Minor Wear (Grade B) | **0.0000** | 0 | 0.000000 | **0.0345** | **0.0345** | ✅ Eligible (SEG_004_D01_D05) |
| Day 03 | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | **0.0000** | 0 | 0.000000 | **0.0345** | **0.0345** | ✅ Eligible (SEG_004_D01_D05) |
| Day 04 | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | **0.3150** | 2 | 0.001154 | **0.3057** | **0.3057** | ✅ Eligible (SEG_004_D01_D05) |
| Day 05 | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Overcast Day | Severe Breakdown (Grade D - Critical) | **0.7302** | 4 | 0.026687 | **0.7446** | **0.7446** | ✅ Eligible (SEG_004_D01_D05) |
| Day 06 | 🌄 Highway Curve Vantage Overlook | Overcast Day | Severe Breakdown (Grade D - Critical) | **0.6612** | 6 | 0.010490 | **0.6213** | **0.6213** | ✅ Eligible (SEG_004_D06_D10) |
| Day 07 | 🌄 Highway Curve Vantage Overlook | Heavy Rain & Wet Road | Severe Breakdown (Grade D - Critical) | **0.8134** | 12 | 0.022511 | **0.7892** | **0.7892** | ✅ Eligible (SEG_004_D06_D10) |
| Day 08 | 🌄 Highway Curve Vantage Overlook | Heavy Rain & Wet Road | Severe Breakdown (Grade D - Critical) | **0.8104** | 11 | 0.018253 | **0.7892** | **0.7892** | ✅ Eligible (SEG_004_D06_D10) |
| Day 09 | 🌄 Highway Curve Vantage Overlook | Golden Hour Sunset | Critical Hazard (Grade F) | **0.0000** | 0 | 0.000000 | **0.0345** | **0.0345** | ✅ Eligible (SEG_004_D06_D10) |
| Day 10 | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Critical Hazard (Grade F) | **0.0000** | 0 | 0.000000 | **0.0345** | **0.0345** | ✅ Eligible (SEG_004_D06_D10) |

---

### Key Research Insights for this Segment

- **Model-Observed Progression (D01–D05 Nadir & D06–D10 Overlook)**: Shows model severity progression across deteriorating simulated states under fixed camera perspectives.
