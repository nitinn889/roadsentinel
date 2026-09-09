# RoadSentinel — SEG_002 Primary Goal Evaluation Summary

## Segment Overview: SEG_002 (10 Physical Captures)

This report covers all 10 physical images captured for `SEG_002`, documenting the single-image current road-health assessment, 90-day XGBoost scenario forecasts (Goal 1), and same-camera temporal change tracking (Goal 2).

### Complete 10-Day Image Inventory & Results Table

| Day | Camera Preset | Lighting Preset | Health State | Current Severity | Defect Count | Defect Area | 90d NORMAL | 90d HEAVY RAIN | Goal 2 Status |
|---|---|---|---|---|---|---|---|---|---|
| Day 01 | 🔍 Low-Angle Pothole Inspection (30° Close-Up) | Clear Noon (70° Sun) | Pristine (Grade A) | **0.3155** | 2 | 0.001281 | **0.3057** | **0.3057** | ✅ Eligible (SEG_002_D01_D02) |
| Day 02 | 🔍 Low-Angle Pothole Inspection (30° Close-Up) | Clear Noon (70° Sun) | Pristine (Grade A) | **0.3498** | 2 | 0.000254 | **0.3108** | **0.3108** | ✅ Eligible (SEG_002_D01_D02) |
| Day 03 | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | **0.2610** | 1 | 0.002742 | **0.2808** | **0.2808** | ❌ VIEWPOINT_CHANGE (Overhead Drone Survey on Day03 isolate) |
| Day 04 | 💧 Waterlogged Pothole Macro View | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | **0.3687** | 3 | 0.001065 | **0.3283** | **0.3283** | ✅ Eligible (SEG_002_D04_D05) |
| Day 05 | 💧 Waterlogged Pothole Macro View | Overcast Day | Moderate Deterioration (Grade C) | **0.7078** | 5 | 0.020932 | **0.7446** | **0.7446** | ✅ Eligible (SEG_002_D04_D05) |
| Day 06 | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | **0.3491** | 2 | 0.000221 | **0.3108** | **0.3108** | ✅ Eligible (SEG_002_D06_D07) |
| Day 07 | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | **0.2551** | 1 | 0.002265 | **0.2808** | **0.2808** | ✅ Eligible (SEG_002_D06_D07) |
| Day 08 | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | **0.3288** | 1 | 0.000730 | **0.3057** | **0.3057** | ❌ VIEWPOINT_CHANGE (Overhead Drone Survey on Day08 isolate) |
| Day 09 | 💧 Waterlogged Pothole Macro View | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | **0.3276** | 1 | 0.000797 | **0.3057** | **0.3057** | ❌ VIEWPOINT_CHANGE (Waterlogged Macro View on Day09 isolate) |
| Day 10 | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Critical Hazard (Grade F) | **0.3275** | 1 | 0.000808 | **0.3057** | **0.3057** | ❌ INSUFFICIENT_CONTIGUOUS_STATES (Overlook single state on Day10) |

---

### Key Research Insights for this Segment

- **Multi-Perspective Pairwise Sequences**: Contains three discrete 2-day same-camera sequences (D01–D02 Low-Angle, D04–D05 Macro, D06–D07 Overlook) demonstrating pairwise temporal tracking across disparate viewpoints.
