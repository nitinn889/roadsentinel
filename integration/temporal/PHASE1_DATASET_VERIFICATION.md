# RoadSentinel — Phase 1: Dataset Verification Report

**Generated**: 2026-09-09T12:40:21.682079+00:00  
**Verification Method**: Strict File-Only Audit (zero Unreal/CARLA interaction)  
**Overall Phase 1 Status**: **PARTIAL PASS — SEG_005 AND SEG_006 STILL TO CAPTURE**  
**Batch A Readiness**: **BATCH_A_NOT_READY_FOR_PHASE_2**

---

## 1. Audit Findings

1. **Images Audit (Batch A)**:
   - Expected: 40 (1920x1080 PNG)
   - Found on disk: **40/40**
   - Valid resolution & format (1920x1080 PNG): **40/40**
   - Corrupted/truncated PNGs: **0**
   - Tiny files (<2000 bytes): **0** (file sizes span 1,173,227 to 5,854,635 bytes)

2. **Sidecar Metadata Audit (Batch A)**:
   - Expected per-image sidecars: 40 (`day_01_metadata.json` ... `day_10_metadata.json`)
   - Found on disk: **39/40**
   - Missing proper sidecars: **1** (SEG_003 Day 10)
   - Note on SEG_003 Day 10: `day_10_metadata.json` does not exist. A legacy segment-level `metadata.json` exists in the folder, but lacks standard schema fields.

3. **Manifest & History Audit**:
   - `dataset_manifest.csv`: **39 entries** (missing `SEG_003 Day 10`)
   - `capture_history.jsonl`: **64 entries**, actively logging.
   - Duplicate/re-capture events: **11** segment-days have multiple logs.
   - Error logs in history: **3** failed events:
     - `SEG_003 Day 10` at 17:57:02 (`capture_succeeded: false, capture_status: "error"`)
     - `SEG_003 Day 10` at 17:57:09 (`capture_succeeded: false, capture_status: "error"`)

4. **Repair Event Verification**:
   - **SEG_005**:
     - Capture Plan: Day 05 = `Critical Hazard (Grade F)` (`pre-repair`), Day 06 = `Pristine (Grade A)` (`repair`) → **YES**
     - Sidecar Metadata: **NOT YET CAPTURED** (awaiting capture of SEG_005)
   - **SEG_006**:
     - Capture Plan: Day 05 = `Critical Hazard (Grade F)` (`pre-repair`), Day 06 = `Pristine (Grade A)` (`repair`) → **YES**
     - Sidecar Metadata: **NOT YET CAPTURED** (awaiting capture of SEG_006)

---

## 2. Phase 2 Readiness Gate for Batch A (SEG_001–SEG_004)

**Gate Evaluation**: **`BATCH_A_NOT_READY_FOR_PHASE_2`**

| Gate Check | Requirement | Batch A Result | Status |
|---|---|---|---|
| **1. Physical Images** | 40/40 PNGs exist | 40/40 exist | PASS |
| **2. Image Readability** | 40/40 PNGs readable & non-zero | 40/40 verified 1920x1080 | PASS |
| **3. Proper Sidecars** | 40/40 sidecars exist | 39/40 exist | FAIL (39/40) |
| **4. Sidecars JSON Valid** | 40/40 sidecars parse | 39/40 parse | FAIL (39/40) |
| **5. Metadata Alignment** | No unresolved metadata mismatch | 40/40 captures contain mismatches | FAIL (40 mismatches) |
| **6. Segment/Day Unambiguous** | No naming or day ambiguity | Verified 1:1 segment/day paths | PASS |

---

## 3. Comprehensive SETTING_MISMATCH Breakdown Table (Batch A)

The table below documents every mismatch between actual sidecar metadata and the planned capture parameters in `temporal_capture_plan.csv`.

| Segment | Day | Field | Expected | Actual |
|---|---|---|---|---|
| `SEG_001` | Day 01 | `pothole_density_per_100m2` | 0.5 | 2.8 |
| `SEG_001` | Day 01 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_002` | Day 01 | `lighting_preset` | Overcast Day | Clear Noon (70° Sun) |
| `SEG_002` | Day 01 | `pothole_sizing_spectrum` | Micro Pitting & Hairlines | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_002` | Day 01 | `pothole_density_per_100m2` | 0.5 | 10.0 |
| `SEG_002` | Day 01 | `pothole_moisture_state` | Mixed Wet / Dry Cavities | 100% Dry Crushed Aggregate |
| `SEG_002` | Day 01 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🔍 Low-Angle Pothole Inspection (30° Close-Up) |
| `SEG_003` | Day 01 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_003` | Day 01 | `pothole_density_per_100m2` | 0.5 | 7.3 |
| `SEG_003` | Day 01 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_004` | Day 01 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_004` | Day 01 | `road_health_state` | Pristine (Grade A) | Moderate Deterioration (Grade C) |
| `SEG_004` | Day 01 | `pothole_sizing_spectrum` | Micro Pitting & Hairlines | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_004` | Day 01 | `pothole_density_per_100m2` | 0.5 | 3.5 |
| `SEG_001` | Day 02 | `lighting_preset` | Golden Hour Sunset | Clear Noon (70° Sun) |
| `SEG_001` | Day 02 | `road_health_state` | Minor Wear (Grade B) | Critical Hazard (Grade F) |
| `SEG_001` | Day 02 | `pothole_density_per_100m2` | 1.5 | 3.8 |
| `SEG_001` | Day 02 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 💧 Waterlogged Pothole Macro View |
| `SEG_002` | Day 02 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_002` | Day 02 | `road_health_state` | Minor Wear (Grade B) | Pristine (Grade A) |
| `SEG_002` | Day 02 | `pothole_sizing_spectrum` | Micro Pitting & Hairlines | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_002` | Day 02 | `pothole_density_per_100m2` | 1.5 | 10.0 |
| `SEG_002` | Day 02 | `pothole_moisture_state` | Mixed Wet / Dry Cavities | 100% Dry Crushed Aggregate |
| `SEG_002` | Day 02 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🔍 Low-Angle Pothole Inspection (30° Close-Up) |
| `SEG_003` | Day 02 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_003` | Day 02 | `road_health_state` | Minor Wear (Grade B) | Pristine (Grade A) |
| `SEG_003` | Day 02 | `pothole_density_per_100m2` | 1.5 | 7.3 |
| `SEG_003` | Day 02 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_004` | Day 02 | `lighting_preset` | Dense Atmospheric Fog | Golden Hour Sunset |
| `SEG_004` | Day 02 | `pothole_sizing_spectrum` | Micro Pitting & Hairlines | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_004` | Day 02 | `pothole_density_per_100m2` | 1.5 | 3.5 |
| `SEG_001` | Day 03 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_001` | Day 03 | `road_health_state` | Minor Wear (Grade B) | Moderate Deterioration (Grade C) |
| `SEG_001` | Day 03 | `pothole_sizing_spectrum` | Micro Pitting & Hairlines | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_001` | Day 03 | `pothole_density_per_100m2` | 1.5 | 4.8 |
| `SEG_001` | Day 03 | `pothole_moisture_state` | Mixed Wet / Dry Cavities | 100% Dry Crushed Aggregate |
| `SEG_001` | Day 03 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_002` | Day 03 | `lighting_preset` | Night Highway with Lamps | Clear Noon (70° Sun) |
| `SEG_002` | Day 03 | `road_health_state` | Minor Wear (Grade B) | Moderate Deterioration (Grade C) |
| `SEG_002` | Day 03 | `pothole_sizing_spectrum` | Micro Pitting & Hairlines | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_002` | Day 03 | `pothole_density_per_100m2` | 1.5 | 3.5 |
| `SEG_002` | Day 03 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_003` | Day 03 | `lighting_preset` | Heavy Rain & Wet Road | Clear Noon (70° Sun) |
| `SEG_003` | Day 03 | `road_health_state` | Minor Wear (Grade B) | Pristine (Grade A) |
| `SEG_003` | Day 03 | `pothole_density_per_100m2` | 1.5 | 7.3 |
| `SEG_003` | Day 03 | `pothole_moisture_state` | 100% Waterlogged Puddles | Mixed Wet / Dry Cavities |
| `SEG_003` | Day 03 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_004` | Day 03 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_004` | Day 03 | `road_health_state` | Minor Wear (Grade B) | Moderate Deterioration (Grade C) |
| `SEG_004` | Day 03 | `pothole_sizing_spectrum` | Micro Pitting & Hairlines | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_004` | Day 03 | `pothole_density_per_100m2` | 1.5 | 3.5 |
| `SEG_001` | Day 04 | `road_health_state` | Moderate Deterioration (Grade C) | Critical Hazard (Grade F) |
| `SEG_001` | Day 04 | `pothole_density_per_100m2` | 3.5 | 4.8 |
| `SEG_001` | Day 04 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_002` | Day 04 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_002` | Day 04 | `pothole_density_per_100m2` | 3.5 | 5.0 |
| `SEG_002` | Day 04 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 💧 Waterlogged Pothole Macro View |
| `SEG_003` | Day 04 | `road_health_state` | Moderate Deterioration (Grade C) | Pristine (Grade A) |
| `SEG_003` | Day 04 | `pothole_sizing_spectrum` | Multi-Scale Organic (20cm - 1.6m) | Micro Pitting & Hairlines |
| `SEG_003` | Day 04 | `pothole_density_per_100m2` | 3.5 | 7.3 |
| `SEG_003` | Day 04 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_003` | Day 04 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_004` | Day 04 | `road_health_state` | Moderate Deterioration (Grade C) | Severe Breakdown (Grade D - Critical) |
| `SEG_004` | Day 04 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_001` | Day 05 | `lighting_preset` | Heavy Rain & Wet Road | Clear Noon (70° Sun) |
| `SEG_001` | Day 05 | `pothole_density_per_100m2` | 3.5 | 5.6 |
| `SEG_001` | Day 05 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_002` | Day 05 | `lighting_preset` | Clear Noon (70° Sun) | Overcast Day |
| `SEG_002` | Day 05 | `pothole_density_per_100m2` | 3.5 | 5.0 |
| `SEG_002` | Day 05 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_002` | Day 05 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 💧 Waterlogged Pothole Macro View |
| `SEG_003` | Day 05 | `lighting_preset` | Night Highway with Lamps | Clear Noon (70° Sun) |
| `SEG_003` | Day 05 | `road_health_state` | Moderate Deterioration (Grade C) | Pristine (Grade A) |
| `SEG_003` | Day 05 | `pothole_sizing_spectrum` | Multi-Scale Organic (20cm - 1.6m) | Micro Pitting & Hairlines |
| `SEG_003` | Day 05 | `pothole_density_per_100m2` | 3.5 | 7.3 |
| `SEG_003` | Day 05 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_003` | Day 05 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_004` | Day 05 | `lighting_preset` | Heavy Rain & Wet Road | Overcast Day |
| `SEG_004` | Day 05 | `road_health_state` | Moderate Deterioration (Grade C) | Severe Breakdown (Grade D - Critical) |
| `SEG_004` | Day 05 | `pothole_moisture_state` | 100% Waterlogged Puddles | 100% Dry Crushed Aggregate |
| `SEG_001` | Day 06 | `lighting_preset` | Golden Hour Sunset | Overcast Day |
| `SEG_001` | Day 06 | `pothole_density_per_100m2` | 3.5 | 5.6 |
| `SEG_001` | Day 06 | `pothole_moisture_state` | Mixed Wet / Dry Cavities | 100% Waterlogged Puddles |
| `SEG_001` | Day 06 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_002` | Day 06 | `lighting_preset` | Heavy Rain & Wet Road | Clear Noon (70° Sun) |
| `SEG_002` | Day 06 | `road_health_state` | Severe Breakdown (Grade D - Critical) | Moderate Deterioration (Grade C) |
| `SEG_002` | Day 06 | `pothole_density_per_100m2` | 6.0 | 5.8 |
| `SEG_002` | Day 06 | `pothole_moisture_state` | 100% Waterlogged Puddles | Mixed Wet / Dry Cavities |
| `SEG_002` | Day 06 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_003` | Day 06 | `lighting_preset` | Overcast Day | Clear Noon (70° Sun) |
| `SEG_003` | Day 06 | `road_health_state` | Moderate Deterioration (Grade C) | Pristine (Grade A) |
| `SEG_003` | Day 06 | `pothole_sizing_spectrum` | Multi-Scale Organic (20cm - 1.6m) | Micro Pitting & Hairlines |
| `SEG_003` | Day 06 | `pothole_density_per_100m2` | 3.5 | 7.3 |
| `SEG_003` | Day 06 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_004` | Day 06 | `pothole_density_per_100m2` | 6.0 | 3.5 |
| `SEG_004` | Day 06 | `pothole_moisture_state` | Mixed Wet / Dry Cavities | 100% Dry Crushed Aggregate |
| `SEG_004` | Day 06 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_001` | Day 07 | `lighting_preset` | Night Highway with Lamps | Overcast Day |
| `SEG_001` | Day 07 | `pothole_density_per_100m2` | 6.0 | 6.6 |
| `SEG_001` | Day 07 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | 100% Waterlogged Puddles |
| `SEG_001` | Day 07 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_002` | Day 07 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_002` | Day 07 | `pothole_density_per_100m2` | 6.0 | 5.8 |
| `SEG_002` | Day 07 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_003` | Day 07 | `lighting_preset` | Night Highway with Lamps | Clear Noon (70° Sun) |
| `SEG_003` | Day 07 | `road_health_state` | Severe Breakdown (Grade D - Critical) | Pristine (Grade A) |
| `SEG_003` | Day 07 | `pothole_sizing_spectrum` | Multi-Scale Organic (20cm - 1.6m) | Micro Pitting & Hairlines |
| `SEG_003` | Day 07 | `pothole_density_per_100m2` | 6.0 | 7.3 |
| `SEG_003` | Day 07 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_003` | Day 07 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_004` | Day 07 | `lighting_preset` | Clear Noon (70° Sun) | Heavy Rain & Wet Road |
| `SEG_004` | Day 07 | `pothole_density_per_100m2` | 6.0 | 3.5 |
| `SEG_004` | Day 07 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_001` | Day 08 | `lighting_preset` | Clear Noon (70° Sun) | Golden Hour Sunset |
| `SEG_001` | Day 08 | `pothole_density_per_100m2` | 6.0 | 7.1 |
| `SEG_001` | Day 08 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_001` | Day 08 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_002` | Day 08 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_002` | Day 08 | `pothole_density_per_100m2` | 6.0 | 3.5 |
| `SEG_003` | Day 08 | `lighting_preset` | Dense Atmospheric Fog | Clear Noon (70° Sun) |
| `SEG_003` | Day 08 | `road_health_state` | Severe Breakdown (Grade D - Critical) | Moderate Deterioration (Grade C) |
| `SEG_003` | Day 08 | `pothole_density_per_100m2` | 6.0 | 3.5 |
| `SEG_004` | Day 08 | `lighting_preset` | Overcast Day | Heavy Rain & Wet Road |
| `SEG_004` | Day 08 | `pothole_density_per_100m2` | 6.0 | 3.5 |
| `SEG_004` | Day 08 | `pothole_moisture_state` | Mixed Wet / Dry Cavities | 100% Dry Crushed Aggregate |
| `SEG_004` | Day 08 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_001` | Day 09 | `lighting_preset` | Golden Hour Sunset | Clear Noon (70° Sun) |
| `SEG_001` | Day 09 | `road_health_state` | Severe Breakdown (Grade D - Critical) | Critical Hazard (Grade F) |
| `SEG_001` | Day 09 | `pothole_sizing_spectrum` | Multi-Scale Organic (20cm - 1.6m) | Micro Pitting & Hairlines |
| `SEG_001` | Day 09 | `pothole_density_per_100m2` | 6.0 | 10.0 |
| `SEG_001` | Day 09 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_002` | Day 09 | `lighting_preset` | Night Highway with Lamps | Clear Noon (70° Sun) |
| `SEG_002` | Day 09 | `road_health_state` | Critical Hazard (Grade F) | Severe Breakdown (Grade D - Critical) |
| `SEG_002` | Day 09 | `pothole_sizing_spectrum` | Large Severe Craters Only | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_002` | Day 09 | `pothole_density_per_100m2` | 8.5 | 5.3 |
| `SEG_002` | Day 09 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_002` | Day 09 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 💧 Waterlogged Pothole Macro View |
| `SEG_003` | Day 09 | `lighting_preset` | Night Highway with Lamps | Clear Noon (70° Sun) |
| `SEG_003` | Day 09 | `pothole_density_per_100m2` | 6.0 | 7.0 |
| `SEG_003` | Day 09 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_004` | Day 09 | `lighting_preset` | Night Highway with Lamps | Golden Hour Sunset |
| `SEG_004` | Day 09 | `pothole_sizing_spectrum` | Large Severe Craters Only | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_004` | Day 09 | `pothole_density_per_100m2` | 8.5 | 4.5 |
| `SEG_004` | Day 09 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_001` | Day 10 | `lighting_preset` | Clear Noon (70° Sun) | Golden Hour Sunset |
| `SEG_001` | Day 10 | `pothole_sizing_spectrum` | Large Severe Craters Only | Micro Pitting & Hairlines |
| `SEG_001` | Day 10 | `pothole_density_per_100m2` | 8.5 | 10.0 |
| `SEG_001` | Day 10 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | 100% Waterlogged Puddles |
| `SEG_001` | Day 10 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_002` | Day 10 | `lighting_preset` | Night Highway with Lamps | Clear Noon (70° Sun) |
| `SEG_002` | Day 10 | `pothole_sizing_spectrum` | Large Severe Craters Only | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_002` | Day 10 | `pothole_density_per_100m2` | 8.5 | 5.3 |
| `SEG_002` | Day 10 | `pothole_moisture_state` | 100% Dry Crushed Aggregate | Mixed Wet / Dry Cavities |
| `SEG_002` | Day 10 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |
| `SEG_003` | Day 10 | `metadata_sidecar` | `day_10_metadata.json` | **MISSING_SIDECAR** |
| `SEG_003` | Day 10 | `lighting_preset` | Overcast Day | MISSING |
| `SEG_003` | Day 10 | `road_health_state` | Critical Hazard (Grade F) | legacy: Extensive Block Cracking |
| `SEG_003` | Day 10 | `pothole_sizing_spectrum` | Large Severe Craters Only | MISSING |
| `SEG_003` | Day 10 | `pothole_density_per_100m2` | 8.5 | MISSING |
| `SEG_003` | Day 10 | `pothole_moisture_state` | Mixed Wet / Dry Cavities | MISSING |
| `SEG_003` | Day 10 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | MISSING |
| `SEG_004` | Day 10 | `lighting_preset` | Overcast Day | Clear Noon (70° Sun) |
| `SEG_004` | Day 10 | `pothole_sizing_spectrum` | Large Severe Craters Only | Multi-Scale Organic (20cm - 1.6m) |
| `SEG_004` | Day 10 | `pothole_density_per_100m2` | 8.5 | 4.5 |
| `SEG_004` | Day 10 | `pothole_moisture_state` | Mixed Wet / Dry Cavities | 100% Dry Crushed Aggregate |
| `SEG_004` | Day 10 | `camera_preset` | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | 🌄 Highway Curve Vantage Overlook |

---

## 4. Complete Planned 60-Image Capture Verification Table

| Segment | Day | Image Status | Sidecar Status | Settings Match | Captures in History | Status | Notes |
|---|---|---|---|---|---|---|---|
| `SEG_001` | 01 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (density: expected 0.5, got 2.8; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities') |
| `SEG_002` | 01 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 2 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Overcast Day', got 'Clear Noon (70° Sun)'; pothole_sizing_spectrum: expected 'Micro Pitting & Hairlines', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 0.5, got 10.0; pothole_moisture_state: expected 'Mixed Wet / Dry Cavities', got '100% Dry Crushed Aggregate'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🔍 Low-Angle Pothole Inspection (30° Close-Up)'); DUPLICATE_HISTORY (2 captures logged) |
| `SEG_003` | 01 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; density: expected 0.5, got 7.3; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_004` | 01 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 2 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Pristine (Grade A)', got 'Moderate Deterioration (Grade C)'; pothole_sizing_spectrum: expected 'Micro Pitting & Hairlines', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 0.5, got 3.5); DUPLICATE_HISTORY (2 captures logged) |
| `SEG_005` | 01 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 01 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 02 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Golden Hour Sunset', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Minor Wear (Grade B)', got 'Critical Hazard (Grade F)'; density: expected 1.5, got 3.8; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '💧 Waterlogged Pothole Macro View') |
| `SEG_002` | 02 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Minor Wear (Grade B)', got 'Pristine (Grade A)'; pothole_sizing_spectrum: expected 'Micro Pitting & Hairlines', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 1.5, got 10.0; pothole_moisture_state: expected 'Mixed Wet / Dry Cavities', got '100% Dry Crushed Aggregate'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🔍 Low-Angle Pothole Inspection (30° Close-Up)') |
| `SEG_003` | 02 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Minor Wear (Grade B)', got 'Pristine (Grade A)'; density: expected 1.5, got 7.3; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_004` | 02 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Golden Hour Sunset'; pothole_sizing_spectrum: expected 'Micro Pitting & Hairlines', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 1.5, got 3.5) |
| `SEG_005` | 02 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 02 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 03 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 2 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Minor Wear (Grade B)', got 'Moderate Deterioration (Grade C)'; pothole_sizing_spectrum: expected 'Micro Pitting & Hairlines', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 1.5, got 4.8; pothole_moisture_state: expected 'Mixed Wet / Dry Cavities', got '100% Dry Crushed Aggregate'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook'); DUPLICATE_HISTORY (2 captures logged) |
| `SEG_002` | 03 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 4 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Night Highway with Lamps', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Minor Wear (Grade B)', got 'Moderate Deterioration (Grade C)'; pothole_sizing_spectrum: expected 'Micro Pitting & Hairlines', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 1.5, got 3.5; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'); DUPLICATE_HISTORY (4 captures logged) |
| `SEG_003` | 03 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Heavy Rain & Wet Road', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Minor Wear (Grade B)', got 'Pristine (Grade A)'; density: expected 1.5, got 7.3; pothole_moisture_state: expected '100% Waterlogged Puddles', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_004` | 03 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 4 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Minor Wear (Grade B)', got 'Moderate Deterioration (Grade C)'; pothole_sizing_spectrum: expected 'Micro Pitting & Hairlines', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 1.5, got 3.5); DUPLICATE_HISTORY (4 captures logged) |
| `SEG_005` | 03 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 03 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 04 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (road_health_state: expected 'Moderate Deterioration (Grade C)', got 'Critical Hazard (Grade F)'; density: expected 3.5, got 4.8; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_002` | 04 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; density: expected 3.5, got 5.0; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '💧 Waterlogged Pothole Macro View') |
| `SEG_003` | 04 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (road_health_state: expected 'Moderate Deterioration (Grade C)', got 'Pristine (Grade A)'; pothole_sizing_spectrum: expected 'Multi-Scale Organic (20cm - 1.6m)', got 'Micro Pitting & Hairlines'; density: expected 3.5, got 7.3; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_004` | 04 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (road_health_state: expected 'Moderate Deterioration (Grade C)', got 'Severe Breakdown (Grade D - Critical)'; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities') |
| `SEG_005` | 04 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 04 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 05 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Heavy Rain & Wet Road', got 'Clear Noon (70° Sun)'; density: expected 3.5, got 5.6; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_002` | 05 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Clear Noon (70° Sun)', got 'Overcast Day'; density: expected 3.5, got 5.0; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '💧 Waterlogged Pothole Macro View') |
| `SEG_003` | 05 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Night Highway with Lamps', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Moderate Deterioration (Grade C)', got 'Pristine (Grade A)'; pothole_sizing_spectrum: expected 'Multi-Scale Organic (20cm - 1.6m)', got 'Micro Pitting & Hairlines'; density: expected 3.5, got 7.3; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_004` | 05 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Heavy Rain & Wet Road', got 'Overcast Day'; road_health_state: expected 'Moderate Deterioration (Grade C)', got 'Severe Breakdown (Grade D - Critical)'; pothole_moisture_state: expected '100% Waterlogged Puddles', got '100% Dry Crushed Aggregate') |
| `SEG_005` | 05 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 05 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 06 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Golden Hour Sunset', got 'Overcast Day'; density: expected 3.5, got 5.6; pothole_moisture_state: expected 'Mixed Wet / Dry Cavities', got '100% Waterlogged Puddles'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_002` | 06 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Heavy Rain & Wet Road', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Severe Breakdown (Grade D - Critical)', got 'Moderate Deterioration (Grade C)'; density: expected 6.0, got 5.8; pothole_moisture_state: expected '100% Waterlogged Puddles', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_003` | 06 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Overcast Day', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Moderate Deterioration (Grade C)', got 'Pristine (Grade A)'; pothole_sizing_spectrum: expected 'Multi-Scale Organic (20cm - 1.6m)', got 'Micro Pitting & Hairlines'; density: expected 3.5, got 7.3; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_004` | 06 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (density: expected 6.0, got 3.5; pothole_moisture_state: expected 'Mixed Wet / Dry Cavities', got '100% Dry Crushed Aggregate'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_005` | 06 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 06 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 07 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Night Highway with Lamps', got 'Overcast Day'; density: expected 6.0, got 6.6; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got '100% Waterlogged Puddles'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_002` | 07 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; density: expected 6.0, got 5.8; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_003` | 07 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Night Highway with Lamps', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Severe Breakdown (Grade D - Critical)', got 'Pristine (Grade A)'; pothole_sizing_spectrum: expected 'Multi-Scale Organic (20cm - 1.6m)', got 'Micro Pitting & Hairlines'; density: expected 6.0, got 7.3; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_004` | 07 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Clear Noon (70° Sun)', got 'Heavy Rain & Wet Road'; density: expected 6.0, got 3.5; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_005` | 07 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 07 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 08 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Clear Noon (70° Sun)', got 'Golden Hour Sunset'; density: expected 6.0, got 7.1; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_002` | 08 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 3 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; density: expected 6.0, got 3.5); DUPLICATE_HISTORY (3 captures logged) |
| `SEG_003` | 08 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 2 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Dense Atmospheric Fog', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Severe Breakdown (Grade D - Critical)', got 'Moderate Deterioration (Grade C)'; density: expected 6.0, got 3.5); DUPLICATE_HISTORY (2 captures logged) |
| `SEG_004` | 08 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Overcast Day', got 'Heavy Rain & Wet Road'; density: expected 6.0, got 3.5; pothole_moisture_state: expected 'Mixed Wet / Dry Cavities', got '100% Dry Crushed Aggregate'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_005` | 08 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 08 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 09 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 5 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Golden Hour Sunset', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Severe Breakdown (Grade D - Critical)', got 'Critical Hazard (Grade F)'; pothole_sizing_spectrum: expected 'Multi-Scale Organic (20cm - 1.6m)', got 'Micro Pitting & Hairlines'; density: expected 6.0, got 10.0; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook'); DUPLICATE_HISTORY (5 captures logged) |
| `SEG_002` | 09 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Night Highway with Lamps', got 'Clear Noon (70° Sun)'; road_health_state: expected 'Critical Hazard (Grade F)', got 'Severe Breakdown (Grade D - Critical)'; pothole_sizing_spectrum: expected 'Large Severe Craters Only', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 8.5, got 5.3; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '💧 Waterlogged Pothole Macro View') |
| `SEG_003` | 09 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 2 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Night Highway with Lamps', got 'Clear Noon (70° Sun)'; density: expected 6.0, got 7.0; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'); DUPLICATE_HISTORY (2 captures logged) |
| `SEG_004` | 09 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Night Highway with Lamps', got 'Golden Hour Sunset'; pothole_sizing_spectrum: expected 'Large Severe Craters Only', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 8.5, got 4.5; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_005` | 09 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 09 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_001` | 10 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 7 | `SETTING_MISMATCH; DUPLICATE_HISTORY` | SETTING_MISMATCH (lighting_preset: expected 'Clear Noon (70° Sun)', got 'Golden Hour Sunset'; pothole_sizing_spectrum: expected 'Large Severe Craters Only', got 'Micro Pitting & Hairlines'; density: expected 8.5, got 10.0; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got '100% Waterlogged Puddles'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook'); DUPLICATE_HISTORY (7 captures logged) |
| `SEG_002` | 10 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Night Highway with Lamps', got 'Clear Noon (70° Sun)'; pothole_sizing_spectrum: expected 'Large Severe Craters Only', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 8.5, got 5.3; pothole_moisture_state: expected '100% Dry Crushed Aggregate', got 'Mixed Wet / Dry Cavities'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_003` | 10 | ✅ Valid | ⚠️ Missing | N/A | 2 | `MISSING_SIDECAR; DUPLICATE_HISTORY` | MISSING_SIDECAR (day_XX_metadata.json missing; found legacy metadata.json); DUPLICATE_HISTORY (2 captures logged) |
| `SEG_004` | 10 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | 1 | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: expected 'Overcast Day', got 'Clear Noon (70° Sun)'; pothole_sizing_spectrum: expected 'Large Severe Craters Only', got 'Multi-Scale Organic (20cm - 1.6m)'; density: expected 8.5, got 4.5; pothole_moisture_state: expected 'Mixed Wet / Dry Cavities', got '100% Dry Crushed Aggregate'; camera_preset: expected '🔭 Overhead Drone Survey (SAM 2 Top-Down)', got '🌄 Highway Curve Vantage Overlook') |
| `SEG_005` | 10 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
| `SEG_006` | 10 | ⚠️ Missing | ⚠️ Missing | N/A | 0 | ⏳ NOT YET CAPTURED | OK |
