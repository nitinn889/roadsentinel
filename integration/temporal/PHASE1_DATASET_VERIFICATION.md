# RoadSentinel — Phase 1: Dataset Verification Report (Phase 1D Reframe)

**Generated**: 2026-09-09T12:49:27.390511+00:00  
**Verification Method**: Strict File-Only Audit (zero Unreal/CARLA interaction)  
**Overall Phase 1 Status**: **PARTIAL PASS / READY FOR EXPERIMENT-A PHASE 2**  
**Experiment A Readiness**: **EXPERIMENT_A_NEAR_READY (39/40 metadata-complete)**  
**Experiment B Status**: **NOT YET CAPTURED (SEG_005 & SEG_006 reserved)**

---

> [!NOTE]
> **REVISION NOTICE (PHASE 1D):**  
> The previous `SETTING_MISMATCH` classification against `temporal_capture_plan.csv` for `SEG_001`–`SEG_004` is officially superseded.  
> The simulation settings were intentionally varied by the Team Lead during interactive capture.  
> The **actual per-image sidecar metadata (`day_XX_metadata.json`) is the authoritative source of truth**.

---

## 1. Dual-Experiment Framing

### Experiment A: Multi-Condition Road-Perception Robustness
- **Segments**: `SEG_001`, `SEG_002`, `SEG_003`, `SEG_004` (40 captures total).
- **Characterization**: Simulated road-inspection states captured under varied road-health, environmental lighting, moisture, defect-density, and camera conditions.
- **Downstream Capabilities**:
  - YOLO object detection benchmarking across viewpoints.
  - DINOv2 foundation feature extraction & SAM 2 prompt-based segmentation.
  - Viewpoint robustness (Overlook vs Drone Survey vs Macro vs Low-Angle).
  - Condition robustness (Clear Noon vs Overcast vs Rain vs Sunset).
  - Severity distribution & qualitative failure mode analysis.

### Experiment B: Controlled Temporal Progression + Repair
- **Segments**: `SEG_005`, `SEG_006` (20 captures total).
- **Characterization**: Reserved for clean, monotonic deterioration and repair sequences captured with consistent top-down drone camera.
- **Sequence**:
  - Days 01–05: Progressive deterioration (Pristine -> Minor -> Moderate -> Severe -> Critical).
  - Day 06: Repair / patch reset intervention (Pristine).
  - Days 07–10: Re-deterioration.
- **Current Status**: **NOT YET CAPTURED** (Team Lead will capture later).

---

## 2. Experiment A Audit Findings

1. **Physical Images**:
   - Total planned: 40 (1920x1080 PNG)
   - Found on disk: **40/40**
   - Valid resolution & raster data: **40/40**
   - Corrupted/truncated PNGs: **0**
   - File size range: 1,173,227 bytes to 5,854,635 bytes (zero tiny files).

2. **Per-Image Sidecar Metadata (`day_XX_metadata.json`)**:
   - Expected: 40 sidecars
   - Found and valid JSON: **39/40**
   - **Missing sidecar**: `SEG_003/day_10_metadata.json` (documented issue; studio logged capture error).
   - Note: Legacy segment-level `metadata.json` in `SEG_003` exists, but is not an authoritative per-image sidecar.

3. **Metadata Logger Outputs**:
   - `dataset_manifest.csv`: **39 rows recorded** (missing `SEG_003 Day 10`).
   - `capture_history.jsonl`: **64 entries**, actively recording captures, duplicates, and errors.

---

## 3. Condition Distribution Summary (Experiment A)

### Lighting / Environmental Presets
| Lighting Preset | Images | Percentage |
|---|---|---|
| `Clear Noon (70° Sun)` | 28 | 71.8% |
| `Overcast Day` | 5 | 12.8% |
| `Golden Hour Sunset` | 4 | 10.3% |
| `Heavy Rain & Wet Road` | 2 | 5.1% |

### Road-Health Severity States
| Road-Health State | Images | Percentage |
|---|---|---|
| `Severe Breakdown (Grade D - Critical)` | 11 | 28.2% |
| `Pristine (Grade A)` | 10 | 25.6% |
| `Moderate Deterioration (Grade C)` | 10 | 25.6% |
| `Critical Hazard (Grade F)` | 7 | 17.9% |
| `Minor Wear (Grade B)` | 1 | 2.6% |

### Camera Viewpoints
| Camera Preset | Images | Percentage | Usability |
|---|---|---|---|
| `🌄 Highway Curve Vantage Overlook` | 23 | 59.0% | Robustness + 8-day & 7-day & 5-day Temporal Candidates |
| `🔭 Overhead Drone Survey (SAM 2 Top-Down)` | 10 | 25.6% | Robustness + 5-day & 2-day Temporal Candidates |
| `💧 Waterlogged Pothole Macro View` | 4 | 10.3% | Macro Robustness + 2-day Temporal Candidate |
| `🔍 Low-Angle Pothole Inspection (30° Close-Up)` | 2 | 5.1% | Low-Angle Robustness + 2-day Temporal Candidate |

### Defect Density Metrics
| Metric | Value |
|---|---|
| Minimum Density | 2.8 / 100m² |
| Maximum Density | 10.0 / 100m² |
| Mean Density | 5.61 / 100m² |
| Median Density | 5.3 / 100m² |

---

## 4. Master Actual-Metadata Inventory (Experiment A)

Full details exported in [actual_capture_inventory.csv](actual_capture_inventory.csv).

| Segment | Day | Image Status | Sidecar Status | Camera Preset | Lighting Preset | Health State | Density | Usability Mode |
|---|---|---|---|---|---|---|---|---|
| `SEG_001` | 01 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Pristine (Grade A) | 2.8 | `ROBUSTNESS_ONLY` |
| `SEG_002` | 01 | ✅ Valid | ✅ Valid | 🔍 Low-Angle Pothole Inspection (30° Close-Up) | Clear Noon (70° Sun) | Pristine (Grade A) | 10.0 | `TEMPORAL_CANDIDATE` |
| `SEG_003` | 01 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Pristine (Grade A) | 7.3 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 01 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 02 | ✅ Valid | ✅ Valid | 💧 Waterlogged Pothole Macro View | Clear Noon (70° Sun) | Critical Hazard (Grade F) | 3.8 | `ROBUSTNESS_ONLY` |
| `SEG_002` | 02 | ✅ Valid | ✅ Valid | 🔍 Low-Angle Pothole Inspection (30° Close-Up) | Clear Noon (70° Sun) | Pristine (Grade A) | 10.0 | `TEMPORAL_CANDIDATE` |
| `SEG_003` | 02 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Pristine (Grade A) | 7.3 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 02 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Golden Hour Sunset | Minor Wear (Grade B) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 03 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | 4.8 | `TEMPORAL_CANDIDATE` |
| `SEG_002` | 03 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | 3.5 | `ROBUSTNESS_ONLY` |
| `SEG_003` | 03 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Pristine (Grade A) | 7.3 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 03 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 04 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Critical Hazard (Grade F) | 4.8 | `TEMPORAL_CANDIDATE` |
| `SEG_002` | 04 | ✅ Valid | ✅ Valid | 💧 Waterlogged Pothole Macro View | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | 5.0 | `TEMPORAL_CANDIDATE` |
| `SEG_003` | 04 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Pristine (Grade A) | 7.3 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 04 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 05 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | 5.6 | `TEMPORAL_CANDIDATE` |
| `SEG_002` | 05 | ✅ Valid | ✅ Valid | 💧 Waterlogged Pothole Macro View | Overcast Day | Moderate Deterioration (Grade C) | 5.0 | `TEMPORAL_CANDIDATE` |
| `SEG_003` | 05 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Pristine (Grade A) | 7.3 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 05 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Overcast Day | Severe Breakdown (Grade D - Critical) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 06 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Overcast Day | Moderate Deterioration (Grade C) | 5.6 | `TEMPORAL_CANDIDATE` |
| `SEG_002` | 06 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | 5.8 | `TEMPORAL_CANDIDATE` |
| `SEG_003` | 06 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Pristine (Grade A) | 7.3 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 06 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Overcast Day | Severe Breakdown (Grade D - Critical) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 07 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Overcast Day | Severe Breakdown (Grade D - Critical) | 6.6 | `TEMPORAL_CANDIDATE` |
| `SEG_002` | 07 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | 5.8 | `TEMPORAL_CANDIDATE` |
| `SEG_003` | 07 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Pristine (Grade A) | 7.3 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 07 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Heavy Rain & Wet Road | Severe Breakdown (Grade D - Critical) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 08 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Golden Hour Sunset | Severe Breakdown (Grade D - Critical) | 7.1 | `TEMPORAL_CANDIDATE` |
| `SEG_002` | 08 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | 3.5 | `ROBUSTNESS_ONLY` |
| `SEG_003` | 08 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Moderate Deterioration (Grade C) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 08 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Heavy Rain & Wet Road | Severe Breakdown (Grade D - Critical) | 3.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 09 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Critical Hazard (Grade F) | 10.0 | `TEMPORAL_CANDIDATE` |
| `SEG_002` | 09 | ✅ Valid | ✅ Valid | 💧 Waterlogged Pothole Macro View | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | 5.3 | `ROBUSTNESS_ONLY` |
| `SEG_003` | 09 | ✅ Valid | ✅ Valid | 🔭 Overhead Drone Survey (SAM 2 Top-Down) | Clear Noon (70° Sun) | Severe Breakdown (Grade D - Critical) | 7.0 | `TEMPORAL_CANDIDATE` |
| `SEG_004` | 09 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Golden Hour Sunset | Critical Hazard (Grade F) | 4.5 | `TEMPORAL_CANDIDATE` |
| `SEG_001` | 10 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Golden Hour Sunset | Critical Hazard (Grade F) | 10.0 | `TEMPORAL_CANDIDATE` |
| `SEG_002` | 10 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Critical Hazard (Grade F) | 5.3 | `ROBUSTNESS_ONLY` |
| `SEG_003` | 10 | ✅ Valid | ⚠️ Missing | UNKNOWN | UNKNOWN | UNKNOWN | N/A | `MISSING_DATA` |
| `SEG_004` | 10 | ✅ Valid | ✅ Valid | 🌄 Highway Curve Vantage Overlook | Clear Noon (70° Sun) | Critical Hazard (Grade F) | 4.5 | `TEMPORAL_CANDIDATE` |

---

## 5. Experiment B Status (SEG_005 & SEG_006)

| Segment ID | Day Range | Planned Perspective | Target Event | Current Status |
|---|---|---|---|---|
| `SEG_005` | Day 01–05 | Top-Down Drone Survey | Monotonic Deterioration (Grade A -> F) | **NOT YET CAPTURED** |
| `SEG_005` | Day 06 | Top-Down Drone Survey | Repair Reset (Pristine Grade A) | **NOT YET CAPTURED** |
| `SEG_005` | Day 07–10 | Top-Down Drone Survey | Post-Repair Re-deterioration | **NOT YET CAPTURED** |
| `SEG_006` | Day 01–05 | Top-Down Drone Survey | Monotonic Deterioration (Grade A -> F) | **NOT YET CAPTURED** |
| `SEG_006` | Day 06 | Top-Down Drone Survey | Repair Reset (Pristine Grade A) | **NOT YET CAPTURED** |
| `SEG_006` | Day 07–10 | Top-Down Drone Survey | Post-Repair Re-deterioration | **NOT YET CAPTURED** |
