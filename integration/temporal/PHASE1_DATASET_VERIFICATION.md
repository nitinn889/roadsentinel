# RoadSentinel — Phase 1: Dataset Verification Report

**Generated**: 2026-09-08T18:44:54.913388+00:00  
**Verification Method**: Strict File-Only Audit (zero Unreal/CARLA interaction)  
**Overall Status**: **PARTIAL PASS — CAPTURE IN PROGRESS**

---

## 1. Audit Findings

1. **Images Audit**:
   - Total expected: 60 (1920x1080 PNG)
   - Found on disk: **21/60**
   - Corrupted/truncated PNGs: **0**
   - Valid resolution & format: **21**

2. **Sidecar Metadata Audit**:
   - Total expected: 60 (`day_XX_metadata.json`)
   - Found on disk: **3/60**
   - Notes: Existing capture directory currently has legacy segment-level `metadata.json` and `segment_history.json`. Individual per-image sidecars (`day_XX_metadata.json`) will be written by `rs_capture_metadata_logger.py` during ongoing captures.

3. **Manifest & History Audit**:
   - `dataset_manifest.csv`: Not yet created (will be created as captures occur)
   - `capture_history.jsonl`: Not yet created (will be created as captures occur)
   - Repeated/duplicate captures in history: **0**

4. **Repair Event Verification**:
   - **SEG_005**:
     - Capture Plan: Day 05 = `Critical Hazard (Grade F)` (`pre-repair`), Day 06 = `Pristine (Grade A)` (`repair`) → **YES**
     - Sidecar Metadata: **PENDING_CAPTURE** (awaiting capture of SEG_005 Days 05 and 06)
   - **SEG_006**:
     - Capture Plan: Day 05 = `Critical Hazard (Grade F)` (`pre-repair`), Day 06 = `Pristine (Grade A)` (`repair`) → **YES**
     - Sidecar Metadata: **PENDING_CAPTURE** (awaiting capture of SEG_006 Days 05 and 06)

---

## 2. Complete Planned 60-Image Capture Table

| Segment | Day | Image Status | Sidecar Status | Settings Match | Overall Status | Notes |
|---|---|---|---|---|---|---|
| `SEG_001` | 01 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 01 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 01 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: missing from metadata; road_health_state: missing standard field (found legacy deterioration_state 'Pristine Asphalt'); pothole_sizing_spectrum: missing from metadata; pothole_density_per_100m2: missing from metadata; pothole_moisture_state: missing from metadata; camera_preset: missing from metadata) |
| `SEG_004` | 01 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 01 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 01 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 02 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 02 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 02 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 02 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 02 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 02 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 03 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 03 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 03 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 03 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 03 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 03 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 04 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 04 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 04 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 04 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 04 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 04 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 05 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 05 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 05 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 05 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 05 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 05 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 06 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 06 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 06 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 06 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 06 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 06 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 07 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 07 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 07 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 07 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 07 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 07 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 08 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 08 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 08 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 08 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 08 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 08 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 09 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_002` | 09 | ✅ Valid | ⚠️ Missing | N/A | `MISSING_METADATA` | MISSING_METADATA |
| `SEG_003` | 09 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 09 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 09 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 09 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_001` | 10 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: missing from metadata; road_health_state: missing standard field (found legacy deterioration_state 'Critical Hazard: Severe Waterlogged Crater'); pothole_sizing_spectrum: missing from metadata; pothole_density_per_100m2: missing from metadata; pothole_moisture_state: missing from metadata; camera_preset: missing from metadata) |
| `SEG_002` | 10 | ✅ Valid | ✅ Valid | ⚠️ Mismatch | `SETTING_MISMATCH` | SETTING_MISMATCH (lighting_preset: missing from metadata; road_health_state: missing standard field (found legacy deterioration_state 'Severe Waterlogged Puddle Hazard'); pothole_sizing_spectrum: missing from metadata; pothole_density_per_100m2: missing from metadata; pothole_moisture_state: missing from metadata; camera_preset: missing from metadata) |
| `SEG_003` | 10 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_004` | 10 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_005` | 10 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
| `SEG_006` | 10 | ⚠️ Missing | ⚠️ Missing | N/A | `MISSING_IMAGE; MISSING_METADATA` | MISSING_IMAGE; MISSING_METADATA |
