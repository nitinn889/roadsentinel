# RoadSentinel — Phase 1: Temporal Dataset Progress

**Date**: 2026-09-09T12:49:27.390511+00:00  
**Overall Phase 1 Status**: **PARTIAL PASS / READY FOR EXPERIMENT-A PHASE 2**  
**Experiment A Readiness**: **EXPERIMENT_A_NEAR_READY (39/40 metadata-complete)**  
**Experiment B Status**: **NOT YET CAPTURED (Reserved for SEG_005 & SEG_006)**

---

> [!IMPORTANT]
> **SUPERSEDING PROJECT DECISION (PHASE 1D):**  
> The Team Lead intentionally varied simulation parameters during the capture of `SEG_001`–`SEG_004` to create a realistic, multi-condition robustness dataset.  
> The previous `SETTING_MISMATCH` classification against `temporal_capture_plan.csv` is **officially superseded**.  
> The **actual per-image sidecar metadata (`day_XX_metadata.json`) is the authoritative source of truth**.

---

## 1. Dual-Experiment Architecture

1. **EXPERIMENT A: Multi-Condition Road-Perception Robustness (SEG_001–SEG_004)**
   - **40 simulated road-inspection states** captured under varying environmental lighting, defect densities, moisture levels, and camera viewpoints.
   - **Intended Purpose**: YOLO defect detection benchmark, DINOv2+SAM 2 zero-shot segmentation, viewpoint angle robustness, severe weather/lighting generalization, and qualitative failure analysis.
   - *Note*: Not treated as continuous physical deterioration sequences, except for identified contiguous same-camera candidate subsets.

2. **EXPERIMENT B: Controlled Temporal Progression + Repair (SEG_005–SEG_006)**
   - Reserved for the cleaner, controlled temporal sequence to be captured manually by the Team Lead.
   - Consistent top-down drone camera perspective across Days 01–10.
   - **Day 01 → Day 05**: Monotonic deterioration (Pristine -> Minor -> Moderate -> Severe -> Critical).
   - **Day 06**: Formal repair intervention (Pristine patch reset).
   - **Day 07 → Day 10**: Post-repair re-deterioration.

---

## 2. Experiment A: Executive Metrics & Completeness

| Metric | Target | Actual | Status |
|---|---|---|---|
| **Physical Images Found** | 40 | 40/40 | ✅ 100% Found (1920x1080 PNG) |
| **Proper Sidecars (`day_XX_metadata.json`)** | 40 | 39/40 | ⚠️ 39/40 (SEG_003 Day 10 missing) |
| **Usable Captures (Image + Valid Sidecar)** | 40 | **39/40** | ✅ **97.5% Complete** |
| **Corrupt / Truncated Images** | 0 | 0 | ✅ Zero corruption |
| **Unique Lighting Conditions** | N/A | 4 | Clear Noon, Overcast, Golden Hour, Heavy Rain |
| **Unique Road Health States** | N/A | 5 | Pristine, Minor, Moderate, Severe, Critical |
| **Unique Camera Viewpoints** | N/A | 4 | Overlook, Drone Survey, Macro View, Low-Angle 30° |
| **Defect Density Range** | N/A | 2.8 – 10.0 / 100m² | Mean: 5.61, Median: 5.3 |

---

## 3. Segment Completeness & Usability

| Segment ID | Physical Images | Proper Sidecars | Usable Captures | Primary Camera Viewpoint | Usability Mode |
|---|---|---|---|---|---|
| **SEG_001** | 10/10 | 10/10 | **10/10** | 8 days Overlook (D03–D10) | Multi-condition robustness + 8-day temporal candidate |
| **SEG_002** | 10/10 | 10/10 | **10/10** | Low-Angle, Macro, Overlook | Viewpoint & moisture robustness + pairwise temporal |
| **SEG_003** | 10/10 | 9/10 | **9/10** | 7 days Overlook (D01–D07) | 7-day stability candidate + 2-day drone (D10 missing sidecar) |
| **SEG_004** | 10/10 | 10/10 | **10/10** | 5 days Drone (D01–D05), 5 days Overlook (D06–D10) | Dual 5-day temporal candidates under varied weather |
| **TOTAL (Exp A)** | **40/40** | **39/40** | **39/40** | **4 distinct perspectives** | **EXPERIMENT_A_NEAR_READY (Ready for Phase 2)** |
| **SEG_005 (Exp B)**| 0/10 | 0/10 | 0/10 | Consistent Top-Down Drone | **NOT YET CAPTURED** (Controlled Temporal + Repair) |
| **SEG_006 (Exp B)**| 0/10 | 0/10 | 0/10 | Consistent Top-Down Drone | **NOT YET CAPTURED** (Controlled Temporal + Repair) |

---

## 4. Same-Camera Temporal Candidate Subsequences (Experiment A)

Although captured as a multi-condition robustness dataset, the actual metadata reveals contiguous same-camera sequences suitable for temporal tracking:

1. **SEG_001 Day 03–10 (8 days)**: `🌄 Highway Curve Vantage Overlook`
   - Progressive density drift from 4.8 to 10.0 / 100m²; health transitions Grade C -> Grade F.
2. **SEG_002 Day 01–02 (2 days)**: `🔍 Low-Angle Pothole Inspection (30° Close-Up)`
3. **SEG_002 Day 04–05 (2 days)**: `💧 Waterlogged Pothole Macro View`
4. **SEG_002 Day 06–07 (2 days)**: `🌄 Highway Curve Vantage Overlook`
5. **SEG_003 Day 01–07 (7 days)**: `🌄 Highway Curve Vantage Overlook`
   - Pristine Grade A roadway at constant density 7.3 under Clear Noon; ideal for false-positive stability testing.
6. **SEG_003 Day 08–09 (2 days)**: `🔭 Overhead Drone Survey (SAM 2 Top-Down)`
7. **SEG_004 Day 01–05 (5 days)**: `🔭 Overhead Drone Survey (SAM 2 Top-Down)`
   - Progressive breakdown from Grade C/B to Grade D across Clear Noon, Sunset, and Overcast skies.
8. **SEG_004 Day 06–10 (5 days)**: `🌄 Highway Curve Vantage Overlook`
   - Heavy breakdown and hazard progression under severe weather (Overcast, Heavy Rain, Sunset).

---

## 5. Experiment A Phase 2 Readiness Gate

**Verdict**: **`EXPERIMENT_A_NEAR_READY`**

- **39 of 40 captures** possess verified 1920x1080 images and complete, valid sidecar metadata.
- **Perception evaluation pipelines (YOLO, DINOv2, SAM 2) may proceed immediately on the 39 verified captures.**
- SEG_003 Day 10 is documented as missing its sidecar; the Team Lead may optionally re-capture it at convenience without blocking Experiment A perception work.
