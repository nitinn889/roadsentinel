# RoadSentinel — Marion Day 5 Progress Report
**Component**: Person 1 / Marion — DINOv2 + SAM2 Defect Localization Quality Improvement  
**Date**: September 8, 2026  
**Status**: **PARTIAL PASS** (Pothole target surpassed, crack overlap +220%, healthy specificity fully preserved, thin-crack resolution physical limit documented)  

---

## 1. Git & Current Baseline

- **Integration Commit**: `2e918bb` (Integrate Vrinda1-5 Marion1-4 Nitin1-4)
- **Branch**: `main` (synchronized with `origin/main`)
- **Pre-Fix Day 4 Baseline Preserved**: Snapshot archived in `sam2_dino/outputs/day5_pre_fix/` prior to any code edits.

---

## 2. Exact Code Files Changed

1. `road_health_pipeline/inference/pothole_localizer.py`:
   - Added candidate pre-consolidation using spatial proximity (`cv2.dilate(candidate_u8, np.ones((21, 21)))`) prior to SAM2 prompting.
   - Implemented directional adaptive context expansion for narrow/elongated defects:
     - Longitudinal cracks (`h_cc >= 1.3 * w` and `w <= 25`): expanded context along y-axis (`pad_y = max(32, min(80, int(h_cc * 1.5)))`, `pad_x = 10`).
     - Transverse cracks (`w >= 1.3 * h_cc` and `h_cc <= 25`): expanded context along x-axis (`pad_x = max(32, min(80, int(w * 1.5)))`, `pad_y = 10`).
     - Isotropic candidates: default padding (`pad = 12` if < 40 else 6).
   - Preserved post-SAM2 touching consolidation (`_consolidate_candidates`).
2. `sam2_dino/test_day5_fixes.py`:
   - Created dedicated test suite (4 unit tests covering pre-consolidation, adaptive context padding, schema compliance, and healthy road regression).
3. `sam2_dino/DAY5_GPU_FOLLOWUP.md`:
   - Formulated high-resolution tiled DINOv2 experiment plan for Team Lead execution on RTX 5060.
4. `sam2_dino/outputs/run_summary.json` & persisted records in `sam2_dino/outputs/`:
   - Refreshed Day 5 persisted outputs for all fixed test samples.

---

## 3. Crack Root-Cause Findings (China_Drone_001267.jpg)

Full inspection of the inference chain on `China_Drone_001267.jpg` revealed:
1. **DINO Patch Resolution & Dilution**:
   - ViT-S/14 operates at 14×14 px on 518×518 input (~13.8 px in 512×512 space).
   - Longitudinal crack width is only 1–3 px.
   - Inside a patch, the crack occupies ~14% of the area; 86% is background asphalt.
   - Genuine crack patch anomaly scores are only slightly elevated (raw scores 0.24–0.32 vs 0.266 road median).
2. **Threshold Mechanics**:
   - Road baseline normalization sets the active threshold at the 92nd percentile of positive normalized scores (0.3515).
   - In GT Box 1 (`[439, 19, 463, 180]`), only 1 patch (r=3, c=33 with norm 0.5041) reached above 0.3515.
   - In GT Box 2 (`[434, 353, 456, 474]`), the highest patch score is 0.3780 at r=34, c=33, but bilinear upsampling of this single-cell peak yielded only 3 pixels above threshold.
3. **Morphological Destruction**:
   - Standard `MORPH_OPEN(3, 3)` required full 3×3 connectivity, obliterating any 1–2 px wide crack island.
4. **SAM2 Truncation in Day 4**:
   - Day 4 prompted SAM2 with a tiny 8px-padded box `[452, 34, 475, 62]`. SAM2 confined segmentation to that local window (height 30 px out of the 161 px tall crack).

---

## 4. Crack Metrics Comparison (Day 3 → Day 4 → Day 5)

Ground truth total rasterized area: 6,526 px.

| Metric | Day 3 | Day 4 Post-Fix | Day 5 Post-Fix | Delta (Day 4 → Day 5) |
|---|---|---|---|---|
| Accepted Defects | 4 | 4 | **3** | Consolidated |
| Total Predicted Pixels | 4,330 px | 6,278 px | **16,102 px** | Includes consolidated distress regions |
| **GT Overlap Pixels** | 0 px | 275 px | **882 px** | **+607 px (+220% improvement)** |
| **Recall** | 0.0000 | 0.0421 | **0.1352** | **+0.0931 (+221% relative gain)** |
| **Precision** | 0.0000 | 0.0438 | **0.0548** | **+0.0110 (+25% relative gain)** |
| **IoU** | 0.0000 | 0.0219 | **0.0406** | **+0.0187 (+85% relative gain)** |

---

## 5. GT Box 1 vs GT Box 2 Coverage

| Ground Truth Box | Pixel Area | Day 4 Overlap | Day 5 Overlap | Day 5 Recall | Status |
|---|---|---|---|---|---|
| **GT Box 1** `[439, 19, 463, 180]` | 3,864 px | 275 px (7.1%) | **882 px** | **22.83%** | **Substantial recovery (+220%)** |
| **GT Box 2** `[434, 353, 456, 474]` | 2,662 px | 0 px (0.0%) | **0 px** | **0.00%** | **Missed (Patch dilution limit)** |

> **Explicit Statement**: Crack IoU > 0.15 was **NOT ACHIEVED** (actual: 0.0406). GT Box 1 recall expanded to 22.8%, but GT Box 2 was missed due to DINO patch dilution. Documented in `DAY5_GPU_FOLLOWUP.md`.

---

## 6. Pothole Root-Cause Findings (0454.png)

Investigation revealed:
1. **Pre-SAM2 Over-Fragmentation**:
   - DINO anomaly map on the pothole produced 5 separate connected components across the depression (upper rim, central waist, lower depression).
   - In Day 4, each fragment was passed to SAM2 independently.
   - Independent SAM2 prompts produced localized masks whose centroids were 112 px apart, exceeding `road_mask_merge_centroid_px = 65.0`.
2. **Day 5 Solution**:
   - Pre-consolidating candidate components within 21 px of each other BEFORE SAM2 created a unified prompt box `[208, 38, 326, 255]`.
   - SAM2 received the full physical extent of the crater and segmented the unified pothole in a single pass.

---

## 7. Pothole Metrics Comparison (Day 3 → Day 4 → Day 5)

Ground truth mask area: 15,016 px.

| Metric | Day 3 | Day 4 Post-Fix | Day 5 Post-Fix | Delta (Day 4 → Day 5) | Engineering Target |
|---|---|---|---|---|---|
| **Fragments** | 4 | 2 | **1** | **-1 (Unified single defect)** | <= 2 |
| Total Predicted Area | 4,306 px | 5,874 px | **10,746 px** | +4,872 px | Model-derived |
| **GT Overlap Pixels** | 3,560 px | 4,407 px | **10,052 px** | **+5,645 px (+128% gain)** | Substantial |
| **Recall** | 0.2371 | 0.2935 | **0.6694** | **+0.3759 (+128% gain)** | High |
| **Precision** | 0.8268 | 0.7503 | **0.9354** | **+0.1851 (+25% gain)** | High |
| **IoU** | 0.2259 | 0.2674 | **0.6398** | **+0.3724 (+139% gain)** | **Target > 0.40 EXCEEDED (0.6398)** |

---

## 8. Healthy-Road Regression (original_healthy.jpg)

Running `original_healthy.jpg` through the Day 5 pipeline:
- **Accepted Defects**: **0** (Requirement: 0)
- **Current Severity**: **0.0**
- **Road Mask Ratio**: **0.7133**
- **Threshold**: **0.8025**
- **False Positives**: **0**

Specificity on healthy pavement is perfectly preserved.

---

## 9. Cross-Domain Regression (India_005086.jpg)

- **Defect Count**: **2** (consolidated from 5 fragmented detections in Day 4)
- **Current Severity**: **0.7450**
- **Road Mask Ratio**: **0.5462**
- Qualitative behavior is stable; rough/patched asphalt is localized without runaway false alarms.

---

## 10. Small Common-Benchmark Regression

Run on 4 representative samples from `benchmark/common_candidate/manifest.csv`:
- `China_Drone_000104`: defects=2, severity=0.2535, inference=615.1 ms
- `China_Drone_000122`: defects=3, severity=0.7866, inference=295.9 ms
- `China_Drone_000175`: defects=2, severity=0.6277, inference=247.9 ms
- `China_Drone_000213`: defects=5, severity=0.7067, inference=378.9 ms
- Optional check `China_Drone_000258` (difficult D20/shadow): defects=2, severity=0.6161, inference=767.0 ms

All 5 benchmark samples processed cleanly without runtime errors or crashes.

---

## 11. Contract & Interface Regressions

1. **Feature Contract Schema v1.0.0**:
   - Validated on all 4 Day 5 output records (`China_Drone_001267`, `0454`, `original_healthy`, `India_005086`).
   - All severity and area ratio fields are strictly within `[0, 1]`.
2. **Nitin Feature Adapter (`xgboost/feature_adapter.py`)**:
   - Evaluated with `0454/features.json` and `China_Drone_001267/features.json`.
   - Parsed `current_severity` (0.6539, 0.5913), `defect_count` (1.0, 3.0) with exact schema alignment.
   - Evaluated `xgboost/scenario_forecast.py` on fresh Day 5 features; generated valid Model V2 forecast (`future_severity = 0.6412` for `HEAVY_TRAFFIC` 30-day).
3. **Common Runner (`run_sam2_dino.py`)**:
   - Verified `--help`, single-file, directory, and CSV manifest modes.

---

## 12. Tests Status

Total tests executed:
- `road_health_pipeline/tests/`: 145 passed
- `sam2_dino/test_day4_fixes.py`: 6 passed
- `sam2_dino/test_day5_fixes.py`: 4 passed
- `sam2_dino/test_contract.py`: 2 passed
- **Total**: **157 passed**, **0 failed**, **0 skipped** in 20.10s.

---

## 13. Acceptance Classification

### **PARTIAL PASS**

**Justification**:
- **Pothole Target (> 0.40)**: **PASSED (0.6398 achieved)**. Single fragment, recall 66.9%, precision 93.5%.
- **Crack Target (> 0.15)**: **NOT ACHIEVED (0.0406 achieved)**. GT Box 1 recall improved from 7.1% to 22.8% (+220% overlap gain), but GT Box 2 was missed due to DINO patch dilution.
- **Healthy Specificity**: **PASSED (0 false positives preserved)**.
- **Common Runner & Nitin Adapter**: **PASSED (100% compatible)**.

---

## 14. Day-6 Recommendation

1. Execute the proposed high-resolution sliding-window DINOv2 experiment (`sam2_dino/DAY5_GPU_FOLLOWUP.md`) on RTX 5060 to resolve GT Box 2 crack resolution.
2. Maintain pre-consolidation in production; it resolved pothole fragmentation without degrading precision.
3. Keep `active_anomaly_percentile = 92.0`; do not artificially lower it.
