# VRINDA1-5 · MARION1-4 · NITIN1-4 Integration Checkpoint

**Integration Date:** 2026-09-08  
**Team Lead:** Integration & regression check across all three streams  
**Starting Commit:** `976bf5d` — Team Lead: complete Marion Day 4 perception stabilization  
**Branch:** `main`  
**GPU:** NVIDIA GeForce RTX 5060 Laptop GPU  

---

## 1. Latest Relevant Commits

| Commit | Stream | Description |
| :--- | :--- | :--- |
| `976bf5d` | Marion Day 4 | Complete Marion Day 4 perception stabilization |
| `0c50305` | Vrinda Day 5 | Complete Vrinda Day 5 benchmark preparation |
| `9234027` | Vrinda Day 4 | Complete Vrinda Day 4 evaluation |
| `ccc1213` | Nitin Day 4 | Nitin Day 4: temporal analytics infrastructure |
| `fb373f6` | Vrinda Day 3 | Vrinda Day 3 taken over |
| `6787e2d` | Vrinda Day 2 | Complete Vrinda Day 2 YOLO training |
| `a7bbf71` | Marion+Nitin | Sync Nitin Days 1–3 and Marion Days 1–3 |

All three streams (YOLO Days 1–5, Marion Days 1–4, Nitin Days 1–4) are present in `main`. ✅

---

## 2. Git / Conflict Status

| Check | Status |
| :--- | :--- |
| Branch | `main` ✅ |
| Sync with origin | Up to date ✅ |
| Merge conflict markers in `*.py *.json *.yaml *.yml` | **NONE** ✅ |
| CarlaUE5 | Modified submodule — NOT staged, NOT touched ✅ |
| `git reset --hard` / `git push --force` used | **NEVER** ✅ |

---

## 3. Shared Contract Status — PASS

**File:** `sam2_dino/feature_contract.schema.json` — `version: 1.0.0`

All required fields present: `image_id`, `segment_id`, `day`, `original_filename`, `condition`,
`current_severity`, `crack_area_ratio`, `defect_area_ratio`, `defect_count`,
`surface_anomaly_score`, `water_flag`, `defects`.

Defect geometry fields defined: `bbox`, `mask_path`, `centroid_x`, `centroid_y`, `area_ratio`, `defect_type`.

- Ratios/severity range `[0, 1]` enforced ✅
- Null for unavailable measurements — permitted by schema ✅
- Null ≠ 0: feature adapter preserves NaN for missing values ✅
- Coordinates in original image space ✅

---

## 4. YOLO Checkpoint Verification — PASS

| Check | Expected | Actual |
| :--- | :--- | :--- |
| SHA-256 | `ddbea38...b5dcf` | `ddbea38...b5dcf` ✅ |
| Classes | D00/D10/D20/D40/Repair | D00/D10/D20/D40/Repair ✅ |
| Ultralytics | — | 8.4.143 |

---

## 5. YOLO Regression — PASS

| Image | Expected | Result | Inference |
| :--- | :--- | :--- | :--- |
| `China_Drone_001267.jpg` | ≥2 D00 | **2 × D00** (conf 0.671, 0.621) | 793 ms |
| `original_healthy.jpg` | 0 detections | **0** | 773 ms |

YOLO output fields verified: `image_name`, `inference_ms`, `defect_count`, `class`, `confidence`, `bbox`.

---

## 6. Marion Asset Verification — PASS

| Asset | Status |
| :--- | :--- |
| DINOv2 checkpoint | Loaded ✅ |
| SAM2 checkpoint (`sam2.1_hiera_small.pt`) | Loaded ✅ |
| FAISS healthy-road memory bank (10,000 vectors) | Loaded ✅ |
| CUDA / RTX 5060 (PyTorch 2.13.0+cu130) | Available ✅ |
| `run_sam2_dino.py --help` | OK ✅ |

---

## 7. Marion Crack Regression (`China_Drone_001267.jpg`) — PARTIAL PASS

Reference Day-3: 0 px overlap, IoU=0.0000.

| Metric | Day 3 | Day 4 | Delta |
| :--- | :---: | :---: | :--- |
| GT Pixel Overlap | 0 px | **275 px** | +275 px |
| IoU | 0.0000 | **0.0219** | Non-zero achieved |
| Recall | 0.0000 | **0.0421** | +0.0421 |
| Precision | 0.0000 | **0.0438** | +0.0438 |
| Defect count | 0 | **4** | +4 |

> Marion crack localization is **PARTIALLY RECOVERED but still weak**. IoU=0.0219 is a breakthrough,
> not a solved problem. Localizer still misses most of the crack extent.

---

## 8. Marion Pothole Regression (`0454.png`) — PARTIAL PASS

| Metric | Day 3 | Day 4 | Delta |
| :--- | :---: | :---: | :--- |
| Fragment count | 4 | **2** | −50% |
| GT pixel overlap | 3,560 px | **4,407 px** | +23.8% |
| IoU | 0.2259 | **0.2674** | +18.4% |
| Recall | 0.2371 | **0.2935** | +23.8% |
| Precision | 0.8268 | **0.7503** | >0.75 preserved |

> Marion pothole segmentation is **IMPROVED but still limited**. IoU=0.2674 captures
> only ~27% of ground-truth area. Day-5 target.

---

## 9. Marion Healthy Regression (`original_healthy.jpg`) — PASS

| Check | Actual |
| :--- | :--- |
| Accepted defect count | **0** |
| Severity | **0.0** |
| Road mask ratio | **0.7133** (exact Day-3 value) |

Zero false positives confirmed. Day-3 regression-free.

---

## 10. Marion Road Mask Stabilization — PASS

| Image | Target | Verified |
| :--- | :---: | :---: |
| `original_healthy` | 0.7133 | 0.7133 ✅ |
| `China_Drone_001267` | 0.7826 | 0.7826 ✅ |
| `0454` | 0.5201 | 0.5201 ✅ |
| `India_005086` | 0.5462 | 0.5462 ✅ |

---

## 11. Marion Common-Manifest Readiness — PASS (after bug fix)

Runner: `run_sam2_dino.py --input benchmark/common_candidate/manifest.csv`

**Bug fixed during integration:** `sam2_dino/run_sam2_dino.py` was computing `ROOT` as `sam2_dino/`
and resolving manifest image paths relative to it. All 18 images reported as not found.
Fix: added `REPO_ROOT = ROOT.parent` used for manifest path resolution.
`test_batch_runner_manifest_collection` now PASS.

Output fields verified: `original_filename`, mask-derived bounding boxes (`defect_boxes_xyxy`),
`inference_time_ms`, `current_severity`, `defect_area_ratio`, `condition`, `viewpoint`, `run_summary.json`.

> SAM2 boxes are **bounding boxes derived from segmentation masks**, not native SAM2 box detections.

---

## 12. Marion → Nitin Feature Adapter — PASS

Input: `sam2_dino/outputs/China_Drone_001267/features.json`

- Schema validation (`validate_feature_record`): PASS
- `current_severity` = 0.6657 arrives correctly: PASS
- Null values remain null/NaN: PASS
- Richer Marion fields (`crack_area_ratio`, etc.) preserved but NOT injected into Model V2: PASS
- No schema incompatibility: PASS

---

## 13. XGBoost Model V2 Regression — PASS

**File:** `xgboost/model/scenario_model_v2.json`

- Model loads ✅
- Scenario presets resolve ✅
- Supported horizons: `[30, 60, 90]` days ✅
- Output bounded `[0, 1]` ✅
- Deterministic ✅
- Feature order preserved: `current_severity`, `rainfall_level`, `traffic_level`, `temperature`, `water_exposure`, `days_ahead` ✅
- Marion perception fields NOT added as model predictors ✅

---

## 14. Goal-1 End-to-End Flow — PASS

```
sam2_dino/outputs/China_Drone_001267/features.json
→ load_feature_record()   [validation: PASS]
→ feature_adapter.build_model_features()
→ scenario_forecast.predict_future_severity()
→ MODEL-BASED FORECAST
```

| Field | Value |
| :--- | :--- |
| input_filename | `sam2_dino/outputs/China_Drone_001267/features.json` |
| current_severity | 0.6657 |
| scenario | `HEAVY_RAIN` |
| days_ahead | 30 |
| **forecast_severity** | **0.6213** |
| severity_change | −0.0444 |
| bounded [0,1] | True |
| forecast_status | `MODEL-BASED FORECAST` |

> This is a MODEL-BASED FORECAST, not an observation. XGBoost does not prove causality.
> LTPP IRI severity has not been calibrated to RoadSentinel image severity.

---

## 15. Nitin Temporal Regression — PASS (TEST_FIXTURE_ONLY)

`xgboost/temporal/test_temporal.py` — **5 passed, 0 failed**

Real multi-day temporal sequence: **NOT VERIFIED — required sequence unavailable.**
Only TEST_FIXTURE_ONLY sequences exist locally. No fabrication performed.

---

## 16. Goal-2 Architecture Readiness — PASS (infrastructure only)

Components verified: `export_temporal.py`, `defect_matching.py`, `progression.py`, `load_sequence.py`.

Goal-1 and Goal-2 remain architecturally separate. No code path sends daily observations
through XGBoost and labels them observed change.

Goal-2 flow:
```
Daily Marion features.json → load_sequence → defect_matching → progression → OBSERVED TEMPORAL CHANGE
```

---

## 17. Common Benchmark Candidate Audit — PASS

`benchmark/common_candidate/manifest.csv`

| Metric | Count |
| :--- | :--- |
| Total rows | 18 |
| GT-compatible labelled RDD images | 14 |
| Diagnostic/incompatible | 4 (`China_Drone_001267`, `original_healthy`, `0454`, `India_005086`) |

- Condition provenance preserved (`VISUAL_LABEL_ONLY` / `UNKNOWN`) ✅
- Visual labels NOT presented as verified weather ✅
- Candidate NOT used for training ✅
- Incompatible annotations NOT silently converted ✅

---

## 18. Same-Image Dual-Pipeline Smoke Test — PASS

3 images from `benchmark/common_candidate/` on RTX 5060:

| Image | YOLO | Marion | YOLO ms | Marion ms | YOLO classes | Marion defects |
| :--- | :---: | :---: | ---: | ---: | :--- | :--- |
| `China_Drone_000104.jpg` | SUCCESS | SUCCESS | 779.6 | 611.5 | 2×Repair | 1 |
| `China_Drone_000122.jpg` | SUCCESS | SUCCESS | 757.9 | 761.5 | 2×Repair | 4 |
| `China_Drone_000175.jpg` | SUCCESS | SUCCESS | 761.9 | 756.4 | 3×D10 | 2 |

6/6 inference runs successful. Interface compatible.

> Class-wise comparison NOT valid: YOLO outputs D00/D10/D20/D40/Repair taxonomy;
> Marion outputs generic anomaly segmentation (road_defect). No shared taxonomy exists.

---

## 19. Comparison Interface Readiness — PASS

**Created:** `integration/comparison/run_comparison.py`

Output: `comparison_records.csv` + `comparison_summary.json`

No ensemble/fusion implemented. Class taxonomy limitation explicitly documented in all outputs.

---

## 20. Test Suite Results

| Suite | Passed | Failed | Status |
| :--- | ---: | ---: | :--- |
| Marion `road_health_pipeline/tests/` | 145 | 0 | PASS |
| Marion `sam2_dino/test_day4_fixes.py` | 5 | 0 | PASS |
| Nitin `test_scenario_forecast.py` + `test_adapter.py` | 6 | 0 | PASS |
| Nitin `test_temporal.py` | 5 | 0 | PASS |
| **TOTAL** | **161** | **0** | ✅ |

---

## 21. Repository Hygiene — PASS

| Check | Status |
| :--- | :--- |
| Datasets staged | NO ✅ |
| Generated image batches staged | NO ✅ |
| SAM2/DINO checkpoints staged | NO ✅ |
| Memory banks staged | NO ✅ |
| CarlaUE5 generated files staged | NO ✅ |
| Secrets staged | NO ✅ |
| `integration/comparison/.gitignore` added | YES (prevents generated outputs) ✅ |

---

## 22. Research Claim Boundaries

| Claim | Status |
| :--- | :--- |
| Marion crack localization is solved | NOT CLAIMED — IoU=0.0219, partial recovery only |
| Real weather robustness verified | NOT CLAIMED — VISUAL_LABEL_ONLY conditions |
| Final YOLO-vs-DINO comparison complete | NOT CLAIMED — taxonomy mismatch |
| Equal model training conditions | NOT CLAIMED — YOLO supervised, Marion zero-shot |
| XGBoost proves causality | NOT CLAIMED — scenario estimates only |
| LTPP severity calibrated to image severity | NOT CLAIMED — scale_transfer_caveat in every output |
| Full 10-day temporal experiment | NOT CLAIMED — TEST_FIXTURE_ONLY |
| Missing defects = repaired roads | NOT CLAIMED — forbidden |

---

## 23. Unresolved Blockers

| Blocker | Severity | Owner |
| :--- | :--- | :--- |
| Marion crack IoU weak (0.0219) | Medium | Marion Day 5 |
| Marion pothole coverage limited (IoU=0.2674) | Medium | Marion Day 5 |
| Real multi-day flight sequence unavailable | High | External data acquisition |
| YOLO/Marion taxonomy mismatch | Known limitation | No workaround without shared taxonomy |
| YOLO D20/D40 domain limitations unaddressed | Medium | Vrinda Day 6 |

---

## 24. Next Task Decisions

**A. YOLO ready for final common comparison?** PARTIAL PASS — checkpoint, standardized output, manifest consumption all verified. D20/D40 weak.

**B. Marion ready for final common comparison?** PARTIAL PASS — manifest ready, interface works. Crack IoU and pothole coverage need Day-5 improvement first.

**C. Nitin Goal-1 ready?** PASS — will receive Marion Day-5 outputs immediately.

**D. Nitin Goal-2 infrastructure ready?** PASS (infrastructure only) — awaiting real multi-day sequence.

**E. Real temporal experiment blocked by?** Real multi-day flight sequence over same road segment — external acquisition dependency.

**F. Real-weather claims blocked by?** Source-verified weather metadata — VISUAL_LABEL_ONLY is insufficient.

**G. GPU corrective work required?** NO — all inference successful on RTX 5060.

**H. Recommended next tasks:**
1. **Marion Day 5** — improve crack IoU and pothole coverage (weak link blocking comparison)
2. **Team Lead common benchmark run** — after Marion Day 5, run full 14-image comparison
3. **Vrinda Day 6** — D20/D40 robustness if benchmark reveals domain gaps
4. **Nitin Day 5** — only after real multi-day sequence available; do not fabricate

---

*Report generated by Team Lead. All data from live integration run 2026-09-08.*
