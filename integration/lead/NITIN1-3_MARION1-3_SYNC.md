# RoadSentinel Team-Lead Sync Report
# Integration of Nitin Days 1–3 and Marion Days 1–3

Date: 2026-09-07  
Role: Team Lead  
Scope: Nitin (XGBoost Days 1–3) + Marion (DINOv2 + SAM2 Days 1–3)

---

## 1. Git State and Commit Hashes

- **Base common ancestor**: `27a4529` (*Simulation ok and did Nitin day 1 and 2, Marion day 1 and 2*)
- **Remote `origin/main` commits**:
  - `05641db` (*Enhance .gitignore with more exclusions*)
  - `27e7372` (*Add road health scoring and deterioration prediction*) — Note: contained unresolved conflict markers
  - `8e20df3` (*Stabilize SAM2 road and defect mask geometry* — Marion Day 3)
- **Local `main` commit**:
  - `a575913` (*trial* — Nitin Day 3)
- **Merge commit**:
  - `25335fe` (*Merge branch 'origin/main' into main*)

---

## 2. Nitin Days 1–3 Files Found

- **Day 1**:
  - `xgboost/DAY1_PROGRESS.md`
  - `xgboost/feature_adapter.py`
  - `xgboost/test_adapter.py`
  - `xgboost/MOCK_scenario.json`
  - `xgboost/smoke_adapter_output.json`
- **Day 2**:
  - `xgboost/DAY2_PROGRESS.md`
  - `xgboost/fetch_nyc_pavement_sample.py`
  - `xgboost/build_training_table.py`
  - `xgboost/train_model_v1.py`
  - `xgboost/predict_model_v1.py`
  - `xgboost/model/model_v1.json`
  - `xgboost/model/linear_baseline_v1.joblib`
  - `xgboost/config/model_v1.json`
  - `xgboost/outputs/evaluation_metrics.json`
  - `xgboost/outputs/split_metadata.json`
  - `xgboost/outputs/normalization_metadata.json`
- **Day 3**:
  - `xgboost/DAY3_PROGRESS.md`
  - `xgboost/build_scenario_training_table.py`
  - `xgboost/train_scenario_model.py`
  - `xgboost/scenario_forecast.py`
  - `xgboost/test_scenario_forecast.py`
  - `xgboost/model/scenario_model_v2.json`
  - `xgboost/config/scenario_model_v2.json`
  - `xgboost/scenario_contract.schema.json`
  - `xgboost/data/processed/scenario_training_pairs.csv`
  - `xgboost/data/processed/scenario_dataset_summary.json`
  - `xgboost/outputs/scenario_model_v2_metrics.json`
  - `xgboost/outputs/day3_heavy_traffic_90_forecast.json`

---

## 3. Marion Days 1–3 Files Found

- **Day 1**:
  - `sam2_dino/DAY1_PROGRESS.md`
  - `sam2_dino/feature_contract.py`
  - `sam2_dino/feature_contract.schema.json`
  - `sam2_dino/run_smoke.py`
  - `sam2_dino/smoke_feature_record.json`
- **Day 2**:
  - `sam2_dino/DAY2_PROGRESS.md`
  - `sam2_dino/export_features.py`
  - `sam2_dino/current_condition.py`
  - `sam2_dino/fixed_test_set.json`
  - `sam2_dino/test_contract.py`
  - `sam2_dino/test_thresholding.py`
  - `sam2_dino/outputs/`
- **Day 3**:
  - `sam2_dino/DAY3_PROGRESS.md`
  - `road_health_pipeline/config.py`
  - `road_health_pipeline/inference/sam2_mask.py`
  - `road_health_pipeline/inference/pothole_localizer.py`
  - `road_health_pipeline/inference/run_inference.py`

---

## 4. Shared Schema Status

- **Schema version**: `1.0.0` (frozen in `sam2_dino/feature_contract.schema.json`).
- **Required fields**:
  - `image_id`, `segment_id`, `day`, `original_filename`, `condition`, `current_severity`, `crack_area_ratio`, `defect_area_ratio`, `defect_count`, `surface_anomaly_score`, `water_flag`, `defects`.
- **Conventions verified**:
  - Severity and ratios strictly bounded in `[0, 1]`.
  - Missing measurements represented strictly as `null` (no fabricated defaults).
  - Defect array elements provide `bbox`, `mask_path`, `centroid_x`, `centroid_y`, `area_ratio`, `defect_type`.
- **Contract tests**: 5 tests passed in `sam2_dino/` (`test_contract.py` + `test_thresholding.py`).

---

## 5. Marion Pipeline Integration Status

- **Road-mask stabilization**: Verified. Implements largest connected component selection, corridor prior fallback, and vegetation suppression for implausibly full-frame masks.
- **Defect clipping to road**: Verified. In `pothole_localizer.py`, SAM2 refined masks intersect with the stabilized road mask (`refined_road = result.mask & road_mask`).
- **Candidate typing**: Verified neutral `road_defect`. Heuristic crack/pothole classification is bypassed.
- **Fragment consolidation**: Verified. Merges masks based on `road_mask_merge_iou` and centroid proximity with dilation in `pothole_localizer.py`.
- **Mask and diagnostic persistence**: Verified. Persists `raw_road_mask.png`, `road_mask.png`, `candidate_mask.png`, per-defect masks, and `diagnostics.json` with `road_mask_diagnostics`.

---

## 6. Marion Day-3 Verification Status

The lead machine possessed all necessary runtime dependencies (`opencv-python 5.0.0`, `torch 2.13.0+cu130`, `faiss 1.15.0`), checkpoints (`sam2.1_hiera_small.pt`, `dinov2_vits14_pretrain.pth`), memory bank (`output/real_memory_bank`), and 4 fixed test images with ground truth annotations.

Execution of `sam2_dino/current_condition.py` across the fixed test set yielded the following:

### A. Road Mask Stabilization
| Image | Category | Day 2 Mask Ratio | Day 3 Raw Ratio | Day 3 Improved Ratio | Selection Reason |
|---|---|---:|---:|---:|---|
| `original_healthy.jpg` | healthy_road | 0.9374 | 0.7211 | 0.7133 | accepted_raw |
| `China_Drone_001267.jpg` | crack | 0.9884 | 0.2039 | 0.7826 | undersized_replaced_with_prior |
| `0454.png` | pothole | 0.9995 | 0.5296 | 0.5201 | accepted_raw |
| `India_005086.jpg` | patched_or_rough | 0.2674 | 0.1842 | 0.5462 | undersized_replaced_with_prior |

Near-full-frame leaks (>0.98 in Day 2) have been eliminated.

### B. Crack Localization (`China_Drone_001267.jpg`)
- Ground truth: 2 D00 boxes `[439, 19, 463, 180]` and `[434, 353, 456, 474]`.
- Output: 4 accepted regions (all typed `road_defect`).
- Overlap with D00 boxes: **0 pixels**.
- IoU: **0.0000**, Recall: **0.0000**, Precision: **0.0000**.
- Status: **Crack recall remains 0 on this benchmark sample**.

### C. Pothole Fragmentation (`0454.png`)
- Ground truth: Pixel mask `RoadSentinel_datasets/pothole_600/Pothole/label/0454.png` (15,016 pixels).
- Output: 4 accepted regions (all typed `road_defect`). Consolidation thresholds were not met due to distance between fragments.
- Overlap with ground truth: 3,560 pixels.
- Predicted union: 4,306 pixels.
- **IoU**: **0.2259** (Day 2 was 0.2983 when mask leaked across full image).
- **Recall**: **0.2371**.
- **Precision**: **0.8268**.

---

## 7. Marion → Nitin Adapter Status

- **Status**: **VERIFIED**.
- Tested real record `sam2_dino/outputs/0454/features.json` through `xgboost/feature_adapter.py`.
- Identifiers preserved: `image_id = "0454"`, `segment_id = "SEG_DAY2_TEST"`, `day = 1`, `original_filename = "0454.png"`.
- `current_severity` (0.6922) mapped to Model V2 input.
- **Perception features consumed by Nitin Model V2**:
  1. `current_severity`
- **Perception features preserved in metadata but unused by Model V2**:
  - `crack_area_ratio`, `defect_area_ratio`, `defect_count`, `surface_anomaly_score`, `water_flag`.
- Missing fields remain `null` / `NaN`; no fabricated defaults.

---

## 8. Model V1 Status

- **Status**: **VERIFIED** (no retraining performed).
- Artifacts present: `xgboost/model/model_v1.json`, `xgboost/config/model_v1.json`.
- Concept: `current_severity`, `days_ahead` → `future_severity`.
- Verified engineering holdout metrics from `xgboost/outputs/evaluation_metrics.json`:
  - MAE: `0.0957999637`
  - RMSE: `0.1277258619`
  - R²: `0.4425254850`

---

## 9. Model V2 Status

- **Status**: **VERIFIED**.
- Artifacts present: `xgboost/model/scenario_model_v2.json`, `xgboost/config/scenario_model_v2.json`, `xgboost/scenario_forecast.py`.
- Trained feature order:
  `current_severity`, `rainfall_level`, `traffic_level`, `temperature`, `water_exposure`, `days_ahead`.
- Scenarios supported:
  `NORMAL`, `HEAVY_RAIN`, `HEAVY_TRAFFIC`, `HIGH_HEAT`, `WET_EXPOSURE`.
- Verification of logic:
  - Scenario values enter the feature vector as model inputs.
  - No post-model increment or rule-based string logic exists (e.g., no `HEAVY_RAIN -> severity + 0.1`).
  - Labelled **MODEL-BASED FORECAST**.

---

## 10. Scenario Smoke-Test Results

Evaluated on real Marion record `0454.png` (`current_severity = 0.6922`), horizon = 90 days:

| Scenario | Input Weather / Traffic Parameters | Predicted Future Severity | Severity Change |
|---|---|---:|---:|
| `NORMAL` | Rain: 2.74 mm/d, Truck AADTT: 346, Temp: 6.40 °C, Wet: 0.566 | 0.744589388 | +0.052389388 |
| `HEAVY_RAIN` | Rain: 3.77 mm/d, Truck AADTT: 346, Temp: 6.40 °C, Wet: 0.566 | 0.744589388 | +0.052389388 |
| `HEAVY_TRAFFIC` | Rain: 2.74 mm/d, Truck AADTT: 784, Temp: 6.40 °C, Wet: 0.566 | 0.764537632 | +0.072337632 |
| `HIGH_HEAT` | Rain: 2.74 mm/d, Truck AADTT: 346, Temp: 21.49 °C, Wet: 0.566 | 0.753558278 | +0.061358278 |
| `WET_EXPOSURE` | Rain: 2.74 mm/d, Truck AADTT: 346, Temp: 6.40 °C, Wet: 0.688 | 0.761363328 | +0.069163328 |

- Identical repeated inputs produced bitwise deterministic outputs.
- Predictions remained safely within `[0, 1]`.
- `NORMAL` and `HEAVY_RAIN` sharing the same leaf output confirms absence of artificial scenario increments.

---

## 11. Unit-Test Results

- **XGBoost tests (`xgboost/test_*.py`)**:
  - `test_reads_shared_schema_without_predicting`: PASS
  - `test_all_scenario_values_are_model_features`: PASS
  - `test_event_name_has_no_hidden_prediction_effect`: PASS
  - `test_identical_inputs_are_deterministic_and_bounded`: PASS
  - `test_same_image_can_produce_scenario_specific_forecasts`: PASS
  - `test_supported_horizons`: PASS
  - **Summary**: 6 passed, 0 failed.

- **SAM2/DINO tests (`sam2_dino/test_*.py`)**:
  - `test_export_from_inference_shape`: PASS
  - `test_final_masks_use_union_and_export_geometry`: PASS
  - `test_temporal_path`: PASS
  - `test_empty_scores_do_not_detect_everything`: PASS
  - `test_policy_uses_configured_percentile_and_clip`: PASS
  - **Summary**: 5 passed, 0 failed.

---

## 12. Current Verified End-to-End Flow

```text
SINGLE IMAGE
    ↓  [VERIFIED]
DINOv2
    ↓  [PARTIALLY VERIFIED] (Crack recall 0.00 on benchmark; lighting/weather unverified)
anomaly candidates
    ↓  [PARTIALLY VERIFIED] (Consolidation incomplete on 0454; neutral road_defect typing)
SAM2 masks
    ↓  [VERIFIED]
current-condition feature record
    ↓  [VERIFIED]
XGBoost scenario model
    ↓  [VERIFIED]
MODEL-BASED future severity forecast
```

---

## 13. Unverified Claims (Research Claim Boundaries)

1. **Causality**: Model V2 associates observational scenarios; rainfall, traffic, and temperature are NOT proven causal deterioration drivers.
2. **Cross-Domain Calibration**: Image-derived severity `[0, 1]` and LTPP IRI-derived severity `[0, 1]` share an interface, but cross-domain physical calibration is **Not verified**.
3. **Short-Term Horizon Sensitivity**: 30, 60, and 90-day bucket forecasts share identical tree leaf assignments on this sample; fine-grained horizon differentiation is **Not verified**.
4. **Adverse Weather Robustness**: Performance under low-light, rain, wet roads, and fog is **Not verified**.
5. **Crack Coverage**: Crack recall on labelled RDD benchmarks is **Not verified** (0 overlap observed).

---

## 14. Integration Bugs Fixed

- **Corrupted Merge Conflict Markers**: Commit `27e7372` had pushed raw conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>> 0e75352`) into:
  - `road_health_pipeline/inference/run_inference.py`
  - `road_health_pipeline/common/schemas.py`
  - `road_health_pipeline/inference/defect_classifier.py`
  - `road_health_pipeline/tests/run_tests.py`
- **Resolution**: Restored clean codebases from `27a4529` and cleanly applied Marion's Day 3 diagnostics logging additions to `run_inference.py`. All syntax errors eliminated; pipeline loading and test execution restored.

---

## 15. Exact Blockers Remaining Before Day 4

1. **Marion Day 4**:
   - DINOv2 patch anomaly scores and coordinate registration must be inspected on crack benchmarks (`China_Drone_001267`) to determine why cracks are missed prior to SAM2 prompting.
   - Pothole fragment consolidation thresholds require adjustment so multiple masks from a single cavity are unified.
2. **Nitin Day 4**:
   - Align additional perception features (`crack_area_ratio`, `defect_area_ratio`) with deterioration models or formulate multi-target regression.
3. **Cross-Domain Calibration**:
   - Empirical alignment between SAM2/DINO image severity and pavement deterioration targets remains the principal cross-subsystem gap.

---

## 16. Files Modified During This Sync

1. `road_health_pipeline/common/schemas.py` (removed conflict markers from 27e7372)
2. `road_health_pipeline/inference/defect_classifier.py` (removed conflict markers from 27e7372)
3. `road_health_pipeline/tests/run_tests.py` (removed conflict markers from 27e7372)
4. `road_health_pipeline/inference/run_inference.py` (resolved conflict markers and integrated Marion Day 3 diagnostics)
5. `sam2_dino/outputs/*` (regenerated Day 3 verification artifacts on fixed test set)
6. `integration/lead/NITIN1-3_MARION1-3_SYNC.md` (team-lead sync report)
