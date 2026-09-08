# RoadSentinel — Nitin Day 4: Temporal Analytics / Goal 2

Date: 2026-09-08
Status: **COMPLETE — Goal-2 infrastructure implemented; no full Marion temporal experiment claimed.**

## Scope and architecture

- Added `xgboost/temporal/` as a separate **OBSERVED TEMPORAL CHANGE** layer. It does not import, retrain, or alter `scenario_model_v2.json` / Model V2.
- Goal 1 remains: one perception record + scenario inputs → **MODEL-BASED FORECAST** through Model V2.
- Goal 2 is now: contract-compliant daily records for one segment → temporal matching and measured deltas → **OBSERVED TEMPORAL CHANGE**. Daily values are never forecast through XGBoost and presented as observations.

## Input and validation

Accepted layout is `SEG_xxx/day_01/features.json` through arbitrary valid days; records are validated against shared schema `1.0.0`.

- Enforces one `segment_id`, one record per day, and matching `day_NN` directory / record day.
- Sorts unordered day directories, reports gaps as `missing_days`, and never fills them.
- Fails safely on malformed records, wrong/mixed segment IDs, directory/day conflicts, duplicate-equivalent days, and invalid contract values.
- Preserves null values. Deltas are null when either measurement is null; no missing value is converted to zero.

## Progression and tracking

- Daily table retains day, filename, condition, severity, crack/defect area ratios, defect count, anomaly score, water flag, consecutive deltas, and Day-1-relative deltas.
- Matching is adjacent-day, same-segment, greedy one-to-one and conservative: compatible defect type plus mask IoU when readable/compatible (minimum 0.50), otherwise bbox IoU (minimum 0.30), otherwise centroid distance (maximum 75 px) with an area-consistency gate (maximum 3× change). Relative mask paths are resolved relative to their feature record.
- Tracks receive `DEFECT_001`-style IDs only after a new observed defect is accepted. Histories retain boxes, centroids, areas, type, and matching method/confidence.
- Events include `NEW_DEFECT`, `MATCHED_EXISTING`, `OBSERVED_AREA_INCREASED`, `OBSERVED_AREA_DECREASED`, and `NOT_OBSERVED`. `NOT_OBSERVED` is not treated as repaired, and area changes are not labelled physical growth/shrinkage.
- Segment-level water tracking reports only the first day with `water_flag=true`; it does not infer rain or causality.

## Outputs and test fixture

`export_temporal.py` writes `xgboost/temporal_outputs/<segment_id>/` with `daily_summary.csv`, `daily_summary.json`, `defect_tracks.json`, `temporal_events.json`, and `progression_summary.json`. Generated outputs are ignored by Git.

No usable multi-day Marion shared-perception sequence exists locally. `sam2_dino/smoke_feature_record.json` is only one Day-1 record. Prior `env/output/temporal_*` artifacts use a different simulation/capture schema; the 20-day temporal results report 3 valid frames of 160, and their renderer/capture history is not converted into this contract.

Therefore a three-day `TEST_FIXTURE_ONLY` sequence was added strictly to validate software behavior. Its exported values are not RoadSentinel experimental findings.

## Verification

- `./.venv/bin/python xgboost/temporal/test_temporal.py`: **5 passed**. Covers unordered and missing days, missing severity, empty/persistent/new/disappearing/multiple defects, ambiguous one-to-one matching, null geometry, malformed/duplicate-equivalent days, and wrong segment IDs.
- Fixture export CLI completed and wrote all five requested machine-readable artifacts.
- Goal-1 regression: `xgboost/test_adapter.py` **1 passed**; `xgboost/test_scenario_forecast.py` **5 passed**. Model V2 loads, preset forecasts remain bounded, and the Marion adapter is unaffected.

## Not verified / Day 5 handoff

- Full Day-1→Day-10 Marion perception experiment: **Not verified / not run**.
- Natural defect correspondences, compatible mask availability, calibrated area changes, physical deterioration, and water/rain causality: **Not verified** until Marion supplies same-segment daily contract records.
- No experimental observed-change score was introduced.

For Day 5, Marion should export one valid `features.json` per available segment/day with stable segment IDs, original filenames, boxes/centroids and compatible mask paths where available. Nitin can then run:

```bash
./.venv/bin/python xgboost/temporal/export_temporal.py SEG_001
```

and review `missing_days`, matching confidence, and `NOT_OBSERVED` events before making any real progression claim.
