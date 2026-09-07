# Day 1 — XGBoost forecasting interface

## Files inspected

- `road_health_pipeline/prediction/progression_model.py`
- `road_health_pipeline/analytics/prediction.py`
- `road_health_pipeline/inference/run_inference.py`
- `road_health_pipeline/tests/test_prediction.py`
- Repository-wide targeted search for XGBoost and requested scenario fields

## Current status

- No XGBoost implementation, trained XGBoost model, training dataset, or `future_severity` evaluation was found. **Not verified**.
- Existing forecasting code is heuristic/rate-based; one optional path trains a Random Forest classifier. It is not XGBoost and does not implement the requested scenario contract.
- The active Python environment does not contain the `xgboost` package. No download was performed.

## Shared input and scenario contract

The adapter reads `sam2_dino/feature_contract.schema.json` version `1.0.0` and preserves `image_id`, `segment_id`, `day`, and `original_filename`.

Fixed numeric model feature order:

1. Current road: `current_severity`, `crack_area_ratio`, `defect_area_ratio`, `defect_count`, `surface_anomaly_score`, `water_flag`
2. Scenario: `rainfall_level` (mm/day), `traffic_level` (vehicles/day), `temperature` (°C), `water_exposure` (fraction of forecast days in `[0,1]`), `days_ahead` (positive integer)

Target: normalized `future_severity` in `[0,1]`.

`condition` is retained as perception metadata but is excluded from the numeric model row. This prevents capture lighting from being treated as a deterioration driver. Missing current measurements become NaN for future XGBoost handling, not invented values.

## Confirmed working

- `feature_adapter.py` validates and reads the exact real-smoke perception record.
- `MOCK_scenario.json` is clearly labelled `MOCK_INTERFACE_ONLY`.
- `smoke_adapter_output.json` preserves identifiers, emits the fixed feature order, and returns `future_severity: null` with `NOT_RUN_NO_TRAINED_XGBOOST_MODEL`.
- Adapter unit test passes.

## Unverified

- XGBoost fitting, prediction, calibration, feature importance, and forecasting accuracy: **Not verified**.
- Availability and validity of longitudinal targets and rainfall/traffic/temperature/water exposure data: **Not verified**.
- Ten-day temporal forecasting performance: **Not verified**.

## Interface and Day 2

Perception writes one validated JSON record per image; the adapter combines it with one user-supplied scenario JSON and prepares a numeric row. It does not predict on Day 1.

Day 2: identify a legitimate longitudinal training source, define leakage-safe segment/time splits, install/pin XGBoost only if approved, train a reproducible baseline, persist model metadata and feature order, and evaluate only against real or explicitly synthetic labelled targets.
