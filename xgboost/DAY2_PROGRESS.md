# XGBoost Day 2 Progress

## Status

- Installed and pinned `xgboost==3.4.1` in the existing `.venv`; import path and version verified.
- FHWA LTPP/InfoPave was inspected first, including `ANALYSIS_IRI`, but the live portal returned **Temporarily Down for Maintenance**. No LTPP data were claimed or downloaded.
- Used one official equivalent source: [NYC Open Data — Street Pavement Ratings (`6yyb-pb25`)](https://data.cityofnewyork.us/d/6yyb-pb25), provided by NYC DOT.

## Real data and target

- Live metadata verified before row download: `OFTCode` (site), borough/street endpoints (location), `InspectionTime`, and `SystemRating` (numeric). The dataset documents `SystemRating` as 1=low/poor through 10=high/good.
- Downloaded 2,010 rows / 226,004 CSV bytes from 200 sites with at least three distinct valid inspection dates. Metadata and selected-site files bring the downloaded total to 253,824 bytes. Selection favors frequently inspected sites.
- Ratings outside the documented `[1,10]` range were excluded, not coerced. Same-site/same-date duplicates were averaged (195 rows collapsed).
- Built 1,615 consecutive forecast pairs across 200 sites.
- Exact normalization: `severity = (10 - SystemRating) / 9`; 10 maps to 0 and 1 maps to 1. `future_severity` applies the same mapping to the next observation. This is a normalized common forecasting scale, **not** a claim that NYC ratings equal RoadSentinel image severity.

## Model V1

- Features actually used, in order: `current_severity`, `days_ahead`.
- Not available in this source / not used: `crack_area_ratio`, `defect_area_ratio`, `defect_count`, `surface_anomaly_score`, `water_flag`, `rainfall_level`, `traffic_level`, `temperature`, `water_exposure`.
- Split: seeded (`42`) site-based 80/20 holdout by `OFTCode`; 160 train sites / 40 test sites, 1,296 / 319 rows, zero site overlap.
- XGBoost holdout: MAE `0.0957999637`, RMSE `0.1277258619`, R² `0.4425254850`.
- Linear Regression holdout: MAE `0.1084265529`, RMSE `0.1410614510`, R² `0.3200389618`.
- Predictions were clipped to `[0,1]` for metric calculation. These are one held-out subset's engineering baseline metrics, not research validation.

## Saved artifacts

- Model: `xgboost/model/model_v1.json`
- Linear baseline: `xgboost/model/linear_baseline_v1.joblib`
- Config/feature order/seed: `xgboost/config/model_v1.json`
- Raw/processed data: `xgboost/data/raw/`, `xgboost/data/processed/training_pairs.csv`
- Metrics, normalization, split metadata: `xgboost/outputs/`
- Reproducible scripts: `fetch_nyc_pavement_sample.py`, `build_training_table.py`, `train_model_v1.py`, `predict_model_v1.py`

## Marion → Nitin interface

- Verified real perception record `sam2_dino/outputs/0454/features.json` through `feature_adapter.py` into Model V1.
- Output: `xgboost/outputs/marion_model_v1_forecast.json`, labelled `MODEL-BASED FORECAST`.
- The scenario file is explicitly `MOCK_INTERFACE_ONLY`; only its `days_ahead` value is consumed. The remaining scenario and perception fields are listed as unused in the output.
- Existing adapter unit test passes. No scenario UI was implemented.

## Unverified / Day 3

- Transfer calibration between image-derived severity and normalized NYC rating: **Not verified**.
- Causal effects for rainfall, traffic, heat, or water exposure: **Not verified** because this table does not contain those variables.
- Day 3: acquire and validate matched longitudinal exposure/loading data before enabling event-specific model effects; calibrate the shared severity scale; add scenario presets only after their model inputs have legitimate training support.
