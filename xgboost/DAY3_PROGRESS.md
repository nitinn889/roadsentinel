# XGBoost Day 3 Progress

## Status

Implemented a user-triggered, single-image scenario forecast interface in
`scenario_forecast.py`. Every result is labelled **MODEL-BASED FORECAST** and
states that it is a scenario-conditioned estimate, not a guaranteed future
condition.

This replaces the Day 2 limitation where only `current_severity` and
`days_ahead` reached Model V1. Scenario Model V2 was trained with rainfall,
traffic, temperature, water exposure, and horizon as actual model features.
There are no event-name branches or fixed scenario severity increments in the
prediction path.

## Implemented scenarios

Preset values are derived from the Model V2 training split, not manually chosen
severity jumps. `NORMAL` uses the training median for every exposure. Each event
preset changes its named exposure to the training P90 while leaving the other
exposures at their medians.

| Scenario | Rainfall (mm/day) | Heavy trucks/day/lane | Temperature (°C) | Wet-day fraction |
|---|---:|---:|---:|---:|
| `NORMAL` | 2.737569 | 346 | 6.396536 | 0.566265 |
| `HEAVY_RAIN` | 3.769006 | 346 | 6.396536 | 0.566265 |
| `HEAVY_TRAFFIC` | 2.737569 | 784 | 6.396536 | 0.566265 |
| `HIGH_HEAT` | 2.737569 | 346 | 21.491477 | 0.566265 |
| `WET_EXPOSURE` | 2.737569 | 346 | 6.396536 | 0.687589 |

The CLI also accepts user overrides for all four exposure values. Values outside
the training domain are rejected instead of silently extrapolated.

## Input schema and interface

Perception input is one validated `sam2_dino/feature_contract.py` record. Model
V2 currently consumes its `current_severity`; the remaining SAM2/DINO fields are
preserved but explicitly listed as unused in each result.

Scenario input:

| Field | Type and unit | Accepted by Day 3 interface |
|---|---|---|
| `scenario_name` | enum | the five names above |
| `scenario_status` | enum | must be `USER_SUPPLIED` |
| `rainfall_level` | float, mm/day | training range `[0.719482, 5.143452]` |
| `traffic_level` | float, heavy trucks/day in the LTPP lane | training range `[58, 1257]` |
| `temperature` | float, °C | training range `[-4.447222, 24.684526]` |
| `water_exposure` | float, fraction of days | training range `[0.261468, 0.777778]` |
| `days_ahead` | integer days | `30`, `60`, or `90` |

Exact trained feature order:

```text
current_severity, rainfall_level, traffic_level, temperature,
water_exposure, days_ahead
```

Clean Python entry points:

```python
scenario = scenario_from_preset("HEAVY_TRAFFIC", 90, overrides={...})
result = predict_future_severity(feature_record, scenario)
```

CLI example:

```bash
./.venv/bin/python xgboost/scenario_forecast.py \
  sam2_dino/outputs/0454/features.json \
  --scenario HEAVY_TRAFFIC \
  --days-ahead 90
```

Optional numeric flags are `--rainfall-level`, `--traffic-level`,
`--temperature`, and `--water-exposure`.

## Training data and model

- Source: [FHWA LTPP InfoPave Standard Data Release 40](https://infopave.fhwa.dot.gov/Data/StandardDataRelease/), using state archives for Alaska, Hawaii, Louisiana, North Dakota, New York, and Rhode Island.
- Built 113 consecutive, same-construction IRI forecast pairs from 23 sites.
- Each pair joins future-interval daily precipitation, daily mean temperature,
  wet-day fraction, and heavy-truck AADTT for the same LTPP section. Climate
  coverage must be at least 80 percent of the interval.
- Observed intervals span 31–728 days. The model horizon feature uses the nearest
  30-day bucket (30–720 days), which gives a defined 30/60/90-day interface
  without altering the observed dates, exposure aggregates, or future IRI
  target.
- Target severity is future MRI/IRI mapped to `[0,1]` using the training sites'
  P01/P99 IRI values. The mapping is fitted on training sites only.
- Seeded site split: 17 train sites / 6 test sites, 89 / 24 rows, zero site
  overlap.
- Held-out engineering metrics: MAE `0.092411`, RMSE `0.114372`, R² `0.805500`.
  This small holdout is not research or deployment validation.
- Every scenario feature received nonzero fitted importance in Model V2.
- Reproducible data/model scripts: `build_scenario_training_table.py` and
  `train_scenario_model.py`.

Model path:

```text
xgboost/model/scenario_model_v2.json
```

Config and learned preset path:

```text
xgboost/config/scenario_model_v2.json
```

## Sample verified runs

Fixed perception input: image `0454`, `current_severity = 0.6757`. All values
below use a 90-day horizon and the exact same perception record.

| Scenario | Predicted future severity |
|---|---:|
| `NORMAL` | 0.744589388 |
| `HEAVY_RAIN` | 0.744589388 |
| `HEAVY_TRAFFIC` | 0.764537632 |
| `HIGH_HEAT` | 0.753558278 |
| `WET_EXPOSURE` | 0.761363328 |

Saved example output: `xgboost/outputs/day3_heavy_traffic_90_forecast.json`.

The unchanged `HEAVY_RAIN` result is a legitimate tree-model outcome: the P90
rain input reaches the model but remains in the same local tree leaves as the
normal rain input for this image. No post-model increment is added to force a
difference. Traffic, heat, and wet exposure demonstrate that the same image can
produce distinct scenario-conditioned forecasts.

The 30, 60, and 90-day values were each accepted, inserted into the feature row,
and returned deterministic bounded outputs. For this particular image and the
five preset exposure combinations, those three horizons currently share the
same learned short-horizon leaves; see limitations below.

Verification command:

```bash
./.venv/bin/python -m unittest discover -s xgboost -p 'test_*.py' -v
```

Result: 6 tests passed. Checks cover `[0,1]` output bounds, identical-input
determinism, all scenario values appearing in the trained row, scenario label
changes having no hidden numerical effect, distinct same-image forecasts, and
30/60/90-day acceptance.

## Limitations

- This is a small engineering baseline: 113 pairs from 23 LTPP sites and six
  states. Scenario associations are observational, not causal.
- Direct short-horizon observations are sparse. The user horizons are 30-day
  buckets, and the first observed interval is 31 days. Model V2 accepts 30, 60,
  and 90 days, but the current fitted trees do not resolve differences among
  those horizons for the verified image. More densely sampled repeated surveys
  are needed before claiming fine-grained short-term timing sensitivity.
- `HIGH_HEAT` means the P90 temperature in this training subset (21.49 °C), not
  an absolute extreme-heat standard. Preset names are relative to this sample.
- Rainfall and wet-day fraction are correlated in the observations. Independent
  user overrides can describe combinations sparsely represented by the joint
  training distribution even when each value is inside its marginal range.
- `traffic_level` is heavy-truck AADTT in the monitored LTPP lane, not total
  passenger-plus-commercial traffic.
- LTPP IRI-derived severity and RoadSentinel image-derived severity both use a
  `[0,1]` interface, but cross-domain calibration has not been established.
- Other SAM2/DINO measurements (`crack_area_ratio`, `defect_area_ratio`,
  `defect_count`, `surface_anomaly_score`, and `water_flag`) are not used because
  the LTPP training table does not contain matching image features.
- Predictions are clipped to `[0,1]` as an output safety check and must be
  presented as **MODEL-BASED FORECAST**, never as a guaranteed future condition.
