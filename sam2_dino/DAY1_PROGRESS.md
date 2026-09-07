# Day 1 — DINOv2 + SAM2

## Files inspected

- `road_health_pipeline/inference/dinov2_embed.py`
- `road_health_pipeline/inference/anomaly_detector.py`
- `road_health_pipeline/inference/sam2_mask.py`
- `road_health_pipeline/inference/pothole_localizer.py`
- `road_health_pipeline/inference/run_inference.py`
- `road_health_pipeline/inference/severity_estimator.py`
- `road_health_pipeline/analytics/severity.py`
- `road_health_pipeline/common/schemas.py`
- `road_health_pipeline/config.py`
- Relevant DINOv2, SAM2, integration, and prediction tests

## Current pipeline status

- DINOv2: extracts ViT-S/14 patch embeddings from a road mask.
- Anomaly scoring: cosine-distance k-NN against a healthy-road FAISS memory bank, followed by local median/IQR normalization.
- Thresholding: the active `infer()` path uses the 92nd percentile of positive normalized patch scores, clipped to `[0.10, 0.85]`; comments/docstrings elsewhere still describe config-driven percentiles.
- SAM2: generates a road mask and refines anomaly candidate boxes into masks. Connected components generate initial boxes.
- Output masks: generated in memory but not persisted by `infer()`; therefore `mask_path` and true mask centroids are unavailable.
- Severity: the active 2D path uses confidence, optional area/depth, water, and surrounding damage on `[0,1]`. Another existing processing path uses `[0,100]`; the exporter normalizes either input scale to `[0,1]`.
- Lighting/weather: low-light enhancement and some shadow/artifact filtering exist. Robustness under rain, wet roads, fog/haze, and broad lighting variation is **Not verified**.
- Temporal defect identity matching across days is **Not verified**.

## Shared schema

`feature_contract.schema.json` freezes version `1.0.0`. Required fields are:

`image_id`, `segment_id`, `day`, `original_filename`, `condition`, `current_severity`, `crack_area_ratio`, `defect_area_ratio`, `defect_count`, `surface_anomaly_score`, `water_flag`, and `defects`.

Severity and ratios use `[0,1]`. Unavailable measurements are JSON `null`; they are never fabricated. `current_severity` is the maximum available detected-defect severity. Area ratios are the sum of serialized mask pixel areas divided by image pixels, capped at 1; mask union is not available after serialization. Individual defects retain only available `bbox`, `mask_path`, centroid, `area_ratio`, and `defect_type` fields.

Temporal inputs follow `SEG_001/day_01/.../day_10/original_filename`; every exported record preserves the parsed segment, day, and filename.

## Confirmed working

- Contract validation and temporal-path tests pass.
- Real one-image CUDA smoke run completed with the installed DINOv2 model, SAM2 checkpoint, and existing `road_health_pipeline/output/real_memory_bank` FAISS index.
- `smoke_feature_record.json` is the resulting machine-readable record. Its values are smoke outputs only, not accuracy metrics or research results. Capture condition was not known and remains `null`.

## Unverified

- Accuracy, calibration, generalization, and causal deterioration claims: **Not verified**.
- Mask-file export, mask-IoU tracking, centroids, new-versus-existing defect association, and ten-day execution: **Not verified**.
- Crack coverage quality and water-detection quality: **Not verified**.

## Interface and Day 2

`export_features.py` converts the existing inference JSON into the shared record; `xgboost/feature_adapter.py` imports the same validator and consumes it.

Day 2: persist per-defect masks, compute mask centroids/union area, add stable cross-day defect IDs and matching, run `SEG_001/day_01..day_10` validation, and test labelled daylight/low-light/rain/wet/shadow/fog sets without treating lighting as a deterioration cause.
