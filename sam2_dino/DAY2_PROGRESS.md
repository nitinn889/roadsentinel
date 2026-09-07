# Day 2 — Stabilize current-condition extraction

## Files modified

- `road_health_pipeline/config.py`
- `road_health_pipeline/inference/anomaly_detector.py`
- `road_health_pipeline/inference/pothole_localizer.py`
- `road_health_pipeline/inference/run_inference.py`
- `sam2_dino/export_features.py`
- `sam2_dino/current_condition.py`
- `sam2_dino/fixed_test_set.json`
- `sam2_dino/test_contract.py`
- `sam2_dino/test_thresholding.py`
- `sam2_dino/DAY2_PROGRESS.md`

## Instability diagnosis and active threshold

Raw scores are FAISS k-NN cosine distances. Local normalization subtracts the per-image road-score median and maps positive elevation with `1 - exp(-elevation / max(0.05, 3*IQR))`. This makes global appearance shifts smaller, but it is self-relative: a nearly uniform healthy frame can still contribute a top tail, while widespread damage can raise the median and hide less-extreme damage. Road-mask errors also contaminate both the normalization baseline and candidates. On the annotated crack sample, SAM2 selected a 98.84% full-image road mask, including vegetation.

The active selector is now one explicit method: `AnomalyDetector.select_active_threshold()`. It uses the 92nd percentile of positive normalized patch scores, clipped to `[0.10,0.85]`; the image score is the 95th percentile of all normalized patch scores. Constants live in `Config`. The four tests applied no clipping. Values were centralized, not tuned.

| Test image | Raw median / IQR | Threshold | Road mask | Components → accepted |
|---|---:|---:|---:|---:|
| `original_healthy.jpg` | 0.2801 / 0.0511 | 0.7941 | 93.74% | 1 → 0 |
| `China_Drone_001267.jpg` | 0.2650 / 0.0442 | 0.3572 | 98.84% | 16 → 4 |
| `0454.png` | 0.3288 / 0.0812 | 0.3906 | 99.95% | 6 → 4 |
| `India_005086.jpg` | 0.4502 / 0.0829 | 0.5356 | 26.74% | 3 → 0 |

## Fixes and outputs

- Added bounded raw/normalized score, median/IQR, threshold, road-mask, connected-component, and SAM2-prompt diagnostics.
- Replaced unconditional per-component console output with debug logging.
- Persisted original, road mask, anomaly heatmap, candidate mask, each final mask, overlap-safe union mask, overlay, diagnostics, and schema-valid features under `sam2_dino/outputs/<image_id>/`.
- Derived each final mask's `[x1,y1,x2,y2]` box, pixel centroid, pixel area, and image area ratio in original-image coordinates.
- `defect_area_ratio` and crack area use mask unions, so overlapping masks are not double-counted. `mask_path` is relative to its `features.json`.
- Export severity remains `[0,1]`; existing `[0,100]` values are divided by 100 before export. A no-accepted-defect result exports 0.0; this is a pipeline result, not ground truth.

## Fixed test set and verification

- Healthy: repository synthetic `original_healthy.jpg` fixture.
- Crack: RDD2022 `China_Drone_001267.jpg`, with two D00 boxes.
- Pothole: Pothole-600 `0454.png`, with a pixel mask.
- Patched/rough: RDD2022 `India_005086.jpg`; patch/rough status is qualitative only, and its annotation is D00 rather than a patch mask.

All four runs completed. Eight accepted regions were all SAM2-refined and persisted. All four `features.json` records pass the unchanged Day 1 schema validator, preserve the original filename, and are readable by the existing XGBoost feature adapter. No XGBoost prediction or training was run.

Feature export root: `sam2_dino/outputs/`. Run summary: `sam2_dino/outputs/run_summary.json`.

## Remaining errors

- Healthy sample: no accepted false positive on this one fixture; broader false-positive rate is **Not verified**.
- Crack sample: prediction union had 0 overlap with both D00 boxes; four accepted masks were typed as potholes, including an off-road vegetation region. Crack recall/precision beyond this sample is **Not verified**.
- Pothole sample: union-mask IoU against the supplied mask was 0.2983, with 0.3066 mask recall and 0.9168 mask precision; the defect was fragmented into four outputs. General performance is **Not verified**.
- Patched/rough sample: no accepted region; its D00 crack was missed, while patch ground truth is unavailable. Patch false-negative status is **Not verified**.
- Water classification and robustness across weather/lighting are **Not verified**.

## Day 3 tasks

Calibrate road-mask candidate selection against labelled road masks; evaluate threshold policy on more labelled healthy/crack/pothole frames; investigate intra-image consolidation of fragmented overlapping masks; and test crack typing/SAM2 prompts against pixel or box ground truth. Do not infer cross-day identity until current-frame geometry is stable.
