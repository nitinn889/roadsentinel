# Day 3 — Road mask and current-frame geometry stabilization

## Files modified

- `road_health_pipeline/config.py`
- `road_health_pipeline/inference/sam2_mask.py`
- `road_health_pipeline/inference/pothole_localizer.py`
- `road_health_pipeline/inference/run_inference.py`
- `sam2_dino/current_condition.py`
- `sam2_dino/DAY3_PROGRESS.md`

## Road-mask changes

- Narrowed SAM2 road prompts to conservative camera-aware corridors instead of a full-frame nadir prompt.
- Persisted `raw_road_mask.png`, improved `road_mask.png`, and road-mask selection diagnostics for each run.
- Added largest-connected-region selection, prompt-corridor coverage checks, full-image-mask rejection, and green/high-saturation vegetation removal only when the raw mask is implausibly full-frame.
- An undersized mask is replaced by the camera prior; this is a fallback, not ground truth road segmentation.

## Crack localization and typing

- SAM2 refined masks are now intersected with the stabilized road mask in both 2D and 3D paths, preventing prompt expansion into off-road regions.
- Candidate typing is deliberately neutral (`road_defect`). Crack/pothole typing is **Not verified** and no longer asserted from the existing shape heuristics.
- Patch-score diagnostics, threshold diagnostics, component counts, raw/improved road masks, candidate masks, SAM2 prompts, and final geometry remain persisted by `current_condition.py`.
- Crack-box overlap on the labelled RDD sample: **Not verified**. The labelled sample is absent from this fresh clone.

## Pothole fragmentation

- Added conservative consolidation: masks merge only when they overlap sufficiently, or when dilated geometry touches and centroids are nearby. The consolidated record uses a union mask and recomputed bbox/area geometry.
- Pothole-600 fragments before/after, IoU, recall, and precision: **Not verified**. The Pothole-600 image, label, SAM2 checkpoint, FAISS memory bank, and OpenCV dependency are absent from this checkout/runtime.

## Road-mask before/after results

- Re-running the fixed Day 2 images: **Not verified**. Required images and model assets are not included in the clone.
- Therefore no replacement numerical road-mask ratios are reported.

## Weather/light test

- Daylight, low-light, and wet/rain tests: **Not verified**. No suitable small labelled or qualitative image set is present locally, and inference dependencies/assets are unavailable.
- No precision, recall, F1, or weather-robustness claim was calculated.

## Validation and compatibility

- Python compilation passed for the modified modules.
- `sam2_dino/test_contract.py`: 3 tests passed.
- `sam2_dino/test_thresholding.py`: **Not verified**; it cannot import because OpenCV (`cv2`) is unavailable in the provided Python runtime.
- The shared schema was not changed. Final defects still export `bbox`, `mask_path`, `centroid_x`, `centroid_y`, `area_ratio`, and `defect_type`; aggregate values remain on the existing `[0,1]` convention.
- XGBoost-adapter compatibility: schema contract passes; adapter runtime compatibility after a new inference export is **Not verified**.

## Remaining critical failures / Day 4

- Run the fixed labelled Day 2 set with the actual SAM2 checkpoint, FAISS memory bank, OpenCV, and dataset assets. Inspect raw versus improved mask overlays before tuning any limits.
- Measure patch scores at the two RDD crack boxes, verify patch-to-image mapping, then evaluate overlap after road-mask stabilization.
- Measure fragmentation and union geometry against the Pothole-600 mask before accepting consolidation thresholds.
- Only then perform a small daylight/low-light/wet qualitative test. Do not start cross-day tracking.
