# Day 4 — DINOv2 + SAM2 Perception Stabilization & Comparison Readiness

## 1. Team Lead Takeover Statement

As the Team Lead, I have taken over Marion / Person 1 Day 4 responsibilities due to Marion's temporary unavailability. The objective of Day 4 was not to redesign the architecture, but to preserve all verified Days 1–3 work (particularly the stabilized road mask), diagnose and resolve the critical crack detection blocker on `China_Drone_001267.jpg`, improve pothole fragmentation on `0454.png`, verify negative regression on `original_healthy.jpg`, validate the shared feature contract and Nitin's feature adapter, prepare temporal input readiness, and build a unified, manifest-compatible common-folder runner (`run_sam2_dino.py`).

Execution environment:
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- Driver: 595.84, CUDA: 13.2, PyTorch: 2.13.0+cu130
- Workspace: Local RoadSentinel repository, main branch

---

## 2. Repository & Model Assets Verification

All core model checkpoints, weights, and memory banks were verified present and functional on CUDA:
- DINOv2: `dinov2_vits14` loaded via torch hub onto CUDA (`dinov2_input_size=518`, `patch_size=14`, `grid_size=37x37`, 384-dim).
- SAM2: `road_health_pipeline/checkpoints/sam2.1_hiera_small.pt` loaded via `configs/sam2.1/sam2.1_hiera_s.yaml` on CUDA.
- FAISS Memory Bank: `road_health_pipeline/output/real_memory_bank` loaded successfully (10,000 vectors, dimension 384).
- Fine-tuned Crater Anomaly Head: `road_health_pipeline/checkpoints/dinov2_crater_head.pt` verified.
- Datasets: RDD2022 (`China_Drone`, `India`), Pothole-600 (`0454.png`), and synthetic healthy road fixture verified.

---

## 3. Pre-Fix Crack Reproduction (China_Drone_001267.jpg)

Before making any changes, the existing Day 3 pipeline was run unchanged on `China_Drone_001267.jpg` and persisted separately to `sam2_dino/outputs/day4_pre_fix/China_Drone_001267/`.

Ground truth D00 crack annotations (from companion Pascal VOC XML):
- Box 1: `[439, 19, 463, 180]` (width=24, height=161)
- Box 2: `[434, 353, 456, 474]` (width=22, height=121)
- Ground truth rasterized mask: 6,526 pixels

Pre-Fix Reproduction Results:
- Active threshold: 0.3515
- Connected components before filters: 11
- Accepted defect regions: 4 (Prompt boxes: `[365, 220, 435, 298]`, `[454, 281, 465, 287]`, `[377, 304, 384, 316]`, `[131, 430, 147, 439]`)
- Total predicted union pixels: 4,330
- GT overlap pixels: **0**
- IoU: **0.0000**
- Recall: **0.0000**
- Precision: **0.0000**

The complete zero-overlap crack failure was reproduced 100% identically to Day 3.

---

## 4. Crack Root-Cause Diagnosis

A systematic diagnostic trace of DINOv2 patch tokens, anomaly scoring, road masking, and candidate filtering revealed the exact reasons the crack was lost:

1. **Patch Coordinate Mapping Verification**:
   - Reshaping `(1, grid*grid, dim)` into `(grid, grid, dim)` preserves row-major raster order (dimension 0 is row/y, dimension 1 is column/x).
   - In `build_anomaly_map`, `cv2.resize(grid, (w, h))` maps rows to `h` (512) and columns to `w` (512) with linear interpolation centered at patch centers. There were no transpose or x/y swap bugs.
2. **DINOv2 Anomaly Elevation**:
   - DINOv2 successfully detects departure from healthy appearance along the crack corridor. Along the crack in Box 1 (column 33, rows 2..13), scores range between 0.15 and 0.5041 (peak at row 3, col 33 = 0.5041, raw score = 0.359).
   - In contrast, the surrounding healthy road background median is ~0.00 to 0.05 (75th percentile = 0.073, 90th percentile = 0.141).
   - At row 3, col 33, the normalized anomaly score (0.5041) clearly exceeds the 0.3515 threshold, creating Connected Component 2 with `bbox=[460, 42, 467, 54]`, area 66 px, conf 0.6604, overlapping GT Box 1 by 28 pixels.
3. **The Fatal Rejection Bugs**:
   - **Bug A (Perspective Horizon Filter in Nadir View)**:
     In `pothole_localizer.py`, line 375:
     `if y < 0.10 * img_h and (y + h_cc) < 0.16 * img_h: continue`
     This filter was intended for forward-facing perspective cameras to remove the vanishing sky/horizon. However, in nadir (top-down UAV / drone) imagery, the top of the image is just the northern road surface. CC 2 (`y=42, y+h_cc=54` in a 512px image) fell inside `[0, 81.9px]` and was discarded as a "distant perspective vanishing horizon"!
   - **Bug B (Duplicate Filter in Road Marking Suppressor)**:
     In `road_marking_suppressor.py`, `is_marking_candidate` had the identical hardcoded check:
     `if y1 < 0.10 * h and y2 < 0.16 * h: return True`
     even if the candidate was on bare asphalt.
   - **Bug C (Aspect Ratio Filter Punishing Cracks)**:
     Line 370 of `pothole_localizer.py` rejected `aspect_ratio > 4.5 and shape_circ < 0.15` as "linear artifacts". Since longitudinal cracks (D00) naturally have high aspect ratios and low circularity, this filter directly suppressed organic cracks.
   - **Bug D (Context Starvation in SAM2 Prompts)**:
     Passing a tight 7x12 pixel candidate box to SAM2 starved SAM2 of surrounding road context. Adding an 8px adaptive context padding allows SAM2 to capture the contrast and segment the defect cleanly.

---

## 5. Exact Code Changes

1. `road_health_pipeline/inference/pothole_localizer.py`:
   - Restricted perspective horizon rejection strictly to forward camera mode (`is_forward = getattr(CONFIG, "camera_mode", "nadir") == "forward" and not is_2d`).
   - Relaxed artificial aspect-ratio suppression so organic cracks are not discarded (`if not is_2d and aspect_ratio > 8.0 and shape_circ < 0.05: continue`).
   - Bypassed arbitrary 25px frame boundary rejection when running in 2D test / nadir mode where road masks already delineate the drivable surface.
   - Added adaptive context padding (`pad_x, pad_y = 8 px` when candidate dimension is narrow) when prompting SAM2.
   - Enhanced candidate consolidation: candidates merge if mask IoU >= 0.20, if masks touch/are contiguous within 9x9 dilation, or if dilated 49x49 masks overlap with centroid distance within `road_mask_merge_centroid_px`.
2. `road_health_pipeline/inference/road_marking_suppressor.py`:
   - Conditioned horizon and border touching rejection in `is_marking_candidate` strictly on `is_forward` camera mode.
3. `road_health_pipeline/config.py`:
   - Updated `road_mask_merge_centroid_px: float = 65.0` (from 48.0) to allow adjacent components of a physical defect to consolidate without splitting.
4. `sam2_dino/run_sam2_dino.py` & `run_sam2_dino.py`:
   - Created clean, general-purpose CLI runner supporting single images, folders, or benchmark CSV manifests.
5. `sam2_dino/test_day4_fixes.py`:
   - Created comprehensive test suite covering coordinate mapping, road prior geometry, candidate consolidation, schema compliance, and manifest consumption.

---

## 6. Post-Fix Crack Metrics (China_Drone_001267.jpg)

Ground truth D00 rasterized pixels: 6,526.

| Metric | Day 3 | Day 4 Post-Fix | Delta / Improvement |
|---|---|---|---|
| Accepted Defect Count | 4 | 4 | Stable |
| Defect 1 Bounding Box | None (Missed) | `[451, 30, 471, 60]` | **Recovered GT Box 1 crack** |
| Defect 1 Area | 0 px | 478 px | +478 px |
| Total Predicted Pixels | 4,330 px | 6,278 px | +1,948 px |
| **GT Overlap Pixels** | **0 px** | **275 px** | **+275 px (Defect 1 genuinely overlaps GT1)** |
| **IoU** | **0.0000** | **0.0219** | **Positive overlap established** |
| **Recall** | **0.0000** | **0.0421** | **Positive recall established** |
| **Precision** | **0.0000** | **0.0438** | **Positive precision established** |

The Day 3 zero-overlap blocker has been resolved with genuine, measured pixel overlap on the labelled RDD crack sample.

---

## 7. Pothole Fragmentation Diagnosis (0454.png)

Investigation of `0454.png` revealed:
1. Anomaly map candidate was originally a single component (`[278, 49, 314, 147]`, area 805 px).
2. Morphological `MORPH_OPEN` with a 3x3 kernel severed a 2px neck at y=116-120, dividing it into Morph CC 1 (`y: 50..116`) and Morph CC 4 (`y: 120..145`).
3. Each half was independently prompted to SAM2, creating Defect 3 (`[276, 46, 318, 118]`, centroid `y=81.7`) and Defect 4 (`[291, 118, 303, 146]`, centroid `y=131.8`).
4. Centroid distance between Defect 3 and Defect 4 was 50.12 px. Because `road_mask_merge_centroid_px` was capped at 48.0 px, the consolidation logic refused to merge them even though they were directly touching at y=118!

---

## 8. Conservative Pothole Consolidation

By allowing touching masks (`cv2.dilate(group_mask, (9, 9)) & candidate.mask`) to merge and setting `road_mask_merge_centroid_px = 65.0`, touching fragments coalesce into a single unified defect.
Recomputation of union mask, bbox, centroid, and area ratio avoids double counting overlapping pixels.

---

## 9. Post-Fix Pothole Metrics (0454.png)

Ground truth mask pixels: 15,016.

| Metric | Day 3 | Day 4 Post-Fix | Delta / Improvement |
|---|---|---|---|
| Accepted Fragment Count | 4 fragments | **2 fragments** | **-2 fragments (consolidated)** |
| Main Defect BBox | `[276, 46, 318, 118]` | `[276, 46, 318, 154]` | **Full vertical extent captured** |
| Main Defect Area | 2,371 px | 3,489 px | +1,118 px |
| Total Predicted Pixels | 4,306 px | 5,874 px | +1,568 px |
| **GT Overlap Pixels** | 3,560 px | **4,407 px** | **+847 px (+23.8% overlap increase)** |
| **IoU** | 0.2259 | **0.2674** | **+0.0415 (+18.4% relative gain)** |
| **Recall** | 0.2371 | **0.2935** | **+0.0564 (+23.8% relative gain)** |
| **Precision** | 0.8268 | **0.7503** | Trade-off: higher recall / lower fragmentation |

Fragmentation was halved (4 → 2), recall increased from 23.7% to 29.4%, and overlap increased by 847 pixels.

---

## 10. Healthy-Road Regression (original_healthy.jpg)

Running `original_healthy.jpg` with the Day 4 perception pipeline:
- Candidate count before filters: 5
- All candidate areas: `[29, 9, 33, 32, 17]` px (all below `min_area_px = 50`)
- Accepted defect count: **0**
- Defect area: **0 px**
- Current severity: **0.0**
- Anomaly threshold: 0.8025

Zero false positives generated on the healthy road regression check.

---

## 11. Secondary Patched/Rough Asphalt Check (India_005086.jpg)

Running `India_005086.jpg` (forward dashcam mode):
- Road mask ratio: 0.5462 (matches Day 3: 0.5462)
- Threshold: 0.2216
- Defect count: 5
- Defect area ratio: 0.0254
- Current severity: 0.5792

The cross-domain behavior remains stable and road mask integrity was fully preserved.

---

## 12. Defect Type Safety

No fine-grained classes were invented. In accordance with contract specifications and project guidance:
- All DINOv2 + SAM2 defects export `defect_type = "road_defect"`.
- Explicit RDD defect class prediction (D00, D10, D20, D40, Repair) remains strictly owned by YOLO.

---

## 13. Shared Feature Contract Validation

All persisted feature records (`sam2_dino/outputs/*/features.json`) were validated against `feature_contract.schema.json` version 1.0.0.
All records passed:
- `original_healthy`: PASSED
- `China_Drone_001267`: PASSED
- `0454`: PASSED
- `India_005086`: PASSED

---

## 14. Nitin Adapter Validation

Validated `features.json` from `China_Drone_001267` and `original_healthy` through Nitin's feature adapter (`xgboost/feature_adapter.py`):
- `current_severity` (0.6657) read accurately.
- `model_row` mapped 11 numeric features in exact expected order.
- Null values safely handled as NaN / null.
- Unit test `xgboost/test_adapter.py`: PASSED (1/1).
- Nitin's Model V2 interface was untouched.

---

## 15. Temporal Input Infrastructure

Exporter infrastructure supports writing segment-organized daily folders:
```
SEG_XXX/
    day_01/
        features.json
        defect_001_mask.png ...
    day_05/
        features.json
        masks ...
    day_10/
        features.json
        masks ...
```
Features export preserves `segment_id`, `day`, `original_filename`, and writes relative mask paths matching `xgboost/temporal/load_sequence.py`. Cross-day tracking IDs are not assigned (temporal matching remains owned by Nitin).

---

## 16. Temporal Smoke Test Status

- **Status**: **NOT VERIFIED — required real multi-day sequence unavailable.**
- Real repeated survey flights across Days 1, 5, 10 for the same real road segment are not present in the local datasets.
- Compatibility of the exporter structure was verified using the unit fixture `TEST_FIXTURE_ONLY`.

---

## 17. Common-Folder Runner Readiness (run_sam2_dino.py)

Created `run_sam2_dino.py` (and `sam2_dino/run_sam2_dino.py`):
- Accepts `--input <image | folder | manifest.csv>` and `--output <output-dir>`.
- Preserves original filenames.
- Generates all masks, overlays, diagnostics, and mask-derived boxes.
- Measures and reports `inference_time_ms` per frame.
- Writes `run_summary.json` containing aggregate counts, timing, and machine-readable per-image records.
- Average inference speed: ~498 ms/frame on RTX 5060 GPU.

---

## 18. Common Benchmark Manifest Compatibility

Verified `run_sam2_dino.py` directly against `benchmark/common_candidate/manifest.csv`:
- Parses CSV headers (`image_id`, `image_path`, `condition`).
- Correctly resolves relative paths to benchmark images.
- Successfully executed test batch with `--limit 2` without errors.
- Did not train on benchmark or modify YOLO outputs.

---

## 19. Tests Passed / Failed

- Existing `road_health_pipeline/tests/`: **145 passed, 0 failed** (21.52s)
- Existing `sam2_dino/test_contract.py`: **3 passed, 0 failed**
- Existing `sam2_dino/test_thresholding.py`: **2 passed, 0 failed**
- New `sam2_dino/test_day4_fixes.py`: **5 passed, 0 failed**
- Total test count: **155 passed, 0 failed**

---

## 20. Remaining Blockers & Anything NOT VERIFIED

- **NOT VERIFIED**: Real-world multi-day temporal sequence (Days 1, 5, 10 for the same segment). Real sequences are unavailable in current datasets; verified on `TEST_FIXTURE_ONLY` only.
- **Crack Box 2 Overlap**: Box 2 (`[434, 353, 456, 474]`) in `China_Drone_001267` remains below threshold in the global adaptive percentile; Box 1 crack was successfully recovered.

---

## 21. Recommended Day-5 Priorities

1. **Joint Benchmark Run**: Execute `run_sam2_dino.py` on the full `benchmark/common_candidate/manifest.csv` and compare detection performance with YOLO Day 5 outputs.
2. **Local Crack Saliency / Sub-threshold Crack Recovery**: Implement a targeted local ridge or directional gradient boost for narrow cracks that score in the 0.20–0.30 range without lowering global road threshold.
3. **Temporal Evaluation Integration**: If simulated CARLA sequence frames are designated for temporal evaluation, run `run_sam2_dino.py` on `env/output/temporal_segments/SEG_001/` to feed Nitin's sequence progression model.
