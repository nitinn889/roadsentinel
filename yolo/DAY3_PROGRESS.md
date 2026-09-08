# RoadSentinel — YOLO Day 3 Robustness and Cross-Domain Evaluation

Date: 2026-09-08
Role: Vrinda / Person 2
Status: **COMPLETE — fixed Day 2 checkpoint evaluated; no retraining performed.**

## Reproducibility and scope

- Repository: `main`, commit `6787e2d` (`origin/main`) after `git fetch`. A rebase was not run because unrelated local changes/artifacts already existed.
- Fixed checkpoint: `yolo/weights/best.pt`, SHA-256 `ddbea38144425aad288f05d90f07a38bcac35eeb467767643c707eb8e2fb5dcf`; loaded and verified as the five-class `D00`, `D10`, `D20`, `D40`, `Repair` model.
- All inference used the frozen checkpoint, CUDA device 0, and one common confidence threshold of **0.25**. No per-image threshold tuning and no model-weight changes occurred.
- No images were downloaded. No DINOv2, SAM2, XGBoost, dashboard, or temporal-pipeline files were modified.

## Condition set

Local datasets did not provide verified real-weather labels for each requested condition. To avoid asserting unverified weather labels, this evaluation uses a compact **synthetic photometric stress set**, not a real-world weather benchmark:

- Six labelled RDD2022 China_Drone validation images: `000001`, `000004`, `000064`, `000112`, `000160`, and `001045`.
- Six geometry-preserving conditions: original `daylight`; deterministic `low_light`, `rain`, `wet_road`, `shadows`, and `fog_haze` transformations.
- 36 images total (six per condition), all with original verified YOLO label paths. The transformations do not move objects, so same-class IoU matching at 0.50 is valid.
- Manifest: `yolo/test_conditions/manifest.csv`; generated local images are intentionally Git-ignored. Each manifest row explicitly identifies the image as a synthetic stress test.

## Quantitative synthetic-stress results

The comparison-ready per-image output is `yolo/outputs/day3_robustness_summary.csv` (36 rows). It reports TP, FP, FN, precision, recall, F1, matched IoU, detections, confidence, and inference time. Aggregate counts are over only 12 ground-truth boxes per condition; do not generalize them as deployment metrics.

| Condition | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Daylight/original | 6 | 10 | 6 | 0.3750 | 0.5000 | 0.4286 |
| Low light (synthetic) | 6 | 4 | 6 | 0.6000 | 0.5000 | 0.5455 |
| Rain (synthetic) | 5 | 6 | 7 | 0.4545 | 0.4167 | 0.4348 |
| Wet road (synthetic) | 5 | 7 | 7 | 0.4167 | 0.4167 | 0.4167 |
| Shadows (synthetic) | 3 | 9 | 9 | 0.2500 | 0.2500 | 0.2500 |
| Fog/haze (synthetic) | 4 | 3 | 8 | 0.5714 | 0.3333 | 0.4211 |

The strongest controlled success is low-light F1 0.5455 on this subset. The strongest controlled failure is shadows: three matches and nine misses/false positives combined. This test is too small for condition-level model selection; **real condition-level quantitative robustness is Not verified**.

## D20 and D40 findings

- Day 2 holdout support was low: D20 had 235 train / 58 validation instances; D40 had 71 train / 15 validation instances. Day 2 class recalls were 0.3793 (D20) and 0.4644 (D40).
- The selected Day 3 subset includes three D20 and three D40 instances per condition. D20 matched 1/3 in daylight, 2/3 in low light and wet-road transforms, 0/3 under shadows, and 1/3 in rain/fog-haze transforms. D40 matched 1/3 in daylight, low-light, rain, wet-road, and shadows, but 0/3 under fog/haze.
- This supports continued concern about both classes under reduced contrast/occlusion-like appearance, especially D20 with shadows and D40 with haze. The set cannot isolate a root cause beyond the documented low support and limited training-domain diversity.

## Cross-domain engineering checks

The fixed runner was rerun on the known four examples; annotated outputs and exact detections are in `yolo/outputs/day3_conditions/cross_domain/results.json`.

| Sample | Result at 0.25 | Interpretation |
|---|---|---|
| `China_Drone_001267.jpg` | Two D00 detections, confidence 0.6710 and 0.6211 | Known China_Drone success; prior verified overlaps were IoU 0.676 and 0.560. |
| `original_healthy.jpg` | No detections | Consistent healthy-road negative engineering check; formal negative-set metric is Not verified. |
| `0454.png` | One marginal D00 (0.2508) | Pothole-600 has a segmentation mask rather than matching five-class detection boxes; this is qualitative evidence of cross-domain/class mismatch, not a detection AP result. |
| `India_005086.jpg` | No detections | Source XML contains a D00 box; this is a verified false negative on a dashcam-view sample and indicates viewpoint/domain-shift sensitivity. |

## Threshold, recommendation, and next steps

- A threshold sweep was **Not run**: the fixed 0.25 threshold supplied a controlled comparison, and this small synthetic subset is not suitable for selecting a production threshold.
- No GPU retraining was run or recommended solely from these 36 transformed images.
- Bounded Team Lead follow-up recommendation: prepare a controlled augmentation experiment with photometric shadow/fog transforms and a limited, clearly held-out D40 supplement from a compatible labelled detection dataset. Preserve the China_Drone validation split and evaluate D20/D40 recall before adopting it.
- Day 4 priorities: obtain a labelled, naturally captured condition-stratified holdout; evaluate the fixed model by real condition and viewpoint; then compare common-benchmark records with other components. Real rain, wet-road, shadows, fog/haze, and low-light robustness remain **Not verified** outside this synthetic stress test.
