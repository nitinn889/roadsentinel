# RoadSentinel YOLO Day 5 Progress

Status: COMPLETE — Team Lead takeover completed after Vrinda's Day 5 work was unavailable/incomplete.

## 1. Takeover and preservation

The repository remained on `main`; existing Vrinda Day 4 and Nitin Day 4 work was
preserved. The unrelated `CarlaUE5` working-tree modification was not staged or
modified. No training was run and `yolo/weights/best.pt` was not changed.

## 2. Frozen checkpoint

- Checkpoint: `yolo/weights/best.pt`
- SHA-256: `ddbea38144425aad288f05d90f07a38bcac35eeb467767643c707eb8e2fb5dcf`
- Classes: `0=D00`, `1=D10`, `2=D20`, `3=D40`, `4=Repair`
- Common confidence threshold: `0.25`

## 3. Dataset audit and acquisition

No new data was downloaded or acquired. The local audit covered RDD2022, MWPD,
Pothole-600, the water-filled/dry pothole collection, India road-damage references,
RoadSentinel simulation frames, and mock fixtures. MWPD and the water-filled/dry
collection are locally present but their schemas/class mappings are not compatible
with the five-class benchmark. The detailed inventory remains in
`yolo/day4_local_condition_inventory.csv`.

## 4. Common benchmark candidate

Created `benchmark/common_candidate/manifest.csv` and an 18-image candidate:

- 14 compatible, naturally captured RDD2022 China_Drone images with YOLO boxes.
- 4 unlabelled/incompatible diagnostic references: `China_Drone_001267.jpg`,
  `original_healthy.jpg`, `0454.png`, and `India_005086.jpg`.
- Conditions represented: daylight (5), shadow (4), and low_light (5) on the
  compatible subset; four references are `unknown`.
- Viewpoints: UAV/nadir for 17 images and dashcam for India_005086; the Pothole-600
  reference is ground-level/nadir.
- Compatible labelled ground truth: 14 images / 24 boxes. Unlabelled or incompatible
  references: 4 images.
- D20 is represented; D40 is not represented in this small candidate because no
  compatible D40-labelled candidate was selected locally.

All condition provenance is `VISUAL_LABEL_ONLY` for the three RDD categories and
`UNKNOWN` for the four diagnostic references. No image is claimed as verified rain,
wet road, or fog/haze.

## 5. YOLO run and metrics

The frozen YOLO checkpoint ran reproducibly on all 18 candidate images with the
standard runner, threshold 0.25, device `0`, producing 31 detections. Outputs are in
`yolo/outputs/day5_common_candidate/`. The standardized CSV preserves image ID,
filename, model, checkpoint hash, class, confidence, box, timing, condition,
provenance, and viewpoint.

Strict IoU metrics are calculated only on the 14 compatible labelled images:

| class | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| D00 | 2 | 0 | 2 | 1.0000 | 0.5000 | 0.6667 |
| D10 | 3 | 3 | 2 | 0.5000 | 0.6000 | 0.5455 |
| D20 | 1 | 5 | 2 | 0.1667 | 0.3333 | 0.2222 |
| D40 | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |
| Repair | 9 | 5 | 3 | 0.6429 | 0.7500 | 0.6923 |

mAP50 and mAP50-95 were not calculated for this small development candidate.
Condition-level metrics are descriptive only; each condition is below a defensible
sample size for a quantitative weather claim. Therefore: **Insufficient sample size
for quantitative condition claim**.

## 6. D20 / D40 and domain evidence

Day 2 baseline recalls remain D20=0.3793 and D40=0.4644. Day 5 adds limited D20
evidence under visual shadow/low-light UAV conditions and no new D40 evidence. The
observed limitations remain consistent with possible low support, scale, viewpoint,
texture, class confusion, and domain shift; causality is not established.

## 7. Marion readiness and comparison

Marion's repository code supports its own fixed-test manifest and produces masks,
mask geometry, features, and diagnostics, but it does not currently provide a stable
common-candidate interface with the same five-class ground truth and condition
manifest. Its existing diagnostic outputs are on a different four-image set.

Cross-model comparison: **deferred pending Marion pipeline readiness**. No YOLO vs
DINOv2+SAM2 comparison was performed. No claim of equal training is made; a future
fair comparison means same images, ground truth, and hardware.

## 8. Real-weather status

Natural rain, wet-road, and fog/haze coverage with source-backed condition metadata
is **NOT VERIFIED / NOT AVAILABLE**. Low-light, daylight, and shadow are visual
categories only, not sensor or metadata-verified weather. The Pothole-600 mask and
water-filled/dry dataset cannot be silently converted to the five-class benchmark;
any mask-derived box must be marked `BBOX_DERIVED_FROM_MASK`.

## 9. GPU retraining recommendation

GPU retraining recommendation = **NO** for Day 5. No compatible labelled training
supplement was found, and the candidate is an evaluation/development set. Do not
train on it. A future proposal can be written only after a defensible D40 and/or
verified-weather training supplement is acquired, with validation preserved and a
pre-registered success/failure criterion.

## 10. Panel demo examples — NOT FINAL BENCHMARK CLAIMS

- Clear positive: `China_Drone_001267.jpg` — two D00 detections.
- Healthy negative: `original_healthy.jpg` — zero detections.
- Difficult visual-condition example: `China_Drone_000258.jpg` — shadow-labelled D20 case.
- Cross-domain limitation: `India_005086.jpg` — dashcam reference missed by YOLO in
  the existing fixed sample; `0454.png` is the weak/marginal Pothole-600 reference.

These are demo/reference examples only and were not used for per-image tuning.

## 11. Regression and remaining work

`yolo/run_yolo.py --help` passed. Regression remained stable: China_Drone_001267.jpg
produced two D00 detections and original_healthy.jpg produced zero.

Day 6 priority: obtain a small source-backed natural-weather set, preserve a separate
untouched evaluation split, then make Marion consume the exact common manifest and
run the deferred same-image comparison. The exact blocker is Marion common-interface
compatibility plus verified adverse-weather ground truth. Do not claim real-weather
robustness until both are resolved.

