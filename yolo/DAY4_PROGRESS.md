# RoadSentinel YOLO Day 4 Progress

Status: COMPLETE — finished by Team Lead after partial Vrinda run.

## 1. Vrinda partial-work recovery

Vrinda's WIP commit `e8d835c` was pulled onto `main` without conflict. The unrelated
working-tree modification under `CarlaUE5` was preserved. No existing YOLO checkpoint
was overwritten and no training was run.

## 2. Work already completed by Vrinda

- Audited the locally visible RoadSentinel datasets and created `day4_local_condition_inventory.csv`.
- Created a small 14-image RDD2022 China_Drone holdout with YOLO labels and manifest.
- Ran fixed-checkpoint inference on all 14 holdout images at confidence 0.25 and saved annotated images/results.
- Recorded visual condition notes and the known cross-domain fixed samples in prior Day 2/3 outputs.

## 3. Team Lead completion

- Verified the active checkpoint hash and class mapping.
- Corrected the inventory to reflect locally present MWPD (3,087 image/label pairs) and the water-filled/dry annotated collection (713 images).
- Normalized holdout metadata paths to repository-relative POSIX paths.
- Added `src/export_day4_comparison.py`, producing a comparison-ready YOLO CSV and strict box metrics JSON.
- Recomputed the standardized export from the existing Vrinda inference results; inference was not repeated because all holdout samples were already processed.
- Ran CLI help and the required positive/negative regression checks.

## 4. Checkpoint

- Active weights: `yolo/weights/best.pt`
- SHA-256: `ddbea38144425aad288f05d90f07a38bcac35eeb467767643c707eb8e2fb5dcf`
- Classes verified: `0=D00`, `1=D10`, `2=D20`, `3=D40`, `4=Repair`.
- Inference threshold: 0.25. No per-image tuning.

## 5. Local dataset inventory and condition availability

The inventory records RDD2022, India/RDD reference data, Pothole-600, MWPD, the
water-filled/dry dataset, simulation frames, and mock fixtures. Only the 14-image
RDD2022 China_Drone holdout is used for this Day 4 labelled analysis.

Holdout coverage: daylight 5 images / 6 labelled boxes; shadow 4 / 9; low_light 5 / 9.
There are no selected natural rain, wet_road, or fog_haze images. The holdout is a
small engineering/development set, not a final paper benchmark.

## 6. Provenance and metric validity

All three holdout conditions are `VISUAL CONDITION LABEL — NOT SENSOR/METADATA VERIFIED`.
They are naturally captured RDD2022 images, but the local source does not verify that
an image was taken during rain, fog, or on a wet road. Therefore real-weather coverage
is **Not available / Not verified**. Metrics are valid only as compatible YOLO
bounding-box detection metrics on the supplied labels; no mAP is claimed for this
small mixed collection.

## 7. Fixed-checkpoint results

The existing Vrinda run processed 14 images and produced 28 detections. At IoU 0.50,
aggregate per-class results are:

| class | TP | FP | FN | precision | recall | F1 | mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| D00 | 2 | 0 | 2 | 1.0000 | 0.5000 | 0.6667 | 0.7435 |
| D10 | 3 | 3 | 2 | 0.5000 | 0.6000 | 0.5455 | 0.7863 |
| D20 | 1 | 5 | 2 | 0.1667 | 0.3333 | 0.2222 | 0.7142 |
| D40 | 0 | 0 | 0 | not evaluated | not evaluated | not evaluated | not evaluated |
| Repair | 9 | 5 | 3 | 0.6429 | 0.7500 | 0.6923 | 0.8975 |

The per-condition aggregate counts are retained in the manifest and underlying
results; they are descriptive only because condition labels are visual.

## 8. Viewpoint/domain findings

The holdout is UAV/nadir RDD2022 imagery, so it does not test dashcam viewpoint or
natural weather. Existing fixed samples remain broadly consistent: `China_Drone_001267.jpg`
has two D00 detections, while `original_healthy.jpg` has zero. Existing engineering
references show weak/marginal D00 on Pothole-600 `0454.png` and a missed D00 on
India `India_005086.jpg`. These support a cautious domain-shift interpretation across
UAV vs dashcam viewpoint, texture, object scale, and pothole/crack appearance; they
do not isolate one cause and were not used for tuning.

## 9. D20 / D40 analysis

Day 2 statistics remain unchanged: D20 had 235 train / 58 validation images with
recall 0.3793; D40 had 71 / 15 with recall 0.4644. The Day 4 holdout contains D20
labels but no D40 labels, so it adds limited qualitative evidence for D20 under
visually shadowed/low-light UAV imagery and no new D40 evidence. Plausible contributors
remain low support, small-object scale, texture variation, viewpoint, class confusion,
and domain shift; the evidence does not justify selecting one definite cause.

## 10. Standardized comparison readiness

`outputs/day4_comparison/yolo_records.csv` preserves image ID, original filename,
model, checkpoint hash, class ID/name, confidence, box, inference time, condition,
condition provenance, and viewpoint. `metrics.json` records the valid IoU-based
holdout metrics. Marion's files were not modified, and no final YOLO vs DINOv2+SAM2
comparison was run.

## 11. Real-weather status and GPU recommendation

Real natural rain, wet-road, fog/haze, and sensor/metadata-verified condition coverage
are **Not Verified**. The evidence does not justify automatic retraining. A bounded
GPU follow-up is justified as a recommendation only: add a small, independently
labelled D40 supplement plus verified weather imagery, train from this baseline with
the supplement, and compare D40 recall and overall mAP50 against the unchanged
baseline. Success requires improved D40 recall without reducing overall mAP50 by
more than 0.02 on the held-out RDD2022 validation set. Do not execute this during Day 4.

## 12. Day 5 recommendation / blocker

First acquire or formally document a small natural-weather set with source-backed
rain, wet-road, fog/haze, and low-light provenance, keeping it separate from tuning.
Then run the planned standardized YOLO vs DINOv2+SAM2 comparison and report results
by viewpoint/domain. Until such data exist, the exact blocker is the absence of
verified real-weather ground truth; do not claim real-weather robustness.

