# RoadSentinel — YOLO Day 2 GPU Training Progress

Date: 2026-09-08
Role: Team Lead
Status: **COMPLETE — Day 2 GPU training executed by Team Lead on RTX 5060.**

## Environment and repository

- Branch: `main`, at `2ff8e57` (`origin/main`) after `git fetch origin`; it was already current. A rebase was not run because pre-existing unrelated workspace changes were present (`CarlaUE5` and local YOLO artifacts).
- The required source/configuration files and the corrective Day 1 report were present. Conflict-marker search found no unresolved merge conflict; separator lines elsewhere in the repository are not conflict markers.
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB); driver 595.84 / CUDA capability reported by `nvidia-smi` as CUDA 13.2.
- Runtime: Python 3.14.4 in `./.venv`, PyTorch 2.13.0+cu130 with CUDA available, Ultralytics 8.4.143.

## Dataset and Day 1 preservation

- Dataset: existing local RDD2022 `China_Drone` subset only; no data was downloaded and the future final benchmark was not used for training.
- Deterministic existing split: 1,921 train and 480 validation images, seed 42. All 2,401 images have matching labels; no missing or dangling labels and no train/validation filename overlap were found.
- Classes verified from `best.pt` and `class_mapping.json`: `0=D00`, `1=D10`, `2=D20`, `3=D40`, `4=Repair`.
- Day 1 artifacts remain intact. The original 3-epoch checkpoint was copied locally to `yolo/weights/day1_best.pt` before training (ignored from Git by design); Day 1 metrics remain in `DAY1_CORRECTIVE_PROGRESS.md`.

## Day 2 controlled training

- Starting weights: `yolov8n.pt` (COCO-pretrained initialization); model head was replaced for the five road-defect classes.
- Command: `./.venv/bin/python yolo/src/train_yolo.py --epochs 25 --batch 32 --imgsz 512 --device 0 --patience 7 --seed 42 --name rdd2022_day2`.
- Configuration: 512 px, batch 32, CUDA device 0, 25 epochs, patience 7, seed 42, AMP enabled, 4 data-loader workers. Ultralytics `optimizer=auto` selected AdamW (lr 0.001111, momentum 0.9).
- Training completed all 25 epochs (early stopping was not triggered) in 163.32 s. GPU memory remained within capacity (about 3.4 GB used during training).
- Full run artifacts are deliberately excluded from Git at `yolo/outputs/training/rdd2022_day2/`; they include `results.csv`, plots, and both run checkpoints.

## Validation and checkpoint selection

`yolo/weights/best.pt` is the best Day 2 validation checkpoint (SHA-256 `ddbea38144425aad288f05d90f07a38bcac35eeb467767643c707eb8e2fb5dcf`). It was independently loaded with Ultralytics and verified to expose exactly the five road-defect classes, not COCO's 80 classes. A fresh validation used the same holdout with `workers=0` because Python 3.14 cannot spawn data-loader workers from an inline validation invocation.

| Metric | Day 1 (3 epochs) | Day 2 (25 epochs) |
|---|---:|---:|
| Precision | 0.3895 | 0.7178 |
| Recall | 0.2935 | 0.6146 |
| mAP50 | 0.2628 | 0.6774 |
| mAP50-95 | 0.1407 | 0.4068 |

The Day 2 holdout metrics are higher on all four measures; this is an engineering validation comparison, not a final research-benchmark claim.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| D00 | 0.7246 | 0.7024 | 0.7246 | 0.3736 |
| D10 | 0.7734 | 0.7422 | 0.8062 | 0.4525 |
| D20 | 0.5670 | 0.3793 | 0.4708 | 0.2523 |
| D40 | 0.7768 | 0.4644 | 0.5581 | 0.3217 |
| Repair | 0.7473 | 0.7846 | 0.8272 | 0.6337 |

## Fixed engineering samples and runner

The standard runner help and folder inference were verified using:

```bash
./.venv/bin/python yolo/run_yolo.py \
  --input yolo/data/smoke_test_images \
  --output yolo/outputs/inference/day2_fixed_samples \
  --weights yolo/weights/best.pt --conf 0.25 --device 0
```

It preserves original filenames and emits `image_name`, `class_id`, `class`, `confidence`, `bbox`, and `inference_ms` per image in `results.json`.

| Image | Day 2 result at conf 0.25 | Inference time |
|---|---|---:|
| `China_Drone_001267.jpg` | 2 D00: 0.6710 `[435,2,463,206]`; 0.6211 `[432,296,457,486]` | 4.61 ms |
| `original_healthy.jpg` | 0 detections | 2.77 ms |
| `0454.png` | 1 low-confidence D00: 0.2508 `[187,180,311,260]` | 867.27 ms (first-run warm-up) |
| `India_005086.jpg` | 0 detections | 3.03 ms |

For the labelled China_Drone sample, both D00 detections overlap the two holdout ground-truth D00 boxes (IoU 0.676 and 0.560). The engineering output is at `yolo/outputs/inference/day2_fixed_samples/results.json` (ignored as generated output).

## Failure analysis and Day 3 handoff

- **D20 remains weakest by recall** (0.3793); its holdout support is only 58 instances.
- **D40 potholes remain constrained by scarcity** (15 validation / 71 training instances). Its recall is 0.4644, below D00, D10, and Repair. The `0454.png` pothole-oriented sample also produced a marginal D00 detection, so cross-domain pothole behavior needs focused inspection.
- `India_005086.jpg` contains a source D00 annotation but received no Day 2 detection at 0.25, indicating viewpoint/domain-shift sensitivity between China drone training imagery and India dashcam imagery.
- Repair is strongest; inspect repair-vs-defect confusions and oversized/undersized boxes in Day 3 rather than changing thresholds from the final benchmark.
- Not verified: daylight/low-light/rain/wet-road/shadow/fog robustness and any full final research benchmark. No Day 3 benchmark or extra data download was started.

For Day 3, Vrinda should run condition-wise evaluations with this fixed checkpoint, retain the existing China_Drone holdout, report degradation by class/condition, and review the D20/D40 and cross-domain failures before proposing a bounded follow-up experiment.
