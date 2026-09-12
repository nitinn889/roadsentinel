# Phase UAV-2A — Manifest Remediation and Low-Cost Multi-UAV YOLO Pilot

**Decision: `INCONCLUSIVE`. Full multi-seed training is not justified by this pilot.**

M-UAV1 improved the source-balanced Macro F1 from **0.2461** to **0.2815**, an absolute gain of **+0.0354** and a relative gain of **+14.4%**. The evidence is not strong enough for `GO`: the group-clustered 95% interval for the gain is **[-0.0095, +0.0703]**, China pothole recall decreased from **0.1563 to 0.1250**, only 15 UAV-PDD potholes were available for validation, group-level gains were not broad in China, and M-UAV0 stopped at epoch 32 while M-UAV1 ran through epoch 40. No predeclared material-regression condition was strong enough for `NO-GO`.

## 1. Scope and manifest correction

The Phase 1 exclusive `role` field was remediated into the top-level partitions `train`, `calibration`, `validation`, `internal_test`, `external_test`, and `severe_shift_test`. Separate Boolean role columns now permit YOLO training, DINOv2 reference-bank use, and SAM2 development use to overlap only inside `train`. Calibration, validation, and test functions remain isolated.

The remediation exposed a Phase 1 grouping defect in UAV-PDD: transformed crops such as `lr_00490_top_left`, `tb_00490_top_left`, and `r180_00490_top_left` derive from the same numbered UAV photograph. The corrected `isolation_group_id` is the original photograph number, not the transformed crop ID. This moved every crop associated with a protected source photograph into the strongest protected partition. The resulting UAV-PDD partition has 5 train photographs/33 images, 11 calibration photographs/98 images, 16 validation photographs/123 images, and 205 held-out photographs/2,178 images.

Valid training-only multi-role reuse recovered **313 manifest rows**, of which **257 are pilot-eligible after canonical-label and empty-image filtering**: 246 China images and 11 UAV-PDD images. The machine-readable correction is in `remediated_role_manifest.csv`.

No training, calibration, validation, or test group crosses a top-level partition. China grouping remains based on the available Phase 1 pHash components; acquisition-sequence metadata are unavailable, so residual China group-independence uncertainty remains.

## 2. Duplicate and UAPD review

All **31** flagged cross-source pHash pairs were manually reviewed side by side at full-image resolution. All 31 were classified **Visually different**; none was classified confirmed duplicate, likely near-duplicate, or unresolved. The decisions are in `phash_manual_review.csv`, with seven paired review sheets under `pilot_figures/phash_review_*.jpg`.

The prior exact-overlap finding is preserved: all **2,401 China_Drone images occur inside UAPD**, so full UAPD is not described or used as independent evaluation data. The remaining **749 UAPD candidates** remain in `external_test` and did not enter training, validation, model selection, or the pilot decision.

China held-out, UAV-PDD held-out, Poznań, and India stress-test roles were preserved. None was evaluated in this pilot.

## 3. Canonical-empty audit

Every one of the **269** canonical-empty images in the Phase 1 harmonized YOLO training view contains an explicit noncanonical source label such as repair, patching, or oblique distress. All 269 were classified **Contains only excluded visible distress** and excluded; none was silently treated as a clean negative.

The broader remediated training-candidate audit contains:

| Decision category | Images | Pilot action |
|---|---:|---|
| Genuine healthy/background image | 5 | Included as verified negatives when in the train partition |
| Contains only excluded visible distress | 323 | Excluded |
| Contains ambiguous distress | 2 | Excluded |
| Annotation is incomplete | 0 in the audited pilot-eligible scope | Excluded by policy if encountered |
| Corrupted or unusable | 0 | Excluded by policy if encountered |

The five unique verified negatives are two China images and three transforms from one UAV-PDD source photograph. The audit is in `canonical_empty_audit.csv`. `canonical_empty_decision_samples.jpg` shows assigned categories and explicit empty-category placeholders where no image was assigned. YOLO ignore regions were not introduced.

## 4. Prepared datasets and class balance

M-UAV0 contains **1,149 China training images in 1,106 groups**. M-UAV1 contains exactly those same 1,149 China images plus **30 UAV-PDD images from 5 original-photograph groups**, for **1,179 images in 1,111 groups**. Three protected-neighbouring or noncanonical-empty UAV-PDD rows among the 33 train-partition images were not eligible. No source-aware or class-aware oversampling was used, no fake labels were generated, and neighbouring crops were not replicated.

### Training distribution

| Model | Source | Images | Groups | Longitudinal | Transverse | Alligator | Pothole | Objects |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M-UAV0 | China | 1,149 | 1,106 | 867 | 770 | 175 | 36 | 1,848 |
| M-UAV1 | China | 1,149 | 1,106 | 867 | 770 | 175 | 36 | 1,848 |
| M-UAV1 | UAV-PDD | 30 | 5 | 27 | 45 | 6 | 0 | 78 |

The UAV-PDD addition contains no pothole training instances. This is a central limitation of the pilot.

### Locked validation distribution

| Source | Images | Groups | Longitudinal | Transverse | Alligator | Pothole | Objects |
|---|---:|---:|---:|---:|---:|---:|---:|
| China | 136 | 127 | 95 | 73 | 24 | 32 | 224 |
| UAV-PDD | 123 | 16 original photographs | 165 | 243 | 39 | 15 | 462 |

Both models used the same 259 validation sample IDs. Neither model used a final held-out or external-test image for training, validation, early stopping, or the decision.

## 5. Exact training and evaluation protocol

Both models began from the same local pretrained `yolov8n.pt`, used seed **20260912**, image size **640**, batch size **16**, device **CUDA:0**, 4 workers, maximum **40 epochs**, patience **10**, AdamW, `lr0=0.001`, `lrf=0.01`, cosine learning rate, momentum **0.937**, weight decay **0.0005**, and 3 warm-up epochs. AMP and deterministic mode were enabled; cache was disabled.

The identical augmentation settings were: `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `degrees=0`, `translate=0.1`, `scale=0.5`, `shear=0`, `perspective=0`, `flipud=0`, `fliplr=0.5`, `bgr=0`, `mosaic=1`, `mixup=0`, `cutmix=0`, `copy_paste=0`, and `close_mosaic=10`. Architecture and loss were unchanged between models.

Fixed evaluation settings were confidence **0.25**, NMS IoU **0.70**, and class-specific greedy matching at IoU **0.50**. AP uses confidence-ranked detections retained from 0.001, 101-point interpolation, and IoU 0.50 for AP50 or the mean over 0.50:0.95 for mAP50:95. The source-balanced primary metric is exactly `mean(China Macro F1, UAV-PDD Macro F1)`; images were not pooled across sources for the decision.

## 6. Smoke tests and environment

Both two-epoch smoke tests passed before pilot training. Images and labels loaded, losses were finite and stable, validation ran, checkpoints were written, and no NaN or CUDA error occurred. Smoke outputs are segregated under `outputs/uav2a_pilot/smoke/` and their records are marked `stage=smoke`.

| Model | Duration | Loss, first → final | Best smoke mAP50:95 | Peak GPU memory | Status |
|---|---:|---:|---:|---:|---|
| M-UAV0 | 22.290 s | 7.99565 → 6.75772 | 0.03594 | 1.8255 GiB | PASS |
| M-UAV1 | 17.567 s | 7.99741 → 6.80904 | 0.02052 | 1.8537 GiB | PASS |

Environment: NVIDIA GeForce RTX 5060 Laptop GPU, driver 595.84, Python 3.14.4, PyTorch 2.13.0+cu130, CUDA 13.0, and Ultralytics 8.4.143.

## 7. Pilot results

### Per-source metrics at confidence 0.25

| Model | Source | P | R | F1 | Macro F1 | mAP50 | mAP50:95 | Mean matched IoU | Pothole R | GT / Pred | FP / FN | No-pred images |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M-UAV0 | China | 0.5472 | 0.5179 | 0.5321 | 0.3998 | 0.4827 | 0.2388 | 0.7498 | 0.1563 | 224 / 212 | 96 / 108 | 34 |
| M-UAV1 | China | 0.4834 | 0.5848 | 0.5293 | 0.4331 | 0.4801 | 0.2525 | 0.7707 | 0.1250 | 224 / 271 | 140 / 93 | 25 |
| M-UAV0 | UAV-PDD | 0.3182 | 0.1364 | 0.1909 | 0.0924 | 0.0740 | 0.0252 | 0.6668 | 0.0000 | 462 / 198 | 135 / 399 | 35 |
| M-UAV1 | UAV-PDD | 0.2685 | 0.2273 | 0.2462 | 0.1299 | 0.0824 | 0.0265 | 0.6722 | 0.0000 | 462 / 391 | 286 / 357 | 16 |

M-UAV1 trades precision for recall on both sources. China Macro F1 rose by **+0.0333 (+8.3%)**, while overall China F1 was nearly flat at **-0.0028**. UAV-PDD Macro F1 rose by **+0.0376 (+40.7%)**, recall by **+0.0909**, and F1 by **+0.0553**, but false positives rose from 135 to 286. China mAP50:95 rose **+0.0136**; UAV-PDD mAP50:95 rose only **+0.0013**.

### Per-class F1 and AP50

| Source | Class | M-UAV0 F1 | M-UAV1 F1 | Δ F1 | M-UAV0 AP50 | M-UAV1 AP50 |
|---|---|---:|---:|---:|---:|---:|
| China | Longitudinal | 0.5366 | 0.5545 | +0.0180 | 0.5668 | 0.5861 |
| China | Transverse | 0.6545 | 0.6096 | -0.0449 | 0.6727 | 0.6744 |
| China | Alligator | 0.1379 | 0.3462 | +0.2082 | 0.1151 | 0.3680 |
| China | Pothole | 0.2703 | 0.2222 | -0.0480 | 0.5760 | 0.2919 |
| UAV-PDD | Longitudinal | 0.1014 | 0.1618 | +0.0604 | 0.0836 | 0.0876 |
| UAV-PDD | Transverse | 0.2680 | 0.3222 | +0.0542 | 0.2104 | 0.2239 |
| UAV-PDD | Alligator | 0.0000 | 0.0357 | +0.0357 | 0.0018 | 0.0182 |
| UAV-PDD | Pothole | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |

Complete per-class precision, recall, F1, AP50, mAP50:95, object counts, errors, and matched IoU are in `per_class_metrics.csv`.

### Source-balanced comparison and uncertainty

| Metric | M-UAV0 | M-UAV1 | Absolute Δ | Relative Δ |
|---|---:|---:|---:|---:|
| Source-balanced Macro F1 | 0.2461 | 0.2815 | **+0.0354** | **+14.4%** |

The paired, source-stratified cluster bootstrap used 2,000 replicates and seed 20260912. UAV-PDD clusters are original photographs; China clusters use Phase 1 pHash components.

| Bootstrap quantity | Point Δ | 95% interval |
|---|---:|---:|
| Source-balanced Macro F1 | +0.0354 | [-0.0095, +0.0703] |
| China Macro F1 | +0.0333 | [-0.0430, +0.0981] |
| UAV-PDD Macro F1 | +0.0376 | [-0.0030, +0.0681] |
| China pothole recall | -0.0313 | [-0.2274, +0.1000] |
| UAV-PDD pothole recall | 0.0000 | [0.0000, 0.0000] |

The intervals are approximate because China pHash groups may not capture every neighbouring frame or acquisition sequence. All three Macro F1 intervals for the difference include zero.

## 8. Training cost

| Model | Epochs completed / max | Best epoch | Optimization steps | Duration | Peak GPU memory | Mean validation inference latency |
|---|---:|---:|---:|---:|---:|---:|
| M-UAV0 | 32 / 40 | 22 | 2,304 | 191.046 s | 1.8331 GiB | 18.95 ms/image |
| M-UAV1 | 40 / 40 | 39 | 2,960 | 237.771 s | 1.8794 GiB | 16.89 ms/image |

M-UAV0 stopped under the shared patience-10 rule; M-UAV1 reached the 40-epoch cap. The settings and stopping rule were identical, but unequal effective steps reduce confidence in a one-seed causal comparison.

## 9. Failure analysis and figures

The dominant pattern is higher recall with substantially more false positives in M-UAV1. It improves alligator-crack detection in China and crack recall in UAV-PDD, but it regresses China transverse F1 and pothole detection. Both models completely miss all 15 UAV-PDD potholes at confidence 0.25. The addition supplied no UAV-PDD pothole training boxes, so this pilot cannot establish pothole transfer.

Artifacts under `reports/uav_multidomain/pilot_figures/` include:

- `side_by_side_metrics.png`, `per_source_macro_f1.png`, and `per_class_f1.png`;
- source-specific confusion matrices and precision-recall curves;
- `representative_improvements.jpg`, `representative_regressions.jpg`, and `representative_failures.jpg`, with ground truth in green and prediction panels labelled by model;
- the pHash review sheets and canonical-empty decision samples.

## 10. Predeclared decision

The exact rule evaluation is stored in `go_no_go_decision.json`.

- Practical-gain condition: **pass** (+0.0354 absolute and +14.4% relative).
- China Macro F1 decline no greater than 0.02: **pass** (+0.0333).
- No pothole-recall decrease on either source: **fail** (China -0.0313).
- Broad improvement across groups: **fail**. Of groups with differing image-level F1, M-UAV1 led in 29/63 China groups (46.0%) and 10/16 UAV-PDD groups (62.5%); the predeclared breadth threshold was at least 60% in each source.
- Evaluation integrity: **pass**.
- Sufficient pothole support: **fail** for UAV-PDD (15 objects; the minimum support used for GO was 20 per source).
- Material NO-GO regression: **not triggered**. China Macro F1 improved, the balanced gain was positive, and the -0.0313 pothole-recall change did not cross the predeclared -0.05 material-regression threshold.

Therefore the result is **`INCONCLUSIVE`**. Full multi-seed training is **not justified now**. The bounded next experiment, if separately authorized, should first acquire or release group-safe UAV-PDD pothole training support and strengthen validation grouping; this phase stops without starting that work.

## 11. Integrity verification

`validate_phase_uav2a.py` passed all gates:

- identical comparable training arguments and the same pretrained initialization;
- identical 259-image validation set and identical 1,149-image China training subset;
- no train/validation isolation-group crossing and no protected-partition sample;
- no HighRPD, UAPD, Poznań, or India row in either materialized pilot dataset;
- no excluded-distress canonical-empty image used as a clean negative;
- complete source, class, and 518-row per-image metric coverage;
- all smoke and pilot run records passed;
- production YOLO weights, DINOv2 embeddings, FAISS index, XGBoost scenario model, and canonical metrics match their frozen SHA-256 values.

DINOv2, SAM2, reliability, XGBoost, the persistence router, dashboard headline cards, and canonical research metrics were not trained, evaluated, or modified. Pilot checkpoints remain only in the Git-ignored `outputs/uav2a_pilot/` tree and were not promoted.

## 12. Files created or changed

Relevant code:

- `scripts/uav_multidomain/prepare_phase_uav2a.py`
- `scripts/uav_multidomain/train_phase_uav2a.py`
- `scripts/uav_multidomain/evaluate_phase_uav2a.py`
- `scripts/uav_multidomain/validate_phase_uav2a.py`

Required artifacts:

- `artifacts/uav_multidomain/pilot/remediated_role_manifest.csv`
- `artifacts/uav_multidomain/pilot/canonical_empty_audit.csv`
- `artifacts/uav_multidomain/pilot/phash_manual_review.csv`
- `artifacts/uav_multidomain/pilot/training_inventory.csv`
- `artifacts/uav_multidomain/pilot/pilot_runs.csv`
- `artifacts/uav_multidomain/pilot/per_source_metrics.csv`
- `artifacts/uav_multidomain/pilot/per_class_metrics.csv`
- `artifacts/uav_multidomain/pilot/per_image_predictions.csv`
- `artifacts/uav_multidomain/pilot/bootstrap_comparison.json`
- `artifacts/uav_multidomain/pilot/go_no_go_decision.json`

Supporting small artifacts include environment/configuration JSON, preparation and integrity validation JSON, the report, and the figures listed above. Raw datasets, generated materialized views, checkpoints, caches, archives, credentials, and temporary review files are excluded from Git.

## 13. Reproduction commands

From the repository root with the project virtual environment:

```bash
.venv/bin/python scripts/uav_multidomain/prepare_phase_uav2a.py
.venv/bin/python scripts/uav_multidomain/train_phase_uav2a.py --stage smoke --model all
.venv/bin/python scripts/uav_multidomain/train_phase_uav2a.py --stage pilot --model all
.venv/bin/python scripts/uav_multidomain/evaluate_phase_uav2a.py
.venv/bin/python scripts/uav_multidomain/validate_phase_uav2a.py
```

Training commands intentionally replace only the corresponding ignored run directory. Re-running them will retrain and overwrite that local pilot run, never `yolo/weights/best.pt`.

## 14. Remaining limitations

- Only one seed was run, as required for this low-cost gate.
- Only 30 UAV-PDD training images from five independent source photographs survived strict source-photo isolation; these add 78 boxes and zero potholes.
- UAV-PDD validation has only 16 source photographs and 15 potholes; both models have zero fixed-threshold pothole recall there.
- China grouping is pHash-based without acquisition sequence metadata, so clustered intervals are approximate.
- M-UAV0 early-stopped at 32 epochs while M-UAV1 reached 40, despite identical stopping settings.
- The confidence interval for the primary delta includes zero, and the observed benefit is recall-heavy with a large false-positive increase.
- External and final held-out sets were intentionally not evaluated, so no production or generalization claim is made.
