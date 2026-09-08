# Marion Day 5 — GPU Follow-Up Experiment Plan
**Author**: Marion (Person 1 Perception) / Directed to Team Lead  
**Target Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (Local)  
**Date**: September 8, 2026  

---

## 1. Context & Motivation

In Day 5 perception stabilization, pothole localization reached **IoU = 0.6398** (exceeding the > 0.40 target), and crack GT1 overlap improved by **+220%** (275 px → 882 px). However, GT Box 2 (`[434, 353, 456, 474]`) on `China_Drone_001267.jpg` remains unrecovered due to a fundamental physical limitation of foundation-model patch resolution:

- DINOv2 ViT-S/14 operates on a 37×37 patch grid over 518×518 pixels (~14×14 px per patch).
- Narrow longitudinal cracks are only 1–3 px wide in drone imagery.
- Inside a 14×14 patch (196 px), the crack occupies only ~14% of the area; 86% is normal asphalt.
- This dilutes the patch embedding anomaly score to 0.2745, falling below the 92nd percentile road baseline threshold (0.3515).
- Lowering the global threshold below 0.35 floods healthy road surfaces with false positives.

To resolve thin longitudinal crack representation without sacrificing healthy specificity, a GPU-intensive high-resolution sliding-window inference is proposed.

---

## 2. Proposed GPU Task

**High-Resolution Tiled / Sliding-Window DINOv2 Patch Inference for Longitudinal Distress Corridor**

1. **Exact Proposed GPU Task**:
   Implement a 2× resolution tiled inference wrapper around `Dinov2Embedder`:
   - Crop the road corridor into overlapping 518×518 tiles with 50% stride along the road heading.
   - Extract patch tokens at native full image resolution (effective patch size shrinks from 14 px to ~7 px in real space).
   - Merge tiled patch grids using maximum inner product against the existing healthy memory bank.

2. **Reason**:
   At 2× zoom, a 2 px crack occupies 28% of a patch (doubling the signal-to-noise ratio). This elevates the anomaly score of genuine crack patches into the top 5% of road anomalies without lowering the healthy background threshold.

3. **Inputs Required**:
   - `RoadSentinel_datasets/rdd2022_full/China_Drone/China_Drone/train/images/China_Drone_001267.jpg`
   - `road_health_pipeline/output/real_memory_bank/` (already built and verified)
   - `checkpoints/sam2.1_hiera_small.pt`

4. **Command / Experiment**:
   ```bash
   road_health_pipeline/.venv/bin/python road_health_pipeline/inference/tiled_dinov2_eval.py \
       --image RoadSentinel_datasets/rdd2022_full/China_Drone/China_Drone/train/images/China_Drone_001267.jpg \
       --tile-size 518 \
       --stride 259 \
       --device cuda
   ```

5. **Success Metric**:
   - Crack IoU on `China_Drone_001267.jpg` reaching **> 0.15** (Day 5 baseline: 0.0406).
   - GT Box 2 non-zero overlap (> 200 px).
   - Healthy road (`original_healthy.jpg`) remaining at **0 accepted defects**.

6. **Baseline to Compare Against (Day 5 Baseline)**:
   - Crack IoU: 0.0406
   - GT Overlap: 882 px (GT1: 882 px, GT2: 0 px)
   - Pothole IoU: 0.6398
   - Healthy False Positives: 0
