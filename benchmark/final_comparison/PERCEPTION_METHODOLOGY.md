# RoadSentinel Perception Benchmark Methodology

## 1. Common Evaluation Benchmark Definition
To establish a rigorous, scientifically valid comparison between task-specific supervised object detection and foundation-model anomaly perception, both systems are evaluated on a **Common Evaluation Benchmark** (designated the **COMMON VALIDATION BENCHMARK**).

- **Benchmark Source**: The official validation split of the RDD2022 China_Drone dataset.
- **Image Resolution**: 512 × 512 pixels across all frames (UAV / nadir camera viewpoint).
- **Scale**: Exactly 480 physical images containing 742 ground-truth defect instances across 5 international damage categories (D00, D10, D20, D40, Repair).
- **Dataset Partitioning & Transparency**: The 480 validation images possess **zero overlap** with the 1,921 training images utilized during supervised YOLOv8n development. However, because this partition originates from the same sensor domain, geography, and annotation team as the training set, it is transparently documented as a validation benchmark rather than an independent blind test set.

---

## 2. Frozen System Configurations
Both perception systems are strictly frozen (`PERCEPTION_PIPELINES_FROZEN = TRUE`). No retraining, fine-tuning, threshold tuning, or post-result hyperparameter adjustments were performed:

1. **System A — Supervised Object Detector (YOLOv8n)**:
   - Architecture: Ultralytics YOLOv8n (3.01M parameters, 8.1 GFLOPs).
   - Weights: `yolo/weights/best.pt` (SHA-256: `ddbea381...`), trained for 25 epochs at 512×512.
   - Inference Settings: Confidence threshold = 0.25, IoU NMS threshold = 0.70, native bounding-box format.

2. **System B — Zero-Shot Foundation Anomaly Pipeline (DINOv2 + SAM2)**:
   - Architecture: Marion Day-5 NORMAL production pipeline.
   - Backbone: DINOv2 ViT-S/14 (`dinov2_vits14`, 384 embedding dim, 37×37 patch grid).
   - Segmenter: Segment Anything Model 2.1 (`sam2.1_hiera_small`).
   - Pipeline Components: Fixed healthy-road memory bank (10,000 vectors sampled from 60 pristine road frames), domain-adaptive 92nd-percentile road anomaly thresholding, vehicle hull suppressor, road marking suppressor, connected-component candidate extraction, and SAM2 prompt-based mask refinement.
   - Output Representation: **Mask-derived bounding boxes** computed directly from binary segmentations (`[min_x, min_y, max_x + 1, max_y + 1]`).

---

## 3. Compatible Binary Defect Collapse
A direct semantic class-to-class comparison between YOLOv8n and DINOv2+SAM2 is fundamentally impossible because DINOv2+SAM2 is an unsupervised anomaly detector that outputs generic road defect regions (`road_defect`), whereas YOLOv8n is supervised on 5 distinct CRDDC classes (`D00`, `D10`, `D20`, `D40`, `Repair`).

To ensure a fair and un-fabricated evaluation:
1. All compatible ground-truth annotations across the 5 classes are collapsed to a single unified binary category: `road_defect`.
2. All YOLOv8n detections across all 5 classes are collapsed to `road_defect`.
3. DINOv2+SAM2 accepted mask-derived boxes are evaluated as `road_defect`.
4. Original semantic annotations and predictions are preserved independently for dedicated YOLO-only semantic evaluation.

---

## 4. Evaluation Matching Protocol
Ground-truth annotations in RDD2022 are provided exclusively as bounding boxes. Therefore, the common quantitative comparison evaluates **bounding-box localization**.

- **Primary Matching Criterion**: Bounding-box Intersection over Union $\text{IoU} \ge 0.50$.
- **Bipartite Matching Rule**: Greedy one-to-one matching per frame. Pairwise IoU values between all predicted boxes and ground-truth boxes are computed. Candidates with $\text{IoU} \ge 0.50$ are sorted in descending order. Each prediction is matched to at most one ground-truth box, and each ground-truth box can be claimed by at most one prediction.
- **Metric Definitions**:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \cdot \text{P} \cdot \text{R}}{\text{P} + \text{R}}$$
  $$\text{Mean Matched IoU} = \frac{1}{\text{TP}} \sum_{i=1}^{\text{TP}} \text{IoU}(P_i, G_i)$$
- **Secondary Sensitivity Analysis**: Evaluated at $\text{IoU} \ge 0.25$ to quantify the impact of aspect-ratio sensitivity on thin, elongated crack geometries.

---

## 5. Hardware Profiling & Timing Protocol
All profiling was executed on the same physical device:
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM, Compute Capability 8.9).
- **Host Environment**: PyTorch 2.13.0+cu130, CUDA 12.8, Linux x86_64.
- **Timing Isolation**:
  - Model loading and memory bank initialization are timed separately and excluded from per-image inference latency.
  - Dedicated warmup iterations are executed before timing begins.
  - Strict CUDA hardware synchronization (`torch.cuda.synchronize()`) is executed immediately before and after each model's inference invocation.
  - Throughput is calculated as $\text{FPS} = 1000.0 / \text{mean\_latency\_ms}$.

---

## 6. Methodological Limitations
1. **Domain Favoritism**: The benchmark favors the supervised detector because it is evaluated within the same dataset/domain family used for its training.
2. **Semantic Asymmetry**: DINOv2+SAM2 cannot distinguish crack types or road repairs; semantic evaluation is strictly YOLO-specific.
3. **Absence of Pixel Ground Truth**: RDD2022 lacks polygon or pixel-level segmentation masks. Converting SAM2 masks to rectangular bounding boxes penalizes tight segmentation contours when evaluated against loose rectangular annotations.
4. **Validation-Set Status**: This split was visible as a development validation target during YOLO training and does not constitute an independent blind benchmark.
