# RoadSentinel — Cross-Domain Evaluation Methodology
## Independent Evaluation of Perception and Reliability Systems under Genuine Distribution Shift

---

## 1. Research Objectives & Motivation

In Phase 8, RoadSentinel developed and validated an interpretable reliability estimation module. The Phase-8 validation demonstrated that:
1. Selective prediction via reliability scoring significantly reduces inspection risk in-domain (60.5% error reduction at 80% coverage).
2. Within the homogeneous **RDD2022 China_Drone** validation domain, detector confidence features dominated failure prediction ($\text{AUROC} = 0.8649$), while DINOv2 feature-space domain distance provided only marginal in-domain failure predictive power ($\text{AUROC} = 0.5971$).

However, in real-world deployment, automated road-inspection systems encounter **severe distribution shifts**: cross-country pavement variations, ground-level vehicle dashcams, extreme perspective angles, variable weather, and unpaved asphalt boundaries. Under severe perspective shifts, object detectors often experience **silent failures** (zero bounding boxes generated, leaving softmax confidences unavailable).

### Primary Scientific Questions for Phase 10:
1. **Perception Generalization**: How severely does frozen supervised YOLOv8n degrade when evaluated zero-shot on an independent forward-facing vehicle dashcam dataset?
2. **DINOv2 Domain-Shift Sensitivity**: Does the non-parametric DINOv2 feature-space reference metric reliably detect genuine domain shift (elevated OOD scores on dashcam imagery)?
3. **Cross-Domain Reliability Value**: Does DINOv2 domain familiarity become more valuable than detector confidence for predicting YOLO failure out-of-domain?
4. **Selective Prediction Utility**: Can the frozen Phase-8 reliability model and risk-coverage rejection mechanism purify inspection quality on cross-domain data without any retraining or domain adaptation?

---

## 2. Dataset Selection & Justification

### 2.1 The "Selection-Before-Results" Protocol
In accordance with strict empirical standards, the cross-domain benchmark dataset was chosen, mapped, and frozen **prior to observing any model inference results**.

### 2.2 Selected Benchmark: RDD2022 India
The benchmark dataset is drawn from the official **RDD2022 India** split released under the Crowdsensing-based Road Damage Detection Challenge (CRDDC 2022).

| Dataset Characteristic | In-Domain Baseline (China_Drone) | Independent Cross-Domain (India) |
|---|---|---|
| **Geography** | China (Urban municipal roads) | India (National highways, state highways, rural corridors) |
| **Sensor / Platform** | UAV Drone (Nadir aerial ~25–100m) | Vehicle Windshield Dashcam / Smartphone (~1.2m) |
| **Camera Viewpoint** | Orthogonal Top-Down (Pitch $-89^\circ$) | Forward-Facing Oblique Perspective |
| **Optical Distortions** | Diffuse top-down daylight | Windshield glare, horizon vanishing lines, vehicle hood reflections |
| **Pavement Characteristics** | Structured urban asphalt | Weathered asphalt, aggregate breakup, unpaved shoulders |
| **Annotation Format** | Pascal VOC XML | Pascal VOC XML |
| **YOLO Training Overlap** | None (Validation Split) | **ZERO Overlap (100% External Cross-Domain)** |

---

## 3. Class Mapping & Benchmark Manifest Freeze

### 3.1 Damage Category Alignment
To ensure a rigorous, fair comparison across heterogeneous defect nomenclatures, all annotated road-damage categories are mapped to a unified binary target: `road_defect`.
- `D00`: Longitudinal Cracking $\rightarrow$ `road_defect`
- `D10`: Transverse Cracking $\rightarrow$ `road_defect`
- `D20`: Alligator Fatigue Cracking $\rightarrow$ `road_defect`
- `D40`: Pothole / Deep Asphalt Cavity $\rightarrow$ `road_defect`
- `Repair`: Pre-existing Maintenance Patch $\rightarrow$ `road_defect`

### 3.2 Deterministic Sample Freeze
- A deterministic subset of $N=300$ non-empty annotated images was sorted and frozen in [`cross_domain/benchmark_manifest.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/cross_domain/benchmark_manifest.csv) and [`cross_domain/ground_truth.json`](file:///home/nitin-nandakumar/Downloads/roadsentinel/cross_domain/ground_truth.json) (totaling 652 ground-truth bounding boxes).

---

## 4. Frozen Perception Systems & Evaluation Protocol

### 4.1 System A: Frozen YOLOv8n
- **Checkpoint**: `yolo/weights/best.pt` (Trained on 1,921 China_Drone images).
- **Inference Config**: Image size $512 \times 512$, confidence threshold $\text{conf} = 0.25$.
- **Matching Criterion**: Greedy one-to-one bounding box $\text{IoU} \ge 0.50$ (primary) and $\text{IoU} \ge 0.25$ (secondary).

### 4.2 System B: Frozen DINOv2 + SAM2 Pipeline
- **Backbone & Segmenter**: `dinov2_vits14` + `sam2.1_hiera_small` (Marion Day-5 NORMAL production configuration).
- **Inference Config**: Forward camera mode, patch anomaly thresholding, heuristic bounding box prompting.
- **Matching Criterion**: Mask-derived bounding boxes evaluated at $\text{IoU} \ge 0.50$ (and $\text{IoU} \ge 0.25$).

---

## 5. DINOv2 Training-Domain Reference & OOD Estimation

### 5.1 Training-Domain Reference Freeze
The domain reference representation remains strictly frozen from Phase 8:
- Built from normalized 384-dimensional DINOv2 CLS tokens extracted across the 1,921 China_Drone training images.
- Metric: $k$-Nearest Neighbors cosine distance ($k=20$).
- Normalization Scale: Calibrated against the 99th percentile of in-domain training distances ($d_{p99} = 0.4491$):
$$\text{OOD Score} = \min\left(1.0, \frac{d_{k\text{NN}}}{0.4491}\right), \quad \text{Familiarity} = 1.0 - \text{OOD Score}$$

---

## 6. Frozen Reliability Modeling & Selective Prediction

### 6.1 Reliability Target Definition
Preserving the exact Phase-8 deterministic target:
- $\text{YOLO\_SUCCESS} = 1$ if image-level YOLO $\text{F1} > 0$ under $\text{IoU} \ge 0.50$.
- $\text{YOLO\_FAILURE} = 1$ if image-level YOLO $\text{F1} == 0$ (misses all defects or generates complete false alarms on positive images).

### 6.2 Zero-Shot Reliability Evaluation
The Phase-8 Logistic Regression classifier (trained on China_Drone in-domain validation data) is applied directly to cross-domain feature vectors without retraining:
- **Model A**: YOLO Confidence Only (`max_conf`, `mean_conf`, `std_conf`, `pred_count`, `num_low_conf`, `num_high_conf`)
- **Model B**: DINOv2 OOD Only (`dino_ood_score`, `dino_knn_dist`, `dino_centroid_dist`)
- **Model C**: Combined Model (Confidence + DINOv2 OOD + Image Contrast/Sharpness/Brightness)

### 6.3 Frozen Operational Reliability Bands
- **HIGH Band**: Predicted Reliability $P \ge 0.85$ $\rightarrow$ Action: `AUTOMATED_ACCEPT`
- **MEDIUM Band**: Predicted Reliability $0.60 \le P < 0.85$ $\rightarrow$ Action: `SECONDARY_INSPECTION`
- **LOW Band**: Predicted Reliability $P < 0.60$ $\rightarrow$ Action: `ESCALATE_MANUAL_REVIEW`

---

## 7. Statistical Hypotheses & Verification

1. **Distribution Shift Hypothesis**: Cross-domain India dashcam imagery exhibits statistically higher DINOv2 OOD scores than in-domain China_Drone aerial imagery (Mann-Whitney $U$ test, $p < 0.001$, Cohen's $d > 1.0$).
2. **Selective Risk Reduction**: Rejecting low-reliability predictions purifies cross-domain inspection outputs, lowering accepted error rates across coverage thresholds ($100\% \rightarrow 90\% \rightarrow 80\% \rightarrow 60\%$).
