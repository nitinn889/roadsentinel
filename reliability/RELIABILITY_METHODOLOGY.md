# RoadSentinel Reliability & Out-of-Distribution (OOD) Methodology

## 1. Problem Formulation & Research Objective
Supervised object detectors such as YOLOv8n achieve strong in-domain empirical performance on road-damage datasets (F1 = 0.7104 on RDD2022 China_Drone). However, in automated infrastructure inspection, silent detector failures (false negatives on critical distresses or spurious false positives on benign pavement) carry high operational risk.

Phase 8 investigates the **Reliability-Aware Perception Framework**:
$$\hat{R}(x) = P(\text{YOLO prediction is trustworthy} \mid x, \hat{y})$$
Where $x$ denotes the input road frame, $\hat{y}$ denotes the raw YOLO detection vector, and $\hat{R}(x) \in [0, 1]$ represents the calibrated reliability score.

---

## 2. Deterministic Target Ground Truth: What is "Unreliable"?
To prevent vague or post-hoc target definitions, reliability is formalized at the image level on the 480-image validation split:

- **Primary Target ($\text{YOLO\_FAILURE} \in \{0, 1\}$)**:
  $$\text{YOLO\_FAILURE} = \begin{cases} 1 & \text{if } \text{F1}_{\text{image}} = 0.0 \quad (\text{zero true positives localized under IoU} \ge 0.50) \\ 0 & \text{if } \text{F1}_{\text{image}} > 0.0 \quad (\text{at least one valid defect localized}) \end{cases}$$
  - Success Count ($Y=0$): **423 images (88.13%)**
  - Failure Count ($Y=1$): **57 images (11.87%)**

- **Secondary Strict Target ($\text{YOLO\_STRICT\_FAILURE} \in \{0, 1\}$)**:
  $$\text{YOLO\_STRICT\_FAILURE} = \begin{cases} 1 & \text{if } \text{F1}_{\text{image}} < 0.50 \\ 0 & \text{if } \text{F1}_{\text{image}} \ge 0.50 \end{cases}$$
  - Failure Count ($Y=1$): **86 images (17.92%)**
  - Success Count ($Y=0$): **394 images (82.08%)**

---

## 3. DINOv2 Training-Domain Reference Construction
To prevent data contamination, the domain reference is constructed exclusively from the **1,921 YOLO training images** (`yolo/data/rdd2022/images/train/`), with zero exposure to validation labels.

### 3.1 Global Feature Representation
- **Backbone**: Meta DINOv2 Small (`dinov2_vits14`, 384-dimensional embedding, $37 \times 37$ patch tokens).
- **Input Transformation**: Bicubic resize to $518 \times 518$, PyTorch ImageNet normalization.
- **Representation**: L2-normalized global CLS token vector $\mathbf{z} \in \mathbb{R}^{384}, \|\mathbf{z}\|_2 = 1$.

### 3.2 Non-Parametric Reference Statistics
For the 1,921 training embeddings $\mathcal{Z}_{\text{train}} = \{\mathbf{z}_1, \dots, \mathbf{z}_N\}$:
- **Centroid**: $\bar{\mathbf{c}} = \frac{1}{N} \sum_{i=1}^N \mathbf{z}_i, \quad \mathbf{c}_{\text{ref}} = \frac{\bar{\mathbf{c}}}{\|\bar{\mathbf{c}}\|_2}$.
- **$k$-Nearest Neighbor Distance ($k=20$)**:
  $$d_{\text{knn}}(\mathbf{z}) = 1 - \frac{1}{k} \sum_{j \in \mathcal{N}_k(\mathbf{z})} \mathbf{z}^\top \mathbf{z}_j$$
- **Reference Calibration Scale**:
  - Training leave-one-out $d_{\text{knn}}$ distribution: $\text{Min} = 0.0336$, $\text{Median} = 0.1730$, $p_{95} = 0.3482$, $p_{99} = 0.4491$, $\text{Max} = 0.6025$.
- **Normalized Domain Familiarity Score**:
  $$\text{dino\_domain\_score}(\mathbf{z}) = \text{clip}\left(1.0 - \frac{d_{\text{knn}}(\mathbf{z}) - d_{\text{min}}}{d_{p99} - d_{\text{min}}}, 0.0, 1.0\right)$$
  $$\text{dino\_ood\_score}(\mathbf{z}) = 1.0 - \text{dino\_domain\_score}(\mathbf{z})$$

---

## 4. Multi-Modal Reliability Feature Schema
The complete 480-sample dataset (`reliability/data/reliability_dataset.csv`) combines three distinct feature modalities:

1. **YOLO Confidence Modality (Supervised Internal Signal)**:
   - `max_confidence`: Maximum detection confidence per frame ($0.0$ if no detections).
   - `mean_confidence`: Arithmetic mean of prediction confidences.
   - `std_confidence`: Standard deviation across candidate boxes.
   - `yolo_pred_count`: Total predicted bounding boxes (conf $\ge 0.25$).
   - `num_low_conf`: Count of borderline predictions with $\text{conf} < 0.40$.
   - `num_high_conf`: Count of high-confidence predictions with $\text{conf} \ge 0.60$.

2. **DINOv2 Domain Modality (Foundation Feature Space)**:
   - `dino_ood_score`: Calibrated out-of-distribution distance $\in [0, 1]$.
   - `dino_knn_distance`: Raw mean cosine distance to 20 nearest training exemplars.
   - `dino_centroid_distance`: Cosine distance to training domain centroid.

3. **Visual Condition Modality (Deterministic Classical Image Processing)**:
   - `brightness`: Mean grayscale pixel luminance $\mu_I$.
   - `contrast`: Standard deviation of pixel intensities $\sigma_I$.
   - `sharpness`: Variance of Laplacian $\text{Var}(\nabla^2 I)$ (blur indicator).
   - `edge_density`: Fraction of high-gradient Canny edge pixels.

---

## 5. Evaluation Protocol & Model Calibration

### 5.1 Leak-Free 5-Fold Stratified Cross-Validation
To guarantee zero evaluation leakage across the 480 benchmark frames:
- 5 stratified folds partitioned on the binary failure target (seed = 42).
- Feature standardizers ($\mu, \sigma$) are fit strictly on training fold partitions and applied out-of-fold.
- Calibrated probability estimates $P(\text{failure})$ are collected out-of-fold across all 480 samples.

### 5.2 Model Architectures Evaluated
1. **Model A (YOLO Confidence Baseline)**: Logistic regression with L2 regularization on the 6 confidence features.
2. **Model B (DINOv2 OOD Baseline)**: Logistic regression on DINOv2 domain distance features.
3. **Model C (Combined Reliability Model)**: Multimodal regularized classifier combining confidence, foundation domain familiarity, and visual condition features.

### 5.3 Metric Definitions
- **AUROC**: Area under the Receiver Operating Characteristic curve.
- **AUPRC**: Area under the Precision-Recall curve (critical for class-imbalanced failure detection; baseline = $0.1188$).
- **Brier Score**: Mean squared error between predicted probabilities and binary outcomes $\frac{1}{N} \sum (\hat{p}_i - y_i)^2$.
- **Expected Calibration Error (ECE)**: Weighted difference between confidence and empirical accuracy across 10 uniform probability bins.

---

## 6. Risk-Coverage Selective Prediction Formulation
In operational deployment, road inspection surveys can reject low-reliability predictions and escalate them to human engineers or secondary inspection passes.

Given a reliability threshold $T$, the accepted sample subset is:
$$\mathcal{A}(T) = \{i \mid \hat{R}(x_i) \ge T\}, \quad \text{Coverage}(T) = \frac{|\mathcal{A}(T)|}{N}$$
$$\text{Accepted Error Rate}(T) = \frac{1}{|\mathcal{A}(T)|} \sum_{i \in \mathcal{A}(T)} \text{YOLO\_FAILURE}_i$$
$$\text{Mean Accepted F1}(T) = \frac{1}{|\mathcal{A}(T)|} \sum_{i \in \mathcal{A}(T)} \text{F1}_{\text{image}, i}$$
The risk-coverage curve profiles how the accepted error rate monotonically drops as the coverage threshold is tightened.
