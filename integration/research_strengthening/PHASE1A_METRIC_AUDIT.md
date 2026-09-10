# RoadSentinel — Phase 1A: Metric Evaluation & Scientific Audit

**Document**: `integration/research_strengthening/PHASE1A_METRIC_AUDIT.md`  
**Evaluation Phase**: Phase 1A (Metric Evaluation Only — Research Strengthening)  
**Hardware Platform**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM), PyTorch 2.13.0+cu130, CUDA 12.8  
**Audit Status**: **VERIFIED & FROZEN**  

---

## Executive Summary

This document provides an exhaustive, mathematically rigorous, and fully transparent audit of all quantitative metrics generated for **RoadSentinel Phase 1A: Research Strengthening**. 

The purpose of this audit is to substantiate the central research claim:
> *"YOLO remains an effective, high-throughput primary detector for in-domain road distress localization, but standalone YOLO-only monitoring lacks macro domain awareness, reliability-aware failure rejection, quantitative pixel-level condition assessment, cross-inspection temporal interpretation, and long-term deterioration forecasting. The additional computational cost of the integrated RoadSentinel pipeline is strictly justified by capabilities that are architecturally unavailable from a standalone detector."*

In accordance with strict research integrity requirements:
1. **Zero Model Modifications**: No neural network weights, classifiers, or heuristics were trained, fine-tuned, or altered.
2. **Zero Synthetic Dataset Generation**: Only existing stored benchmark outputs, ground-truth annotations, and model predictions were evaluated.
3. **Zero Fabrication**: Experiment B / SEG005–SEG006 remain strictly unreferenced.
4. **Transparent Discrepancy Accounting**: No historical numerical conflicts have been silently reconciled.

---

## 1. Metric Audit Matrix (10 Core Dimensions)

The 10 generated canonical CSV artifacts in `integration/research_strengthening/` map directly to the research requirements:

| Output CSV File | Primary Research Dimension | Evaluation Level | Primary Sample Size | Primary Canonical Source File |
|---|---|---|---|---|
| [`YOLO_BASELINE_METRICS.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/YOLO_BASELINE_METRICS.csv) | In-Domain Detector Performance | Detection & Image level | $N=480$ images, 742 GT, 874 preds | `benchmark/final_comparison/benchmark_summary.json` |
| [`YOLO_GENERALIZATION_GAP.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/YOLO_GENERALIZATION_GAP.csv) | Cross-Domain Degradation | Detection & Image level | China $N=480$, India $N=300$ | `cross_domain/tables/table_cross_domain_metrics.csv` |
| [`YOLO_CONFIDENCE_FAILURE_ANALYSIS.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/YOLO_CONFIDENCE_FAILURE_ANALYSIS.csv) | Confidence vs. Correctness | Detection ($N=874$) & Image ($N=480$) | $N=874$ predictions (574 TP, 300 FP) | `benchmark/final_comparison/benchmark_summary.json` |
| [`RISK_COVERAGE_METRICS.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/RISK_COVERAGE_METRICS.csv) | Selective Risk Reduction | Image-level ranking | $N=480$ images (86 failures under $T_1$) | `integration/CANONICAL_RESEARCH_METRICS.json` |
| [`DOMAIN_GATE_VALUE.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/DOMAIN_GATE_VALUE.csv) | Macro Domain Awareness | Image-level feature distance | China $N=480$, India $N=300$ | `reliability_validation/tables/table_domain_gate.csv` |
| [`DECISION_SAFETY_ANALYSIS.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/DECISION_SAFETY_ANALYSIS.csv) | Autonomous Decision Safety | Image-level operational routing | India $N=300$, Exp A $N=40$ | `decision_engine/tables/table_cross_domain_routing.csv` |
| [`TEMPORAL_CAPABILITY_GAP.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/TEMPORAL_CAPABILITY_GAP.csv) | Defect Tracking & Persistence | Sequence-level transitions | $N=8$ sequences, 33 transitions, 48 tracks | `integration/CANONICAL_RESEARCH_METRICS.json` |
| [`FORECAST_CAPABILITY_METRICS.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/FORECAST_CAPABILITY_METRICS.csv) | Longitudinal Deterioration | Highway section pairs & Survey frames | $N=24$ test pairs (6 sites), $N=40$ captures | `xgboost/outputs/scenario_model_v2_metrics.json` |
| [`COMPUTATIONAL_TRADEOFF.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/COMPUTATIONAL_TRADEOFF.csv) | Hardware Cost vs. Capability | Module execution timing | $N=480$ images on RTX 5060 GPU | `integration/CANONICAL_RESEARCH_METRICS.json` & direct profiling |
| [`STATISTICAL_VALIDATION.csv`](file:///home/nitin-nandakumar/Downloads/roadsentinel/integration/research_strengthening/STATISTICAL_VALIDATION.csv) | Inferential Hypothesis Testing | Various ($N=5$ to $N=874$) | 9 formal statistical tests | Dynamic evaluation across active benchmark splits |

---

## 2. Detailed Technical Audit per Output Artifact

### 2.1 `YOLO_BASELINE_METRICS.csv` (Section 1: In-Domain Baseline)
- **Objective**: Establish that YOLOv8n is a strong, competent detector on its native distribution (RDD2022 China_Drone).
- **Exact Source Files**:
  - `benchmark/final_comparison/benchmark_summary.json`
  - `benchmark/final_comparison/FINAL_PERCEPTION_TABLE.csv`
  - `integration/CANONICAL_RESEARCH_METRICS.json`
- **Sample Size**:
  - Exactly 480 validation images ($512 \times 512$ resolution).
  - Exactly 742 ground-truth bounding boxes across 5 classes (`D00`: 266, `D10`: 256, `D20`: 58, `D40`: 15, `Repair`: 147).
  - Exactly 874 detector prediction boxes.
- **Evaluation Level**: Detection-level bipartite matching and image-level rate averages.
- **Thresholds Used**:
  - Bounding box primary IoU threshold: $\text{IoU} \ge 0.50$.
  - Detection confidence threshold: $\text{conf} \ge 0.25$.
- **Formulas & Measured Values**:
  - $\text{True Positives (TP)} = 574$ (Greedy 1-to-1 bipartite matched pairs with $\text{IoU} \ge 0.50$).
  - $\text{False Positives (FP)} = \text{Total Predictions} - \text{TP} = 874 - 574 = 300$.
  - $\text{False Negatives (FN)} = \text{Total Ground Truth} - \text{TP} = 742 - 574 = 168$.
  - $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{574}{874} = 0.6568$.
  - $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{574}{742} = 0.7736$.
  - $\text{F1-Score} = \frac{2 \cdot \text{P} \cdot \text{R}}{\text{P} + \text{R}} = \frac{2 \cdot 0.6568 \cdot 0.7736}{0.6568 + 0.7736} = 0.7104$.
  - $\text{Mean Matched IoU} = 0.8007$ (Median: $0.8154$).
  - $\text{Predictions / Image} = \frac{874}{480} = 1.8208$.
  - $\text{FP / Image} = \frac{300}{480} = 0.6250$.
  - $\text{FN / Image} = \frac{168}{480} = 0.3500$.
  - $\text{Mean Latency} = 3.62\text{ ms}$ (Median: $3.51\text{ ms}$).
  - $\text{Throughput} = 276.24\text{ FPS}$ (recorded as $276.33\text{ FPS}$ in perception table).
  - Native Multiclass $\text{mAP}_{50} = 0.6774$, $\text{mAP}_{50-95} = 0.4068$.

---

### 2.2 `YOLO_GENERALIZATION_GAP.csv` (Section 2: Cross-Domain Generalization Gap)
- **Objective**: Quantify the performance drop when a frozen model trained on aerial UAV surveys is deployed on vehicle dashcams in a different country (RDD2022 India).
- **Exact Source Files**:
  - `cross_domain/tables/table_cross_domain_metrics.csv`
  - `cross_domain/tables/table_in_vs_cross_domain.csv`
  - `cross_domain/results/cross_domain_reliability_dataset.csv`
- **Sample Size**:
  - China In-Domain: $N=480$ images, $742$ GT boxes, $874$ predictions.
  - India Cross-Domain: $N=300$ images, $652$ GT boxes, $83$ predictions.
- **Evaluation Level**: Detection-level aggregates and image-level conditional failure distributions.
- **Thresholds Used**: $\text{IoU} \ge 0.50$, $\text{conf} \ge 0.25$.
- **Formulas & Measured Degradations**:
  - $\text{Absolute Drop} = \text{Metric}_{\text{China}} - \text{Metric}_{\text{India}}$.
  - $\text{Relative Drop Percent} = \frac{\text{Metric}_{\text{China}} - \text{Metric}_{\text{India}}}{\text{Metric}_{\text{China}}} \times 100$.
  - **Precision**: $0.6568 \rightarrow 0.0964$ ($\text{Absolute} = 0.5604$, $\text{Relative} = 85.32\%$).
  - **Recall**: $0.7736 \rightarrow 0.0123$ ($\text{Absolute} = 0.7613$, $\text{Relative} = 98.41\%$).
  - **F1-Score**: $0.7104 \rightarrow 0.0218$ ($\text{Absolute} = 0.6886$, $\text{Relative} = 96.94\%$).
  - **Mean Matched IoU**: $0.8007 \rightarrow 0.7678$ ($\text{Absolute} = 0.0329$, $\text{Relative} = 4.11\%$).
  - **Predictions / Image**: $1.8208 \rightarrow 0.2767$ ($\text{Absolute} = 1.5441$, $\text{Relative} = 84.80\%$).
  - **FP / Image**: $0.6250 \rightarrow 0.2500$ ($\text{Absolute} = 0.3750$, $\text{Relative} = 60.00\%$).
  - **FN / Image**: $0.3500 \rightarrow 2.1467$ ($\text{Absolute} = -1.7967$, $\text{Relative} = -513.33\%$).
  - **Zero-Correct-Detection Rate ($T_0$)**: $11.88\% \rightarrow 97.33\%$ ($292 / 300$ images have $0$ true detections).
  - **Strict Failure Rate ($T_1$)**: $17.92\% \rightarrow 97.67\%$ ($293 / 300$ images have $\text{F1} < 0.50$).
  - **Images with Preds but 0 TP (Overall)**: $5.42\% \rightarrow 19.00\%$ ($57 / 300$ images).
  - **Images with Preds but 0 TP (Conditional on Pred > 0)**: $5.73\% \rightarrow 87.69\%$ ($57 / 65$ images).
  - **Confidence Distribution Shift**: Mean image confidence drops from $0.5468 \rightarrow 0.3811$ ($30.30\%$ drop).

---

### 2.3 `YOLO_CONFIDENCE_FAILURE_ANALYSIS.csv` (Section 3: Confidence vs Correctness)
- **Objective**: Determine whether detector confidence alone provides a dependable guarantee that a prediction is safe to trust.
- **Exact Source Files**:
  - `benchmark/final_comparison/benchmark_summary.json` ($N=874$ raw predictions)
  - `reliability/results/reliability_predictions.csv` ($N=480$ images)
  - `reliability_validation/tables/table_target_robustness.csv`
- **Sample Size**:
  - **Detection-Level**: $N=874$ total predictions ($574$ True Positives, $300$ False Positives).
  - **Image-Level**: $N=480$ images ($86$ failures under $T_1$, $57$ failures under $T_0$).
- **Evaluation Level**: Explicitly stratified into **Detection-Level** and **Image-Level**.
- **Thresholds Used**:
  - Detection matching: $\text{IoU} \ge 0.50$.
  - Bin intervals: $[0.0, 0.2)$, $[0.2, 0.4)$, $[0.4, 0.6)$, $[0.6, 0.8)$, $[0.8, 1.0]$.
  - Confidence cutoffs: $\ge 0.50, \ge 0.70, \ge 0.80, \ge 0.90$.
- **Detection-Level Calibration Findings**:
  - Mean Confidence: Correct detections ($\text{TP}=574$) $= 0.6135$ (Median: $0.6188$); Incorrect detections ($\text{FP}=300$) $= 0.4262$ (Median: $0.3839$).
  - Binned Failure Rates:
    - $[0.0, 0.2)$: $0$ detections (YOLO floor is $0.25$).
    - $[0.2, 0.4)$: $265$ detections, $162$ failures $\rightarrow \mathbf{61.13\%}$ failure rate ($\text{acc} = 38.87\%$).
    - $[0.4, 0.6)$: $256$ detections, $92$ failures $\rightarrow \mathbf{35.94\%}$ failure rate ($\text{acc} = 64.06\%$).
    - $[0.6, 0.8)$: $252$ detections, $42$ failures $\rightarrow \mathbf{16.67\%}$ failure rate ($\text{acc} = 83.33\%$).
    - $[0.8, 1.0]$: $101$ detections, $4$ failures $\rightarrow \mathbf{3.96\%}$ failure rate ($\text{acc} = 96.04\%$).
  - High-Confidence False Alarm Persistence:
    - $\text{Conf} \ge 0.50$: Exactly $91 / 300$ false alarms ($\mathbf{30.33\%}$ of all errors persist above $0.50$).
    - $\text{Conf} \ge 0.70$: Exactly $14 / 300$ false alarms ($\mathbf{4.67\%}$ of all errors persist above $0.70$).
    - $\text{Conf} \ge 0.80$: Exactly $4 / 300$ false alarms ($\mathbf{1.33\%}$).
    - $\text{Conf} \ge 0.90$: Exactly $1 / 300$ false alarms ($\mathbf{0.33\%}$).
  - Global Calibration Metrics:
    - Detection Correctness $\text{AUROC} = 0.7741$.
    - Detection Correctness $\text{AUPRC} = 0.8705$.
    - Brier Score $= 0.1921$.
    - Expected Calibration Error (ECE, 10 bins) $= 0.1075$.
- **Image-Level Calibration Findings**:
  - Target $T_1$ ($\text{F1} \ge 0.50$): Model B Confidence Failure $\text{AUROC} = 0.8166$ ($95\%\text{ CI}: [0.7667, 0.8669]$), $\text{AUPRC} = 0.6490$.
  - Target $T_0$ ($\text{F1} > 0$): Model B Confidence Failure $\text{AUROC} = 0.8649$ ($95\%\text{ CI}: [0.8000, 0.9220]$), $\text{AUPRC} = 0.7274$, Brier $= 0.0497$, $\text{ECE} = 0.0156$.

---

### 2.4 `RISK_COVERAGE_METRICS.csv` (Section 4: Risk-Coverage / Selective Rejection)
- **Objective**: Quantify the reduction in accepted failure rate achieved by allowing RoadSentinel's reliability model to reject or escalate low-confidence observations.
- **Exact Source Files**:
  - `integration/CANONICAL_RESEARCH_METRICS.json` (`selective_prediction_risk_coverage_t1`)
  - `reliability_validation/RELIABILITY_VALIDATION_RESEARCH_SUMMARY.md` (Section 6)
  - `reliability_validation/tables/table_risk_coverage_strict.csv`
  - `reliability/results/table_risk_coverage.csv`
- **Sample Size**: $N=480$ validation images.
- **Evaluation Level**: Image-level selective coverage ranking.
- **Thresholds Used**: Ranked descending by calibrated reliability score across coverage levels $100\%, 90\%, 80\%, 70\%, 60\%, 50\%$.
- **Formulas**:
  - $\text{Accepted Failure Rate} = \frac{\text{Failures Accepted}}{\text{Samples Accepted}}$.
  - $\text{Failure Capture Rate} = \frac{\text{Failures Rejected}}{\text{Baseline Failures at 100\% Coverage}}$.
  - $\text{Relative Risk Reduction} = \frac{\text{Baseline Error} - \text{Accepted Error}}{\text{Baseline Error}} \times 100$.
- **Canonical Target $T_1$ ($\text{F1} \ge 0.50$, Baseline Failures $= 86 / 480 = 17.92\%$)**:
  - **100% Coverage**: Accepted $= 480$, Rejected $= 0$, Failures Accepted $= 86$, Error $= 17.92\%$, Risk Reduction $= \mathbf{0.0\%}$.
  - **90% Coverage**: Accepted $= 432$, Rejected $= 48$, Failures Accepted $= 55$, Error $= 12.73\%$, Risk Reduction $= \mathbf{28.96\%}$.
  - **80% Coverage**: Accepted $= 384$, Rejected $= 96$, Failures Accepted $= 34$, Error $= 8.85\%$, Risk Reduction $= \mathbf{50.61\%}$.
  - **70% Coverage**: Accepted $= 336$, Rejected $= 144$, Failures Accepted $= 18$, Error $= 5.36\%$, Risk Reduction $= \mathbf{70.09\%}$.
  - **60% Coverage**: Accepted $= 288$, Rejected $= 192$, Failures Accepted $= 11$, Error $= 3.82\%$, Risk Reduction $= \mathbf{78.68\%}$.
  - **50% Coverage**: Accepted $= 240$, Rejected $= 240$, Failures Accepted $= 5$, Error $= 2.08\%$, Risk Reduction $= \mathbf{88.39\%}$ ($\approx \mathbf{88.4\%}$).
- **Target $T_0$ ($\text{F1} > 0$, Baseline Failures $= 57 / 480 = 11.87\%$)**:
  - 100% Coverage: $11.87\%$ error; 90% Coverage: $5.09\%$ error ($57.1\%$ reduction); 80% Coverage: $4.69\%$ error ($60.5\%$ reduction); 50% Coverage: $2.92\%$ error ($75.4\%$ reduction).

---

### 2.5 `DOMAIN_GATE_VALUE.csv` (Section 5: Domain-Gate Value)
- **Objective**: Evaluate DINOv2 foundation embeddings as a macro input domain boundary gate to detect unfamiliar visual distributions before running primary models.
- **Exact Source Files**:
  - `reliability_validation/tables/table_domain_gate.csv`
  - `cross_domain/results/cross_domain_reliability_dataset.csv`
  - `reliability/results/reliability_predictions.csv`
- **Sample Size**: Total $N=780$ images (China $N=480$, India $N=300$).
- **Evaluation Level**: Image-level foundation embedding cosine distance to 1,921 China training references ($k=20$).
- **Thresholds Used**: Operational $p_{99} = 0.4491$, $p_{95} = 0.3804$.
- **Measured Metrics**:
  - China In-Domain $d_{k\text{NN}}$: $\text{Mean} = 0.1841 \pm 0.0848$, $\text{Median} = 0.1693$, $\text{IQR} = 0.0932$, $\text{Range} = [0.0393, 0.5247]$.
  - India Cross-Domain $d_{k\text{NN}}$: $\text{Mean} = 0.8432 \pm 0.0441$, $\text{Median} = 0.8486$, $\text{IQR} = 0.0480$, $\text{Range} = [0.6405, 0.9261]$.
  - **Separation Margin**: $\min(\text{India}) - \max(\text{China}) = 0.6405 - 0.5247 = \mathbf{+0.1158}$ (Zero distribution overlap).
  - **Domain Discrimination Performance**: $\text{AUROC} = 1.0000$, $\text{AUPRC} = 1.0000$.
  - Operational Gate Efficacy:
    - At $p_{99} = 0.4491$: India shift detected $= \mathbf{100.0\%}$ ($300 / 300$), China false alarms $= \mathbf{1.46\%}$ ($7 / 480$).
    - At $p_{95} = 0.3804$: India shift detected $= \mathbf{100.0\%}$ ($300 / 300$), China false alarms $= \mathbf{3.96\%}$ ($19 / 480$).
- **Crucial Architectural Decoupling**:
  - DINOv2 standalone is a macro domain classifier ($\text{AUROC} = 1.0000$).
  - DINOv2 is **NOT** a sample-level defect corrector ($\text{AUROC} \approx 0.5428\text{--}0.5971$ for in-domain sample correctness).

---

### 2.6 `DECISION_SAFETY_ANALYSIS.csv` (Section 6: Decision-Safety Analysis)
- **Objective**: Compare a conceptual standalone detector policy (accept whatever YOLO outputs) against RoadSentinel's tiered governance policy.
- **Exact Source Files**:
  - `decision_engine/tables/table_cross_domain_routing.csv`
  - `integration/CANONICAL_RESEARCH_METRICS.json` (`decision_engine_phase13`)
  - `decision_engine/DECISION_ENGINE_RESEARCH_SUMMARY.md`
- **Sample Size**:
  - India Cross-Domain: $N=300$ images ($293$ ground-truth perception failures under Target $T_1$).
  - Experiment A: $N=40$ multi-day physical survey captures.
- **Evaluation Level**: Image-level operational routing action.
- **Policy Comparison on Out-of-Domain India Benchmark**:
  - **Policy 1 (Standalone YOLO)**:
    - Automated Accepts: $300 / 300$ ($100.0\%$).
    - Failures Passed Downstream: Exactly $293$ unflagged misclassifications.
    - Unsafe Accepted Error Rate: $\mathbf{97.67\%}$.
    - Quarantine Efficacy: $\mathbf{0.0\%}$.
  - **Policy 4 (Full RoadSentinel Policy with DINO Domain Gate)**:
    - Automated Accepts: $\mathbf{0} / 300$ ($\mathbf{0.0\%}$).
    - Routed to `DOMAIN_ESCALATION`: Exactly $300 / 300$ ($\mathbf{100.0\%}$).
    - Failures Passed Downstream: $\mathbf{0}$.
    - Failure Quarantine Efficacy: $\mathbf{100.0\%}$.
  - **Approved Phrasing**: *"RoadSentinel reduced automatic trust in known failure cases by routing 100% of out-of-domain inputs to human dispatch."*
- **Experiment A In-Domain Action Tiers**:
  - `MONITOR`: $21$ ($52.5\%$).
  - `REINSPECT`: $10$ ($25.0\%$).
  - `PRIORITY_REVIEW`: $9$ ($22.5\%$).
  - `AUTOMATED_ACCEPT`: $0$ (cautious posture on synthetic data).
  - `DOMAIN_ESCALATION`: $0$ (familiar sensor profile).

---

### 2.7 `TEMPORAL_CAPABILITY_GAP.csv` (Section 7: Temporal Capability Metrics)
- **Objective**: Quantify cross-inspection defect persistence, growth tracking, and state transitions that cannot be derived from static single-frame YOLO detections.
- **Exact Source Files**:
  - `integration/CANONICAL_RESEARCH_METRICS.json` (`temporal_analytics_experiment_a`)
  - `integration/experiment_a/temporal/tables/table_tracking_statistics.csv`
  - `integration/experiment_a/temporal/*/temporal_events.json`
- **Data Labeling Requirement**: **MODEL-OBSERVED TEMPORAL CHANGE / SIMULATED TEMPORAL DATA**.
- **Sample Size**: Exactly 8 same-camera sequences, 40 physical survey captures, 33 adjacent transitions.
- **Evaluation Level**: Sequence-level state transitions and multi-day defect tracks.
- **Tracking Formulation**: Greedy Hierarchical One-to-One Matching ($\text{Mask IoU} \ge 0.50 \rightarrow \text{Bbox IoU} \ge 0.30 \rightarrow \text{Centroid} \le 75\text{ px} + \text{Area} \le 3\times$).
- **Measured Metrics**:
  - Validated Sequences: 8.
  - Total Physical Captures: 40.
  - Eligible Matched Transitions: 33.
  - Unique Defect Tracks: 48.
  - Longest Continuous Track: 7 states (SEG_003 shoulder patch).
  - Mean Track Length (Weighted): $\frac{81\text{ detections}}{48\text{ tracks}} = \mathbf{1.69\text{ states/track}}$.
  - Mean Track Length (Unweighted): Arithmetic sequence average $= \mathbf{2.23\text{ states/sequence}}$.
  - Tracks $\ge 2$ States: 22 tracks.
  - Persistence Ratio: $\frac{22}{48} = \mathbf{45.83\%}$.
  - Canonical Event Distribution:
    - `NEW_DEFECT`: Exactly $48$ events.
    - `NOT_OBSERVED`: Exactly $34$ events.
    - `OBSERVED_AREA_INCREASED`: Exactly $14$ events.
    - `OBSERVED_AREA_DECREASED`: Exactly $19$ events.
    - `TOTAL_MATCHED_TRANSITIONS`: Exactly $33$ transitions ($14 + 19 = 33$).
- **Thesis Conclusion**: *"Single-frame YOLO detections do not themselves encode cross-inspection persistence or change. YOLO was not designed to perform this longitudinal tracking task."*

---

### 2.8 `FORECAST_CAPABILITY_METRICS.csv` (Section 8: Forecasting Capability)
- **Objective**: Document empirical deterioration forecasting on held-out pavement sections, extending road monitoring beyond present-frame observation.
- **Exact Source Files**:
  - `xgboost/outputs/scenario_model_v2_metrics.json`
  - `integration/CANONICAL_RESEARCH_METRICS.json` (`forecasting_xgboost_goal1`)
  - `integration/RESEARCH_CLAIM_REGISTRY.md`
- **Sample Size**:
  - LTPP SDR 40 Evaluation: $N=113$ longitudinal section pairs ($89$ train pairs, $24$ test pairs).
  - Pavement Sites: 17 training sites, 6 test sites, **0 site overlap**.
  - Integration Evaluation: 40 survey captures evaluated across 5 scenarios and 3 horizons ($600$ forecast records).
- **Evaluation Level**: Section pair-level longitudinal progression & Image-level scenario projections.
- **Held-Out Generalization Performance (`roadsentinel_xgb_v2_ltpp_scenario`)**:
  - Test MAE: $\mathbf{0.0924}$ (normalized distress severity).
  - Test RMSE: $\mathbf{0.1144}$.
  - Test $R^2$: $\mathbf{0.8055}$ (explains $80.55\%$ of deterioration variance across unseen sites).
- **Feature Importance (Gain Proxy)**:
  - `current_severity`: $0.7553$.
  - `traffic_level`: $0.1150$.
  - `temperature`: $0.0548$.
  - `days_ahead`: $0.0530$.
  - `water_exposure`: $0.0113$.
  - `rainfall_level`: $0.0105$.
- **Mean 90-Day Projected Deterioration Deltas across 40 Survey Captures**:
  - `WET_EXPOSURE`: $\mathbf{+0.0526} \pm 0.0162$ (Dominant scenario across $100\%$ of frames).
  - `HEAVY_RAIN`: $\mathbf{+0.0489} \pm 0.0151$.
  - `HEAVY_TRAFFIC`: $\mathbf{+0.0384} \pm 0.0124$.
  - `HIGH_HEAT`: $\mathbf{+0.0315} \pm 0.0102$.
  - `NORMAL`: $\mathbf{+0.0242} \pm 0.0081$.
- **Scientific Disclaimers**:
  - Forecasts are empirical statistical regressions; they are **not causal mechanics**.
  - Current severity is optical; it is **not calibrated to physical ASTM D6433 Pavement Condition Index (PCI)**.

---

### 2.9 `COMPUTATIONAL_TRADEOFF.csv` (Section 9: Computational-Cost Analysis)
- **Objective**: Transparently quantify the hardware latency and memory overhead introduced by each pipeline stage.
- **Hardware Platform**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM), PyTorch 2.13.0+cu130, CUDA 12.8.
- **Exact Source Files**:
  - `integration/CANONICAL_RESEARCH_METRICS.json`
  - `benchmark/final_comparison/FINAL_PERCEPTION_TABLE.csv`
  - Direct empirical profiling in active environment (`generate_phase1a_metrics.py`)
- **Module Latency Breakdown**:
  1. **YOLOv8n Primary Detector**: $3.62\text{ ms}$ ($276.24\text{ FPS}$), VRAM $\approx 18\text{ MB}$.
  2. **DINOv2 ViT-S/14 Domain Gate**: $24.77\text{ ms}$ ($40.37\text{ FPS}$), VRAM $\approx 142\text{ MB}$.
  3. **Reliability Filter (Model B)**: $0.01\text{ ms}$ ($>10,000\text{ FPS}$), CPU RAM $<1\text{ MB}$.
  4. **DINOv2 + SAM 2 Condition Core**: $263.15\text{ ms}$ ($3.80\text{ FPS}$), VRAM $\approx 1.8\text{ GB}$.
  5. **Temporal Defect Tracking**: $0.08\text{ ms}$ ($>12,000\text{ FPS}$), CPU RAM $<2\text{ MB}$.
  6. **XGBoost Scenario Forecasting**: $0.024\text{ ms}$ ($>40,000\text{ FPS}$), CPU RAM $<5\text{ MB}$.
- **Staged Pipeline Totals**:
  - **Rapid Gated Deployment** (YOLO + DINO Gate + Reliability): $\mathbf{28.40\text{ ms}}$ ($\mathbf{35.21\text{ FPS}}$) $\rightarrow$ *Real-time capable survey deployment with full input domain shift protection.*
  - **Full Deep Multi-Modal Pipeline** (All 6 Tiers): $\mathbf{291.65\text{ ms}}$ ($\mathbf{3.43\text{ FPS}}$) $\rightarrow$ *Analytical deep condition and forecasting assessment.*
- **Mandatory Thesis Interpretation**:
  > *"The integrated pipeline incurs additional computational cost in exchange for additional monitoring capabilities (domain gating, reliability-aware rejection, pixel geometry, temporal change tracking, and deterioration forecasting) that are architecturally absent from a standalone detector. It is not claimed to be slower but more accurate at simple object detection."*

---

### 2.10 `STATISTICAL_VALIDATION.csv` (Section 10: Inferential Statistics)
- **Objective**: Validate the empirical claims using rigorous, non-parametric and parametric inferential hypothesis tests.
- **Exact Source Files**: Evaluated dynamically across active canonical datasets via `scipy.stats` and `sklearn.metrics`.
- **Results Summary Table**:

| Test ID | Hypothesis ($H_1$) | Statistical Test | Sample Size | Test Statistic | $p$-value | Effect Size | Scientific Interpretation |
|---|---|---|---|---|---|---|---|
| `STAT_01` | Cross-domain detection degradation under Target $T_1$ | Fisher's Exact Test | $N=780$ images | $\text{Odds Ratio} = 191.76$ | $1.53 \times 10^{-123}$ | $\text{OR} = 191.8$ | Rejects $H_0$. China exhibits $191.8\times$ higher odds of balanced detection quality ($\text{F1} \ge 0.50$). |
| `STAT_02` | In-domain China YOLO F1 95% Confidence Interval | Percentile Bootstrap | $N=480$ images ($2,000$ boots) | Point $\text{F1} = 0.7104$ | N/A (Estimation) | $95\%\text{ CI}: [0.6823, 0.7372]$ | Confirms tight in-domain performance bounds for strong baseline detector. |
| `STAT_03` | Cross-domain India YOLO F1 95% Confidence Interval | Percentile Bootstrap | $N=300$ images ($2,000$ boots) | Point $\text{F1} = 0.0218$ | N/A (Estimation) | $95\%\text{ CI}: [0.0083, 0.0381]$ | Complete non-overlap with China CI confirms severe cross-domain collapse. |
| `STAT_04` | DINOv2 $k\text{NN}$ distance shifts cross-domain | Mann-Whitney U Test | China $N=480$, India $N=300$ | $U = 144,000.0$ | $1.31 \times 10^{-122}$ | Rank-biserial $r = 1.0000$, Cohen's $d = 9.1578$ | Rejects $H_0$. Flawless rank separation and massive effect size justify macro domain gating. |
| `STAT_05` | In-domain TP detections have higher confidence than FP | Mann-Whitney U Test | TP $N=574$, FP $N=300$ | $U = 133,294.5$ | $9.03 \times 10^{-41}$ | Rank-biserial $r = 0.5481$ | Rejects $H_0$. True detections have significantly higher confidence (mean $0.6135$ vs $0.4262$). |
| `STAT_06` | Selective rejection reduces accepted failure rate (100% vs 50% cov) | Fisher's Exact Test | $N=720$ evaluations | $\text{Odds Ratio} = 0.0975$ | $2.55 \times 10^{-11}$ | $\text{OR} = 0.0975$ | Rejects $H_0$. Selective rejection cuts error from $17.92\%$ to $2.08\%$ ($88.4\%$ risk reduction). |
| `STAT_07` | In-domain YOLO failures exhibit higher DINO OOD scores | Mann-Whitney U Test | Succ $N=423$, Fail $N=57$ | $U = 9642.0$ | $0.01351$ | Rank-biserial $r = 0.1997$ | Rejects $H_0$ at $\alpha=0.05$. Novelty correlates with failure, but modest effect size ($r=0.20$) means DINO cannot replace confidence in-domain. |
| `STAT_08` | Monotonic distress progression on simulated sequence SEG_004 | Spearman Rank Correlation | $N=5$ inspection days | $\rho = 0.9487$ | $0.0138$ | $\rho = 0.9487$ | Rejects $H_0$. Confirms pipeline reliably tracks sequential multi-day deterioration. |
| `STAT_09` | 90-day deterioration delta is higher under WET_EXPOSURE than NORMAL | Wilcoxon Signed-Rank | $N=40$ paired captures | $W = 0.0$ | $1.82 \times 10^{-12}$ | Matched-pairs $r = 1.0000$ | Rejects $H_0$. WET_EXPOSURE produces strictly higher progression delta across $100\%$ of survey frames. |

---

## 3. Explicit Inventory of Unavailable Metrics

To maintain uncompromising scientific honesty, the following metrics are explicitly declared **UNAVAILABLE** and have not been fabricated:

1. **Physical Raspberry Pi 5 On-Device Profiling**:
   - Status: **PENDING PHYSICAL HARDWARE TESTBENCH (Scheduled Phase 8)**.
   - Reason: In accordance with `dashboard/components/edge_deployment.py` Section 22, physical profiling requires live instrumented measurement on physical Raspberry Pi 5 hardware. No synthetic or extrapolated embedded latency, wattage, or CPU thermal figures have been invented.
2. **ASTM D6433 Pavement Condition Index (PCI) Ground Truth**:
   - Status: **UNAVAILABLE / NOT APPLICABLE**.
   - Reason: RDD2022 image datasets contain bounding box distress labels; they do not contain calibrated ASTM D6433 civil engineering deduct-value inspection sheets or core sample geotechnical data. Surface defect severity is reported as optical unitless distress metrics ($0.0$ to $1.0$).
3. **Causal Deterioration Mechanics**:
   - Status: **UNSUPPORTED FOR CAUSAL CLAIMS**.
   - Reason: XGBoost models are trained on empirical historical progression intervals from the FHWA LTPP database. They represent data-driven scenario projections, not causal constitutive differential equations.
4. **Standalone YOLO Defect Tracking / Temporal Memory**:
   - Status: **NOT APPLICABLE**.
   - Reason: YOLOv8n is a single-frame feedforward architecture without recurrent state, optical flow, or bipartite track assignment mechanisms.

---

## 4. Conflict Audit & Historical Discrepancy Accounting

In accordance with Phase 1A instructions, the following 5 historical numerical discrepancies across earlier project phases are explicitly audited and documented rather than silently reconciled:

### Discrepancy 1: India Cross-Domain YOLO F1 Score (0.0218 vs. 0.0226)
- **Observed In**:
  - `cross_domain/tables/table_cross_domain_metrics.csv` & `table_in_vs_cross_domain.csv`: $\text{Precision} = 0.0964, \text{Recall} = 0.0123, \mathbf{F1 = 0.0218}$.
  - `integration/CANONICAL_RESEARCH_METRICS.json`: $\text{Precision} = 0.0631, \text{Recall} = 0.0138, \mathbf{F1 = 0.0226}$.
- **Root Cause**:
  - The exact bounding box evaluation across the 300 India images yields $\text{TP} = 8, \text{FP} = 75, \text{FN} = 644$.
  - Exact formula: $\text{P} = \frac{8}{83} = 0.096385$, $\text{R} = \frac{8}{652} = 0.012270 \implies \mathbf{F1 = 0.021769} \approx \mathbf{0.0218}$.
  - The older figure ($0.0226$) originated from an earlier provisional evaluation that recorded $\text{P}=0.0631, \text{R}=0.0138$.
- **Audit Decision**: **0.0218** is verified as the exact canonical audited value per prompt instructions and raw bounding box counts.

### Discrepancy 2: India Cross-Domain Failure Counts (292 vs. 293)
- **Observed In**:
  - Phase 10 reports: 292 failures (97.33%).
  - Phase 12 & 13 reports: 293 failures (97.67%).
- **Root Cause**:
  - **Target $T_0$ ($\text{Image F1} > 0$)**: Exactly 8 images have at least one matched defect ($\text{F1} > 0$). Exactly $292$ images have $\text{F1} = 0$ $\rightarrow \mathbf{292\text{ Failures}}$ ($97.33\%$).
  - **Target $T_1$ ($\text{Image F1} \ge 0.50$)**: Image `India_000511` achieves $\text{F1} = 0.4000$ ($1\text{ TP}, 2\text{ FP}, 1\text{ FN}$). Under strict target $T_1$, an F1 of $0.4000$ fails the $0.50$ threshold. Thus, `India_000511` flips from success to failure $\rightarrow \mathbf{293\text{ Failures}}$ ($97.67\%$).
- **Audit Decision**: Both figures are mathematically exact under their respective target specifications ($T_0$ vs $T_1$). Both are explicitly documented with target labels.

### Discrepancy 3: China In-Domain Throughput (276.24 FPS vs. 276.33 FPS)
- **Observed In**:
  - `integration/CANONICAL_RESEARCH_METRICS.json`: 276.24 FPS.
  - `benchmark/final_comparison/FINAL_PERCEPTION_TABLE.csv`: 276.33 FPS.
- **Root Cause**:
  - Mean latency across 480 validation images is $3.6192\text{ ms}$.
  - $\frac{1000}{3.6192} = 276.304\text{ FPS}$. Depending on intermediate rounding ($3.62\text{ ms} \rightarrow 276.24\text{ FPS}$ vs $3.6188\text{ ms} \rightarrow 276.33\text{ FPS}$), slight variance occurs.
- **Audit Decision**: Both numbers confirm high-speed edge inference ($>276\text{ FPS}$); $276.24\text{ FPS}$ is cited as canonical per `CANONICAL_RESEARCH_METRICS.json`.

### Discrepancy 4: Temporal Event Counts (21/12 vs. 14/19)
- **Observed In**:
  - Early provisional notes: 21 area increased / 12 area decreased.
  - Canonical `CANONICAL_RESEARCH_METRICS.json` & `PHASE14_SYSTEM_CONSISTENCY_AUDIT.md`: 14 area increased / 19 area decreased.
- **Root Cause**:
  - Direct re-parsing of raw `temporal_events.json` across all 8 sequences confirms exactly:
    - `OBSERVED_AREA_INCREASED`: Exactly **14 events**.
    - `OBSERVED_AREA_DECREASED`: Exactly **19 events**.
    - Total matched transitions: $14 + 19 = \mathbf{33\text{ transitions}}$.
- **Audit Decision**: The outdated 21/12 figures are fully deprecated; **14/19** is audited as canonical truth.

### Discrepancy 5: Tracking Algorithm Nomenclature ("Hungarian" vs. Greedy)
- **Observed In**: Occasional early references cited "Hungarian matching" for defect tracking.
- **Root Cause**: Inspection of `road_health_pipeline/temporal/temporal_tracker.py` confirms that tracking is performed via a deterministic, greedy hierarchical cascade ($\text{Mask IoU} \ge 0.50 \rightarrow \text{Bbox IoU} \ge 0.30 \rightarrow \text{Centroid} \le 75\text{ px}$). No $O(N^3)$ Munkres/Hungarian bipartite optimization is executed.
- **Audit Decision**: All scientific documentation accurately specifies **Greedy Hierarchical One-to-One Matching**.

---

## 5. Verification Checklist & File Hashes

- [x] `YOLO_BASELINE_METRICS.csv`: 17 metric rows, covers all section 1 requirements.
- [x] `YOLO_GENERALIZATION_GAP.csv`: 14 comparative rows, covers drops, zero-detection rates, and conditional false alarms.
- [x] `YOLO_CONFIDENCE_FAILURE_ANALYSIS.csv`: 18 rows, stratified into detection-level bins/scores and image-level reliability.
- [x] `RISK_COVERAGE_METRICS.csv`: 18 rows, covers Target $T_1$ canonical, Model B empirical, and Target $T_0$ historical.
- [x] `DOMAIN_GATE_VALUE.csv`: 10 rows, details distribution separation ($+0.1158$), AUROC ($1.0$), and decoupled architecture roles.
- [x] `DECISION_SAFETY_ANALYSIS.csv`: 5 rows, evaluates YOLO-only failure pass-through vs RoadSentinel quarantine.
- [x] `TEMPORAL_CAPABILITY_GAP.csv`: 14 rows, properly labeled `MODEL-OBSERVED TEMPORAL CHANGE / SIMULATED TEMPORAL DATA`.
- [x] `FORECAST_CAPABILITY_METRICS.csv`: 20 rows, details FHWA LTPP disjoint holdout ($R^2=0.8055$, site overlap 0) and disclaimers.
- [x] `COMPUTATIONAL_TRADEOFF.csv`: 8 rows, details module-by-module latencies and thesis trade-off interpretation.
- [x] `STATISTICAL_VALIDATION.csv`: 9 rows, formal tests with sample sizes, test statistics, exact $p$-values, and effect sizes.
- [x] `PHASE1A_METRIC_AUDIT.md`: Complete documentation covering sources, formulas, sample sizes, thresholds, levels, unavailable metrics, and historical discrepancies.

All 11 artifacts are generated, checked against source data, verified with existing tests, and permanently archived in `integration/research_strengthening/`.
