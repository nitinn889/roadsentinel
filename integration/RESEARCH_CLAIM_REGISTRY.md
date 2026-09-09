# RoadSentinel Research Claim Registry

**Document**: `integration/RESEARCH_CLAIM_REGISTRY.md`  
**Purpose**: Official registry of scientifically verified claims, empirical evidence anchors, approved wording, and prohibited overclaims for paper writing, defense presentations, and dashboard text.

---

## 1. Registry Table of Approved Claims

| # | Research Claim Area | Supporting Empirical Artifact | Verified Metric Anchor | Approved Wording | Prohibited Overclaim |
|:---:|---|---|---|---|---|
| **1** | **In-Domain Perception Benchmark** | `benchmark/final_comparison/` (Phase 5/6) | YOLO: F1 = 0.7104 @ 3.62 ms<br>DINO/SAM: F1 = 0.0267 @ 263.15 ms | "YOLOv8n substantially outperformed the zero-shot DINOv2+SAM2 pipeline on in-domain distress localization with approximately $72.7\times$ lower inference latency." | ❌ "YOLO is universally superior to foundation models in all vision tasks."<br>❌ "DINO/SAM2 is completely useless." |
| **2** | **Cross-Domain Generalization** | `cross_domain/results/` (Phase 10) | YOLO F1: 0.7104 $\rightarrow$ 0.0226 ($-96.8\%$ relative drop)<br>DINO/SAM F1: 0.0000 | "Both perception approaches suffered catastrophic degradation when transferred zero-shot from drone survey to dashcam perspectives." | ❌ "Road damage detection is impossible across countries."<br>❌ "YOLO cannot be fine-tuned for dashcams." |
| **3** | **DINOv2 Foundation Domain Awareness** | `reliability_validation/tables/table_domain_gate.csv` (Phase 12) | Domain AUROC = 1.0000, AUPRC = 1.0000, $d_{k\text{NN}}$ separation margin $+0.1158$ | "DINOv2 foundation embeddings cleanly separated the evaluated in-domain and out-of-distribution benchmark datasets, detecting 100% of the evaluated Indian frames at $p_{99} = 0.4491$." | ❌ "DINOv2 is an infallible universal OOD detector."<br>❌ "DINOv2 solves the domain shift problem." |
| **4** | **Perception Reliability & Selective Prediction** | `reliability_validation/tables/table_full_ablation.csv` (Phase 12) | Model B AUROC = 0.8166–0.8650<br>Error reduced from 17.9% to 2.1% (88.4% reduction) | "Calibrated confidence reliability allows selective rejection of ambiguous perception samples, reducing accepted in-domain inspection error by up to 88.4%." | ❌ "Reliability guarantees zero errors."<br>❌ "Reliability replaces detector retraining." |
| **5** | **Model-Observed Temporal Analytics** | `integration/experiment_a/temporal/` (Phase 3/4) | 8 sequences, 33 transitions, 48 unique tracks, 14 area increased, 19 area decreased | "Greedy hierarchical matching successfully tracked persistent defect candidates across same-camera inspection states and quantified model-observed area transitions." | ❌ "Proves true physical asphalt deterioration physics."<br>❌ "Tracks defects across arbitrary moving cameras." |
| **6** | **Scenario-Conditioned Forecasting** | `integration/primary_goals/goal1_forecasts.csv` (Phase 11) | 600 records, 5 scenarios, 3 horizons<br>WET_EXPOSURE delta $= +0.0526$ | "XGBoost Model V2 generated scenario-conditioned 90-day future severity projections from LTPP pavement records, demonstrating heightened sensitivity to wet-exposure conditions." | ❌ "Forecasts are exact deterministic physical predictions."<br>❌ "Guarantees monotonic severity growth over time." |
| **7** | **Reliability-Aware Decision Governance** | `decision_engine/ROAD_HEALTH_DECISIONS.csv` (Phase 13) | 40 Experiment A frames: 21 Monitor, 10 Reinspect, 9 Priority Review<br>0 / 293 India failures auto-accepted | "The deterministic decision engine combines current assessment, temporal trends, scenario forecasts, and reliability tiers to prioritize inspection actions without uncalibrated score scaling." | ❌ "Directly prescribes civil engineering pavement repair work orders."<br>❌ "Autonomously manages municipal road budgets." |

---

## 2. Key Scientific Terminology Standards

1. **Current Assessment**: Always label as **`CURRENT MODEL ASSESSMENT`**. Never imply it is physical ground truth.
2. **Temporal Trends**: Always label as **`MODEL-OBSERVED TEMPORAL CHANGE`**. Avoid terms like "true deterioration".
3. **Future Projections**: Always label as **`MODEL-BASED SCENARIO FORECAST`**.
4. **Reliability & Domain Status**: Always distinguish **Sample-Level Reliability** (YOLO confidence) from **Macro Domain Gating** (DINOv2 foundation embeddings).
