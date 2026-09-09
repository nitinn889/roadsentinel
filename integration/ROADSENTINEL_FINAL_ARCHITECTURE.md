# RoadSentinel: Final Canonical Technical Architecture

**Document**: `integration/ROADSENTINEL_FINAL_ARCHITECTURE.md`  
**Status**: **FROZEN & VERIFIED** (Phases 1 through 13 Complete)  
**System Classification**: Decoupled Two-Tier Reliability-Aware Road-Health Perception & Decision Framework

---

## 1. Executive Summary & Design Philosophy

RoadSentinel is a multi-modal computer vision and machine learning framework designed for automated, reliable road distress assessment and inspection prioritization.

### Central Engineering Tenets:
1. **Decoupled Perception Safety**: Foundation models (DINOv2) provide macro operational domain shift detection at the input boundary, while specialized lightweight detectors (YOLOv8n) provide real-time distress localization.
2. **Evidence Stream Independence**: Current model assessment, model-observed temporal tracking, scenario-conditioned forecasts, and perception reliability remain mathematically independent data fields without uncalibrated multiplication.
3. **Transparent Decision Governance**: An interpretable deterministic policy translates perception outputs, temporal trajectories, and environmental forecasts into five actionable inspection review tiers.

---

## 2. End-to-End Architecture Diagram

```
                              [ Incoming Road Capture ]
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │          TIER 1: MACRO DOMAIN GATE             │
                 │   DINOv2 ViT-S/14 Foundation Embeddings        │
                 │   kNN Cosine Distance vs. Training ODD Ref     │
                 └───────────────────────┬─────────────────────────┘
                                         │
                 ┌───────────────────────┴─────────────────────────┐
                 │                                                 │
          [ d_kNN > 0.4491 ]                               [ d_kNN ≤ 0.4491 ]
                 │                                                 │
                 ▼                                                 ▼
     ┌──────────────────────┐                    ┌──────────────────────────────────┐
     │  DOMAIN_ESCALATION   │                    │    TIER 2: PERCEPTION CORE       │
     │  Immediate Human     │                    │  • YOLOv8n Bbox Detection        │
     │  Dispatch / Review   │                    │  • DINOv2+SAM2 Surface Geometry  │
     └──────────────────────┘                    └─────────────────┬────────────────┘
                                                                   │
                                                                   ▼
                                                 ┌──────────────────────────────────┐
                                                 │   TIER 3: RELIABILITY FILTER     │
                                                 │  Model B Calibrated Confidence   │
                                                 │  HIGH (≥0.85) / MED / LOW (<0.60)│
                                                 └─────────────────┬────────────────┘
                                                                   │
                                         ┌─────────────────────────┴─────────────────────────┐
                                         │                                                   │
                                  [ LOW (<0.60) ]                                     [ HIGH / MEDIUM ]
                                         │                                                   │
                                         ▼                                                   ▼
                              ┌──────────────────────┐                     ┌──────────────────────────────────┐
                              │ REINSPECT / REVIEW   │                     │  TIER 4 & 5: TEMPORAL & FORECAST │
                              │ Quarantined Sensor   │                     │  • Same-Camera Defect Tracking   │
                              │ Reinspection         │                     │  • XGBoost 90-Day Projections    │
                              └──────────────────────┘                     └─────────────────┬────────────────┘
                                                                                             │
                                                                                             ▼
                                                                           ┌──────────────────────────────────┐
                                                                           │   TIER 6: DECISION & PRIORITY    │
                                                                           │  • PRIORITY_REVIEW               │
                                                                           │  • MONITOR                       │
                                                                           │  • AUTOMATED_ACCEPT              │
                                                                           └──────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### Tier 1: Macro Domain Gate (DINOv2 ViT-S/14)
- **Model**: `facebookresearch/dinov2` (`dinov2_vits14`, 384-dimensional CLS token).
- **Reference ODD**: 1,921 training embeddings from RDD2022 China_Drone.
- **Metric**: Mean cosine distance across $k=20$ nearest neighbors ($d_{k\text{NN}}$).
- **Operational Thresholds**:
  - $\text{DOMAIN\_FAMILIAR}$: $d_{k\text{NN}} \le 0.3804$ ($p_{95}$)
  - $\text{DOMAIN\_WARNING}$: $0.3804 < d_{k\text{NN}} \le 0.4491$ ($p_{95}$ to $p_{99}$)
  - $\text{EXTREME\_DOMAIN\_SHIFT}$: $d_{k\text{NN}} > 0.4491$ ($p_{99}$)
- **Performance**: $\text{AUROC} = 1.0000$, $\text{AUPRC} = 1.0000$, $100.0\%$ cross-domain shift detection on RDD2022 India.

### Tier 2: Perception Core
- **YOLOv8n Supervised Detector**:
  - Task: Bounding box road damage localization (D00, D10, D20, D40, Repair).
  - Checkpoint: `yolo/weights/best.pt` (Input size $512\times 512$, $\text{conf}=0.25$).
  - In-Domain Validation ($N=480$): Precision $= 0.6568$, Recall $= 0.7736$, F1 $= 0.7104$.
- **DINOv2 + SAM 2 Current Road Health Pipeline**:
  - Pipeline: Marion Day-5 NORMAL pipeline (`dinov2_vits14` patch features + `sam2.1_hiera_small`).
  - Output Metrics: `current_severity`, `defect_count`, `defect_area_ratio`, `crack_area_ratio`, `surface_anomaly_score`, `water_flag`.

### Tier 3: Perception Reliability Filter
- **Model**: Standardized Logistic Regression / Model B Confidence Calibration.
- **Features**: `max_confidence`, `mean_confidence`, `std_confidence`, `yolo_pred_count`, `num_low_conf`, `num_high_conf`.
- **In-Domain Failure Prediction ($N=480$)**: $\text{AUROC} = 0.8166$ to $0.8650$, $\text{AUPRC} = 0.6490$ to $0.7298$.
- **Operational Bands**:
  - `HIGH` ($\ge 0.85$): $91.97\%$ success rate under Target $T_1$.
  - `MEDIUM` ($[0.60, 0.85)$): Borderline confidence.
  - `LOW` ($< 0.60$): Isolates $74.51\%$ to $89.19\%$ failure rate.

### Tier 4: Temporal Engine
- **Algorithm**: Greedy hierarchical matching (Mask IoU $\rightarrow$ Bbox IoU $\rightarrow$ Centroid/Area fallback).
- **Invariance Rule**: Matching is strictly restricted to valid same-camera viewpoints.
- **Canonical Experiment A Metrics**: 48 unique tracks, 33 matched events, 48 new-defect events, 34 not-observed events, 14 area-increased events, 19 area-decreased events.

### Tier 5: Scenario Forecasting Engine (XGBoost V2)
- **Model**: Scenario-Conditioned XGBoost Model V2 trained on LTPP pavement deterioration records.
- **Horizons**: 30-day, 60-day, 90-day future severity projections.
- **Environmental Scenarios**: `NORMAL`, `HEAVY_RAIN`, `HEAVY_TRAFFIC`, `HIGH_HEAT`, `WET_EXPOSURE`.
- **Empirical Sensitivity**: `WET_EXPOSURE` creates the largest 90-day deterioration delta across all evaluated road segments.

### Tier 6: Decision & Priority Engine
- **Output Action Tiers**:
  1. `DOMAIN_ESCALATION`: Out-of-distribution input; automated acceptance blocked.
  2. `REINSPECT`: Low perception reliability; sensor verification required.
  3. `PRIORITY_REVIEW`: High current severity, rapid temporal growth, or severe forecast escalation.
  4. `MONITOR`: Moderate severity, stable response, or rising scenario vulnerability.
  5. `AUTOMATED_ACCEPT`: High reliability, pristine condition, and stable temporal history.

---

## 4. Verification & Validation Summary

| Test Domain / Benchmark | Dataset Size | Key Validation Result | Safety Significance |
|---|:---:|---|---|
| **RDD2022 China_Drone** | 480 frames | Reliability selective prediction reduces in-domain error from $17.92\%$ to $2.08\%$ ($+88.4\%$ reduction). | Proves risk-controlled autonomous acceptance. |
| **RDD2022 India Dashcam** | 300 frames | DINOv2 Domain Gate intercepts $300/300$ frames ($0/293$ perception failures accepted). | Prevents catastrophic silent failures under domain shift. |
| **Experiment A (Unreal Engine)** | 40 physical frames | Multi-stream hierarchy assigns 21 Monitor, 10 Reinspect, 9 Priority Review. | Demonstrates coherent multi-day inspection triage. |
