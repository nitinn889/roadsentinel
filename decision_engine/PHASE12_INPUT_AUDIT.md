# RoadSentinel Phase 12 Input Audit for Decision Engine Integration

**Document**: `decision_engine/PHASE12_INPUT_AUDIT.md`  
**Purpose**: Formal extraction of frozen Phase-12 empirical parameters and findings before implementing the Phase-13 Decision Engine.

---

## 1. Audited Reliability Models & Performance

From `reliability_validation/tables/table_full_ablation.csv` and `reliability_validation/RELIABILITY_VALIDATION_RESEARCH_SUMMARY.md`:

| Perceptual Target | Best In-Domain Model | Features Included | In-Domain AUROC [95% CI] | In-Domain AUPRC | In-Domain Failure F1 | Cross-Domain India Transfer AUROC |
|---|---|---|:---:|:---:|:---:|:---:|
| **$T_0$ ($\text{F1} > 0$)** | **Model E** | Confidence + DINO OOD | **0.8650** [0.7985, 0.9210] | 0.7298 | 0.7111 | 0.9217 |
| **$T_1$ ($\text{F1} \ge 0.50$)** | **Model B** | YOLO Confidence Features | **0.8166** [0.7667, 0.8669] | 0.6490 | 0.5839 | 0.9093 |
| **$T_2$ ($\text{Recall} \ge 0.50$)** | **Model A** | YOLO Max Confidence | **0.8239** [0.7721, 0.8800] | 0.6686 | 0.6341 | 0.9221 |
| **$T_3$ ($\text{Recall} \ge 0.75$)** | **Model A** | YOLO Max Confidence | **0.6849** [0.6311, 0.7392] | 0.5893 | 0.5000 | 0.8927 |
| **$T_4$ ($\text{Recall} = 1.0$)** | **Model A** | YOLO Max Confidence | **0.6777** [0.6247, 0.7337] | 0.5850 | 0.4984 | 0.8927 |

### Core Model Findings:
1. **YOLO Confidence Features (Model B)** provide the dominant predictive signal for in-domain sample-level failure prediction.
2. **DINOv2 Sample-Level Contribution**: Standalone DINO OOD (Model C) achieves only $\text{AUROC} = 0.5428$ to $0.6061$ and adds $\Delta \le +0.0001$ over confidence features alone.
3. **Operational Reliability Bands**:
   - $\text{HIGH Band}: \text{Reliability} \ge 0.85$ (Yields $91.97\%$ success rate under $T_1$, $95.38\%$ under $T_0$).
   - $\text{MEDIUM Band}: 0.60 \le \text{Reliability} < 0.85$ (Borderline confidence; routes to secondary inspection).
   - $\text{LOW Band}: \text{Reliability} < 0.60$ (Isolates $74.51\%$ failure rate under $T_1$, $89.19\%$ under $T_0$).

---

## 2. Audited DINOv2 Macro Domain Gate

From `reliability_validation/tables/table_domain_gate.csv`:

| Parameter / Metric | Empirical Value | Source / Derivation |
|---|---|---|
| **China Validation Raw $k$-NN Distance** | Mean $0.1841 \pm 0.0847$, Range $[0.0393, 0.5247]$ | `dino_knn_distance` on $N=480$ validation frames |
| **India Cross-Domain Raw $k$-NN Distance** | Mean $0.8432 \pm 0.0441$, Range $[0.6405, 0.9261]$ | `dino_knn_distance` on $N=300$ cross-domain frames |
| **Separation Margin** | **$+0.1158$** cosine distance | $\min(\text{India}) - \max(\text{China}) = 0.6405 - 0.5247$ (Zero distribution overlap) |
| **Domain Classification AUROC / AUPRC** | **1.0000 / 1.0000** | Standalone binary domain classification |
| **Domain Gate Threshold ($p_{99}$)** | **$0.4491$** | China development reference 99th percentile |
| **Domain Warning Threshold ($p_{95}$)** | **$0.3804$** | China development reference 95th percentile |
| **India Domain-Shift Detection Rate** | **100.0%** (300 / 300 flagged as shifted) | Evaluated at $p_{99} = 0.4491$ |
| **China False Warning Rate** | **1.46%** (7 / 480 flagged as borderline) | Evaluated at $p_{99} = 0.4491$ |

---

## 3. Audited Temporal Feature Ablation

From `reliability_validation/tables/table_temporal_ablation.csv`:

| Model Configuration | AUROC | AUPRC | Balanced Acc | Failure F1 | Key Research Conclusion |
|---|:---:|:---:|:---:|:---:|---|
| **Model B (Confidence Only)** | 0.1556 | 0.0698 | 0.5000 | 0.0000 | Fails during environmental lighting confounds (overcast inflates confidence). |
| **Model I (Confidence + Temporal)** | **0.7111** | **0.5370** | **0.6667** | **0.5000** | Causal temporal tracking features flag sudden drops in defect persistence ($\Delta \text{AUROC} = +0.5555$). |
| **Model J (Confidence + DINO + Temp)** | 0.7111 | 0.5051 | 0.6667 | 0.5000 | Matches Model I; DINO provides no extra gain at sequence level. |
| **Model K (All Causally Valid Features)** | 0.7111 | 0.5051 | 0.6667 | 0.5000 | Confirms temporal tracking is the key causal feature. |

*Sample Size Note*: Evaluated on $N=33$ sequence states (25 consecutive step transitions) across Experiment A. Classified as preliminary evidence pending larger multi-day datasets.

---

## 4. Final Recommended Architecture Freeze

**Architecture C**: **DINOv2 Macro Domain Gate + YOLOv8n + Confidence-Based Reliability Estimation**
- **Tier 1 (Input Gate)**: DINOv2 cosine distance to training reference ($d \le 0.4491$). If violated $\implies \text{DOMAIN\_ESCALATION}$.
- **Tier 2 (Perception Core)**: YOLOv8n object detection & DINOv2+SAM2 current condition estimation.
- **Tier 3 (Reliability Filter)**: Model B confidence estimation assigns $\text{HIGH} \ge 0.85$, $\text{MEDIUM} \in [0.60, 0.85)$, $\text{LOW} < 0.60$.
- **Tier 4 (Temporal & Forecast Interpretation)**: Model-observed temporal change and XGBoost 90-day scenario forecasts modulate final inspection review priority.
