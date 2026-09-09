# RoadSentinel Phase 14: System-Wide Consistency & Numerical Audit

**Document**: `integration/PHASE14_SYSTEM_CONSISTENCY_AUDIT.md`  
**Phase**: 14 (System-Wide Consistency Audit & Dashboard V2)  
**Status**: **COMPLETE & RECONCILED**  

---

## 1. Executive Summary of Audit Findings

A rigorous audit of all quantitative outputs and textual descriptions across Phases 4 through 13 was conducted to ensure absolute scientific precision, mathematical reproducibility, and eliminate ambiguous claims before finalizing Dashboard V2 and thesis publications.

| Audit Item | Initial State / Discrepancy | Root Cause & Resolution | Canonical Approved Value |
|---|---|---|---|
| **India Perception Failures** | Phase 10: 292 failures<br>Phase 13: 293 failures | **Target Definition Difference**: Phase 10 evaluated under $T_0$ ($\text{F1} > 0$), whereas Phase 12/13 evaluated under strict target $T_1$ ($\text{F1} \ge 0.50$). Frame `India_000511` ($\text{F1}=0.4000$) flips from success under $T_0$ to failure under $T_1$. | **$T_0$: 292 Failures / 8 Successes**<br>**$T_1$: 293 Failures / 7 Successes** |
| **Temporal Event Totals** | Discrepancy between historical 21/12 and canonical 14/19 | Recomputed directly from raw `temporal_events.json` event logs across all 8 sequences. | **14 Area Increased / 19 Area Decreased**<br>(33 Total Matched Transitions, 34 Not Observed, 48 New Defects, 48 Tracks) |
| **Tracking Algorithm Wording** | Occasional references to "Hungarian matching" | Audited actual code: Greedy one-to-one hierarchical matching (Mask IoU $\ge 0.50 \rightarrow$ Bbox IoU $\ge 0.30 \rightarrow$ Centroid/Area fallback). | **Greedy Hierarchical One-to-One Matching** (No Hungarian assignment) |
| **XGBoost Monotonicity Claims** | Claims of "convexity constraints" or "cannot mathematically reverse" | Audited `xgboost/train_scenario_model.py`: Model is a standard `XGBRegressor` with post-prediction clipping to $[0.0, 1.0]$. No hard tree constraints exist. | **Data-driven empirical regression with bounded $[0.0, 1.0]$ projection** |
| **Live Inference Scope** | Potential ambiguity over live DINO/SAM2 vs YOLO | Audited `dashboard/components/single_image.py`: YOLOv8n runs live on GPU; DINOv2+SAM2 runs in precomputed mode for responsive, offline-safe demonstration. | **Live Mode: YOLOv8n GPU**<br>**Precomputed Demo: Complete Multi-Modal Pipeline** |
| **Cross-Domain Claims** | Overclaims such as "solves domain shift" | Softened to defensible empirical scope: DINO embeddings cleanly separated the evaluated China and India datasets. | **"Separated the evaluated China and India benchmark domains ($d_{k\text{NN}}$ margin $+0.1158$)"** |
| **Component Roles** | Implication that DINO/SAM replaces YOLO | Reaffirmed primary detector role of YOLO; DINOv2 serves as Macro Domain Gating. | **YOLO: Primary Detector**<br>**DINOv2: Domain Awareness Gate** |
| **Temporal Wording** | Calling simulated sequences "true physical deterioration" | Replaced with strict observational phrasing. | **"MODEL-OBSERVED TEMPORAL CHANGE across simulated road states"** |

---

## 2. Detailed Technical Reconciliations

### 2.1 India Cross-Domain Failure Discrepancy (292 vs. 293)
An exhaustive inspection of `cross_domain/results/cross_domain_reliability_dataset.csv` ($N=300$ images, 652 GT boxes) explains the exact origin of both numbers:

1. **Target $T_0$ ($\text{Image-level F1} > 0$, Historical Phase 8 Target)**:
   - Exactly **8 images** have $\text{F1} > 0$:
     - `India_000312` ($\text{F1}=1.0000$)
     - `India_000319` ($\text{F1}=0.6667$)
     - `India_000320` ($\text{F1}=0.6667$)
     - `India_000511` ($\text{F1}=0.4000$)
     - `India_000596` ($\text{F1}=0.5000$)
     - `India_000656` ($\text{F1}=0.6667$)
     - `India_000758` ($\text{F1}=1.0000$)
     - `India_000760` ($\text{F1}=1.0000$)
   - Result: **292 Failures (97.33%) / 8 Successes (2.67%)**. (Reported in Phase 10).

2. **Target $T_1$ ($\text{Image-level F1} \ge 0.50$, Strict Phase 12/13 Target)**:
   - Image `India_000511` has $\text{F1} = 0.4000 < 0.50$ ($1 \text{ TP}, 2 \text{ FP}, 1 \text{ FN}$).
   - Under $T_1$, `India_000511` is classified as a **Failure**.
   - Result: **293 Failures (97.67%) / 7 Successes (2.33%)**. (Reported in Phase 12 & 13).

**Conclusion**: Both numbers are 100% mathematically correct. When citing cross-domain perception metrics, the document/dashboard must specify the active target ($T_0$ vs $T_1$).

---

### 2.2 Canonical Temporal Event Accounting
Parsed directly from `integration/experiment_a/temporal/*/temporal_events.json`:
- **`NEW_DEFECT`**: **48 events**
- **`NOT_OBSERVED`**: **34 events**
- **`OBSERVED_AREA_INCREASED`**: **14 events**
- **`OBSERVED_AREA_DECREASED`**: **19 events**
- **Total Matched Area Transitions**: **33 transitions** ($14 + 19 = 33$)
- **Total Unique Tracks**: **48 tracks**
- **Longest Track**: **7 states** (SEG_003 D01–D07 invariant shoulder patch)
- **Total Evaluated Same-Camera Sequences**: **8 sequences**

All references to outdated provisional numbers (such as 21/12) are fully deprecated.

---

### 2.3 Perception Latency Ratio
From Phase 5/6 common validation benchmark on RTX 5060 ($N=480$ images):
- **YOLOv8n**: $3.62 \text{ ms}$ per image ($276.2 \text{ FPS}$).
- **DINOv2 + SAM 2**: $263.15 \text{ ms}$ per image ($3.80 \text{ FPS}$).
- **Latency Ratio**: $\frac{263.15}{3.62} = 72.693\times \approx \mathbf{72.7\times}$.
- Approved Phrasing: *"YOLO exhibits approximately $72.7\times$ lower inference latency than the DINOv2+SAM2 pipeline."*

---

### 2.4 Evidence Stream Decoupling Verification
Verified that across `decision_engine/ROAD_HEALTH_DECISIONS.csv` and all integration scripts:
1. `current_severity` is reported strictly as measured by perception ($0.0000$ to $0.7302$).
2. `reliability_score` is reported strictly as calculated by confidence calibration ($0.0400$ to $0.9800$).
3. No mathematical multiplication ($\text{severity} \times \text{reliability}$) or division occurs anywhere in the codebase.
4. `observed_severity_change` is computed strictly from same-camera temporal histories.
5. `predicted_future_severity` is generated strictly by scenario-conditioned XGBoost models.
