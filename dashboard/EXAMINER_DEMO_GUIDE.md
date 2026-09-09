# RoadSentinel — Examiner Demonstration Guide (5-Minute Walkthrough)

This guide provides a standardized, rigorous, 5-minute demonstration script for examiners, committee members, and research panels. It walks sequentially through the entire 6-tier RoadSentinel architecture, highlighting empirical strengths while preserving scientific honesty.

---

## Pre-Flight Setup

Launch the dashboard locally:
```bash
streamlit run dashboard/app.py
```
*Expected Startup*: Loads instantly at `http://localhost:8501` in offline-safe Verified Demo Mode.

---

## 5-Minute Scripted Walkthrough

### STEP 1: System Architecture & Research Overview (0:00 – 0:45)
1. Navigate to: `1. System Overview`
2. **Talking Points**:
   - Introduce RoadSentinel as an integrated perception, reliability, and deterioration forecasting intelligence framework.
   - Point to the **Canonical Architecture Diagram**: Raw images pass through (1) Macro Domain Gate, (2) Perception Core, (3) Reliability Filter, (4) Temporal Tracking, (5) XGBoost Forecaster, and (6) Decision Engine.
   - Highlight the **Headline Findings**:
     - YOLO dominates in-domain detection ($F_1 = 0.7104$ vs $0.0267$, $72.7\times$ lower latency).
     - DINOv2 provides perfect macro domain gating ($\text{AUROC} = 1.0000$).
     - Confidence calibration yields up to $88.4\%$ error reduction under selective prediction.
     - Multi-evidence fusion prevents 100% of evaluated cross-domain misdetections from being accepted.

---

### STEP 2: Single-Image Assessment (0:45 – 1:30)
1. Navigate to: `2. Single-Image Assessment`
2. Set **Segment** to `SEG_004`, **Day** to `Day 01` (Baseline pristine condition).
3. **Talking Points**:
   - Label: **CURRENT MODEL ASSESSMENT**.
   - Model Severity is `0.0000`, Defect Count is `0`.
   - Domain status: `IN_DOMAIN` ($d_{k\text{NN}} = 0.1841$, well below threshold $0.4491$).
   - Decision Engine outputs `MONITOR` with primary rationale: *"Low model-observed severity with stable temporal evidence."*
4. Toggle **Day** to `Day 05`:
   - Point out severity increase to `0.7302`, defect count `2`.
   - Decision immediately updates to `PRIORITY_REVIEW`.

---

### STEP 3: Multi-Day Temporal Progression (1:30 – 2:15)
1. Navigate to: `5. Temporal Change Analysis`
2. Select sequence `SEG_004_D01_D05 — Progression Sequence`.
3. **Talking Points**:
   - Label: **MODEL-OBSERVED TEMPORAL CHANGE**.
   - Review the severity progression table ($0.0000 \to 0.7302$, $\Delta = +0.7302$).
   - Emphasize tracking algorithm: Greedy one-to-one matching (`Mask IoU ≥ 0.50` $\to$ `Bbox IoU ≥ 0.30` $\to$ `Centroid ≤ 75px`).
   - Switch to `SEG_003_D01_D07 — Stability Control Test`:
     - Highlight the $0.67\%$ Coefficient of Variation across 7 invariant states, proving perception repeatability under fixed lighting.
   - Point out the **Canonical Event Counts**: 48 New Defects, 33 Matched Transitions (14 Area Increased, 19 Area Decreased), 34 Not Observed.

---

### STEP 4: XGBoost Scenario Deterioration Forecasting (2:15 – 3:00)
1. Navigate to: `4. XGBoost Future Forecast`
2. Select `SEG_004`, `Day 05`, Horizon `90 Days Ahead`.
3. **Talking Points**:
   - Label: **MODEL-BASED FORECAST** (Trained on FHWA LTPP pavement database, $N=28,834$).
   - Compare the 5 climate/traffic scenarios: `NORMAL`, `HIGH_HEAT`, `HEAVY_TRAFFIC`, `HEAVY_RAIN`, `WET_EXPOSURE`.
   - Point out that **`WET_EXPOSURE`** generates the highest projected severity ($0.7614$, $\Delta +0.0312$), confirming pavement sensitivity to prolonged moisture penetration.
   - Read the disclaimer: *Forecasts are empirical regression projections, not deterministic physical deterioration laws.*

---

### STEP 5: Perception Benchmark & Research Balance (3:00 – 3:45)
1. Navigate to: `6. YOLO vs DINO/SAM Research`
2. **Talking Points**:
   - Primary Benchmark (China_Drone validation split, 480 images, 742 ground truth boxes).
   - YOLOv8n: Precision $0.6568$, Recall $0.7736$, F1 $0.7104$, Latency $3.62\text{ ms}$.
   - DINOv2+SAM2: Precision $0.0221$, Recall $0.0337$, F1 $0.0267$, Latency $263.15\text{ ms}$.
   - YOLO exhibits $72.7\times$ lower latency ($276.2\text{ FPS}$ vs $3.8\text{ FPS}$).
   - Emphasize scientific context: China_Drone is in-domain for YOLO; DINO/SAM is zero-shot anomaly localization without class supervision.

---

### STEP 6 & 7: Cross-Domain Collapse & Domain Quarantine (3:45 – 4:30)
1. Navigate to: `7. Cross-Domain Generalization`
2. **Talking Points**:
   - Transfer from China Drone (aerial) to India Dashcam (ground vehicle, 300 images, 652 GT boxes).
   - Supervised YOLO F1 collapses by $-96.9\%$ ($0.7104 \to 0.0218$, $292/300$ failures under $T_0$, $293/300$ under $T_1$).
   - DINOv2 Foundation Gating: $d_{k\text{NN}}$ cleanly separates the two domains ($\text{AUROC} = 1.0000$, margin $+0.1158$).
   - **Crucial Safety Outcome**: Decision engine routes **$300/300$ (100.0%)** India benchmark images to `DOMAIN_ESCALATION`, preventing all 293 silent perception failures from automated acceptance.

---

### STEP 8: Decision Engine & Conclusion (4:30 – 5:00)
1. Navigate to: `8. Decision & Inspection Priority`
2. **Talking Points**:
   - Shows the full 5-tier action taxonomy (`AUTOMATED_ACCEPT`, `MONITOR`, `REINSPECT`, `PRIORITY_REVIEW`, `DOMAIN_ESCALATION`).
   - For Experiment A: $21\text{ MONITOR}$, $10\text{ REINSPECT}$, $9\text{ PRIORITY\_REVIEW}$, $0\text{ AUTOMATED\_ACCEPT}$, $0\text{ DOMAIN\_ESCALATION}$.
   - Conclude: RoadSentinel achieves end-to-end multi-evidence synthesis, transparent safety gating, and reproducible empirical rigor.
