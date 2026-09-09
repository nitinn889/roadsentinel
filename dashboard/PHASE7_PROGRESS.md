# RoadSentinel — Phase 7 Progress Report: Examiner-Facing Interactive Dashboard

**Status**: COMPLETED & VERIFIED  
**Date**: September 2026  
**Execution Environment**: Linux x86_64, NVIDIA GeForce RTX 5060 Laptop GPU, Streamlit 1.63.0  
**Simulation Policy Compliance**: 100% compliant. Zero interaction with Unreal Engine, CARLA, or Interactive Studio.

---

## 1. Executive Summary

Phase 7 establishes the **first fully unified, examiner-facing interactive research dashboard** for the RoadSentinel project. The dashboard consolidates the frozen scientific findings of Phases 2 through 6 into a modular, highly polished, and offline-resilient user interface:
- **Supervised Damage Detection**: Ultralytics YOLOv8n (5 classes, conf=0.25, imgsz=512).
- **Foundation Anomaly Perception**: Meta DINOv2 (ViT-S/14) + SAM2 (Hiera-S) zero-shot anomaly localization and segmentation.
- **Multi-Day Temporal Analytics**: Experiment A tracking across 40 captures, featuring the SEG_003 stability control (CV = 0.67%) and SEG_004 progression sequence (Δ = +0.7302).
- **Scenario-Conditioned Forecasting**: Frozen XGBoost Model V2 trained with monotone directional constraints on 6 observational environmental and operational stressor features.
- **Comparative Research & Failure Modes**: Comprehensive benchmark evaluation on the 480-image RDD2022 China_Drone validation split, paired with transparent documentation of algorithmic failure modes and IoU sensitivity.
- **Edge Deployment Roadmap**: Transparently structures the embedded evaluation framework while marking Raspberry Pi 5 benchmarking as **PENDING PHASE 8** with zero fabricated data.

---

## 2. Technical Framework & Execution Specs

- **Dashboard Framework**: Streamlit (v1.63.0) with custom Dark Glassmorphism CSS design system.
- **Launch Command**:
  ```bash
  .venv/bin/streamlit run dashboard/app.py
  ```
- **Cold-Start Latency**: **0.42 – 0.47 seconds** (measured across 3 fresh cold-start launches).
- **Default Port**: `8501` (configurable via `--server.port`).
- **Execution Architecture**:
  - **Verified Demo Mode (Default)**: Precomputed, cached, and validated outputs ensure instant, 100% deterministic operation during examiner presentations without requiring GPU initialization or external network calls.
  - **Live Inference Mode (Optional)**: Provides on-demand GPU inference for YOLOv8n (while DINOv2+SAM2 displays frozen precomputed segmentations), equipped with graceful try-except fallbacks that automatically revert to precomputed assets if CUDA errors or memory shortages occur.

---

## 3. Dashboard Structure & Navigation Sections

The application is structured into 8 modular sections navigated via an expandable sidebar:

```
dashboard/
├── app.py                      # Main entrypoint, sidebar routing, CSS injection
├── data_loader.py              # Cached, defensive loader for manifests, models, & tables
├── assets/
│   └── style.css               # Modern dark glassmorphism, Outfit/Inter typography, status pills
├── components/
│   ├── overview.py             # 1. System Overview & architecture pipeline diagram
│   ├── single_image.py         # 2. Single-Image Road Assessment (Side-by-side cards)
│   ├── forecast.py             # 3. Future Condition Forecast (XGBoost Model V2)
│   ├── temporal.py             # 4. Temporal Road Monitoring (SEG_003 & SEG_004)
│   ├── benchmark_view.py       # 5. YOLOv8n vs DINOv2+SAM2 Benchmark
│   ├── failure_modes.py        # 6. Failure Mode Analysis (DINO FPs, YOLO D20, IoU 0.25)
│   ├── edge_deployment.py      # 7. Edge Deployment (Raspberry Pi 5 — Pending Phase 8)
│   └── methodology.py          # 8. Research Methodology & Artifact Evidence Links
├── EXAMINER_DEMO_GUIDE.md      # Chronological 11-step presentation walkthrough script
├── PHASE7_PROGRESS.md          # This verification report
└── README.md                   # Operations, startup, and troubleshooting manual
```

---

## 4. Subsystem Integration Verification

### 4.1 Single-Image Road Assessment
- Presents a 3-card side-by-side comparison:
  1. Original Road Frame
  2. YOLOv8n Detections (bounding boxes, class label, confidence)
  3. DINOv2+SAM2 Anomaly Segmentation (mask-derived bounding box, generic `road_defect` label)
- Pavement state metrics:
  - Current Severity Index ($S_0$): Computed on Experiment A multi-modal captures as a weighted combination of defect area ($m^2$/pixels), estimated depth (dynamically re-weighted in RGB mode), water hazard presence/confidence, and surrounding damage extent. On 2D detection benchmark frames, multi-modal severity is marked N/A.
  - Defect Count
  - Defect Area Ratio (%)
  - Surface Anomaly Score
- Session state continuity: $S_0$ automatically seeds the starting baseline for the Future Condition Forecast page when available.

### 4.2 Scenario-Conditioned Deterioration Forecasting (XGBoost Model V2)
- Exactly matches frozen 6-feature signature: `[current_severity, rainfall_level, traffic_level, temperature, water_exposure, days_ahead]`.
- Implements 5 pre-calibrated scenario presets from observational quantiles:
  - `NORMAL`, `HEAVY_RAIN`, `HEAVY_TRAFFIC`, `HIGH_HEAT`, `WET_EXPOSURE`, plus full `CUSTOM` slider overrides.
- Clearly demarcated output: **STATUS: MODEL-BASED FORECAST**.
- Displays mandatory scientific disclaimer emphasizing observational conditioning over deterministic physical certainty.

### 4.3 Temporal Road Monitoring & Multi-Day Analytics
- Exposes all 4 primary sequence regimes from Experiment A:
  - **SEG_003 D01–D07**: Invariant stability control proving model repeatability (**CV = 0.67%**, mean severity 0.2444, 7-state track retention). Explicitly notes detected region is a systematic false-positive candidate.
  - **SEG_004 D01–D05**: Model-observed severity progression across simulated road states under a fixed camera viewpoint (**Δ = +0.7302**).
  - **SEG_001 D03–D10**: Demonstrates optical sensitivity and false suppression under weather shifts (cloud cover inflating severity to 0.7285, sunset glare suppressing defects).
  - **SEG_004 D06–D10**: Rain puddle reflections (12 false clusters) and sunset glare confounds.
- Comprehensive 5-event taxonomy counters: `NEW_DEFECT` (48), `MATCHED_EXISTING` (33), `AREA_INCREASED` (14), `AREA_DECREASED` (19), `NOT_OBSERVED` (34). Reaffirms `NOT_OBSERVED` is **never marked as REPAIRED**.
- Bipartite tracking algorithm: Evaluated using hierarchical greedy matching: Mask IoU ($\ge 0.50$) $\to$ BBox IoU fallback ($\ge 0.30$) $\to$ Centroid/Area fallback ($\le 75\,\text{px}, \le 3.0\times$).

### 4.4 Perception Benchmark: YOLOv8n vs DINOv2+SAM2
- Renders frozen headline metrics on RDD2022 China_Drone (480 images, 742 GT boxes):
  - YOLOv8n: Precision = 0.6568, Recall = 0.7736, F1 = 0.7104, Mean IoU = 0.8007, Latency = 3.62 ms, FPS = 276.33.
  - DINOv2+SAM2: Precision = 0.0221, Recall = 0.0337, F1 = 0.0267, Mean IoU = 0.6533, Latency = 263.15 ms, FPS = 3.80.
- Contextual integrity callout: Clearly articulates that China_Drone aligns with YOLO's supervised training distribution while DINO/SAM is zero-shot.
- YOLO per-class table: Includes explicit **D40 Low Support Warning** (GT = 15) and highlights **D20 Alligator Cracking** as the weakest class (F1 = 0.4259).
- Curated panel browser: Interactive selector across all 8 verified panel case studies.

### 4.5 Failure Mode & Sensitivity Analysis
- Transparent analysis of DINOv2+SAM2 failures: coarse gravel aggregate false alarms (1,055 unmatched regions), 14×14 px thin crack patch dilution, long crack SAM2 fragmentation, specular puddle reflections, and sunset shadow suppression.
- Transparent analysis of YOLOv8n failures: D20 confusion, roadside soil FPs, worn thermoplastic marking wear, and absence of detections on forward dashcam domain shift (India_005086).
- IoU sensitivity analysis: IoU 0.50 (TP = 25) vs IoU 0.25 (TP = 74), with rigorous explanation that relaxed criteria reveal partial mask overlap rather than true precision parity.

### 4.6 Edge Deployment (Raspberry Pi 5)
- Prominently displays: **RASPBERRY PI 5 EDGE EVALUATION — STATUS: PENDING PHASE 8**.
- Zero fabricated latency, FPS, or power metrics.
- Comprehensive evaluation schema prepared for latency, throughput, RAM, CPU load, and model loading consistency.

---

## 5. Presentation Reliability Verification

Extensive automated and manual stress tests were performed to guarantee fail-safe examiner demonstration:

1. **Triple Cold-Start Benchmark**:
   - Launch #1: Health verified in 0.47s
   - Launch #2: Health verified in 0.43s
   - Launch #3: Health verified in 0.42s
   - *Result*: Flawless startup, zero port binding conflicts, average startup time < 0.45s.
2. **Headless Navigation & Exception Audit (Streamlit AppTest)**:
   - Evaluated all 8 navigation routes programmatically.
   - `Page [1. System Overview]`: 0 exceptions
   - `Page [2. Single-Image Road Assessment]`: 0 exceptions
   - `Page [3. Future Condition Forecast]`: 0 exceptions
   - `Page [4. Temporal Road Monitoring]`: 0 exceptions
   - `Page [5. YOLO vs DINOv2+SAM2]`: 0 exceptions
   - `Page [6. Failure Mode Analysis]`: 0 exceptions
   - `Page [7. Edge Deployment]`: 0 exceptions
   - `Page [8. Research / Methodology]`: 0 exceptions
3. **Session State Continuity**:
   - Single-Image Assessment selected image severity seamlessly populates the baseline slider on the Future Condition Forecast page when available.
4. **Missing Artifact Defensive Fallbacks**:
   - Tested behavior under simulated missing manifests and optional files; data loaders return graceful empty structures without throwing uncaught exceptions.

---

## 6. Known Limitations & Research Boundaries

1. **Benchmark Distribution Bias**: The validation benchmark uses RDD2022 China_Drone, which directly matches YOLOv8n's supervised training domain. DINOv2+SAM2 is zero-shot and penalized on standard bounding box IoU metrics.
2. **Distinct Optical Sensitivities**: DINOv2+SAM2 is vulnerable to specular puddle reflections and sunset shadow gradients confusing heuristic suppression, whereas YOLOv8n is sensitive to road-border terrain textures and camera viewpoint shifts.
3. **Synthetic Deterioration Horizon**: Multi-day temporal progressions (Experiment A) capture simulated road deterioration states rather than multi-year physical weathering, necessitating validation against long-term highway infrastructure datasets.

---

## 7. Phase 8 Readiness Assessment

- **Readiness**: YES.
- All Phase 1–7 artifacts, datasets, and perception/forecasting models are frozen for evaluation as of the Phase-7 commit.
- Edge deployment target specifications and metric schemas are established.
- The repository is completely primed for Phase 8 embedded deployment and on-device profiling on the Raspberry Pi 5 platform.
