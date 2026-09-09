# RoadSentinel — Examiner Demonstration Guide

This guide provides a standardized, chronological script for demonstrating the RoadSentinel research pipeline to examiners and review panels. The walkthrough highlights technical strengths while maintaining strict scientific transparency.

---

## 0. Pre-Flight Setup

Launch the dashboard from the repository root:
```bash
.venv/bin/streamlit run dashboard/app.py
```
*Expected Startup*: Browser automatically opens at `http://localhost:8501` within 0.5 seconds in **Verified Demo Mode**.

---

## 1. System Overview (Landing Page)
1. **Navigate to**: `1. System Overview` (Sidebar default).
2. **Key Talking Points**:
   - Introduce RoadSentinel as an integrated dual-perception and multi-scale deterioration intelligence system.
   - Point out the **Pipeline Status Badges**: Perception and Temporal stages are locked and frozen; Edge Deployment is clearly marked as *Pending Phase 8*.
   - Walk through the **End-to-End System Architecture** diagram:
     - Optical imagery passes simultaneously to a supervised fast detector (YOLOv8n) and an unsupervised foundation anomaly segmentor (DINOv2 + SAM2).
     - Single-frame outputs feed into both multi-day temporal tracking and scenario-conditioned deterioration forecasting (XGBoost Model V2).
   - Point out the **Empirical Findings at a Glance** cards: In-domain F1 of 0.7104 vs 0.0267, 72.7× latency advantage, and 0.67% control stability CV.

---

## 2. Single-Image Inspection Selection
1. **Navigate to**: `2. Single-Image Road Assessment`.
2. **Action**: Open the *Select Road Inspection Sample* dropdown.
3. **Select**: `China_Drone_001063 — both_success`.
4. **Key Talking Points**:
   - Explain that this frame contains a prominent transverse crack evaluated on the common validation benchmark.
   - Note the **Verified Demo Mode** setting, ensuring instant deterministic rendering.

---

## 3. Compare YOLOv8n Detections
1. **Action**: Direct attention to Card 2 (*YOLOv8n Detections*).
2. **Key Talking Points**:
   - Point out the tight rectangular bounding box labeled `D10 0.82` (transverse crack).
   - Emphasize that YOLO directly classifies damage type and predicts coordinates in just **3.6 ms** on the RTX 5060 GPU.

---

## 4. Compare DINOv2 + SAM2 Zero-Shot Segmentation
1. **Action**: Direct attention to Card 3 (*DINOv2 + SAM2 Generic Defect*).
2. **Key Talking Points**:
   - Point out the mask-derived bounding box labeled `road_defect`.
   - **Crucial Scientific Distinction**: Clarify to the examiner that DINOv2+SAM2 does **not** predict supervised CRDDC classes (`D00`, `D10`, etc.); it segments geometric anomalies relative to a healthy asphalt memory bank without damage-specific training labels.

---

## 5. Review Current Severity Metrics
1. **Action**: Review the *Pavement State Metrics* cards below the images.
2. **Key Talking Points**:
   - Explain the **Current Severity Index**: On multi-modal inspection captures (Experiment A), severity is computed per defect as a weighted combination of defect area ($m^2$/pixels), estimated depth (dynamically re-weighted in RGB mode), water hazard presence/confidence, and surrounding damage extent. On 2D detection benchmark frames evaluated solely on bounding boxes, multi-modal severity is marked N/A.
   - Mention that measured severity values are preserved in session state to initialize future condition forecasts.

---

## 6. Configure Future Forecast Scenario
1. **Action**: Navigate to `3. Future Condition Forecast`.
2. **Key Talking Points**:
   - Show that the starting baseline severity ($S_0$) is automatically pre-populated from the single-image assessment (or adjustable via slider).
   - Switch the *Scenario Preset* dropdown from `NORMAL` to `HEAVY_RAIN` (or `HIGH_HEAT`).
   - Note how precipitation, traffic, and temperature sliders adjust to match empirically grounded 90th-percentile stressor thresholds.
   - Set the *Forecast Horizon* slider to **60 Days** or **90 Days**.

---

## 7. Evaluate Model-Based XGBoost Forecast
1. **Action**: Scroll to Section 3 (*Model-Based Forecast Output*).
2. **Key Talking Points**:
   - Emphasize the headline label: **STATUS: MODEL-BASED FORECAST**.
   - Review the Baseline Severity, Projected Future Severity, and estimated change ($\Delta$).
   - Read or highlight the **Mandatory Scientific Disclaimer**:
     > *"The forecast is scenario-conditioned and derived from observational training data; it should not be interpreted as a deterministic physical deterioration prediction."*
   - Mention that the model incorporates **monotone directional constraints** (`monotone_constraints: (1,1,1,1,1,1)`) ensuring tree predictions do not project lower deterioration under increased stress levels or longer forecast horizons.

---

## 8. Open Temporal Road Monitoring
1. **Action**: Navigate to `4. Temporal Road Monitoring`.
2. **Key Talking Points**:
   - Introduce the multi-day inspection evaluation from Experiment A (40 captures across 4 sequences).
   - Review the **Temporal Event Taxonomy** cards: `NEW_DEFECT` (48), `MATCHED_EXISTING` (33), `AREA_INCREASED` (14), `AREA_DECREASED` (19), and `NOT_OBSERVED` (34).
   - Reiterate that `NOT_OBSERVED` is **never fabricated as REPAIRED** without maintenance metadata.
   - Explain the **One-to-One Hierarchical Matching Algorithm**: Mask IoU ($\ge 0.50$) $\to$ BBox IoU fallback ($\ge 0.30$) $\to$ Centroid/Area fallback ($\le 75\,\text{px}, \le 3.0\times$).

---

## 9. Demonstrate Progression vs Stability Sequences
1. **Action A (Progression)**: Select `SEG_004_D01_D05 — Monotonic Progression Sequence`.
   - Show the model-observed severity progression across simulated road states under a fixed camera viewpoint (`0.0000` to `0.7302`, $\Delta = +0.7302$).
   - Point out Figure 2 showing defect area expansion across successive states.
2. **Action B (Stability Control)**: Switch to `SEG_003_D01_D07 — Stability Control Test`.
   - Highlight the **0.67% Coefficient of Variation** across 7 consecutive observation states under invariant lighting.
   - Explain the research takeaway: the perception pipeline has outstanding repeatability when environmental conditions are controlled, though the detected region is a systematic false-positive candidate on the road shoulder.

---

## 10. Examine Benchmark Comparison & Research Balance
1. **Action**: Navigate to `5. YOLO vs DINOv2+SAM2`.
2. **Key Talking Points**:
   - Review the Primary Performance Table:
     - YOLOv8n: F1 = **0.7104**, Latency = **3.62 ms** (276 FPS).
     - DINOv2+SAM2: F1 = **0.0267**, Latency = **263.15 ms** (3.8 FPS).
   - State the **Scientific Context**:
     - The common validation benchmark is RDD2022 China_Drone (supervised training domain of YOLO).
     - DINO/SAM is zero-shot and has no concept of crack classification standards.
   - Point out the **D40 Low Support Warning**: Pothole ground truth has only 15 instances, precluding strong class-level conclusions.
   - Point out **D20 Alligator Cracking**: Weakest meaningful YOLO class (F1 = 0.4259).

---

## 11. Transparent Failure Mode Discussion & Conclude
1. **Action**: Navigate to `6. Failure Mode Analysis`.
2. **Key Talking Points**:
   - Highlight that RoadSentinel explicitly documents failure modes rather than concealing them.
   - Discuss **DINOv2 coarse gravel aggregate false positives** and **14×14 px thin crack patch dilution**.
   - Walk through the **IoU Sensitivity Analysis**:
     - Relaxing IoU from 0.50 to 0.25 increases DINO/SAM true positives from **25 to 74**.
     - Emphasize: *This indicates partial spatial overlap of organic mask boundaries, NOT that DINO/SAM is suddenly accurate.*
3. **Action (Edge Readiness)**: Briefly show `7. Edge Deployment`.
   - Confirm that Raspberry Pi 5 evaluation is visibly marked as **PENDING PHASE 8**, with zero fabricated metrics.
4. **Conclusion**:
   - RoadSentinel delivers an honest, rigorous, and fully reproducible perception and deterioration intelligence platform.
