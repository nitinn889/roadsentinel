# RoadSentinel — Road Health Scoring, Defect Analytics & Deterioration Prediction

Post-inference analytics and degradation prediction layer for RoadSentinel. Turns raw DINOv2 + SAM2 defect detections into measurable physical properties, transparent multi-factor severity ratings, an explainable 0–100 segment-level Road Health Index, CARLA ground-truth evaluation metrics, and a longitudinal deterioration prediction baseline.

```
RGB image → DINOv2 patch features → Healthy-road memory bank → Anomaly map
         → Candidate regions → SAM2 segmentation → Defect masks
         → Area & Depth measurements → Multi-factor Severity Breakdown
         → 0–100 Road Health Scoring → Segment Spatial/Temporal Aggregation
         → Temporal Deterioration & Pothole Prediction → Structured JSON + Overlays
```

---

## Scientific Status

| Component | Status | Details |
|---|---|---|
| **DINOv2 Anomaly Detection** | **IMPLEMENTED** | ViT-S/14 patch embeddings + FAISS cosine distance kNN |
| **Defect Classification & Filtering** | **IMPLEMENTED** | Distinguishes `pothole`, `water_filled_pothole`, `crack_or_damage`, `unknown_road_anomaly` + shadow suppression |
| **SAM2 Prompting & Refinement** | **IMPLEMENTED** | Bounding box prompt refinement into binary masks |
| **Area Estimation** | **IMPLEMENTED** | Closed-form ground projection from altitude & camera FOV |
| **Depth Estimation** | **PROTOTYPE** | NullDepthEstimator default; metric RGB depth requires calibrated mono-depth model |
| **Multi-Factor Severity Scoring** | **IMPLEMENTED** | Configurable, explainable continuous [0, 1] score & qualitative classes with component breakdown |
| **0–100 Road Health Index** | **IMPLEMENTED** | Segment-level score (100 = healthy, 0 = hazardous) with explicit penalty decomposition |
| **Road Segment Aggregation** | **IMPLEMENTED** | Geospatial / segment-ID grouping retaining 100% defect-level traceability |
| **CARLA Ground-Truth Evaluation** | **IMPLEMENTED** | Error metrics for area MAE, depth RMSE, 3D location, and water classification |
| **Temporal Progression Simulation** | **CARLA-SYNTHETIC ONLY** | Synthetic longitudinal wear sequences ($t_1 \to t_4$) for controlled model benchmarking |
| **Deterioration Prediction Model** | **PROTOTYPE** | Random Forest baseline on rate-of-change features + rule-based heuristic fallback |
| **Real Longitudinal Validation** | **REQUIRES REAL DATA** | Longitudinal real-world drone inspection datasets required for calibrated field deployment |

---

## 1. Defect Severity Formulation

Severity is computed per defect as an explainable weighted combination of physical attributes:

$$\text{Severity Score} = \frac{\sum w_i \cdot c_i}{\sum w_i} + \Delta_{\text{water}}$$

- **Confidence ($c_{\text{conf}}$)**: Detection confidence $[0, 1]$ ($w = 0.30$)
- **Area ($c_{\text{area}}$)**: $\min(1.0, \text{Area}_{m^2} / 2.0)$ ($w = 0.25$)
- **Depth ($c_{\text{depth}}$)**: $\min(1.0, \text{Depth}_m / 0.15)$ ($w = 0.20$)
- **Water Hazard ($c_{\text{water}}$)**: Water pooling indicator $[0, 1]$ ($w = 0.15$)
- **Surrounding Damage ($c_{\text{damage}}$)**: Pavement fatigue / cracking variance $[0, 1]$ ($w = 0.10$)

Qualitative classification:
- **Low**: $[0.00, 0.35)$
- **Medium**: $[0.35, 0.65)$
- **High**: $[0.65, 0.85)$
- **Critical**: $[0.85, 1.00]$

---

## 2. Road Health Score (0–100 Index)

$$\text{Road Health Score} = 100.0 - \left( P_{\text{count}} + P_{\text{severity}} + P_{\text{crack}} + P_{\text{water}} + P_{\text{surface}} \right)$$

- **Pothole Count Penalty ($P_{\text{count}}$)**: $\text{Weight (25.0)} \times \min(1.0, \text{Count} / 5)$
- **Pothole Severity Penalty ($P_{\text{severity}}$)**: $\text{Weight (30.0)} \times (0.65 \times \text{MaxSev} + 0.35 \times \text{MeanSev})$
- **Crack Extent Penalty ($P_{\text{crack}}$)**: $\text{Weight (20.0)} \times \min(1.0, \text{CrackArea}_{m^2} / 5.0)$
- **Water Hazard Penalty ($P_{\text{water}}$)**: $\text{Weight (15.0)} \times \min(1.0, \text{WaterPotholes} / 2)$
- **Surface Roughness Penalty ($P_{\text{surface}}$)**: $\text{Weight (10.0)} \times \min(1.0, \text{MeanAnomaly} \times 2.0)$

Condition classes:
- **Good**: $80.0 - 100.0$
- **Fair**: $60.0 - 79.9$
- **Poor**: $40.0 - 59.9$
- **Critical**: $0.0 - 39.9$

---

## 3. Deterioration Prediction Architecture

The prediction interface accepts longitudinal segment histories:
1. **Temporal Feature Extractor**: Computes rates of change ($\Delta \text{Health} / \Delta t$, $\Delta \text{Area} / \Delta t$, $\Delta \text{Severity} / \Delta t$).
2. **Machine Learning Baseline**: Random Forest model trained on synthetic CARLA sequences without segment identity leakage between train/test.
3. **Outputs**:
   - `deterioration_probability`: Risk of significant pavement condition drop over horizon (e.g. 30 days).
   - `pothole_formation_probability`: Likelihood of crack/wear developing into a structural pothole.
   - `progression_direction`: `stable`, `improving`, `degrading`, or `critical`.

---

## 4. Running the Analytics E2E Demo

```bash
# Run end-to-end analytics demo generating JSON and visualization overlays:
python scripts/run_analytics_e2e.py
```

Outputs generated in `output/analytics_demo/`:
- `result.json`: Comprehensive segment summary with full defect traceability
- `detection_overlay.jpg`: Defect boxes with classification and confidence
- `severity_overlay.jpg`: Color-coded severity masks
- `road_health_overlay.jpg`: Segment health index banner
- `work_orders.json`: Automated text-based repair work orders generated via VLM

---

## 5. Automated VLM Work Order Generation

The `vlm_work_order_gen.py` module ingests analytics results, filters critical and high-severity defects, matches visual defect bounding regions, and prompts a Vision-Language Model to draft actionable 3-sentence municipal maintenance work orders.

```bash
# Generate repair work orders from analytics demo outputs:
python vlm_work_order_gen.py \
    --result-json output/analytics_demo/result.json \
    --output output/analytics_demo/work_orders.json
```

---

## 6. Running Automated Tests

```bash
pytest tests/ -v
```

94 unit and integration tests covering:
- VLM prompt construction, PIL visual matching, severity filtering, and JSON export
- Defect severity calculation and rebalancing
- 0-100 road health scoring and penalty contributions
- Segment aggregation and geospatial ID generation
- Temporal feature extraction and deterioration prediction
- CARLA simulation ground-truth metrics
- Edge cases (no defects, missing depth, missing GPS, low confidence)

---

## 7. Git Branch & Safety

All work is committed to branch `feature/vlm-work-orders`:

```bash
git checkout feature/vlm-work-orders
git push -u origin feature/vlm-work-orders
```


