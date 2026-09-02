# RoadSentinel — Road Health Scoring, Severity Estimation & Deterioration Prediction

## 1. Overview
The RoadSentinel Road-Health & Analytics Pipeline operates directly on the output of the trained DINOv2 + SAM2 defect detection layer. It converts raw segmented anomaly masks into traceable physical metrics (area $m^2$, depth, water presence, crack extent), calculates transparent and explainable defect severity and segment-level road health scores ($0\text{--}100$), and provides a modular temporal forecasting interface for road deterioration and pothole formation.

---

## 2. Scientific Status
In accordance with rigorous scientific standards, all components are transparently categorized:

| Module / Component | Scientific Status | Description / Validation Basis |
| :--- | :--- | :--- |
| **Area Estimation** | `IMPLEMENTED` | Ground-projected nadir camera ray tracing calibrated by altitude & FOV. |
| **Depth Evaluation** | `IMPLEMENTED` | Standardized MAE, RMSE, AbsRel, $\delta$-thresholds against ground truth. |
| **Defect Classification** | `IMPLEMENTED` | Morphological compactness, aspect ratio, solidity & spectral water cues. |
| **Severity Breakdown** | `IMPLEMENTED` | Multi-criteria explainable formulation with graceful missing-depth reweighting. |
| **Road Health Scoring** | `IMPLEMENTED` | Segment-level 0–100 index with calibrated penalties & condition classes. |
| **Segment Aggregation** | `IMPLEMENTED` | Geospatial grid binning retaining 100% individual defect traceability. |
| **CARLA GT Evaluation** | `CARLA-SYNTHETIC ONLY`| Controlled synthetic validation against simulated road geometries. |
| **Deterioration Model** | `CARLA-SYNTHETIC ONLY`| Temporal degradation & hazard forecasting evaluated on leak-free synthetic sequences. |
| **Real-World Forecasting** | `REQUIRES REAL DATA` | Real-world validation requires repeated longitudinal multi-month drone surveys. |

---

## 3. Mathematical Formulations

### 3.1 Defect Severity Estimation
For an individual detected defect $i$, the severity score $S_i \in [0, 100]$ is computed as:
$$S_i = c_i \cdot \frac{\sum_{k \in \mathcal{K}_{\text{avail}}} w_k S_{i,k}}{\sum_{k \in \mathcal{K}_{\text{avail}}} w_k}$$

Where:
- $S_{i, \text{area}} = \min\left(100, \frac{\text{Area}_i}{\text{Area}_{\text{high}}} \times 100\right)$
- $S_{i, \text{depth}} = \min\left(100, \frac{\text{Depth}_i}{\text{Depth}_{\text{high}}} \times 100\right)$ (omitted and re-weighted when depth is unavailable)
- $S_{i, \text{water}} = \text{WaterConfidence}_i \times 100 + \text{Bonus}_{\text{water}}$
- $S_{i, \text{damage}} = \min\left(100, \frac{\text{CrackExtent}_i}{1.0\text{m}} \times 100\right)$
- $c_i = 0.5 + 0.5 \times \text{Confidence}_i$

Qualitative mapping:
- **Low**: $S_i < 25$
- **Medium**: $25 \le S_i < 50$
- **High**: $50 \le S_i < 75$
- **Critical**: $S_i \ge 75$

### 3.2 Segment Road-Health Score
The segment road health score $H \in [0, 100]$ evaluates a road segment (where $100$ is pristine and $0$ is impassable):
$$H = \max\left(0, 100.0 - \left(P_{\text{pothole}} + P_{\text{crack}} + P_{\text{water}} + P_{\text{surface}}\right)\right)$$

Where penalties are capped by configurable limits:
- $P_{\text{pothole}} = \min\left(P_{\text{max, pothole}}, N_{\text{potholes}} \times 8.0 + \sum \frac{S_i}{100} \times 15.0\right)$
- $P_{\text{crack}} = \min\left(P_{\text{max, crack}}, N_{\text{cracks}} \times 4.0 + \text{Area}_{\text{cracks}} \times 15.0\right)$
- $P_{\text{water}} = \min\left(P_{\text{max, water}}, N_{\text{water}} \times 7.5\right)$
- $P_{\text{surface}} = \min\left(P_{\text{max, surface}}, N_{\text{surface}} \times 2.5\right)$

Condition Classes:
- **Good**: $80 \le H \le 100$ (Optimal condition; routine monitoring)
- **Fair**: $60 \le H < 80$ (Minor wear/cracking; scheduled maintenance recommended)
- **Poor**: $40 \le H < 60$ (Significant pothole/structural hazards; high priority repair)
- **Critical**: $H < 40$ (Hazardous structural failure; immediate road closure/emergency repair)

---

## 4. Temporal Deterioration & Pothole Formation Forecasting

### 4.1 Feature Extraction
For road segment $s$ with inspection history $\mathcal{H}_s = \{(t_0, H_0), \dots, (t_n, H_n)\}$, the model calculates:
- Health velocity: $v_H = \frac{H_n - H_0}{t_n - t_0}$ (points/day)
- Damaged area growth rate: $g_A = \frac{\Delta \text{Area}}{\Delta t}$ ($m^2$/day)
- Water exposure ratio: $r_{\text{water}} = \frac{1}{n} \sum \mathbb{I}(\text{water present})$

### 4.2 Prediction Hazard Output
- **Deterioration Probability**: $P(\Delta H > 10\text{ pts within horizon } T) = \sigma\left(\frac{-v_H \cdot T \cdot (1 + 1.2 r_w) - \theta_{\text{det}}}{6.0}\right)$
- **Pothole Formation Probability**: $P(\text{Pothole emergence}) = \sigma\left(0.4 N_{\text{crack}} + 8.0 g_A \cdot T + 1.5 r_w - 1.2\right)$

---

## 5. Standard Output Schema

```json
{
  "image_id": "IMG_2026-08-30T12:00:00Z",
  "timestamp": "2026-08-30T12:00:00Z",
  "road_segment_id": "SEG_X+002_Y+006",
  "geolocation": {
    "lat": 13.0827,
    "lon": 80.2707
  },
  "image_shape": [720, 1280, 3],
  "anomaly_threshold": 98.0,
  "anomaly_score": 0.87,
  "detections": [
    {
      "defect_id": "DEF_2026-08-30T12:00:00Z_000",
      "defect_type": "water_filled_pothole",
      "confidence": 0.95,
      "bbox": [695, 365, 805, 475],
      "mask_area_pixels": 9503,
      "estimated_area_m2": 0.5842,
      "estimated_depth_m": 0.08,
      "is_water_filled": true,
      "water_confidence": 0.92,
      "crack_or_damage_extent": null,
      "road_segment_id": "SEG_X+002_Y+006",
      "timestamp": "2026-08-30T12:00:00Z",
      "latitude": 13.0827,
      "longitude": 80.2707,
      "severity": {
        "severity": "critical",
        "severity_score": 88.4,
        "severity_components": {
          "area": 100.0,
          "depth": 80.0,
          "water": 100.0,
          "surrounding_damage": 0.0
        }
      },
      "depth_source": "unavailable",
      "notes": []
    }
  ],
  "road_health": {
    "road_health_score": 42.5,
    "condition_class": "Poor",
    "components": {
      "pothole_penalty": 38.5,
      "crack_penalty": 11.5,
      "water_penalty": 7.5,
      "surface_penalty": 0.0
    },
    "confidence": 0.95,
    "explanation": "Condition: Poor (42.5/100). 1 pothole(s) (-38.5 pts); 1 water puddle hazard(s) (-7.5 pts)"
  },
  "prediction": {
    "deterioration_probability": 0.785,
    "pothole_formation_probability": 0.642,
    "prediction_horizon_days": 30,
    "progression_trend": "deteriorating",
    "scientific_status": "CARLA-SYNTHETIC ONLY",
    "features_used": {
      "latest_health": 42.5,
      "health_velocity": -0.65,
      "area_growth_rate": 0.015,
      "severity_velocity": 1.2
    },
    "notes": []
  }
}
```

---

## 6. Generated Visual Artifacts

The pipeline automatically renders diagnostic visual overlays saved under `outputs/`:
1. `outputs/detection_overlay.jpg`: Defect bounding boxes, masks, defect type tags, and areas.
2. `outputs/severity_overlay.jpg`: Color-coded severity heatmaps (Low = Green, Medium = Yellow, High = Orange, Critical = Red).
3. `outputs/road_health_overlay.jpg`: Segment HUD overlay featuring 0-100 health gauge, condition status, penalty breakdown, and 30-day forecast.
4. `outputs/result.json`: Fully structured, traceable JSON record.
