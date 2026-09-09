# RoadSentinel — Experiment B Methodology
## Controlled Deterioration, Repair, and Re-Deterioration Lifecycle Evaluation

---

## 1. Executive Summary & Research Goal

Experiment B is designed as RoadSentinel's primary **controlled longitudinal road-health experiment**. While Experiment A evaluated perceptual and temporal robustness across complex environmental confounding (varying sun angles, rain, sunset, and camera vantage shifts), Experiment B isolates the **physical / simulated road-state progression** across a fixed viewpoint under controlled conditions.

### Core Scientific Question:
> *Can the frozen RoadSentinel multi-modal perception + temporal tracking pipeline observe a consistent, monotonic defect lifecycle across controlled repeated inspections, detect scripted maintenance interventions (disappearance of tracked defects upon repair), and monitor subsequent post-repair re-deterioration?*

---

## 2. Experimental Design & Target State Schedule

Experiment B comprises two independent road segments:
- **`SEG_005`**: Primary Controlled Sequence (Curved/Straight Highway Section)
- **`SEG_006`**: Replication Sequence (Independent Highway Section)

Each segment is evaluated over an ordered 10-inspection lifecycle:

```
           PRE-REPAIR PROGRESSION           REPAIR            POST-REPAIR PROGRESSION
   [ Day 01 ──> Day 02 ──> Day 03 ──> Day 04 ──> Day 05 ] ──> [ Day 06 ] ──> [ Day 07 ──> Day 08 ──> Day 09 ──> Day 10 ]
   Pristine     Minor     Moderate    Severe    Critical       Pristine       Minor     Moderate    Severe    Critical
   Grade A     Grade B    Grade C    Grade D    Grade F        Reset A       Grade B    Grade C    Grade D    Grade F
```

### Planned State-by-Day Schedule:
| Inspection | Lifecycle Phase | Simulated Condition State | Defect Density | Target Event |
|---|---|---|---|---|
| **Day 01** | `PRE_REPAIR` | Pristine (Grade A) / Baseline | $\le 2.0 / 100\text{m}^2$ | Baseline unblemished road surface |
| **Day 02** | `PRE_REPAIR` | Minor Wear (Grade B) | $3.0 - 4.0 / 100\text{m}^2$ | Initial hairline fissure onset |
| **Day 03** | `PRE_REPAIR` | Moderate Deterioration (Grade C) | $4.5 - 5.5 / 100\text{m}^2$ | Surface defect widening / cavity growth |
| **Day 04** | `PRE_REPAIR` | Severe Breakdown (Grade D) | $6.0 - 7.5 / 100\text{m}^2$ | Deep cavity formation & crack propagation |
| **Day 05** | `PRE_REPAIR` | Critical Hazard (Grade F) | $8.0 - 10.0 / 100\text{m}^2$ | Maximum pre-repair deterioration |
| **Day 06** | `REPAIR` | Pristine (Grade A) Reset | $0.0 / 100\text{m}^2$ | **Scripted Repair Intervention Reset** |
| **Day 07** | `POST_REPAIR` | Minor Post-Repair Wear (Grade B) | $2.5 - 3.5 / 100\text{m}^2$ | Early recurrence / asphalt fatigue |
| **Day 08** | `POST_REPAIR` | Moderate Recurrence (Grade C) | $4.5 - 6.0 / 100\text{m}^2$ | Recurrent defect expansion |
| **Day 09** | `POST_REPAIR` | Severe Recurrence (Grade D) | $6.5 - 8.0 / 100\text{m}^2$ | Secondary structural degradation |
| **Day 10** | `POST_REPAIR` | Critical Recurrence (Grade F) | $8.5 - 10.0 / 100\text{m}^2$ | Maximum post-repair deterioration |

---

## 3. Strict Experimental Controls

### 3.1 Fixed Camera Viewpoint Requirement
- **Camera Preset**: `🔭 Overhead Drone Survey (SAM 2 Top-Down)` (Nadir angle, $\text{Pitch} \approx -89^\circ$, Altitude $= 25.0\text{m}$).
- **Consistency Constraint**: The camera viewpoint, location, altitude, and orientation must remain strictly identical across all 10 days for each segment. Viewpoint switching within a sequence is forbidden as it introduces geometric perspective confounding.

### 3.2 Environmental Lighting & Moisture Invariance
- **Lighting Preset**: `Clear Noon (70° Sun)` across all 10 days.
- **Moisture State**: `100% Dry Crushed Aggregate` across all 10 days.
- **Rationale**: Isolates damage progression by eliminating shadows, specular glare, wet halo puddles, and low-light feature degradation.

### 3.3 Intervention Metadata Protocol
- Days 01–05: `intervention: "PRE_REPAIR"`
- Day 06: `intervention: "REPAIR"`
- Days 07–10: `intervention: "POST_REPAIR"`

---

## 4. Frozen Perception & Analytics Pipelines

### 4.1 Frozen Perception Models (Zero Retraining / Zero Retuning)
1. **DINOv2 + SAM2 Pipeline**:
   - Backbone: `dinov2_vits14` (frozen ViT-S/14)
   - Segmenter: `sam2.1_hiera_small`
   - Calibration: Marion Day-5 NORMAL production configuration ($\text{RoadMask}$, $\text{RoadMarkingSuppressor}$, heuristic box prompts, unified severity formula).
2. **YOLOv8n Object Detector**:
   - Checkpoint: `yolo/weights/best.pt`
   - Resolution: $512 \times 512$, $\text{conf} = 0.25$.
3. **Phase-8 Reliability Estimator**:
   - Logistic Calibration Head over YOLO confidence, DINOv2 domain familiarity ($k\text{-NN}$ $k=20$), and image contrast/sharpness.

### 4.2 Frozen Hierarchical Temporal Tracker
The greedy one-to-one temporal matching algorithm is applied strictly across consecutive days:
1. **Tier 1**: Mask $\text{IoU} \ge 0.50$
2. **Tier 2**: BBox $\text{IoU} \ge 0.30$
3. **Tier 3**: Centroid Euclidean Distance $\le 75\text{px}$ AND Area Ratio $\le 3.0\times$

### 4.3 Raw Temporal Event Vocabulary
- `NEW_DEFECT`: Candidate region observed with no prior temporal match.
- `MATCHED_EXISTING`: Candidate region matches prior active track.
- `OBSERVED_AREA_INCREASED`: Matched defect exhibits $> 15\%$ surface pixel expansion.
- `OBSERVED_AREA_DECREASED`: Matched defect exhibits $> 15\%$ surface pixel contraction.
- `NOT_OBSERVED`: Active track from $t-1$ not detected at $t$.

---

## 5. Repair Interpretation & Lifecycle Handoff Rules

1. **The Repair Ingestion Rule (Day 05 $\rightarrow$ Day 06)**:
   - When active tracks from Day 05 are absent on Day 06, the temporal tracker outputs raw `NOT_OBSERVED` events.
   - The research interpretation combines the raw event with the authoritative simulation metadata: *"NOT OBSERVED FOLLOWING SCRIPTED REPAIR INTERVENTION"*.
   - The raw event vocabulary is preserved; causal terminology (`REPAIRED`) is applied at the lifecycle reporting layer only.

2. **Post-Repair Recurrence Rule (Day 06 $\rightarrow$ Day 10)**:
   - Defects appearing on Day 07 are ingested as `NEW_DEFECT` (or `NEW_POST_REPAIR_DEFECT` in lifecycle aggregation) unless physical spatial overlap with pre-repair coordinates supports spatial recurrence.

---

## 6. Statistical & Methodological Limitations

1. **Sample Size ($N=5$ per sub-sequence)**:
   - Monotonicity is measured via Spearman rank correlation $\rho_s$. With $N=5$, rank correlation provides descriptive monotonicity rather than high-power statistical significance.
2. **Simulated Ground Truth**:
   - Deterioration rates reflect Unreal Engine procedural material blending and mesh displacement, not empirical multi-year asphalt mechanics.
3. **Absence vs True Disappearance**:
   - An unobserved region reflects detector thresholding under current perception heuristics. Metadata confirms physical repair.
