# RoadSentinel Decision Engine Methodology: Reliability-Aware Road-Health Priority Framework

**Document**: `decision_engine/DECISION_ENGINE_METHODOLOGY.md`  
**Phase**: 13 (System Integration & Decision Layer)  
**Status**: **COMPLETE**

---

## 1. Executive Overview & Scope

The RoadSentinel Decision Engine integrates all preceding research modules—visual perception, macro domain gating, sample-level reliability estimation, model-observed temporal tracking, and scenario-conditioned forecasting—into a transparent, deterministic decision framework.

The primary objective is to answer:
1. **What is the current road condition?** (Current Model Assessment)
2. **How has it changed over time?** (Model-Observed Temporal Change)
3. **What does the model forecast under future scenarios?** (Model-Based Forecast)
4. **Can the perception result be trusted?** (DINOv2 Domain Gate + YOLO Reliability)
5. **What inspection action should be recommended?**

> [!IMPORTANT]
> **Operational Scope Limitation**: The Decision Engine outputs *inspection and review priorities* (`AUTOMATED_ACCEPT`, `MONITOR`, `REINSPECT`, `PRIORITY_REVIEW`, `DOMAIN_ESCALATION`). It does **not** prescribe physical civil-engineering pavement maintenance or compute standardized pavement condition ratings (such as ASTM D6433 PCI).

---

## 2. Four Decoupled Evidence Streams

To maintain scientific integrity and prevent invalid feature conflation, the engine keeps four evidence streams strictly separated:

```
[Raw Image + Metadata]
       │
       ├── 1. DOMAIN & PERCEPTION RELIABILITY
       │      ├── DINOv2 Domain Familiarity (kNN distance d_kNN vs p99=0.4491)
       │      └── YOLOv8n Confidence-Based Reliability (Model B: HIGH / MED / LOW)
       │
       ├── 2. CURRENT MODEL ASSESSMENT
       │      └── Severity, Defect Count, Defect Area Ratio, Surface Anomaly
       │
       ├── 3. MODEL-OBSERVED TEMPORAL CHANGE
       │      └── Same-Camera State Delta, Matched Defect Tracks, Persistence Ratio
       │
       └── 4. MODEL-BASED SCENARIO FORECAST
              └── XGBoost 90-Day Projections (Normal, Rain, Traffic, Heat, Wet Exposure)
```

**Non-Multiplication Constraint**: Reliability scores are **never** multiplied by or used to scale severity scores (e.g. no $\text{severity} \times \text{reliability}$). Reliability determines the *action/trust tier*, while severity reports the *damage estimate*.

---

## 3. Deterministic Decision Hierarchy

Each road inspection observation is evaluated through a strict, deterministic rule precedence:

| Tier | Priority Rule | Condition | Assigned Decision | Explanation / Rationale |
|:---:|---|---|:---:|---|
| **1** | **Domain Gate Rule** | $\text{Raw DINO Distance} > 0.4491$ ($p_{99}$) on real capture | `DOMAIN_ESCALATION` | Intercepts visual distribution shift before perception output can be mistakenly accepted. |
| **1b** | **Metadata Diagnostic Rule** | $\text{Metadata Status} == \text{MISSING}$ (SEG_003 Day 10) | `MONITOR` | Preserves pipeline continuity under fallback parameters while documenting metadata limitation. |
| **2** | **Low Reliability Severe Rule** | $\text{Reliability} < 0.60$ ($\text{LOW}$) $\land \text{Severity} > 0.50$ | `PRIORITY_REVIEW` | High apparent distress requires urgent human verification despite perceptual ambiguity. |
| **2b** | **Low Reliability Moderate Rule**| $\text{Reliability} < 0.60$ ($\text{LOW}$) $\land \text{Severity} \le 0.50$ | `REINSPECT` | Perceptual uncertainty quarantines observation for sensor reinspection. |
| **3** | **High Severity Confident Rule** | $\text{Severity} > 0.50 \land \text{Reliability} \ge 0.60$ | `PRIORITY_REVIEW` | Confident detector agreement on severe surface breakdown. |
| **4** | **Rapid Temporal Growth Rule** | $\text{Trend} == \text{INCREASING} \land \Delta \text{Severity} > 0.15$ | `PRIORITY_REVIEW` | Rapid model-observed damage progression across consecutive inspections. |
| **5** | **Moderate Severity / Forecast Rule** | $\text{Severity} \in [0.20, 0.50] \lor \Delta_{\text{Forecast}} > 0.20$ | `MONITOR` | Moderate distress or high scenario-conditioned vulnerability requires ongoing surveillance. |
| **6** | **Pristine Stable Confident Rule** | $\text{Severity} < 0.20 \land \text{Trend} == \text{STABLE} \land \text{Reliability} \ge 0.85$ | `AUTOMATED_ACCEPT` | Invariant response, low severity, and high reliability justify automated acceptance. |

---

## 4. Empirical Threshold Derivations

All thresholds are derived empirically from development and benchmark data:
1. **DINOv2 Domain Thresholds**: Derived from China development embeddings ($N=1921$):
   - $p_{95} = 0.3804$ (`DOMAIN_WARNING`)
   - $p_{99} = 0.4491$ (`EXTREME_DOMAIN_SHIFT`)
2. **Perception Reliability Bands**: Derived from Phase 8/12 logistic calibration on out-of-fold validation predictions:
   - $\text{HIGH} \ge 0.85$
   - $\text{MEDIUM} \in [0.60, 0.85)$
   - $\text{LOW} < 0.60$
3. **Current Severity Descriptive Bands**:
   - `LOW_MODEL_SEVERITY`: $\text{Severity} < 0.20$
   - `MODERATE_MODEL_SEVERITY`: $0.20 \le \text{Severity} \le 0.50$
   - `HIGH_MODEL_SEVERITY`: $\text{Severity} > 0.50$
4. **Forecast Delta Bands**:
   - `LOW_FORECAST_CHANGE`: $\Delta_{90d} < 0.05$
   - `MODERATE_FORECAST_CHANGE`: $0.05 \le \Delta_{90d} \le 0.20$
   - `HIGH_FORECAST_CHANGE`: $\Delta_{90d} > 0.20$
