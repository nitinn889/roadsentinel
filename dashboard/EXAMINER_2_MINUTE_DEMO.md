# RoadSentinel — Examiner 2-Minute Rapid Review Script

This concise 2-minute demonstration is designed for high-impact executive reviews, short conference demos, and quick committee evaluations.

---

## 2-Minute Quick-Fire Demonstration

| Time | Dashboard Page | Key Action & Talking Point |
|---|---|---|
| **0:00 – 0:30** | `1. System Overview` | **Multi-Stream Architecture**: Show the 6-tier pipeline connecting Macro Domain Gating (DINOv2), Supervised Perception (YOLOv8n), Sample Reliability, Temporal Tracking, and XGBoost Forecasting into an explainable Decision Engine. |
| **0:30 – 0:55** | `2. Single-Image Assessment` | **CURRENT MODEL ASSESSMENT**: Select `SEG_004 Day 01` (Severity 0.0000 $\to$ `MONITOR`), then switch to `SEG_004 Day 05` (Severity 0.7302 $\to$ `PRIORITY_REVIEW`). Point out the decoupled reliability and explainable decision card. |
| **0:55 – 1:20** | `5. Temporal Change Analysis` | **MODEL-OBSERVED CHANGE**: Open `SEG_004 D01-D05` progression ($\Delta = +0.7302$) and `SEG_003 D01-D07` control ($\text{CV} = 0.67\%$). Highlight canonical tracking: 14 Area Increased, 19 Area Decreased, 34 Not Observed. |
| **1:20 – 1:45** | `4. XGBoost Future Forecast` | **MODEL-BASED FORECAST**: Show 90-day multi-scenario forecast. Point out `WET_EXPOSURE` as dominant degradation stressor ($\Delta +0.0312$). Mention LTPP training baseline and monotone constraints. |
| **1:45 – 2:00** | `7. Cross-Domain Generalization` | **Domain Shift & Safety Quarantine**: Show $-96.9\%$ YOLO collapse on Indian dashcams, DINOv2 perfect separation ($\text{AUROC} = 1.0000$), and 100% quarantine ($300/300$ routed to `DOMAIN_ESCALATION`). |

---

## Core Soundbites for Reviewers
1. *"RoadSentinel does not rely on a single fallible detector; it evaluates four decoupled evidence streams before committing an inspection recommendation."*
2. *"DINOv2 serves as a macro domain gate rather than an unguided detector, achieving AUROC 1.0000 on evaluated cross-domain transfer."*
3. *"100% of evaluated cross-domain perception failures were safely quarantined away from automated acceptance."*
