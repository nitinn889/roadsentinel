# RoadSentinel Phase 14: Dashboard V2 & Canonical Research Integration Report

**Document**: `dashboard/PHASE14_DASHBOARD_V2_REPORT.md`  
**Phase**: 14 (Dashboard V2 + System-Wide Consistency Audit)  
**Status**: **COMPLETE & VERIFIED**  
**Date**: September 2026  

---

## 1. Executive Summary

Phase 14 delivers the unified, offline-first **Dashboard V2** platform alongside a system-wide consistency audit across all completed phases (Phases 4 through 13). The upgraded dashboard provides an interactive interface for examiners, researchers, and transportation authorities to explore the complete multi-modal RoadSentinel architecture without requiring live Unreal Engine connections or external network access.

### Key Milestones Achieved
1. **System-Wide Consistency Audit**: Reconciled historical numerical nuances across all project artifacts (e.g., India failure counts under Target $T_0$ vs $T_1$, canonical 14/19 temporal event totals, greedy one-to-one tracking hierarchy, and LTPP XGBoost constraints).
2. **Canonical Metrics Single Source of Truth**: Created `integration/CANONICAL_RESEARCH_METRICS.json` and `integration/RESEARCH_CLAIM_REGISTRY.md` to prevent discrepancies across documentation and UI rendering.
3. **Multi-Page Dashboard V2 Architecture**: Expanded from an 8-page prototype to a comprehensive 12-section research interface incorporating the Decoupled Road-Health Decision Engine, Two-Tier Reliability Architecture, Cross-Domain Generalization, and Multi-Day Temporal Tracking.
4. **Complete Capture Access**: Enabled seamless inspection across all 40 Experiment A captures (`SEG_001` through `SEG_004`, Days 01–10) with explicit accounting for temporal ineligibility reasons (e.g., `VIEWPOINT_CHANGE`, `INSUFFICIENT_CONTIGUOUS_STATES`).
5. **Decoupled Evidence Integrity**: Ensured all four primary evidence streams (**Current Model Assessment**, **Model-Observed Temporal Change**, **Scenario-Conditioned Forecasts**, and **Perception Reliability**) remain strictly unmingled and explainable.

---

## 2. Dashboard V2 Section Breakdown

| Section | Component File | Key Data Sources | Core Capabilities & Visualizations |
|---|---|---|---|
| **1. System Overview** | `components/overview.py` | `CANONICAL_RESEARCH_METRICS.json` | 6-tier architecture flow, real-time status badges, 4 headline metric cards, 7 examiner research takeaways. |
| **2. Single-Image Assessment** | `components/single_image.py` | `ROAD_HEALTH_DECISIONS.csv`, `ROADSENTINEL_PRIMARY_RESULTS.csv` | All 40 physical captures accessible, Current Model Assessment metrics, optional live YOLOv8n GPU inference, explainable decision card. |
| **3. Domain & Reliability** | `components/reliability_view.py` | `table_full_ablation.csv`, `table_domain_gate.csv` | Two-tier safety framework, DINO domain separation (AUROC 1.0000), Model B confidence calibration, selective prediction risk-coverage curves. |
| **4. XGBoost Future Forecast** | `components/forecast.py` | `goal1_forecasts.csv` (600 records) | 30/60/90-day horizon forecasts across 5 environmental scenarios (`NORMAL`, `HIGH_HEAT`, `HEAVY_TRAFFIC`, `HEAVY_RAIN`, `WET_EXPOSURE`). |
| **5. Temporal Change Analysis** | `components/temporal.py` | `table_temporal_sequences.csv`, `temporal_events.json` | All 8 evaluated sequences (33 transitions, 48 tracks), stability control ($\text{CV}=0.67\%$), drone progression ($\Delta=+0.7302$), canonical event counts (14 increased, 19 decreased, 34 not observed), ineligibility accounting. |
| **6. YOLO vs DINO/SAM Research** | `components/benchmark_view.py` | `FINAL_PERCEPTION_TABLE.csv`, `yolo_semantic_metrics.csv`, `panel_examples.csv` | In-domain validation benchmark on China_Drone ($N=480$, 742 GT boxes), $72.7\times$ latency ratio, 5-class semantic breakdown, curated qualitative panels. |
| **7. Cross-Domain Generalization** | `components/cross_domain_view.py` | `table_in_vs_cross_domain.csv`, `cross_domain_reliability_dataset.csv` | RDD2022 India dashcam zero-shot transfer ($N=300$), $-96.9\%$ YOLO collapse, DINO domain separation margin $+0.1158$, 100% failure quarantine ($300/300$ escalated). |
| **8. Decision & Priority Engine** | `components/decision_view.py` | `ROAD_HEALTH_DECISIONS.csv`, `table_decision_ablation.csv` | 5-tier action taxonomy, Experiment A distribution ($21\text{ MONITOR}$, $10\text{ REINSPECT}$, $9\text{ PRIORITY\_REVIEW}$), interactive explainable decision cards, multi-evidence ablation. |
| **9. Failure Mode Taxonomy** | `components/failure_modes.py` | `common_binary_metrics.csv`, benchmark figures | 6 DINO/SAM failure modes (aggregate FPs, thin crack dilution, glare), YOLO semantic weaknesses (D20 lattice fragmentation), IoU 0.50 vs 0.25 sensitivity. |
| **10. Research Methodology** | `components/methodology.py` | Repository documentation & configs | Complete mathematical formulations, dataset specifications, LTPP XGBoost features, reproducibility seeds, and direct artifact file links. |
| **11. Controlled Experiment B** | `components/experiment_b.py` | Phase 9 specification | 10-day deterioration $\to$ repair $\to$ re-deterioration protocol (`SEG_005` & `SEG_006`), clearly marked **PLANNED / DEFERRED FOR NEXT REVIEW**. |
| **12. Edge Deployment Roadmap** | `components/edge_deployment.py` | Hardware profiling specifications | Raspberry Pi 5 Cortex-A76 deployment architecture, ONNX/NCNN export roadmap, clearly marked **PENDING PHASE 8**. |

---

## 3. Data Loading & Performance Optimizations

### 3.1 Centralized Cached Loader (`dashboard/data_loader.py`)
- **Streamlit Caching**: All tabular datasets (`.csv`) and canonical configuration manifests (`.json`) are wrapped with `@st.cache_data(show_spinner=False)`.
- **Zero Heavy Import Overhead on Startup**: Heavy deep-learning frameworks (`torch`, `ultralytics`, `transformers`) are only imported on-demand if the user explicitly switches to Live Inference Mode.
- **Instant Cold-Start**: Dashboard cold start to first rendered frame is under **0.45 seconds** on local hardware.

### 3.2 Offline-First Reliability
- **Zero External Dependencies**: Dashboard operates with zero internet connectivity, external API keys, or active IPC sockets.
- **Deterministic Presentation**: All precomputed metrics, bounding boxes, segmentation masks, and decision explanations are bundled locally in the repository.

---

## 4. Summary of Consistency Audit Fixes

| Area | Audit Item | Resolution & Approved Wording |
|---|---|---|
| **Perception** | India Cross-Domain Failures | Explicitly disambiguated Target $T_0$ ($\text{F1}>0$, **292 Failures**) vs Target $T_1$ ($\text{F1}\ge 0.50$, **293 Failures**). Frame `India_000511` ($\text{F1}=0.4000$) documented. |
| **Temporal** | Event Counts Discrepancy | Reconciled directly from raw event logs: **14 Area Increased**, **19 Area Decreased**, **34 Not Observed**, **48 New Defects**, **33 Total Matched Transitions**. |
| **Tracking** | Association Algorithm | Corrected all documentation to **Greedy Hierarchical One-to-One Matching** (Mask IoU $\ge 0.50 \to$ Bbox IoU $\ge 0.30 \to$ Centroid/Area fallback). Removed references to Hungarian matching. |
| **Forecasting** | XGBoost Monotonicity | Verified Model V2 parameters (`monotone_constraints: (1,1,1,1,1,1)` and $[0.0, 1.0]$ clipping). Removed unsupported claims of convexity constraints. |
| **Safety** | Claim Precision | Softened out-of-distribution claims to strictly evaluated datasets: *"DINOv2 patch embeddings perfectly separated the evaluated China and India benchmark domains."* |

---

## 5. Verification & Test Suite

The automated test suite (`dashboard/test_dashboard_v2.py`) validates:
1. All 12 dashboard component modules import cleanly without syntax or dependency errors.
2. All 40 Experiment A image assets are physically present and resolvable on disk.
3. All primary CSV and JSON artifacts load correctly with non-empty DataFrames.
4. Numerical parity is maintained between `CANONICAL_RESEARCH_METRICS.json`, `ROAD_HEALTH_DECISIONS.csv`, and `goal1_forecasts.csv`.
5. Cold-start import latency remains under 1.0 second.

**Status**: **ALL TESTS PASSED**.
