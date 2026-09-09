# RoadSentinel — Examiner-Facing Interactive Research Dashboard

**Phase 7 Primary Dashboard Package**  
Integrating Supervised Damage Detection (YOLOv8n), Zero-Shot Foundation Anomaly Perception (DINOv2 + SAM2), Multi-Day Temporal Tracking (Phase 4), and Scenario-Conditioned Deterioration Forecasting (XGBoost Model V2).

---

## 1. Quick Launch (Single Command)

From the project root directory:

```bash
.venv/bin/streamlit run dashboard/app.py
```

By default, the dashboard will open automatically in your browser at:
- **URL**: `http://localhost:8501`
- **Network URL**: displayed in console output for local network demonstration.

To specify a custom port or prevent auto-opening:
```bash
.venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

---

## 2. Architecture & Design Principles

The dashboard adheres to the following core tenets:
1. **Offline-First / Verified Demo Mode Default**: Precomputed and cached research assets ensure that panel demonstrations never crash or hang due to GPU driver mismatches, CUDA memory limits, or missing model weights.
2. **Defensive Fallback Handling**: Every data loader function in `dashboard/data_loader.py` includes try/except wrappers and synthetic defaults so that missing optional files never cause an unhandled exception.
3. **Lazy Asset & Model Loading**: Heavy deep-learning frameworks (PyTorch, Ultralytics, SAM2) are never loaded at startup. The application cold-starts in **under 0.5 seconds**.
4. **Absolute Simulation Isolation**: Complies with the project's strict rule never to interface with or launch Unreal Engine, CARLA, or Interactive Studio. All data is read from frozen disk records.

---

## 3. Directory Structure

```
dashboard/
├── app.py                      # Main entrypoint, sidebar routing, and theme injection
├── data_loader.py              # Cached, defensive loader for manifests, models, & tables
├── assets/
│   └── style.css               # Dark glassmorphism styling, Outfit/Inter typography, badges
├── components/
│   ├── overview.py             # 1. System Overview & end-to-end architecture flow
│   ├── single_image.py         # 2. Single-Image Road Assessment (Side-by-side cards)
│   ├── forecast.py             # 3. Future Condition Forecast (XGBoost Model V2)
│   ├── temporal.py             # 4. Temporal Road Monitoring (SEG_003 & SEG_004)
│   ├── benchmark_view.py       # 5. YOLOv8n vs DINOv2+SAM2 (In-domain comparison)
│   ├── failure_modes.py        # 6. Failure Mode Analysis (DINO FPs, YOLO D20, IoU 0.25)
│   ├── edge_deployment.py      # 7. Edge Deployment (Raspberry Pi 5 — Pending Phase 8)
│   └── methodology.py          # 8. Research Methodology & Artifact Evidence Links
├── EXAMINER_DEMO_GUIDE.md      # Step-by-step 11-stage examiner presentation guide
├── PHASE7_PROGRESS.md          # Comprehensive Phase 7 validation & readiness report
└── README.md                   # This setup, execution, and troubleshooting manual
```

---

## 4. Navigation Sections

| Section | Key Content & Examiner Focus |
|---|---|
| **1. System Overview** | End-to-end architecture diagram, frozen pipeline status badges, headline comparison metrics. |
| **2. Single-Image Road Assessment** | Side-by-side inspection cards (Original \| YOLO \| DINO/SAM), severity calculation, educational guide. |
| **3. Future Condition Forecast** | Frozen XGBoost Model V2, scenario stress presets (`HEAVY_RAIN`, `HIGH_HEAT`, etc.), delta progression. |
| **4. Temporal Road Monitoring** | Multi-day tracking sequences: `SEG_003` stability control (CV: 0.67%), `SEG_004` progression (Δ: +0.7302), and event taxonomy. |
| **5. YOLO vs DINOv2+SAM2** | Final frozen comparison table, YOLO 5-class breakdown (with D40 low support caveat), curated panel gallery. |
| **6. Failure Mode Analysis** | Transparent breakdown of DINOv2 texture FPs, thin crack patch dilution, YOLO D20 confusion, and IoU 0.25 sensitivity. |
| **7. Edge Deployment** | Embedded deployment architecture, Cortex-A76 target specs, planned metrics, and `PENDING PHASE 8` badge. |
| **8. Research / Methodology** | Mathematical formulation of monotone forecasting, hierarchical greedy tracking rules, and repository links. |

---

## 5. Execution Modes

### A. Verified Demo Mode (Default & Recommended)
- Uses precomputed inference records from the 480-image common validation benchmark and curated panel crops.
- Zero GPU compute required; deterministic latency (< 10 ms per click).
- Safest mode for live presentations and academic examinations.

### B. Live YOLOv8n Inference Mode (Optional)
- Available on the Single-Image Road Assessment page.
- Dynamically invokes PyTorch and CUDA for YOLOv8n on `best.pt` (RTX 5060 Laptop GPU), while DINOv2+SAM2 displays frozen precomputed segmentations.
- If GPU memory is exhausted or CUDA is uninitialized, the UI catches the error and seamlessly falls back to precomputed results without crashing.

---

## 6. Troubleshooting

- **Port already in use**:
  ```bash
  .venv/bin/streamlit run dashboard/app.py --server.port 8502
  ```
- **Virtual environment activation**:
  If `.venv` is not in your current PATH:
  ```bash
  source .venv/bin/activate
  streamlit run dashboard/app.py
  ```
- **Dependencies check**:
  Verify required libraries are installed:
  ```bash
  .venv/bin/python -c "import streamlit, xgboost, pandas, cv2, torch; print('All dependencies satisfied.')"
  ```
