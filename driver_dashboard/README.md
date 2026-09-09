# RoadSentinel Driver User Dashboard

**Module Path**: `driver_dashboard/`  
**Classification**: Driver-Facing Real-Time Road-Health Simulation & Hazard Warning System  
**Status**: **PASS** (100% Offline-Ready & Canonical Data Integrated)

---

## 🌟 Overview & Purpose

The **RoadSentinel Driver User Dashboard** is a separate, user-centric simulation interface connecting RoadSentinel's perception outputs, multi-day inspection data, and scenario forecasts into a 2.4 km continuous simulated road corridor.

Unlike the examiner-facing research dashboard (`dashboard/`), the Driver Dashboard presents non-technical, real-time driver metrics:
- **Vehicle Speed**: Live speed display (0–100 km/h) with speed-context warnings.
- **Road Health Score Ahead**: Model-derived 400m look-ahead window score $[0, 100]$.
- **Nearest Hazard Ahead**: Distance, ETA, and hazard type (Pothole / Road Defect).
- **Proactive Alerts**: Non-directive Advisory ($150\,\text{m}$), Warning ($80\,\text{m}$), and Urgent ($30\,\text{m}$) driver alerts with single-trigger deduplication.
- **2.4 km Road Strip & Map**: Live PyDeck visualization showing vehicle movement and segment status (SEG_001 through SEG_006).
- **Citizen Road-Damage Reporting + AI Verification**: Upload road defect/pothole photos for multi-model AI verification (YOLO + DINOv2 domain gate + SAM2 defect segmentation + reliability calibration + model severity).

---

## 🚀 Launch Instructions

Launch ONLY the Driver Dashboard (Port 8502):

```bash
./launch_driver_dashboard.sh
```

Or run via Streamlit directly:

```bash
.venv/bin/streamlit run driver_dashboard/app.py --server.port 8502
```

---

## 🛣️ 2.4 km Road Geometry

| Segment ID | Local Distance | Global Corridor Range | Inspection Status |
|:---:|:---:|:---:|:---:|
| **SEG_001** | $0 - 400\,\text{m}$ | $0 - 400\,\text{m}$ | Inspected (Days 01–10) |
| **SEG_002** | $0 - 400\,\text{m}$ | $400 - 800\,\text{m}$ | Inspected (Days 01–10) |
| **SEG_003** | $0 - 400\,\text{m}$ | $800 - 1200\,\text{m}$ | Inspected (Days 01–10) |
| **SEG_004** | $0 - 400\,\text{m}$ | $1200 - 1600\,\text{m}$ | Inspected (Days 01–10) |
| **SEG_005** | $0 - 400\,\text{m}$ | $1600 - 2000\,\text{m}$ | DATA PENDING (Uninspected) |
| **SEG_006** | $0 - 400\,\text{m}$ | $2000 - 2400\,\text{m}$ | DATA PENDING (Uninspected) |

---

## 📸 Citizen Road-Damage Reporting + AI Verification

The **Citizen Road-Damage Reporting** module allows drivers and citizens to report road hazards with automated multi-model verification:
1. **Input**: Image upload (JPG, PNG, WEBP) + optional GPS coordinates.
2. **YOLO Detection**: Detects defects (Class `D40` mapped to `Pothole`, `D00`/`D10`/`D20` to `Road Defect`, `Repair` to `Repair / Patch`).
3. **DINOv2 Domain Gate**: Computes kNN cosine distance to training reference embeddings to identify out-of-distribution visual domains.
4. **SAM2 Instance Segmentation**: Prompts SAM2 with defect bounding boxes to extract fine-grained masks and compute defect area ratios.
5. **Model-Derived Severity**: Multi-factor distress rating $[0.0, 1.0]$ explicitly labeled `MODEL-DERIVED SEVERITY — NOT PCI`.
6. **Perception Reliability**: Calibrated score $[0.0, 1.0]$ (`HIGH`, `MEDIUM`, `LOW`).
7. **Report Outcomes**:
   - `VERIFIED REPORT`: Strong detection with high/medium reliability in familiar visual domain.
   - `NEEDS MANUAL REVIEW`: Plausible defect with low confidence, low reliability, or domain shift.
   - `RETAKE / NO VERIFIED DAMAGE`: Unusable image, no defect detected, or invalid input.
8. **Local Persistence**: Reports stored in `driver_dashboard/data/citizen_reports/` (`RS-CR-XXXX`) with raw images, annotated overlays, and JSON/CSV summary manifests.

---

## 🧪 Testing

Run the automated driver dashboard unit and integration test suites:

```bash
PYTHONPATH=driver_dashboard PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/pytest driver_dashboard/tests/
```

Tests verify Haversine calculations, route generation, boundary mapping, geofencing, hazard filtering, alert deduplication, 400m look-ahead health, simulation movement, and complete citizen reporting AI analysis and persistence.
