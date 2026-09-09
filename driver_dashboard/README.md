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

---

## 🚀 Launch Instructions

Launch ONLY the Driver Dashboard:

```bash
./launch_driver_dashboard.sh
```

Or run via Streamlit directly:

```bash
streamlit run driver_dashboard/app.py
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

## 🧪 Testing

Run the automated driver dashboard unit test suite:

```bash
python -m unittest driver_dashboard/tests/test_driver_dashboard.py
```

Tests verify Haversine calculations, route generation, boundary mapping, geofencing, hazard filtering, alert deduplication, 400m look-ahead health, and simulation movement.
