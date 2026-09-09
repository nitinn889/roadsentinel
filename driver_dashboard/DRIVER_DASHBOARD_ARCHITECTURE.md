# RoadSentinel Driver Dashboard — Architecture Specification

## 1. Overview & System Design

The **RoadSentinel Driver User Dashboard** is a standalone real-time user-facing driver assistance UI. It abstracts complex research metrics (e.g., DINOv2 patch anomaly maps, SAM2 segmentations, IoU, k-NN) into intuitive, actionable driver telemetry:
- Current Vehicle Speed & Deterministic SIMULATED GPS Position
- 400 m Weighted Look-Ahead Road Health Score (0–100 scale)
- Nearest Hazard Ahead Detection (Pothole / Road Defect) with distance and real-time ETA
- Context-Aware Multi-Tier Alerts (Advisory, Warning, Urgent) with stateful deduplication
- Continuous 2.4 km Visual Corridor Map (PyDeck) and Segment Status Strip

---

## 2. High-Level Modular Architecture

```
                                  +-----------------------+
                                  | Primary CSV & Temporal|
                                  | Research Artifacts    |
                                  +-----------+-----------+
                                              |
                                              v
+------------------+             +-----------------------+
|  route.py        |             |  data_loader.py       |
|  - 2.4km Route   +------------>|  - Day 01..10 Ingest  |
|  - Waypoints     |             |  - Pending Handling   |
+--------+---------+             +-----------+-----------+
         |                                   |
         v                                   v
+------------------+             +-----------------------+
|  gps.py          |             |  hazard_engine.py     |
|  - Haversine     +------------>|  - Manifest Build     |
|  - Geofence (50m)|             |  - Nearest Hazard     |
+------------------+             |  - Alert Evaluator    |
                                 +-----------+-----------+
                                             |
                                             v
                                 +-----------------------+
                                 |  simulation.py        |
                                 |  - Car Telemetry      |
                                 |  - 400m Look-Ahead    |
                                 +-----------+-----------+
                                             |
                                             v
                                 +-----------------------+
                                 |  app.py (Streamlit UI)|
                                 |  - Driver Telemetry   |
                                 |  - PyDeck Route Map   |
                                 |  - 6-Segment Strip    |
                                 +-----------------------+
```

---

## 3. Key Components & Implementation Details

### 3.1 `config.py`
Defines core system constants, including total road length (2,400 m), 6 temporal segments of 400 m each (`SEG_001` through `SEG_006`), route starting coordinate `(8.8932, 76.6141)`, default speed (40 km/h), geofence radius (50 m), and alert thresholds (Advisory: 150 m, Warning: 80 m, Urgent: 30 m).

### 3.2 `gps.py`
Provides `haversine_distance()` using spherical trigonometry ($R = 6,371,000\,\text{m}$) and internal radian conversions. Provides `check_geofence()` to verify whether a GPS coordinate is within 50 m of the route corridor (`ON_ROADSENTINEL_ROUTE` vs `OUTSIDE_MONITORED_ROUTE`).

### 3.3 `route.py`
Generates a deterministic 2.4 km simulated route with waypoints spaced every 10 meters (241 waypoints total). Provides `interpolate_gps(distance_m)` for smooth vehicle spatial tracking and route boundary mapping.

### 3.4 `data_loader.py`
Parses existing canonical RoadSentinel artifacts (`ROADSENTINEL_PRIMARY_RESULTS.csv`, `ROAD_HEALTH_DECISIONS.csv`, YOLO prediction summaries). Supports temporal segment selection (Day 01–Day 10). Gracefully flags missing segment data (`SEG_005`/`SEG_006`) as `DATA UNAVAILABLE / PENDING INSPECTION` without fabricating results.

### 3.5 `hazard_engine.py`
Constructs a normalized hazard manifest from research perception outputs. Maps YOLO class `D40` to `Pothole` and other defects to `Road Defect`. Assigns deterministic simulated GPS coordinates per defect track ID to maintain spatial stability across inspection days. Evaluates nearest hazard ahead and applies multi-tier deduplicated alerts.

### 3.6 `simulation.py`
Manages car state (Start, Pause, Reset, Speed 0–100 km/h). Computes the 400 m weighted look-ahead severity across segment boundaries and converts it to a user-facing Road Health Score ($100 \times (1 - \text{weighted\_severity})$).

### 3.7 `app.py`
The primary Streamlit driver dashboard interface. Renders top driver metrics, active warning banners, dynamic PyDeck map visualization, interactive 6-segment status strip, day-over-day comparison expander, alert history log, manual GPS developer testing mode, and safety disclaimer.

---

## 4. Safety & System Isolation

- **Zero Modifications to Core Code:** Does not touch `dashboard/`, `integration/`, `decision_engine/`, `Unreal`, `CARLA`, or `IPC`.
- **Pure Output Consumer:** Reads frozen research data read-only.
