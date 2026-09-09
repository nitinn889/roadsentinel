# RoadSentinel Driver Dashboard — Demo Guide

## 1. Quick Start Guide

To launch the RoadSentinel Driver Dashboard independently, run the executable bash launcher from the repository root:

```bash
./launch_driver_dashboard.sh
```

Alternatively, invoke Streamlit directly:

```bash
streamlit run driver_dashboard/app.py
```

The app will start and open automatically in your browser (typically at `http://localhost:8502`).

---

## 2. Deterministic Demo Scenario (Day 05 @ 40 km/h)

Follow these steps to demonstrate all core driver dashboard features:

### Step 1: Initialize Inspection Day 05
1. In the sidebar under **Simulated Road Inspection Day**, select **Day 05**.
2. Notice the road status banner updates to show available data for `SEG_001`–`SEG_004` and `DATA UNAVAILABLE / PENDING INSPECTION` for `SEG_005`–`SEG_006`.

### Step 2: Start Vehicle Simulation
1. Set **Driving Speed** to **40 km/h** (default).
2. Click the **▶ START** button in the sidebar.
3. Observe the vehicle telemetry updating continuously:
   - **Current Speed:** 40 km/h
   - **Road Health Ahead:** Updates smoothly as the car traverses segments.
   - **Distance Travelled / Distance Remaining:** Progresses along 2,400 m route.
   - **Nearest Hazard:** Shows nearest pothole/defect ahead with distance (m) and ETA (sec).

### Step 3: Observe Context-Aware Alerts
As the car approaches defects on `SEG_001` and `SEG_002`:
- **> 150 m:** No active alert banner.
- **80 m – 150 m:** **ADVISORY** banner ("Road damage detected XX m ahead.").
- **30 m – 80 m:** **WARNING** banner ("Pothole XX m ahead. Approach with caution.").
- **<= 30 m:** **URGENT** banner ("Road hazard XX m ahead. Reduce speed.").
- **Speed-Context Warning:** Increase speed slider to **70 km/h** while near a hazard (<= 80 m) to trigger high-speed warning: `"Road damage XX m ahead at current speed. Reduce speed."`
- **Hazard Clearance:** Once the car passes a hazard, the hazard is ignored and the telemetry updates to the next hazard ahead.

### Step 4: Map & Segment Strip Visualization
- **PyDeck Route Map:** View the green start point, red finish line, yellow vehicle position marker, orange hazard pins, and red nearest hazard highlight pin.
- **6-Segment Road Strip:** Highlights the active segment in blue (`SEG_001` -> `SEG_002` -> `SEG_003` -> `SEG_004` -> `SEG_005` -> `SEG_006`).
- **Missing Data Handling:** When entering `SEG_005` and `SEG_006`, observe the `ROAD HEALTH DATA PENDING` indicator.

---

## 3. Developer Manual GPS Testing Mode

1. In the sidebar under **Operating Mode**, switch from **Simulated Drive** to **Manual GPS Test**.
2. Test **ON_ROADSENTINEL_ROUTE**:
   - Enter Latitude: `8.8940`
   - Enter Longitude: `76.6150`
   - Observe status: `ON_ROADSENTINEL_ROUTE` (inside 50m corridor). Telemetry and hazard calculations run normally.
3. Test **OUTSIDE_MONITORED_ROUTE**:
   - Enter Latitude: `9.0000`
   - Enter Longitude: `77.0000`
   - Observe status: `OUTSIDE_MONITORED_ROUTE`. Road hazard alerts and road health metrics are suppressed as required.

---

## 4. Reset & Day Switching Verification

1. Click **🔄 RESET** to restore vehicle distance to 0 m and clear stateful alert logs.
2. Select **Day 01** or **Day 03** to observe how temporal track IDs maintain defect positions across inspection days while health scores reflect temporal evolution.
