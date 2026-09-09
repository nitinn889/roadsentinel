# RoadSentinel Driver Dashboard — GPS & Alert Logic Specification

**Document**: `driver_dashboard/GPS_AND_ALERT_LOGIC.md`  
**Purpose**: Complete mathematical and logical specification for Haversine distance, geofence boundary check, 400m look-ahead health calculation, and driver alert deduplication.

---

## 1. Haversine Distance Formula

The great-circle distance between two GPS coordinates $(\text{lat}_1, \text{lon}_1)$ and $(\text{lat}_2, \text{lon}_2)$ is computed using:

$$\phi_1 = \text{radians}(\text{lat}_1), \quad \phi_2 = \text{radians}(\text{lat}_2)$$
$$\Delta \phi = \text{radians}(\text{lat}_2 - \text{lat}_1), \quad \Delta \lambda = \text{radians}(\text{lon}_2 - \text{lon}_1)$$

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1) \cdot \cos(\phi_2) \cdot \sin^2\left(\frac{\Delta \lambda}{2}\right)$$

$$c = 2 \cdot \operatorname{atan2}\left(\sqrt{a}, \sqrt{1-a}\right)$$

$$d = R \cdot c, \quad \text{where } R = 6,371,000\,\text{m}$$

All calculations use floating-point double precision in radians.

---

## 2. Road Geofence Logic

- **Geofence Radius**: $50.0\,\text{m}$
- **Geofence Evaluation**: Finds nearest route waypoint using Haversine distance.
  - If $d_{\text{min}} \le 50.0\,\text{m} \implies$ **`ON_ROADSENTINEL_ROUTE`**
  - If $d_{\text{min}} > 50.0\,\text{m} \implies$ **`OUTSIDE_MONITORED_ROUTE`**
- **Safety Rule**: When outside the monitored route, all hazard alerts and look-ahead score calculations are safely suspended.

---

## 3. 400m Look-Ahead Road Health Score

Translates model severity $[0.0, 1.0]$ into a driver-facing score $[0, 100]$:

$$\text{Health Score} = 100 \times (1 - \text{Severity})$$

For the $400\,\text{m}$ window ahead of the vehicle $[d, d + 400\,\text{m}]$:
- If the window lies within a single segment, that segment's severity is used.
- If the window spans across segment boundaries (e.g. $100\,\text{m}$ in SEG_001 and $300\,\text{m}$ in SEG_002):

$$\text{Weighted Severity} = \frac{L_1 \cdot S_1 + L_2 \cdot S_2}{400}$$

Where $L_i$ is the overlap length and $S_i$ is the segment severity.

---

## 4. Alert Thresholds & Deduplication

### Distance Thresholds:
- **$> 150\,\text{m}$**: No alert.
- **$80\,\text{m} - 150\,\text{m}$**: **`ADVISORY`** ("Road damage detected XX m ahead.")
- **$30\,\text{m} - 80\,\text{m}$**: **`WARNING`** ("Pothole / road defect XX m ahead. Approach with caution.")
- **$\le 30\,\text{m}$**: **`URGENT`** ("Road hazard XX m ahead. Reduce speed.")

### Speed-Context Escalation:
If distance $\le 80\,\text{m}$ AND speed $\ge 60\,\text{km/h}$:
> *"WARNING: Road damage XX m ahead at current speed. Reduce speed."*

### Deduplication Policy:
- Unique alert trigger key: `(hazard_id, alert_level)`.
- Each alert level fires only ONCE per hazard ID.
- Fired alerts set is cleared when:
  1. Simulation is reset.
  2. Inspection day changes.
- **Non-Directive Safety Rule**: Alerts NEVER instruct drivers to swerve or change lanes.
