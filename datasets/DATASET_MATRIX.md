# RoadSentinel Dataset Master Matrix

This matrix maps each acquired dataset to its suitability for various RoadSentinel tasks, domains, and model evaluations.

| Dataset | Pothole Det | Pothole Seg | Crack Seg | Road Health Scoring | Water Det | Depth Val | Drone Domain | Indian Domain | Weather Div | Temporal Data | Prediction Potential | SAM2 Suit | DINOv2 Suit | Pi5 Suit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **rdd2022 (existing)** | Yes | No | No | No | No | No | Yes | No | No | No | No | No | No | Yes | Yes (dev subset) |
| **rdd2022_full (China_Drone)** | Yes | No | No | Partial | No | No | Yes | No | No | No | No | No | No | Yes | No |
| **rdd2022_full (India)** | Yes | No | No | Partial | No | No | No | Yes | No | No | No | No | No | Yes | No |
| **pothole_mix** | Yes | Yes | Partial | Yes | No | Yes (videos) | No | No | No | No | No | Yes | Yes | Yes | No |
| **water_filled_potholes** | Yes | No | No | Yes | Yes | No | No | No | Yes | No | No | No | No | Yes | No |
| **pothole_600** | Yes | Yes | No | Yes | No | Yes | No | No | No | No | No | Yes | Yes | Yes | No |
| **chitholian** | Yes | No | No | No | No | No | No | No | No | No | No | No | No | Yes | Yes (Pi5 subset) |
| **mwpd** | Yes | No | No | Yes | No | No | No | No | Yes | No | No | No | No | Yes | No |
| **qr4change** | No | No | No | Yes | No | No | No | Yes | Yes | No | No | No | No | Yes | No |

## Suitability Definitions:
- **Yes:** Fully supported by official annotations and domains.
- **Partial:** Bounding boxes only, or subset provides partial mask capability.
- **No:** Not supported or verified.
- **Not verified:** Gaps in current research data.
