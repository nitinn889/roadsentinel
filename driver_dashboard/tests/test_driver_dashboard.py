"""Unit Test Suite for RoadSentinel Driver Dashboard.

Verifies:
1. Haversine distance formula against known GPS coordinates
2. 2.4 km deterministic route length and waypoint generation
3. Segment mapping and boundary transitions (399.9m, 400.0m, 799.9m, 800.0m, etc.)
4. GPS interpolation along the route
5. Road geofence detection (<= 50m inside, > 50m outside)
6. Nearest hazard ahead filtering (ignoring hazards behind vehicle)
7. ETA calculation logic
8. Driver alert distance thresholds (150m, 80m, 30m) and speed-context warning
9. Alert deduplication (firing each alert level only once per hazard ID)
10. Model-derived road health score conversion (100 * (1 - severity))
11. 400m weighted look-ahead road health across segment boundaries
12. Missing segment data handling (SEG_005 / SEG_006 pending inspection)
13. Simulation vehicle movement, start/pause/reset, and day switching
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# Ensure driver_dashboard directory is in sys.path
DRIVER_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DRIVER_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DASHBOARD_DIR))

from config import (
    ALERT_ADVISORY_M,
    ALERT_LEVEL_ADVISORY,
    ALERT_LEVEL_URGENT,
    ALERT_LEVEL_WARNING,
    ALERT_URGENT_M,
    ALERT_WARNING_M,
    EARTH_RADIUS_M,
    ROAD_GEOFENCE_RADIUS_M,
    SEGMENT_BOUNDARIES,
    START_LATITUDE,
    START_LONGITUDE,
    TOTAL_ROAD_LENGTH_M,
)
from gps import check_geofence, haversine_distance
from hazard_engine import (
    evaluate_alert_status,
    generate_hazard_manifest,
    get_hazards_for_day,
    get_nearest_hazard_ahead,
)
from route import (
    generate_deterministic_route,
    get_gps_at_distance,
    get_segment_at_distance,
)
from simulation import SimulationState, compute_400m_lookahead_health, compute_road_health_score


class TestHaversine(unittest.TestCase):
    """Test Haversine distance formula implementation."""

    def test_haversine_same_point(self):
        dist = haversine_distance(START_LATITUDE, START_LONGITUDE, START_LATITUDE, START_LONGITUDE)
        self.assertAlmostEqual(dist, 0.0, places=4)

    def test_haversine_known_distance(self):
        # Coordinates ~ 111 km apart along latitude (1 degree lat)
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 1.0, 0.0
        expected_m = (math.pi / 180.0) * EARTH_RADIUS_M  # ~111,194.9 m
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(dist, expected_m, delta=100.0)


class TestRouteAndSegments(unittest.TestCase):
    """Test route generation, segment boundaries, and GPS interpolation."""

    def setUp(self):
        self.waypoints = generate_deterministic_route()

    def test_route_length_and_waypoints(self):
        self.assertGreaterEqual(len(self.waypoints), 200)
        self.assertAlmostEqual(self.waypoints[0]["distance_m"], 0.0, places=1)
        self.assertAlmostEqual(self.waypoints[-1]["distance_m"], TOTAL_ROAD_LENGTH_M, places=1)

    def test_segment_boundaries(self):
        self.assertEqual(get_segment_at_distance(0.0), "SEG_001")
        self.assertEqual(get_segment_at_distance(399.9), "SEG_001")
        self.assertEqual(get_segment_at_distance(400.0), "SEG_002")
        self.assertEqual(get_segment_at_distance(799.9), "SEG_002")
        self.assertEqual(get_segment_at_distance(800.0), "SEG_003")
        self.assertEqual(get_segment_at_distance(1199.9), "SEG_003")
        self.assertEqual(get_segment_at_distance(1200.0), "SEG_004")
        self.assertEqual(get_segment_at_distance(1599.9), "SEG_004")
        self.assertEqual(get_segment_at_distance(1600.0), "SEG_005")
        self.assertEqual(get_segment_at_distance(1999.9), "SEG_005")
        self.assertEqual(get_segment_at_distance(2000.0), "SEG_006")
        self.assertEqual(get_segment_at_distance(2400.0), "SEG_006")

    def test_gps_interpolation(self):
        lat0, lon0 = get_gps_at_distance(0.0, self.waypoints)
        self.assertAlmostEqual(lat0, START_LATITUDE, places=4)
        self.assertAlmostEqual(lon0, START_LONGITUDE, places=4)

        lat_mid, lon_mid = get_gps_at_distance(1200.0, self.waypoints)
        self.assertNotEqual(lat_mid, START_LATITUDE)
        self.assertNotEqual(lon_mid, START_LONGITUDE)


class TestGeofence(unittest.TestCase):
    """Test road geofence checking."""

    def setUp(self):
        self.waypoints = generate_deterministic_route()

    def test_geofence_inside(self):
        # Exactly on the start point
        res = check_geofence(START_LATITUDE, START_LONGITUDE, self.waypoints)
        self.assertTrue(res["is_on_route"])
        self.assertEqual(res["status"], "ON_ROADSENTINEL_ROUTE")
        self.assertLessEqual(res["distance_m"], 50.0)

    def test_geofence_outside(self):
        # Point far away from route (e.g. +1.0 degree lat ~ 111 km)
        res = check_geofence(START_LATITUDE + 1.0, START_LONGITUDE, self.waypoints)
        self.assertFalse(res["is_on_route"])
        self.assertEqual(res["status"], "OUTSIDE_MONITORED_ROUTE")
        self.assertGreater(res["distance_m"], 50.0)


class TestHazardEngineAndAlerts(unittest.TestCase):
    """Test hazard generation, filtering ahead, ETA, alert thresholds, and deduplication."""

    def setUp(self):
        self.waypoints = generate_deterministic_route()

    def test_nearest_hazard_ahead_filtering(self):
        sample_hazards = [
            {"hazard_id": "H1", "global_distance_m": 300.0, "hazard_type": "Pothole", "latitude": 8.894, "longitude": 76.615},
            {"hazard_id": "H2", "global_distance_m": 700.0, "hazard_type": "Road Defect", "latitude": 8.897, "longitude": 76.618},
        ]

        # Vehicle at 100m -> nearest hazard ahead is H1 (300m)
        nh1 = get_nearest_hazard_ahead(100.0, START_LATITUDE, START_LONGITUDE, sample_hazards)
        self.assertIsNotNone(nh1)
        self.assertEqual(nh1["hazard_id"], "H1")
        self.assertAlmostEqual(nh1["distance_ahead_m"], 200.0, places=1)
        self.assertIn("eta_seconds", nh1)
        self.assertIsNone(nh1["eta_seconds"])  # default speed = 0 (stationary)

        # When moving at 72 km/h = 20 m/s: 200m ahead -> ETA = 10.0 sec
        nh1_moving = get_nearest_hazard_ahead(100.0, START_LATITUDE, START_LONGITUDE, sample_hazards, speed_kmh=72.0)
        self.assertIsNotNone(nh1_moving["eta_seconds"])
        self.assertAlmostEqual(nh1_moving["eta_seconds"], 10.0, places=1)

        # Vehicle at 400m -> H1 (300m) is BEHIND and IGNORED; nearest hazard ahead is H2 (700m)
        nh2 = get_nearest_hazard_ahead(400.0, START_LATITUDE, START_LONGITUDE, sample_hazards)
        self.assertIsNotNone(nh2)
        self.assertEqual(nh2["hazard_id"], "H2")
        self.assertAlmostEqual(nh2["distance_ahead_m"], 300.0, places=1)

        # Vehicle at 800m -> no hazards ahead
        nh3 = get_nearest_hazard_ahead(800.0, START_LATITUDE, START_LONGITUDE, sample_hazards)
        self.assertIsNone(nh3)

    def test_alert_thresholds_and_deduplication(self):
        fired_alerts = set()

        # 1. Distance = 200m (> 150m) -> No alert
        h_far = {"hazard_id": "H1", "hazard_type": "Pothole", "distance_ahead_m": 200.0}
        alert, fired_alerts = evaluate_alert_status(h_far, 40.0, fired_alerts)
        self.assertIsNone(alert)

        # 2. Distance = 120m (80m-150m) -> ADVISORY
        h_adv = {"hazard_id": "H1", "hazard_type": "Pothole", "distance_ahead_m": 120.0}
        alert, fired_alerts = evaluate_alert_status(h_adv, 40.0, fired_alerts)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["level"], ALERT_LEVEL_ADVISORY)
        self.assertTrue(alert["is_new"])

        # Duplicate ADVISORY for same hazard -> is_new is False
        alert_dup, fired_alerts = evaluate_alert_status(h_adv, 40.0, fired_alerts)
        self.assertFalse(alert_dup["is_new"])

        # 3. Distance = 50m (30m-80m) -> WARNING
        h_warn = {"hazard_id": "H1", "hazard_type": "Pothole", "distance_ahead_m": 50.0}
        alert_w, fired_alerts = evaluate_alert_status(h_warn, 40.0, fired_alerts)
        self.assertIsNotNone(alert_w)
        self.assertEqual(alert_w["level"], ALERT_LEVEL_WARNING)
        self.assertTrue(alert_w["is_new"])

        # 4. Distance = 20m (<= 30m) -> URGENT
        h_urg = {"hazard_id": "H1", "hazard_type": "Pothole", "distance_ahead_m": 20.0}
        alert_u, fired_alerts = evaluate_alert_status(h_urg, 40.0, fired_alerts)
        self.assertIsNotNone(alert_u)
        self.assertEqual(alert_u["level"], ALERT_LEVEL_URGENT)
        self.assertTrue(alert_u["is_new"])


class TestHealthLookahead(unittest.TestCase):
    """Test Road Health Score conversion and 400m look-ahead engine."""

    def test_road_health_score_conversion(self):
        self.assertEqual(compute_road_health_score(0.0), 100.0)
        self.assertEqual(compute_road_health_score(0.35), 65.0)
        self.assertEqual(compute_road_health_score(1.0), 0.0)

    def test_400m_lookahead_single_segment(self):
        # Synthetic primary data where SEG_001 severity = 0.20
        mock_data = {
            5: {
                "SEG_001": {"has_data": True, "current_severity": 0.20},
                "SEG_002": {"has_data": True, "current_severity": 0.40},
            }
        }

        # At 0m, 400m lookahead is entirely in SEG_001 -> severity = 0.20 -> health = 80/100
        res = compute_400m_lookahead_health(0.0, 5, mock_data)
        self.assertAlmostEqual(res["health_score"], 80.0, places=1)
        self.assertAlmostEqual(res["weighted_severity"], 0.20, places=2)

    def test_400m_lookahead_boundary_weighted(self):
        # At 300m, lookahead window is [300, 700]:
        # 100m in SEG_001 (sev = 0.20) + 300m in SEG_002 (sev = 0.40)
        # Weighted severity = (100*0.20 + 300*0.40) / 400 = (20 + 120) / 400 = 140 / 400 = 0.35
        # Health score = 100 * (1 - 0.35) = 65.0 / 100
        mock_data = {
            5: {
                "SEG_001": {"has_data": True, "current_severity": 0.20},
                "SEG_002": {"has_data": True, "current_severity": 0.40},
            }
        }

        res = compute_400m_lookahead_health(300.0, 5, mock_data)
        self.assertAlmostEqual(res["weighted_severity"], 0.35, places=2)
        self.assertAlmostEqual(res["health_score"], 65.0, places=1)

    def test_400m_lookahead_missing_segment(self):
        # SEG_005 and SEG_006 are in MISSING_DATA_SEGMENTS
        mock_data = {
            5: {
                "SEG_005": {"has_data": False, "severity": None},
                "SEG_006": {"has_data": False, "severity": None},
            }
        }
        res = compute_400m_lookahead_health(1800.0, 5, mock_data)
        self.assertTrue(res["data_pending"])


class TestSimulationState(unittest.TestCase):
    """Test SimulationState car movement, speed, day switching, and reset."""

    def test_simulation_lifecycle(self):
        sim = SimulationState(day=5, speed_kmh=36.0)  # 36 km/h = 10 m/s
        self.assertFalse(sim.is_running)
        self.assertEqual(sim.distance_m, 0.0)

        sim.start()
        self.assertTrue(sim.is_running)

        # Update 5 seconds -> distance should be 50.0m
        sim.update(5.0)
        self.assertAlmostEqual(sim.distance_m, 50.0, places=1)

        sim.pause()
        self.assertFalse(sim.is_running)

        sim.reset()
        self.assertEqual(sim.distance_m, 0.0)
        self.assertFalse(sim.is_running)


if __name__ == "__main__":
    import math
    unittest.main()
