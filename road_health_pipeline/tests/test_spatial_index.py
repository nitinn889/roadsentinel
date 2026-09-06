import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.spatial_index import DefectSpatialIndex, haversine_distance_m


def test_spatial_index_kd_tree():
    sample_defects = [
        {
            "defect_id": "def-1",
            "defect_type": "pothole",
            "latitude": 13.08270,
            "longitude": 80.27070,
            "is_water_filled": False,
            "severity": {"severity": "high"},
        },
        {
            "defect_id": "def-2",
            "defect_type": "water_filled_pothole",
            "latitude": 13.08310,
            "longitude": 80.27110,
            "is_water_filled": True,
            "severity": {"severity": "critical"},
        },
        {
            "defect_id": "def-3",
            "defect_type": "crack",
            "latitude": 13.08800,
            "longitude": 80.27500,
            "is_water_filled": False,
            "severity": {"severity": "low"},
        },
    ]

    index = DefectSpatialIndex(sample_defects)

    # 1. Query radius 60m around defect 1
    nearby = index.query_radius(13.08272, 80.27072, radius_m=60.0)
    assert len(nearby) >= 1
    assert nearby[0]["defect_id"] == "def-1"
    assert nearby[0]["distance_m"] < 10.0

    # 2. Query nearest
    nearest = index.query_nearest(13.08311, 80.27109, k=1)
    assert len(nearest) == 1
    assert nearest[0][0]["defect_id"] == "def-2"
    assert nearest[0][1] < 5.0

    # 3. Geofence zones
    zones = index.create_geofence_zones(default_radius_m=50.0)
    assert len(zones) == 3
    crit_zones = [z for z in zones if z["hazard_level"] == "critical"]
    assert len(crit_zones) == 1
    assert crit_zones[0]["defect_id"] == "def-2"

    # 4. Driver proximity evaluation
    alert_clear = index.evaluate_driver_hazard(13.0900, 80.2800, warning_radius_m=50.0)
    assert not alert_clear["alert_triggered"]
    assert alert_clear["status"] == "CLEAR"

    alert_warn = index.evaluate_driver_hazard(13.08275, 80.27075, warning_radius_m=50.0, critical_radius_m=10.0)
    assert alert_warn["alert_triggered"]
    assert alert_warn["hazard_level"] in ["warning", "critical"]
    assert alert_warn["speed_advisory_kmph"] in [30, 50]
