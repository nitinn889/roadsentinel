"""
test_interactive_sim.py
------------------------
Unit tests for RoadSentinel interactive launcher, procedural road generation,
6 defect shape families, water-filled states, seed reproducibility,
ground truth integrity, and CARLA readiness detection.
"""

import json
import math
import sys
from pathlib import Path

# Paths
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "env"))

import run_simulation as rsim
from env import config
from env.road_injector import ProceduralRoadGenerator, generate_shape_polygon


def test_interactive_configuration_parsing():
    """Verify configuration dictionary parsing and default handling."""
    raw_cfg = {
        "scenario": "poor",
        "weather": "post_rain",
        "altitude": 100.0,
        "speed": 30.0,
        "water": "Automatic by Scenario",
        "seed": "42",
        "duration": 60.0,
        "rendering": "GUI",
    }
    validated = rsim.validate_configuration(raw_cfg)
    assert validated["scenario"] == "poor"
    assert validated["weather"] == "post_rain"
    assert validated["altitude"] == 100.0
    assert validated["speed"] == 30.0
    assert validated["seed"] == 42
    assert validated["duration"] == 60.0
    assert validated["headless"] is False


def test_configuration_validation_errors():
    """Verify validation bounds on scenario, weather, altitude, speed, and duration."""
    # Invalid scenario
    try:
        rsim.validate_configuration({"scenario": "apocalyptic"})
        assert False, "Should fail on invalid scenario"
    except ValueError:
        pass

    # Invalid altitude (negative or >500m)
    try:
        rsim.validate_configuration({"altitude": -10.0})
        assert False, "Should fail on negative altitude"
    except ValueError:
        pass

    # Invalid speed
    try:
        rsim.validate_configuration({"speed": 500.0})
        assert False, "Should fail on excessive speed"
    except ValueError:
        pass

    # Invalid duration
    try:
        rsim.validate_configuration({"duration": -5.0})
        assert False, "Should fail on negative duration"
    except ValueError:
        pass


def test_scenario_and_weather_selection():
    """Verify that all scenarios and weather presets validate correctly."""
    for s in ["healthy", "moderate", "poor", "critical"]:
        cfg = rsim.validate_configuration({"scenario": s})
        assert cfg["scenario"] == s

    for w in ["clear", "overcast", "wet", "rain", "post_rain", "low_light", "sunset", "early_morning"]:
        cfg = rsim.validate_configuration({"weather": w})
        assert cfg["weather"] == w


def test_seed_and_altitude_propagation():
    """Verify seed and altitude propagation into drone sim command line."""
    cfg = rsim.validate_configuration({
        "scenario": "moderate",
        "weather": "clear",
        "altitude": 80.0,
        "speed": 20.0,
        "seed": 999,
        "duration": 45.0,
        "rendering": "Headless",
    })
    out_dir = Path("/tmp/test_rs_output")
    cmd = rsim.build_drone_cmd(cfg, out_dir, standalone=True)

    assert "--altitude" in cmd and cmd[cmd.index("--altitude") + 1] == "80.0"
    assert "--speed" in cmd and cmd[cmd.index("--speed") + 1] == "20.0"
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "999"
    assert "--headless" in cmd
    assert "--standalone" in cmd
    assert "--auto-fly" in cmd


def test_procedural_seed_reproducibility():
    """Verify that identical seeds produce bitwise-identical defect specifications."""
    gen1 = ProceduralRoadGenerator(scenario="poor", seed=42)
    plan1 = gen1.generate_corridor_plan(segment_length_m=120.0, defects_count=10)

    gen2 = ProceduralRoadGenerator(scenario="poor", seed=42)
    plan2 = gen2.generate_corridor_plan(segment_length_m=120.0, defects_count=10)

    assert len(plan1) == len(plan2)
    for (a1, l1, s1), (a2, l2, s2) in zip(plan1, plan2):
        assert math.isclose(a1, a2, abs_tol=1e-5)
        assert math.isclose(l1, l2, abs_tol=1e-5)
        assert s1.defect_id == s2.defect_id
        assert s1.shape_category == s2.shape_category
        assert s1.is_water_filled == s2.is_water_filled
        assert math.isclose(s1.depth_m, s2.depth_m, abs_tol=1e-4)

    # Different seed must yield different arrangement
    gen3 = ProceduralRoadGenerator(scenario="poor", seed=99)
    plan3 = gen3.generate_corridor_plan(segment_length_m=120.0, defects_count=10)
    assert plan1[0][0] != plan3[0][0] or plan1[0][1] != plan3[0][1]


def test_pothole_geometry_diversity():
    """Verify representation of all 6 physical shape families across generated defects."""
    gen = ProceduralRoadGenerator(scenario="critical", seed=123)
    plan = gen.generate_corridor_plan(segment_length_m=300.0, defects_count=40)

    shape_families = {spec.shape_category for _, _, spec in plan}
    expected_families = {
        "elongated_longitudinal",
        "elongated_transverse",
        "irregular_natural",
        "jagged",
        "compound_cluster",
        "partially_connected",
    }
    # In 40 defects with critical scenario, all 6 families should be instantiated
    for fam in expected_families:
        assert fam in shape_families, f"Shape family '{fam}' missing from generated plan"

    # Verify polygon generation for all 6 families produces valid 2D coordinates
    for fam in expected_families:
        pts = generate_shape_polygon(cx=100, cy=100, rx=20, ry=15, shape_cat=fam, seed=42)
        assert len(pts) >= 48
        assert pts.shape[1] == 2


def test_water_state_diversity():
    """Verify water coverage fractions, turbidity values, and capillary halos."""
    gen = ProceduralRoadGenerator(scenario="poor", seed=777)
    plan = gen.generate_corridor_plan(segment_length_m=200.0, defects_count=20, water_ratio_override=0.60)

    water_defects = [spec for _, _, spec in plan if spec.is_water_filled]
    assert len(water_defects) > 0, "Expected water-filled defects with water_ratio=0.60"

    for w_spec in water_defects:
        assert 0.25 <= w_spec.water_coverage_frac <= 1.0
        assert 0.0 <= w_spec.turbidity <= 1.0
        assert 0.10 <= w_spec.wet_halo_radius_m <= 0.60
        assert w_spec.water_depth_m > 0.0


def test_ground_truth_correctness():
    """Verify ground truth structure adheres to required schema without shortcuts."""
    gen = ProceduralRoadGenerator(scenario="poor", seed=42)
    plan = gen.generate_corridor_plan(segment_length_m=100.0, defects_count=5)
    spec = plan[0][2]

    # Required fields verification
    assert spec.defect_id.startswith("pothole_poor_")
    assert spec.defect_type in ("pothole", "water_filled_pothole")
    assert spec.length_m > 0.0
    assert spec.width_m > 0.0
    assert spec.depth_m > 0.0
    assert spec.aspect_ratio > 0.0
    assert 0.0 <= spec.orientation_deg <= 360.0
    assert spec.cluster_id.startswith("cluster_")


def test_launcher_carla_readiness_detection():
    """Verify CARLA socket probing helper logic."""
    # Probing an unassigned high port should reliably return False
    assert rsim.is_port_open("127.0.0.1", 59999, timeout=0.1) is False
    # Localhost check should execute without uncaught exceptions
    active = rsim.check_carla_running("127.0.0.1", 2000, timeout=0.5)
    assert isinstance(active, bool)
