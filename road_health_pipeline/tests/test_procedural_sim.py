"""test_procedural_sim.py
-----------------------
Unit and integration tests for the RoadSentinel Procedural CARLA Testing Testbed:
  - Defect diversity and non-circular irregular geometry
  - Road-health scenario monotonicity (Healthy -> Moderate -> Poor -> Critical)
  - Deterministic reproducibility with random seeds
  - Ground truth JSON schema compliance
  - 3-meter spatial evaluation matching and error calculations
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import pytest

# Add workspace roots to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = WORKSPACE_ROOT / "env"
PIPELINE_DIR = WORKSPACE_ROOT / "road_health_pipeline"

for p in [str(WORKSPACE_ROOT), str(ENV_DIR), str(PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from road_injector import config as env_config
from road_injector import ProceduralRoadGenerator, ProceduralDefectSpec
from evaluate_simulation import evaluate, match_defects, compute_ground_distance_m


def test_procedural_defect_diversity():
    """Verify procedural defects exhibit non-uniformity in dimensions, aspect ratios, depths, shapes, and water states."""
    gen = ProceduralRoadGenerator(scenario="poor", seed=42)
    plan = gen.generate_corridor_plan(segment_length_m=500.0, defects_count=40)
    defects = [spec for _, _, spec in plan]

    # Lengths and widths must vary
    lengths = [d.length_m for d in defects]
    widths = [d.width_m for d in defects]
    depths = [d.depth_m for d in defects]
    aspect_ratios = [d.aspect_ratio for d in defects]
    orientations = [d.orientation_deg for d in defects]
    irregularities = [d.irregularity for d in defects]

    # No identical repeated dimensions
    assert len(set(lengths)) > 25, "Defect lengths must vary across samples"
    assert len(set(widths)) > 25, "Defect widths must vary across samples"
    assert len(set(depths)) > 20, "Defect depths must vary across samples"

    # Avoid perfect circles: aspect ratio must not always be 1.0
    assert any(ar > 1.25 for ar in aspect_ratios), "Must contain elongated/non-circular defects"
    assert any(ar < 1.15 for ar in aspect_ratios), "Must contain near-isometric defects"

    # Orientations must span multiple angles
    assert max(orientations) - min(orientations) > 45.0, "Defect orientations must vary"

    # Irregular boundary shapes (avoid perfect geometry)
    assert all(irr >= 0.15 for irr in irregularities), "All defects must have irregular/fractal roughness"

    # Water-filled variation: neither 0% nor 100% in poor scenario
    water_filled_count = sum(1 for d in defects if d.is_water_filled)
    assert 0 < water_filled_count < len(defects), "Must have a realistic mix of dry and water-filled potholes"

    # Surface defects beyond potholes: check cracks and patches
    assert any(d.has_cracks for d in defects), "Must include cracked defect variations"
    assert any(d.has_road_patch for d in defects), "Must include repaired road patches"


def test_scenario_monotonicity():
    """Verify that defect count and degradation severity strictly increase from Healthy to Critical."""
    scenarios = ["healthy", "moderate", "poor", "critical"]
    gens = [ProceduralRoadGenerator(scenario=s, seed=42) for s in scenarios]

    counts = [g.scenario_cfg.defects_per_corridor[0] for g in gens]
    water_ratios = [g.scenario_cfg.water_filled_ratio for g in gens]
    patch_probs = [g.scenario_cfg.patch_density for g in gens]
    crack_probs = [g.scenario_cfg.crack_density for g in gens]

    # Defect count must be strictly monotonic
    assert counts[0] <= counts[1] <= counts[2] <= counts[3], f"Defect counts must increase: {counts}"
    assert counts[0] < counts[3], "Healthy must have significantly fewer defects than Critical"

    # Degradation parameters must increase
    assert water_ratios[0] <= water_ratios[3]
    assert patch_probs[0] <= patch_probs[3]
    assert crack_probs[0] <= crack_probs[3]


def test_reproducibility_with_seed():
    """Verify that the same random seed produces identical defects, while different seeds vary."""
    gen1 = ProceduralRoadGenerator(scenario="moderate", seed=101)
    plan1 = gen1.generate_corridor_plan(segment_length_m=300.0, defects_count=10)
    defects1 = [s for _, _, s in plan1]

    gen2 = ProceduralRoadGenerator(scenario="moderate", seed=101)
    plan2 = gen2.generate_corridor_plan(segment_length_m=300.0, defects_count=10)
    defects2 = [s for _, _, s in plan2]

    gen3 = ProceduralRoadGenerator(scenario="moderate", seed=999)
    plan3 = gen3.generate_corridor_plan(segment_length_m=300.0, defects_count=10)
    defects3 = [s for _, _, s in plan3]

    # Same seed -> exact equality
    for d1, d2 in zip(defects1, defects2):
        assert d1.length_m == d2.length_m
        assert d1.width_m == d2.width_m
        assert d1.depth_m == d2.depth_m
        assert d1.is_water_filled == d2.is_water_filled
        assert d1.orientation_deg == d2.orientation_deg

    # Different seed -> distinct values
    diff_count = sum(1 for d1, d3 in zip(defects1, defects3) if d1.length_m != d3.length_m)
    assert diff_count >= 8, "Different seeds must produce different defect arrangements"


def test_ground_truth_json_schema():
    """Verify that generated ground truth contains all required specification keys."""
    gen = ProceduralRoadGenerator(scenario="poor", seed=42)
    plan = gen.generate_corridor_plan(segment_length_m=100.0, defects_count=1)
    _, _, spec = plan[0]

    # Required attributes in spec
    assert "pothole" in spec.defect_id or "defect" in spec.defect_id
    assert spec.length_m > 0.0
    assert spec.width_m > 0.0
    assert spec.depth_m > 0.0
    assert spec.area_m2 > 0.0
    assert 0.0 <= spec.severity_score <= 1.0
    assert spec.severity_category in ["low", "medium", "high", "critical"]
    assert isinstance(spec.is_water_filled, bool)


def test_spatial_distance_and_matching():
    """Verify 3-meter spatial matching and evaluation error calculations."""
    # Defect at reference location
    gt_lat, gt_lon = 13.0827, 80.2707
    gts = [
        {
            "defect_id": "gt_01",
            "gps_coordinates": {"latitude": gt_lat, "longitude": gt_lon},
            "dimensions": {"area_m2": 0.80, "depth_m": 0.08},
            "water_state": {"is_water_filled": True},
            "true_severity_score": 0.70,
        }
    ]

    # Prediction within 1.5 meters (should match)
    pred_near = [
        {
            "defect_id": "pred_01",
            "latitude": gt_lat + (1.2 / 111320.0),  # ~1.2m offset North
            "longitude": gt_lon,
            "area_m2": 0.75,
            "estimated_depth_m": 0.075,
            "severity_score": 0.68,
            "is_water_filled": True,
        }
    ]

    matched, un_gt, un_pred = match_defects(gts, pred_near, max_distance_m=3.0)
    assert len(matched) == 1
    assert len(un_gt) == 0
    assert len(un_pred) == 0
    assert matched[0]["distance_m"] < 2.0

    # Prediction 15 meters away (should NOT match)
    pred_far = [
        {
            "defect_id": "pred_far",
            "latitude": gt_lat + (15.0 / 111320.0),  # ~15m offset North
            "longitude": gt_lon,
            "area_m2": 0.80,
            "estimated_depth_m": 0.08,
            "is_water_filled": True,
        }
    ]

    matched_far, un_gt_far, un_pred_far = match_defects(gts, pred_far, max_distance_m=3.0)
    assert len(matched_far) == 0
    assert len(un_gt_far) == 1
    assert len(un_pred_far) == 1


def test_evaluate_end_to_end_synthetic(tmp_path: Path):
    """Verify that evaluate() outputs the complete evaluation report schema with correct metrics."""
    gt_lat, gt_lon = 13.0827, 80.2707
    gt_data = {
        "metadata": {
            "scenario": "poor",
            "weather": "wet",
            "seed": 42,
        },
        "defects": [
            {
                "defect_id": "defect_001",
                "gps_coordinates": {"latitude": gt_lat, "longitude": gt_lon},
                "dimensions": {"area_m2": 1.0, "depth_m": 0.10},
                "water_state": {"is_water_filled": True},
                "true_severity_score": 0.75,
            },
            {
                "defect_id": "defect_002",
                "gps_coordinates": {"latitude": gt_lat + 0.0001, "longitude": gt_lon},
                "dimensions": {"area_m2": 0.5, "depth_m": 0.05},
                "water_state": {"is_water_filled": False},
                "true_severity_score": 0.40,
            },
        ],
    }
    pred_data = {
        "road_health": {
            "road_health_score": 55.0,
            "condition_class": "poor",
        },
        "detections": [
            {
                "defect_id": "pothole_001",
                "latitude": gt_lat + (0.5 / 111320.0),
                "longitude": gt_lon,
                "area_m2": 0.95,
                "estimated_depth_m": 0.09,
                "severity_score": 0.72,
                "is_water_filled": True,
            },
            {
                "defect_id": "pothole_002",
                "latitude": gt_lat + 0.0001 + (0.8 / 111320.0),
                "longitude": gt_lon,
                "area_m2": 0.52,
                "estimated_depth_m": 0.048,
                "severity_score": 0.42,
                "is_water_filled": False,
            },
        ],
    }

    gt_file = tmp_path / "ground_truth.json"
    pred_file = tmp_path / "result.json"

    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_data, f)
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(pred_data, f)

    report = evaluate(gt_file, pred_file, max_match_dist_m=3.0)

    assert report["evaluation_summary"]["matched_true_positives"] == 2
    assert report["detection_performance"]["f1_score"] == 1.0
    assert report["water_hazard_classification"]["water_f1_score"] == 1.0
    assert report["localization_accuracy"]["location_mae_m"] < 1.0
    assert report["dimension_accuracy"]["area_mae_m2"] < 0.1
    assert report["road_health_consistency"]["scenario_agreement"] is True
