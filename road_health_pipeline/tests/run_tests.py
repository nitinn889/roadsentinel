from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Add pipeline root to path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import tests.test_analytics_layer as tal
import tests.test_spatial_index as tsi


def run_all_unit_tests():
    print("=" * 65)
    print("      RUNNING ROADSENTINEL ANALYTICS & PREDICTION TEST SUITE     ")
    print("=" * 65)

    test_functions = [
        ("Area Estimation & Ray-tracing", tal.test_area_estimation),
        ("Depth Metric Calculations", tal.test_depth_metrics_computation),
        ("Severity Calculation (With & Without Depth)", tal.test_severity_calculation_with_and_without_depth),
        ("Road Health Scoring & Deductions", tal.test_road_health_scoring),
        ("Segment Aggregation & Defect Traceability", tal.test_segment_aggregation_traceability),
        ("Spatial Index KD-Tree & Geofencing", tsi.test_spatial_index_kd_tree),
        ("CARLA Ground Truth Validation", tal.test_carla_ground_truth_comparison),
        ("Temporal Prediction Interface", tal.test_temporal_prediction_interface),
        ("CARLA Temporal Dataset & Split Integrity", tal.test_carla_synthetic_temporal_dataset_and_evaluator),
    ]

    passed = 0
    failed = 0

    for name, func in test_functions:
        try:
            func()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}")
            traceback.print_exc()
            failed += 1

    # End to end test
    try:
        tmp_dir = PIPELINE_ROOT / "outputs" / "test_run"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tal.test_end_to_end_pipeline_and_json_schema(tmp_dir)
        print(f" [PASS] End-to-End Pipeline & JSON Schema Validation")
        passed += 1
    except Exception as e:
        print(f" [FAIL] End-to-End Pipeline & JSON Schema Validation")
        traceback.print_exc()
        failed += 1

    print("=" * 65)
    print(f" Test Summary: {passed} passed, {failed} failed out of {passed + failed} total.")
    print("=" * 65)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_unit_tests()
