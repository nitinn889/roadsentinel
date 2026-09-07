from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT_DIR = HERE.parent / "sam2_dino"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from feature_adapter import MODEL_FEATURES, TARGET, build_model_features
from feature_contract import validate_feature_record


class AdapterTest(unittest.TestCase):
    def test_reads_shared_schema_without_predicting(self) -> None:
        record = {
            "schema_version": "1.0.0",
            "image_id": "MOCK_IMAGE",
            "segment_id": "SEG_001",
            "day": 1,
            "original_filename": "mock.jpg",
            "condition": None,
            "current_severity": None,
            "crack_area_ratio": 0.0,
            "defect_area_ratio": 0.0,
            "defect_count": 0,
            "surface_anomaly_score": 0.1,
            "water_flag": False,
            "defects": [],
        }
        scenario = {
            "scenario_status": "MOCK_INTERFACE_ONLY",
            "rainfall_level": 12.0,
            "traffic_level": 1500.0,
            "temperature": 32.0,
            "water_exposure": 0.4,
            "days_ahead": 30,
        }
        validate_feature_record(record)
        features = build_model_features(record, scenario)
        self.assertEqual(tuple(features), MODEL_FEATURES)
        self.assertTrue(math.isnan(features["current_severity"]))
        self.assertEqual(TARGET, "future_severity")


if __name__ == "__main__":
    unittest.main()
