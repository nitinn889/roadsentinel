from __future__ import annotations

import unittest

import numpy as np

from export_features import build_feature_record, parse_temporal_image_path
from feature_contract import validate_feature_record


class FeatureContractTest(unittest.TestCase):
    def test_export_from_inference_shape(self) -> None:
        inference = {
            "image_id": "MOCK_IMAGE",
            "image_shape": [100, 200, 3],
            "anomaly_score": 0.25,
            "detections": [
                {
                    "bbox": [10, 20, 30, 40],
                    "mask_area_pixels": 200,
                    "defect_type": "crack",
                    "severity": {"severity_score": 0.3},
                    "is_water_filled": False,
                }
            ],
        }
        record = build_feature_record(
            inference,
            segment_id="SEG_001",
            day=1,
            original_filename="mock.jpg",
            condition=None,
        )
        validate_feature_record(record)
        self.assertEqual(record["defect_count"], 1)
        self.assertAlmostEqual(record["defect_area_ratio"], 0.01)
        self.assertAlmostEqual(record["crack_area_ratio"], 0.01)

    def test_temporal_path(self) -> None:
        self.assertEqual(
            parse_temporal_image_path("SEG_001/day_10/road.jpg"),
            ("SEG_001", 10, "road.jpg"),
        )

    def test_final_masks_use_union_and_export_geometry(self) -> None:
        first = np.zeros((4, 4), dtype=bool)
        second = np.zeros((4, 4), dtype=bool)
        first[0:2, 0:2] = True
        second[1:3, 1:3] = True
        inference = {
            "image_id": "MASK_TEST",
            "image_shape": [4, 4, 3],
            "anomaly_score": 0.5,
            "detections": [
                {
                    "defect_type": "crack",
                    "severity": {"severity_score": 30.0},
                    "is_water_filled": False,
                },
                {
                    "defect_type": "pothole",
                    "severity": {"severity_score": 0.6},
                    "is_water_filled": False,
                },
            ],
        }
        record = build_feature_record(
            inference,
            segment_id="SEG_001",
            day=1,
            original_filename="mask_test.png",
            defect_masks=[first, second],
            mask_paths=["defect_001_mask.png", "defect_002_mask.png"],
        )
        validate_feature_record(record)
        self.assertAlmostEqual(record["defect_area_ratio"], 7 / 16)
        self.assertAlmostEqual(record["crack_area_ratio"], 4 / 16)
        self.assertAlmostEqual(record["current_severity"], 0.6)
        self.assertEqual(record["defects"][0]["bbox"], [0, 0, 2, 2])
        self.assertAlmostEqual(record["defects"][0]["centroid_x"], 0.5)
        self.assertAlmostEqual(record["defects"][0]["centroid_y"], 0.5)


if __name__ == "__main__":
    unittest.main()
