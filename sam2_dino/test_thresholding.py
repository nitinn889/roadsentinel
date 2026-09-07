from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "road_health_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config import CONFIG
from inference.anomaly_detector import AnomalyDetector


class ActiveThresholdTest(unittest.TestCase):
    def test_policy_uses_configured_percentile_and_clip(self) -> None:
        scores = np.array([0.0, 0.1, 0.2, 0.9], dtype=np.float32)
        threshold, image_score, details = AnomalyDetector.select_active_threshold(scores)
        expected = np.clip(
            np.percentile(scores[scores > 1e-6], CONFIG.active_anomaly_percentile),
            CONFIG.active_anomaly_floor,
            CONFIG.active_anomaly_ceiling,
        )
        self.assertAlmostEqual(threshold, float(expected))
        self.assertAlmostEqual(image_score, float(np.percentile(scores, 95.0)))
        self.assertEqual(details["policy"], "positive_score_percentile_with_clip")

    def test_empty_scores_do_not_detect_everything(self) -> None:
        threshold, image_score, details = AnomalyDetector.select_active_threshold(
            np.array([], dtype=np.float32)
        )
        self.assertEqual(threshold, 1.0)
        self.assertEqual(image_score, 0.0)
        self.assertEqual(details["clip_applied"], "empty_scores")


if __name__ == "__main__":
    unittest.main()
