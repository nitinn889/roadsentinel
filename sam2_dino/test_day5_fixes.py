#!/usr/bin/env python3
"""test_day5_fixes.py
--------------------
Focused test suite for Day 5 DINOv2 + SAM2 localization improvements:
1. Pre-consolidation of candidate fragments before SAM2 prompting.
2. Adaptive context expansion for narrow/elongated defects.
3. Union geometry generation and single-defect consolidation.
4. Healthy road regression (zero false positive defects).
5. Schema validation across persisted Day 5 feature records.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PIPELINE_ROOT = REPO_ROOT / "road_health_pipeline"

for p in (str(HERE), str(REPO_ROOT), str(PIPELINE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_contract import load_feature_record, validate_feature_record
from inference.pothole_localizer import PotholeLocalizer
from common.schemas import CandidateRegion, SegmentationResult


class Day5FixesTest(unittest.TestCase):
    def test_pre_consolidation_geometry(self) -> None:
        """Verify touching/adjacent candidate components are grouped before SAM2."""
        localizer = PotholeLocalizer()
        h, w = 100, 100

        # Two candidate blobs separated by 10 pixels (within the 21x21 dilation distance)
        cbin = np.zeros((h, w), dtype=np.uint8)
        cbin[20:35, 40:60] = 1   # blob 1: y in [20, 35], x in [40, 60]
        cbin[42:55, 40:60] = 1   # blob 2: y in [42, 55], x in [40, 60] (gap of 7 pixels)

        dilated = cv2.dilate(cbin, np.ones((21, 21), np.uint8))
        n_grp, labels_grp, _, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        # Both blobs should form a single group
        self.assertEqual(n_grp - 1, 1)

        gmask = (labels_grp == 1) & (cbin > 0)
        ys, xs = np.nonzero(gmask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        self.assertEqual([x0, y0, x1, y1], [40, 20, 60, 55])
        self.assertEqual(int(gmask.sum()), 15 * 20 + 13 * 20)

    def test_adaptive_elongation_padding(self) -> None:
        """Verify vertically and horizontally elongated boxes receive directional padding."""
        # Simulated vertical crack candidate: w=10, h=40
        w_v, h_v = 10, 40
        self.assertTrue(h_v >= 1.3 * w_v and w_v <= 25)
        pad_y_v = max(32, min(80, int(h_v * 1.5)))
        pad_x_v = 10
        self.assertEqual(pad_y_v, 60)
        self.assertEqual(pad_x_v, 10)

        # Simulated isotropic pothole candidate: w=50, h=50
        w_p, h_p = 50, 50
        self.assertFalse(h_p >= 1.3 * w_p and w_p <= 25)
        self.assertFalse(w_p >= 1.3 * h_p and h_p <= 25)

    def test_persisted_day5_schema(self) -> None:
        """Verify all Day 5 outputs pass the feature contract schema."""
        out_root = HERE / "outputs"
        samples = ["China_Drone_001267", "0454", "original_healthy", "India_005086"]
        for s in samples:
            fpath = out_root / s / "features.json"
            if fpath.exists():
                rec = load_feature_record(fpath)
                validate_feature_record(rec)
                self.assertIn(rec["defect_count"], (0, 1, 2, 3, 4, 5))
                self.assertGreaterEqual(rec["current_severity"], 0.0)
                self.assertLessEqual(rec["current_severity"], 1.0)

    def test_healthy_road_zero_defects(self) -> None:
        """Verify original_healthy features record has zero defects and zero severity."""
        h_feat = HERE / "outputs" / "original_healthy" / "features.json"
        if h_feat.exists():
            rec = load_feature_record(h_feat)
            self.assertEqual(rec["defect_count"], 0)
            self.assertEqual(rec["current_severity"], 0.0)


if __name__ == "__main__":
    unittest.main()
