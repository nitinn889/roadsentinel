#!/usr/bin/env python3
"""test_day4_fixes.py
--------------------
Focused test suite for Day 4 DINOv2 + SAM2 perception stabilization:
1. Patch coordinate mapping (orientation, no x/y swap or transpose)
2. Road mask geometry & corridor prior
3. Component consolidation & touching mask merging
4. Union mask area & bbox recomputation
5. Schema validity on persisted Day 4 records
6. Batch runner interface & manifest compatibility
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PIPELINE_ROOT = REPO_ROOT / "road_health_pipeline"

for p in (str(HERE), str(REPO_ROOT), str(PIPELINE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_contract import load_feature_record, validate_feature_record
from inference.pothole_localizer import PotholeLocalizer
from common.schemas import CandidateRegion, SegmentationResult
from run_sam2_dino import collect_images_from_input


class Day4FixesTest(unittest.TestCase):
    def test_patch_coordinate_mapping(self) -> None:
        """Verify grid row/col maps to original image y/x without transposition."""
        grid_size = 37
        h, w = 512, 512
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        # Put high score at row 10 (y), col 25 (x)
        grid[10, 25] = 1.0

        amap = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
        max_y, max_x = np.unravel_index(np.argmax(amap), amap.shape)

        # Expected center: row 10 -> (10 + 0.5)/37 * 512 = 145.3 px (y)
        #                  col 25 -> (25 + 0.5)/37 * 512 = 353.1 px (x)
        self.assertAlmostEqual(max_y, (10 + 0.5) / grid_size * h, delta=2.0)
        self.assertAlmostEqual(max_x, (25 + 0.5) / grid_size * w, delta=2.0)

    def test_road_prior_geometry(self) -> None:
        """Verify camera-aware nadir prior produces correct non-full-frame road corridor."""
        from config import CONFIG
        from inference.sam2_mask import RoadMasker

        # Create a mock instance with CPU to test prior geometry
        masker = RoadMasker.__new__(RoadMasker)
        masker.camera_mode = "nadir"
        prior = masker._road_prior(512, 512)

        self.assertEqual(prior.shape, (512, 512))
        ratio = float(prior.mean())
        self.assertGreater(ratio, 0.40)
        self.assertLess(ratio, 0.94)

    def test_touching_component_consolidation(self) -> None:
        """Verify touching candidate masks merge and recompute bbox/area properly."""
        localizer = PotholeLocalizer()
        h, w = 200, 200

        # Create two vertically adjacent fragments of one defect
        m1 = np.zeros((h, w), dtype=bool)
        m1[20:50, 40:80] = True   # y: 20..49, x: 40..79

        m2 = np.zeros((h, w), dtype=bool)
        m2[50:80, 40:80] = True   # y: 50..79, x: 40..79 (touching m1 at y=50!)

        c1 = CandidateRegion(
            mask=m1,
            bbox_xyxy=[40, 20, 80, 50],
            anomaly_score=0.4,
            pothole_confidence=0.7,
            defect_type="road_defect",
            shape_circularity=0.5,
            aspect_ratio=1.3,
            surrounding_damage=0.1,
        )
        c2 = CandidateRegion(
            mask=m2,
            bbox_xyxy=[40, 50, 80, 80],
            anomaly_score=0.45,
            pothole_confidence=0.75,
            defect_type="road_defect",
            shape_circularity=0.5,
            aspect_ratio=1.3,
            surrounding_damage=0.1,
        )

        consolidated = localizer._consolidate_candidates([c1, c2])
        self.assertEqual(len(consolidated), 1)

        merged = consolidated[0]
        self.assertEqual(merged.bbox_xyxy, [40, 20, 80, 80])
        self.assertEqual(int(merged.mask.sum()), int(m1.sum() + m2.sum()))
        self.assertEqual(merged.defect_type, "road_defect")

    def test_schema_validity_persisted_outputs(self) -> None:
        """Verify all outputs in sam2_dino/outputs pass feature contract schema."""
        out_root = HERE / "outputs"
        tested = 0
        for sample_dir in out_root.iterdir():
            if not sample_dir.is_dir():
                continue
            feat_path = sample_dir / "features.json"
            if feat_path.exists():
                rec = load_feature_record(feat_path)
                validate_feature_record(rec)
                self.assertIn("defect_count", rec)
                self.assertIn("current_severity", rec)
                self.assertIn("defects", rec)
                tested += 1
        self.assertGreaterEqual(tested, 4)

    def test_batch_runner_manifest_collection(self) -> None:
        """Verify run_sam2_dino collects images from common manifest."""
        manifest_path = REPO_ROOT / "benchmark" / "common_candidate" / "manifest.csv"
        items = collect_images_from_input(manifest_path)
        self.assertGreaterEqual(len(items), 18)
        # Check first item structure
        img_id, img_path, cond = items[0]
        self.assertTrue(img_path.exists())
        self.assertTrue(isinstance(img_id, str))


if __name__ == "__main__":
    unittest.main()
