"""Unit and integration tests for Citizen Road-Damage Reporting & AI Verification.

Tests:
1. Valid pothole/defect image analysis (verification & detection)
2. Valid non-pothole road image (clean road baseline -> RETAKE / NO VERIFIED DAMAGE)
3. Invalid / non-image upload rejection
4. No GPS handling -> LOCATION NOT PROVIDED
5. Valid GPS within corridor -> ON_ROADSENTINEL_ROUTE & segment assignment
6. Valid GPS outside corridor -> OUTSIDE_MONITORED_ROUTE
7. Low-confidence / unfamiliar domain detection handling
8. Report persistence, unique ID generation (RS-CR-XXXX), and manifest reload
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

import citizen_reporting
from citizen_reporting import (
    CitizenReportingPipeline,
    analyze_citizen_report,
    evaluate_report_gps,
    load_all_reports,
    save_citizen_report,
    validate_image_bytes,
    STATUS_MANUAL_REVIEW,
    STATUS_RETAKE,
    STATUS_VERIFIED,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCitizenReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize pipeline on CPU for deterministic CI testing
        cls.pipeline = CitizenReportingPipeline(device="cpu")

        # Test image paths from repo
        cls.crack_repair_img = REPO_ROOT / "benchmark/common_candidate/images/China_Drone_000104.jpg"
        cls.healthy_img = REPO_ROOT / "benchmark/common_candidate/images/original_healthy.jpg"

    def test_image_validation_valid_and_invalid(self):
        """Test image decoding and corrupt input rejection."""
        # 1. Invalid bytes
        is_val, img, msg = validate_image_bytes(b"corrupted_binary_stream_not_an_image")
        self.assertFalse(is_val)
        self.assertIsNone(img)
        self.assertIn("validation failed", msg.lower())

        # 2. Empty bytes
        is_val, img, msg = validate_image_bytes(b"")
        self.assertFalse(is_val)
        self.assertIsNone(img)

        # 3. Valid bytes from existing file
        if self.healthy_img.exists():
            valid_bytes = self.healthy_img.read_bytes()
            is_val, img, msg = validate_image_bytes(valid_bytes)
            self.assertTrue(is_val)
            self.assertIsNotNone(img)
            self.assertEqual(len(img.shape), 3)

    def test_valid_defect_image_verification(self):
        """Test detection and verification on a valid damage image."""
        if not self.crack_repair_img.exists():
            self.skipTest(f"Sample {self.crack_repair_img} not found")

        bytes_in = self.crack_repair_img.read_bytes()
        res = analyze_citizen_report(bytes_in, self.pipeline, latitude=8.8935, longitude=76.6145)

        self.assertTrue(res["is_valid"])
        self.assertTrue(res["defect_detected"])
        self.assertIn(res["status"], [STATUS_VERIFIED, STATUS_MANUAL_REVIEW])
        self.assertGreater(res["yolo_confidence"], 0.30)
        self.assertIn(res["reliability_level"], ["HIGH", "MEDIUM", "LOW"])
        self.assertGreater(res["model_severity"], 0.0)
        self.assertIsNotNone(res["annotated_image_bytes"])

    def test_valid_non_pothole_clean_road_image(self):
        """Test that pristine/healthy road image returns NO VERIFIED DAMAGE."""
        if not self.healthy_img.exists():
            self.skipTest(f"Sample {self.healthy_img} not found")

        bytes_in = self.healthy_img.read_bytes()
        res = analyze_citizen_report(bytes_in, self.pipeline)

        self.assertTrue(res["is_valid"])
        self.assertFalse(res["defect_detected"])
        self.assertEqual(res["status"], STATUS_RETAKE)
        self.assertEqual(res["defect_type"], "None")
        self.assertEqual(res["model_severity"], 0.0)

    def test_no_gps_handling(self):
        """Test submission when user omits GPS coordinates."""
        gps = evaluate_report_gps(None, None)
        self.assertFalse(gps["has_gps"])
        self.assertEqual(gps["gps_display"], "LOCATION NOT PROVIDED")
        self.assertEqual(gps["geofence_status"], "NO_GPS_SUPPLIED")
        self.assertEqual(gps["nearest_segment"], "N/A")

    def test_valid_gps_inside_route_corridor(self):
        """Test GPS inside the 50m corridor correctly maps to route and segment."""
        # Start of SEG_001 is at (8.8932, 76.6141)
        lat = 8.89325
        lon = 76.61415
        gps = evaluate_report_gps(lat, lon)

        self.assertTrue(gps["has_gps"])
        self.assertEqual(gps["geofence_status"], "ON_ROADSENTINEL_ROUTE")
        self.assertTrue(gps["is_on_route"])
        self.assertIn(gps["nearest_segment"], ["SEG_001", "SEG_002"])
        self.assertLessEqual(gps["distance_to_corridor_m"], 50.0)

    def test_valid_gps_outside_route_corridor(self):
        """Test GPS outside corridor is flagged as OUTSIDE_MONITORED_ROUTE."""
        # 10 km away from start coordinate
        lat = 9.0500
        lon = 76.8000
        gps = evaluate_report_gps(lat, lon)

        self.assertTrue(gps["has_gps"])
        self.assertEqual(gps["geofence_status"], "OUTSIDE_MONITORED_ROUTE")
        self.assertFalse(gps["is_on_route"])
        self.assertGreater(gps["distance_to_corridor_m"], 50.0)

    def test_invalid_gps_range(self):
        """Test invalid latitude or longitude values are safely caught."""
        gps = evaluate_report_gps(999.0, -250.0)
        self.assertFalse(gps["has_gps"])
        self.assertEqual(gps["geofence_status"], "INVALID_COORDINATES")

    def test_report_persistence_and_reload(self):
        """Test saving a report locally, generating sequential ID, and reloading from manifest."""
        import cv2

        # Create valid dummy JPEG bytes so saved files are fully identifiable images
        valid_bytes = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))[1].tobytes()

        mock_analysis = {
            "status": STATUS_VERIFIED,
            "decision_reason": "Unit test verified defect",
            "is_valid": True,
            "defect_detected": True,
            "defect_type": "Pothole",
            "yolo_confidence": 0.88,
            "reliability_score": 0.92,
            "reliability_level": "HIGH",
            "domain_distance": 0.25,
            "domain_familiarity": "IN_DOMAIN",
            "model_severity": 0.78,
            "severity_band": "HIGH",
            "detections": [{"class": "D40", "confidence": 0.88, "bbox": [10, 10, 50, 50]}],
            "has_sam2_mask": True,
            "mask_area_ratio": 0.045,
            "gps_info": evaluate_report_gps(8.8932, 76.6141),
            "annotated_image_bytes": valid_bytes,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_reports_dir = Path(tmpdir)

            # Save report into isolated temporary directory
            report_id = save_citizen_report(valid_bytes, mock_analysis, reports_dir=temp_reports_dir)
            self.assertTrue(report_id.startswith("RS-CR-"))
            self.assertEqual(report_id, "RS-CR-0001")

            # Verify files were created
            raw_file = temp_reports_dir / f"{report_id}_raw.jpg"
            ann_file = temp_reports_dir / f"{report_id}_annotated.jpg"
            meta_file = temp_reports_dir / f"{report_id}.json"
            self.assertTrue(raw_file.exists())
            self.assertTrue(ann_file.exists())
            self.assertTrue(meta_file.exists())

            # Verify saved images are valid decodable images
            with Image.open(raw_file) as img:
                img.verify()
            with Image.open(ann_file) as img:
                img.verify()

            # Reload reports manifest from temporary directory
            all_reports = load_all_reports(reports_dir=temp_reports_dir)
            self.assertEqual(len(all_reports), 1)

            saved_item = all_reports[0]
            self.assertEqual(saved_item["report_id"], report_id)
            self.assertEqual(saved_item["status"], STATUS_VERIFIED)
            self.assertEqual(saved_item["defect_type"], "Pothole")
            self.assertEqual(saved_item["geofence_status"], "ON_ROADSENTINEL_ROUTE")
            self.assertEqual(saved_item["nearest_segment"], "SEG_001")



class TestSafeImageRendering(unittest.TestCase):
    """Test defensive handling of corrupted, empty, or missing images."""

    def test_safe_render_handles_corrupt_and_missing_inputs(self):
        """Ensure safe_render_image does not raise unhandled exceptions on invalid data."""
        from app import safe_render_image
        import tempfile

        # 1. Test with None
        try:
            safe_render_image(None)
        except Exception as e:
            self.fail(f"safe_render_image raised unexpectedly on None: {e}")

        # 2. Test with empty bytes
        try:
            safe_render_image(b"")
        except Exception as e:
            self.fail(f"safe_render_image raised unexpectedly on empty bytes: {e}")

        # 3. Test with non-existent file
        try:
            safe_render_image(Path("/tmp/nonexistent_image_12345.jpg"))
        except Exception as e:
            self.fail(f"safe_render_image raised unexpectedly on missing file: {e}")

        # 4. Test with corrupted file containing text bytes
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(b"mock_raw_jpeg_data")
            tf_path = Path(tf.name)

        try:
            safe_render_image(tf_path)
        except Exception as e:
            self.fail(f"safe_render_image raised unexpectedly on corrupt file: {e}")
        finally:
            if tf_path.exists():
                tf_path.unlink()


if __name__ == "__main__":
    unittest.main()

