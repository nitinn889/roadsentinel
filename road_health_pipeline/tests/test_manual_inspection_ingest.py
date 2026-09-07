"""
test_manual_inspection_ingest.py
--------------------------------
Unit and integration tests for user-captured Unreal Engine inspection ingestion:
- Input validation & blank/extreme frame rejection (all-black, all-white, empty, corrupt).
- Diagnostic error overlay creation.
- Scanning SEG_001 to SEG_008 across Days 1-20.
- Trend prediction on Days 1-14 with unseen evaluation on Days 15-20.
- Work order generation for observed and forecast risks.
- Rebuilding dashboard dataset in RoadSentinelTemporal20Day/v1 schema.
"""

import json
from pathlib import Path
import numpy as np
import cv2
import pytest

import ingest_manual_inspections as ingest
from ingest_manual_inspections import (
    ALL_SEGMENT_SPECS,
    create_diagnostic_error_overlay,
    generate_work_orders_for_inspections,
    ingest_all_manual_inspections,
    predict_held_out_robust,
    validate_inspection_image,
)


def test_validation_rejects_all_black_image(tmp_path):
    """Ensure pitch-black frames (e.g. nighttime SceneCapture2D without light) are rejected."""
    black_img_path = tmp_path / "black.png"
    black = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.imwrite(str(black_img_path), black)

    is_valid, reason, img = validate_inspection_image(black_img_path)
    assert is_valid is False
    assert reason is not None
    assert "pitch black" in reason.lower() or "exposure failure" in reason.lower()
    assert img is not None


def test_validation_rejects_all_white_image(tmp_path):
    """Ensure washed-out all-white frames (overexposed sun lux) are rejected."""
    white_img_path = tmp_path / "white.png"
    white = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(white_img_path), white)

    is_valid, reason, img = validate_inspection_image(white_img_path)
    assert is_valid is False
    assert reason is not None
    assert "all-white" in reason.lower() or "blown out" in reason.lower()
    assert img is not None


def test_validation_rejects_empty_and_corrupt_files(tmp_path):
    """Ensure missing, 0-byte, and non-image files are rejected."""
    missing_path = tmp_path / "nonexistent.png"
    is_valid, reason, img = validate_inspection_image(missing_path)
    assert is_valid is False
    assert "not found" in reason.lower()

    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")
    is_valid, reason, img = validate_inspection_image(empty_path)
    assert is_valid is False
    assert "too small" in reason.lower()

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"NOT_A_VALID_PNG_HEADER_DATA_STREAM" * 100)
    is_valid, reason, img = validate_inspection_image(corrupt_path)
    assert is_valid is False
    assert "corrupted" in reason.lower() or "failed decoding" in reason.lower()


def test_validation_accepts_genuine_road_image(tmp_path):
    """Ensure genuine road texture with varied pixel distribution passes validation."""
    valid_img_path = tmp_path / "road.png"
    rng = np.random.RandomState(42)
    # Simulate textured asphalt (grey 90-130 with variation)
    road = rng.randint(85, 140, (1080, 1920, 3), dtype=np.uint8)
    # Add road lane marking
    cv2.line(road, (960, 0), (960, 1080), (220, 220, 220), 10)
    cv2.imwrite(str(valid_img_path), road)

    is_valid, reason, img = validate_inspection_image(valid_img_path)
    assert is_valid is True
    assert reason is None
    assert img is not None
    assert img.shape == (1080, 1920, 3)


def test_diagnostic_error_overlay_generation(tmp_path):
    """Verify high-visibility diagnostic error overlay creates a valid 1080x1920 image with red warning elements."""
    overlay = create_diagnostic_error_overlay(
        raw_img=None,
        segment_id="SEG_003",
        day=5,
        weather="Night Highway with Lamps",
        reason="Frame rejected: image is pitch black (100.0% zero pixels)",
    )
    assert overlay.shape == (1080, 1920, 3)
    # Border should be red (BGR: 0, 0, 255)
    assert overlay[2, 2, 2] == 255  # Red channel active
    assert overlay[2, 2, 0] == 0    # Blue channel zero


def test_predict_held_out_robust_excludes_invalid_frames():
    """Ensure trend prediction only fits on valid training days in Days 1-14."""
    history = [
        {"day": 1, "is_valid": True, "severity_score": 0.10},
        {"day": 2, "is_valid": False, "validation_error": "All-black", "severity_score": None},
        {"day": 3, "is_valid": True, "severity_score": 0.16},
        {"day": 4, "is_valid": True, "severity_score": 0.19},
        {"day": 5, "is_valid": False, "validation_error": "All-white", "severity_score": None},
        {"day": 6, "is_valid": True, "severity_score": 0.25},
        {"day": 7, "is_valid": True, "severity_score": 0.28},
        {"day": 8, "is_valid": True, "severity_score": 0.32},
        {"day": 9, "is_valid": True, "severity_score": 0.36},
        {"day": 10, "is_valid": True, "severity_score": 0.40},
        {"day": 11, "is_valid": True, "severity_score": 0.43},
        {"day": 12, "is_valid": True, "severity_score": 0.47},
        {"day": 13, "is_valid": True, "severity_score": 0.51},
        {"day": 14, "is_valid": True, "severity_score": 0.55},
        # Evaluation days 15-20
        {"day": 15, "is_valid": True, "severity_score": 0.58},
        {"day": 16, "is_valid": True, "severity_score": 0.62},
        {"day": 17, "is_valid": False, "validation_error": "All-black", "severity_score": None},
        {"day": 18, "is_valid": True, "severity_score": 0.70},
        {"day": 19, "is_valid": True, "severity_score": 0.74},
        {"day": 20, "is_valid": True, "severity_score": 0.78},
    ]

    pred = predict_held_out_robust(history)
    assert pred["status"] == "valid_forecast"
    # Days 2 and 5 must be excluded from training
    assert 2 not in pred["training_days"]
    assert 5 not in pred["training_days"]
    assert 1 in pred["training_days"]
    assert 14 in pred["training_days"]
    assert pred["prediction_horizon_days"] == 7
    assert pred["predicted_future_day"] == 27
    assert 0.0 <= pred["predicted_future_severity"] <= 1.0
    assert pred["unseen_mae"] is not None


def test_work_order_generation_on_observed_and_forecast_risk():
    """Ensure work orders trigger on either observed current severity OR predicted future severity."""
    segment_list = [
        {
            "road_segment_id": "SEG_001",
            "location": {"latitude": 13.0827, "longitude": 80.2707},
            "current": {"day": 20, "severity_score": 0.20, "condition": "Healthy"},
            "prediction": {"predicted_future_severity": 0.25},
        },
        {
            "road_segment_id": "SEG_003",
            "location": {"latitude": 13.0828, "longitude": 80.2715},
            "current": {"day": 20, "severity_score": 0.45, "condition": "Moderate"},
            "prediction": {"predicted_future_severity": 0.72},  # Forecast triggers WO (>= 0.65)
        },
        {
            "road_segment_id": "SEG_004",
            "location": {"latitude": 13.0829, "longitude": 80.2720},
            "current": {"day": 20, "severity_score": 0.88, "condition": "Severe Pothole", "has_water_hazard": True},
            "prediction": {"predicted_future_severity": 0.95},
        },
    ]

    orders = generate_work_orders_for_inspections(segment_list, threshold=0.65)
    order_segs = [o["road_segment_id"] for o in orders]
    # SEG_001 is healthy -> no work order
    assert "SEG_001" not in order_segs
    # SEG_003 has forecast >= 0.65 -> work order generated
    assert "SEG_003" in order_segs
    # SEG_004 has severe pothole + water hazard -> P1 Immediate work order generated
    assert "SEG_004" in order_segs
    wo4 = next(o for o in orders if o["road_segment_id"] == "SEG_004")
    assert wo4["priority"] == "P1 - Immediate"


def test_all_segment_specs_include_seg001_to_seg008():
    """Verify ALL_SEGMENT_SPECS covers SEG_001 through SEG_008 with stable IDs."""
    ids = [s.segment_id for s in ALL_SEGMENT_SPECS]
    assert len(ids) == 8
    assert ids == [f"SEG_{i:03d}" for i in range(1, 9)]


def test_ingest_all_manual_inspections_e2e(tmp_path, monkeypatch):
    """Integration test: ingest mock inspection captures with valid, blank, and unrecorded segments."""
    inspections_dir = tmp_path / "manual_inspections"
    output_dir = tmp_path / "temporal_20_day"

    # Setup fixtures for Days 1 to 20
    rng = np.random.RandomState(42)
    for day in range(1, 21):
        day_dir = inspections_dir / f"day_{day:02d}"
        # SEG_001: Genuine road capture
        s1 = day_dir / "SEG_001"
        s1.mkdir(parents=True)
        img1 = rng.randint(90, 130, (1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(s1 / "raw.png"), img1)
        (s1 / "metadata.json").write_text(json.dumps({
            "day": day,
            "segment_id": "SEG_001",
            "weather_preset": "Dense Atmospheric Fog",
            "world_coordinates": {"along_m": 35.0, "across_m": -1.35},
        }))

        # SEG_002: Day 1-10 all-black failure, Day 11-20 valid road
        s2 = day_dir / "SEG_002"
        s2.mkdir(parents=True)
        if day <= 10:
            img2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        else:
            img2 = rng.randint(90, 130, (1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(s2 / "raw.png"), img2)
        (s2 / "metadata.json").write_text(json.dumps({
            "day": day,
            "segment_id": "SEG_002",
            "weather_preset": "Night Highway with Lamps" if day <= 10 else "Heavy Rain & Wet Road",
            "world_coordinates": {"along_m": 95.0, "across_m": 1.35},
        }))

        # SEG_003: Overexposed all-white
        s3 = day_dir / "SEG_003"
        s3.mkdir(parents=True)
        img3 = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(s3 / "raw.png"), img3)
        (s3 / "metadata.json").write_text(json.dumps({
            "day": day,
            "segment_id": "SEG_003",
            "weather_preset": "Clear Noon (70° Sun)",
            "world_coordinates": {"along_m": 155.0, "across_m": -1.35},
        }))
        # SEG_004 through SEG_008 intentionally omitted to test unrecorded handling

    # Mock infer to avoid loading the full PyTorch models during unit test
    class MockDefect:
        def to_dict(self):
            return {"defect_id": "mock_pothole", "bbox": [100, 100, 200, 200], "defect_type": "pothole"}

    class MockLegacy:
        anomaly_score = 0.45

    class MockHealth:
        road_health_score = 75.0

    class MockResult:
        potholes = [MockLegacy()]
        detections = [MockDefect()]
        road_health = MockHealth()
        anomaly_score = 0.45
        anomaly_threshold = 0.35
        warnings = []

    def mock_infer(*args, **kwargs):
        return MockResult()

    monkeypatch.setattr(ingest, "infer", mock_infer)
    monkeypatch.setattr(ingest, "load_pipeline", lambda **k: "mock_pipeline")
    monkeypatch.setattr(ingest, "load_rgb", lambda p: np.zeros((1080, 1920, 3), dtype=np.uint8))
    monkeypatch.setattr(
        ingest,
        "generate_visual_overlays",
        lambda rgb, p, r: {
            "detection_overlay": np.zeros((100, 100, 3), dtype=np.uint8),
            "severity_overlay": np.zeros((100, 100, 3), dtype=np.uint8),
            "road_health_overlay": np.zeros((100, 100, 3), dtype=np.uint8),
        },
    )

    result = ingest_all_manual_inspections(
        inspections_dir=inspections_dir,
        output_dir=output_dir,
        work_order_threshold=0.65,
        rebuild_dashboard=True,
    )

    assert result["schema_version"] == "RoadSentinelTemporal20Day/v1"
    # Covers all 8 segments
    seg_map = {s["road_segment_id"]: s for s in result["segments"]}
    assert len(seg_map) == 8
    assert "SEG_001" in seg_map
    assert "SEG_008" in seg_map

    # SEG_001: All 20 days valid
    s1_hist = seg_map["SEG_001"]["history"]
    assert len(s1_hist) == 20
    assert all(x["is_valid"] is True for x in s1_hist)

    # SEG_002: Days 1-10 rejected, Days 11-20 valid
    s2_hist = seg_map["SEG_002"]["history"]
    assert len(s2_hist) == 20
    assert all(x["is_valid"] is False for x in s2_hist[:10])
    assert all(x["is_valid"] is True for x in s2_hist[10:])

    # SEG_003: All days rejected (all-white)
    s3_hist = seg_map["SEG_003"]["history"]
    assert all(x["is_valid"] is False for x in s3_hist)
    assert "all-white" in s3_hist[0]["validation_error"].lower() or "overexposed" in s3_hist[0]["validation_error"].lower()

    # SEG_007: Unrecorded
    s7_hist = seg_map["SEG_007"]["history"]
    assert all(x["is_valid"] is False for x in s7_hist)
    assert s7_hist[0]["condition"] == "No Capture"

    # Verify output files exist
    assert (inspections_dir / "temporal_results.json").is_file()
    assert (output_dir / "temporal_results.json").is_file()
    assert (output_dir / "work_orders.json").is_file()
