"""
test_inspection_tool.py
-----------------------
Unit tests for the RoadSentinel user-driven inspection capture tool.

These tests run entirely without Unreal Engine — they verify:
  1. Manifest segment persistence across all 20 days.
  2. Capture path construction.
  3. PNG validation (blank/small rejection and acceptance).
  4. Metadata schema correctness.

Run with:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest env/scripts/tests/test_inspection_tool.py -v
"""

from __future__ import annotations

import io
import json
import struct
import zlib
from pathlib import Path

import pytest

# Make the scripts directory importable
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_ROOT = WORKSPACE_ROOT / "env" / "output" / "temporal_20_day" / "manifests"

import rs_inspection_capture as ic


# ===========================================================================
# 1. Manifest segment persistence
# ===========================================================================

class TestManifestPersistence:
    """Verify that segment along_m / across_m are identical across all 20 days.

    This is the core guarantee of the temporal inspection system: a camera
    posed at SEG_003 on Day 1 points at exactly the same road patch as one
    posed at SEG_003 on Day 20.
    """

    @pytest.fixture(scope="class")
    def all_manifests(self):
        manifests = {}
        for day in range(1, 21):
            manifests[day] = ic.load_manifest(day)
        return manifests

    def test_all_20_manifests_present(self, all_manifests):
        assert len(all_manifests) == 20, "Expected 20 day manifests"

    def test_segment_ids_stable(self, all_manifests):
        """Every manifest must list the same set of segment IDs."""
        reference_ids = sorted(
            s["road_segment_id"] for s in all_manifests[1].get("segments", [])
        )
        for day, manifest in all_manifests.items():
            ids = sorted(s["road_segment_id"] for s in manifest.get("segments", []))
            assert ids == reference_ids, (
                f"Day {day} has different segment IDs: {ids} vs {reference_ids}"
            )

    def test_along_m_stable_across_days(self, all_manifests):
        """along_m (road longitudinal position) must not change between days."""
        reference_positions = {
            s["road_segment_id"]: s["along_m"]
            for s in all_manifests[1].get("segments", [])
        }
        for day in range(2, 21):
            for seg in all_manifests[day].get("segments", []):
                sid = seg["road_segment_id"]
                assert sid in reference_positions, f"Day {day} has unexpected segment {sid}"
                assert seg["along_m"] == pytest.approx(reference_positions[sid], abs=1e-9), (
                    f"Day {day} segment {sid}: along_m changed from "
                    f"{reference_positions[sid]} to {seg['along_m']}"
                )

    def test_across_m_stable_across_days(self, all_manifests):
        """across_m (lateral lane position) must not change between days."""
        reference_positions = {
            s["road_segment_id"]: s["across_m"]
            for s in all_manifests[1].get("segments", [])
        }
        for day in range(2, 21):
            for seg in all_manifests[day].get("segments", []):
                sid = seg["road_segment_id"]
                assert seg["across_m"] == pytest.approx(reference_positions[sid], abs=1e-9), (
                    f"Day {day} segment {sid}: across_m changed from "
                    f"{reference_positions[sid]} to {seg['across_m']}"
                )

    def test_condition_score_varies_across_days(self, all_manifests):
        """Condition scores should change over time for non-stable segments."""
        # SEG_002 (slow deterioration) should have a higher score by day 20 than day 1
        score_day1 = ic.get_segment_info(all_manifests[1], "SEG_002").get(
            "renderer_condition_score", 0.0
        )
        score_day20 = ic.get_segment_info(all_manifests[20], "SEG_002").get(
            "renderer_condition_score", 0.0
        )
        assert score_day20 > score_day1, (
            f"SEG_002 score should increase over time: day1={score_day1}, day20={score_day20}"
        )


# ===========================================================================
# 2. Capture path construction
# ===========================================================================

class TestCapturePaths:

    def test_basic_path_structure(self, tmp_path):
        out = ic.build_output_dir(1, "SEG_001", root=tmp_path)
        assert out == tmp_path / "day_01" / "SEG_001"

    def test_day_zero_padded(self, tmp_path):
        for day in [1, 9, 10, 20]:
            out = ic.build_output_dir(day, "SEG_001", root=tmp_path)
            part = out.parts[-2]
            assert part.startswith("day_"), f"Expected day_XX format, got {part}"
            assert len(part) == 6, f"Expected day_XX (6 chars), got {part!r}"

    def test_all_segments_unique_paths(self, tmp_path):
        paths = [ic.build_output_dir(5, seg, root=tmp_path) for seg in ic.KNOWN_SEGMENT_IDS]
        assert len(set(paths)) == len(ic.KNOWN_SEGMENT_IDS), "Segment paths must be unique"

    def test_invalid_day_raises(self, tmp_path):
        with pytest.raises(ValueError, match="day must be 1–20"):
            ic.build_output_dir(0, "SEG_001", root=tmp_path)
        with pytest.raises(ValueError, match="day must be 1–20"):
            ic.build_output_dir(21, "SEG_001", root=tmp_path)

    def test_invalid_segment_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a known segment"):
            ic.build_output_dir(1, "SEG_007", root=tmp_path)
        with pytest.raises(ValueError, match="not a known segment"):
            ic.build_output_dir(1, "SEG_000", root=tmp_path)

    def test_expected_output_path_format(self, tmp_path):
        """Verify full expected path pattern for a specific day/segment combo."""
        out = ic.build_output_dir(10, "SEG_003", root=tmp_path)
        assert str(out).endswith("day_10/SEG_003")


# ===========================================================================
# 3. PNG validation
# ===========================================================================

def _make_minimal_png(width: int, height: int, fill_byte: int = 0x80) -> bytes:
    """Create a minimal valid PNG with a single scanline row of constant colour."""
    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(ctype: bytes, cdata: bytes) -> bytes:
        length = struct.pack(">I", len(cdata))
        crc = struct.pack(">I", zlib.crc32(ctype + cdata) & 0xFFFFFFFF)
        return length + ctype + cdata + crc

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = chunk(b"IHDR", ihdr_data)

    # IDAT — one scanline per row, filter byte 0 followed by RGB values
    raw_rows = b""
    for _ in range(height):
        raw_rows += bytes([0]) + bytes([fill_byte] * (width * 3))
    idat = chunk(b"IDAT", zlib.compress(raw_rows, level=1))

    # IEND
    iend = chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


class TestValidatePng:

    def test_valid_normal_image(self, tmp_path):
        png_data = _make_minimal_png(1920, 1080, fill_byte=0x60)
        f = tmp_path / "test.png"
        f.write_bytes(png_data)
        result = ic.validate_png(f, expected_width=1920, expected_height=1080)
        assert result["valid"] is True, f"Expected valid, got: {result}"

    def test_file_not_exist(self, tmp_path):
        result = ic.validate_png(tmp_path / "nonexistent.png")
        assert result["valid"] is False
        assert "does not exist" in result["reason"].lower()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.png"
        f.write_bytes(b"")
        result = ic.validate_png(f, min_size_bytes=100)
        assert result["valid"] is False
        assert "too small" in result["reason"].lower()

    def test_wrong_dimensions(self, tmp_path):
        png_data = _make_minimal_png(640, 480, fill_byte=0x80)
        f = tmp_path / "wrong_dim.png"
        f.write_bytes(png_data)
        result = ic.validate_png(f, expected_width=1920, expected_height=1080)
        assert result["valid"] is False
        assert "dimensions" in result["reason"].lower()

    def test_all_black_rejected(self, tmp_path):
        png_data = _make_minimal_png(1920, 1080, fill_byte=0x00)
        f = tmp_path / "all_black.png"
        f.write_bytes(png_data)
        result = ic.validate_png(
            f, expected_width=1920, expected_height=1080, near_uniform_tolerance=0.95
        )
        assert result["valid"] is False
        assert "black" in result["reason"].lower()

    def test_all_white_rejected(self, tmp_path):
        png_data = _make_minimal_png(1920, 1080, fill_byte=0xFF)
        f = tmp_path / "all_white.png"
        f.write_bytes(png_data)
        result = ic.validate_png(
            f, expected_width=1920, expected_height=1080, near_uniform_tolerance=0.95
        )
        assert result["valid"] is False
        assert "white" in result["reason"].lower()

    def test_non_png_file_rejected(self, tmp_path):
        f = tmp_path / "not_a_png.png"
        f.write_bytes(b"This is not a PNG" * 200)
        result = ic.validate_png(f)
        assert result["valid"] is False
        assert "png" in result["reason"].lower()


# ===========================================================================
# 4. Metadata schema
# ===========================================================================

class TestMetadataSchema:

    REQUIRED_FIELDS = [
        "schema_version",
        "day",
        "segment_id",
        "timestamp",
        "camera_pose",
        "world_coordinates",
        "condition_score",
        "image_path",
        "capture_source",
    ]

    CAMERA_POSE_FIELDS = ["x_cm", "y_cm", "z_cm", "pitch_deg", "yaw_deg", "roll_deg"]
    WORLD_COORD_FIELDS = ["x_cm", "y_cm", "z_cm", "along_m", "across_m"]

    @pytest.fixture()
    def written_metadata(self, tmp_path):
        out_dir = tmp_path / "day_05" / "SEG_003"
        meta_path = ic.write_metadata(
            output_dir=out_dir,
            day=5,
            segment_id="SEG_003",
            camera_pose={"x_cm": 15500.0, "y_cm": -135.0, "z_cm": 2590.0,
                         "pitch_deg": -89.0, "yaw_deg": 0.0, "roll_deg": 0.0},
            world_coordinates={"x_cm": 15500.0, "y_cm": -135.0, "z_cm": 0.0,
                                "along_m": 155.0, "across_m": -1.35},
            condition_score=0.2598,
        )
        return meta_path, json.loads(meta_path.read_text(encoding="utf-8"))

    def test_required_top_level_fields(self, written_metadata):
        _, record = written_metadata
        for field in self.REQUIRED_FIELDS:
            assert field in record, f"Missing required field: {field!r}"

    def test_day_value(self, written_metadata):
        _, record = written_metadata
        assert record["day"] == 5

    def test_segment_id_value(self, written_metadata):
        _, record = written_metadata
        assert record["segment_id"] == "SEG_003"

    def test_camera_pose_subfields(self, written_metadata):
        _, record = written_metadata
        for field in self.CAMERA_POSE_FIELDS:
            assert field in record["camera_pose"], (
                f"camera_pose missing field: {field!r}"
            )

    def test_world_coordinates_subfields(self, written_metadata):
        _, record = written_metadata
        for field in self.WORLD_COORD_FIELDS:
            assert field in record["world_coordinates"], (
                f"world_coordinates missing field: {field!r}"
            )

    def test_condition_score_type(self, written_metadata):
        _, record = written_metadata
        assert isinstance(record["condition_score"], float)

    def test_image_path_string(self, written_metadata):
        meta_path, record = written_metadata
        assert isinstance(record["image_path"], str)
        assert record["image_path"].endswith("raw.png")

    def test_timestamp_iso8601(self, written_metadata):
        _, record = written_metadata
        from datetime import datetime
        # Must parse without error
        dt = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        assert dt.year >= 2024

    def test_extra_fields_merged(self, tmp_path):
        out_dir = tmp_path / "day_01" / "SEG_001"
        meta_path = ic.write_metadata(
            output_dir=out_dir,
            day=1,
            segment_id="SEG_001",
            camera_pose={"x_cm": 3500.0, "y_cm": -135.0, "z_cm": 2590.0,
                         "pitch_deg": -89.0, "yaw_deg": 0.0},
            world_coordinates={"x_cm": 3500.0, "y_cm": -135.0, "z_cm": 0.0,
                                "along_m": 35.0, "across_m": -1.35},
            condition_score=0.025,
            extra={"custom_field": "hello", "total_defects_day": 0},
        )
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        assert record.get("custom_field") == "hello"
        assert record.get("total_defects_day") == 0
