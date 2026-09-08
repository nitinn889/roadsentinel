from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
XGBOOST_DIR = HERE.parent
if str(XGBOOST_DIR) not in sys.path:
    sys.path.insert(0, str(XGBOOST_DIR))

from temporal.export_temporal import export_sequence
from temporal.defect_matching import match_adjacent
from temporal.load_sequence import SequenceLoadError, load_sequence, missing_days


FIXTURE = HERE / "fixtures" / "TEST_FIXTURE_ONLY" / "SEG_FIXTURE"


class TemporalTest(unittest.TestCase):
    def test_fixture_orders_tracks_and_exports(self) -> None:
        records = load_sequence(FIXTURE)
        self.assertEqual([record.day for record in records], [1, 2, 3])
        with tempfile.TemporaryDirectory() as directory:
            target = export_sequence(FIXTURE, directory)
            summary = json.loads((target / "progression_summary.json").read_text())
            events = json.loads((target / "temporal_events.json").read_text())["events"]
            self.assertEqual(summary["num_tracks"], 2)
            self.assertEqual(summary["water_first_seen_day"], 3)
            self.assertIn("OBSERVED_AREA_INCREASED", [event["event"] for event in events])
            self.assertIn("NEW_DEFECT", [event["event"] for event in events])

    def test_missing_day_is_reported_not_filled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "SEG_FIXTURE"
            for day in (1, 3):
                source = FIXTURE / f"day_{day:02d}" / "features.json"
                target = root / f"day_{day:02d}"; target.mkdir(parents=True)
                target.joinpath("features.json").write_text(source.read_text())
            self.assertEqual(missing_days(load_sequence(root)), [2])

    def test_duplicate_day_wrong_segment_and_malformed_fail_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "SEG_FIXTURE"
            for dirname in ("day_01", "day_02"):
                target = root / dirname; target.mkdir(parents=True)
                target.joinpath("features.json").write_text((FIXTURE / "day_01" / "features.json").read_text())
            with self.assertRaises(SequenceLoadError): load_sequence(root)
            (root / "day_02" / "features.json").write_text('{"not": "a record"}')
            with self.assertRaises(SequenceLoadError): load_sequence(root)

    def test_wrong_segment_and_unordered_days_fail_or_sort_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "SEG_FIXTURE"
            for day in (3, 1):
                target = root / f"day_{day:02d}"; target.mkdir(parents=True)
                target.joinpath("features.json").write_text((FIXTURE / f"day_{day:02d}" / "features.json").read_text())
            self.assertEqual([item.day for item in load_sequence(root)], [1, 3])
            target = root / "day_02"; target.mkdir()
            wrong = json.loads((FIXTURE / "day_02" / "features.json").read_text())
            wrong["segment_id"] = "SEG_OTHER"
            target.joinpath("features.json").write_text(json.dumps(wrong))
            with self.assertRaises(SequenceLoadError): load_sequence(root)

    def test_empty_disappearing_ambiguous_and_null_geometry_do_not_invent_matches(self) -> None:
        defect = {"bbox": [0, 0, 10, 10], "centroid_x": 5, "centroid_y": 5, "area_ratio": 0.01, "defect_type": "pothole"}
        self.assertEqual(match_adjacent([], []), [])  # no defects
        self.assertEqual(match_adjacent([defect], []), [])  # not-observed is handled by tracking
        # Two current candidates overlap the same old defect: one-to-one logic emits one match only.
        self.assertEqual(len(match_adjacent([defect], [defect, dict(defect)])), 1)
        null_geometry = {"bbox": None, "centroid_x": None, "centroid_y": None, "area_ratio": 0.01, "defect_type": "pothole"}
        self.assertEqual(match_adjacent([null_geometry], [null_geometry]), [])


if __name__ == "__main__":
    unittest.main()
