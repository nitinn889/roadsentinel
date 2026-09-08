"""Strict loader for shared-perception records arranged by segment and day."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = ROOT / "sam2_dino"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))
from feature_contract import ContractError, load_feature_record  # noqa: E402


class SequenceLoadError(ValueError):
    """A sequence cannot safely be analysed."""


@dataclass(frozen=True)
class DailyRecord:
    record: dict[str, Any]
    source_path: Path

    @property
    def day(self) -> int:
        return int(self.record["day"])


def load_sequence(segment_dir: Path | str) -> list[DailyRecord]:
    """Load ``day_*/features.json`` records, enforcing one segment and one record/day."""
    root = Path(segment_dir)
    if not root.is_dir():
        raise SequenceLoadError(f"Sequence directory does not exist: {root}")
    candidates = sorted(root.glob("day_*/features.json"))
    if not candidates:
        # Supports a Marion layout that writes a feature file directly below day folders.
        candidates = sorted(root.glob("day_*/*.json"))
    if not candidates:
        raise SequenceLoadError(f"No day_*/features.json records found below {root}")
    records: list[DailyRecord] = []
    segment_id: str | None = None
    days: set[int] = set()
    for path in candidates:
        try:
            record = load_feature_record(path)
        except (OSError, ContractError, ValueError) as exc:
            raise SequenceLoadError(f"Malformed feature record {path}: {exc}") from exc
        day_dir = path.parent.name
        expected = f"day_{record['day']:02d}"
        if day_dir != expected:
            raise SequenceLoadError(f"Day directory {day_dir!r} conflicts with record day {record['day']} in {path}")
        if segment_id is None:
            segment_id = record["segment_id"]
        elif record["segment_id"] != segment_id:
            raise SequenceLoadError(f"Mixed segment IDs: {segment_id!r} and {record['segment_id']!r}")
        if record["day"] in days:
            raise SequenceLoadError(f"Duplicate day {record['day']} in {root}")
        days.add(record["day"])
        # Resolve relative mask references at load time; the contract remains
        # unchanged and matching can now prefer masks from Marion layouts.
        record = dict(record)
        record["defects"] = [dict(defect) for defect in record["defects"]]
        for defect in record["defects"]:
            mask = defect.get("mask_path")
            if mask and not Path(mask).is_absolute():
                defect["mask_path"] = str((path.parent / mask).resolve())
        records.append(DailyRecord(record=record, source_path=path))
    return sorted(records, key=lambda item: item.day)


def missing_days(records: list[DailyRecord]) -> list[int]:
    """Return gaps between supplied days; absence is reported, never filled."""
    if not records:
        return []
    known = {item.day for item in records}
    return [day for day in range(min(known), max(known) + 1) if day not in known]
