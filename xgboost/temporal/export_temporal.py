"""Export Goal-2 temporal observations in dashboard-ready CSV and JSON forms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

if __package__:
    from .load_sequence import load_sequence
    from .progression import build_daily_summary, build_tracks, progression_summary
else:  # Supports the documented ``python xgboost/temporal/export_temporal.py`` CLI.
    from load_sequence import load_sequence
    from progression import build_daily_summary, build_tracks, progression_summary


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_sequence(segment_dir: Path | str, output_root: Path | str) -> Path:
    records = load_sequence(segment_dir)
    daily = build_daily_summary(records)
    tracks, events = build_tracks(records)
    summary = progression_summary(records, tracks, events)
    target = Path(output_root) / records[0].record["segment_id"]
    target.mkdir(parents=True, exist_ok=True)
    fields = list(daily[0])
    with (target / "daily_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(daily)
    _write_json(target / "daily_summary.json", daily)
    _write_json(target / "defect_tracks.json", {"segment_id": summary["segment_id"], "tracks": list(tracks.values())})
    _write_json(target / "temporal_events.json", {"segment_id": summary["segment_id"], "events": events})
    _write_json(target / "progression_summary.json", summary)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segment_dir", type=Path, help="SEG_xxx directory containing day_*/features.json")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "temporal_outputs")
    args = parser.parse_args()
    print(export_sequence(args.segment_dir, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
