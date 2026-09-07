#!/usr/bin/env python3
"""Convert NYC pavement ratings into consecutive longitudinal forecast pairs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data" / "raw" / "nyc_pavement_ratings_sample.csv"
OUTPUT_DIR = HERE / "data" / "processed"
OUTPUT = OUTPUT_DIR / "training_pairs.csv"


def rating_to_severity(rating: float) -> float:
    """Invert the documented NYC 1 (worst) to 10 (best) rating onto [0, 1]."""
    if not 1.0 <= rating <= 10.0:
        raise ValueError(f"SystemRating outside documented [1,10] range: {rating}")
    return (10.0 - rating) / 9.0


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Run fetch_nyc_pavement_sample.py first: {SOURCE}")

    by_site_date: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_site_date[(row["oftcode"], row["inspection"])].append(row)

    observations: dict[str, list[dict[str, object]]] = defaultdict(list)
    duplicate_rows_collapsed = 0
    for (site_id, inspection), rows in by_site_date.items():
        duplicate_rows_collapsed += len(rows) - 1
        ratings = [float(row["systemrating"]) for row in rows]
        rating = sum(ratings) / len(ratings)
        sample = rows[0]
        observations[site_id].append(
            {
                "site_id": site_id,
                "inspection_date": inspection[:10],
                "rating": rating,
                "borough": sample["boroughname"],
                "on_street": sample["onstreetna"],
                "from_street": sample["fromstreet"],
                "to_street": sample["tostreetna"],
            }
        )

    pairs: list[dict[str, object]] = []
    for site_id, site_observations in observations.items():
        site_observations.sort(key=lambda row: str(row["inspection_date"]))
        for current, future in zip(site_observations, site_observations[1:]):
            current_date = datetime.fromisoformat(str(current["inspection_date"])).date()
            future_date = datetime.fromisoformat(str(future["inspection_date"])).date()
            days_ahead = (future_date - current_date).days
            if days_ahead <= 0:
                continue
            pairs.append(
                {
                    "site_id": site_id,
                    "current_date": current_date.isoformat(),
                    "future_date": future_date.isoformat(),
                    "days_ahead": days_ahead,
                    "current_rating": f"{float(current['rating']):.6f}",
                    "future_rating": f"{float(future['rating']):.6f}",
                    "current_severity": f"{rating_to_severity(float(current['rating'])):.9f}",
                    "future_severity": f"{rating_to_severity(float(future['rating'])):.9f}",
                    "borough": current["borough"],
                    "on_street": current["on_street"],
                    "from_street": current["from_street"],
                    "to_street": current["to_street"],
                }
            )

    if not pairs:
        raise RuntimeError("No positive-time longitudinal pairs were built")
    pairs.sort(key=lambda row: (str(row["site_id"]), str(row["current_date"])))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(pairs[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    summary = {
        "source": str(SOURCE.relative_to(HERE)),
        "output": str(OUTPUT.relative_to(HERE)),
        "source_rows": sum(len(rows) for rows in by_site_date.values()),
        "unique_sites": len(observations),
        "unique_site_dates": len(by_site_date),
        "duplicate_same_site_date_rows_collapsed_by_mean": duplicate_rows_collapsed,
        "forecast_pairs": len(pairs),
        "target": "future_severity",
        "target_mapping": "severity = (10 - SystemRating) / 9",
        "mapping_bounds": {"SystemRating_10": 0.0, "SystemRating_1": 1.0},
        "interpretation": (
            "Normalized inversion of the NYC pavement rating, used as a common forecasting "
            "scale; it is not asserted to be identical to RoadSentinel image severity."
        ),
    }
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
