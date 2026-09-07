#!/usr/bin/env python3
"""Build a scenario-aware forecast table from FHWA LTPP SDR 40 records.

The table joins consecutive measured IRI visits to the precipitation,
temperature, wet-day, and representative heavy-truck traffic observed over
the same future interval.  No scenario-specific severity increments are
invented by this script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import urllib.request
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

from access_parser import AccessParser


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "data" / "processed"
OUTPUT = OUTPUT_DIR / "scenario_training_pairs.csv"
SUMMARY = OUTPUT_DIR / "scenario_dataset_summary.json"

SDR_VERSION = 40
STATE_CODES = ("AK", "HI", "LA", "ND", "NY", "RI")
DOWNLOAD_TEMPLATE = (
    "https://du993ylnpbddg.cloudfront.net/SDR/40/By_State_Province/"
    "SDR40_{state}.zip"
)

# Very short repeat-profile runs are not deterioration observations.  The
# upper bound keeps the learned interface focused on near-term forecasting.
MIN_PAIR_DAYS = 7
MAX_PAIR_DAYS = 730
MIN_CLIMATE_COVERAGE = 0.80
MODEL_HORIZON_STEP_DAYS = 30


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "RoadSentinel-Day3/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def _rows(table: Mapping[str, list[Any]]) -> Iterable[dict[str, Any]]:
    row_count = len(next(iter(table.values()), []))
    for index in range(row_count):
        yield {name: values[index] for name, values in table.items()}


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value[:19])


def _is_eligible(row: Mapping[str, Any]) -> bool:
    return row.get("RECORD_STATUS") in (None, "E")


class DailySeries:
    """Prefix sums for one daily climate variable, keyed by VWS station."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        date_field: str,
        value_field: str,
    ) -> None:
        grouped: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        for row in rows:
            value = row.get(value_field)
            date_value = row.get(date_field)
            station = row.get("VWS_ID")
            if _is_eligible(row) and value is not None and date_value and station:
                grouped[str(station)].append((_date(str(date_value)), float(value)))

        self._series: dict[str, tuple[list[datetime], list[float], list[int]]] = {}
        for station, observations in grouped.items():
            observations.sort()
            dates: list[datetime] = []
            sums = [0.0]
            wet_counts = [0]
            for date_value, value in observations:
                dates.append(date_value)
                sums.append(sums[-1] + value)
                wet_counts.append(wet_counts[-1] + int(value > 0.0))
            self._series[station] = (dates, sums, wet_counts)

    def interval(self, station: str, start: datetime, end: datetime) -> dict[str, float] | None:
        series = self._series.get(station)
        if series is None:
            return None
        dates, sums, wet_counts = series
        # Forecast exposure starts after the current measurement and includes
        # the future measurement day: exactly ``(end - start).days`` slots.
        lower = bisect_right(dates, start)
        upper = bisect_right(dates, end)
        count = upper - lower
        if count <= 0:
            return None
        return {
            "mean": (sums[upper] - sums[lower]) / count,
            "wet_fraction": (wet_counts[upper] - wet_counts[lower]) / count,
            "count": float(count),
        }


def _traffic_tables(
    database: AccessParser,
) -> tuple[
    dict[tuple[int, str], dict[int, float]],
    dict[tuple[int, str], list[tuple[datetime, datetime, float]]],
]:
    annual_values: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in _rows(database.parse_table("TRF_MEPDG_AADTT_LTPP_LN")):
        value = row.get("AADTT_LTPPLN")
        if _is_eligible(row) and value is not None:
            key = (int(row["STATE_CODE"]), str(row["SHRP_ID"]), int(row["YEAR"]))
            annual_values[key].append(float(value))
    annual: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
    for (state, section, year), values in annual_values.items():
        annual[(state, section)][year] = statistics.median(values)

    representative: dict[
        tuple[int, str], list[tuple[datetime, datetime, float]]
    ] = defaultdict(list)
    for row in _rows(database.parse_table("TRF_REP")):
        if (
            not _is_eligible(row)
            or row.get("REP_AADTT") is None
            or not row.get("ASSIGN_DATE")
            or not row.get("DEASSIGN_DATE")
        ):
            continue
        key = (int(row["STATE_CODE"]), str(row["SHRP_ID"]))
        representative[key].append(
            (
                _date(str(row["ASSIGN_DATE"])),
                _date(str(row["DEASSIGN_DATE"])),
                float(row["REP_AADTT"]),
            )
        )
    return dict(annual), dict(representative)


def _mean_traffic(
    key: tuple[int, str],
    start: datetime,
    end: datetime,
    annual: Mapping[tuple[int, str], Mapping[int, float]],
    representative: Mapping[tuple[int, str], list[tuple[datetime, datetime, float]]],
) -> tuple[float, str] | None:
    by_year = annual.get(key, {})
    annual_matches = [
        value for year, value in by_year.items() if start.year <= year <= end.year
    ]
    if annual_matches:
        return statistics.mean(annual_matches), "TRF_MEPDG_AADTT_LTPP_LN"

    overlaps: list[tuple[int, float]] = []
    for assigned, deassigned, value in representative.get(key, []):
        overlap_days = (min(end, deassigned) - max(start, assigned)).days
        if overlap_days > 0:
            overlaps.append((overlap_days, value))
    if not overlaps:
        return None
    total_days = sum(days for days, _ in overlaps)
    return (
        sum(days * value for days, value in overlaps) / total_days,
        "TRF_REP",
    )


def _process_database(database_path: Path, state_abbreviation: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    database = AccessParser(str(database_path))
    iri_table = database.parse_table("ANALYSIS_IRI")
    link_table = database.parse_table("CLM_SITE_VWS_LINK")
    precipitation = DailySeries(
        _rows(database.parse_table("CLM_VWS_PRECIP_DAILY")),
        date_field="VWS_DATE",
        value_field="DAY_PRECIPITATION",
    )
    temperature = DailySeries(
        _rows(database.parse_table("CLM_VWS_TEMP_DAILY")),
        date_field="VWS_DATE",
        value_field="MEAN_DAY_TEMP",
    )
    annual_traffic, representative_traffic = _traffic_tables(database)

    station_by_section = {
        (int(row["STATE_CODE"]), str(row["SHRP_ID"])): str(row["VWS_ID"])
        for row in _rows(link_table)
        if _is_eligible(row)
    }

    measurements: dict[
        tuple[int, str, int], dict[datetime, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in _rows(iri_table):
        if (
            _is_eligible(row)
            and row.get("MRI") is not None
            and row.get("VISIT_DATE")
        ):
            key = (
                int(row["STATE_CODE"]),
                str(row["SHRP_ID"]),
                int(row["CONSTRUCTION_NO"]),
            )
            measurements[key][_date(str(row["VISIT_DATE"]))].append(float(row["MRI"]))

    output: list[dict[str, Any]] = []
    rejected: dict[str, int] = defaultdict(int)
    for (state_code, section, construction), visits in measurements.items():
        collapsed = sorted(
            (date_value, statistics.median(values))
            for date_value, values in visits.items()
        )
        for (current_date, current_iri), (future_date, future_iri) in zip(
            collapsed, collapsed[1:]
        ):
            days_ahead = (future_date - current_date).days
            if not MIN_PAIR_DAYS <= days_ahead <= MAX_PAIR_DAYS:
                rejected["horizon_outside_7_730_days"] += 1
                continue

            section_key = (state_code, section)
            station = station_by_section.get(section_key)
            if station is None:
                rejected["missing_climate_station_link"] += 1
                continue
            rain = precipitation.interval(station, current_date, future_date)
            heat = temperature.interval(station, current_date, future_date)
            if rain is None or heat is None:
                rejected["missing_climate_interval"] += 1
                continue
            if (
                rain["count"] / days_ahead < MIN_CLIMATE_COVERAGE
                or heat["count"] / days_ahead < MIN_CLIMATE_COVERAGE
            ):
                rejected["climate_coverage_below_80_percent"] += 1
                continue
            traffic = _mean_traffic(
                section_key,
                current_date,
                future_date,
                annual_traffic,
                representative_traffic,
            )
            if traffic is None:
                rejected["missing_traffic_interval"] += 1
                continue

            output.append(
                {
                    "site_id": f"{state_code:02d}-{section}",
                    "state": state_abbreviation,
                    "state_code": state_code,
                    "shrp_id": section,
                    "construction_no": construction,
                    "current_date": current_date.date().isoformat(),
                    "future_date": future_date.date().isoformat(),
                    "observed_days_ahead": days_ahead,
                    # The user interface operates at monthly 30/60/90-day
                    # horizons.  Bucket the measured interval to the nearest
                    # 30-day model feature; climate aggregation and the target
                    # still use the unmodified observation dates above.
                    "days_ahead": max(
                        MODEL_HORIZON_STEP_DAYS,
                        int(
                            (days_ahead + MODEL_HORIZON_STEP_DAYS / 2)
                            // MODEL_HORIZON_STEP_DAYS
                            * MODEL_HORIZON_STEP_DAYS
                        ),
                    ),
                    "current_iri_m_per_km": f"{current_iri:.9f}",
                    "future_iri_m_per_km": f"{future_iri:.9f}",
                    "rainfall_level": f"{rain['mean']:.9f}",
                    "traffic_level": f"{traffic[0]:.9f}",
                    "temperature": f"{heat['mean']:.9f}",
                    "water_exposure": f"{rain['wet_fraction']:.9f}",
                    "rain_observation_days": int(rain["count"]),
                    "temperature_observation_days": int(heat["count"]),
                    "traffic_source": traffic[1],
                }
            )
    return output, dict(rejected)


def main() -> int:
    all_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    rejection_totals: dict[str, int] = defaultdict(int)

    for state in STATE_CODES:
        url = DOWNLOAD_TEMPLATE.format(state=state)
        with TemporaryDirectory(prefix=f"roadsentinel_ltpp_{state.lower()}_") as temp_dir:
            archive = Path(temp_dir) / f"SDR40_{state}.zip"
            _download(url, archive)
            archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
            with ZipFile(archive) as zip_file:
                primary_name = next(
                    name for name in zip_file.namelist() if "Primary_Data" in name
                )
                zip_file.extract(primary_name, temp_dir)
            state_rows, rejected = _process_database(Path(temp_dir) / primary_name, state)
            all_rows.extend(state_rows)
            for reason, count in rejected.items():
                rejection_totals[reason] += count
            sources.append(
                {
                    "state": state,
                    "url": url,
                    "downloaded_bytes": archive.stat().st_size,
                    "sha256": archive_sha256,
                    "accepted_pairs": len(state_rows),
                    "rejected_pairs": rejected,
                }
            )

    if not all_rows:
        raise RuntimeError("No fully joined LTPP scenario training rows were produced")
    all_rows.sort(key=lambda row: (row["site_id"], row["construction_no"], row["current_date"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "dataset": "FHWA LTPP InfoPave Standard Data Release 40",
        "source_page": "https://infopave.fhwa.dot.gov/Data/StandardDataRelease/",
        "states": list(STATE_CODES),
        "source_files": sources,
        "output": str(OUTPUT.relative_to(HERE)),
        "accepted_pair_count": len(all_rows),
        "unique_site_count": len({row["site_id"] for row in all_rows}),
        "rejected_pair_counts": dict(sorted(rejection_totals.items())),
        "observed_pair_horizon_days": {
            "minimum": min(int(row["observed_days_ahead"]) for row in all_rows),
            "maximum": max(int(row["observed_days_ahead"]) for row in all_rows),
        },
        "model_horizon_days": {
            "minimum": min(int(row["days_ahead"]) for row in all_rows),
            "maximum": max(int(row["days_ahead"]) for row in all_rows),
            "bucket_size": MODEL_HORIZON_STEP_DAYS,
            "supported_interface_values": [30, 60, 90],
        },
        "joins": {
            "condition": "ANALYSIS_IRI consecutive visits within one construction number",
            "rainfall_level": "mean DAY_PRECIPITATION (mm/day) over the future interval",
            "temperature": "mean MEAN_DAY_TEMP (degrees C) over the future interval",
            "water_exposure": "fraction of interval precipitation observations above 0 mm",
            "traffic_level": (
                "mean observed AADTT_LTPPLN, with overlapping REP_AADTT fallback; "
                "heavy trucks/day in the LTPP lane"
            ),
        },
        "climate_minimum_coverage": MIN_CLIMATE_COVERAGE,
        "severity_jumps_added": False,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
