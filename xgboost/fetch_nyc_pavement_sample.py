#!/usr/bin/env python3
"""Fetch a small, reproducible longitudinal sample from NYC Open Data.

The script verifies the live schema before requesting rows. It selects the 200
street segments with the most distinct rated inspection dates, then downloads
only the columns needed for an auditable longitudinal baseline.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path


DATASET_ID = "6yyb-pb25"
METADATA_URL = f"https://data.cityofnewyork.us/api/views/{DATASET_ID}"
RESOURCE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.csv"
SITE_LIMIT = 200
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"

EXPECTED_FIELDS = {
    "oftcode": "text",
    "boroughname": "text",
    "onstreetna": "text",
    "fromstreet": "text",
    "tostreetna": "text",
    "systemrating": "number",
    "inspection": "calendar_date",
}


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "RoadSentinel-Day2/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _query(params: dict[str, str]) -> bytes:
    return _get(f"{RESOURCE_URL}?{urllib.parse.urlencode(params)}")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    metadata_bytes = _get(METADATA_URL)
    metadata = json.loads(metadata_bytes)
    actual_fields = {
        column["fieldName"]: column["dataTypeName"]
        for column in metadata.get("columns", [])
    }
    mismatches = {
        name: {"expected": expected, "actual": actual_fields.get(name)}
        for name, expected in EXPECTED_FIELDS.items()
        if actual_fields.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"NYC dataset schema mismatch: {mismatches}")

    sites_bytes = _query(
        {
            "$select": "oftcode,count(distinct inspection) as observation_dates",
            "$where": "systemrating between 1 and 10 and inspection is not null",
            "$group": "oftcode",
            "$having": "count(distinct inspection) >= 3",
            "$order": "observation_dates desc,oftcode",
            "$limit": str(SITE_LIMIT),
        }
    )
    site_rows = list(csv.DictReader(io.StringIO(sites_bytes.decode("utf-8"))))
    site_ids = [row["oftcode"] for row in site_rows]
    if len(site_ids) != SITE_LIMIT:
        raise RuntimeError(f"Expected {SITE_LIMIT} eligible sites, received {len(site_ids)}")

    quoted_ids = ",".join(f"'{site_id}'" for site_id in site_ids)
    records_bytes = _query(
        {
            "$select": (
                "oftcode,boroughname,onstreetna,fromstreet,tostreetna,"
                "inspection,systemrating"
            ),
            "$where": (
                "systemrating between 1 and 10 and inspection is not null "
                f"and oftcode in ({quoted_ids})"
            ),
            "$order": "oftcode,inspection",
            "$limit": "50000",
        }
    )
    records = list(csv.DictReader(io.StringIO(records_bytes.decode("utf-8"))))
    if not records:
        raise RuntimeError("NYC query returned no pavement records")

    metadata_path = RAW_DIR / "nyc_dataset_metadata.json"
    sites_path = RAW_DIR / "nyc_selected_sites.csv"
    records_path = RAW_DIR / "nyc_pavement_ratings_sample.csv"
    metadata_path.write_bytes(metadata_bytes)
    sites_path.write_bytes(sites_bytes)
    records_path.write_bytes(records_bytes)

    summary = {
        "dataset_id": DATASET_ID,
        "dataset_name": metadata.get("name"),
        "source": f"https://data.cityofnewyork.us/d/{DATASET_ID}",
        "metadata_url": METADATA_URL,
        "api_resource": RESOURCE_URL,
        "selection": (
            "Top 200 OFTCode street segments by count of distinct rated inspection dates; "
            "at least 3 distinct dates; SystemRating within documented [1,10] range; "
            "non-null InspectionTime."
        ),
        "verified_fields": EXPECTED_FIELDS,
        "selected_site_count": len(site_ids),
        "downloaded_row_count": len(records),
        "downloaded_csv_bytes": len(records_bytes),
        "metadata_rows_updated_at_epoch": metadata.get("rowsUpdatedAt"),
        "files": {
            "metadata": str(metadata_path.relative_to(HERE)),
            "selected_sites": str(sites_path.relative_to(HERE)),
            "records": str(records_path.relative_to(HERE)),
        },
    }
    summary_path = RAW_DIR / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
