"""RoadSentinel's frozen Day 1 perception/forecast feature contract.

Unknown values stay ``None`` and become JSON ``null``. The module has no
third-party dependencies so both perception and forecasting code can use it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0.0"
REQUIRED_FIELDS = (
    "schema_version",
    "image_id",
    "segment_id",
    "day",
    "original_filename",
    "condition",
    "current_severity",
    "crack_area_ratio",
    "defect_area_ratio",
    "defect_count",
    "surface_anomaly_score",
    "water_flag",
    "defects",
)
NULLABLE_RATIO_FIELDS = (
    "current_severity",
    "crack_area_ratio",
    "defect_area_ratio",
    "surface_anomaly_score",
)


class ContractError(ValueError):
    """Raised when a feature record violates the shared contract."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_feature_record(record: Mapping[str, Any]) -> None:
    """Validate the dependency-free subset of feature_contract.schema.json."""
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        raise ContractError(f"Missing required fields: {', '.join(missing)}")
    unknown = sorted(set(record) - set(REQUIRED_FIELDS))
    if unknown:
        raise ContractError(f"Unknown top-level fields: {', '.join(unknown)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION!r}")
    for name in ("image_id", "segment_id", "original_filename"):
        if not isinstance(record[name], str) or not record[name]:
            raise ContractError(f"{name} must be a non-empty string")
    if not isinstance(record["day"], int) or isinstance(record["day"], bool) or not 1 <= record["day"] <= 10:
        raise ContractError("day must be an integer from 1 through 10")
    if record["condition"] is not None and not isinstance(record["condition"], str):
        raise ContractError("condition must be a string or null")
    for name in NULLABLE_RATIO_FIELDS:
        value = record[name]
        if value is not None and (not _is_number(value) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
            raise ContractError(f"{name} must be a finite number in [0,1] or null")
    if not isinstance(record["defect_count"], int) or isinstance(record["defect_count"], bool) or record["defect_count"] < 0:
        raise ContractError("defect_count must be a non-negative integer")
    if record["water_flag"] is not None and not isinstance(record["water_flag"], bool):
        raise ContractError("water_flag must be boolean or null")
    defects = record["defects"]
    if not isinstance(defects, list) or len(defects) != record["defect_count"]:
        raise ContractError("defects must be a list whose length equals defect_count")
    allowed = {"bbox", "mask_path", "centroid_x", "centroid_y", "area_ratio", "defect_type"}
    for index, defect in enumerate(defects):
        if not isinstance(defect, dict):
            raise ContractError(f"defects[{index}] must be an object")
        extra = sorted(set(defect) - allowed)
        if extra:
            raise ContractError(f"defects[{index}] has unknown fields: {', '.join(extra)}")
        bbox = defect.get("bbox")
        if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(v) for v in bbox)):
            raise ContractError(f"defects[{index}].bbox must contain four numbers")
        for name in ("centroid_x", "centroid_y"):
            if name in defect and not _is_number(defect[name]):
                raise ContractError(f"defects[{index}].{name} must be numeric")
        if "area_ratio" in defect and (not _is_number(defect["area_ratio"]) or not 0.0 <= float(defect["area_ratio"]) <= 1.0):
            raise ContractError(f"defects[{index}].area_ratio must be in [0,1]")
        for name in ("mask_path", "defect_type"):
            if name in defect and (not isinstance(defect[name], str) or not defect[name]):
                raise ContractError(f"defects[{index}].{name} must be a non-empty string")


def load_feature_record(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ContractError("Feature record root must be an object")
    validate_feature_record(record)
    return record


def save_feature_record(record: Mapping[str, Any], path: Path | str) -> Path:
    validate_feature_record(record)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return output
