#!/usr/bin/env python3
"""Read the shared perception schema and prepare a future XGBoost row.

Day 1 intentionally does not fit a model or emit a prediction. Missing
perception measurements remain NaN in the numeric row for future XGBoost use.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
CONTRACT_DIR = HERE.parent / "sam2_dino"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from feature_contract import load_feature_record


CURRENT_FEATURES = (
    "current_severity",
    "crack_area_ratio",
    "defect_area_ratio",
    "defect_count",
    "surface_anomaly_score",
    "water_flag",
)
SCENARIO_FEATURES = (
    "rainfall_level",
    "traffic_level",
    "temperature",
    "water_exposure",
    "days_ahead",
)
MODEL_FEATURES = CURRENT_FEATURES + SCENARIO_FEATURES
TARGET = "future_severity"


class ScenarioError(ValueError):
    pass


def validate_scenario(scenario: Mapping[str, Any]) -> None:
    allowed = {"scenario_status", *SCENARIO_FEATURES}
    missing = sorted(allowed - set(scenario))
    extra = sorted(set(scenario) - allowed)
    if missing:
        raise ScenarioError(f"Missing scenario fields: {', '.join(missing)}")
    if extra:
        raise ScenarioError(f"Unknown scenario fields: {', '.join(extra)}")
    if scenario["scenario_status"] not in {"USER_SUPPLIED", "MOCK_INTERFACE_ONLY"}:
        raise ScenarioError("scenario_status must be USER_SUPPLIED or MOCK_INTERFACE_ONLY")
    for name in SCENARIO_FEATURES:
        value = scenario[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ScenarioError(f"{name} must be a finite number")
    if scenario["rainfall_level"] < 0 or scenario["traffic_level"] < 0:
        raise ScenarioError("rainfall_level and traffic_level must be non-negative")
    if not 0.0 <= float(scenario["water_exposure"]) <= 1.0:
        raise ScenarioError("water_exposure must be in [0,1]")
    if not isinstance(scenario["days_ahead"], int) or isinstance(scenario["days_ahead"], bool) or scenario["days_ahead"] < 1:
        raise ScenarioError("days_ahead must be a positive integer")


def build_model_features(record: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    """Return one numeric feature mapping in fixed ``MODEL_FEATURES`` order."""
    validate_scenario(scenario)
    values: dict[str, float] = {}
    for name in CURRENT_FEATURES:
        value = record[name]
        if isinstance(value, bool):
            values[name] = float(value)
        elif value is None:
            values[name] = math.nan
        else:
            values[name] = float(value)
    for name in SCENARIO_FEATURES:
        values[name] = float(scenario[name])
    return values


def load_scenario(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        scenario = json.load(handle)
    if not isinstance(scenario, dict):
        raise ScenarioError("Scenario root must be an object")
    validate_scenario(scenario)
    return scenario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_record", type=Path)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    record = load_feature_record(args.feature_record)
    scenario = load_scenario(args.scenario)
    features = build_model_features(record, scenario)
    payload = {
        "interface_status": scenario["scenario_status"],
        "schema_version": record["schema_version"],
        "image_id": record["image_id"],
        "segment_id": record["segment_id"],
        "day": record["day"],
        "original_filename": record["original_filename"],
        "feature_names": list(MODEL_FEATURES),
        "model_row": [None if math.isnan(features[name]) else features[name] for name in MODEL_FEATURES],
        "target_name": TARGET,
        "future_severity": None,
        "prediction_status": "NOT_RUN_NO_TRAINED_XGBOOST_MODEL",
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
