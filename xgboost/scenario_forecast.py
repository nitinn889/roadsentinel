#!/usr/bin/env python3
"""Forecast one road image under a user-selected deterioration scenario."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from xgboost import XGBRegressor

from feature_adapter import (
    SCENARIO_FEATURES,
    SCENARIO_NAMES,
    ScenarioError,
    build_model_features,
)

HERE = Path(__file__).resolve().parent
CONTRACT_DIR = HERE.parent / "sam2_dino"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from feature_contract import load_feature_record, validate_feature_record


CONFIG_PATH = HERE / "config" / "scenario_model_v2.json"
MODEL_PATH = HERE / "model" / "scenario_model_v2.json"
FORECAST_LABEL = "MODEL-BASED FORECAST"


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_id",
        "feature_order",
        "feature_ranges",
        "scenario_presets",
        "supported_horizons_days",
    }
    missing = sorted(required - set(config))
    if missing:
        raise RuntimeError(
            f"Scenario model is not trained; config lacks: {', '.join(missing)}"
        )
    return config


def scenario_from_preset(
    scenario_name: str,
    days_ahead: int,
    *,
    overrides: Mapping[str, float | None] | None = None,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Resolve one named preset to the numeric model inputs.

    Preset values are training-distribution quantiles saved by
    ``train_scenario_model.py``.  Optional user values replace those numeric
    inputs before inference; the name itself is never sent to XGBoost.
    """
    name = scenario_name.upper()
    if name not in SCENARIO_NAMES:
        raise ScenarioError(
            f"scenario_name must be one of: {', '.join(SCENARIO_NAMES)}"
        )
    config = _load_config(config_path)
    scenario: dict[str, Any] = {
        "scenario_status": "USER_SUPPLIED",
        "scenario_name": name,
        **config["scenario_presets"][name],
        "days_ahead": days_ahead,
    }
    for field, value in (overrides or {}).items():
        if field not in SCENARIO_FEATURES[:-1]:
            raise ScenarioError(f"Unknown numeric scenario override: {field}")
        if value is not None:
            scenario[field] = value
    _validate_supported_scenario(scenario, config)
    return scenario


def _validate_supported_scenario(
    scenario: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    # ``build_model_features`` performs the common type/schema checks.  The
    # trained interface adds explicit horizon and interpolation-domain checks.
    placeholder = {
        "current_severity": 0.5,
        "crack_area_ratio": None,
        "defect_area_ratio": None,
        "defect_count": 0,
        "surface_anomaly_score": None,
        "water_flag": None,
    }
    build_model_features(placeholder, scenario)
    if scenario.get("scenario_status") != "USER_SUPPLIED":
        raise ScenarioError("Day 3 forecasts require scenario_status USER_SUPPLIED")
    horizons = [int(value) for value in config["supported_horizons_days"]]
    if scenario["days_ahead"] not in horizons:
        raise ScenarioError(
            f"days_ahead must be one of the trained interface horizons: {horizons}"
        )
    for name in SCENARIO_FEATURES[:-1]:
        value = float(scenario[name])
        bounds = config["feature_ranges"][name]
        if not float(bounds["minimum"]) <= value <= float(bounds["maximum"]):
            raise ScenarioError(
                f"{name}={value} is outside the training range "
                f"[{bounds['minimum']}, {bounds['maximum']}]"
            )


def predict_future_severity(
    feature_record: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    model_path: Path = MODEL_PATH,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Return one deterministic, scenario-conditioned model forecast.

    Every numeric scenario value is placed directly in the trained feature
    row.  The event name is metadata only and cannot alter the prediction.
    """
    validate_feature_record(feature_record)
    config = _load_config(config_path)
    _validate_supported_scenario(scenario, config)
    all_features = build_model_features(feature_record, scenario)
    feature_order = list(config["feature_order"])
    missing = [name for name in feature_order if math.isnan(all_features[name])]
    if missing:
        raise ValueError(f"Scenario Model V2 requires measured values for: {', '.join(missing)}")

    row = np.asarray([[all_features[name] for name in feature_order]], dtype=float)
    model = XGBRegressor()
    model.load_model(model_path)
    raw_prediction = float(model.predict(row)[0])
    future_severity = min(1.0, max(0.0, raw_prediction))
    current_severity = float(feature_record["current_severity"])
    scenario_name = str(scenario.get("scenario_name", "CUSTOM")).upper()

    return {
        "forecast_status": FORECAST_LABEL,
        "forecast_caveat": (
            "Scenario-conditioned model estimate; not a guaranteed future condition."
        ),
        "model_id": config["model_id"],
        "model_path": str(model_path),
        "image_id": feature_record["image_id"],
        "segment_id": feature_record["segment_id"],
        "original_filename": feature_record["original_filename"],
        "scenario_status": scenario["scenario_status"],
        "scenario_name": scenario_name,
        "current_severity": current_severity,
        "future_severity": future_severity,
        "severity_change": future_severity - current_severity,
        "prediction_clipped_to_0_1": raw_prediction != future_severity,
        "features_used": {name: all_features[name] for name in feature_order},
        "target_definition": (
            "Future LTPP MRI/IRI mapped to [0,1] with training-site P01/P99 bounds."
        ),
        "scale_transfer_caveat": (
            "RoadSentinel image severity has not been calibrated to the LTPP IRI severity scale."
        ),
        "unused_perception_fields": [
            name
            for name in (
                "crack_area_ratio",
                "defect_area_ratio",
                "defect_count",
                "surface_anomaly_score",
                "water_flag",
            )
            if name not in feature_order
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_record", type=Path)
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, required=True)
    parser.add_argument("--days-ahead", type=int, choices=(30, 60, 90), required=True)
    parser.add_argument("--rainfall-level", type=float)
    parser.add_argument("--traffic-level", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--water-exposure", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    overrides = {
        "rainfall_level": args.rainfall_level,
        "traffic_level": args.traffic_level,
        "temperature": args.temperature,
        "water_exposure": args.water_exposure,
    }
    scenario = scenario_from_preset(
        args.scenario,
        args.days_ahead,
        overrides=overrides,
    )
    payload = predict_future_severity(load_feature_record(args.feature_record), scenario)
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
