#!/usr/bin/env python3
"""Run the trained Model V1 through the shared RoadSentinel feature adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from feature_adapter import MODEL_FEATURES, build_model_features, load_scenario

import sys

HERE = Path(__file__).resolve().parent
CONTRACT_DIR = HERE.parent / "sam2_dino"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from feature_contract import load_feature_record


CONFIG_PATH = HERE / "config" / "model_v1.json"
MODEL_PATH = HERE / "model" / "model_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_record", type=Path)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    record = load_feature_record(args.feature_record)
    scenario = load_scenario(args.scenario)
    all_features = build_model_features(record, scenario)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    feature_order = list(config["feature_order"])
    missing = [name for name in feature_order if math.isnan(all_features[name])]
    if missing:
        raise ValueError(f"Model V1 requires measured values for: {', '.join(missing)}")

    row = np.asarray([[all_features[name] for name in feature_order]], dtype=float)
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    raw_prediction = float(model.predict(row)[0])
    forecast = min(1.0, max(0.0, raw_prediction))

    payload = {
        "forecast_status": "MODEL-BASED FORECAST",
        "scenario_status": scenario["scenario_status"],
        "model_id": config["model_id"],
        "image_id": record["image_id"],
        "segment_id": record["segment_id"],
        "day": record["day"],
        "original_filename": record["original_filename"],
        "future_severity": forecast,
        "prediction_clipped_to_0_1": raw_prediction != forecast,
        "features_used": {
            name: all_features[name]
            for name in feature_order
        },
        "interface_fields_not_used_by_model_v1": [
            name for name in MODEL_FEATURES if name not in feature_order
        ],
        "target_definition": "Normalized future NYC pavement-rating severity on [0,1].",
        "target_caveat": (
            "The training target is not asserted to be identical to RoadSentinel "
            "image-derived severity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
