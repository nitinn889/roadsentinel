#!/usr/bin/env python3
"""Train the scenario-aware RoadSentinel XGBoost model on joined LTPP data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
TABLE = HERE / "data" / "processed" / "scenario_training_pairs.csv"
CONFIG_PATH = HERE / "config" / "scenario_model_v2.json"
MODEL_PATH = HERE / "model" / "scenario_model_v2.json"
METRICS_PATH = HERE / "outputs" / "scenario_model_v2_metrics.json"

SCENARIO_FIELDS = (
    "rainfall_level",
    "traffic_level",
    "temperature",
    "water_exposure",
)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sites = sorted({row["site_id"] for row in rows})
    if len(sites) < 10:
        raise RuntimeError(f"Insufficient joined LTPP sites for a site split: {len(sites)}")

    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)
    shuffled_sites = np.asarray(sites, dtype=object)
    rng.shuffle(shuffled_sites)
    test_count = max(1, int(round(len(sites) * float(config["test_fraction_by_site"]))))
    test_sites = set(shuffled_sites[:test_count].tolist())
    train_sites = set(shuffled_sites[test_count:].tolist())
    train_rows = [row for row in rows if row["site_id"] in train_sites]
    test_rows = [row for row in rows if row["site_id"] in test_sites]
    if not train_rows or not test_rows or train_sites & test_sites:
        raise RuntimeError("Invalid site-based scenario-model split")

    # Fit the IRI-to-severity mapping only on training sites.  Quantile scaling
    # is data-derived and avoids inventing a fixed severity jump or IRI ceiling.
    train_iri = np.asarray(
        [
            float(row[name])
            for row in train_rows
            for name in ("current_iri_m_per_km", "future_iri_m_per_km")
        ],
        dtype=float,
    )
    iri_low = float(np.quantile(train_iri, 0.01))
    iri_high = float(np.quantile(train_iri, 0.99))
    if not iri_high > iri_low:
        raise RuntimeError("Cannot normalize IRI: training quantiles are not distinct")

    def severity(iri: float) -> float:
        return float(np.clip((iri - iri_low) / (iri_high - iri_low), 0.0, 1.0))

    feature_order = list(config["feature_order"])

    def matrix(selected: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        for row in selected:
            values = {
                "current_severity": severity(float(row["current_iri_m_per_km"])),
                **{name: float(row[name]) for name in SCENARIO_FIELDS},
                "days_ahead": float(row["days_ahead"]),
            }
            x_rows.append([values[name] for name in feature_order])
            y_rows.append(severity(float(row["future_iri_m_per_km"])))
        return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)

    x_train, y_train = matrix(train_rows)
    x_test, y_test = matrix(test_rows)
    params = dict(config["xgboost_parameters"])
    params["random_state"] = seed
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    raw_test_predictions = model.predict(x_test)
    test_predictions = np.clip(raw_test_predictions, 0.0, 1.0)

    feature_ranges = {
        name: {
            "minimum": float(np.min(x_train[:, index])),
            "maximum": float(np.max(x_train[:, index])),
        }
        for index, name in enumerate(feature_order)
    }
    scenario_quantiles = {
        name: _quantiles(x_train[:, feature_order.index(name)])
        for name in SCENARIO_FIELDS
    }
    normal = {name: scenario_quantiles[name]["median"] for name in SCENARIO_FIELDS}
    presets = {"NORMAL": dict(normal)}
    for preset_name, feature_name in (
        ("HEAVY_RAIN", "rainfall_level"),
        ("HEAVY_TRAFFIC", "traffic_level"),
        ("HIGH_HEAT", "temperature"),
        ("WET_EXPOSURE", "water_exposure"),
    ):
        presets[preset_name] = dict(normal)
        presets[preset_name][feature_name] = scenario_quantiles[feature_name]["p90"]

    config.update(
        {
            "model_path": str(MODEL_PATH.relative_to(HERE)),
            "training_table": str(TABLE.relative_to(HERE)),
            "training_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_site_count": len(train_sites),
            "test_site_count": len(test_sites),
            "training_horizon_range_days": [
                int(feature_ranges["days_ahead"]["minimum"]),
                int(feature_ranges["days_ahead"]["maximum"]),
            ],
            "feature_ranges": feature_ranges,
            "scenario_quantiles": scenario_quantiles,
            "scenario_presets": presets,
            "iri_severity_mapping": {
                "formula": "clip((IRI - training_p01) / (training_p99 - training_p01), 0, 1)",
                "training_p01_m_per_km": iri_low,
                "training_p99_m_per_km": iri_high,
                "fit_on_training_sites_only": True,
            },
        }
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "model_id": config["model_id"],
        "dataset": "FHWA LTPP InfoPave SDR 40 joined scenario table",
        "xgboost_version": xgboost.__version__,
        "split_strategy": "seeded site-based holdout",
        "random_seed": seed,
        "train_sites": sorted(train_sites),
        "test_sites": sorted(test_sites),
        "site_overlap_count": len(train_sites & test_sites),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "feature_order": feature_order,
        "feature_importance_gain_proxy": {
            name: float(value)
            for name, value in zip(feature_order, model.feature_importances_)
        },
        "test_metrics": _metrics(y_test, test_predictions),
        "test_prediction_clipped_count": int(
            np.count_nonzero(raw_test_predictions != test_predictions)
        ),
        "model_path": str(MODEL_PATH.relative_to(HERE)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
