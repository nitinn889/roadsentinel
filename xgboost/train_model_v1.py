#!/usr/bin/env python3
"""Train RoadSentinel Model V1 and a linear baseline on real longitudinal data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
TABLE = HERE / "data" / "processed" / "training_pairs.csv"
CONFIG_PATH = HERE / "config" / "model_v1.json"
MODEL_DIR = HERE / "model"
OUTPUT_DIR = HERE / "outputs"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = int(config["random_seed"])
    feature_order = list(config["feature_order"])

    with TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sites = sorted({row["site_id"] for row in rows})
    if len(sites) < 5:
        raise RuntimeError(f"Insufficient unique sites for site split: {len(sites)}")

    rng = np.random.default_rng(seed)
    shuffled_sites = np.asarray(sites, dtype=object)
    rng.shuffle(shuffled_sites)
    test_count = max(1, int(round(len(sites) * float(config["test_fraction_by_site"]))))
    test_sites = set(shuffled_sites[:test_count].tolist())
    train_sites = set(shuffled_sites[test_count:].tolist())
    train_rows = [row for row in rows if row["site_id"] in train_sites]
    test_rows = [row for row in rows if row["site_id"] in test_sites]
    if not train_rows or not test_rows or train_sites & test_sites:
        raise RuntimeError("Invalid site-based split")

    def matrix(selected: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([[float(row[name]) for name in feature_order] for row in selected])
        y = np.asarray([float(row[config["target"]]) for row in selected])
        return x, y

    x_train, y_train = matrix(train_rows)
    x_test, y_test = matrix(test_rows)

    params = dict(config["xgboost_parameters"])
    params["random_state"] = seed
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    xgb_prediction = np.clip(model.predict(x_test), 0.0, 1.0)

    baseline = LinearRegression()
    baseline.fit(x_train, y_train)
    baseline_prediction = np.clip(baseline.predict(x_test), 0.0, 1.0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model_v1.json"
    baseline_path = MODEL_DIR / "linear_baseline_v1.joblib"
    model.save_model(model_path)
    joblib.dump(baseline, baseline_path)

    split = {
        "strategy": "site-based holdout using OFTCode",
        "random_seed": seed,
        "test_fraction_requested": config["test_fraction_by_site"],
        "train_site_count": len(train_sites),
        "test_site_count": len(test_sites),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "site_overlap_count": len(train_sites & test_sites),
        "train_sites": sorted(train_sites),
        "test_sites": sorted(test_sites),
    }
    normalization = {
        "source_variable": "NYC SystemRating",
        "source_documented_range": [1.0, 10.0],
        "source_direction": "1 is low/poor; 10 is high/good",
        "formula": "severity = (10 - SystemRating) / 9",
        "output_range": [0.0, 1.0],
        "fit_on_training_data": False,
        "caveat": (
            "This normalized rating is a common forecasting scale and is not asserted "
            "to be identical to RoadSentinel image-derived severity."
        ),
    }
    results = {
        "model_id": config["model_id"],
        "dataset": "NYC Open Data Street Pavement Ratings (6yyb-pb25)",
        "feature_order": feature_order,
        "target": config["target"],
        "xgboost_version": xgboost.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost": metrics(y_test, xgb_prediction),
        "linear_regression": metrics(y_test, baseline_prediction),
        "prediction_clipping": "Both model predictions clipped to [0,1] for evaluation.",
        "model_path": str(model_path.relative_to(HERE)),
        "baseline_path": str(baseline_path.relative_to(HERE)),
    }
    (OUTPUT_DIR / "split_metadata.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "normalization_metadata.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "feature_order.json").write_text(
        json.dumps({"feature_order": feature_order}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
