from __future__ import annotations

import unittest

from feature_adapter import ScenarioError
from scenario_forecast import (
    FORECAST_LABEL,
    SCENARIO_NAMES,
    predict_future_severity,
    scenario_from_preset,
)


FEATURE_RECORD = {
    "schema_version": "1.0.0",
    "image_id": "DAY3_FIXED_IMAGE",
    "segment_id": "SEG_DAY3_TEST",
    "day": 1,
    "original_filename": "fixed.jpg",
    "condition": None,
    "current_severity": 0.6757,
    "crack_area_ratio": 0.016,
    "defect_area_ratio": 0.019,
    "defect_count": 1,
    "surface_anomaly_score": 0.3744,
    "water_flag": False,
    "defects": [{"area_ratio": 0.019, "defect_type": "crack_or_damage"}],
}


class ScenarioForecastTest(unittest.TestCase):
    def test_identical_inputs_are_deterministic_and_bounded(self) -> None:
        scenario = scenario_from_preset("NORMAL", 30)
        first = predict_future_severity(FEATURE_RECORD, scenario)
        second = predict_future_severity(FEATURE_RECORD, scenario)
        self.assertEqual(first, second)
        self.assertEqual(first["forecast_status"], FORECAST_LABEL)
        self.assertGreaterEqual(first["future_severity"], 0.0)
        self.assertLessEqual(first["future_severity"], 1.0)
        self.assertIn("not a guaranteed", first["forecast_caveat"])

    def test_all_scenario_values_are_model_features(self) -> None:
        scenario = scenario_from_preset("HEAVY_RAIN", 60)
        result = predict_future_severity(FEATURE_RECORD, scenario)
        for name in (
            "rainfall_level",
            "traffic_level",
            "temperature",
            "water_exposure",
            "days_ahead",
        ):
            self.assertEqual(result["features_used"][name], float(scenario[name]))

    def test_event_name_has_no_hidden_prediction_effect(self) -> None:
        first_scenario = scenario_from_preset("NORMAL", 30)
        second_scenario = dict(first_scenario, scenario_name="HEAVY_RAIN")
        first = predict_future_severity(FEATURE_RECORD, first_scenario)
        second = predict_future_severity(FEATURE_RECORD, second_scenario)
        self.assertEqual(first["features_used"], second["features_used"])
        self.assertEqual(first["future_severity"], second["future_severity"])

    def test_same_image_can_produce_scenario_specific_forecasts(self) -> None:
        predictions = {
            name: predict_future_severity(
                FEATURE_RECORD, scenario_from_preset(name, 90)
            )["future_severity"]
            for name in SCENARIO_NAMES
        }
        self.assertGreater(len(set(predictions.values())), 1, predictions)

    def test_supported_horizons(self) -> None:
        for horizon in (30, 60, 90):
            result = predict_future_severity(
                FEATURE_RECORD, scenario_from_preset("NORMAL", horizon)
            )
            self.assertEqual(result["features_used"]["days_ahead"], float(horizon))
        with self.assertRaises(ScenarioError):
            scenario_from_preset("NORMAL", 120)


if __name__ == "__main__":
    unittest.main()
