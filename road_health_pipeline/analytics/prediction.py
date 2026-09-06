"""Modular Road Deterioration & Pothole Formation Prediction Interface.

Provides an interpretable machine learning baseline (Gradient Boosting / Random Forest)
and feature engineering pipeline to predict road segment deterioration and pothole formation
probabilities over configurable horizons (e.g. 30, 60, 90 days).

Scientific Status:
- CARLA-SYNTHETIC ONLY / PROTOTYPE
- Standard image datasets lack true longitudinal progression; all evaluations must be
  validated on temporal simulation sequences or clearly marked as requiring real temporal data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Union
import logging
import numpy as np

from config import CONFIG
from common.schemas import PredictionResult, SegmentSummary

log = logging.getLogger(__name__)


@dataclass
class SegmentObservation:
    """Historical observation snapshot for a road segment."""

    timestamp: str  # ISO-8601 UTC string or day offset float
    road_health_score: float
    pothole_count: int = 0
    total_defects: int = 0
    damaged_area_m2: float = 0.0
    max_severity: float = 0.0
    avg_severity: float = 0.0
    has_water_hazard: bool = False
    day_offset: Optional[float] = None  # Optional explicit days from t0


def parse_timestamp_days(ts_str: str) -> float:
    """Convert ISO timestamp or numeric string to day float for relative difference."""
    try:
        # Try numeric string (e.g. "0.0", "15.5")
        return float(ts_str)
    except ValueError:
        pass
    try:
        # Parse ISO-8601
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp() / 86400.0  # seconds to days
    except Exception:
        return 0.0


def extract_temporal_features(observations: Sequence[SegmentObservation]) -> dict[str, float]:
    """Extract engineered rate-of-change and state features from observation history.

    Parameters
    ----------
    observations:
        Ordered list of chronological inspections for a single segment.

    Returns
    -------
    dict[str, float]
        Dictionary of statistical and physical progression features.
    """
    if not observations:
        return {
            "health_score_latest": 100.0,
            "health_score_slope_per_day": 0.0,
            "damaged_area_latest_m2": 0.0,
            "damaged_area_growth_rate": 0.0,
            "max_severity_latest": 0.0,
            "severity_slope_per_day": 0.0,
            "pothole_count_latest": 0.0,
            "pothole_growth_rate": 0.0,
            "water_hazard_latest": 0.0,
            "inspection_count": 0.0,
            "timespan_days": 0.0,
        }

    # Sort observations chronologically
    sorted_obs = sorted(
        observations,
        key=lambda o: o.day_offset if o.day_offset is not None else parse_timestamp_days(o.timestamp),
    )

    latest = sorted_obs[-1]
    first = sorted_obs[0]
    n_obs = len(sorted_obs)

    # Compute timespan in days
    if latest.day_offset is not None and first.day_offset is not None:
        timespan_days = max(0.1, latest.day_offset - first.day_offset)
    else:
        t_first = parse_timestamp_days(first.timestamp)
        t_last = parse_timestamp_days(latest.timestamp)
        timespan_days = max(0.1, t_last - t_first)

    # Linear slopes (deltas per day)
    health_delta = latest.road_health_score - first.road_health_score
    health_slope = health_delta / timespan_days if n_obs > 1 else 0.0

    area_delta = latest.damaged_area_m2 - first.damaged_area_m2
    area_growth_rate = area_delta / timespan_days if n_obs > 1 else 0.0

    sev_delta = latest.max_severity - first.max_severity
    sev_slope = sev_delta / timespan_days if n_obs > 1 else 0.0

    pothole_delta = latest.pothole_count - first.pothole_count
    pothole_growth_rate = pothole_delta / timespan_days if n_obs > 1 else 0.0

    return {
        "health_score_latest": float(latest.road_health_score),
        "health_score_slope_per_day": float(health_slope),
        "damaged_area_latest_m2": float(latest.damaged_area_m2),
        "damaged_area_growth_rate": float(area_growth_rate),
        "max_severity_latest": float(latest.max_severity),
        "severity_slope_per_day": float(sev_slope),
        "pothole_count_latest": float(latest.pothole_count),
        "pothole_growth_rate": float(pothole_growth_rate),
        "water_hazard_latest": 1.0 if latest.has_water_hazard else 0.0,
        "inspection_count": float(n_obs),
        "timespan_days": float(timespan_days),
    }


class RoadDeteriorationPredictor:
    """Interpretable temporal predictor for road deterioration and pothole emergence."""

    def __init__(self, horizon_days: int = CONFIG.prediction_default_horizon_days):
        self.horizon_days = horizon_days
        self._classifier = None
        self._feature_names = [
            "health_score_latest",
            "health_score_slope_per_day",
            "damaged_area_latest_m2",
            "damaged_area_growth_rate",
            "max_severity_latest",
            "severity_slope_per_day",
            "pothole_count_latest",
            "pothole_growth_rate",
            "water_hazard_latest",
            "timespan_days",
        ]

    def fit(self, X_features: np.ndarray, y_labels: np.ndarray) -> None:
        """Fit a baseline Random Forest classifier on temporal progression features."""
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=CONFIG.seed,
        )
        clf.fit(X_features, y_labels)
        self._classifier = clf

    def _heuristic_predict(self, features: dict[str, float]) -> tuple[float, float, str]:
        """Explainable rule-based degradation model when ML model or data is limited."""
        score = features["health_score_latest"]
        slope = features["health_score_slope_per_day"]  # negative = degrading
        potholes = features["pothole_count_latest"]
        water = features["water_hazard_latest"]
        growth = features["damaged_area_growth_rate"]

        # Base degradation risk driven by current health deficit
        health_deficit = (100.0 - score) / 100.0
        
        # Accelerated risk from observed rate of score drop
        degrade_rate_penalty = float(np.clip(-slope * self.horizon_days / 30.0, 0.0, 0.50))
        
        # Area expansion acceleration
        area_growth_penalty = float(np.clip(growth * self.horizon_days / 5.0, 0.0, 0.30))

        # Water pooling increases deterioration rate drastically
        water_penalty = 0.15 if water > 0 else 0.0

        deterioration_prob = float(np.clip(
            0.50 * health_deficit + degrade_rate_penalty + area_growth_penalty + water_penalty,
            0.02,
            0.98,
        ))

        # Pothole formation probability (for healthy / cracked road forming new potholes)
        if potholes == 0:
            pothole_prob = float(np.clip(
                0.35 * health_deficit + degrade_rate_penalty * 0.8 + water_penalty * 1.2,
                0.01,
                0.90,
            ))
        else:
            # Already has potholes -> formation of additional potholes / enlargement
            pothole_prob = float(np.clip(
                0.50 + 0.30 * health_deficit + area_growth_penalty,
                0.10,
                0.99,
            ))

        if deterioration_prob > 0.70 or pothole_prob > 0.75:
            direction = "critical"
        elif deterioration_prob > 0.40 or slope < -0.1:
            direction = "degrading"
        elif slope > 0.05:
            direction = "improving"
        else:
            direction = "stable"

        return round(deterioration_prob, 4), round(pothole_prob, 4), direction

    def predict(
        self,
        observations: Sequence[SegmentObservation],
        road_segment_id: Optional[str] = None,
    ) -> PredictionResult:
        """Predict deterioration and pothole emergence probabilities for a segment."""
        features = extract_temporal_features(observations)
        notes = []

        if len(observations) < 2:
            notes.append("Single inspection baseline: temporal rate-of-change estimated via standard degradation prior.")
        else:
            notes.append(f"Multi-temporal analysis computed across {len(observations)} historical observations.")

        if self._classifier is not None:
            feat_vec = np.array([[features[k] for k in self._feature_names]], dtype=np.float32)
            try:
                probs = self._classifier.predict_proba(feat_vec)[0]
                p_deteriorate = float(probs[1]) if len(probs) > 1 else float(probs[0])
                p_pothole = float(np.clip(p_deteriorate * (1.2 if features["pothole_count_latest"] > 0 else 0.8), 0, 1))
                direction = "critical" if p_deteriorate > 0.7 else ("degrading" if p_deteriorate > 0.4 else "stable")
                notes.append("Probabilities computed using trained Random Forest temporal baseline.")
            except Exception as e:
                log.warning("Classifier prediction failed (%s); using rule-based baseline.", e)
                p_deteriorate, p_pothole, direction = self._heuristic_predict(features)
        else:
            p_deteriorate, p_pothole, direction = self._heuristic_predict(features)

        # Feature importances / contributions
        importances = {
            "health_score": round(features["health_score_latest"], 2),
            "health_slope_per_day": round(features["health_score_slope_per_day"], 4),
            "damaged_area_m2": round(features["damaged_area_latest_m2"], 3),
            "area_growth_rate": round(features["damaged_area_growth_rate"], 4),
            "max_severity": round(features["max_severity_latest"], 3),
            "pothole_count": int(features["pothole_count_latest"]),
        }

        return PredictionResult(
            deterioration_probability=p_deteriorate,
            pothole_formation_probability=p_pothole,
            prediction_horizon_days=self.horizon_days,
            progression_direction=direction,
            feature_importances=importances,
            notes=notes,
        )
