from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from common.schemas import PredictionResult
from config import CONFIG, PredictionConfig


@dataclass
class TemporalInspectionRecord:
    """Historical snapshot of a road segment at a specific inspection timestamp."""
    timestamp_days: float  # days elapsed from baseline or epoch
    road_health_score: float  # 0-100
    total_damaged_area_m2: float
    pothole_count: int
    crack_count: int
    max_severity_score: float
    water_present: bool = False


class DeteriorationPredictor:
    """Interpretable temporal deterioration and pothole formation predictor.

    Accepts historical segment inspections, computes rate of structural change,
    and estimates probabilities of future health drop and pothole emergence.

    Scientific status: CARLA-SYNTHETIC ONLY (requires multi-temporal survey data for real validation).
    """

    def __init__(self, cfg: Optional[PredictionConfig] = None):
        self.cfg = cfg or CONFIG.prediction

    def extract_features(self, history: List[TemporalInspectionRecord]) -> Dict[str, float]:
        """Extracts engineered velocity and acceleration features from inspection history."""
        if not history:
            return {
                "latest_health": 100.0,
                "health_velocity": 0.0,
                "area_growth_rate": 0.0,
                "severity_velocity": 0.0,
                "latest_crack_count": 0.0,
                "latest_pothole_count": 0.0,
                "water_exposure_ratio": 0.0,
                "history_length": 0.0,
            }

        # Sort chronologically
        sorted_hist = sorted(history, key=lambda x: x.timestamp_days)
        latest = sorted_hist[-1]
        n = len(sorted_hist)

        if n == 1:
            return {
                "latest_health": latest.road_health_score,
                "health_velocity": 0.0,
                "area_growth_rate": 0.0,
                "severity_velocity": 0.0,
                "latest_crack_count": float(latest.crack_count),
                "latest_pothole_count": float(latest.pothole_count),
                "water_exposure_ratio": 1.0 if latest.water_present else 0.0,
                "history_length": 1.0,
            }

        # Compute time intervals and delta rates
        t_first = sorted_hist[0].timestamp_days
        t_last = sorted_hist[-1].timestamp_days
        dt_total = max(0.1, t_last - t_first)

        health_diff = sorted_hist[-1].road_health_score - sorted_hist[0].road_health_score
        health_velocity = health_diff / dt_total  # negative if degrading (points/day)

        area_diff = sorted_hist[-1].total_damaged_area_m2 - sorted_hist[0].total_damaged_area_m2
        area_growth_rate = max(0.0, area_diff / dt_total)  # m^2/day

        sev_diff = sorted_hist[-1].max_severity_score - sorted_hist[0].max_severity_score
        severity_velocity = sev_diff / dt_total  # severity points/day

        water_exposure = sum(1 for h in sorted_hist if h.water_present) / float(n)

        return {
            "latest_health": latest.road_health_score,
            "health_velocity": float(health_velocity),
            "area_growth_rate": float(area_growth_rate),
            "severity_velocity": float(severity_velocity),
            "latest_crack_count": float(latest.crack_count),
            "latest_pothole_count": float(latest.pothole_count),
            "water_exposure_ratio": float(water_exposure),
            "history_length": float(n),
        }

    def predict(self,
                road_segment_id: str,
                history: List[TemporalInspectionRecord],
                horizon_days: Optional[int] = None) -> PredictionResult:
        """Predicts future deterioration and pothole formation probabilities for horizon_days."""
        horizon = horizon_days or self.cfg.default_horizon_days
        feats = self.extract_features(history)

        latest_health = feats["latest_health"]
        health_velocity = feats["health_velocity"]
        area_growth = feats["area_growth_rate"]
        crack_count = feats["latest_crack_count"]
        pothole_count = feats["latest_pothole_count"]
        water_ratio = feats["water_exposure_ratio"]
        hist_len = feats["history_length"]

        notes = []

        if hist_len < self.cfg.min_history_steps:
            # Single-observation heuristic prior
            notes.append(f"Heuristic baseline used: only {int(hist_len)} inspection observation(s) available.")
            # Higher prior if cracks or existing potholes are present
            base_hazard = (100.0 - latest_health) / 100.0
            deterioration_prob = float(1.0 / (1.0 + math.exp(-3.0 * (base_hazard - 0.3))))
            
            # Pothole formation is driven primarily by existing cracks + water exposure
            pothole_hazard = (crack_count * 0.25) + (water_ratio * 0.3)
            pothole_formation_prob = float(1.0 / (1.0 + math.exp(-4.0 * (pothole_hazard - 0.4))))
        else:
            # Multi-temporal rate-based progression model
            # Projected health drop over horizon = - health_velocity * horizon
            projected_health_loss = max(0.0, -health_velocity * float(horizon))
            
            # Water acceleration factor
            water_multiplier = 1.0 + (1.2 * water_ratio)
            accelerated_loss = projected_health_loss * water_multiplier

            # Logistic calibration for deterioration probability (P(health drop > threshold))
            logits_det = (accelerated_loss - self.cfg.deterioration_threshold_health_drop) / 6.0
            deterioration_prob = float(1.0 / (1.0 + math.exp(-logits_det)))

            # Pothole formation probability: presence of expanding cracks and high moisture
            projected_crack_growth = area_growth * float(horizon)
            pothole_logits = (crack_count * 0.4 + projected_crack_growth * 8.0 + water_ratio * 1.5) - 1.2
            pothole_formation_prob = float(1.0 / (1.0 + math.exp(-pothole_logits)))

        # Determine qualitative progression trend
        if health_velocity < -0.5 or area_growth > 0.02:
            trend = "rapidly_deteriorating"
        elif health_velocity < -0.1 or area_growth > 0.005 or deterioration_prob > 0.5:
            trend = "deteriorating"
        else:
            trend = "stable"

        return PredictionResult(
            deterioration_probability=round(np.clip(deterioration_prob, 0.0, 1.0), 4),
            pothole_formation_probability=round(np.clip(pothole_formation_prob, 0.0, 1.0), 4),
            prediction_horizon_days=horizon,
            progression_trend=trend,
            scientific_status="CARLA-SYNTHETIC ONLY",
            features_used={
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in feats.items()
            },
            notes=notes,
        )
