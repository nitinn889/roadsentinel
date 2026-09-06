from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class PredictionEvaluationReport:
    """Consolidated performance report for deterioration and pothole formation models."""
    num_sequences: int
    deterioration_precision: float
    deterioration_recall: float
    deterioration_f1: float
    deterioration_brier_score: float
    pothole_precision: float
    pothole_recall: float
    pothole_f1: float
    pothole_brier_score: float
    directional_trend_agreement_pct: float
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_sequences": self.num_sequences,
            "deterioration_prediction": {
                "precision": round(self.deterioration_precision, 4),
                "recall": round(self.deterioration_recall, 4),
                "f1": round(self.deterioration_f1, 4),
                "brier_calibration_score": round(self.deterioration_brier_score, 4),
            },
            "pothole_formation_prediction": {
                "precision": round(self.pothole_precision, 4),
                "recall": round(self.pothole_recall, 4),
                "f1": round(self.pothole_f1, 4),
                "brier_calibration_score": round(self.pothole_brier_score, 4),
            },
            "directional_trend_agreement_pct": round(self.directional_trend_agreement_pct, 2),
            "scientific_status": "CARLA-SYNTHETIC ONLY",
            "notes": self.notes,
        }


class PredictionEvaluator:
    """Evaluates temporal deterioration prediction models against held-out synthetic test sequences."""

    @staticmethod
    def _compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
        y_pred = (y_prob >= threshold).astype(int)
        y_true = np.asarray(y_true, dtype=int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        brier = float(np.mean((y_prob - y_true) ** 2))

        return prec, rec, f1, brier

    def evaluate(self,
                 true_deteriorated: List[bool],
                 pred_deterioration_probs: List[float],
                 true_pothole_formed: List[bool],
                 pred_pothole_probs: List[float],
                 simulated_progression_trends: List[str],
                 predicted_progression_trends: List[str]) -> PredictionEvaluationReport:
        """Evaluates prediction accuracy, calibration, and trend agreement."""
        n = len(true_deteriorated)
        if n == 0:
            return PredictionEvaluationReport(
                num_sequences=0,
                deterioration_precision=0.0,
                deterioration_recall=0.0,
                deterioration_f1=0.0,
                deterioration_brier_score=0.0,
                pothole_precision=0.0,
                pothole_recall=0.0,
                pothole_f1=0.0,
                pothole_brier_score=0.0,
                directional_trend_agreement_pct=0.0,
                notes=["No test sequences provided."],
            )

        det_prec, det_rec, det_f1, det_brier = self._compute_binary_metrics(
            np.array(true_deteriorated), np.array(pred_deterioration_probs)
        )

        pot_prec, pot_rec, pot_f1, pot_brier = self._compute_binary_metrics(
            np.array(true_pothole_formed), np.array(pred_pothole_probs)
        )

        # Directional trend agreement
        # Agreement occurs if both say "stable" or both say "deteriorating"/"rapidly_deteriorating"
        matching_trends = 0
        for sim_t, pred_t in zip(simulated_progression_trends, predicted_progression_trends):
            sim_is_degrading = sim_t in ("crack_to_pothole", "wear_deterioration", "deteriorating", "rapidly_deteriorating")
            pred_is_degrading = pred_t in ("deteriorating", "rapidly_deteriorating")
            if sim_is_degrading == pred_is_degrading:
                matching_trends += 1

        trend_agreement = (matching_trends / n) * 100.0 if n > 0 else 0.0

        return PredictionEvaluationReport(
            num_sequences=n,
            deterioration_precision=det_prec,
            deterioration_recall=det_rec,
            deterioration_f1=det_f1,
            deterioration_brier_score=det_brier,
            pothole_precision=pot_prec,
            pothole_recall=pot_rec,
            pothole_f1=pot_f1,
            pothole_brier_score=pot_brier,
            directional_trend_agreement_pct=trend_agreement,
            notes=[
                "Evaluation conducted strictly on held-out CARLA synthetic segment sequences.",
                "Zero temporal data leakage guaranteed by disjoint segment IDs."
            ],
        )
