"""Probability calibration, threshold determination, and reliability metrics.

Transforms uncalibrated raw anomaly / heuristic scores into well-calibrated
posterior probabilities:
    P(Defect | s) = 1 / (1 + exp(-(a * s + b)))

Provides:
- Platt scaling / Logistic calibration
- Expected Calibration Error (ECE) computation
- Optimal threshold determination via F1-maximization on validation data
- Reliability diagram metrics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    brier_score: float
    expected_calibration_error: float
    optimal_threshold: float
    max_f1: float
    auc_roc: float
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "brier_score": round(self.brier_score, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "max_f1": round(self.max_f1, 4),
            "auc_roc": round(self.auc_roc, 4),
            "bin_accuracies": [round(x, 4) for x in self.bin_accuracies],
            "bin_confidences": [round(x, 4) for x in self.bin_confidences],
            "bin_counts": self.bin_counts,
        }


class ProbabilityCalibrator:
    """Calibrator mapping raw anomaly / confidence scores to true empirical probabilities."""

    def __init__(
        self,
        slope: float = 6.0,
        intercept: float = -2.5,
        optimal_threshold: float = 0.50,
    ) -> None:
        self.slope = slope
        self.intercept = intercept
        self.optimal_threshold = optimal_threshold
        self.is_fitted = False

    def predict_probability(self, raw_score: float | np.ndarray) -> float | np.ndarray:
        """Apply Platt scaling sigmoid to map raw score into [0, 1] probability."""
        z = self.slope * np.asarray(raw_score, dtype=np.float32) + self.intercept
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        if isinstance(raw_score, (float, int)):
            return float(prob)
        return prob.astype(np.float32)

    def fit(
        self,
        raw_scores: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Fit Platt scaling parameters (slope, intercept) using logistic regression."""
        s = np.asarray(raw_scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)

        if len(s) < 10 or len(np.unique(y)) < 2:
            log.warning("Insufficient samples or single-class data for calibration fit.")
            return self.evaluate(s, y, n_bins)

        # Simple Newton-Raphson or SGD logistic regression
        a, b = 4.0, -2.0
        lr = 0.05
        for _ in range(200):
            p = 1.0 / (1.0 + np.exp(-np.clip(a * s + b, -20.0, 20.0)))
            grad_a = np.mean((p - y) * s)
            grad_b = np.mean(p - y)
            a -= lr * grad_a
            b -= lr * grad_b

        self.slope = float(a)
        self.intercept = float(b)
        self.is_fitted = True

        # Find optimal F1 threshold
        probs = self.predict_probability(s)
        best_f1, best_th = 0.0, 0.50
        for th in np.linspace(0.1, 0.9, 81):
            preds = probs >= th
            tp = np.sum((preds == 1) & (y == 1))
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = (2 * prec * rec) / max(1e-6, prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)

        self.optimal_threshold = best_th
        log.info(
            "Fitted Calibrator: slope=%.3f, intercept=%.3f, optimal_th=%.3f (F1: %.3f)",
            self.slope, self.intercept, self.optimal_threshold, best_f1,
        )
        return self.evaluate(s, y, n_bins)

    def evaluate(
        self,
        raw_scores: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Compute Brier Score, ECE, AUC-ROC, and reliability diagram bins."""
        probs = self.predict_probability(raw_scores)
        y = np.asarray(labels, dtype=np.float64)

        brier = float(np.mean((probs - y) ** 2))

        # Expected Calibration Error (ECE)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []
        ece = 0.0
        n_total = len(y)

        for i in range(n_bins):
            in_bin = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
            count = int(np.sum(in_bin))
            bin_counts.append(count)
            if count > 0:
                acc = float(np.mean(y[in_bin]))
                conf = float(np.mean(probs[in_bin]))
                bin_accs.append(acc)
                bin_confs.append(conf)
                ece += (count / n_total) * abs(acc - conf)
            else:
                bin_accs.append(0.0)
                bin_confs.append(float((bins[i] + bins[i + 1]) / 2.0))

        # Trapezoidal AUC-ROC approximation
        thresholds = np.linspace(0.0, 1.0, 50)
        tprs, fprs = [], []
        pos = max(1, np.sum(y == 1))
        neg = max(1, np.sum(y == 0))
        for th in thresholds:
            preds = probs >= th
            tpr = np.sum((preds == 1) & (y == 1)) / pos
            fpr = np.sum((preds == 1) & (y == 0)) / neg
            tprs.append(tpr)
            fprs.append(fpr)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc_roc = float(abs(trapz_fn(tprs, fprs))) if trapz_fn else 0.5

        preds_opt = probs >= self.optimal_threshold
        tp = np.sum((preds_opt == 1) & (y == 1))
        fp = np.sum((preds_opt == 1) & (y == 0))
        fn = np.sum((preds_opt == 0) & (y == 1))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        max_f1 = float((2 * prec * rec) / max(1e-6, prec + rec))

        return CalibrationMetrics(
            brier_score=brier,
            expected_calibration_error=float(ece),
            optimal_threshold=self.optimal_threshold,
            max_f1=max_f1,
            auc_roc=auc_roc,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
        )

    def save(self, path: Path) -> None:
        """Save calibrator parameters to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "slope": self.slope,
            "intercept": self.intercept,
            "optimal_threshold": self.optimal_threshold,
            "is_fitted": self.is_fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Saved ProbabilityCalibrator to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """Load calibrator parameters from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cal = cls(
            slope=data.get("slope", 6.0),
            intercept=data.get("intercept", -2.5),
            optimal_threshold=data.get("optimal_threshold", 0.50),
        )
        cal.is_fitted = data.get("is_fitted", False)
        return cal
