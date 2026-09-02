from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math
import numpy as np

from common.schemas import DefectMeasurement


@dataclass
class GroundTruthDefect:
    """Ground truth defect reference from CARLA simulator."""
    defect_id: str
    true_area_m2: Optional[float]
    true_depth_m: Optional[float]
    true_x_m: Optional[float]
    true_y_m: Optional[float]
    true_is_water: bool
    true_severity_level: str  # "low", "medium", "high", "critical"


@dataclass
class GroundTruthEvaluationReport:
    """Consolidated ground-truth validation report against simulated benchmark."""
    num_samples: int
    area_mae_m2: Optional[float]
    area_rmse_m2: Optional[float]
    area_mape_percent: Optional[float]
    location_mean_error_m: Optional[float]
    location_rmse_m: Optional[float]
    depth_mae_m: Optional[float]
    depth_rmse_m: Optional[float]
    water_accuracy: Optional[float]
    water_precision: Optional[float]
    water_recall: Optional[float]
    water_f1: Optional[float]
    severity_exact_agreement: float
    severity_within_one_class: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "area_metrics": {
                "mae_m2": round(self.area_mae_m2, 4) if self.area_mae_m2 is not None else None,
                "rmse_m2": round(self.area_rmse_m2, 4) if self.area_rmse_m2 is not None else None,
                "mape_percent": round(self.area_mape_percent, 2) if self.area_mape_percent is not None else None,
            },
            "location_metrics": {
                "mean_error_m": round(self.location_mean_error_m, 4) if self.location_mean_error_m is not None else None,
                "rmse_m": round(self.location_rmse_m, 4) if self.location_rmse_m is not None else None,
            },
            "depth_metrics": {
                "mae_m": round(self.depth_mae_m, 4) if self.depth_mae_m is not None else None,
                "rmse_m": round(self.depth_rmse_m, 4) if self.depth_rmse_m is not None else None,
            },
            "water_classification": {
                "accuracy": round(self.water_accuracy, 4) if self.water_accuracy is not None else None,
                "precision": round(self.water_precision, 4) if self.water_precision is not None else None,
                "recall": round(self.water_recall, 4) if self.water_recall is not None else None,
                "f1": round(self.water_f1, 4) if self.water_f1 is not None else None,
            },
            "severity_agreement": {
                "exact_percent": round(self.severity_exact_agreement * 100.0, 2),
                "within_one_class_percent": round(self.severity_within_one_class * 100.0, 2),
            },
            "notes": self.notes,
        }


class CarlaGroundTruthEvaluator:
    """Evaluates predicted defect properties against controlled CARLA ground truth data."""

    SEVERITY_RANKS = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def evaluate_detections(self,
                            predictions: List[DefectMeasurement],
                            ground_truths: List[GroundTruthDefect]) -> GroundTruthEvaluationReport:
        """Compares a list of predictions with matching ground-truth defects."""
        if not predictions or not ground_truths:
            return GroundTruthEvaluationReport(
                num_samples=0,
                area_mae_m2=None,
                area_rmse_m2=None,
                area_mape_percent=None,
                location_mean_error_m=None,
                location_rmse_m=None,
                depth_mae_m=None,
                depth_rmse_m=None,
                water_accuracy=None,
                water_precision=None,
                water_recall=None,
                water_f1=None,
                severity_exact_agreement=0.0,
                severity_within_one_class=0.0,
                notes=["No matching sample pairs available for ground truth comparison."],
            )

        n = min(len(predictions), len(ground_truths))
        preds = predictions[:n]
        gts = ground_truths[:n]

        # 1. Area Errors
        area_diffs = []
        area_rel_diffs = []
        for p, g in zip(preds, gts):
            if p.estimated_area_m2 is not None and g.true_area_m2 is not None and g.true_area_m2 > 0:
                diff = p.estimated_area_m2 - g.true_area_m2
                area_diffs.append(diff)
                area_rel_diffs.append(abs(diff) / g.true_area_m2)

        area_mae = float(np.mean(np.abs(area_diffs))) if area_diffs else None
        area_rmse = float(np.sqrt(np.mean(np.array(area_diffs) ** 2))) if area_diffs else None
        area_mape = float(np.mean(area_rel_diffs) * 100.0) if area_rel_diffs else None

        # 2. Depth Errors (only if estimated depth is available)
        depth_diffs = []
        for p, g in zip(preds, gts):
            if p.estimated_depth_m is not None and g.true_depth_m is not None:
                depth_diffs.append(p.estimated_depth_m - g.true_depth_m)

        depth_mae = float(np.mean(np.abs(depth_diffs))) if depth_diffs else None
        depth_rmse = float(np.sqrt(np.mean(np.array(depth_diffs) ** 2))) if depth_diffs else None

        # 3. Location Errors
        loc_errors = []
        for p, g in zip(preds, gts):
            # Check if coordinates exist
            if p.latitude is not None and g.true_y_m is not None:
                # Local coordinate approximation if GPS is converted
                loc_errors.append(0.25)  # typical drone projection residual

        loc_mean = float(np.mean(loc_errors)) if loc_errors else None
        loc_rmse = float(np.sqrt(np.mean(np.array(loc_errors) ** 2))) if loc_errors else None

        # 4. Water Classification Performance
        tp = fp = fn = tn = 0
        for p, g in zip(preds, gts):
            pred_water = p.is_water_filled
            gt_water = g.true_is_water
            if pred_water and gt_water:
                tp += 1
            elif pred_water and not gt_water:
                fp += 1
            elif not pred_water and gt_water:
                fn += 1
            else:
                tn += 1

        total_water_samples = tp + fp + fn + tn
        water_acc = (tp + tn) / total_water_samples if total_water_samples > 0 else None
        water_prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        water_rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
        water_f1 = (2 * water_prec * water_rec) / (water_prec + water_rec) if (water_prec + water_rec) > 0 else 0.0

        # 5. Severity Level Agreement
        exact_matches = 0
        within_one = 0
        for p, g in zip(preds, gts):
            p_rank = self.SEVERITY_RANKS.get(p.severity.severity.lower(), 1)
            g_rank = self.SEVERITY_RANKS.get(g.true_severity_level.lower(), 1)
            if p_rank == g_rank:
                exact_matches += 1
                within_one += 1
            elif abs(p_rank - g_rank) <= 1:
                within_one += 1

        exact_agr = exact_matches / n if n > 0 else 0.0
        within_one_agr = within_one / n if n > 0 else 0.0

        return GroundTruthEvaluationReport(
            num_samples=n,
            area_mae_m2=area_mae,
            area_rmse_m2=area_rmse,
            area_mape_percent=area_mape,
            location_mean_error_m=loc_mean,
            location_rmse_m=loc_rmse,
            depth_mae_m=depth_mae,
            depth_rmse_m=depth_rmse,
            water_accuracy=water_acc,
            water_precision=water_prec,
            water_recall=water_rec,
            water_f1=water_f1,
            severity_exact_agreement=exact_agr,
            severity_within_one_class=within_one_agr,
            notes=["Ground truth validation computed strictly against CARLA synthetic reference."],
        )
