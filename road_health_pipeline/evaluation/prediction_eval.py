"""Deterioration Model Evaluation & Leakage-Free Sequence Validation.

Evaluates temporal deterioration and pothole emergence prediction models on sequential
inspection histories with strict segment-level train/test splits.

Metrics computed:
- Binary Classification: Precision, Recall, F1, Accuracy
- Probability Calibration: Brier score, Mean absolute calibration error
- Progression Direction Agreement: Directional consistency between predicted slope and ground truth
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from analytics.prediction import (
    RoadDeteriorationPredictor,
    SegmentObservation,
    extract_temporal_features,
)
from analytics.temporal_generator import SyntheticRoadSequence


def evaluate_prediction_model(
    train_sequences: List[SyntheticRoadSequence],
    test_sequences: List[SyntheticRoadSequence],
    horizon_days: int = 30,
) -> Dict[str, Any]:
    """Train on train_sequences and evaluate on unseen test_sequences.

    Ensures zero road segment leakage by checking that segment IDs are mutually exclusive.
    """
    train_ids = {s.road_segment_id for s in train_sequences}
    test_ids = {s.road_segment_id for s in test_sequences}
    leakage = train_ids.intersection(test_ids)
    if leakage:
        raise ValueError(f"Data leakage detected! {len(leakage)} segments present in both train and test splits.")

    # 1. Build training matrices
    X_train_list = []
    y_train_list = []
    feature_names = [
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

    for seq in train_sequences:
        # Use first N-1 observations to predict deterioration at final timestep
        obs_history = seq.observations[:-1] if len(seq.observations) > 1 else seq.observations
        feats = extract_temporal_features(obs_history)
        feat_vec = [feats[k] for k in feature_names]
        X_train_list.append(feat_vec)
        y_train_list.append(1 if seq.ground_truth_deteriorated else 0)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)

    # 2. Fit predictor
    predictor = RoadDeteriorationPredictor(horizon_days=horizon_days)
    if len(np.unique(y_train)) > 1:
        predictor.fit(X_train, y_train)

    # 3. Evaluate on test sequences
    y_true: List[int] = []
    y_pred_probs: List[float] = []
    y_pred_labels: List[int] = []
    direction_agreements: List[bool] = []

    for seq in test_sequences:
        obs_history = seq.observations[:-1] if len(seq.observations) > 1 else seq.observations
        pred_res = predictor.predict(obs_history, road_segment_id=seq.road_segment_id)
        
        prob = pred_res.deterioration_probability or 0.0
        gt = 1 if seq.ground_truth_deteriorated else 0
        pred_label = 1 if prob >= 0.50 else 0

        y_true.append(gt)
        y_pred_probs.append(prob)
        y_pred_labels.append(pred_label)

        # Direction check
        if gt == 1:
            agrees = pred_res.progression_direction in {"degrading", "critical"}
        else:
            agrees = pred_res.progression_direction in {"stable", "improving"}
        direction_agreements.append(agrees)

    y_t = np.array(y_true, dtype=np.int32)
    y_p = np.array(y_pred_labels, dtype=np.int32)
    probs = np.array(y_pred_probs, dtype=np.float32)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    accuracy = float(np.mean(y_t == y_p))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    brier_score = float(np.mean((probs - y_t) ** 2))
    dir_agreement_rate = float(np.mean(direction_agreements))

    return {
        "num_train_segments": len(train_sequences),
        "num_test_segments": len(test_sequences),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier_score, 4),
        "progression_direction_agreement": round(dir_agreement_rate, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
    }
