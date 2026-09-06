"""Unit tests for synthetic CARLA temporal sequence generation and leakage-free evaluation."""

import pytest
from analytics.temporal_generator import (
    generate_synthetic_benchmark_dataset,
    generate_synthetic_segment_sequence,
)
from evaluation.prediction_eval import evaluate_prediction_model


def test_generate_synthetic_sequence_types():
    seq_fail = generate_synthetic_segment_sequence("seg_fail", "accelerated_failure", num_timesteps=4)
    seq_wear = generate_synthetic_segment_sequence("seg_wear", "gradual_wear", num_timesteps=4)
    seq_stable = generate_synthetic_segment_sequence("seg_stable", "stable_healthy", num_timesteps=4)

    assert len(seq_fail.observations) == 4
    assert seq_fail.observations[-1].road_health_score < seq_fail.observations[0].road_health_score
    assert seq_fail.ground_truth_deteriorated is True

    assert seq_stable.observations[-1].road_health_score >= 85.0
    assert seq_stable.ground_truth_deteriorated is False


def test_evaluate_prediction_model_leakage_free():
    dataset = generate_synthetic_benchmark_dataset(num_segments=20, timesteps=4, seed=42)
    train_seqs = dataset[:12]
    test_seqs = dataset[12:]

    metrics = evaluate_prediction_model(train_seqs, test_seqs, horizon_days=30)
    assert metrics["num_train_segments"] == 12
    assert metrics["num_test_segments"] == 8
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["progression_direction_agreement"] <= 1.0


def test_evaluate_prediction_model_detects_leakage():
    dataset = generate_synthetic_benchmark_dataset(num_segments=10, timesteps=4, seed=42)
    with pytest.raises(ValueError, match="Data leakage detected"):
        evaluate_prediction_model(dataset, dataset)
