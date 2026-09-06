"""CARLA Synthetic Temporal Sequence Generator for Pavement Deterioration.

Generates controlled, reproducible synthetic progression sequences across road segments
(e.g., t1: healthy/microcrack -> t2: fatigue cracking -> t3: surface depression -> t4: pothole formation -> t5: water ponding)
for training and evaluating deterioration prediction models without leaking segment identity between train and test.

Scientific Status:
- CARLA-SYNTHETIC ONLY
- Simulates realistic wear dynamics with noise; not claiming real-world validated degradation dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from analytics.prediction import SegmentObservation


@dataclass
class SyntheticRoadSequence:
    """A multi-timestamp synthetic progression sequence for a specific road segment."""

    road_segment_id: str
    progression_type: str  # "accelerated_failure", "gradual_wear", "stable_healthy", "repaired"
    observations: List[SegmentObservation]
    ground_truth_deteriorated: bool
    ground_truth_pothole_formed: bool


def generate_synthetic_segment_sequence(
    segment_id: str,
    progression_type: str = "gradual_wear",
    num_timesteps: int = 4,
    time_step_days: float = 14.0,
    seed: int = 42,
) -> SyntheticRoadSequence:
    """Generate a single road segment longitudinal inspection history.

    Parameters
    ----------
    segment_id:
        Unique identifier for the road segment.
    progression_type:
        Type of simulation progression:
        - "accelerated_failure": Healthy road undergoes rapid fatigue, cracking, and severe pothole emergence.
        - "gradual_wear": Healthy road slowly degrades into surface roughness and minor cracking.
        - "stable_healthy": Pristine road remains healthy with minor environmental noise.
        - "repaired": Deteriorated road receives maintenance and returns to healthy.
    num_timesteps:
        Number of inspection timestamps in sequence (e.g. 4 for t1..t4).
    time_step_days:
        Interval between successive drone inspection passes.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    SyntheticRoadSequence
    """
    rng = np.random.default_rng(seed)
    observations: List[SegmentObservation] = []

    base_score = 95.0 + rng.uniform(-2.0, 2.0)
    base_area = 0.0
    potholes = 0
    max_sev = 0.0
    water = False

    for t in range(num_timesteps):
        day = t * time_step_days

        if progression_type == "accelerated_failure":
            # Rapid degradation: healthy -> cracks -> deep pothole -> water
            if t == 0:
                health = base_score
                potholes = 0
                area = 0.0
                max_sev = 0.05
                water = False
            elif t == 1:
                health = base_score - 15.0 - rng.uniform(0, 5)
                potholes = 0
                area = 0.4 + rng.uniform(0, 0.2)
                max_sev = 0.35 + rng.uniform(0, 0.1)
                water = False
            elif t == 2:
                health = base_score - 40.0 - rng.uniform(0, 8)
                potholes = 1
                area = 1.2 + rng.uniform(0, 0.3)
                max_sev = 0.70 + rng.uniform(0, 0.1)
                water = rng.random() > 0.5
            else:
                health = base_score - 65.0 - rng.uniform(0, 10)
                potholes = 2
                area = 2.5 + rng.uniform(0, 0.5)
                max_sev = 0.90 + rng.uniform(0, 0.08)
                water = True

        elif progression_type == "gradual_wear":
            # Slow wear: 95 -> 90 -> 84 -> 78
            drop = t * (4.5 + rng.uniform(0, 1.5))
            health = max(40.0, base_score - drop)
            potholes = 0 if t < 3 else (1 if rng.random() > 0.6 else 0)
            area = max(0.0, t * 0.25 + rng.uniform(-0.05, 0.05))
            max_sev = max(0.0, t * 0.15 + rng.uniform(-0.02, 0.02))
            water = False

        elif progression_type == "stable_healthy":
            # Noise around 95
            health = base_score + rng.uniform(-2.5, 2.5)
            potholes = 0
            area = 0.0
            max_sev = rng.uniform(0.0, 0.05)
            water = False

        elif progression_type == "repaired":
            # Poor -> Repaired
            if t < num_timesteps // 2:
                health = 45.0 + rng.uniform(-3, 3)
                potholes = 1
                area = 1.0
                max_sev = 0.65
                water = False
            else:
                health = 92.0 + rng.uniform(-2, 2)
                potholes = 0
                area = 0.0
                max_sev = 0.05
                water = False
        else:
            raise ValueError(f"Unknown progression type: {progression_type}")

        health_clipped = float(np.clip(health, 0.0, 100.0))
        obs = SegmentObservation(
            timestamp=f"T{t:02d}",
            road_health_score=round(health_clipped, 2),
            pothole_count=potholes,
            total_defects=potholes + (1 if area > 0 else 0),
            damaged_area_m2=round(float(area), 3),
            max_severity=round(float(max_sev), 3),
            avg_severity=round(float(max_sev * 0.8), 3),
            has_water_hazard=water,
            day_offset=float(day),
        )
        observations.append(obs)

    gt_deteriorated = (observations[-1].road_health_score < observations[0].road_health_score - 15.0)
    gt_pothole_formed = (observations[-1].pothole_count > 0 and observations[0].pothole_count == 0)

    return SyntheticRoadSequence(
        road_segment_id=segment_id,
        progression_type=progression_type,
        observations=observations,
        ground_truth_deteriorated=gt_deteriorated,
        ground_truth_pothole_formed=gt_pothole_formed,
    )


def generate_synthetic_benchmark_dataset(
    num_segments: int = 40,
    timesteps: int = 4,
    seed: int = 42,
) -> List[SyntheticRoadSequence]:
    """Generate a diverse synthetic benchmark dataset with split-ready segments."""
    types = ["accelerated_failure", "gradual_wear", "stable_healthy", "repaired"]
    sequences: List[SyntheticRoadSequence] = []
    
    for i in range(num_segments):
        ptype = types[i % len(types)]
        seg_id = f"sim_segment_{i:04d}"
        seq = generate_synthetic_segment_sequence(
            segment_id=seg_id,
            progression_type=ptype,
            num_timesteps=timesteps,
            time_step_days=14.0,
            seed=seed + i * 13,
        )
        sequences.append(seq)
    
    return sequences
