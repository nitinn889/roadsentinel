from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from prediction.progression_model import TemporalInspectionRecord


@dataclass
class SyntheticSegmentSequence:
    """A simulated multi-timestep inspection sequence for a single CARLA road segment."""
    road_segment_id: str
    progression_type: str  # "healthy_stable", "crack_to_pothole", "wear_deterioration"
    inspections: List[TemporalInspectionRecord]
    ground_truth_deteriorated: bool  # whether health dropped > 10 pts by final day
    ground_truth_pothole_formed: bool  # whether a pothole emerged by final day


class CarlaTemporalSequenceGenerator:
    """Generates controlled synthetic temporal progression sequences modeled after CARLA observations.

    Temporal stages:
    - t1 (day 0): baseline healthy or minor hairline crack
    - t2 (day 15): crack propagation & moisture ingress
    - t3 (day 30): asphalt depression / surface dislodgement
    - t4 (day 45-60): pothole opening & water accumulation

    Explicitly labeled as CARLA-SYNTHETIC ONLY.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_sequence(self, segment_id: str, progression_type: str) -> SyntheticSegmentSequence:
        days = [0.0, 15.0, 30.0, 45.0, 60.0]
        inspections: List[TemporalInspectionRecord] = []

        if progression_type == "healthy_stable":
            for d in days:
                noise_health = float(self.rng.normal(0, 1.0))
                inspections.append(TemporalInspectionRecord(
                    timestamp_days=d,
                    road_health_score=float(np.clip(96.0 + noise_health, 88.0, 100.0)),
                    total_damaged_area_m2=0.0,
                    pothole_count=0,
                    crack_count=0,
                    max_severity_score=0.0,
                    water_present=False,
                ))
            gt_det = False
            gt_pothole = False

        elif progression_type == "crack_to_pothole":
            # Progression: hairline crack -> wide crack + water -> depression -> open pothole
            health_traj = [88.0, 78.0, 65.0, 48.0, 35.0]
            area_traj = [0.02, 0.08, 0.18, 0.35, 0.55]
            pothole_counts = [0, 0, 0, 1, 1]
            crack_counts = [1, 2, 2, 1, 1]
            severity_traj = [15.0, 32.0, 55.0, 78.0, 88.0]
            water_traj = [False, True, True, True, True]

            for i, d in enumerate(days):
                noise = float(self.rng.normal(0, 1.5))
                inspections.append(TemporalInspectionRecord(
                    timestamp_days=d,
                    road_health_score=float(np.clip(health_traj[i] + noise, 0.0, 100.0)),
                    total_damaged_area_m2=float(max(0.0, area_traj[i] + self.rng.normal(0, 0.01))),
                    pothole_count=pothole_counts[i],
                    crack_count=crack_counts[i],
                    max_severity_score=float(np.clip(severity_traj[i] + noise, 0.0, 100.0)),
                    water_present=water_traj[i],
                ))
            gt_det = True
            gt_pothole = True

        else:  # "wear_deterioration"
            # Steady abrasion without full pothole formation
            health_traj = [92.0, 85.0, 78.0, 72.0, 66.0]
            area_traj = [0.01, 0.04, 0.09, 0.14, 0.20]
            pothole_counts = [0, 0, 0, 0, 0]
            crack_counts = [1, 1, 2, 2, 2]
            severity_traj = [10.0, 20.0, 35.0, 42.0, 48.0]

            for i, d in enumerate(days):
                noise = float(self.rng.normal(0, 1.2))
                inspections.append(TemporalInspectionRecord(
                    timestamp_days=d,
                    road_health_score=float(np.clip(health_traj[i] + noise, 0.0, 100.0)),
                    total_damaged_area_m2=float(max(0.0, area_traj[i] + self.rng.normal(0, 0.01))),
                    pothole_count=pothole_counts[i],
                    crack_count=crack_counts[i],
                    max_severity_score=float(np.clip(severity_traj[i] + noise, 0.0, 100.0)),
                    water_present=False,
                ))
            gt_det = True
            gt_pothole = False

        return SyntheticSegmentSequence(
            road_segment_id=segment_id,
            progression_type=progression_type,
            inspections=inspections,
            ground_truth_deteriorated=gt_det,
            ground_truth_pothole_formed=gt_pothole,
        )

    def generate_dataset(self, num_segments: int = 40, train_ratio: float = 0.70
                         ) -> Tuple[List[SyntheticSegmentSequence], List[SyntheticSegmentSequence]]:
        """Generates a dataset with strict segment-level splitting to prevent temporal leakage."""
        types = ["healthy_stable", "crack_to_pothole", "wear_deterioration"]
        all_sequences: List[SyntheticSegmentSequence] = []

        for idx in range(num_segments):
            ptype = types[idx % len(types)]
            seg_id = f"CARLA_SEG_{idx:03d}"
            all_sequences.append(self.generate_sequence(seg_id, ptype))

        # Shuffle segments
        indices = np.arange(num_segments)
        self.rng.shuffle(indices)

        n_train = int(num_segments * train_ratio)
        train_seqs = [all_sequences[i] for i in indices[:n_train]]
        test_seqs = [all_sequences[i] for i in indices[n_train:]]

        return train_seqs, test_seqs
