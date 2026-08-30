from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SeverityWeights:
    """Configurable weights and thresholds for individual defect severity scoring."""
    # Component weights (sum to 1.0)
    weight_area: float = 0.35
    weight_depth: float = 0.30
    weight_water: float = 0.20
    weight_surrounding_damage: float = 0.15

    # Area thresholds in m^2 (small < 0.1m^2, med 0.1-0.5m^2, large > 0.5m^2)
    area_low_m2: float = 0.05
    area_med_m2: float = 0.20
    area_high_m2: float = 0.50

    # Depth thresholds in metres (shallow < 2.5cm, medium 2.5-5cm, deep > 5cm)
    depth_low_m: float = 0.025
    depth_med_m: float = 0.050
    depth_high_m: float = 0.100

    # Water multiplier when water-filled
    water_hazard_bonus: float = 25.0

    # Qualitative severity thresholds (0-100)
    threshold_low: float = 25.0
    threshold_medium: float = 50.0
    threshold_high: float = 75.0


@dataclass(frozen=True)
class RoadHealthWeights:
    """Configurable weights for segment-level road health (0-100, 100 = perfect)."""
    # Max deductions per category
    max_pothole_penalty: float = 50.0
    max_crack_penalty: float = 25.0
    max_water_penalty: float = 15.0
    max_surface_penalty: float = 10.0

    # Density scaling factors
    pothole_density_scale: float = 10.0  # potholes per 100m^2
    crack_area_ratio_scale: float = 0.15  # fraction of road area cracked

    # Condition class thresholds
    condition_good_min: float = 80.0
    condition_fair_min: float = 60.0
    condition_poor_min: float = 40.0
    # < 40 is Critical


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for temporal deterioration and pothole prediction models."""
    default_horizon_days: int = 30
    min_history_steps: int = 2
    deterioration_threshold_health_drop: float = 10.0  # drop in health score to flag deterioration
    pothole_formation_crack_threshold_m2: float = 0.10


@dataclass(frozen=True)
class Config:
    # Paths
    healthy_roads_dir: Path = ROOT / "data" / "healthy_roads"
    memory_bank_dir: Path = ROOT / "output" / "memory_bank"
    output_dir: Path = ROOT / "output"
    sam2_checkpoint: Path = ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # Models
    dinov2_model_name: str = "dinov2_vits14"
    patch_size: int = 14
    dinov2_input_size: int = 518
    device: str = os.getenv("ROADSENTINEL_DEVICE", "cuda")
    batch_size: int = 1
    seed: int = 42

    # Road camera geometry / SAM2 prompt strategy
    camera_mode: str = os.getenv("ROADSENTINEL_CAMERA_MODE", "nadir")  # nadir|forward
    forward_roi_box_fractions: tuple[float, float, float, float] = (0.05, 0.35, 0.95, 1.0)
    nadir_roi_box_fractions: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    road_patch_fraction: float = 0.50

    # Memory bank
    coreset_ratio: float = 0.10
    coreset_max_points: int = 20_000
    coreset_presample_max: int = 200_000
    coreset_block_size: int = 8192
    knn_k: int = 5

    # Inference thresholds. These should be calibrated on a labeled validation set.
    anomaly_percentile: float = 98.0
    candidate_min_area_px: int = 100
    candidate_max_area_fraction: float = 0.35
    pothole_confidence_threshold: float = 0.55

    # Camera / flight
    image_width: int = 1280
    image_height: int = 720
    horizontal_fov_deg: float = 90.0
    capture_interval_s: float = 3.0
    max_speed_mps: float = 30.0 / 3.6
    target_overlap: float = 0.70
    altitude_m: float = 30.0

    # CARLA georeferencing. This is a configurable simulation reference, not real GPS.
    carla_origin_lat: float = 13.0827
    carla_origin_lon: float = 80.2707
    carla_x_is_east: bool = True
    carla_y_is_north: bool = True

    # Segment clustering
    segment_grid_size_m: float = 50.0  # spatial binning for road segments

    # Severity & Road Health & Prediction Configs
    severity: SeverityWeights = field(default_factory=SeverityWeights)
    road_health: RoadHealthWeights = field(default_factory=RoadHealthWeights)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)

    # Pi edge
    edge_max_image_side: int = 1600
    edge_jpeg_quality: int = 88
    edge_sharpness_min: float = 30.0
    edge_brightness_min: float = 10.0
    edge_brightness_max: float = 245.0
    edge_duplicate_phash_distance: int = 4
    edge_queue_dir: Path = ROOT / "output" / "pi_queue"
    uploader_url: str = os.getenv("ROADSENTINEL_UPLOADER_URL", "http://127.0.0.1:8000/infer")


CONFIG = Config()
