from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent


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
