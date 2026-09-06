#!/usr/bin/env python3
"""
rs_phase4_drone_sensor.py
-------------------------
RoadSentinel - Phase 4: Sensor Rig Integration & Drone Flight Controller.
Simulates a downward-facing high-resolution RGB camera on a proxy drone actor (RS_Drone),
captures lossless frames, projects Phase 2 defect bounding boxes into image coordinates,
generates ground-truth JSON annotations and preview images, and provides 3 flight modes.
"""

import os
import sys
import json
import math
import time
import shutil
import queue
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Workspace paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

MANIFEST_PATH = WORKSPACE_ROOT / "env" / "output" / "logs" / "rs_defects_manifest.json"
FRAMES_DIR = WORKSPACE_ROOT / "env" / "output" / "images" / "frames"
ANNOTATIONS_DIR = WORKSPACE_ROOT / "env" / "output" / "images" / "annotations"
PREVIEW_DIR = WORKSPACE_ROOT / "env" / "output" / "images" / "preview"

# CARLA import (graceful)
try:
    import carla
    HAS_CARLA = True
except ImportError:
    carla = None
    HAS_CARLA = False


# ==============================================================================
# Camera Sensor Configuration
# ==============================================================================

@dataclass
class CameraConfig:
    width: int = int(os.environ.get("RS_CAM_WIDTH", "1920"))
    height: int = int(os.environ.get("RS_CAM_HEIGHT", "1080"))
    fov: float = float(os.environ.get("RS_CAM_FOV", "90.0"))
    shutter_speed: float = float(os.environ.get("RS_CAM_SHUTTER", "0.004"))
    iso: int = int(os.environ.get("RS_CAM_ISO", "400"))
    lens_k: float = float(os.environ.get("RS_CAM_LENS_K", "-0.1"))
    altitude_m: float = 25.0  # Default 25m AGL

    @property
    def focal_length(self) -> float:
        # Pinhole focal length from horizontal FOV
        return (self.width / 2.0) / math.tan(math.radians(self.fov / 2.0))

    @property
    def principal_point(self) -> Tuple[float, float]:
        return (self.width / 2.0, self.height / 2.0)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        f = self.focal_length
        cx, cy = self.principal_point
        return np.array([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


# ==============================================================================
# Safety & Output Integrity
# ==============================================================================

def check_disk_space_gb(target_path: Path) -> float:
    """Returns free disk space in Gigabytes for the filesystem containing target_path."""
    try:
        usage = shutil.disk_usage(str(target_path.parent if target_path.parent.exists() else WORKSPACE_ROOT))
        return usage.free / (1024.0 ** 3)
    except Exception:
        return 999.0


def check_disk_space_guard(target_path: Path) -> Tuple[bool, str]:
    """Safety check: free space must not fall below 2.0 GB."""
    free_gb = check_disk_space_gb(target_path)
    if free_gb < 2.0:
        msg = f"[RoadSentinel Warning] Free disk space critical: {free_gb:.2f} GB remaining (threshold: 2.0 GB)!"
        print(msg, file=sys.stderr)
        return False, msg
    return True, f"Free space: {free_gb:.2f} GB"


# ==============================================================================
# 3D to 2D Bounding Box Projection
# ==============================================================================

def project_world_point_to_camera(
    world_pt_m: Tuple[float, float, float],
    drone_pose_m: Tuple[float, float, float, float],  # x, y, z, yaw_deg
    cam_cfg: CameraConfig
) -> Optional[Tuple[float, float, float]]:
    """
    Projects a 3D world coordinate (meters) to 2D pixel coordinates (u, v)
    with downward-facing camera (pitch -90°).
    Returns (u, v, z_c) or None if behind camera.
    """
    wx, wy, wz = world_pt_m
    dx, dy, dz, yaw_deg = drone_pose_m

    # Downward camera: altitude difference is optical depth Zc
    # Camera optical axis +Zc is straight down (-Z_world)
    z_c = dz - wz
    if z_c <= 0.1:  # Behind or at camera plane
        return None

    # Translation in drone horizontal frame
    delta_x = wx - dx
    delta_y = wy - dy

    rad = math.radians(yaw_deg)
    cos_yaw = math.cos(rad)
    sin_yaw = math.sin(rad)

    # Rotate into drone forward (+X) / right (+Y) frame
    # Camera image coordinates:
    # +X_camera points right (along image u): corresponds to drone +Y
    # +Y_camera points down (along image v): corresponds to -drone +X
    drone_fwd = delta_x * cos_yaw + delta_y * sin_yaw
    drone_right = -delta_x * sin_yaw + delta_y * cos_yaw

    x_c = drone_right
    y_c = -drone_fwd

    # Pinhole projection
    f = cam_cfg.focal_length
    cx, cy = cam_cfg.principal_point

    u_ideal = cx + f * (x_c / z_c)
    v_ideal = cy + f * (y_c / z_c)

    # Apply radial lens distortion (k = -0.1)
    norm_x = (u_ideal - cx) / f
    norm_y = (v_ideal - cy) / f
    r2 = norm_x * norm_x + norm_y * norm_y
    distort_factor = 1.0 + cam_cfg.lens_k * r2

    u_distorted = cx + (u_ideal - cx) * distort_factor
    v_distorted = cy + (v_ideal - cy) * distort_factor

    return (u_distorted, v_distorted, z_c)


def project_defect_bounding_box(
    defect_entry: dict,
    drone_pose_m: Tuple[float, float, float, float],
    cam_cfg: CameraConfig
) -> Optional[dict]:
    """
    Takes a defect entry from Phase 2 manifest (coordinates in Unreal cm)
    and projects its 8-corner bounding box into camera image coordinates.
    """
    bbox_corners_cm = defect_entry.get("bounding_box_world", [])
    if len(bbox_corners_cm) != 8:
        return None

    proj_corners = []
    for corner in bbox_corners_cm:
        # Convert Unreal cm to meters
        world_pt_m = (corner[0] / 100.0, corner[1] / 100.0, corner[2] / 100.0)
        proj = project_world_point_to_camera(world_pt_m, drone_pose_m, cam_cfg)
        if proj:
            proj_corners.append(proj)

    if not proj_corners:
        return None

    us = [p[0] for p in proj_corners]
    vs = [p[1] for p in proj_corners]

    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)

    # Check if inside frame
    W, H = cam_cfg.width, cam_cfg.height
    if u_max < 0 or u_min > W or v_max < 0 or v_min > H:
        return None

    # Clamped 2D box
    box_xmin = max(0.0, min(float(W), u_min))
    box_ymin = max(0.0, min(float(H), v_min))
    box_xmax = max(0.0, min(float(W), u_max))
    box_ymax = max(0.0, min(float(H), v_max))

    # Visibility estimate (overlap with frame)
    orig_area = max(1.0, (u_max - u_min) * (v_max - v_min))
    clamped_area = max(0.0, (box_xmax - box_xmin) * (box_ymax - box_ymin))
    visibility = round(min(1.0, clamped_area / orig_area), 3)

    if visibility <= 0.05:
        return None

    return {
        "defect_type": defect_entry.get("defect_type", "Unknown"),
        "actor_id": defect_entry.get("actor_id", ""),
        "bbox_2d": [round(box_xmin, 1), round(box_ymin, 1), round(box_xmax, 1), round(box_ymax, 1)],
        "bbox_3d_world": bbox_corners_cm,
        "visibility": visibility
    }


# ==============================================================================
# Flight Modes Controller
# ==============================================================================

class DroneFlightController:
    """
    Implements 3 flight modes for the drone camera rig:
    - hover: stationary at altitude (default 25m AGL) above GPS waypoint.
    - transect: fly straight-line transect above road segment at speed (default 5 m/s) and altitude.
    - survey_grid: lawnmower pattern over bounding box with configurable overlap (default 70%).
    """

    def __init__(
        self,
        flight_mode: str = "hover",
        altitude_m: float = 25.0,
        speed_mps: float = 5.0,
        origin_xy: Tuple[float, float] = (50.0, 0.0)
    ):
        self.flight_mode = flight_mode
        self.altitude_m = altitude_m
        self.speed_mps = speed_mps
        self.origin_x, self.origin_y = origin_xy
        self.current_x = self.origin_x
        self.current_y = self.origin_y
        self.current_yaw_deg = 0.0
        self.elapsed_time_s = 0.0

        # Survey Grid parameters
        self.grid_bounds = (0.0, 200.0, -15.0, 15.0)  # Xmin, Xmax, Ymin, Ymax
        self.grid_overlap_pct = 70.0
        self.grid_lane_idx = 0
        self.grid_direction = 1.0

    def update(self, dt_s: float) -> Tuple[float, float, float, float]:
        """Advances flight by dt_s and returns updated drone pose (x, y, z, yaw_deg)."""
        self.elapsed_time_s += dt_s

        if self.flight_mode == "hover":
            self.current_x = self.origin_x
            self.current_y = self.origin_y
            self.current_yaw_deg = 0.0

        elif self.flight_mode == "transect":
            # Fly along road (+X axis) at constant speed
            self.current_x += self.speed_mps * dt_s
            self.current_y = self.origin_y
            self.current_yaw_deg = 0.0
            # Wrap around at 250m
            if self.current_x > 250.0:
                self.current_x = 0.0

        elif self.flight_mode == "survey_grid":
            # Lawnmower pattern
            xmin, xmax, ymin, ymax = self.grid_bounds
            lane_spacing_m = self.altitude_m * math.tan(math.radians(45.0)) * 2.0 * (1.0 - self.grid_overlap_pct / 100.0)
            lane_spacing_m = max(5.0, lane_spacing_m)

            self.current_x += self.grid_direction * self.speed_mps * dt_s
            if self.grid_direction > 0 and self.current_x >= xmax:
                self.current_x = xmax
                self.grid_direction = -1.0
                self.current_y += lane_spacing_m
                self.current_yaw_deg = 180.0
            elif self.grid_direction < 0 and self.current_x <= xmin:
                self.current_x = xmin
                self.grid_direction = 1.0
                self.current_y += lane_spacing_m
                self.current_yaw_deg = 0.0

            if self.current_y > ymax:
                self.current_y = ymin

        return (self.current_x, self.current_y, self.altitude_m, self.current_yaw_deg)


# ==============================================================================
# Frame Capture & Ground-Truth Overlay Pipeline
# ==============================================================================

class DroneSensorPipeline:
    """
    Sensor capture pipeline:
    - Downward RGB camera capture (lossless PNG).
    - Defect projection and annotation generation (JSON).
    - Lossy preview rendering with co-registered bounding boxes (JPEG).
    - Disk safety validation (> 2 GB).
    """

    def __init__(self, cam_cfg: Optional[CameraConfig] = None):
        self.cam_cfg = cam_cfg or CameraConfig()
        self.flight_ctrl = DroneFlightController(altitude_m=self.cam_cfg.altitude_m)
        self.manifest_defects: List[dict] = []
        self.load_manifest()

        # Ensure output directories exist
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

        self.last_frame_path: Optional[Path] = None
        self.last_annotation_path: Optional[Path] = None
        self.last_preview_path: Optional[Path] = None
        self.total_frames_captured: int = 0
        self.is_capturing: bool = False

    def load_manifest(self):
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.manifest_defects = data.get("defects", [])
            except Exception as e:
                print(f"[RoadSentinel] Error loading manifest: {e}")

    def render_synthetic_drone_frame(
        self,
        drone_pose_m: Tuple[float, float, float, float],
        weather_preset: str = "ClearNoon"
    ) -> Image.Image:
        """
        Renders a realistic synthetic downward road view when CARLA server is not attached.
        Renders asphalt texture, road markings, wetness reflections, and surface defect craters.
        """
        W, H = self.cam_cfg.width, self.cam_cfg.height
        dx, dy, dz, yaw = drone_pose_m

        # Base asphalt background
        base_color = (48, 50, 52)
        if "Night" in weather_preset:
            base_color = (18, 20, 24)
        elif "Rain" in weather_preset:
            base_color = (32, 34, 38)
        elif "Dusk" in weather_preset:
            base_color = (60, 52, 45)

        img = Image.new("RGB", (W, H), base_color)
        draw = ImageDraw.Draw(img)

        # Draw road lane markings (white dashed & yellow solid)
        # Centerline is at world Y = 0.0
        center_pt = project_world_point_to_camera((dx, 0.0, 0.0), drone_pose_m, self.cam_cfg)
        if center_pt:
            cu = center_pt[0]
            # Yellow center double stripe
            draw.line([(cu - 4, 0), (cu - 4, H)], fill=(220, 180, 40), width=4)
            draw.line([(cu + 4, 0), (cu + 4, H)], fill=(220, 180, 40), width=4)

            # Lane dividers (-3.75m and +3.75m)
            for offset_y in [-3.75, 3.75]:
                lane_pt = project_world_point_to_camera((dx, offset_y, 0.0), drone_pose_m, self.cam_cfg)
                if lane_pt:
                    lu = lane_pt[0]
                    # Dashed white lines
                    for y_start in range(0, H, 60):
                        draw.line([(lu, y_start), (lu, y_start + 30)], fill=(230, 230, 230), width=3)

        # Render visual representation of defects in frame
        for defect in self.manifest_defects:
            proj = project_defect_bounding_box(defect, drone_pose_m, self.cam_cfg)
            if proj:
                bx0, by0, bx1, by1 = proj["bbox_2d"]
                dtype = proj["defect_type"]
                if dtype == "PotholeWet":
                    # Dark puddle with blueish highlight
                    draw.ellipse([bx0, by0, bx1, by1], fill=(15, 25, 35), outline=(30, 70, 110), width=2)
                elif dtype == "PotholeDry":
                    draw.ellipse([bx0, by0, bx1, by1], fill=(25, 22, 20), outline=(10, 10, 10), width=2)
                elif dtype == "LongitudinalCrack":
                    draw.line([(bx0, (by0+by1)/2), (bx1, (by0+by1)/2)], fill=(12, 12, 12), width=3)
                else:  # AlligatorCrack
                    draw.rectangle([bx0, by0, bx1, by1], fill=(30, 28, 26), outline=(20, 18, 15), width=1)

        return img

    def capture_frame(
        self,
        carla_image: Optional[Any] = None,
        weather_preset: str = "ClearNoon"
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a single frame capture tick:
        1. Checks disk space (>= 2 GB guard).
        2. Retrieves or renders camera image.
        3. Projects defect bounding boxes into image coordinates.
        4. Writes raw frame, annotation JSON, and preview overlay.
        """
        ok, msg = check_disk_space_guard(FRAMES_DIR)
        if not ok:
            return None

        timestamp_ms = int(time.time() * 1000)
        frame_id = f"frame_{timestamp_ms:015d}"

        # Get current drone pose
        drone_pose = (
            self.flight_ctrl.current_x,
            self.flight_ctrl.current_y,
            self.flight_ctrl.altitude_m,
            self.flight_ctrl.current_yaw_deg
        )

        # Obtain RGB image
        if carla_image is not None and HAS_CARLA:
            # Convert CARLA raw image buffer
            arr = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            arr = arr.reshape((self.cam_cfg.height, self.cam_cfg.width, 4))
            rgb_arr = arr[:, :, :3]  # BGRA to BGR / RGB
            rgb_arr = rgb_arr[:, :, ::-1]  # BGR to RGB
            pil_img = Image.fromarray(rgb_arr)
        else:
            # Synthetic render
            pil_img = self.render_synthetic_drone_frame(drone_pose, weather_preset)

        # 1. Save raw lossless PNG frame
        raw_frame_path = FRAMES_DIR / f"{frame_id}.png"
        pil_img.save(raw_frame_path, format="PNG")
        self.last_frame_path = raw_frame_path

        # 2. Project defect bounding boxes
        visible_defects = []
        for defect in self.manifest_defects:
            proj = project_defect_bounding_box(defect, drone_pose, self.cam_cfg)
            if proj:
                visible_defects.append(proj)

        # 3. Write ground-truth annotation JSON
        camera_pose_dict = {
            "x_m": round(drone_pose[0], 3),
            "y_m": round(drone_pose[1], 3),
            "z_m": round(drone_pose[2], 3),
            "yaw_deg": round(drone_pose[3], 2),
            "pitch_deg": -90.0,
            "roll_deg": 0.0
        }

        annotation_data = {
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "camera_pose": camera_pose_dict,
            "weather_preset": weather_preset,
            "image_dimensions": {"width": self.cam_cfg.width, "height": self.cam_cfg.height},
            "intrinsic_matrix": self.cam_cfg.intrinsic_matrix.tolist(),
            "total_defects_in_frame": len(visible_defects),
            "defects": visible_defects
        }

        annotation_path = ANNOTATIONS_DIR / f"{frame_id}.json"
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(annotation_data, f, indent=2)
        self.last_annotation_path = annotation_path

        # 4. Render preview with 2D bounding boxes and save as JPEG (quality 85)
        preview_img = pil_img.copy()
        draw_preview = ImageDraw.Draw(preview_img)

        defect_colors = {
            "PotholeWet": (0, 200, 255),
            "PotholeDry": (255, 140, 0),
            "LongitudinalCrack": (255, 50, 50),
            "AlligatorCrack": (200, 0, 255)
        }

        for d in visible_defects:
            b = d["bbox_2d"]
            col = defect_colors.get(d["defect_type"], (0, 255, 0))
            draw_preview.rectangle(b, outline=col, width=3)
            label = f"{d['defect_type']} ({d['visibility']:.2f})"
            draw_preview.text((b[0] + 4, max(0, b[1] - 16)), label, fill=col)

        # Telemetry overlay banner
        banner_text = (
            f"RoadSentinel Drone Sensor Rig | Mode: {self.flight_ctrl.flight_mode.upper()} | "
            f"Alt: {drone_pose[2]:.1f}m | Weather: {weather_preset} | Defects: {len(visible_defects)}"
        )
        draw_preview.rectangle([0, 0, self.cam_cfg.width, 32], fill=(0, 0, 0, 180))
        draw_preview.text((12, 8), banner_text, fill=(255, 255, 255))

        preview_path = PREVIEW_DIR / f"{frame_id}_preview.jpg"
        preview_img.save(preview_path, format="JPEG", quality=85)
        self.last_preview_path = preview_path

        self.total_frames_captured += 1

        return {
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "raw_frame_path": str(raw_frame_path),
            "annotation_path": str(annotation_path),
            "preview_path": str(preview_path),
            "defects_count": len(visible_defects)
        }


# Singleton sensor pipeline instance
_SENSOR_PIPELINE: Optional[DroneSensorPipeline] = None

def get_sensor_pipeline() -> DroneSensorPipeline:
    global _SENSOR_PIPELINE
    if _SENSOR_PIPELINE is None:
        _SENSOR_PIPELINE = DroneSensorPipeline()
    return _SENSOR_PIPELINE


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Drone Camera Sensor & Capture Pipeline")
    parser.add_argument("--mode", type=str, default="hover", choices=["hover", "transect", "survey_grid"],
                        help="Drone flight mode")
    parser.add_argument("--altitude", type=float, default=25.0, help="Drone altitude in meters AGL")
    parser.add_argument("--speed", type=float, default=5.0, help="Drone flight speed in m/s")
    parser.add_argument("--frames", type=int, default=3, help="Number of frames to capture")
    parser.add_argument("--weather", type=str, default="ClearNoon", help="Active weather preset name")
    parser.add_argument("--status", action="store_true", help="Print sensor status and exit")

    args, unknown = parser.parse_known_args()

    pipeline = get_sensor_pipeline()
    pipeline.flight_ctrl.flight_mode = args.mode
    pipeline.flight_ctrl.altitude_m = args.altitude
    pipeline.flight_ctrl.speed_mps = args.speed

    if args.status:
        free_gb = check_disk_space_gb(FRAMES_DIR)
        print("\nRoadSentinel Drone Sensor Status:")
        print(f"  - Flight Mode: {pipeline.flight_ctrl.flight_mode}")
        print(f"  - Altitude: {pipeline.flight_ctrl.altitude_m} m")
        print(f"  - Camera: {pipeline.cam_cfg.width}x{pipeline.cam_cfg.height} (FOV: {pipeline.cam_cfg.fov}°)")
        print(f"  - Free Disk Space: {free_gb:.2f} GB")
        print(f"  - Total Captured Frames: {pipeline.total_frames_captured}")
        print(f"  - Manifest Defects Loaded: {len(pipeline.manifest_defects)}")
        return

    print(f"\nStarting capture run ({args.frames} frames in '{args.mode}' mode at {args.altitude}m)...")
    for i in range(args.frames):
        pipeline.flight_ctrl.update(0.5)
        res = pipeline.capture_frame(weather_preset=args.weather)
        if res:
            print(f"  [Frame {i+1:02d}] {res['frame_id']} -> {res['defects_count']} defects detected.")
            print(f"             Preview: {res['preview_path']}")
        time.sleep(0.1)

    print(f"\nCapture complete! Output saved to: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
