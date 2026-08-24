from __future__ import annotations

import math
import numpy as np


def horizontal_vertical_fov(horizontal_fov_deg: float, width: int, height: int) -> tuple[float, float]:
    hfov = math.radians(horizontal_fov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (height / width))
    return hfov, vfov


def footprint_m(width: int, height: int, altitude_m: float, horizontal_fov_deg: float) -> tuple[float, float]:
    hfov, vfov = horizontal_vertical_fov(horizontal_fov_deg, width, height)
    return 2 * altitude_m * math.tan(hfov / 2), 2 * altitude_m * math.tan(vfov / 2)


def target_capture_speed_mps(altitude_m: float, horizontal_fov_deg: float, width: int, height: int,
                             interval_s: float, target_overlap: float) -> float:
    _, footprint_y = footprint_m(width, height, altitude_m, horizontal_fov_deg)
    desired_distance = footprint_y * (1.0 - target_overlap)
    return desired_distance / interval_s


def actual_forward_overlap(speed_mps: float, interval_s: float, altitude_m: float,
                           horizontal_fov_deg: float, width: int, height: int) -> float:
    _, footprint_y = footprint_m(width, height, altitude_m, horizontal_fov_deg)
    if footprint_y <= 0:
        return 0.0
    travel = speed_mps * interval_s
    return max(0.0, min(1.0, 1.0 - travel / footprint_y))


def intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float) -> tuple[float, float, float, float]:
    hfov = math.radians(horizontal_fov_deg)
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def pixel_rays_on_ground(points_xy: np.ndarray, width: int, height: int,
                         altitude_m: float, horizontal_fov_deg: float) -> np.ndarray:
    """Nadir camera, zero roll/pitch: maps image pixels to local ground plane.

    Returns XY coordinates in metres relative to the camera's ground projection.
    """
    fx, fy, cx, cy = intrinsics_from_fov(width, height, horizontal_fov_deg)
    pts = np.asarray(points_xy, dtype=np.float64)
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    # For nadir, normalized image axes map to ground axes with scale altitude.
    gx = altitude_m * x
    gy = altitude_m * y
    return np.column_stack([gx, gy])


def polygon_area_m2(mask: np.ndarray, altitude_m: float, horizontal_fov_deg: float) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 0.0
    width = mask.shape[1]
    height = mask.shape[0]
    # Exact-ish projection by summing local pixel footprint. This assumes nadir and planar ground.
    hfov, vfov = horizontal_vertical_fov(horizontal_fov_deg, width, height)
    pixel_w = 2 * altitude_m * math.tan(hfov / 2) / width
    pixel_h = 2 * altitude_m * math.tan(vfov / 2) / height
    return float(mask.sum() * pixel_w * pixel_h)
