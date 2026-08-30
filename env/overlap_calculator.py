"""
overlap_calculator.py
---------------------
All the aerial-survey geometry in one place. Given the current altitude,
camera FOV/resolution and constant flight speed, this works out:

  1. the ground footprint of a single photo (width x height, in metres)
  2. the ground sampling distance (GSD - cm covered by one pixel)
  3. the time interval between shots needed to hit the requested forward
     overlap percentage

Recomputed live (not just at startup) so that if you change altitude
in-flight with R/F, the capture interval adapts to keep ~70% overlap
rather than silently drifting off target.
"""

import math
import config


def compute_footprint_m(altitude_m: float,
                         hfov_deg: float = config.CAMERA_FOV_DEG,
                         image_width: int = config.IMAGE_WIDTH,
                         image_height: int = config.IMAGE_HEIGHT):
    """
    Returns (footprint_width_m, footprint_height_m) for a nadir camera at
    the given altitude. "Height" here means the extent along the flight
    direction (i.e. what determines forward overlap), "width" is
    perpendicular to it (side overlap, not currently used but handy if you
    add multi-pass strips later).
    """
    hfov = math.radians(hfov_deg)
    aspect = image_width / image_height  # width / height

    footprint_width_m = 2.0 * altitude_m * math.tan(hfov / 2.0)

    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) / aspect)
    footprint_height_m = 2.0 * altitude_m * math.tan(vfov / 2.0)

    return footprint_width_m, footprint_height_m


def compute_gsd_cm_per_px(altitude_m: float,
                           hfov_deg: float = config.CAMERA_FOV_DEG,
                           image_width: int = config.IMAGE_WIDTH,
                           image_height: int = config.IMAGE_HEIGHT):
    """Ground sampling distance in cm/pixel (same in both axes by construction)."""
    _, footprint_height_m = compute_footprint_m(altitude_m, hfov_deg, image_width, image_height)
    return (footprint_height_m / image_height) * 100.0


def compute_capture_interval_s(speed_mps: float,
                                altitude_m: float,
                                overlap: float = config.FORWARD_OVERLAP,
                                hfov_deg: float = config.CAMERA_FOV_DEG,
                                image_width: int = config.IMAGE_WIDTH,
                                image_height: int = config.IMAGE_HEIGHT) -> float:
    """
    Time between shots (seconds) such that consecutive photos overlap by
    `overlap` fraction, given constant forward speed.

        spacing_m = footprint_height_m * (1 - overlap)
        interval_s = spacing_m / speed_mps
    """
    _, footprint_height_m = compute_footprint_m(altitude_m, hfov_deg, image_width, image_height)
    spacing_m = footprint_height_m * (1.0 - overlap)
    return spacing_m / speed_mps


def startup_report() -> str:
    """Human-readable summary printed once at launch, and any time altitude changes."""
    fw, fh = compute_footprint_m(config.ALTITUDE_M)
    gsd = compute_gsd_cm_per_px(config.ALTITUDE_M)
    interval = compute_capture_interval_s(config.SPEED_MPS, config.ALTITUDE_M)
    lines = [
        "--- Flight / capture parameters ---",
        f"Speed:            {config.SPEED_KMPH:.1f} km/h ({config.SPEED_MPS:.3f} m/s), constant",
        f"Altitude:         {config.ALTITUDE_M:.1f} m",
        f"Camera FOV (H):   {config.CAMERA_FOV_DEG:.1f} deg, {config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}",
        f"Ground footprint: {fw:.1f} m (across) x {fh:.1f} m (along flight path)",
        f"GSD:              {gsd:.2f} cm/pixel",
        f"Target overlap:   {config.FORWARD_OVERLAP * 100:.0f}%",
        f"=> Capture interval: {interval:.2f} s (requested range was 6-7 s; "
        f"tune ALTITUDE_M/CAMERA_FOV_DEG in config.py to shift this - see README)",
        "------------------------------------",
    ]
    return "\n".join(lines)
