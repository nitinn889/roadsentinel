#!/usr/bin/env python3
"""
drone_sim.py
============
RoadSentinel – Configurable & Procedurally Generated CARLA Drone Simulation Testbed.

Simulates a nadir aerial drone survey flight over highway/rural road corridors
with procedural road degradation (potholes, water hazards, cracks, patches) across
configurable road-health scenarios (Healthy, Moderate, Poor, Critical) and weather
conditions (Clear, Overcast, Morning, Sunset, Low-light, Wet, Rain, Post-rain).

Preserves full compatibility with CARLA 0.9.16, the existing drone flight dynamics,
camera sensor configuration, forward photogrammetric overlap (~70%), and standard
RoadSentinel ML ingestion output directories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import pygame

import config
import geo_utils
import overlap_calculator
import road_utils
from drone_controller import DroneController
from metadata_writer import MetadataWriter
from road_injector import (
    ProceduralRoadGenerator,
    RoadDefectManager,
    inject_road_defects,
    cleanup_road_defects,
    project_defects_onto_frame,
)

try:
    import carla
    _HAS_CARLA = True
except ImportError:
    _HAS_CARLA = False


PREVIEW_WIDTH, PREVIEW_HEIGHT = 1280, 720


def image_to_rgb_array(image: "carla.Image") -> np.ndarray:
    """Convert CARLA BGRA raw camera image to RGB NumPy array."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return np.ascontiguousarray(arr[:, :, [2, 1, 0]])


def connect() -> Tuple["carla.Client", "carla.World"]:
    """Connect to local CARLA 0.9.16 server and load configured map."""
    if not _HAS_CARLA:
        raise RuntimeError("CARLA Python package not installed in this environment.")

    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT_S)

    world = client.get_world()
    current_map = world.get_map().name.split("/")[-1]
    if current_map != config.CARLA_MAP:
        print(f"[RoadSentinel] Loading map {config.CARLA_MAP} (currently {current_map})...")
        world = client.load_world(config.CARLA_MAP)
    return client, world


def apply_carla_weather(world: "carla.World", weather_name: str, overrides: Optional[Dict[str, float]] = None) -> None:
    """Apply weather parameters to CARLA world from config.WEATHER_PRESETS."""
    if not _HAS_CARLA or world is None:
        return
    params = config.WEATHER_PRESETS.get(weather_name.lower(), config.WEATHER_PRESETS["clear"])
    if overrides:
        params = {**params, **overrides}

    weather = carla.WeatherParameters(
        cloudiness=float(params.get("cloudiness", 10.0)),
        precipitation=float(params.get("precipitation", 0.0)),
        precipitation_deposits=float(params.get("precipitation_deposits", 0.0)),
        wetness=float(params.get("wetness", 0.0)),
        fog_density=float(params.get("fog_density", 0.0)),
        sun_altitude_angle=float(params.get("sun_altitude_angle", 75.0)),
        sun_azimuth_angle=float(params.get("sun_azimuth_angle", 180.0)),
    )
    world.set_weather(weather)
    print(f"[RoadSentinel] Applied CARLA weather [{weather_name.upper()}]: "
          f"sun_alt={weather.sun_altitude_angle:.0f}°, wetness={weather.wetness:.0f}%, "
          f"rain={weather.precipitation:.0f}%, puddles={weather.precipitation_deposits:.0f}%, "
          f"clouds={weather.cloudiness:.0f}%")


def make_synchronous(world: "carla.World") -> "carla.WorldSettings":
    """Configure world for fixed-step synchronous simulation ticks."""
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = config.FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    return original


def restore_settings(world: "carla.World", original_settings: "carla.WorldSettings") -> None:
    """Restore original world settings upon shutdown."""
    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)


def spawn_drone_camera(world: "carla.World", spawn_transform: "carla.Transform", fov_deg: Optional[float] = None) -> "carla.Actor":
    """Spawn downward-facing nadir RGB camera sensor with calibrated survey lens."""
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(config.IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(config.IMAGE_HEIGHT))
    fov_val = fov_deg if fov_deg is not None else config.CAMERA_FOV_DEG
    camera_bp.set_attribute("fov", str(fov_val))
    camera_bp.set_attribute("sensor_tick", "0.0")
    camera = world.spawn_actor(camera_bp, spawn_transform)
    return camera


def build_hud_lines(
    controller: DroneController,
    meta_writer: MetadataWriter,
    sim_time_s: float,
    next_capture_in_s: float,
    gsd_cm_px: float,
    num_segments: int,
    scenario: str = "moderate",
    weather: str = "clear",
) -> List[str]:
    """Generate on-screen telemetry lines for Pygame HUD."""
    loc = controller.transform.location
    rot = controller.transform.rotation
    return [
        f"RoadSentinel Testbed — Scenario: [{scenario.upper()}]  Weather: [{weather.upper()}]",
        f"Speed: {config.SPEED_KMPH:.1f} km/h (const)   Altitude: {loc.z:.1f} m   Heading: {rot.yaw % 360:.0f}°",
        f"Captured: {meta_writer._image_count} photos   Next shot in: {max(0.0, next_capture_in_s):.1f} s   [{'PAUSED' if controller.paused else 'RECORDING'}]",
        f"GSD: {gsd_cm_px:.2f} cm/px   Corridor: {controller.road_segment_index % max(1, num_segments)}/{num_segments}",
        "W/S flight  A/D strafe  R/F altitude  N/P corridor  SPACE pause  C capture  ESC quit",
    ]


def draw_hud_overlay(screen, hud_lines: List[str], font: Any, frame_rgb: np.ndarray) -> None:
    """Render HUD text lines on screen using Pygame font or fallback OpenCV putText."""
    if font is not None and screen is not None:
        surface = pygame.image.frombuffer(frame_rgb.tobytes(), (frame_rgb.shape[1], frame_rgb.shape[0]), "RGB")
        screen.blit(surface, (0, 0))
        for i, line in enumerate(hud_lines):
            text_surf = font.render(line, True, (0, 242, 254) if i == 0 else (255, 255, 0))
            screen.blit(text_surf, (15, 15 + i * 22))
    elif screen is not None:
        frame_hud = frame_rgb.copy()
        for i, line in enumerate(hud_lines):
            color = (254, 242, 0) if i == 0 else (0, 255, 255)  # RGB
            # Draw black outline then colored text
            cv2.putText(frame_hud, line, (15, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame_hud, line, (15, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        surface = pygame.image.frombuffer(frame_hud.tobytes(), (frame_hud.shape[1], frame_hud.shape[0]), "RGB")
        screen.blit(surface, (0, 0))


def run_standalone_flight_simulator(screen, font, clock, args) -> None:
    """Standalone Procedural Flight Simulator (runs seamlessly with or without CARLA server)."""
    meta_writer = MetadataWriter()
    out_dir = Path(args.output_dir or config.OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / config.IMAGES_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)

    meta_writer.output_dir = str(out_dir)
    meta_writer.images_dir = str(images_dir)
    meta_writer.geo_path = str(out_dir / "geo.txt")
    meta_writer.csv_path = str(out_dir / "metadata.csv")
    meta_writer.log_path = str(out_dir / "capture_log.json")

    duration = args.duration if (args.duration > 0 and args.duration < 99999) else 0.0
    infinite_mode = (duration <= 0)
    sim_time_s = 0.0
    time_since_last_capture = 0.0

    speed_mps = float(args.speed) / 3.6 if args.speed else config.SPEED_MPS
    alt_m = float(args.altitude) if args.altitude else config.ALTITUDE_M
    next_interval_s = overlap_calculator.compute_capture_interval_s(speed_mps, alt_m)

    x_m, y_m = 0.0, 0.0
    yaw_deg = 0.0
    paused = False

    # 1. Procedurally generate road defect plan for the corridor
    corridor_length_m = max(180.0, duration * speed_mps + 50.0) if duration > 0 else 600.0
    generator = ProceduralRoadGenerator(scenario=args.scenario, seed=args.seed)
    plan = generator.generate_corridor_plan(
        segment_length_m=corridor_length_m,
        defects_count=args.num_defects,
        water_ratio_override=args.water_ratio,
    )

    # 2. Build structured ground truth records
    ground_truth_records: List[Dict[str, Any]] = []
    for along_m, across_m, spec in plan:
        lat, lon = geo_utils.local_xy_to_latlon(across_m, along_m)
        ground_truth_records.append({
            "defect_id": spec.defect_id,
            "actor_ids": [1000 + len(ground_truth_records)],
            "segment_index": 0,
            "road_segment_id": "seg_carla_town04_0042",
            "defect_type": spec.defect_type,
            "shape_category": spec.shape_category,
            "carla_location": {"x": round(across_m, 3), "y": round(along_m, 3), "z": 0.035},
            "gps_coordinates": {"latitude": round(lat, 8), "longitude": round(lon, 8), "altitude_m": round(alt_m, 2)},
            "lane_position": spec.lane_position,
            "dimensions": {
                "length_m": round(spec.length_m, 3),
                "width_m": round(spec.width_m, 3),
                "diameter_m": round(spec.diameter_m, 3),
                "depth_m": round(spec.depth_m, 3),
                "area_m2": round(spec.area_m2, 3),
                "aspect_ratio": round(spec.aspect_ratio, 2),
                "orientation_deg": round(spec.orientation_deg, 1),
            },
            "surface_properties": {
                "irregularity": round(spec.irregularity, 3),
                "edge_breakup": round(spec.edge_breakup, 3),
                "roughness": round(spec.roughness, 3),
            },
            "water_state": {
                "is_water_filled": spec.is_water_filled,
                "water_level_m": round(spec.water_depth_m, 3),
                "water_coverage_frac": round(spec.water_coverage_frac, 2),
                "turbidity": round(spec.turbidity, 2),
                "wet_halo_radius_m": round(spec.wet_halo_radius_m, 2),
            },
            "associated_defects": {
                "has_cracks": spec.has_cracks,
                "crack_pattern": spec.crack_pattern,
                "has_road_patch": spec.has_road_patch,
            },
            "clustering": {
                "is_clustered": bool(spec.cluster_id),
                "cluster_id": spec.cluster_id,
                "is_overlapping": spec.is_overlapping,
            },
            "scenario": args.scenario.lower(),
            "severity_category": spec.severity_category,
            "true_severity_score": round(spec.severity_score, 3),
            "generation_seed": args.seed,
        })

    images_sub = out_dir / config.IMAGES_SUBDIR
    images_sub.mkdir(parents=True, exist_ok=True)
    # Clear stale images from older sessions to prevent telemetry mismatch
    for old_f in list(images_sub.glob("*.jpg")) + list(images_sub.glob("*.png")):
        try:
            old_f.unlink()
        except OSError:
            pass

    # Save ground truth metadata
    gt_export = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scenario": args.scenario.lower(),
            "weather": args.weather.lower(),
            "seed": args.seed,
            "altitude_m": alt_m,
            "speed_kmph": args.speed,
            "total_defects": len(ground_truth_records),
            "total_water_filled": sum(1 for d in ground_truth_records if d["water_state"]["is_water_filled"]),
            "corridor_length_m": corridor_length_m,
        },
        "defects": ground_truth_records,
    }

    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_export, f, indent=2)
    with open(out_dir / "road_defects_ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_export, f, indent=2)

    print(f"\n[RoadSentinel] Standalone Procedural Engine initialized [{args.scenario.upper()} | {args.weather.upper()} | seed={args.seed}].")
    print(f"[RoadSentinel] Generated {len(plan)} procedural defects. Ground truth saved to: {out_dir / 'ground_truth.json'}")

    weather_cfg = config.WEATHER_PRESETS.get(args.weather.lower(), config.WEATHER_PRESETS["clear"])
    is_wet_weather = weather_cfg.get("wetness", 0.0) >= 50.0

    # Main flight loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused

        keys = pygame.key.get_pressed()
        dt = config.FIXED_DELTA_SECONDS

        if not paused:
            vy = speed_mps
            if keys[pygame.K_s]:
                vy = -speed_mps * 0.5
            if keys[pygame.K_a]:
                x_m -= 3.0 * dt
            if keys[pygame.K_d]:
                x_m += 3.0 * dt
            if keys[pygame.K_r]:
                alt_m = min(120.0, alt_m + 5.0 * dt)
            if keys[pygame.K_f]:
                alt_m = max(15.0, alt_m - 5.0 * dt)

            y_m += vy * dt
            sim_time_s += dt
            time_since_last_capture += dt

            if not infinite_mode and duration > 0 and sim_time_s >= duration:
                running = False

        # Synthesize Nadir Aerial Camera Frame
        h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
        base_shade = 34 if is_wet_weather else 54
        tex_noise = np.random.normal(0, 3.8, (h, w)).astype(np.int16)
        r_ch = np.clip(base_shade - 2 + tex_noise, 10, 240).astype(np.uint8)
        g_ch = np.clip(base_shade + 2 + tex_noise, 10, 240).astype(np.uint8)
        b_ch = np.clip(base_shade + 6 + tex_noise, 10, 240).astype(np.uint8)
        frame = np.stack([r_ch, g_ch, b_ch], axis=-1)

        # Multi-lane highway road markings
        cx = w // 2
        lane_w_px = int(w * 0.22)

        # White road edge lines
        cv2.line(frame, (cx - lane_w_px * 2, 0), (cx - lane_w_px * 2, h), (210, 210, 210), 3)
        cv2.line(frame, (cx + lane_w_px * 2, 0), (cx + lane_w_px * 2, h), (210, 210, 210), 3)

        # Yellow center divider
        cv2.line(frame, (cx, 0), (cx, h), (220, 180, 45), 2)

        # Dashed lane markings
        dash_offset = int((y_m * 12) % 60)
        for yd in range(-60 + dash_offset, h + 60, 60):
            cv2.line(frame, (cx - lane_w_px, yd), (cx - lane_w_px, yd + 25), (200, 200, 200), 2)
            cv2.line(frame, (cx + lane_w_px, yd), (cx + lane_w_px, yd + 25), (200, 200, 200), 2)

        # Scale factor: pixels per meter based on altitude & FOV
        hfov_rad = math.radians(config.CAMERA_FOV_DEG)
        ground_w_m = 2.0 * alt_m * math.tan(hfov_rad / 2.0)
        px_per_m = float(w) / max(1.0, ground_w_m)

        # Draw procedural defects scrolling under drone using photorealistic 3D depression engine
        frame = project_defects_onto_frame(
            frame=frame,
            drone_x=x_m,
            drone_y=y_m,
            drone_z=alt_m,
            drone_yaw_deg=yaw_deg,
            fov_deg=args.fov or config.CAMERA_FOV_DEG,
            defects=ground_truth_records,
        )

        # Shutter capture trigger
        if not paused and (time_since_last_capture >= next_interval_s):
            idx = meta_writer._image_count
            path, name = meta_writer.image_path_for_index(idx)
            Image.fromarray(frame).save(path, quality=config.JPEG_QUALITY)

            lat, lon = geo_utils.local_xy_to_latlon(x_m, y_m)
            gsd = overlap_calculator.compute_gsd_cm_per_px(alt_m)

            meta_writer.record(
                image_name=name, lat=lat, lon=lon, alt_m=alt_m,
                x_m=x_m, y_m=y_m, yaw_deg=yaw_deg, pitch_deg=-90.0, roll_deg=0.0,
                sim_time_s=sim_time_s, gsd_cm_px=gsd,
            )
            time_since_last_capture = 0.0

        # Preview update
        if not args.headless and screen is not None:
            surf_frame = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            time_txt = f"{max(0.0, duration - sim_time_s):.1f}s REMAINING" if duration > 0 else f"T+{sim_time_s:.1f}s (INFINITE)"
            hud = [
                f"ROADSENTINEL STANDALONE — Scenario: [{args.scenario.upper()}]  Weather: [{args.weather.upper()}]",
                f"Flight [{time_txt}]  Speed: {speed_mps * 3.6:.1f} km/h   Altitude: {alt_m:.1f}m   GSD: {overlap_calculator.compute_gsd_cm_per_px(alt_m):.2f} cm/px",
                f"Position: N+{y_m:.1f}m, E+{x_m:.1f}m   Photos: {meta_writer._image_count}   Next in: {max(0.0, next_interval_s - time_since_last_capture):.1f}s",
                "Controls: W/S flight  A/D strafe  R/F altitude  SPACE pause  ESC finalize output",
            ]
            draw_hud_overlay(screen, hud, font, surf_frame)
            pygame.display.flip()
            clock.tick(60)

    meta_writer.write_run_log({"final_sim_time_s": sim_time_s, "sim_mode": "procedural_standalone"})
    pygame.quit()
    print(f"\n[RoadSentinel] Flight session finalized. {meta_writer._image_count} images + metadata.csv written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="RoadSentinel Configurable & Procedural CARLA Drone Simulation Testbed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", type=str, default="moderate",
                        choices=["healthy", "moderate", "poor", "critical"],
                        help="Predefined road-health degradation scenario")
    parser.add_argument("--weather", type=str, default="clear",
                        choices=list(config.WEATHER_PRESETS.keys()),
                        help="Configurable environmental weather and lighting condition")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible procedural defect placement")
    parser.add_argument("--altitude", type=float, default=config.ALTITUDE_M,
                        help="Drone flight survey altitude in meters")
    parser.add_argument("--speed", type=float, default=config.SPEED_KMPH,
                        help="Drone flight survey speed in km/h")
    parser.add_argument("--num-defects", type=int, default=None,
                        help="Override number of defects to generate per corridor")
    parser.add_argument("--water-ratio", type=float, default=None,
                        help="Override water-filled pothole fraction [0.0 - 1.0]")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Flight duration in seconds (0 = manual infinite flight)")
    parser.add_argument("--auto-fly", action="store_true",
                        help="Automatically cruise forward along corridor at survey speed")
    parser.add_argument("--headless", action="store_true",
                        help="Run without opening graphical Pygame window")
    parser.add_argument("--standalone", action="store_true",
                        help="Force standalone procedural simulation mode without CARLA server")
    parser.add_argument("--fov", type=float, default=config.CAMERA_FOV_DEG,
                        help="Horizontal camera field-of-view in degrees (default 60° survey lens)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Destination output directory for images and telemetry")
    args = parser.parse_args()

    infinite_flight = (args.duration <= 0 or args.duration >= 99999)

    if args.headless or not os.environ.get("DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    screen = pygame.display.set_mode((PREVIEW_WIDTH, PREVIEW_HEIGHT)) if not args.headless else None
    if screen:
        pygame.display.set_caption(f"RoadSentinel - CARLA Drone ({args.scenario.upper()} | {args.weather.upper()})")
        try:
            py_font = getattr(pygame, "font", None)
            if py_font is not None and hasattr(py_font, "SysFont"):
                font = py_font.SysFont("consolas", 18)
            else:
                font = None
        except Exception:
            font = None
    else:
        font = None
    clock = pygame.time.Clock()

    print("\n" + "=" * 68)
    print("      ROADSENTINEL PROCEDURAL CARLA TESTING TESTBED v3.0")
    print("=" * 68)
    print(f"  Scenario:   {args.scenario.upper()}")
    print(f"  Weather:    {args.weather.upper()}")
    print(f"  Seed:       {args.seed}")
    print(f"  Altitude:   {args.altitude:.1f} m")
    print(f"  Speed:      {args.speed:.1f} km/h")
    print(f"  Duration:   {'Infinite (ESC to stop)' if infinite_flight else f'{args.duration:.1f}s'}")
    print("=" * 68 + "\n")

    # If standalone mode forced or CARLA import unavailable, branch to standalone engine directly
    if args.standalone or not _HAS_CARLA:
        run_standalone_flight_simulator(screen, font, clock, args)
        return

    # Attempt connection to live CARLA server
    try:
        client, world = connect()
    except Exception as exc:
        print(f"\n[RoadSentinel] Notice: Could not connect to CARLA server at {config.CARLA_HOST}:{config.CARLA_PORT} ({exc}).")
        print("[RoadSentinel] Transitioning automatically to Standalone Procedural Engine...")
        run_standalone_flight_simulator(screen, font, clock, args)
        return

    # 1. Apply synchronous mode settings
    original_settings = make_synchronous(world)
    carla_map = world.get_map()
    print(f"[RoadSentinel] CARLA Map Active: {carla_map.name}")

    # 2. Apply configured weather & lighting preset
    apply_carla_weather(world, args.weather)

    # 3. Locate straight highway corridors
    segments = road_utils.find_straight_segments(carla_map)
    print(f"[RoadSentinel] Located {len(segments)} candidate highway corridor(s).")

    # 4. Procedurally inject defects onto the road surface
    out_dir = Path(args.output_dir or config.OUTPUT_DIR).resolve()
    defect_manager = inject_road_defects(
        world=world,
        segments=segments,
        scenario=args.scenario,
        seed=args.seed,
        defects_per_segment=args.num_defects,
        water_ratio=args.water_ratio,
        output_dir=out_dir,
        verbose=True,
    )

    # 5. Spawn Nadir Drone Camera
    first_wp = segments[0]
    origin_x, origin_y = first_wp.transform.location.x, first_wp.transform.location.y
    drone_alt = float(args.altitude) if args.altitude else config.ALTITUDE_M
    fov_val = float(args.fov) if args.fov else config.CAMERA_FOV_DEG
    spawn_transform = carla.Transform(
        carla.Location(x=origin_x, y=origin_y, z=drone_alt),
        carla.Rotation(pitch=config.CAMERA_PITCH_DEG, yaw=first_wp.transform.rotation.yaw, roll=0.0),
    )

    camera = spawn_drone_camera(world, spawn_transform, fov_deg=fov_val)
    image_queue: queue.Queue = queue.Queue()
    camera.listen(image_queue.put)

    speed_mps = float(args.speed) / 3.6 if args.speed else config.SPEED_MPS
    controller = DroneController(spawn_transform, speed_mps=speed_mps)
    meta_writer = MetadataWriter()
    meta_writer.output_dir = str(out_dir)
    images_sub = out_dir / config.IMAGES_SUBDIR
    images_sub.mkdir(parents=True, exist_ok=True)
    # Clear stale images from older sessions to prevent telemetry mismatch
    for old_f in list(images_sub.glob("*.jpg")) + list(images_sub.glob("*.png")):
        try:
            old_f.unlink()
        except OSError:
            pass
    meta_writer.images_dir = str(images_sub)
    meta_writer.geo_path = str(out_dir / "geo.txt")
    meta_writer.csv_path = str(out_dir / "metadata.csv")
    meta_writer.log_path = str(out_dir / "capture_log.json")

    sim_time_s = 0.0
    time_since_last_capture = 0.0
    last_applied_segment_index = 0
    next_interval_s = overlap_calculator.compute_capture_interval_s(speed_mps, drone_alt)

    print("[RoadSentinel] Live CARLA flight initialized. Focus preview window or press ESC to finalize.\n")

    try:
        while True:
            events = pygame.event.get()
            controller.handle_events(events)
            if controller.quit_requested:
                break

            if not infinite_flight and args.duration > 0 and sim_time_s >= args.duration:
                print(f"[RoadSentinel] Flight duration target {args.duration:.1f}s reached.")
                break

            keys = pygame.key.get_pressed()
            if args.auto_fly and not controller.is_moving_horizontally(keys):
                keys_list = list(keys)
                keys_list[pygame.K_w] = 1
                keys = keys_list

            seg_idx = controller.road_segment_index % len(segments)
            if seg_idx != last_applied_segment_index:
                wp = segments[seg_idx]
                loc = controller.transform.location
                new_transform = carla.Transform(
                    carla.Location(x=wp.transform.location.x, y=wp.transform.location.y, z=loc.z),
                    carla.Rotation(pitch=config.CAMERA_PITCH_DEG, yaw=wp.transform.rotation.yaw, roll=0.0),
                )
                controller.transform = new_transform
                last_applied_segment_index = seg_idx
                print(f"[RoadSentinel] Switched to road corridor {seg_idx}.")

            new_transform = controller.update(config.FIXED_DELTA_SECONDS, keys)
            camera.set_transform(new_transform)

            world.tick()
            sim_time_s += config.FIXED_DELTA_SECONDS

            try:
                carla_image = image_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            next_interval_s = overlap_calculator.compute_capture_interval_s(
                speed_mps, new_transform.location.z
            )

            rgb = image_to_rgb_array(carla_image)
            # Project photorealistic 3D road depressions onto asphalt
            rgb = project_defects_onto_frame(
                frame=rgb,
                drone_x=new_transform.location.x,
                drone_y=new_transform.location.y,
                drone_z=new_transform.location.z,
                drone_yaw_deg=new_transform.rotation.yaw,
                fov_deg=fov_val,
                defects=defect_manager.ground_truth_records,
            )
            time_since_last_capture += config.FIXED_DELTA_SECONDS
            should_capture = (not controller.paused and time_since_last_capture >= next_interval_s)
            if controller.manual_capture_requested:
                should_capture = True
                controller.manual_capture_requested = False

            if should_capture:
                idx = meta_writer._image_count
                path, name = meta_writer.image_path_for_index(idx)
                Image.fromarray(rgb).save(path, quality=config.JPEG_QUALITY)

                x_m = new_transform.location.x
                y_m = new_transform.location.y
                lat, lon, alt_geo = geo_utils.carla_transform_to_geolocation(carla_map, new_transform.location)
                gsd = overlap_calculator.compute_gsd_cm_per_px(new_transform.location.z)

                meta_writer.record(
                    image_name=name, lat=lat, lon=lon, alt_m=alt_geo or new_transform.location.z,
                    x_m=x_m, y_m=y_m,
                    yaw_deg=new_transform.rotation.yaw, pitch_deg=new_transform.rotation.pitch,
                    roll_deg=new_transform.rotation.roll,
                    sim_time_s=sim_time_s, gsd_cm_px=gsd,
                )
                if idx % 10 == 0:
                    meta_writer.flush()
                time_since_last_capture = 0.0

            if not args.headless and screen is not None:
                small = cv2.resize(rgb, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                gsd_now = overlap_calculator.compute_gsd_cm_per_px(new_transform.location.z)
                hud_lines = build_hud_lines(
                    controller, meta_writer, sim_time_s,
                    next_interval_s - time_since_last_capture, gsd_now, len(segments),
                    scenario=args.scenario, weather=args.weather,
                )
                draw_hud_overlay(screen, hud_lines, font, small)
                pygame.display.flip()
                clock.tick(60)

    finally:
        print("[RoadSentinel] Finalizing flight session...")
        cleanup_road_defects()
        meta_writer.write_run_log({
            "final_sim_time_s": sim_time_s,
            "scenario": args.scenario,
            "weather": args.weather,
            "seed": args.seed,
        })
        camera.stop()
        camera.destroy()
        restore_settings(world, original_settings)
        pygame.quit()
        print(f"[RoadSentinel] Done. {meta_writer._image_count} photos + metadata.csv written to {out_dir}")


if __name__ == "__main__":
    main()
