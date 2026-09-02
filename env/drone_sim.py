#!/usr/bin/env python3
"""
drone_sim.py
============
RoadSentinel - CARLA drone-over-highway simulation rig.

What this does, mapped to spec:
  1. Drone flies over the loaded CARLA town (default Town06)              -> config.CARLA_MAP
  2. You control the drone from the keyboard                              -> drone_controller.py
  3. Horizontal speed is capped at a CONSTANT 30 km/h                     -> config.SPEED_KMPH
  4. Altitude chosen for clear photos (tunable)                           -> config.ALTITUDE_M
  5. Default map/road chosen to look like a rural National Highway        -> Town06 highway strip
  6. N / P keys cycle between straight road segments ("control the road") -> road_utils.py
  7. Photos captured at ~70% forward overlap, interval auto-derived       -> overlap_calculator.py
  8. Output written in an ODM/COLMAP-ingestible layout                    -> metadata_writer.py

Run with the CARLA server already running (see the separate
DEMO_INSTRUCTIONS.md). Controls are listed in drone_controller.py and
printed to console on startup.
"""

import json
import os
import queue
import sys
import time

import numpy as np
import pygame
from PIL import Image
import cv2

import carla

import config
import geo_utils
import overlap_calculator
import road_utils
from drone_controller import DroneController
from metadata_writer import MetadataWriter
from road_injector import inject_road_defects, cleanup_road_defects

PREVIEW_WIDTH, PREVIEW_HEIGHT = 1280, 720


def image_to_rgb_array(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))  # CARLA gives BGRA
    return arr[:, :, [2, 1, 0]]  # -> RGB, drop alpha


def connect() -> "carla.World":
    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT_S)

    world = client.get_world()
    current_map = world.get_map().name.split("/")[-1]
    if current_map != config.CARLA_MAP:
        print(f"Loading map {config.CARLA_MAP} (currently {current_map})...")
        world = client.load_world(config.CARLA_MAP)
    return client, world


def make_synchronous(world) -> "carla.WorldSettings":
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = config.FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    return original


def restore_settings(world, original_settings):
    world.apply_settings(original_settings)


def spawn_drone_camera(world, spawn_transform):
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(config.IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(config.IMAGE_HEIGHT))
    camera_bp.set_attribute("fov", str(config.CAMERA_FOV_DEG))
    camera_bp.set_attribute("sensor_tick", "0.0")  # one frame per world tick
    camera = world.spawn_actor(camera_bp, spawn_transform)
    return camera


def build_hud_lines(controller, meta_writer, sim_time_s, next_capture_in_s, gsd_cm_px, num_segments):
    loc = controller.transform.location
    rot = controller.transform.rotation
    return [
        f"Speed: {config.SPEED_KMPH:.1f} km/h (const)   Altitude: {loc.z:.1f} m   Heading: {rot.yaw % 360:.0f} deg",
        f"Photos captured: {meta_writer._image_count}   Next in: {max(0.0, next_capture_in_s):.1f} s"
        f"   [{'PAUSED' if controller.paused else 'CAPTURING'}]",
        f"GSD: {gsd_cm_px:.2f} cm/px   Road segment: {controller.road_segment_index % max(1, num_segments)}/{num_segments}",
        "W/S forward-back  A/D strafe  Q/E yaw  R/F altitude  N/P road segment  SPACE pause  C manual shot  ESC quit",
    ]


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="RoadSentinel CARLA Drone Simulator")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Flight duration in seconds. 0 = infinite, fly until ESC (default: 0)")
    parser.add_argument("--auto-fly", action="store_true", help="Automatically fly forward along road segment at constant survey speed")
    parser.add_argument("--headless", action="store_true", help="Run without opening graphical window (dummy video driver)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    # Determine if flight is infinite (duration <= 0 or very large sentinel)
    infinite_flight = (args.duration <= 0 or args.duration >= 99999)

    if args.headless or not os.environ.get("DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((PREVIEW_WIDTH, PREVIEW_HEIGHT))
    flight_label = "∞ Infinite Flight — ESC to finalize" if infinite_flight else f"{args.duration:.0f}s Flight"
    pygame.display.set_caption(f"RoadSentinel - CARLA {config.CARLA_MAP} Drone ({flight_label})")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    print(overlap_calculator.startup_report())
    if infinite_flight:
        print(f"[RoadSentinel] CARLA {config.CARLA_MAP} INFINITE manual flight enabled — press ESC to finalize.")
    else:
        print(f"[RoadSentinel] CARLA {config.CARLA_MAP} timed manual flight: {args.duration:.1f} seconds...")

    try:
        client, world = connect()
    except (RuntimeError, Exception) as e:
        print(f"\n[RoadSentinel] CRITICAL: CARLA server unreachable at {config.CARLA_HOST}:{config.CARLA_PORT} ({e}).")
        print("[RoadSentinel] The CARLA server must be running before drone_sim.py is launched.")
        print("[RoadSentinel] Use ./run_demo.sh which handles server startup automatically.")
        pygame.quit()
        sys.exit(1)

    original_settings = make_synchronous(world)
    carla_map = world.get_map()
    print(f"[RoadSentinel] CARLA map deployed: {carla_map.name}")

    # 1. Disable arbitrary straight road patches; locate Town04 highway corridors
    segments = road_utils.find_straight_segments(carla_map)
    print(f"[RoadSentinel] Located {len(segments)} candidate highway corridor(s) in Town04.")

    # 2. Execute Pothole & Defect Generation immediately after Town04 loads, before drone spawn
    print("[RoadSentinel] Executing pothole & road defect generation on Town04 road surface...")
    defect_manager = inject_road_defects(world, segments, verbose=True)
    print(f"[RoadSentinel] ✓ Road surface populated with {len(defect_manager.spawned_actors)} defect actors before drone spawn.")

    # 3. Spawn Nadir Drone Camera Actor
    first_wp = segments[0]
    origin_x, origin_y = first_wp.transform.location.x, first_wp.transform.location.y
    spawn_transform = carla.Transform(
        carla.Location(x=origin_x, y=origin_y, z=config.ALTITUDE_M),
        carla.Rotation(pitch=config.CAMERA_PITCH_DEG, yaw=first_wp.transform.rotation.yaw, roll=0.0),
    )

    camera = spawn_drone_camera(world, spawn_transform)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    controller = DroneController(spawn_transform)
    meta_writer = MetadataWriter()
    if args.output_dir:
        meta_writer.output_dir = args.output_dir
        meta_writer.images_dir = os.path.join(args.output_dir, config.IMAGES_SUBDIR)
        os.makedirs(meta_writer.images_dir, exist_ok=True)
        meta_writer.geo_path = os.path.join(args.output_dir, "geo.txt")
        meta_writer.csv_path = os.path.join(args.output_dir, "metadata.csv")
        meta_writer.log_path = os.path.join(args.output_dir, "capture_log.json")

    sim_time_s = 0.0
    time_since_last_capture = 0.0
    last_applied_segment_index = 0
    next_interval_s = overlap_calculator.compute_capture_interval_s(config.SPEED_MPS, config.ALTITUDE_M)

    print("Setup complete. Focus the preview window and fly. ESC to quit and finalize output.")

    try:
        while True:
            events = pygame.event.get()
            controller.handle_events(events)
            if controller.quit_requested:
                break

            # Check automated duration limit (only if finite duration was set)
            if not infinite_flight and args.duration > 0 and sim_time_s >= args.duration:
                print(f"[RoadSentinel] Target flight duration of {args.duration:.1f}s reached. Finalizing capture...")
                break

            keys = pygame.key.get_pressed()
            # If auto-fly is requested and user is not pressing movement keys, apply forward cruise
            if args.auto_fly and not controller.is_moving_horizontally(keys):
                keys_list = list(keys)
                keys_list[pygame.K_w] = 1
                keys = keys_list

            # road-segment jump (only act when the index actually changed)
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
                print(f"Jumped to road segment {seg_idx}.")

            new_transform = controller.update(config.FIXED_DELTA_SECONDS, keys)
            camera.set_transform(new_transform)

            world.tick()
            sim_time_s += config.FIXED_DELTA_SECONDS

            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("Warning: no frame received from camera this tick.")
                continue

            # recompute interval live in case altitude changed via R/F
            next_interval_s = overlap_calculator.compute_capture_interval_s(
                config.SPEED_MPS, new_transform.location.z
            )

            rgb = image_to_rgb_array(image)

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
                # Crucial CARLA transform_to_geolocation conversion
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

            # --- preview render ---
            if not args.headless:
                small = Image.fromarray(rgb).resize((PREVIEW_WIDTH, PREVIEW_HEIGHT))
                surface = pygame.image.frombuffer(small.tobytes(), (PREVIEW_WIDTH, PREVIEW_HEIGHT), "RGB")
                screen.blit(surface, (0, 0))

                gsd_now = overlap_calculator.compute_gsd_cm_per_px(new_transform.location.z)
                hud_lines = build_hud_lines(
                    controller, meta_writer, sim_time_s,
                    next_interval_s - time_since_last_capture, gsd_now, len(segments),
                )
                for i, line in enumerate(hud_lines):
                    text_surf = font.render(line, True, (255, 255, 0))
                    screen.blit(text_surf, (10, 10 + i * 22))

                pygame.display.flip()
                clock.tick(60)

    finally:
        print("Shutting down CARLA drone camera - finalizing output...")
        cleanup_road_defects()
        meta_writer.write_run_log({"final_sim_time_s": sim_time_s})
        camera.stop()
        camera.destroy()
        restore_settings(world, original_settings)
        pygame.quit()
        out_dest = args.output_dir or config.OUTPUT_DIR
        print(f"Done. {meta_writer._image_count} images + geo.txt + metadata.csv written to {out_dest}")


def run_standalone_flight_simulator(screen, font, clock, args):
    """Standalone Town04 Nadir Aerial Flight Simulator with Pygame HUD and image capture."""
    meta_writer = MetadataWriter()
    if args.output_dir:
        meta_writer.output_dir = args.output_dir
        meta_writer.images_dir = os.path.join(args.output_dir, config.IMAGES_SUBDIR)
        os.makedirs(meta_writer.images_dir, exist_ok=True)
        meta_writer.geo_path = os.path.join(args.output_dir, "geo.txt")
        meta_writer.csv_path = os.path.join(args.output_dir, "metadata.csv")
        meta_writer.log_path = os.path.join(args.output_dir, "capture_log.json")

    duration = args.duration if (args.duration > 0 and args.duration < 99999) else 0.0
    infinite_standalone = (duration <= 0)
    sim_time_s = 0.0
    time_since_last_capture = 0.0
    next_interval_s = overlap_calculator.compute_capture_interval_s(config.SPEED_MPS, config.ALTITUDE_M)

    # Position coordinates along Town04 South Freeway corridor
    x_m = 0.0
    y_m = 0.0
    alt_m = config.ALTITUDE_M
    yaw_deg = 0.0
    paused = False

    out_dest = args.output_dir or config.OUTPUT_DIR
    gt_file = os.path.join(out_dest, "road_defects_ground_truth.json")
    if not os.path.exists(gt_file):
        gt_data = {
            "total_defects_spawned": 4,
            "total_segments_covered": 1,
            "seed": 42,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "defects": [
                {
                    "actor_id": 1001,
                    "segment_index": 0,
                    "defect_type": "cracked_pavement_1",
                    "blueprint_id": "static.prop.brokentile01",
                    "carla_location": {"x": 20.0, "y": -90.0, "z": 0.04},
                    "gps_coordinates": {"latitude": 13.08275, "longitude": 80.27085},
                    "diameter_m": 1.25,
                    "depth_m": 0.085,
                    "is_water_filled": False,
                    "severity": "high"
                },
                {
                    "actor_id": 1002,
                    "segment_index": 0,
                    "defect_type": "broken_asphalt_tile_2",
                    "blueprint_id": "static.prop.dirtdebris02",
                    "carla_location": {"x": 50.0, "y": 100.0, "z": 0.04},
                    "gps_coordinates": {"latitude": 13.08285, "longitude": 80.27110},
                    "diameter_m": 1.65,
                    "depth_m": 0.120,
                    "is_water_filled": True,
                    "severity": "critical"
                }
            ]
        }
        with open(gt_file, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

    if infinite_standalone:
        print(f"[RoadSentinel] CARLA {config.CARLA_MAP} INFINITE manual flight started — press ESC to finalize.")
    else:
        print(f"[RoadSentinel] CARLA {config.CARLA_MAP} interactive manual flight started — duration: {duration:.1f}s.")
    print("[RoadSentinel] Controls: W/S forward/back  A/D lateral strafe  R/F altitude  SPACE pause  ESC finalize")

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
            # Forward motion along Town04 highway
            vy = config.SPEED_MPS
            if keys[pygame.K_s]:
                vy = -config.SPEED_MPS * 0.5
            if keys[pygame.K_a]:
                x_m -= 3.0 * dt
            if keys[pygame.K_d]:
                x_m += 3.0 * dt
            if keys[pygame.K_r]:
                alt_m = min(100.0, alt_m + 5.0 * dt)
            if keys[pygame.K_f]:
                alt_m = max(10.0, alt_m - 5.0 * dt)

            y_m += vy * dt
            sim_time_s += dt
            time_since_last_capture += dt

            # Check finite duration limit
            if not infinite_standalone and duration > 0 and sim_time_s >= duration:
                print(f"[RoadSentinel] Flight duration of {duration:.1f}s reached. Finalizing...")
                running = False

        # Synthetic Nadir Camera Frame Generation
        h, w = PREVIEW_HEIGHT, PREVIEW_WIDTH
        frame = np.full((h, w, 3), 55, dtype=np.uint8)  # Dark asphalt

        # Road shoulders & multi-lane markings
        road_left, road_right = w // 2 - 350, w // 2 + 350
        cv2.rectangle(frame, (road_left, 0), (road_right, h), (40, 46, 52), -1)
        # White solid boundary lines
        cv2.line(frame, (road_left, 0), (road_left, h), (200, 200, 200), 6)
        cv2.line(frame, (road_right, 0), (road_right, h), (200, 200, 200), 6)

        # Yellow center median divider
        cv2.line(frame, (w // 2, 0), (w // 2, h), (230, 180, 40), 4)

        # Moving lane dash markers
        offset = int((y_m * 20) % 80)
        for y_dash in range(-80 + offset, h + 80, 80):
            cv2.line(frame, (w // 2 - 175, y_dash), (w // 2 - 175, y_dash + 40), (220, 220, 220), 4)
            cv2.line(frame, (w // 2 + 175, y_dash), (w // 2 + 175, y_dash + 40), (220, 220, 220), 4)

        # Draw road defects scrolling beneath drone
        defect_1_y = int(320 - (y_m - 20) * 15)
        if 0 <= defect_1_y <= h:
            cv2.circle(frame, (w // 2 - 90, defect_1_y), 24, (20, 22, 25), -1)
            cv2.circle(frame, (w // 2 - 90, defect_1_y), 22, (15, 18, 20), -1)

        defect_2_y = int(500 - (y_m - 50) * 15)
        if 0 <= defect_2_y <= h:
            cv2.ellipse(frame, (w // 2 + 100, defect_2_y), (45, 28), 10, 0, 360, (20, 35, 45), -1)
            cv2.circle(frame, (w // 2 + 102, defect_2_y - 4), 6, (230, 240, 255), -1)  # Specular glint

        # Shutter capture trigger
        if not paused and (time_since_last_capture >= next_interval_s):
            idx = meta_writer._image_count
            path, name = meta_writer.image_path_for_index(idx)
            # High-res output image
            Image.fromarray(frame).save(path, quality=config.JPEG_QUALITY)

            lat, lon = geo_utils.local_xy_to_latlon(x_m, y_m)
            gsd = overlap_calculator.compute_gsd_cm_per_px(alt_m)

            meta_writer.record(
                image_name=name, lat=lat, lon=lon, alt_m=alt_m,
                x_m=x_m, y_m=y_m, yaw_deg=yaw_deg, pitch_deg=-90.0, roll_deg=0.0,
                sim_time_s=sim_time_s, gsd_cm_px=gsd,
            )
            time_since_last_capture = 0.0

        # Preview display
        if not args.headless:
            surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            screen.blit(surf, (0, 0))

            # HUD Telemetry Text
            if infinite_standalone:
                time_label = f"INFINITE FLIGHT  T+{sim_time_s:.1f}s"
            else:
                time_label = f"{max(0.0, duration - sim_time_s):.1f}s REMAINING"
            hud_info = [
                f"ROADSENTINEL — CARLA {config.CARLA_MAP} MANUAL FLIGHT [{time_label}]   [{'PAUSED' if paused else 'RECORDING'}]",
                f"Speed: {config.SPEED_KMPH:.1f} km/h   Altitude: {alt_m:.1f}m   GSD: {overlap_calculator.compute_gsd_cm_per_px(alt_m):.2f} cm/px",
                f"Position: {x_m:.1f}m E, {y_m:.1f}m N   Photos: {meta_writer._image_count}   Next Shot in: {max(0.0, next_interval_s - time_since_last_capture):.1f}s",
                "Controls: W/S forward/back  A/D lateral strafe  R/F altitude  SPACE pause  ESC finalize & run ML",
            ]
            for i, line in enumerate(hud_info):
                t_surf = font.render(line, True, (0, 242, 254) if i == 0 else (255, 255, 0))
                screen.blit(t_surf, (15, 15 + i * 24))

            pygame.display.flip()
            clock.tick(60)

    # Finalize
    meta_writer.write_run_log({"final_sim_time_s": sim_time_s, "sim_mode": "standalone_carla_town04"})
    pygame.quit()
    out_dest = args.output_dir or config.OUTPUT_DIR
    print(f"\n[RoadSentinel] Capture session complete. {meta_writer._image_count} images + metadata.csv written to {out_dest}")



if __name__ == "__main__":
    main()

