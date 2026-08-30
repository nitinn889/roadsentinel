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

import queue
import sys

import numpy as np
import pygame
from PIL import Image

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
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((PREVIEW_WIDTH, PREVIEW_HEIGHT))
    pygame.display.set_caption("RoadSentinel - Drone Preview (this is a live preview only; saved images are full-res)")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    print(overlap_calculator.startup_report())

    try:
        client, world = connect()
    except RuntimeError as e:
        print(f"Could not connect to CARLA at {config.CARLA_HOST}:{config.CARLA_PORT} - "
              f"is the simulator running? ({e})")
        sys.exit(1)

    original_settings = make_synchronous(world)
    carla_map = world.get_map()

    segments = road_utils.find_straight_segments(carla_map)
    print(f"Found {len(segments)} candidate straight road segment(s). Use N/P to cycle.")

    # Spawn road defects along the candidate flight corridors
    inject_road_defects(world, segments, verbose=True)

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

            keys = pygame.key.get_pressed()

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

                x_m = new_transform.location.x - origin_x
                y_m = new_transform.location.y - origin_y
                lat, lon = geo_utils.local_xy_to_latlon(x_m, y_m)
                gsd = overlap_calculator.compute_gsd_cm_per_px(new_transform.location.z)

                meta_writer.record(
                    image_name=name, lat=lat, lon=lon, alt_m=new_transform.location.z,
                    x_m=x_m, y_m=y_m,
                    yaw_deg=new_transform.rotation.yaw, pitch_deg=new_transform.rotation.pitch,
                    roll_deg=new_transform.rotation.roll,
                    sim_time_s=sim_time_s, gsd_cm_px=gsd,
                )
                if idx % 10 == 0:
                    meta_writer.flush()
                time_since_last_capture = 0.0

            # --- preview render ---
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
        print("Shutting down - finalizing output...")
        cleanup_road_defects()
        meta_writer.write_run_log({"final_sim_time_s": sim_time_s})
        camera.stop()
        camera.destroy()
        restore_settings(world, original_settings)
        pygame.quit()
        print(f"Done. {meta_writer._image_count} images + geo.txt + metadata.csv written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
