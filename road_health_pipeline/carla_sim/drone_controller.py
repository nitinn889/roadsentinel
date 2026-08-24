from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import carla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CONFIG
from common.geometry import actual_forward_overlap, footprint_m, target_capture_speed_mps
from common.io_utils import utc_iso, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("carla_drone")


def speed_mps(vehicle: carla.Actor) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--altitude", type=float, default=CONFIG.altitude_m)
    ap.add_argument("--speed", type=float, default=CONFIG.max_speed_mps)
    ap.add_argument("--interval", type=float, default=CONFIG.capture_interval_s)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "carla_run")
    args = ap.parse_args()

    overlap = actual_forward_overlap(args.speed, args.interval, args.altitude, CONFIG.horizontal_fov_deg,
                                      CONFIG.image_width, CONFIG.image_height)
    target_speed = target_capture_speed_mps(args.altitude, CONFIG.horizontal_fov_deg, CONFIG.image_width,
                                             CONFIG.image_height, args.interval, CONFIG.target_overlap)
    fw, fh = footprint_m(CONFIG.image_width, CONFIG.image_height, args.altitude, CONFIG.horizontal_fov_deg)
    log.info("Ground footprint %.2fm x %.2fm", fw, fh)
    log.info("At speed %.2fm/s and %.2fs interval, forward overlap = %.1f%%", args.speed, args.interval, 100 * overlap)
    log.info("Approximate speed for %.0f%% overlap = %.2fm/s (%.1fkm/h)", 100*CONFIG.target_overlap, target_speed, target_speed*3.6)
    if abs(overlap - CONFIG.target_overlap) > 0.10:
        log.warning("Configured flight parameters do not achieve the target overlap within 10 percentage points")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.*")[0]
    spawn = carla.Transform(carla.Location(x=-103.2, y=-14.4, z=args.altitude), carla.Rotation(yaw=-89.4, pitch=0, roll=0))
    vehicle = world.try_spawn_actor(vehicle_bp, spawn)
    if vehicle is None:
        raise RuntimeError("Could not spawn drone proxy vehicle")

    camera = None
    try:
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CONFIG.image_width))
        cam_bp.set_attribute("image_size_y", str(CONFIG.image_height))
        cam_bp.set_attribute("fov", str(CONFIG.horizontal_fov_deg))
        cam_bp.set_attribute("sensor_tick", "0.1")
        camera_transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=-90, yaw=0, roll=0))
        camera = world.spawn_actor(cam_bp, camera_transform, attach_to=vehicle)

        args.out.mkdir(parents=True, exist_ok=True)
        state = {"frame_id": 0, "last_capture": 0.0}

        def save_rgb(image: carla.Image):
            now = time.monotonic()
            if now - state["last_capture"] < args.interval:
                return
            state["last_capture"] = now
            frame_id = state["frame_id"]
            state["frame_id"] += 1
            image_path = args.out / "rgb" / f"frame_{frame_id:08d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save_to_disk(str(image_path))
            tr = vehicle.get_transform()
            v = speed_mps(vehicle)
            metadata = {
                "timestamp": utc_iso(),
                "frame_id": frame_id,
                "latitude": None,
                "longitude": None,
                "altitude_m": float(tr.location.z),
                "heading_deg": float(tr.rotation.yaw),
                "world_x": float(tr.location.x),
                "world_y": float(tr.location.y),
                "world_z": float(tr.location.z),
                "speed_mps": v,
                "camera_mode": "nadir",
                "ground_footprint_width_m": fw,
                "ground_footprint_length_m": fh,
            }
            save_json(metadata, image_path.with_suffix(".json"))
            log.info("Captured %s speed=%.2f m/s", image_path, v)

        camera.listen(save_rgb)
        log.info("Controls: W forward, S reverse, D stop, A/E rotate, ESC exit")
        while True:
            key = input("Control [w/s/d/a/e/q]: ").strip().lower()[:1]
            if key == "q":
                break
            control = carla.VehicleControl()
            if key == "w":
                control.throttle = 0.6
            elif key == "s":
                control.brake = 0.3
                control.throttle = 0.0
            elif key == "d":
                control.brake = 1.0
            elif key == "a":
                current = vehicle.get_transform().rotation.yaw
                vehicle.set_transform(carla.Transform(vehicle.get_transform().location,
                                                       carla.Rotation(yaw=current - 10, pitch=-0.0, roll=0)))
            elif key == "e":
                current = vehicle.get_transform().rotation.yaw
                vehicle.set_transform(carla.Transform(vehicle.get_transform().location,
                                                       carla.Rotation(yaw=current + 10, pitch=-0.0, roll=0)))
            vehicle.apply_control(control)
            time.sleep(0.1)
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        vehicle.destroy()


if __name__ == "__main__":
    main()
