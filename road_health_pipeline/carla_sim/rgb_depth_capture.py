from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import threading

import carla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CONFIG
from common.io_utils import save_json, utc_iso


class SynchronizedCapture:
    def __init__(self, out_dir: Path, far_clip_m: float = 1000.0):
        self.out_dir = Path(out_dir)
        self.q_rgb = queue.Queue()
        self.q_depth = queue.Queue()
        self.running = True
        self.far_clip_m = far_clip_m

    def start(self, host="127.0.0.1", port=2000):
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        bp = world.get_blueprint_library()
        vehicle_bp = bp.filter("vehicle.*")[0]
        vehicle = world.try_spawn_actor(vehicle_bp, carla.Transform(carla.Location(x=-103.2, y=-14.4, z=CONFIG.altitude_m), carla.Rotation(yaw=-89.4)))
        if vehicle is None:
            raise RuntimeError("Vehicle spawn failed")
        rgb = depth = None
        try:
            rgb_bp = bp.find("sensor.camera.rgb")
            depth_bp = bp.find("sensor.camera.depth")
            for camera_bp in (rgb_bp, depth_bp):
                camera_bp.set_attribute("image_size_x", str(CONFIG.image_width))
                camera_bp.set_attribute("image_size_y", str(CONFIG.image_height))
                camera_bp.set_attribute("fov", str(CONFIG.horizontal_fov_deg))
            tf = carla.Transform(carla.Location(), carla.Rotation(pitch=-90))
            rgb = world.spawn_actor(rgb_bp, tf, attach_to=vehicle)
            depth = world.spawn_actor(depth_bp, tf, attach_to=vehicle)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.out_dir.joinpath("rgb").mkdir(exist_ok=True)
            self.out_dir.joinpath("depth_gt").mkdir(exist_ok=True)
            lock = threading.Lock()
            pending = {}

            def on_rgb(image):
                with lock:
                    pending.setdefault(image.frame, {})["rgb"] = image
                    self._try_save(image.frame, pending)

            def on_depth(image):
                with lock:
                    pending.setdefault(image.frame, {})["depth"] = image
                    self._try_save(image.frame, pending)

            rgb.listen(on_rgb)
            depth.listen(on_depth)
            print("Synchronized RGB/depth capture running. Press Ctrl-C to stop.")
            while True:
                world.wait_for_tick()
        finally:
            if rgb:
                rgb.stop(); rgb.destroy()
            if depth:
                depth.stop(); depth.destroy()
            vehicle.destroy()

    def _try_save(self, frame, pending):
        item = pending.get(frame, {})
        if "rgb" not in item or "depth" not in item:
            return
        rgb = item["rgb"]
        depth = item["depth"]
        rgb_path = self.out_dir / "rgb" / f"frame_{frame:08d}.png"
        depth_path = self.out_dir / "depth_gt" / f"frame_{frame:08d}.png"
        rgb.save_to_disk(str(rgb_path))
        depth.save_to_disk(str(depth_path), carla.ColorConverter.Raw)
        tf = depth.transform
        meta = {
            "timestamp": utc_iso(),
            "frame_id": int(frame),
            "altitude_m": float(tf.location.z),
            "heading_deg": float(tf.rotation.yaw),
            "world_x": float(tf.location.x),
            "world_y": float(tf.location.y),
            "world_z": float(tf.location.z),
            "camera_mode": "nadir",
            "depth_far_clip_m": self.far_clip_m,
        }
        save_json(meta, rgb_path.with_suffix(".json"))
        del pending[frame]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "carla_sync")
    args = ap.parse_args()
    SynchronizedCapture(args.out).start(args.host, args.port)


if __name__ == "__main__":
    main()
