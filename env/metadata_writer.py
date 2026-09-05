"""
metadata_writer.py
-------------------
Produces the hand-off artifacts for the next pipeline stage
(DINOv2/SAM2 anomaly detection + OpenDroneMap/COLMAP photogrammetry):

  output/
    images/                  sequential JPGs
    geo.txt                  OpenDroneMap geolocation file (lat, lon, alt)
    metadata.csv             full 6-DOF pose + timestamp per image
    capture_log.json         run summary (written at shutdown)

geo.txt follows ODM's simple geolocation-file format:
    EPSG:4326
    <image_name> <lon> <lat> <alt>
(ODM accepts lon/lat/alt in that column order for EPSG:4326.)
"""

import csv
import json
import os
import time
import config


class MetadataWriter:
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR
        self.images_dir = os.path.join(self.output_dir, config.IMAGES_SUBDIR)
        os.makedirs(self.images_dir, exist_ok=True)

        self.geo_path = os.path.join(self.output_dir, "geo.txt")
        self.csv_path = os.path.join(self.output_dir, "metadata.csv")
        self.trajectory_path = os.path.join(self.output_dir, "drone_trajectory.csv")
        self.log_path = os.path.join(self.output_dir, "capture_log.json")

        self._geo_lines = ["EPSG:4326"]
        self._csv_rows = []
        self._trajectory_rows = []
        self._run_start_wall = time.time()
        self._image_count = 0

    def image_path_for_index(self, index: int) -> str:
        name = f"road_{index:05d}.{config.IMAGE_FORMAT}"
        return os.path.join(self.images_dir, name), name

    def record(self, image_name: str, lat: float, lon: float, alt_m: float,
               x_m: float, y_m: float, yaw_deg: float, pitch_deg: float,
               roll_deg: float, sim_time_s: float, gsd_cm_px: float,
               z_m: float = None):
        self._geo_lines.append(f"{image_name} {lon:.8f} {lat:.8f} {alt_m:.2f}")
        self._csv_rows.append({
            "image_name": image_name,
            "sim_time_s": round(sim_time_s, 3),
            "local_x_m": round(x_m, 3),
            "local_y_m": round(y_m, 3),
            "altitude_m": round(alt_m, 2),
            "latitude": round(lat, 8),
            "longitude": round(lon, 8),
            "yaw_deg": round(yaw_deg, 2),
            "pitch_deg": round(pitch_deg, 2),
            "roll_deg": round(roll_deg, 2),
            "gsd_cm_per_px": round(gsd_cm_px, 3),
        })

        z_val = z_m if z_m is not None else alt_m
        self._trajectory_rows.append({
            "frame": self._image_count,
            "x": round(x_m, 4),
            "y": round(y_m, 4),
            "z": round(z_val, 4),
            "pitch": round(pitch_deg, 4),
            "yaw": round(yaw_deg, 4),
            "roll": round(roll_deg, 4),
        })
        self._image_count += 1

    def flush(self):
        """Write geo.txt, metadata.csv, and drone_trajectory.csv to disk."""
        with open(self.geo_path, "w") as f:
            f.write("\n".join(self._geo_lines) + "\n")

        if self._csv_rows:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._csv_rows)

        if self._trajectory_rows:
            with open(self.trajectory_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["frame", "x", "y", "z", "pitch", "yaw", "roll"])
                writer.writeheader()
                writer.writerows(self._trajectory_rows)
            try:
                root_traj = os.path.join(os.getcwd(), "drone_trajectory.csv")
                with open(root_traj, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["frame", "x", "y", "z", "pitch", "yaw", "roll"])
                    writer.writeheader()
                    writer.writerows(self._trajectory_rows)
            except Exception:
                pass

    def write_run_log(self, extra: dict):
        self.flush()
        summary = {
            "total_images": self._image_count,
            "wall_clock_duration_s": round(time.time() - self._run_start_wall, 2),
            "config_snapshot": {
                "speed_kmph": config.SPEED_KMPH,
                "altitude_m": config.ALTITUDE_M,
                "camera_fov_deg": config.CAMERA_FOV_DEG,
                "image_width": config.IMAGE_WIDTH,
                "image_height": config.IMAGE_HEIGHT,
                "forward_overlap": config.FORWARD_OVERLAP,
                "carla_map": config.CARLA_MAP,
            },
        }
        summary.update(extra)
        with open(self.log_path, "w") as f:
            json.dump(summary, f, indent=2)
