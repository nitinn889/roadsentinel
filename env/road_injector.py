"""
road_injector.py
================
RoadSentinel – CARLA Road Defect Injector.

Spawns physical road defects (pothole damage zones, broken asphalt tiles,
dirt/debris clusters, and road wear hazards) along the road flight corridors
in CARLA Town04, and records machine-readable ground-truth metadata (GPS,
local coordinates, defect dimensions, severity, and status).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import geo_utils
import road_utils


# ---------------------------------------------------------------------------
# Defect Blueprints Available in CARLA 0.9.16
# ---------------------------------------------------------------------------
DEFECT_BLUEPRINTS = {
    "broken_asphalt_tile_1": "static.prop.dirtdebris01",
    "broken_asphalt_tile_2": "static.prop.dirtdebris02",
    "broken_asphalt_tile_3": "static.prop.dirtdebris03",
    "cracked_pavement_1":     "static.prop.brokentile01",
    "cracked_pavement_2":     "static.prop.brokentile02",
    "surface_breakup_1":      "static.prop.brokentile03",
    "surface_breakup_2":      "static.prop.brokentile04",
}

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]


class RoadDefectManager:
    """Manages spawning, tracking, and clean destruction of road defects in CARLA."""

    def __init__(self, world):
        self.world = world
        self.bp_lib = world.get_blueprint_library()
        self.spawned_actors: List = []
        self.ground_truth_records: List[Dict] = []

    def spawn_defects_along_corridors(
        self,
        segments: List,
        defects_per_segment: int = 12,
        seed: int = 42,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Spawns road defects along the specified road segments.

        Parameters
        ----------
        segments : List[carla.Waypoint]
            Start waypoints for straight road segments.
        defects_per_segment : int
            Average number of defect clusters to spawn per segment.
        seed : int
            Random seed for reproducible placement.
        output_dir : Path
            Where to save ground-truth JSON.
        verbose : bool
            Whether to print progress.
        """
        rng = np.random.default_rng(seed)
        t0 = time.time()

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent / "output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bp_keys = list(DEFECT_BLUEPRINTS.keys())
        total_spawned = 0

        if verbose:
            print(f"[RoadInjector] Spawning defects across {len(segments)} road corridor(s) ...")

        for seg_idx, start_wp in enumerate(segments):
            # Trace waypoints forward along this segment (up to 150m)
            wps = [start_wp]
            curr = start_wp
            for _ in range(30):  # 30 * 5m = 150m
                nxts = curr.next(5.0)
                if not nxts:
                    break
                curr = nxts[0]
                wps.append(curr)

            if len(wps) < 4:
                continue

            # Select spaced waypoint locations along the corridor
            step_spacing = max(2, len(wps) // defects_per_segment)
            chosen_wps = wps[1::step_spacing]

            for wp in chosen_wps:
                # Decide cluster size (1 to 3 defect items per spot)
                cluster_size = rng.integers(1, 3)
                
                # Wheel-path bias (-1.4m left wheel track, +1.4m right wheel track)
                base_lat = float(rng.choice([-1.4, 1.4]))
                yaw_rad = math.radians(wp.transform.rotation.yaw)

                for c_i in range(cluster_size):
                    defect_key = str(rng.choice(bp_keys))
                    bp_id = DEFECT_BLUEPRINTS[defect_key]
                    bp = self.bp_lib.find(bp_id)

                    # Lateral & longitudinal jitter
                    lat_jitter = base_lat + float(rng.normal(0, 0.35))
                    along_jitter = float(rng.uniform(-1.8, 1.8))

                    # Calculate CARLA 3D coordinates
                    dx_lat = -math.sin(yaw_rad) * lat_jitter
                    dy_lat =  math.cos(yaw_rad) * lat_jitter
                    dx_along = math.cos(yaw_rad) * along_jitter
                    dy_along = math.sin(yaw_rad) * along_jitter

                    loc_x = float(wp.transform.location.x + dx_lat + dx_along)
                    loc_y = float(wp.transform.location.y + dy_lat + dy_along)
                    loc_z = float(wp.transform.location.z + 0.04)  # slightly above asphalt

                    rot_yaw = float(rng.uniform(0, 360))

                    spawn_transform = self.world.get_map().get_waypoint(wp.transform.location).transform
                    spawn_transform.location.x = loc_x
                    spawn_transform.location.y = loc_y
                    spawn_transform.location.z = loc_z
                    spawn_transform.rotation.yaw = rot_yaw

                    actor = self.world.try_spawn_actor(bp, spawn_transform)
                    if actor is not None:
                        self.spawned_actors.append(actor)
                        total_spawned += 1

                        # Physical defect metadata
                        is_water = bool(rng.random() < 0.30)
                        diameter_m = round(float(rng.uniform(0.60, 1.80)), 2)
                        depth_m = round(float(rng.uniform(0.04, 0.15)), 3)
                        severity = str(rng.choice(SEVERITY_LEVELS))

                        # Global GPS coordinates
                        origin_x = segments[0].transform.location.x
                        origin_y = segments[0].transform.location.y
                        lat, lon = geo_utils.local_xy_to_latlon(loc_x - origin_x, loc_y - origin_y)

                        self.ground_truth_records.append({
                            "actor_id": int(actor.id),
                            "segment_index": int(seg_idx),
                            "defect_type": defect_key,
                            "blueprint_id": bp_id,
                            "carla_location": {
                                "x": round(loc_x, 3),
                                "y": round(loc_y, 3),
                                "z": round(loc_z, 3),
                            },
                            "gps_coordinates": {
                                "latitude": round(lat, 8),
                                "longitude": round(lon, 8),
                            },
                            "diameter_m": diameter_m,
                            "depth_m": depth_m,
                            "is_water_filled": is_water,
                            "severity": severity,
                        })

        # Save ground truth file
        gt_file = output_dir / "road_defects_ground_truth.json"
        gt_data = {
            "total_defects_spawned": len(self.spawned_actors),
            "total_segments_covered": len(segments),
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "defects": self.ground_truth_records,
        }
        with open(gt_file, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

        elapsed = time.time() - t0
        if verbose:
            print(f"[RoadInjector] ✓ Successfully placed {len(self.spawned_actors)} road defects in {elapsed:.2f}s!")
            print(f"[RoadInjector] Ground truth saved to: {gt_file}")

        return self.ground_truth_records

    def cleanup(self):
        """Destroy all spawned defect actors."""
        count = len(self.spawned_actors)
        for actor in self.spawned_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.spawned_actors.clear()
        print(f"[RoadInjector] Cleaned up {count} defect actors.")


# Global singleton instance
_GLOBAL_MANAGER: Optional[RoadDefectManager] = None


def inject_road_defects(world, segments=None, verbose: bool = True) -> RoadDefectManager:
    """Convenience function to inject road defects into CARLA world."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        _GLOBAL_MANAGER.cleanup()

    if segments is None:
        carla_map = world.get_map()
        segments = road_utils.find_straight_segments(carla_map)

    _GLOBAL_MANAGER = RoadDefectManager(world)
    _GLOBAL_MANAGER.spawn_defects_along_corridors(segments, verbose=verbose)
    return _GLOBAL_MANAGER


def cleanup_road_defects():
    """Cleans up all spawned defects."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        _GLOBAL_MANAGER.cleanup()
        _GLOBAL_MANAGER = None
