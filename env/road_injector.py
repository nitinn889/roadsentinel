"""
road_injector.py
================
RoadSentinel – Procedural CARLA Road Defect & Degradation Generator.

Generates physically diverse, reproducible road defects along CARLA road corridors
(Town04 / Town06 / Town07) with realistic variations in:
  - Dimensions: length, width, depth, area, aspect ratio, orientation
  - Shape categories: circular, elongated (longitudinal/transverse), irregular/jagged, compound clusters
  - Water pooling states: dry vs water-filled (with variable turbidity, depth, wet halo)
  - Road-health scenarios: Healthy, Moderate, Poor, Critical
  - Associated road degradation: repaired asphalt patches, cracks, surface breakups, debris
  - Structured ground truth metadata (GPS, local Cartesian, physical metrics, severity)
"""

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Securely load env-local modules regardless of sys.path collision
_ENV_DIR = Path(__file__).resolve().parent

def _load_local_module(name: str, filename: str):
    fpath = _ENV_DIR / filename
    spec = importlib.util.spec_from_file_location(name, fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

config = _load_local_module("env_config", "config.py")
geo_utils = _load_local_module("env_geo_utils", "geo_utils.py")
road_utils = _load_local_module("env_road_utils", "road_utils.py")

try:
    import carla
    _HAS_CARLA = True
except ImportError:
    _HAS_CARLA = False


# ---------------------------------------------------------------------------
# CARLA Blueprint Registry for Physical Props (Excludes ugly square tiles)
# ---------------------------------------------------------------------------
DEBRIS_BLUEPRINTS = [
    "static.prop.dirtdebris01",
    "static.prop.dirtdebris02",
    "static.prop.dirtdebris03",
]

CONSTRUCTION_BLUEPRINTS = [
    "static.prop.warningconstruction",
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
]


class ProceduralDefectSpec:
    """Detailed physical specification of a single procedurally generated defect."""

    def __init__(
        self,
        defect_id: str,
        defect_type: str,
        shape_category: str,
        length_m: float,
        width_m: float,
        depth_m: float,
        aspect_ratio: float,
        orientation_deg: float,
        irregularity: float,
        edge_breakup: float,
        roughness: float,
        is_water_filled: bool,
        water_depth_m: float = 0.0,
        water_coverage_frac: float = 0.0,
        turbidity: float = 0.0,
        wet_halo_radius_m: float = 0.0,
        has_cracks: bool = False,
        crack_pattern: str = "none",
        has_road_patch: bool = False,
        lane_position: str = "lane_center",
        cluster_id: Optional[str] = None,
        is_overlapping: bool = False,
    ):
        self.defect_id = defect_id
        self.defect_type = defect_type
        self.shape_category = shape_category
        self.length_m = max(0.15, float(length_m))
        self.width_m = max(0.15, float(width_m))
        self.depth_m = max(0.01, float(depth_m))
        self.aspect_ratio = max(0.5, float(aspect_ratio))
        self.orientation_deg = float(orientation_deg % 360.0)
        self.irregularity = float(np.clip(irregularity, 0.0, 1.0))
        self.edge_breakup = float(np.clip(edge_breakup, 0.0, 1.0))
        self.roughness = float(np.clip(roughness, 0.0, 1.0))
        
        # Equivalent diameter and metric surface area proxy
        self.diameter_m = math.sqrt(self.length_m * self.width_m)
        # Approximate ellipse area with irregularity discount
        self.area_m2 = round((math.pi / 4.0) * self.length_m * self.width_m * (1.0 - self.irregularity * 0.12), 3)

        self.is_water_filled = bool(is_water_filled)
        self.water_depth_m = round(float(water_depth_m), 3) if self.is_water_filled else 0.0
        self.water_coverage_frac = round(float(water_coverage_frac), 2) if self.is_water_filled else 0.0
        self.turbidity = round(float(turbidity), 2) if self.is_water_filled else 0.0
        self.wet_halo_radius_m = round(float(wet_halo_radius_m), 2) if self.is_water_filled else 0.0

        self.has_cracks = bool(has_cracks)
        self.crack_pattern = str(crack_pattern)
        self.has_road_patch = bool(has_road_patch)
        self.lane_position = str(lane_position)
        self.cluster_id = cluster_id
        self.is_overlapping = bool(is_overlapping)

        # Compute continuous severity index and categorical tier
        area_comp = min(1.0, self.area_m2 / 1.5)
        depth_comp = min(1.0, self.depth_m / 0.15)
        water_comp = 0.85 if self.is_water_filled else 0.10
        breakup_comp = min(1.0, self.edge_breakup * 0.6 + self.roughness * 0.4)
        
        self.severity_score = round(
            0.35 * area_comp + 0.30 * depth_comp + 0.20 * water_comp + 0.15 * breakup_comp, 3
        )
        if self.severity_score >= 0.85 or (self.depth_m >= 0.12 and self.is_water_filled):
            self.severity_category = "critical"
        elif self.severity_score >= 0.65:
            self.severity_category = "high"
        elif self.severity_score >= 0.35:
            self.severity_category = "medium"
        else:
            self.severity_category = "low"


class ProceduralRoadGenerator:
    """Procedural road defect generation engine for CARLA simulation."""

    def __init__(self, scenario: str = "moderate", seed: int = 42):
        self.scenario_name = scenario.lower()
        self.scenario_cfg = config.SCENARIOS.get(self.scenario_name, config.SCENARIOS["moderate"])
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def generate_corridor_plan(
        self,
        segment_length_m: float = 150.0,
        defects_count: Optional[int] = None,
        water_ratio_override: Optional[float] = None,
    ) -> List[Tuple[float, float, ProceduralDefectSpec]]:
        """Plan procedural defects along a straight corridor in local metric space (along_m, across_m)."""
        if defects_count is None:
            min_d, max_d = self.scenario_cfg.defects_per_corridor
            count = int(self.rng.integers(min_d, max_d + 1))
        else:
            count = max(0, int(defects_count))

        if count == 0:
            return []

        water_ratio = (
            float(water_ratio_override)
            if water_ratio_override is not None
            else self.scenario_cfg.water_filled_ratio
        )

        plan: List[Tuple[float, float, ProceduralDefectSpec]] = []
        defect_idx = 1

        # Determine how many defects are clustered vs isolated
        cluster_prob = self.scenario_cfg.cluster_probability
        min_size, max_size = self.scenario_cfg.pothole_size_range_m
        min_depth, max_depth = self.scenario_cfg.pothole_depth_range_m

        while defect_idx <= count:
            # 1. Longitudinal placement along corridor (leaving 8m margin at ends)
            along_m = float(self.rng.uniform(8.0, max(12.0, segment_length_m - 8.0)))

            # 2. Lateral placement (wheel-path biased)
            lane_roll = self.rng.random()
            if lane_roll < 0.42:
                # Left wheel track
                across_m = float(self.rng.normal(-1.40, 0.28))
                lane_pos = "left_wheel_track"
            elif lane_roll < 0.84:
                # Right wheel track
                across_m = float(self.rng.normal(1.40, 0.28))
                lane_pos = "right_wheel_track"
            elif lane_roll < 0.94:
                # Lane center
                across_m = float(self.rng.normal(0.0, 0.35))
                lane_pos = "lane_center"
            else:
                # Shoulder / boundary
                side = float(self.rng.choice([-1.0, 1.0]))
                across_m = side * float(self.rng.uniform(2.40, 3.10))
                lane_pos = "lane_boundary"

            # 3. Shape category & dimensions (6 distinct realistic families)
            shape_roll = self.rng.random()
            base_diam = float(self.rng.uniform(min_size, max_size))

            if shape_roll < 0.18:
                # 1. Elongated longitudinal (rutting / wheel-wear tearing along lane)
                aspect_ratio = float(self.rng.uniform(2.0, 3.8))
                length_m = base_diam * math.sqrt(aspect_ratio)
                width_m = base_diam / math.sqrt(aspect_ratio)
                shape_cat = "elongated_longitudinal"
                orientation_deg = float(self.rng.normal(0.0, 8.0))
            elif shape_roll < 0.36:
                # 2. Elongated transverse (joint / contraction fracture across lane)
                aspect_ratio = float(self.rng.uniform(2.0, 3.5))
                width_m = base_diam * math.sqrt(aspect_ratio)
                length_m = base_diam / math.sqrt(aspect_ratio)
                shape_cat = "elongated_transverse"
                orientation_deg = float(self.rng.normal(90.0, 8.0))
            elif shape_roll < 0.54:
                # 3. Irregular natural (multi-lobed organic fractal)
                aspect_ratio = float(self.rng.uniform(1.1, 1.7))
                length_m = base_diam * math.sqrt(aspect_ratio)
                width_m = base_diam / math.sqrt(aspect_ratio)
                shape_cat = "irregular_natural"
                orientation_deg = float(self.rng.uniform(0.0, 360.0))
            elif shape_roll < 0.72:
                # 4. Jagged (sharp angular crumbling edges)
                aspect_ratio = float(self.rng.uniform(1.05, 1.8))
                length_m = base_diam * math.sqrt(aspect_ratio)
                width_m = base_diam / math.sqrt(aspect_ratio)
                shape_cat = "jagged"
                orientation_deg = float(self.rng.uniform(0.0, 360.0))
            elif shape_roll < 0.86:
                # 5. Compound pothole cluster
                aspect_ratio = float(self.rng.uniform(1.1, 1.6))
                length_m = base_diam * 1.25
                width_m = base_diam * 1.15
                shape_cat = "compound_cluster"
                orientation_deg = float(self.rng.uniform(0.0, 360.0))
            else:
                # 6. Partially connected twin potholes
                aspect_ratio = float(self.rng.uniform(1.6, 2.6))
                length_m = base_diam * math.sqrt(aspect_ratio)
                width_m = base_diam / math.sqrt(aspect_ratio)
                shape_cat = "partially_connected"
                orientation_deg = float(self.rng.choice([0.0, 45.0, 90.0]) + self.rng.normal(0, 10.0))

            # Depth
            depth_m = float(self.rng.uniform(min_depth, max_depth))
            depth_m = float(np.clip(depth_m * (0.8 + 0.4 * (base_diam / max(0.1, max_size))), min_depth, max_depth))

            # Water state (variable coverage 0.25 to 1.0, turbidity clear->muddy, wet halo)
            is_water = bool(self.rng.random() < water_ratio)
            water_cov = float(np.clip(self.rng.uniform(0.25, 1.0), 0.25, 1.0)) if is_water else 0.0
            water_depth = depth_m * float(self.rng.uniform(0.35, 0.90)) * water_cov if is_water else 0.0
            turbidity = float(self.rng.uniform(0.05, 0.85)) if is_water else 0.0
            wet_halo = float(self.rng.uniform(0.18, 0.45)) if is_water else 0.0

            # Surface features
            has_cracks = bool(self.rng.random() < self.scenario_cfg.crack_density)
            patterns = ["radial", "longitudinal", "transverse", "alligator"]
            crack_pat = str(self.rng.choice(patterns)) if has_cracks else "none"
            has_patch = bool(self.rng.random() < self.scenario_cfg.patch_density)

            defect_id = f"pothole_{self.scenario_name}_{defect_idx:03d}"
            cluster_id = f"cluster_{defect_idx:03d}"

            spec = ProceduralDefectSpec(
                defect_id=defect_id,
                defect_type="water_filled_pothole" if is_water else "pothole",
                shape_category=shape_cat,
                length_m=length_m,
                width_m=width_m,
                depth_m=depth_m,
                aspect_ratio=aspect_ratio,
                orientation_deg=orientation_deg,
                irregularity=float(self.rng.uniform(0.35, 0.85)),
                edge_breakup=float(self.rng.uniform(0.30, 0.90)),
                roughness=float(self.rng.uniform(0.25, 0.80)),
                is_water_filled=is_water,
                water_depth_m=water_depth,
                water_coverage_frac=water_cov,
                turbidity=turbidity,
                wet_halo_radius_m=wet_halo,
                has_cracks=has_cracks,
                crack_pattern=crack_pat,
                has_road_patch=has_patch,
                lane_position=lane_pos,
                cluster_id=cluster_id,
                is_overlapping=False,
            )
            plan.append((along_m, across_m, spec))
            defect_idx += 1

            # Check if this defect spawns a cluster or overlapping sibling
            if defect_idx <= count and self.rng.random() < cluster_prob:
                is_overlap = bool(self.rng.random() < 0.45)
                # Offset: small for overlap (<1.4m), larger for cluster (1.5 - 3.2m)
                if is_overlap:
                    d_along = float(self.rng.uniform(0.60, 1.35))
                    d_across = float(self.rng.uniform(-0.50, 0.50))
                else:
                    d_along = float(self.rng.uniform(1.60, 3.20))
                    d_across = float(self.rng.uniform(-0.80, 0.80))

                along_c = float(np.clip(along_m + d_along, 6.0, segment_length_m - 6.0))
                across_c = float(across_m + d_across)

                c_diam = float(base_diam * self.rng.uniform(0.60, 1.15))
                c_depth = float(np.clip(depth_m * self.rng.uniform(0.70, 1.20), min_depth, max_depth))
                c_water = is_water if self.rng.random() < 0.70 else bool(self.rng.random() < water_ratio)
                c_cov = float(np.clip(self.rng.uniform(0.25, 1.0), 0.25, 1.0)) if c_water else 0.0

                c_id = f"pothole_{self.scenario_name}_{defect_idx:03d}"
                c_spec = ProceduralDefectSpec(
                    defect_id=c_id,
                    defect_type="water_filled_pothole" if c_water else "pothole",
                    shape_category="compound_cluster" if not is_overlap else "partially_connected",
                    length_m=c_diam * float(self.rng.uniform(0.9, 1.4)),
                    width_m=c_diam,
                    depth_m=c_depth,
                    aspect_ratio=float(self.rng.uniform(0.9, 1.4)),
                    orientation_deg=float(self.rng.uniform(0.0, 360.0)),
                    irregularity=float(self.rng.uniform(0.40, 0.85)),
                    edge_breakup=float(self.rng.uniform(0.35, 0.90)),
                    roughness=float(self.rng.uniform(0.30, 0.85)),
                    is_water_filled=c_water,
                    water_depth_m=c_depth * 0.6 * c_cov if c_water else 0.0,
                    water_coverage_frac=c_cov,
                    turbidity=turbidity,
                    wet_halo_radius_m=wet_halo,
                    has_cracks=has_cracks,
                    crack_pattern=crack_pat,
                    has_road_patch=has_patch,
                    lane_position=lane_pos,
                    cluster_id=cluster_id,
                    is_overlapping=is_overlap,
                )
                plan.append((along_c, across_c, c_spec))
                defect_idx += 1

        return plan


class RoadDefectManager:
    """Manages physical defect actor spawning, tracking, and ground truth in CARLA 0.9.16."""

    def __init__(self, world=None):
        self.world = world
        self.bp_lib = world.get_blueprint_library() if world is not None else None
        self.spawned_actors: List[Any] = []
        self.ground_truth_records: List[Dict[str, Any]] = []

    def spawn_procedural_defects(
        self,
        segments: List[Any],
        scenario: str = "moderate",
        seed: int = 42,
        defects_per_segment: Optional[int] = None,
        water_ratio: Optional[float] = None,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Spawns procedural road defects with complete variation and records ground truth."""
        t0 = time.time()
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent / "output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = ProceduralRoadGenerator(scenario=scenario, seed=seed)
        self.ground_truth_records.clear()
        self.cleanup()

        mesh_plane_path = Path(__file__).resolve().parent / "assets" / "pothole_plane.obj"
        has_mesh_bp = False
        mesh_bp = None
        if self.bp_lib is not None:
            try:
                mesh_bp = self.bp_lib.find("static.prop.mesh")
                if mesh_bp is not None and mesh_plane_path.is_file():
                    has_mesh_bp = True
            except Exception:
                has_mesh_bp = False

        if verbose:
            print(f"[RoadInjector] Spawning procedural [{scenario.upper()}] defects (seed={seed}) across {len(segments)} corridor(s)...")

        total_spawned_actors = 0

        for seg_idx, start_wp in enumerate(segments):
            # Waypoint tracing along road segment (150m)
            wps = [start_wp]
            curr = start_wp
            for _ in range(30):
                nxts = curr.next(5.0)
                if not nxts:
                    break
                curr = nxts[0]
                wps.append(curr)

            if len(wps) < 4:
                continue

            seg_length_m = len(wps) * 5.0
            plan = generator.generate_corridor_plan(
                segment_length_m=seg_length_m,
                defects_count=defects_per_segment,
                water_ratio_override=water_ratio,
            )

            start_loc = start_wp.transform.location
            road_seg_id = f"seg_carla_{getattr(self.world.get_map(), 'name', 'Town04').split('/')[-1]}_{seg_idx:04d}" if self.world else f"seg_carla_town04_{seg_idx:04d}"

            for along_m, across_m, spec in plan:
                # Interpolate along waypoints to find center transform
                wp_idx = min(len(wps) - 1, int(along_m / 5.0))
                ref_wp = wps[wp_idx]
                yaw_rad = math.radians(ref_wp.transform.rotation.yaw)

                # Local Cartesian offset
                dx_along = math.cos(yaw_rad) * (along_m - wp_idx * 5.0)
                dy_along = math.sin(yaw_rad) * (along_m - wp_idx * 5.0)
                dx_lat = -math.sin(yaw_rad) * across_m
                dy_lat =  math.cos(yaw_rad) * across_m

                loc_x = float(ref_wp.transform.location.x + dx_along + dx_lat)
                loc_y = float(ref_wp.transform.location.y + dy_along + dy_lat)
                loc_z = float(ref_wp.transform.location.z + 0.035)

                rot_yaw = float((ref_wp.transform.rotation.yaw + spec.orientation_deg) % 360.0)

                actor_ids: List[int] = []

                # Georeference to GPS
                if self.world is not None and hasattr(self.world, "get_map"):
                    lat, lon, alt_geo = geo_utils.carla_transform_to_geolocation(
                        self.world.get_map(), (loc_x, loc_y, loc_z)
                    )
                else:
                    lat, lon = geo_utils.local_xy_to_latlon(loc_x - start_loc.x, loc_y - start_loc.y)
                    alt_geo = loc_z

                record = {
                    "defect_id": spec.defect_id,
                    "actor_ids": actor_ids,
                    "segment_index": int(seg_idx),
                    "road_segment_id": road_seg_id,
                    "defect_type": spec.defect_type,
                    "shape_category": spec.shape_category,
                    "carla_location": {
                        "x": round(loc_x, 3),
                        "y": round(loc_y, 3),
                        "z": round(loc_z, 3),
                    },
                    "gps_coordinates": {
                        "latitude": round(lat, 8),
                        "longitude": round(lon, 8),
                        "altitude_m": round(alt_geo or 30.0, 2),
                    },
                    "lane_position": spec.lane_position,
                    "dimensions": {
                        "length_m": round(spec.length_m, 3),
                        "width_m": round(spec.width_m, 3),
                        "diameter_m": round(spec.diameter_m, 3),
                        "depth_m": round(spec.depth_m, 3),
                        "area_m2": round(spec.area_m2, 3),
                        "aspect_ratio": round(spec.aspect_ratio, 2),
                        "orientation_deg": round(rot_yaw, 1),
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
                    "scenario": scenario.lower(),
                    "severity_category": spec.severity_category,
                    "true_severity_score": round(spec.severity_score, 3),
                    "generation_seed": seed,
                }
                self.ground_truth_records.append(record)

        # Save structured ground truth to files
        gt_export = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scenario": scenario.lower(),
                "seed": seed,
                "total_defects": len(self.ground_truth_records),
                "total_water_filled": sum(1 for d in self.ground_truth_records if d["water_state"]["is_water_filled"]),
                "total_actors_spawned": total_spawned_actors,
                "total_segments_covered": len(segments),
            },
            "defects": self.ground_truth_records,
        }

        # Write to both ground_truth.json and road_defects_ground_truth.json
        gt_path = output_dir / "ground_truth.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_export, f, indent=2)

        legacy_path = output_dir / "road_defects_ground_truth.json"
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(gt_export, f, indent=2)

        elapsed = time.time() - t0
        if verbose:
            print(f"[RoadInjector] ✓ Placed {len(self.ground_truth_records)} procedural defects in {elapsed:.2f}s.")
            print(f"[RoadInjector] Ground truth exported to: {gt_path}")

        return self.ground_truth_records

    def cleanup(self):
        """Cleanly destroy all spawned defect actors."""
        count = len(self.spawned_actors)
        for actor in self.spawned_actors:
            try:
                if actor is not None and getattr(actor, "is_alive", True):
                    actor.destroy()
            except Exception:
                pass
        self.spawned_actors.clear()
        if count > 0:
            print(f"[RoadInjector] Cleaned up {count} defect actor(s).")


# ---------------------------------------------------------------------------
# Photorealistic 3D Depression Geometry & Water Surface Projection Engine
# ---------------------------------------------------------------------------

def generate_shape_polygon(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    shape_cat: str,
    irregularity: float = 0.65,
    orientation_deg: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate irregular polygon vertices for one of the 6 realistic defect shape families."""
    rng = np.random.default_rng(seed)
    rad_yaw = math.radians(orientation_deg)
    cos_y, sin_y = math.cos(rad_yaw), math.sin(rad_yaw)

    n_pts = 72
    angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)

    if shape_cat == "elongated_longitudinal":
        r_mod = 1.0 + 0.22 * np.sin(2 * angles) + rng.normal(0, 0.04, n_pts)
        ex = rx * r_mod * np.cos(angles)
        ey = ry * r_mod * np.sin(angles)
    elif shape_cat == "elongated_transverse":
        r_mod = 1.0 + 0.22 * np.cos(2 * angles) + rng.normal(0, 0.05, n_pts)
        ex = rx * r_mod * np.cos(angles)
        ey = ry * r_mod * np.sin(angles)
    elif shape_cat == "irregular_natural":
        r_mod = np.ones(n_pts)
        for k in range(1, 5):
            r_mod += (0.35 * irregularity / (k ** 1.1)) * np.sin(k * angles + rng.uniform(0, 2 * math.pi))
        for _ in range(int(rng.integers(3, 6))):
            a_notch = rng.uniform(0, 2 * math.pi)
            diff = np.abs(np.angle(np.exp(1j * (angles - a_notch))))
            r_mod -= 0.28 * np.exp(-(diff / 0.30) ** 2)
        ex = rx * r_mod * np.cos(angles)
        ey = ry * r_mod * np.sin(angles)
    elif shape_cat == "jagged":
        r_mod = 1.0 + rng.uniform(-0.35, 0.35, n_pts) * irregularity
        ex = rx * r_mod * np.cos(angles)
        ey = ry * r_mod * np.sin(angles)
    elif shape_cat == "compound_cluster":
        centers = [(0, 0, 0.70), (rx * 0.40, ry * 0.25, 0.55), (-rx * 0.35, -ry * 0.20, 0.60)]
        r_vals = []
        for a in angles:
            best_r = 0.0
            for ocx, ocy, scale in centers:
                r_c = (scale * rx) * (1.0 + 0.18 * math.sin(3 * a + rng.uniform(0, 1)))
                d = math.hypot(ocx + r_c * math.cos(a), ocy + r_c * math.sin(a))
                if d > best_r:
                    best_r = d
            r_vals.append(best_r)
        ex = np.array(r_vals) * np.cos(angles)
        ey = np.array(r_vals) * np.sin(angles)
    elif shape_cat == "partially_connected":
        r_mod = 0.75 + 0.45 * np.abs(np.cos(angles)) ** 0.8
        ex = rx * r_mod * np.cos(angles)
        ey = ry * (0.60 + 0.40 * np.abs(np.sin(angles))) * np.sin(angles)
    else:
        r_mod = 1.0 + rng.normal(0, 0.08, n_pts)
        ex = rx * r_mod * np.cos(angles)
        ey = ry * r_mod * np.sin(angles)

    rot_x = cx + ex * cos_y - ey * sin_y
    rot_y = cy + ex * sin_y + ey * cos_y
    return np.column_stack([rot_x, rot_y]).astype(np.int32)


def draw_meandering_crack(
    canvas: np.ndarray,
    start_pt: Tuple[int, int],
    angle_rad: float,
    length_px: float,
    seed: int = 42,
) -> None:
    """Draws a meandering, naturalistic asphalt fracture branching from a defect."""
    import cv2
    rng = np.random.default_rng(seed)
    curr = np.array(start_pt, dtype=np.float32)
    pts = [curr.astype(np.int32)]
    steps = max(4, int(length_px / 4.0))
    cur_ang = angle_rad
    for _ in range(steps):
        cur_ang += float(rng.normal(0, 0.35))
        step_len = float(rng.uniform(2.5, 5.0))
        curr += np.array([math.cos(cur_ang), math.sin(cur_ang)]) * step_len
        pts.append(curr.astype(np.int32))
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts_arr], False, (55, 55, 58), 1, cv2.LINE_AA)


def render_3d_pothole_on_canvas(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    shape_cat: str = "irregular_natural",
    depth_m: float = 0.08,
    is_water: bool = False,
    water_cov: float = 0.80,
    turbidity: float = 0.30,
    orientation_deg: float = 0.0,
    has_patch: bool = False,
    has_cracks: bool = False,
    crack_pattern: str = "radial",
    seed: int = 42,
    sun_dir: Tuple[float, float] = (-0.65, -0.75),
) -> None:
    """Renders a single photorealistic 3D road depression directly onto canvas."""
    import cv2
    rng = np.random.default_rng(seed)
    h, w, _ = canvas.shape

    pad = int(max(rx, ry) * 2.5) + 35
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(w, cx + pad), min(h, cy + pad)

    local = np.ascontiguousarray(canvas[y1:y2, x1:x2].astype(np.float32))
    lh, lw, _ = local.shape
    if lh < 4 or lw < 4:
        return

    # Outer polygon for depression
    poly_pts = generate_shape_polygon(
        cx - x1, cy - y1, rx, ry, shape_cat,
        irregularity=0.72, orientation_deg=orientation_deg, seed=seed,
    )
    mask = np.zeros((lh, lw), dtype=np.float32)
    cv2.fillPoly(mask, [poly_pts], 1.0)

    # 1. Spalling / crumbling asphalt lip tightly hugging the perimeter
    dist_out = cv2.distanceTransform((1.0 - mask).astype(np.uint8), cv2.DIST_L2, 3)
    spall_lip = np.clip(1.0 - dist_out / 4.0, 0.0, 1.0)
    spall_noise = rng.uniform(0.78, 1.12, (lh, lw, 1))
    local = np.where(spall_lip[..., None] > 0.05, local * (1.0 - spall_lip[..., None] * 0.15) * spall_noise, local)

    # 2. Wet halo around water-filled pothole (capillary damp asphalt)
    if is_water:
        halo_width = max(4.0, min(rx, ry) * 0.45)
        halo_factor = np.clip(1.0 - dist_out / halo_width, 0.0, 1.0)
        halo_factor = cv2.GaussianBlur(halo_factor, (9, 9), 2.5)
        local *= (1.0 - halo_factor[..., None] * 0.36)

    # 3. 3D Concave Bowl Depth Map
    dist_in = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    max_d = np.max(dist_in) if np.max(dist_in) > 0 else 1.0
    bowl = np.clip(dist_in / max_d, 0.0, 1.0) ** 1.15

    # Directional sun shadow & specular rim
    s_norm = np.array(sun_dir, dtype=np.float32)
    s_norm /= (np.linalg.norm(s_norm) + 1e-9)
    gy, gx = np.gradient(bowl)
    slope_shadow = np.clip(-(gx * s_norm[0] + gy * s_norm[1]) * 2.2, -0.38, 0.32)

    # Ambient occlusion: deeper cavity = darker
    ao = 1.0 - (0.58 + 0.32 * (depth_m / 0.15)) * bowl

    # Cavity sub-base aggregate / gravel
    gravel_noise = rng.normal(0, 10.0, (lh, lw))
    cavity_base = np.array([38.0, 36.0, 34.0])

    # Inner slope transition from road asphalt down into cavity
    alpha = cv2.GaussianBlur(mask, (3, 3), 0.7)[..., None]

    if not is_water:
        # Dry pothole
        bed_col = cavity_base + gravel_noise[..., None]
        shaded_cavity = bed_col * (ao[..., None] + slope_shadow[..., None])
        shaded_cavity = np.clip(shaded_cavity, 14.0, 120.0)

        # Smooth sloped transition: road color drops into cavity
        cavity_depth_tone = local * (1.0 - bowl[..., None] * 0.85)
        cavity_composite = cavity_depth_tone * (1.0 - bowl[..., None]) + shaded_cavity * bowl[..., None]
        local = local * (1.0 - alpha) + cavity_composite * alpha
    else:
        # Water-filled pothole: water pools in bottom based on water_coverage
        water_threshold = 1.0 - water_cov
        water_mask = np.clip((bowl - water_threshold) / max(0.01, water_cov), 0.0, 1.0)
        water_mask = cv2.GaussianBlur(water_mask, (3, 3), 0.6)

        clear_water = np.array([45.0, 42.0, 36.0])   # Dark reflective road water
        muddy_water = np.array([55.0, 72.0, 85.0])   # Turbid silt runoff
        water_col = clear_water * (1.0 - turbidity) + muddy_water * turbidity

        # Sky glint
        glint = np.exp(-((bowl - 0.62) ** 2) / 0.03) * water_mask * 95.0

        bed_shade = (cavity_base + gravel_noise[..., None] * 0.4) * (ao[..., None] + slope_shadow[..., None])
        water_surf = water_col * (0.88 + 0.12 * turbidity) + glint[..., None]

        water_opacity = np.clip(0.38 + turbidity * 0.52, 0.32, 0.94)
        submerged = bed_shade * (1.0 - water_mask[..., None] * water_opacity) + water_surf * (water_mask[..., None] * water_opacity)

        exposed_bed = cavity_base * 1.3 + gravel_noise[..., None]
        composite = np.where(water_mask[..., None] > 0.05, submerged, exposed_bed)

        cavity_composite = local * (1.0 - bowl[..., None]) + composite * bowl[..., None]
        local = local * (1.0 - alpha) + cavity_composite * alpha

    # 4. Realistic meandering fatigue cracks
    if has_cracks:
        for i, a_deg in enumerate([20, 85, 155, 230, 310]):
            rad = math.radians(a_deg + orientation_deg)
            sx = int(cx - x1 + math.cos(rad) * rx * 0.95)
            sy = int(cy - y1 + math.sin(rad) * ry * 0.95)
            crack_len = rng.uniform(rx * 0.6, rx * 1.4)
            draw_meandering_crack(local, (sx, sy), rad, crack_len, seed=seed + i * 17)

    canvas[y1:y2, x1:x2] = np.clip(local, 0, 255).astype(np.uint8)


def project_defects_onto_frame(
    frame: np.ndarray,
    drone_x: float,
    drone_y: float,
    drone_z: float,
    drone_yaw_deg: float,
    fov_deg: float,
    defects: List[Dict[str, Any]],
    sun_dir: Tuple[float, float] = (-0.65, -0.75),
) -> np.ndarray:
    """Project structured ground truth defects and surface repairs onto a nadir drone camera frame."""
    if not defects:
        return frame

    import cv2
    image_h, image_w = frame.shape[:2]
    hfov_rad = math.radians(fov_deg)
    ground_w_m = 2.0 * drone_z * math.tan(hfov_rad / 2.0)
    px_per_m = float(image_w) / max(1.0, ground_w_m)

    rad_yaw = math.radians(drone_yaw_deg)
    cos_y = math.cos(rad_yaw)
    sin_y = math.sin(rad_yaw)

    # 1. Pass 1: Render large realistic asphalt resurfacing patches first (underlying the road surface)
    for d in defects:
        assoc = d.get("associated_defects", {})
        if not assoc.get("has_road_patch", False):
            continue

        loc = d.get("carla_location", {})
        dx = loc.get("x", 0.0) - drone_x
        dy = loc.get("y", 0.0) - drone_y

        fwd_m = dx * cos_y + dy * sin_y
        right_m = -dx * sin_y + dy * cos_y

        px = int(image_w / 2.0 + right_m * px_per_m)
        py = int(image_h / 2.0 - fwd_m * px_per_m)

        # Resurfacing patch size: 3.5m to 5.5m along road, 2.4m to 3.2m across lane
        pw = int(2.8 * px_per_m / 2.0)
        ph = int(4.5 * px_per_m / 2.0)
        margin = max(pw, ph) * 2 + 50
        if -margin <= px <= image_w + margin and -margin <= py <= image_h + margin:
            # Align patch with the road lane (vertical in nadir camera with slight paving jitter)
            patch_deg = float((int(d.get("generation_seed", 42)) % 5) - 2)
            p_rot = math.radians(patch_deg)
            cp_o, sp_o = math.cos(p_rot), math.sin(p_rot)
            corners = []
            for cdx, cdy in [(-pw, -ph), (pw, -ph), (pw, ph), (-pw, ph)]:
                corners.append([int(px + cdx * cp_o - cdy * sp_o), int(py + cdx * sp_o + cdy * cp_o)])
            patch_poly = np.array([corners], dtype=np.int32)

            p_mask = np.zeros((image_h, image_w), dtype=np.float32)
            cv2.fillPoly(p_mask, patch_poly, 1.0)
            p_mask = cv2.GaussianBlur(p_mask, (3, 3), 0.8)

            # Subtle asphalt tone shift (10% darker fresh bitumen)
            frame_f = frame.astype(np.float32)
            frame_f = frame_f * (1.0 - p_mask[..., None] * 0.10)
            # Tar sealant border along patch edges
            border_mask = cv2.Canny((p_mask * 255).astype(np.uint8), 50, 150)
            frame_f = np.where(border_mask[..., None] > 50, frame_f * 0.75, frame_f)
            frame[:] = np.clip(frame_f, 0, 255).astype(np.uint8)

    # 2. Pass 2: Render 3D depression geometry and water surfaces
    for d in defects:
        loc = d.get("carla_location", {})
        dx = loc.get("x", 0.0) - drone_x
        dy = loc.get("y", 0.0) - drone_y

        fwd_m = dx * cos_y + dy * sin_y
        right_m = -dx * sin_y + dy * cos_y

        px = int(image_w / 2.0 + right_m * px_per_m)
        py = int(image_h / 2.0 - fwd_m * px_per_m)

        dim = d.get("dimensions", {})
        rx = max(3, int(dim.get("width_m", 0.8) * px_per_m / 2.0))
        ry = max(3, int(dim.get("length_m", 0.8) * px_per_m / 2.0))

        margin = max(rx, ry) * 2 + 50
        if -margin <= px <= image_w + margin and -margin <= py <= image_h + margin:
            water = d.get("water_state", {})
            assoc = d.get("associated_defects", {})
            render_3d_pothole_on_canvas(
                canvas=frame,
                cx=px,
                cy=py,
                rx=rx,
                ry=ry,
                shape_cat=d.get("shape_category", "irregular_natural"),
                depth_m=dim.get("depth_m", 0.08),
                is_water=water.get("is_water_filled", False),
                water_cov=water.get("water_coverage_frac", 0.80),
                turbidity=water.get("turbidity", 0.30),
                orientation_deg=dim.get("orientation_deg", 0.0),
                has_patch=assoc.get("has_road_patch", False),
                has_cracks=assoc.get("has_cracks", False),
                crack_pattern=assoc.get("crack_pattern", "radial"),
                seed=d.get("generation_seed", 42) + int(d.get("defect_id", "0").split("_")[-1] if "_" in d.get("defect_id", "") else 0),
                sun_dir=sun_dir,
            )

    return frame


# Global singleton manager
_GLOBAL_MANAGER: Optional[RoadDefectManager] = None


def inject_road_defects(
    world,
    segments=None,
    scenario: str = "moderate",
    seed: int = 42,
    defects_per_segment: Optional[int] = None,
    water_ratio: Optional[float] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> RoadDefectManager:
    """Entrypoint function to procedurally generate defects into CARLA world."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        _GLOBAL_MANAGER.cleanup()

    if segments is None and world is not None:
        carla_map = world.get_map()
        segments = road_utils.find_straight_segments(carla_map)

    _GLOBAL_MANAGER = RoadDefectManager(world)
    _GLOBAL_MANAGER.spawn_procedural_defects(
        segments=segments or [],
        scenario=scenario,
        seed=seed,
        defects_per_segment=defects_per_segment,
        water_ratio=water_ratio,
        output_dir=output_dir,
        verbose=verbose,
    )
    return _GLOBAL_MANAGER


def cleanup_road_defects():
    """Destroy all currently spawned road defect actors."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        _GLOBAL_MANAGER.cleanup()
        _GLOBAL_MANAGER = None
