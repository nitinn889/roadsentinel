"""
rs_phase2_surface_defects.py
----------------------------
RoadSentinel - Phase 2 & 3: Photorealistic Highway Testbed, Multi-Scale Defect Scatter,
Interactive 3D Studio & Drone Camera Flight Controller.

Features:
1. Photorealistic 3D Highway Infrastructure:
   - Continuous 4-lane asphalt highway (250m straight + 160m curved arc).
   - High-fidelity PBR asphalt aggregate texture, 800m x 600m roadside landscape verge.
   - Painted highway markings (double yellow centerlines, dashed white lane dividers, outer fog lines).
   - Galvanized steel W-beam guardrails with support posts and streetlamps.
2. Organic Multi-Scale Pothole & Distress Scatter (Non-Uniform, Varying Sizes):
   - Wide spectrum of sizes:
     * Small Pitting & Incipient Spalls: 20cm - 38cm
     * Medium Standard Potholes: 42cm - 78cm
     * Large Severe Crater Basins: 85cm - 160cm (1.6m!)
   - Multi-scale spatial clustering: realistic damage epicenters + dispersed isolated defects.
   - Full roadway coverage: straight section, curved section, across all 4 lanes, dividers, and shoulders.
   - Natural irregular aspect ratios (non-circular) with randomized 360° orientations.
   - Photorealistic PBR textures: dry crushed rock cavity (T_RS_Pothole_Dry_D), wet waterlogged puddle (T_RS_Pothole_Wet_D),
     alligator cracking (T_RS_Crack_Alligator_D), and dynamic translucent water shader.
3. Drone Flight Navigation & Photo Capture ('C' Key):
   - Standard 6-DOF drone navigation in Unreal Engine (WASD + Mouse + Q/E).
   - Instant drone camera teleportation to key inspection vantage points (Overhead SAM 2 Survey, Low-Angle 30°, Wet Puddle Macro, Curve Vantage).
   - Press 'C' (or click button) to capture high-res drone photo to env/output/captures/ with full telemetry log.
4. Interactive GUI Studio (PySide6):
   - Floating dark-mode control window with live dropdowns for:
     * Lighting Conditions (Clear Noon, Golden Sunset, Overcast, Heavy Rain & Slick Road, Dense Fog, Night Highway)
     * Road Health (Pristine, Minor Wear, Moderate Deterioration, Severe Breakdown, Critical Hazard)
     * Defect Sizing (Multi-Scale Organic, Large Severe Craters, Micro Pitting)
     * Defect Density Slider (0.5 to 10.0 / 100m²)
     * Dynamic 'Generate World' button that updates 3D scene in real-time.
"""

import os
import sys
import json
import math
import time
import shutil
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# Unreal Engine module import (with graceful fallback for standalone testing/linting)
try:
    import unreal
    HAS_UNREAL = True
except ImportError:
    unreal = None
    HAS_UNREAL = False




WORKSPACE_ROOT = "/home/nitin-nandakumar/Downloads/roadsentinel"
CAPTURES_DIR = os.path.join(WORKSPACE_ROOT, "env", "output", "captures")
MANIFEST_DIR = os.path.join(WORKSPACE_ROOT, "env", "output", "logs")
TEXTURES_DIR = os.path.join(WORKSPACE_ROOT, "RoadSentinelSim", "Content", "RS_Roads", "Textures")
TEMPORAL_OUTPUT_ROOT = Path(WORKSPACE_ROOT) / "env" / "output" / "temporal_20_day"
TEMPORAL_MANIFEST_ROOT = TEMPORAL_OUTPUT_ROOT / "manifests"
TEMPORAL_CAPTURE_ROOT = TEMPORAL_OUTPUT_ROOT / "captures"

# The UE temporal runner owns this state while the editor is running.  It is
# deliberately separate from the random interactive scatter state: temporal
# defect IDs, road coordinates, camera poses and ground-truth records must be
# stable across all inspection days.
_TEMPORAL_STATE: Dict[str, Any] = {
    "day": None,
    "manifest_path": None,
    "segments": {},
    "ground_truth": [],
    "camera_config": {},
}

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)


# ==============================================================================
# 1. Road Network Geometry
# ==============================================================================

@dataclass
class RoadWaypoint:
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float
    lane_width_m: float = 3.75
    num_lanes: int = 4

    @property
    def total_width_m(self) -> float:
        return self.lane_width_m * self.num_lanes


@dataclass
class RoadSegment:
    segment_id: str
    length_m: float
    width_m: float
    waypoints: List[RoadWaypoint]

    @property
    def area_m2(self) -> float:
        return self.length_m * self.width_m

    def get_interpolated_point(self, s_m: float, t_m: float) -> Tuple[float, float, float, float]:
        if not self.waypoints:
            return 0.0, 0.0, 0.0, 0.0
        if len(self.waypoints) == 1:
            wp = self.waypoints[0]
            rad = math.radians(wp.yaw_deg)
            nx, ny = -math.sin(rad), math.cos(rad)
            return wp.x_m + t_m * nx, wp.y_m + t_m * ny, wp.z_m, wp.yaw_deg

        norm_s = max(0.0, min(self.length_m, s_m))
        step = self.length_m / max(1, len(self.waypoints) - 1)
        idx = int(norm_s / step)
        idx = min(idx, len(self.waypoints) - 2)
        alpha = (norm_s - idx * step) / max(1e-5, step)

        wp0 = self.waypoints[idx]
        wp1 = self.waypoints[idx + 1]

        cx = wp0.x_m + alpha * (wp1.x_m - wp0.x_m)
        cy = wp0.y_m + alpha * (wp1.y_m - wp0.y_m)
        cz = wp0.z_m + alpha * (wp1.z_m - wp0.z_m)
        yaw = wp0.yaw_deg + alpha * (wp1.yaw_deg - wp0.yaw_deg)

        rad = math.radians(yaw)
        nx = -math.sin(rad)
        ny = math.cos(rad)

        return cx + t_m * nx, cy + t_m * ny, cz, yaw


def get_road_network() -> List[RoadSegment]:
    """Constructs 250m straight highway and 157m curved highway arc (radius 150m, 60°)."""
    segments: List[RoadSegment] = []
    width = 16.0  # 4 lanes @ 3.75m + shoulders

    # Segment 1: Straight Highway (250m)
    seg1_wps = []
    for s in range(0, 251, 10):
        seg1_wps.append(RoadWaypoint(x_m=float(s), y_m=0.0, z_m=0.0, yaw_deg=0.0, num_lanes=4))
    segments.append(RoadSegment("seg_01_highway_straight", 250.0, width, seg1_wps))

    # Segment 2: Curved Highway (Arc 60 degrees, radius 150m)
    seg2_wps = []
    r_curve = 150.0
    start_x = 250.0
    start_y = 0.0
    center_x = start_x
    center_y = start_y + r_curve
    for deg in range(0, 61, 3):
        rad = math.radians(deg)
        cx = center_x + r_curve * math.sin(rad)
        cy = center_y - r_curve * math.cos(rad)
        seg2_wps.append(RoadWaypoint(x_m=cx, y_m=cy, z_m=0.0, yaw_deg=float(deg), num_lanes=4))
    arc_length = r_curve * math.radians(60.0)
    segments.append(RoadSegment("seg_02_highway_curve", arc_length, width, seg2_wps))

    return segments


# ==============================================================================
# 2. Defect Sizing & Multi-Scale Organic Scatter
# ==============================================================================

@dataclass
class DefectSpec:
    defect_type: str  # PotholeWet, PotholeDry, AlligatorCrack, LongitudinalCrack
    diameter_m: float
    depth_m: float
    water_fill_pct: float
    length_m: float = 0.0
    width_m: float = 0.0
    aspect_x: float = 1.0
    aspect_y: float = 1.0
    rotation_yaw: float = 0.0


def sample_defect_spec(
    chosen_type: str,
    size_profile: str,
    rng: random.Random
) -> DefectSpec:
    """Generates realistic varying sizes from micro-pitting (20cm) to massive craters (1.6m)."""
    # Base size tier selection
    if size_profile == "Large Severe Craters Only":
        size_tier = "large"
    elif size_profile == "Micro Pitting & Hairlines":
        size_tier = "small"
    else:  # "Multi-Scale Organic (Mixed)"
        # 25% small, 50% medium, 25% large
        size_tier = rng.choices(["small", "medium", "large"], weights=[0.25, 0.50, 0.25], k=1)[0]

    # Diameter according to tier
    if size_tier == "small":
        diameter = rng.uniform(0.20, 0.38)
        depth = rng.uniform(0.02, 0.06)
    elif size_tier == "medium":
        diameter = rng.uniform(0.42, 0.78)
        depth = rng.uniform(0.06, 0.14)
    else:  # large severe
        diameter = rng.uniform(0.85, 1.55)
        depth = rng.uniform(0.12, 0.24)

    aspect_x = diameter * rng.uniform(0.85, 1.25)
    aspect_y = diameter * rng.uniform(0.85, 1.25)
    rot_yaw = rng.uniform(0.0, 360.0)

    water_pct = 0.0
    if chosen_type == "PotholeWet":
        water_pct = rng.uniform(40.0, 95.0)

    length_m = 0.0
    width_m = 0.0
    if chosen_type == "LongitudinalCrack":
        length_m = rng.uniform(1.5, 4.8)
        width_m = rng.uniform(0.03, 0.07)
        depth = rng.uniform(0.02, 0.04)

    return DefectSpec(
        defect_type=chosen_type,
        diameter_m=round(diameter, 3),
        depth_m=round(depth, 3),
        water_fill_pct=round(water_pct, 1),
        length_m=round(length_m, 3),
        width_m=round(width_m, 3),
        aspect_x=aspect_x,
        aspect_y=aspect_y,
        rotation_yaw=rot_yaw
    )


def generate_varied_scatter(
    road_segments: List[RoadSegment],
    density_per_100m2: float,
    size_profile: str,
    road_health: str,
    water_ratio: str,
    rng: random.Random
) -> List[Tuple[RoadSegment, float, float, DefectSpec]]:
    """
    Generates non-uniform, clustered and dispersed defect points across the entire road network.
    """
    results: List[Tuple[RoadSegment, float, float, DefectSpec]] = []
    if road_health == "Pristine (Grade A)":
        return results

    # Lane offsets across 16m road (-8m to +8m)
    lane_zones = [
        (-7.0, -6.2),  # Left shoulder
        (-6.0, -4.2),  # Lane 1 (outer left)
        (-3.9, -3.6),  # Lane 1-2 divider
        (-3.4, -1.2),  # Lane 2 (inner left)
        (-0.4, 0.4),   # Centerline
        (1.2, 3.4),    # Lane 3 (inner right)
        (3.6, 3.9),    # Lane 3-4 divider
        (4.2, 6.0),    # Lane 4 (outer right)
        (6.2, 7.0),    # Right shoulder
    ]

    for segment in road_segments:
        target_count = int(math.ceil((segment.area_m2 / 100.0) * density_per_100m2))
        if target_count <= 0:
            continue

        # 1. Create damage "epicenters" (clustered wear zones)
        num_clusters = max(2, int(segment.length_m / 40.0))
        cluster_centers = [rng.uniform(10.0, segment.length_m - 10.0) for _ in range(num_clusters)]

        cluster_defects_count = int(target_count * 0.65)
        dispersed_defects_count = target_count - cluster_defects_count

        # Spawn clustered defects
        for _ in range(cluster_defects_count):
            c_s = rng.choice(cluster_centers)
            s_m = max(2.0, min(segment.length_m - 2.0, c_s + rng.gauss(0.0, 4.0)))

            zone_min, zone_max = rng.choice(lane_zones)
            t_m = rng.uniform(zone_min, zone_max)

            # Determine defect type
            if water_ratio == "100% Waterlogged Puddles":
                d_type = rng.choice(["PotholeWet", "AlligatorCrack"])
            elif water_ratio == "100% Dry Crushed Aggregate":
                d_type = rng.choice(["PotholeDry", "AlligatorCrack", "LongitudinalCrack"])
            else:
                d_type = rng.choices(["PotholeWet", "PotholeDry", "AlligatorCrack", "LongitudinalCrack"],
                                     weights=[0.38, 0.32, 0.18, 0.12], k=1)[0]

            spec = sample_defect_spec(d_type, size_profile, rng)
            results.append((segment, s_m, t_m, spec))

        # Spawn dispersed defects (scattered across the full road)
        for _ in range(dispersed_defects_count):
            s_m = rng.uniform(3.0, segment.length_m - 3.0)
            zone_min, zone_max = rng.choice(lane_zones)
            t_m = rng.uniform(zone_min, zone_max)

            if water_ratio == "100% Waterlogged Puddles":
                d_type = rng.choice(["PotholeWet", "AlligatorCrack"])
            elif water_ratio == "100% Dry Crushed Aggregate":
                d_type = rng.choice(["PotholeDry", "AlligatorCrack", "LongitudinalCrack"])
            else:
                d_type = rng.choices(["PotholeWet", "PotholeDry", "AlligatorCrack", "LongitudinalCrack"],
                                     weights=[0.35, 0.35, 0.15, 0.15], k=1)[0]

            spec = sample_defect_spec(d_type, size_profile, rng)
            results.append((segment, s_m, t_m, spec))

    return results


def compute_8_corner_bbox_world(
    center_cm: Tuple[float, float, float],
    yaw_deg: float,
    half_extents_cm: Tuple[float, float, float]
) -> List[List[float]]:
    lx, ly, lz = center_cm
    sx, sy, sz = half_extents_cm
    rad = math.radians(yaw_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    corners = []
    for dx in [-1.0, 1.0]:
        for dy in [-1.0, 1.0]:
            for dz in [-1.0, 1.0]:
                ox = sx * dx
                oy = sy * dy
                oz = sz * dz
                rx = ox * cos_a - oy * sin_a
                ry = ox * sin_a + oy * cos_a
                rz = oz
                corners.append([
                    round(lx + rx, 2),
                    round(ly + ry, 2),
                    round(lz + rz, 2)
                ])
    return corners


# ==============================================================================
# 3. PBR Textures & Materials Importer
# ==============================================================================

def import_texture_to_unreal(image_path: str, destination_path: str = "/Game/RS_Roads/Textures") -> Optional[Any]:
    if not HAS_UNREAL or not os.path.exists(image_path):
        return None
    asset_name = os.path.splitext(os.path.basename(image_path))[0]
    full_pkg = f"{destination_path}/{asset_name}"
    existing = unreal.EditorAssetLibrary.load_asset(full_pkg)
    if existing:
        return existing

    task = unreal.AssetImportTask()
    task.set_editor_property("filename", os.path.abspath(image_path))
    task.set_editor_property("destination_path", destination_path)
    task.set_editor_property("destination_name", asset_name)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)
    task.set_editor_property("replace_existing", True)

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return unreal.EditorAssetLibrary.load_asset(full_pkg)


def get_or_create_textured_material(
    name: str,
    texture_path: Optional[str] = None,
    fallback_color: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    roughness: float = 0.85,
    metallic: float = 0.0,
    u_tiling: float = 1.0,
    v_tiling: float = 1.0,
    force_recreate: bool = False
) -> Optional[Any]:
    if not HAS_UNREAL:
        return None
    package_path = "/Game/RS_Roads/Materials"
    full_path = f"{package_path}/{name}"

    if force_recreate and unreal.EditorAssetLibrary.does_asset_exist(full_path):
        unreal.EditorAssetLibrary.delete_asset(full_path)

    loaded = unreal.EditorAssetLibrary.load_asset(full_path)
    if loaded and not force_recreate:
        return loaded

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    factory = unreal.MaterialFactoryNew()
    mat = asset_tools.create_asset(name, package_path, unreal.Material, factory)
    if not mat:
        return None

    mel = unreal.MaterialEditingLibrary
    tex_asset = None
    if texture_path and os.path.exists(texture_path):
        tex_asset = import_texture_to_unreal(texture_path)

    if tex_asset:
        tex_sample = mel.create_material_expression(mat, unreal.MaterialExpressionTextureSample, -350, -200)
        tex_sample.set_editor_property("texture", tex_asset)
        if u_tiling != 1.0 or v_tiling != 1.0:
            tex_coord = mel.create_material_expression(mat, unreal.MaterialExpressionTextureCoordinate, -600, -200)
            tex_coord.set_editor_property("u_tiling", u_tiling)
            tex_coord.set_editor_property("v_tiling", v_tiling)
            mel.connect_material_expressions(tex_coord, "", tex_sample, "UVs")
        mel.connect_material_property(tex_sample, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    else:
        col = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -400, -100)
        col.set_editor_property("parameter_name", "BaseColor")
        col.set_editor_property("default_value", unreal.LinearColor(fallback_color[0], fallback_color[1], fallback_color[2], 1.0))
        mel.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)

    rough_exp = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -400, 100)
    rough_exp.set_editor_property("parameter_name", "Roughness")
    rough_exp.set_editor_property("default_value", roughness)
    mel.connect_material_property(rough_exp, "", unreal.MaterialProperty.MP_ROUGHNESS)

    if metallic > 0.0:
        met_exp = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -400, 250)
        met_exp.set_editor_property("parameter_name", "Metallic")
        met_exp.set_editor_property("default_value", metallic)
        mel.connect_material_property(met_exp, "", unreal.MaterialProperty.MP_METALLIC)

    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def ensure_water_material_exists() -> Optional[Any]:
    if not HAS_UNREAL:
        return None
    full_path = "/Game/RS_Roads/Materials/M_RS_WaterPothole"
    loaded = unreal.EditorAssetLibrary.load_asset(full_path)
    if loaded:
        return loaded

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    factory = unreal.MaterialFactoryNew()
    mat = asset_tools.create_asset("M_RS_WaterPothole", "/Game/RS_Roads/Materials", unreal.Material, factory)
    if not mat:
        return None

    mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_DEFAULT_LIT)
    mat.set_editor_property("two_sided", True)

    mel = unreal.MaterialEditingLibrary
    ior_param = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -350, 250)
    ior_param.set_editor_property("parameter_name", "RS_WaterIOR")
    ior_param.set_editor_property("default_value", 1.333)
    mel.connect_material_property(ior_param, "", unreal.MaterialProperty.MP_REFRACTION)

    water_col = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -500, -250)
    water_col.set_editor_property("parameter_name", "WaterBase")
    water_col.set_editor_property("default_value", unreal.LinearColor(0.04, 0.06, 0.08, 0.9))
    mel.connect_material_property(water_col, "", unreal.MaterialProperty.MP_BASE_COLOR)

    rough_param = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -300, 50)
    rough_param.set_editor_property("parameter_name", "Roughness")
    rough_param.set_editor_property("default_value", 0.03)
    mel.connect_material_property(rough_param, "", unreal.MaterialProperty.MP_ROUGHNESS)

    op_param = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -300, 180)
    op_param.set_editor_property("parameter_name", "Opacity")
    op_param.set_editor_property("default_value", 0.88)
    mel.connect_material_property(op_param, "", unreal.MaterialProperty.MP_OPACITY)

    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


# ==============================================================================
# 4. Lighting & Atmospheric Presets Controller
# ==============================================================================

LIGHTING_PRESETS = {
    "Clear Noon (70° Sun)": {
        "pitch": -70.0, "yaw": 180.0, "sun_lux": 100000.0, "temp": 6500.0,
        "sky_int": 1.2, "fog_density": 0.001, "road_wetness": 0.0
    },
    "Golden Hour Sunset": {
        "pitch": -12.0, "yaw": 250.0, "sun_lux": 45000.0, "temp": 4200.0,
        "sky_int": 1.8, "fog_density": 0.006, "road_wetness": 0.0
    },
    "Overcast Day": {
        "pitch": -45.0, "yaw": 160.0, "sun_lux": 25000.0, "temp": 7200.0,
        "sky_int": 3.5, "fog_density": 0.012, "road_wetness": 0.10
    },
    "Heavy Rain & Wet Road": {
        "pitch": -30.0, "yaw": 180.0, "sun_lux": 8000.0, "temp": 6500.0,
        "sky_int": 2.2, "fog_density": 0.035, "road_wetness": 0.95
    },
    "Dense Atmospheric Fog": {
        "pitch": -35.0, "yaw": 180.0, "sun_lux": 6000.0, "temp": 6200.0,
        "sky_int": 1.5, "fog_density": 0.08, "road_wetness": 0.40
    },
    "Night Highway with Lamps": {
        "pitch": 25.0, "yaw": 180.0, "sun_lux": 0.0, "temp": 6500.0,
        "sky_int": 0.15, "fog_density": 0.004, "road_wetness": 0.05
    },
}


def apply_environment_lighting(preset_name: str):
    """Dynamically applies Sun angle, intensity, sky light, fog, and wetness to the 3D scene."""
    if not HAS_UNREAL:
        return
    params = LIGHTING_PRESETS.get(preset_name, LIGHTING_PRESETS["Clear Noon (70° Sun)"])

    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
    sun_actor = None
    sky_actor = None
    fog_actor = None

    for a in all_actors:
        if not a:
            continue
        cname = a.get_class().get_name()
        if cname == "DirectionalLight" or a.get_component_by_class(unreal.DirectionalLightComponent):
            sun_actor = a
        elif cname == "SkyLight" or a.get_component_by_class(unreal.SkyLightComponent):
            sky_actor = a
        elif cname == "ExponentialHeightFog" or a.get_component_by_class(unreal.ExponentialHeightFogComponent):
            fog_actor = a

    # Sun DirectionalLight
    if sun_actor:
        rot = unreal.Rotator(roll=0.0, pitch=params["pitch"], yaw=params["yaw"])
        sun_actor.set_actor_rotation(rot, False)
        scomp = sun_actor.get_component_by_class(unreal.DirectionalLightComponent)
        if scomp:
            scomp.set_editor_property("intensity", max(0.01, params["sun_lux"] / 10.0))
            scomp.set_editor_property("use_temperature", True)
            scomp.set_editor_property("temperature", params["temp"])

    # SkyLight
    if sky_actor:
        kcomp = sky_actor.get_component_by_class(unreal.SkyLightComponent)
        if kcomp:
            kcomp.set_editor_property("intensity", params["sky_int"])

    # Fog
    if fog_actor:
        fcomp = fog_actor.get_component_by_class(unreal.ExponentialHeightFogComponent)
        if fcomp:
            fcomp.set_editor_property("fog_density", params["fog_density"])


# ==============================================================================
# 5. Scene Construction & Defect Spawner
# ==============================================================================

def spawn_full_world(
    road_health: str = "Moderate Deterioration (Grade C)",
    lighting_preset: str = "Clear Noon (70° Sun)",
    size_profile: str = "Multi-Scale Organic (Mixed)",
    density_per_100m2: float = 3.5,
    water_ratio: str = "Mixed Wet/Dry",
    random_seed: int = 42
) -> List[dict]:
    """
    Spawns complete world according to parameters:
    Highway + landscape + varied defect scatter + lighting conditions.
    """
    rng = random.Random(random_seed)
    road_segments = get_road_network()

    if HAS_UNREAL:
        # 1. Clear previous actors
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        for actor in all_actors:
            if actor and (actor.actor_has_tag(unreal.Name("RS_Road")) or
                          actor.actor_has_tag(unreal.Name("RS_Marking")) or
                          actor.actor_has_tag(unreal.Name("RS_Guardrail")) or
                          actor.actor_has_tag(unreal.Name("RS_Streetlamp")) or
                          actor.actor_has_tag(unreal.Name("RS_Terrain")) or
                          actor.actor_has_tag(unreal.Name("RS_Defect")) or
                          actor.get_name() == "Floor"):
                unreal.EditorLevelLibrary.destroy_actor(actor)

        cube_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube")
        cyl_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cylinder.Cylinder")

        # Textures & Materials
        tex_asphalt = os.path.join(TEXTURES_DIR, "T_RS_Asphalt_D.jpg")
        tex_terrain = os.path.join(TEXTURES_DIR, "T_RS_Terrain_Verge_D.jpg")
        tex_pothole_dry = os.path.join(TEXTURES_DIR, "T_RS_Pothole_Dry_D.jpg")
        tex_pothole_wet = os.path.join(TEXTURES_DIR, "T_RS_Pothole_Wet_D.jpg")
        tex_alligator = os.path.join(TEXTURES_DIR, "T_RS_Crack_Alligator_D.jpg")

        mat_asphalt = get_or_create_textured_material("M_RS_Asphalt_Clean", tex_asphalt, roughness=0.82, u_tiling=25.0, v_tiling=4.0)
        mat_terrain = get_or_create_textured_material("M_RS_Terrain_Grass", tex_terrain, roughness=0.94, u_tiling=60.0, v_tiling=60.0)
        mat_white = get_or_create_textured_material("M_RS_Marking_White", fallback_color=(0.90, 0.90, 0.90), roughness=0.35)
        mat_yellow = get_or_create_textured_material("M_RS_Marking_Yellow", fallback_color=(0.95, 0.70, 0.04), roughness=0.35)
        mat_metal = get_or_create_textured_material("M_RS_Guardrail_Metal", fallback_color=(0.75, 0.77, 0.80), roughness=0.28, metallic=0.95)

        mat_pothole_dry = get_or_create_textured_material("M_RS_Pothole_Cavity", tex_pothole_dry, roughness=0.98)
        mat_pothole_wet = get_or_create_textured_material("M_RS_Pothole_Wet_PBR", tex_pothole_wet, roughness=0.15)
        mat_alligator = get_or_create_textured_material("M_RS_Crack_Distress", tex_alligator, roughness=0.94)
        mat_water = ensure_water_material_exists()

        # 2. Grand Landscape Ground Plane (800m x 600m)
        terrain_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.StaticMeshActor,
            unreal.Vector(25000.0, 5000.0, -18.0),
            unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
        )
        if terrain_actor and cube_mesh:
            terrain_actor.tags.append(unreal.Name("RS_Terrain"))
            tc = terrain_actor.static_mesh_component
            if tc:
                tc.set_static_mesh(cube_mesh)
                tc.set_world_scale3d(unreal.Vector(800.0, 600.0, 0.10))
                if mat_terrain:
                    tc.set_material(0, mat_terrain)

        # 3. Straight Highway Slab (250m continuous)
        road_width_cm = 1600.0
        slab_thick_cm = 20.0
        marking_thick_cm = 4.0

        straight_road = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.StaticMeshActor,
            unreal.Vector(12500.0, 0.0, -(slab_thick_cm / 2.0)),
            unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
        )
        if straight_road and cube_mesh:
            straight_road.tags.append(unreal.Name("RS_Road"))
            sc = straight_road.static_mesh_component
            if sc:
                sc.set_static_mesh(cube_mesh)
                sc.set_world_scale3d(unreal.Vector(250.0, road_width_cm / 100.0, slab_thick_cm / 100.0))
                if mat_asphalt:
                    sc.set_material(0, mat_asphalt)

        # Double yellow center lines
        for offset_cm in [-14.0, 14.0]:
            ylw = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(12500.0, offset_cm, 2.0),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
            )
            if ylw and cube_mesh:
                ylw.tags.append(unreal.Name("RS_Marking"))
                yc = ylw.static_mesh_component
                if yc:
                    yc.set_static_mesh(cube_mesh)
                    yc.set_world_scale3d(unreal.Vector(250.0, 0.22, marking_thick_cm / 100.0))
                    if mat_yellow:
                        yc.set_material(0, mat_yellow)

        # Outer white fog lines
        for fog_offset_cm in [-750.0, 750.0]:
            fog = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(12500.0, fog_offset_cm, 2.0),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
            )
            if fog and cube_mesh:
                fog.tags.append(unreal.Name("RS_Marking"))
                fc = fog.static_mesh_component
                if fc:
                    fc.set_static_mesh(cube_mesh)
                    fc.set_world_scale3d(unreal.Vector(250.0, 0.20, marking_thick_cm / 100.0))
                    if mat_white:
                        fc.set_material(0, mat_white)

        # Dashed white lane lines
        for x_m in range(6, 246, 12):
            for lane_offset_cm in [-375.0, 375.0]:
                dash = unreal.EditorLevelLibrary.spawn_actor_from_class(
                    unreal.StaticMeshActor,
                    unreal.Vector(x_m * 100.0, lane_offset_cm, 2.0),
                    unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
                )
                if dash and cube_mesh:
                    dash.tags.append(unreal.Name("RS_Marking"))
                    dc = dash.static_mesh_component
                    if dc:
                        dc.set_static_mesh(cube_mesh)
                        dc.set_world_scale3d(unreal.Vector(3.0, 0.20, marking_thick_cm / 100.0))
                        if mat_white:
                            dc.set_material(0, mat_white)

        # Guardrails
        for side_sign in [-1.0, 1.0]:
            rail_y = side_sign * 820.0
            rail = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(12500.0, rail_y, 55.0),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
            )
            if rail and cube_mesh:
                rail.tags.append(unreal.Name("RS_Guardrail"))
                rc = rail.static_mesh_component
                if rc:
                    rc.set_static_mesh(cube_mesh)
                    rc.set_world_scale3d(unreal.Vector(250.0, 0.12, 0.35))
                    if mat_metal:
                        rc.set_material(0, mat_metal)

            for post_x in range(0, 251, 6):
                post = unreal.EditorLevelLibrary.spawn_actor_from_class(
                    unreal.StaticMeshActor,
                    unreal.Vector(post_x * 100.0, rail_y, 25.0),
                    unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
                )
                if post and cube_mesh:
                    post.tags.append(unreal.Name("RS_Guardrail"))
                    pc = post.static_mesh_component
                    if pc:
                        pc.set_static_mesh(cube_mesh)
                        pc.set_world_scale3d(unreal.Vector(0.12, 0.12, 0.60))
                        if mat_metal:
                            pc.set_material(0, mat_metal)

        # Streetlamps
        for lamp_x in range(20, 250, 40):
            lamp_y = 960.0
            pole = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(lamp_x * 100.0, lamp_y, 300.0),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
            )
            if pole and cyl_mesh:
                pole.tags.append(unreal.Name("RS_Streetlamp"))
                plc = pole.static_mesh_component
                if plc:
                    plc.set_static_mesh(cyl_mesh)
                    plc.set_world_scale3d(unreal.Vector(0.15, 0.15, 3.0))
                    if mat_metal:
                        plc.set_material(0, mat_metal)

            arm = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(lamp_x * 100.0, lamp_y - 120.0, 600.0),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
            )
            if arm and cube_mesh:
                arm.tags.append(unreal.Name("RS_Streetlamp"))
                ac = arm.static_mesh_component
                if ac:
                    ac.set_static_mesh(cube_mesh)
                    ac.set_world_scale3d(unreal.Vector(0.12, 2.4, 0.12))
                    if mat_metal:
                        ac.set_material(0, mat_metal)

        # 4. Curved Highway Arc (Segment 2)
        r_curve = 150.0
        start_x = 250.0
        start_y = 0.0
        center_x = start_x
        center_y = start_y + r_curve

        for deg in range(0, 60, 2):
            deg_mid = deg + 1.0
            rad_mid = math.radians(deg_mid)
            yaw_deg = deg_mid

            cx_cm = (center_x + r_curve * math.sin(rad_mid)) * 100.0
            cy_cm = (center_y - r_curve * math.cos(rad_mid)) * 100.0

            c_road = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(cx_cm, cy_cm, -(slab_thick_cm / 2.0)),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=yaw_deg)
            )
            if c_road and cube_mesh:
                c_road.tags.append(unreal.Name("RS_Road"))
                crc = c_road.static_mesh_component
                if crc:
                    crc.set_static_mesh(cube_mesh)
                    crc.set_world_scale3d(unreal.Vector(5.5, road_width_cm / 100.0, slab_thick_cm / 100.0))
                    if mat_asphalt:
                        crc.set_material(0, mat_asphalt)

            c_ylw = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(cx_cm, cy_cm, 2.0),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=yaw_deg)
            )
            if c_ylw and cube_mesh:
                c_ylw.tags.append(unreal.Name("RS_Marking"))
                cyc = c_ylw.static_mesh_component
                if cyc:
                    cyc.set_static_mesh(cube_mesh)
                    cyc.set_world_scale3d(unreal.Vector(5.5, 0.25, marking_thick_cm / 100.0))
                    if mat_yellow:
                        cyc.set_material(0, mat_yellow)

    # 5. Scatter Defects with Non-Uniform Sizing
    scattered = generate_varied_scatter(
        road_segments=road_segments,
        density_per_100m2=density_per_100m2,
        size_profile=size_profile,
        road_health=road_health,
        water_ratio=water_ratio,
        rng=rng
    )

    manifest_entries: List[dict] = []
    actor_counter = 0

    for segment, s_m, t_m, spec in scattered:
        actor_counter += 1
        x_m, y_m, z_m, yaw_deg = segment.get_interpolated_point(s_m, t_m)
        world_loc_cm = (x_m * 100.0, y_m * 100.0, z_m * 100.0 + 0.3)
        actor_id = f"RS_Defect_{spec.defect_type}_{actor_counter:04d}"

        half_extents = (spec.diameter_m * 50.0, spec.diameter_m * 50.0, spec.depth_m * 50.0)
        bbox = compute_8_corner_bbox_world(world_loc_cm, spec.rotation_yaw, half_extents)

        entry = {
            "actor_id": actor_id,
            "defect_type": spec.defect_type,
            "location": {"x": round(world_loc_cm[0], 2), "y": round(world_loc_cm[1], 2), "z": round(world_loc_cm[2], 2)},
            "road_segment_id": segment.segment_id,
            "diameter_m": spec.diameter_m,
            "depth_m": spec.depth_m,
            "water_fill_pct": spec.water_fill_pct,
            "bounding_box_world": bbox
        }
        manifest_entries.append(entry)

        if HAS_UNREAL:
            loc_vec = unreal.Vector(world_loc_cm[0], world_loc_cm[1], world_loc_cm[2])
            rot_val = unreal.Rotator(roll=0.0, pitch=0.0, yaw=spec.rotation_yaw)

            # Spawn Pothole Dry
            if spec.defect_type == "PotholeDry":
                cavity = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, loc_vec, rot_val)
                if cavity and cyl_mesh:
                    cavity.tags.append(unreal.Name("RS_Defect"))
                    cavity.set_actor_label(actor_id)
                    comp = cavity.static_mesh_component
                    if comp:
                        comp.set_static_mesh(cyl_mesh)
                        comp.set_world_scale3d(unreal.Vector(spec.aspect_x, spec.aspect_y, 0.005))
                        if mat_pothole_dry:
                            comp.set_material(0, mat_pothole_dry)

            # Spawn Pothole Wet
            elif spec.defect_type == "PotholeWet":
                cavity = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, loc_vec, rot_val)
                if cavity and cyl_mesh:
                    cavity.tags.append(unreal.Name("RS_Defect"))
                    cavity.set_actor_label(actor_id)
                    comp = cavity.static_mesh_component
                    if comp:
                        comp.set_static_mesh(cyl_mesh)
                        comp.set_world_scale3d(unreal.Vector(spec.aspect_x, spec.aspect_y, 0.005))
                        if mat_pothole_wet:
                            comp.set_material(0, mat_pothole_wet)

                # Concentric specular water puddle layer
                if mat_water and cyl_mesh:
                    water_loc = unreal.Vector(world_loc_cm[0], world_loc_cm[1], world_loc_cm[2] + 0.2)
                    water_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, water_loc, rot_val)
                    if water_actor:
                        water_actor.tags.append(unreal.Name("RS_Defect"))
                        wcomp = water_actor.static_mesh_component
                        if wcomp:
                            wcomp.set_static_mesh(cyl_mesh)
                            p_scale = spec.diameter_m * 0.78
                            wcomp.set_world_scale3d(unreal.Vector(p_scale, p_scale, 0.003))
                            wcomp.set_material(0, mat_water)

            # Spawn Alligator Crack
            elif spec.defect_type == "AlligatorCrack":
                crack_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, loc_vec, rot_val)
                if crack_actor and cyl_mesh:
                    crack_actor.tags.append(unreal.Name("RS_Defect"))
                    comp = crack_actor.static_mesh_component
                    if comp:
                        comp.set_static_mesh(cyl_mesh)
                        comp.set_world_scale3d(unreal.Vector(spec.diameter_m, spec.diameter_m, 0.004))
                        if mat_alligator:
                            comp.set_material(0, mat_alligator)

            # Spawn Longitudinal Crack
            elif spec.defect_type == "LongitudinalCrack":
                crack_rot = unreal.Rotator(roll=0.0, pitch=0.0, yaw=yaw_deg + rng.uniform(-4.0, 4.0))
                long_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, loc_vec, crack_rot)
                if long_actor and cube_mesh:
                    long_actor.tags.append(unreal.Name("RS_Defect"))
                    comp = long_actor.static_mesh_component
                    if comp:
                        comp.set_static_mesh(cube_mesh)
                        comp.set_world_scale3d(unreal.Vector(spec.length_m, max(0.04, spec.width_m), 0.004))
                        if mat_alligator:
                            comp.set_material(0, mat_alligator)

    # 6. Apply Lighting & Weather Condition
    apply_environment_lighting(lighting_preset)

    # 7. Write Ground Truth Manifest
    manifest_path = os.path.join(MANIFEST_DIR, "rs_defects_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "road_health": road_health,
            "lighting_preset": lighting_preset,
            "size_profile": size_profile,
            "density_per_100m2": density_per_100m2,
            "total_defects": len(manifest_entries),
            "defects": manifest_entries
        }, f, indent=2)

    if HAS_UNREAL:
        unreal.log(f"Generated world: {road_health}, {lighting_preset}, {len(manifest_entries)} defects.")

    return manifest_entries


# ==============================================================================
# 6. Deterministic 20-Day Temporal Scene Materialisation
# ==============================================================================

def _resolve_temporal_path(raw_path: str, purpose: str) -> Path:
    """Resolve a temporal artifact while keeping it under ``env/output``."""
    if not raw_path:
        raise ValueError(f"Missing {purpose} path")
    path = Path(str(raw_path)).expanduser().resolve()
    allowed_root = (Path(WORKSPACE_ROOT) / "env" / "output").resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"{purpose} must be inside {allowed_root}") from exc
    return path


def _finite_number(value: Any, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return number


def _temporal_route_pose(along_m: float, across_m: float) -> Tuple[float, float, float, float]:
    """Map a cumulative road distance to the fixed UE highway geometry.

    The first 250 m is the straight, followed by the persistent curved road
    section.  This is the one mapping used for both the actual defect meshes
    and the camera poses, so a SEG id cannot drift between days.
    """
    remaining = along_m
    road_segments = get_road_network()
    total_length = sum(segment.length_m for segment in road_segments)
    if not 0.0 <= along_m <= total_length:
        raise ValueError(f"along_m={along_m} falls outside the Unreal road route (0..{total_length:.1f} m)")
    for segment in road_segments:
        if remaining <= segment.length_m + 1e-6:
            return segment.get_interpolated_point(max(0.0, remaining), across_m)
        remaining -= segment.length_m
    return road_segments[-1].get_interpolated_point(road_segments[-1].length_m, across_m)


def _temporal_capture_config(manifest: Dict[str, Any]) -> Dict[str, float]:
    """Return a bounded, fixed camera configuration for the whole experiment."""
    raw = manifest.get("unreal_capture", {})
    if not isinstance(raw, dict):
        raw = {}
    # Do not inherit CARLA's historical 100 m default from fixed_flight.  The
    # native UE road is much smaller and 25 m provides defect-visible imagery.
    return {
        "altitude_m": _finite_number(raw.get("altitude_m", 25.0), "unreal_capture.altitude_m", 8.0, 80.0),
        "pitch_deg": _finite_number(raw.get("pitch_deg", -89.0), "unreal_capture.pitch_deg", -90.0, -45.0),
        "fov_deg": _finite_number(raw.get("fov_deg", 70.0), "unreal_capture.fov_deg", 35.0, 110.0),
        "width": int(_finite_number(raw.get("width", 1920), "unreal_capture.width", 320, 4096)),
        "height": int(_finite_number(raw.get("height", 1080), "unreal_capture.height", 240, 4096)),
    }


def _temporal_materials() -> Dict[str, Any]:
    """Load the same native PBR assets used by the interactive UE world."""
    if not HAS_UNREAL:
        raise RuntimeError("The temporal scene must run inside Unreal Engine")
    return {
        "cube": unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube"),
        "cylinder": unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cylinder.Cylinder"),
        "dry": get_or_create_textured_material(
            "M_RS_Pothole_Cavity", os.path.join(TEXTURES_DIR, "T_RS_Pothole_Dry_D.jpg"), roughness=0.98
        ),
        "wet": get_or_create_textured_material(
            "M_RS_Pothole_Wet_PBR", os.path.join(TEXTURES_DIR, "T_RS_Pothole_Wet_D.jpg"), roughness=0.15
        ),
        "crack": get_or_create_textured_material(
            "M_RS_Crack_Distress", os.path.join(TEXTURES_DIR, "T_RS_Crack_Alligator_D.jpg"), roughness=0.94
        ),
        "water": ensure_water_material_exists(),
    }


def _tag_temporal_actor(actor: Any, segment_id: str, defect_id: str) -> None:
    if not actor:
        return
    for tag in ("RS_Defect", "RS_TemporalDefect", f"RS_Segment_{segment_id}", f"RS_Defect_{defect_id}"):
        actor.tags.append(unreal.Name(tag))


def _spawn_temporal_defect(raw: Dict[str, Any], materials: Dict[str, Any], day: int) -> Dict[str, Any]:
    """Instantiate one manifest defect at its persistent UE road coordinate."""
    segment_id = str(raw.get("road_segment_id", "")).strip()
    defect_id = str(raw.get("defect_id", "")).strip()
    if not segment_id or not defect_id:
        raise ValueError("Every temporal defect needs road_segment_id and defect_id")

    along_m = _finite_number(raw.get("along_m"), f"{defect_id}.along_m", 0.0, 420.0)
    across_m = _finite_number(raw.get("across_m"), f"{defect_id}.across_m", -7.5, 7.5)
    dims = raw.get("dimensions", {})
    if not isinstance(dims, dict):
        raise ValueError(f"{defect_id}.dimensions must be an object")
    length_m = _finite_number(dims.get("length_m", dims.get("diameter_m", 0.4)), f"{defect_id}.length_m", 0.03, 4.0)
    width_m = _finite_number(dims.get("width_m", dims.get("diameter_m", 0.4)), f"{defect_id}.width_m", 0.02, 4.0)
    depth_m = _finite_number(dims.get("depth_m", 0.01), f"{defect_id}.depth_m", 0.001, 0.5)
    local_orientation = _finite_number(dims.get("orientation_deg", 0.0), f"{defect_id}.orientation_deg", -360.0, 360.0)
    defect_type = str(raw.get("defect_type", "pothole")).strip().lower()
    water_state = raw.get("water_state", {})
    if not isinstance(water_state, dict):
        water_state = {}
    is_water = bool(water_state.get("is_water_filled", False)) or defect_type == "water_filled_pothole"

    x_m, y_m, z_m, road_yaw_deg = _temporal_route_pose(along_m, across_m)
    world_yaw_deg = road_yaw_deg + local_orientation
    loc = unreal.Vector(x_m * 100.0, y_m * 100.0, z_m * 100.0 + 0.55)
    rot = unreal.Rotator(roll=0.0, pitch=0.0, yaw=world_yaw_deg)
    actor_label = f"RS_Temporal_{segment_id}_{defect_id}"
    actor_ids: List[str] = []

    if defect_type == "crack":
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, loc, rot)
        if actor and materials["cube"]:
            actor.set_actor_label(actor_label)
            _tag_temporal_actor(actor, segment_id, defect_id)
            comp = actor.static_mesh_component
            comp.set_static_mesh(materials["cube"])
            comp.set_world_scale3d(unreal.Vector(length_m, max(0.025, width_m), 0.004))
            if materials["crack"]:
                comp.set_material(0, materials["crack"])
            actor_ids.append(actor.get_name())
    else:
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, loc, rot)
        if actor and materials["cylinder"]:
            actor.set_actor_label(actor_label)
            _tag_temporal_actor(actor, segment_id, defect_id)
            comp = actor.static_mesh_component
            comp.set_static_mesh(materials["cylinder"])
            comp.set_world_scale3d(unreal.Vector(length_m, width_m, max(0.006, depth_m * 0.06)))
            if is_water and materials["wet"]:
                comp.set_material(0, materials["wet"])
            elif materials["dry"]:
                comp.set_material(0, materials["dry"])
            actor_ids.append(actor.get_name())

        if is_water and materials["water"] and materials["cylinder"]:
            water = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(loc.x, loc.y, loc.z + 0.35),
                rot,
            )
            if water:
                water.set_actor_label(f"{actor_label}_Water")
                _tag_temporal_actor(water, segment_id, defect_id)
                water_comp = water.static_mesh_component
                water_comp.set_static_mesh(materials["cylinder"])
                coverage = _finite_number(
                    water_state.get("water_coverage_frac", 0.75),
                    f"{defect_id}.water_coverage_frac",
                    0.05,
                    1.0,
                )
                water_comp.set_world_scale3d(unreal.Vector(length_m * coverage, width_m * coverage, 0.003))
                water_comp.set_material(0, materials["water"])
                actor_ids.append(water.get_name())

    half_extents = (length_m * 50.0, width_m * 50.0, max(1.0, depth_m * 50.0))
    ground_truth = {
        "actor_id": actor_label,
        "actor_names": actor_ids,
        "defect_id": defect_id,
        "road_segment_id": segment_id,
        "day": day,
        "defect_type": "water_filled_pothole" if is_water else defect_type,
        "location": {"x_cm": round(loc.x, 2), "y_cm": round(loc.y, 2), "z_cm": round(loc.z, 2)},
        "route_location": {"along_m": round(along_m, 3), "across_m": round(across_m, 3), "road_yaw_deg": round(road_yaw_deg, 3)},
        "dimensions": {"length_m": length_m, "width_m": width_m, "depth_m": depth_m},
        "water_state": {"is_water_filled": is_water, **water_state},
        "bounding_box_world": compute_8_corner_bbox_world((loc.x, loc.y, loc.z), world_yaw_deg, half_extents),
        "renderer_condition_score": raw.get("true_severity_score"),
    }
    return ground_truth


_CURRENT_SELECTED_STATE: Dict[str, Any] = {
    "segment_id": "SEG_001",
    "day": 1
}

DETERMINISTIC_SEGMENT_LOCATIONS = {
    "SEG_001": {"along_m": 40.0, "across_m": 0.0, "desc": "Paved Highway Wheeltrack - Progressive Fatigue"},
    "SEG_002": {"along_m": 90.0, "across_m": 0.0, "desc": "Rapid Pothole & Waterlogging"},
    "SEG_003": {"along_m": 140.0, "across_m": 0.0, "desc": "Thermal Transverse & Block Cracking"},
    "SEG_004": {"along_m": 190.0, "across_m": 0.0, "desc": "Cluster Pothole Damage"},
    "SEG_005": {"along_m": 240.0, "across_m": 0.0, "desc": "High-Resilience Control Segment"},
    "SEG_006": {"along_m": 300.0, "across_m": 0.0, "desc": "Curved Highway Edge Shear Damage"},
}


def get_deterministic_segment_defects(segment_id: str, day: int) -> Tuple[List[Dict[str, Any]], str, float]:
    """Return deterministic defect specifications, state name, and severity for (segment_id, day)."""
    segment_id = segment_id.strip().upper()
    day = int(day)
    along_base = DETERMINISTIC_SEGMENT_LOCATIONS.get(segment_id, {}).get("along_m", 40.0)

    # -------------------------------------------------------------------------
    # SEG_001: Progressive Fatigue (Wheeltrack Cracking -> Severe Waterlogged Crater)
    # -------------------------------------------------------------------------
    if segment_id == "SEG_001":
        if day == 1:
            return [], "Pristine Asphalt", 0.02
        elif day == 2:
            return [
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.35, "width_m": 0.03, "depth_m": 0.004}, "true_severity_score": 0.08}
            ], "Subtle Surface Wear", 0.08
        elif day == 3:
            return [
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.65, "width_m": 0.04, "depth_m": 0.004}, "true_severity_score": 0.16}
            ], "Hairline Crack Extension", 0.16
        elif day == 4:
            return [
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.95, "width_m": 0.06, "depth_m": 0.004}, "true_severity_score": 0.28},
                {"defect_id": "SEG_001_C2", "defect_type": "crack", "along_m": along_base - 0.2, "across_m": -1.75, "dimensions": {"length_m": 0.55, "width_m": 0.04, "orientation_deg": 65.0}, "true_severity_score": 0.28}
            ], "Longitudinal & Branch Crack", 0.28
        elif day == 5:
            return [
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 1.30, "width_m": 0.08, "depth_m": 0.004}, "true_severity_score": 0.44},
                {"defect_id": "SEG_001_C2", "defect_type": "crack", "along_m": along_base - 0.2, "across_m": -1.75, "dimensions": {"length_m": 0.85, "width_m": 0.06, "orientation_deg": 65.0}, "true_severity_score": 0.44},
                {"defect_id": "SEG_001_S1", "defect_type": "pothole", "along_m": along_base + 0.2, "across_m": -1.8, "dimensions": {"length_m": 0.38, "width_m": 0.35, "depth_m": 0.006}, "true_severity_score": 0.44}
            ], "Visible Fatigue & Alligator Cracking", 0.44
        elif day == 6:
            return [
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 1.45, "width_m": 0.09, "depth_m": 0.004}, "true_severity_score": 0.58},
                {"defect_id": "SEG_001_P1", "defect_type": "pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.55, "width_m": 0.50, "depth_m": 0.04}, "true_severity_score": 0.58}
            ], "Crack Enlargement & Incipient Pothole", 0.58
        elif day == 7:
            return [
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 1.55, "width_m": 0.09, "depth_m": 0.004}, "true_severity_score": 0.68},
                {"defect_id": "SEG_001_P1", "defect_type": "pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.75, "width_m": 0.68, "depth_m": 0.06}, "true_severity_score": 0.68}
            ], "Small Dry Pothole Cavity", 0.68
        elif day == 8:
            return [
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 1.65, "width_m": 0.09, "depth_m": 0.004}, "true_severity_score": 0.78},
                {"defect_id": "SEG_001_P1", "defect_type": "pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.95, "width_m": 0.85, "depth_m": 0.08}, "true_severity_score": 0.78},
                {"defect_id": "SEG_001_P2", "defect_type": "pothole", "along_m": along_base - 0.8, "across_m": -1.85, "dimensions": {"length_m": 0.40, "width_m": 0.35, "depth_m": 0.03}, "true_severity_score": 0.78}
            ], "Larger Dry Pothole Cavity", 0.78
        elif day == 9:
            return [
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": -1.75, "dimensions": {"length_m": 1.10, "width_m": 0.09, "orientation_deg": -20.0}, "true_severity_score": 0.86},
                {"defect_id": "SEG_001_P1", "defect_type": "pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 1.15, "width_m": 1.05, "depth_m": 0.09}, "true_severity_score": 0.86},
                {"defect_id": "SEG_001_P2", "defect_type": "pothole", "along_m": along_base - 0.8, "across_m": -1.85, "dimensions": {"length_m": 0.45, "width_m": 0.40, "depth_m": 0.04}, "true_severity_score": 0.86}
            ], "Severe Pothole", 0.86
        else: # Day 10
            return [
                {"defect_id": "SEG_001_W1", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 1.40, "width_m": 1.20, "depth_m": 0.10}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.90}, "true_severity_score": 0.95},
                {"defect_id": "SEG_001_C1", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": -1.75, "dimensions": {"length_m": 1.10, "width_m": 0.09, "orientation_deg": -20.0}, "true_severity_score": 0.95},
                {"defect_id": "SEG_001_W2", "defect_type": "water_filled_pothole", "along_m": along_base - 0.8, "across_m": -1.85, "dimensions": {"length_m": 0.50, "width_m": 0.45, "depth_m": 0.04}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.50}, "true_severity_score": 0.95}
            ], "Critical Hazard: Severe Waterlogged Crater", 0.95

    # -------------------------------------------------------------------------
    # SEG_002: Rapid Pothole Formation (Stripping -> Raveling -> Large Wet Pothole)
    # -------------------------------------------------------------------------
    elif segment_id == "SEG_002":
        if day <= 1:
            return [], "Pristine Asphalt", 0.02
        elif day == 2:
            return [
                {"defect_id": "SEG_002_S1", "defect_type": "pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 0.28, "width_m": 0.25, "depth_m": 0.008}, "true_severity_score": 0.10}
            ], "Aggregate Stripping", 0.10
        elif day == 3:
            return [
                {"defect_id": "SEG_002_S1", "defect_type": "pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 0.45, "width_m": 0.40, "depth_m": 0.02}, "true_severity_score": 0.22}
            ], "Surface Raveling & Depression", 0.22
        elif day == 4:
            return [
                {"defect_id": "SEG_002_P1", "defect_type": "pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 0.60, "width_m": 0.55, "depth_m": 0.04}, "true_severity_score": 0.40}
            ], "Incipient Pothole", 0.40
        elif day == 5:
            return [
                {"defect_id": "SEG_002_P1", "defect_type": "pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 0.80, "width_m": 0.72, "depth_m": 0.06}, "true_severity_score": 0.58}
            ], "Distinct Dry Pothole", 0.58
        elif day == 6:
            return [
                {"defect_id": "SEG_002_W1", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 0.95, "width_m": 0.85, "depth_m": 0.07}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.50}, "true_severity_score": 0.72}
            ], "Early Water Accumulation Pothole", 0.72
        elif day == 7:
            return [
                {"defect_id": "SEG_002_W1", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 1.10, "width_m": 0.98, "depth_m": 0.08}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.70}, "true_severity_score": 0.80}
            ], "Expanding Wet Pothole", 0.80
        elif day == 8:
            return [
                {"defect_id": "SEG_002_W1", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 1.25, "width_m": 1.10, "depth_m": 0.09}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.80}, "true_severity_score": 0.86}
            ], "Large Waterlogged Pothole", 0.86
        elif day == 9:
            return [
                {"defect_id": "SEG_002_W1", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 1.40, "width_m": 1.25, "depth_m": 0.10}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.90}, "true_severity_score": 0.91}
            ], "Deep Waterlogged Crater", 0.91
        else: # Day 10
            return [
                {"defect_id": "SEG_002_W1", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": 1.8, "dimensions": {"length_m": 1.55, "width_m": 1.40, "depth_m": 0.12}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.95}, "true_severity_score": 0.96}
            ], "Severe Waterlogged Puddle Hazard", 0.96

    # -------------------------------------------------------------------------
    # SEG_003: Crack-Dominated Deterioration (Transverse -> Block Cracking -> Edge Spalling)
    # -------------------------------------------------------------------------
    elif segment_id == "SEG_003":
        if day <= 1:
            return [], "Pristine Asphalt", 0.02
        elif day == 2:
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 1.50, "width_m": 0.03, "orientation_deg": 90.0}, "true_severity_score": 0.10}
            ], "Transverse Crack Onset", 0.10
        elif day == 3:
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 2.80, "width_m": 0.04, "orientation_deg": 90.0}, "true_severity_score": 0.20}
            ], "Transverse Crack Extension", 0.20
        elif day == 4:
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 3.20, "width_m": 0.05, "orientation_deg": 90.0}, "true_severity_score": 0.32},
                {"defect_id": "SEG_003_C2", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": 0.0, "dimensions": {"length_m": 2.20, "width_m": 0.04, "orientation_deg": 90.0}, "true_severity_score": 0.32}
            ], "Parallel Transverse Cracks", 0.32
        elif day == 5:
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 3.50, "width_m": 0.06, "orientation_deg": 90.0}, "true_severity_score": 0.45},
                {"defect_id": "SEG_003_C2", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": 0.0, "dimensions": {"length_m": 3.00, "width_m": 0.05, "orientation_deg": 90.0}, "true_severity_score": 0.45},
                {"defect_id": "SEG_003_C3", "defect_type": "crack", "along_m": along_base + 0.4, "across_m": -1.0, "dimensions": {"length_m": 0.80, "width_m": 0.04, "orientation_deg": 0.0}, "true_severity_score": 0.45}
            ], "Transverse & Longitudinal Crack Network", 0.45
        elif day == 6:
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 3.80, "width_m": 0.07, "orientation_deg": 90.0}, "true_severity_score": 0.55},
                {"defect_id": "SEG_003_C2", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": 0.0, "dimensions": {"length_m": 3.20, "width_m": 0.06, "orientation_deg": 90.0}, "true_severity_score": 0.55},
                {"defect_id": "SEG_003_C3", "defect_type": "crack", "along_m": along_base + 0.4, "across_m": -1.0, "dimensions": {"length_m": 0.80, "width_m": 0.05, "orientation_deg": 0.0}, "true_severity_score": 0.55},
                {"defect_id": "SEG_003_C4", "defect_type": "crack", "along_m": along_base + 0.4, "across_m": 1.0, "dimensions": {"length_m": 0.80, "width_m": 0.05, "orientation_deg": 0.0}, "true_severity_score": 0.55}
            ], "Block Cracking Onset", 0.55
        elif day == 7:
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 4.00, "width_m": 0.08, "orientation_deg": 90.0}, "true_severity_score": 0.65},
                {"defect_id": "SEG_003_C2", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": 0.0, "dimensions": {"length_m": 3.50, "width_m": 0.07, "orientation_deg": 90.0}, "true_severity_score": 0.65},
                {"defect_id": "SEG_003_C3", "defect_type": "crack", "along_m": along_base + 0.4, "across_m": -1.2, "dimensions": {"length_m": 0.80, "width_m": 0.06, "orientation_deg": 0.0}, "true_severity_score": 0.65},
                {"defect_id": "SEG_003_C4", "defect_type": "crack", "along_m": along_base + 0.4, "across_m": 1.2, "dimensions": {"length_m": 0.80, "width_m": 0.06, "orientation_deg": 0.0}, "true_severity_score": 0.65}
            ], "Extensive Block Cracking", 0.65
        elif day == 8:
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 4.20, "width_m": 0.09, "orientation_deg": 90.0}, "true_severity_score": 0.72},
                {"defect_id": "SEG_003_C2", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": 0.0, "dimensions": {"length_m": 3.80, "width_m": 0.08, "orientation_deg": 90.0}, "true_severity_score": 0.72},
                {"defect_id": "SEG_003_S1", "defect_type": "pothole", "along_m": along_base + 0.4, "across_m": -1.2, "dimensions": {"length_m": 0.35, "width_m": 0.30, "depth_m": 0.02}, "true_severity_score": 0.72}
            ], "Widened Block Cracking & Edge Spall", 0.72
        elif day == 9:
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 4.50, "width_m": 0.10, "orientation_deg": 90.0}, "true_severity_score": 0.78},
                {"defect_id": "SEG_003_C2", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": 0.0, "dimensions": {"length_m": 4.00, "width_m": 0.09, "orientation_deg": 90.0}, "true_severity_score": 0.78},
                {"defect_id": "SEG_003_S1", "defect_type": "pothole", "along_m": along_base + 0.4, "across_m": -1.2, "dimensions": {"length_m": 0.45, "width_m": 0.40, "depth_m": 0.03}, "true_severity_score": 0.78}
            ], "Heavy Block Cracking & Edge Spalling", 0.78
        else: # Day 10
            return [
                {"defect_id": "SEG_003_C1", "defect_type": "crack", "along_m": along_base, "across_m": 0.0, "dimensions": {"length_m": 4.80, "width_m": 0.12, "orientation_deg": 90.0}, "true_severity_score": 0.83},
                {"defect_id": "SEG_003_C2", "defect_type": "crack", "along_m": along_base + 0.8, "across_m": 0.0, "dimensions": {"length_m": 4.20, "width_m": 0.10, "orientation_deg": 90.0}, "true_severity_score": 0.83},
                {"defect_id": "SEG_003_S1", "defect_type": "pothole", "along_m": along_base + 0.4, "across_m": -1.2, "dimensions": {"length_m": 0.55, "width_m": 0.48, "depth_m": 0.04}, "true_severity_score": 0.83},
                {"defect_id": "SEG_003_S2", "defect_type": "pothole", "along_m": along_base + 0.4, "across_m": 1.2, "dimensions": {"length_m": 0.45, "width_m": 0.40, "depth_m": 0.03}, "true_severity_score": 0.83}
            ], "Severe Block Cracking & Edge Dislodgement", 0.83

    # -------------------------------------------------------------------------
    # SEG_004: Cluster Potholes (Multi-Epicenter Potholes in Proximity)
    # -------------------------------------------------------------------------
    elif segment_id == "SEG_004":
        if day <= 1:
            return [], "Pristine Asphalt", 0.02
        elif day == 2:
            return [
                {"defect_id": "SEG_004_P1", "defect_type": "pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 0.30, "width_m": 0.28, "depth_m": 0.01}, "true_severity_score": 0.12}
            ], "Micro Spall 1", 0.12
        elif day == 3:
            return [
                {"defect_id": "SEG_004_P1", "defect_type": "pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 0.40, "width_m": 0.35, "depth_m": 0.02}, "true_severity_score": 0.25},
                {"defect_id": "SEG_004_P2", "defect_type": "pothole", "along_m": along_base + 0.6, "across_m": -1.2, "dimensions": {"length_m": 0.32, "width_m": 0.28, "depth_m": 0.01}, "true_severity_score": 0.25}
            ], "Micro Spall 1 & 2", 0.25
        elif day == 4:
            return [
                {"defect_id": "SEG_004_P1", "defect_type": "pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 0.55, "width_m": 0.48, "depth_m": 0.04}, "true_severity_score": 0.38},
                {"defect_id": "SEG_004_P2", "defect_type": "pothole", "along_m": along_base + 0.6, "across_m": -1.2, "dimensions": {"length_m": 0.45, "width_m": 0.40, "depth_m": 0.03}, "true_severity_score": 0.38}
            ], "Dual Incipient Potholes", 0.38
        elif day == 5:
            return [
                {"defect_id": "SEG_004_P1", "defect_type": "pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 0.70, "width_m": 0.60, "depth_m": 0.05}, "true_severity_score": 0.52},
                {"defect_id": "SEG_004_P2", "defect_type": "pothole", "along_m": along_base + 0.6, "across_m": -1.2, "dimensions": {"length_m": 0.58, "width_m": 0.50, "depth_m": 0.04}, "true_severity_score": 0.52},
                {"defect_id": "SEG_004_P3", "defect_type": "pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.35, "width_m": 0.30, "depth_m": 0.02}, "true_severity_score": 0.52}
            ], "Multiple Small Potholes", 0.52
        elif day == 6:
            return [
                {"defect_id": "SEG_004_P1", "defect_type": "pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 0.85, "width_m": 0.75, "depth_m": 0.07}, "true_severity_score": 0.66},
                {"defect_id": "SEG_004_P2", "defect_type": "pothole", "along_m": along_base + 0.6, "across_m": -1.2, "dimensions": {"length_m": 0.72, "width_m": 0.65, "depth_m": 0.05}, "true_severity_score": 0.66},
                {"defect_id": "SEG_004_P3", "defect_type": "pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.48, "width_m": 0.42, "depth_m": 0.03}, "true_severity_score": 0.66}
            ], "Triple Pothole Cluster", 0.66
        elif day == 7:
            return [
                {"defect_id": "SEG_004_P1", "defect_type": "pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 1.00, "width_m": 0.88, "depth_m": 0.08}, "true_severity_score": 0.76},
                {"defect_id": "SEG_004_P2", "defect_type": "pothole", "along_m": along_base + 0.6, "across_m": -1.2, "dimensions": {"length_m": 0.85, "width_m": 0.75, "depth_m": 0.06}, "true_severity_score": 0.76},
                {"defect_id": "SEG_004_P3", "defect_type": "pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.60, "width_m": 0.52, "depth_m": 0.04}, "true_severity_score": 0.76}
            ], "Expanding Pothole Cluster", 0.76
        elif day == 8:
            return [
                {"defect_id": "SEG_004_P1", "defect_type": "pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 1.15, "width_m": 1.00, "depth_m": 0.09}, "true_severity_score": 0.84},
                {"defect_id": "SEG_004_P2", "defect_type": "pothole", "along_m": along_base + 0.6, "across_m": -1.2, "dimensions": {"length_m": 0.98, "width_m": 0.85, "depth_m": 0.07}, "true_severity_score": 0.84},
                {"defect_id": "SEG_004_P3", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.72, "width_m": 0.62, "depth_m": 0.05}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.60}, "true_severity_score": 0.84}
            ], "Clustered Deep Potholes", 0.84
        elif day == 9:
            return [
                {"defect_id": "SEG_004_P1", "defect_type": "pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 1.30, "width_m": 1.12, "depth_m": 0.10}, "true_severity_score": 0.90},
                {"defect_id": "SEG_004_P2", "defect_type": "water_filled_pothole", "along_m": along_base + 0.6, "across_m": -1.2, "dimensions": {"length_m": 1.10, "width_m": 0.95, "depth_m": 0.08}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.70}, "true_severity_score": 0.90},
                {"defect_id": "SEG_004_P3", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.85, "width_m": 0.75, "depth_m": 0.06}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.80}, "true_severity_score": 0.90}
            ], "Severe Clustered Potholes", 0.90
        else: # Day 10
            return [
                {"defect_id": "SEG_004_W1", "defect_type": "water_filled_pothole", "along_m": along_base - 0.5, "across_m": -1.5, "dimensions": {"length_m": 1.45, "width_m": 1.25, "depth_m": 0.11}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.85}, "true_severity_score": 0.95},
                {"defect_id": "SEG_004_W2", "defect_type": "water_filled_pothole", "along_m": along_base + 0.6, "across_m": -1.2, "dimensions": {"length_m": 1.22, "width_m": 1.08, "depth_m": 0.09}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.80}, "true_severity_score": 0.95},
                {"defect_id": "SEG_004_W3", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": -1.8, "dimensions": {"length_m": 0.95, "width_m": 0.82, "depth_m": 0.07}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.90}, "true_severity_score": 0.95}
            ], "Critical Hazard: Multi-Pothole Cluster", 0.95

    # -------------------------------------------------------------------------
    # SEG_005: High-Resilience Control Segment (Max Severity <= 0.35)
    # -------------------------------------------------------------------------
    elif segment_id == "SEG_005":
        if day <= 4:
            return [], "Pristine Asphalt", round(0.01 + day * 0.01, 2)
        elif day <= 7:
            return [
                {"defect_id": "SEG_005_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.35, "dimensions": {"length_m": 0.30 + (day - 5) * 0.1, "width_m": 0.02, "depth_m": 0.003}, "true_severity_score": round(0.08 + (day - 5) * 0.04, 2)}
            ], "Faint Surface Wear", round(0.08 + (day - 5) * 0.04, 2)
        else: # Day 8-10
            return [
                {"defect_id": "SEG_005_C1", "defect_type": "crack", "along_m": along_base, "across_m": -1.35, "dimensions": {"length_m": 0.55 + (day - 8) * 0.1, "width_m": 0.03, "depth_m": 0.004}, "true_severity_score": round(0.22 + (day - 8) * 0.04, 2)}
            ], "Minor Hairline Crack", round(0.22 + (day - 8) * 0.04, 2)

    # -------------------------------------------------------------------------
    # SEG_006: Curved Highway Edge Shear Damage
    # -------------------------------------------------------------------------
    elif segment_id == "SEG_006":
        if day <= 1:
            return [], "Pristine Asphalt", 0.02
        elif day == 2:
            return [
                {"defect_id": "SEG_006_C1", "defect_type": "crack", "along_m": along_base, "across_m": 6.2, "dimensions": {"length_m": 1.20, "width_m": 0.05, "orientation_deg": 15.0}, "true_severity_score": 0.12}
            ], "Outer Lane Edge Wear", 0.12
        elif day == 3:
            return [
                {"defect_id": "SEG_006_C1", "defect_type": "crack", "along_m": along_base, "across_m": 6.2, "dimensions": {"length_m": 1.80, "width_m": 0.06, "orientation_deg": 15.0}, "true_severity_score": 0.22}
            ], "Edge Shear Crack", 0.22
        elif day == 4:
            return [
                {"defect_id": "SEG_006_P1", "defect_type": "pothole", "along_m": along_base, "across_m": 6.0, "dimensions": {"length_m": 2.00, "width_m": 0.25, "depth_m": 0.03}, "true_severity_score": 0.34}
            ], "Outer Curve Shear Gouge Onset", 0.34
        elif day == 5:
            return [
                {"defect_id": "SEG_006_P1", "defect_type": "pothole", "along_m": along_base, "across_m": 6.0, "dimensions": {"length_m": 2.30, "width_m": 0.40, "depth_m": 0.05}, "true_severity_score": 0.48}
            ], "Shear Deformation & Edge Breakup", 0.48
        elif day == 6:
            return [
                {"defect_id": "SEG_006_P1", "defect_type": "pothole", "along_m": along_base, "across_m": 6.0, "dimensions": {"length_m": 2.50, "width_m": 0.55, "depth_m": 0.07}, "true_severity_score": 0.62}
            ], "Deep Edge Gouge", 0.62
        elif day == 7:
            return [
                {"defect_id": "SEG_006_P1", "defect_type": "pothole", "along_m": along_base, "across_m": 6.0, "dimensions": {"length_m": 2.70, "width_m": 0.70, "depth_m": 0.08}, "true_severity_score": 0.72}
            ], "Expanding Edge Crater", 0.72
        elif day == 8:
            return [
                {"defect_id": "SEG_006_P1", "defect_type": "pothole", "along_m": along_base, "across_m": 6.0, "dimensions": {"length_m": 2.90, "width_m": 0.85, "depth_m": 0.09}, "true_severity_score": 0.80}
            ], "Deep Edge Crater Basin", 0.80
        elif day == 9:
            return [
                {"defect_id": "SEG_006_W1", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": 6.0, "dimensions": {"length_m": 3.10, "width_m": 0.95, "depth_m": 0.10}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.75}, "true_severity_score": 0.88}
            ], "Severe Edge Crater Hazard", 0.88
        else: # Day 10
            return [
                {"defect_id": "SEG_006_W1", "defect_type": "water_filled_pothole", "along_m": along_base, "across_m": 6.0, "dimensions": {"length_m": 3.30, "width_m": 1.10, "depth_m": 0.12}, "water_state": {"is_water_filled": True, "water_coverage_frac": 0.90}, "true_severity_score": 0.94}
            ], "Critical Hazard: Severe Curve Edge Crater", 0.94

    return [], "Unknown State", 0.0


def materialize_segment_day(segment_id: str, day: int) -> Dict[str, Any]:
    """Materialise a specific (segment_id, day) deterministic road deterioration state.

    1. Removes previous temporal defect actors.
    2. Spawns the exact deterministic defect meshes and water overlays for (segment_id, day).
    3. Moves the Unreal Editor viewport camera to the segment's fixed downward top-down pose.
    4. Sets camera height Z = 12m, pitch = -89°, FOV = 70° so road fills ~95% of image width.
    """
    if not HAS_UNREAL:
        raise RuntimeError("materialize_segment_day must run inside Unreal Engine Python")

    segment_id = str(segment_id).strip().upper()
    if segment_id not in DETERMINISTIC_SEGMENT_LOCATIONS:
        raise ValueError(f"Unknown segment_id {segment_id!r}")
    day = int(day)
    if not 1 <= day <= 10:
        raise ValueError(f"day must be between 1 and 10, got {day!r}")

    _CURRENT_SELECTED_STATE["segment_id"] = segment_id
    _CURRENT_SELECTED_STATE["day"] = day

    loc_info = DETERMINISTIC_SEGMENT_LOCATIONS[segment_id]
    along_m = loc_info["along_m"]
    across_m = loc_info["across_m"]

    x_m, y_m, z_m, yaw_deg = _temporal_route_pose(along_m, across_m)

    cam_pose = {
        "x_cm": round(x_m * 100.0, 2),
        "y_cm": round(y_m * 100.0, 2),
        "z_cm": round((z_m + 12.0) * 100.0, 2),
        "pitch_deg": -89.0,
        "yaw_deg": round(yaw_deg, 3),
        "roll_deg": 0.0,
        "fov_deg": 70.0
    }

    # Move viewport camera
    try:
        cam_loc = unreal.Vector(cam_pose["x_cm"], cam_pose["y_cm"], cam_pose["z_cm"])
        cam_rot = unreal.Rotator(roll=0.0, pitch=cam_pose["pitch_deg"], yaw=cam_pose["yaw_deg"])
        unreal.EditorLevelLibrary.set_level_viewport_camera_info(cam_loc, cam_rot)
    except Exception as exc:
        print(f"Error setting viewport camera: {exc}")

    # Clear previous defect actors
    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
    for a in all_actors:
        if a and (a.actor_has_tag(unreal.Name("RS_Defect")) or a.actor_has_tag(unreal.Name("RS_TemporalDefect"))):
            unreal.EditorLevelLibrary.destroy_actor(a)

    # Check if road baseline exists; if not, spawn it
    has_road = any(a and a.actor_has_tag(unreal.Name("RS_Road")) for a in all_actors)
    if not has_road:
        spawn_full_world(
            road_health="Pristine (Grade A)",
            lighting_preset="Clear Noon (70° Sun)",
            density_per_100m2=0.0,
            random_seed=2026
        )
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        for a in all_actors:
            if a and a.get_name() == "Floor":
                unreal.EditorLevelLibrary.destroy_actor(a)
            elif a and a.actor_has_tag(unreal.Name("RS_Marking")):
                l_m = a.get_actor_location()
                s_m = a.get_actor_scale3d()
                a.set_actor_location(unreal.Vector(l_m.x, l_m.y, 2.0), False, False)
                a.set_actor_scale3d(unreal.Vector(s_m.x, s_m.y, 0.04))

    defects_spec, state_name, severity_val = get_deterministic_segment_defects(segment_id, day)
    materials = _temporal_materials()

    spawned_defects = []
    for raw in defects_spec:
        raw["road_segment_id"] = segment_id
        gt = _spawn_temporal_defect(raw, materials, day)
        spawned_defects.append(gt)

    segment_state = {
        "road_segment_id": segment_id,
        "day": day,
        "deterioration_state": state_name,
        "renderer_condition_score": severity_val,
        "persistent_location": {"along_m": along_m, "across_m": across_m},
        "world_location_cm": {"x": round(x_m * 100.0, 2), "y": round(y_m * 100.0, 2), "z": round(z_m * 100.0, 2)},
        "camera_pose": cam_pose,
        "defects": spawned_defects
    }

    _TEMPORAL_STATE["segments"][segment_id] = segment_state
    _TEMPORAL_STATE["day"] = day
    _TEMPORAL_STATE["ground_truth"] = spawned_defects
    _TEMPORAL_STATE["camera_config"] = {"altitude_m": 12.0, "pitch_deg": -89.0, "fov_deg": 70.0, "width": 1920, "height": 1080}

    try:
        unreal.log(f"[RoadSentinel] Materialised {segment_id} Day {day:02d} ({state_name}): {len(spawned_defects)} defects, severity {severity_val:.2f}")
    except Exception:
        print(f"[RoadSentinel] Materialised {segment_id} Day {day:02d} ({state_name}): {len(spawned_defects)} defects, severity {severity_val:.2f}")

    return segment_state



def materialize_temporal_manifest(manifest_path: str, ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
    """Apply one deterministic temporal day to the native UE road scene.

    This function consumes the condition manifest directly.  It never invokes
    the interactive random scatter generator for temporal defects, ensuring
    that SEG_001, etc. retain both their ID and physical coordinates through
    all 20 days.
    """
    if not HAS_UNREAL:
        raise RuntimeError("materialize_temporal_manifest must be run by Unreal Engine Python")
    resolved_manifest = _resolve_temporal_path(manifest_path, "temporal manifest")
    if not resolved_manifest.is_file():
        raise FileNotFoundError(f"Temporal manifest does not exist: {resolved_manifest}")
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "RoadSentinelTemporalManifest/v1":
        raise ValueError("Unsupported temporal manifest schema")
    day = int(_finite_number(payload.get("day"), "day", 1.0, 20.0))
    raw_segments = payload.get("segments", [])
    raw_defects = payload.get("defects", [])
    if not isinstance(raw_segments, list) or not isinstance(raw_defects, list):
        raise ValueError("Temporal manifest segments and defects must be arrays")

    camera_config = _temporal_capture_config(payload)
    segment_state: Dict[str, Dict[str, Any]] = {}
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ValueError("Each temporal segment must be an object")
        segment_id = str(raw.get("road_segment_id", "")).strip()
        if not segment_id or segment_id in segment_state:
            raise ValueError("Temporal segment ids must be non-empty and unique")
        along_m = _finite_number(raw.get("along_m"), f"{segment_id}.along_m", 0.0, 420.0)
        across_m = _finite_number(raw.get("across_m"), f"{segment_id}.across_m", -7.5, 7.5)
        x_m, y_m, z_m, yaw_deg = _temporal_route_pose(along_m, across_m)
        segment_state[segment_id] = {
            "road_segment_id": segment_id,
            "persistent_location": {"along_m": along_m, "across_m": across_m},
            "world_location_cm": {"x": round(x_m * 100.0, 2), "y": round(y_m * 100.0, 2), "z": round(z_m * 100.0, 2)},
            "camera_pose": {
                "x_cm": round(x_m * 100.0, 2),
                "y_cm": round(y_m * 100.0, 2),
                "z_cm": round((z_m + camera_config["altitude_m"]) * 100.0, 2),
                "pitch_deg": camera_config["pitch_deg"],
                "yaw_deg": round(yaw_deg, 3),
                "roll_deg": 0.0,
            },
            "renderer_condition_score": raw.get("renderer_condition_score"),
        }

    # Rebuild exactly the same UE road and lighting each day with *zero*
    # random defects.  Only the following manifest-driven meshes differ.
    spawn_full_world(
        road_health="Pristine (Grade A)",
        lighting_preset="Clear Noon (70° Sun)",
        size_profile="Multi-Scale Organic (Mixed)",
        density_per_100m2=0.0,
        water_ratio="Mixed Wet/Dry",
        random_seed=2026,
    )
    materials = _temporal_materials()
    ground_truth: List[Dict[str, Any]] = []
    for raw in raw_defects:
        if not isinstance(raw, dict):
            raise ValueError("Each temporal defect must be an object")
        if str(raw.get("road_segment_id", "")) not in segment_state:
            raise ValueError("Every temporal defect must refer to a declared segment")
        ground_truth.append(_spawn_temporal_defect(raw, materials, day))

    _TEMPORAL_STATE.update({
        "day": day,
        "manifest_path": str(resolved_manifest),
        "segments": segment_state,
        "ground_truth": ground_truth,
        "camera_config": camera_config,
    })
    result = {
        "schema_version": "RoadSentinelUnrealTemporalGroundTruth/v1",
        "engine": "Unreal Engine native scene",
        "day": day,
        "manifest_path": str(resolved_manifest),
        "segments": list(segment_state.values()),
        "defects": ground_truth,
        "camera_config": camera_config,
    }
    if ground_truth_path:
        resolved_ground_truth = _resolve_temporal_path(ground_truth_path, "ground-truth")
        resolved_ground_truth.parent.mkdir(parents=True, exist_ok=True)
        resolved_ground_truth.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["ground_truth_path"] = str(resolved_ground_truth)
    unreal.log(f"[RoadSentinel] Materialised temporal day {day:02d}: {len(ground_truth)} manifest defects.")
    return result


# ==============================================================================
# 7. Drone Camera Navigation & Photo Capture ('C')
# ==============================================================================

DRONE_VIEWPOINTS = {
    "🔭 Overhead Drone Survey (SAM 2 Top-Down)": {
        "location": (4500.0, 0.0, 2500.0),
        "rotation": (0.0, -89.0, 0.0),
        "desc": "High altitude orthographic survey view"
    },
    "🔍 Low-Angle Pothole Inspection (30° Close-Up)": {
        "location": (3200.0, -220.0, 320.0),
        "rotation": (0.0, -28.0, 20.0),
        "desc": "Close inspection of broken asphalt crater"
    },
    "💧 Waterlogged Pothole Macro View": {
        "location": (3800.0, -180.0, 180.0),
        "rotation": (0.0, -42.0, 45.0),
        "desc": "Direct macro view of sky reflection in puddle"
    },
    "🛣 Highway Cruise View (Forward Drone)": {
        "location": (1200.0, -190.0, 220.0),
        "rotation": (0.0, -6.0, 0.0),
        "desc": "Forward flying view along highway lane"
    },
    "🌄 Highway Curve Vantage Overlook": {
        "location": (24000.0, -1500.0, 1800.0),
        "rotation": (0.0, -15.0, 35.0),
        "desc": "Panoramic overview of the curved road & edge of town"
    },
}


def teleport_drone_camera(preset_name: str):
    """Flies/teleports the active Unreal viewport camera to the chosen vantage point."""
    if not HAS_UNREAL:
        return
    vp = DRONE_VIEWPOINTS.get(preset_name)
    if not vp:
        return
    loc = unreal.Vector(*vp["location"])
    rot = unreal.Rotator(roll=vp["rotation"][0], pitch=vp["rotation"][1], yaw=vp["rotation"][2])
    try:
        unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)
    except Exception as e:
        print(f"Error setting camera info: {e}")


def capture_drone_photo() -> str:
    """Captures a photo from the current drone/inspection camera ('C' key)."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if HAS_UNREAL and _CURRENT_SELECTED_STATE.get("segment_id"):
        seg_id = _CURRENT_SELECTED_STATE["segment_id"]
        day_num = _CURRENT_SELECTED_STATE["day"]
        try:
            res = _ipc_inspection_capture(day_num, seg_id)
            print(f"📸 RoadSentinel Inspection Photo Captured: {res.get('path')}")
            return res.get("path", "")
        except Exception as e:
            print(f"Error capturing inspection photo on 'C': {e}")

    capture_filename = f"drone_capture_{timestamp}.png"
    target_path = os.path.join(CAPTURES_DIR, capture_filename)

    camera_info = {}
    if HAS_UNREAL:
        try:
            loc, rot = unreal.EditorLevelLibrary.get_level_viewport_camera_info()
            camera_info = {
                "x_cm": loc.x, "y_cm": loc.y, "z_cm": loc.z,
                "pitch_deg": rot.pitch, "yaw_deg": rot.yaw, "roll_deg": rot.roll
            }
        except Exception:
            pass

        cmd = f"HighResShot 1920x1080 filename=\"{target_path}\""
        unreal.SystemLibrary.execute_console_command(None, cmd)
        unreal.SystemLibrary.execute_console_command(None, "Shot")

    log_file = os.path.join(CAPTURES_DIR, "drone_captures_log.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": timestamp,
            "filename": capture_filename,
            "path": target_path,
            "camera": camera_info
        }) + "\n")

    print(f"📸 Drone photo captured: {target_path}")
    return target_path


# ==============================================================================
# 7a. Inspection Capture Helper (SceneCapture2D, validate, metadata)
# ==============================================================================

def _ipc_inspection_capture(day: int, segment_id: str) -> dict:
    """Perform a full SceneCapture2D render for the given day+segment.

    Saves:
      env/output/temporal_segments/SEG_XXX/day_XX.png
      env/output/temporal_segments/SEG_XXX/segment_history.json
    """
    if not HAS_UNREAL:
        raise RuntimeError("_ipc_inspection_capture must run inside Unreal Engine Python")

    _helper_dir = os.path.join(WORKSPACE_ROOT, "env", "scripts")
    if _helper_dir not in sys.path:
        sys.path.insert(0, _helper_dir)
    import rs_inspection_capture as _ic

    segments = _TEMPORAL_STATE.get("segments", {})
    if not segments or segment_id not in segments:
        materialize_segment_day(segment_id, day)
        segments = _TEMPORAL_STATE.get("segments", {})

    seg_state = segments[segment_id]
    camera_pose = seg_state["camera_pose"]
    world_location_cm = seg_state["world_location_cm"]
    persistent_location = seg_state["persistent_location"]
    condition_score = seg_state.get("renderer_condition_score") or 0.0
    camera_config = _TEMPORAL_STATE.get("camera_config", {
        "altitude_m": 12.0, "pitch_deg": -89.0, "fov_deg": 70.0,
        "width": 1920, "height": 1080,
    })

    out_dir = _ic.build_output_dir(day, segment_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = _ic.build_image_path(day, segment_id)

    # ------- SceneCapture2D render -------
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
    except Exception:
        subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        world = subsystem.get_editor_world()
    if not world:
        raise RuntimeError("Unreal Editor world is not available for SceneCapture2D")

    # Create render target
    render_format = getattr(
        unreal.TextureRenderTargetFormat,
        "RTF_RGBA8_SRGB",
        unreal.TextureRenderTargetFormat.RTF_RGBA8,
    )
    clear = unreal.LinearColor(0.0, 0.0, 0.0, 1.0)
    try:
        target = unreal.RenderingLibrary.create_render_target2d(
            world, int(camera_config["width"]), int(camera_config["height"]),
            render_format, clear_color=clear, auto_generate_mips=False,
        )
    except TypeError:
        target = unreal.RenderingLibrary.create_render_target2d(
            world, int(camera_config["width"]), int(camera_config["height"]), render_format,
        )
    if not target:
        raise RuntimeError("Could not create SceneCapture2D render target for inspection")

    # Spawn SceneCapture2D at the segment's fixed camera pose
    loc = unreal.Vector(
        float(camera_pose["x_cm"]),
        float(camera_pose["y_cm"]),
        float(camera_pose["z_cm"]),
    )
    rot = unreal.Rotator(
        roll=float(camera_pose.get("roll_deg", 0.0)),
        pitch=float(camera_pose["pitch_deg"]),
        yaw=float(camera_pose["yaw_deg"]),
    )
    cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
    if not cam:
        raise RuntimeError("Could not spawn SceneCapture2D actor for inspection capture")

    try:
        comp = None
        try:
            comp = cam.get_capture_component2d()
        except Exception:
            pass
        if not comp:
            comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
        if not comp:
            raise RuntimeError("SceneCapture2D actor has no capture component")

        comp.set_editor_property("texture_target", target)
        comp.set_editor_property("fov_angle", float(camera_config["fov_deg"]))
        comp.set_editor_property("capture_every_frame", False)
        comp.set_editor_property("capture_on_movement", False)
        comp.set_editor_property("always_persist_rendering_state", True)
        comp.set_editor_property("inherit_main_view_camera_post_process_settings", True)
        try:
            comp.set_editor_property(
                "capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
            )
        except (AttributeError, TypeError):
            pass

        try:
            pps = comp.get_editor_property("post_process_settings")
            pps.set_editor_property("override_auto_exposure_bias", True)
            pps.set_editor_property("auto_exposure_bias", 2.6)
            comp.set_editor_property("post_process_settings", pps)
            comp.set_editor_property("post_process_blend_weight", 1.0)
        except Exception:
            pass

        comp.capture_scene()
        time.sleep(0.25)
        comp.capture_scene()

        unreal.RenderingLibrary.export_render_target(
            world, target, str(out_dir), raw_path.name,
        )

        deadline = time.time() + 20.0
        actual_path = raw_path
        alt_path = raw_path.with_name(raw_path.name + ".png")
        while time.time() < deadline:
            if raw_path.is_file() and raw_path.stat().st_size > 0:
                actual_path = raw_path
                break
            if alt_path.is_file() and alt_path.stat().st_size > 0:
                alt_path.replace(raw_path)
                actual_path = raw_path
                break
            time.sleep(0.10)
        else:
            raise RuntimeError(
                f"UE did not export inspection PNG within 20s: {raw_path}"
            )

    finally:
        try:
            unreal.EditorLevelLibrary.destroy_actor(cam)
        except Exception:
            pass

    val = _ic.validate_png(
        actual_path,
        expected_width=int(camera_config["width"]),
        expected_height=int(camera_config["height"]),
    )

    meta_path = _ic.write_metadata(
        output_dir=out_dir,
        day=day,
        segment_id=segment_id,
        camera_pose=camera_pose,
        world_coordinates={
            "x_cm": world_location_cm.get("x", 0.0),
            "y_cm": world_location_cm.get("y", 0.0),
            "z_cm": world_location_cm.get("z", 0.0),
            "along_m": persistent_location.get("along_m", 0.0),
            "across_m": persistent_location.get("across_m", 0.0),
        },
        condition_score=condition_score,
        image_filename=raw_path.name,
        extra={
            "capture_day": day,
            "deterioration_state": seg_state.get("deterioration_state", "observed_deterioration"),
            "defect_types": [d.get("defect_type") for d in seg_state.get("defects", [])],
            "total_defects_day": len(seg_state.get("defects", [])),
        },
    )

    unreal.log(
        f"[RoadSentinel] Inspection capture day {day:02d} {segment_id}: "
        f"{'VALID' if val['valid'] else 'INVALID – ' + str(val.get('reason'))}"
    )

    return {
        "status": "ok" if val["valid"] else "warning",
        "path": str(actual_path),
        "metadata_path": str(meta_path),
        "validated": val["valid"],
        "validation_reason": val.get("reason"),
        "file_size_bytes": val.get("file_size_bytes", 0),
    }


# ==============================================================================
# 7. Unreal Engine IPC Server (Receives commands from Studio GUI)
# ==============================================================================

import socket
import struct
import zlib
import select
import subprocess

IPC_HOST = "127.0.0.1"
IPC_PORT = 8899

_ipc_server_socket = None
_ipc_client_sockets = []
_ipc_tick_handle = None


def start_ipc_server():
    """Initializes non-blocking TCP server on 127.0.0.1:8899 for GUI commands."""
    global _ipc_server_socket, _ipc_tick_handle
    if _ipc_server_socket:
        return

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((IPC_HOST, IPC_PORT))
        s.listen(5)
        s.setblocking(False)
        _ipc_server_socket = s
        print(f"[✓] RoadSentinel IPC server listening on {IPC_HOST}:{IPC_PORT}")

        if HAS_UNREAL:
            _ipc_tick_handle = unreal.register_slate_post_tick_callback(process_ipc_commands)
    except Exception as e:
        print(f"[!] RoadSentinel IPC server error: {e}")


def process_ipc_commands(delta_time=0.0):
    """Processes incoming GUI commands on Unreal Engine's Slate thread."""
    global _ipc_server_socket, _ipc_client_sockets
    if not _ipc_server_socket:
        return

    try:
        rlist, _, _ = select.select([_ipc_server_socket] + _ipc_client_sockets, [], [], 0.0)
        for sock in rlist:
            if sock is _ipc_server_socket:
                client, addr = _ipc_server_socket.accept()
                client.setblocking(False)
                _ipc_client_sockets.append(client)
            else:
                try:
                    data = sock.recv(4096)
                    if not data:
                        _ipc_client_sockets.remove(sock)
                        sock.close()
                        continue

                    lines = data.decode("utf-8").strip().split("\n")
                    for line in lines:
                        if not line:
                            continue
                        msg = json.loads(line)
                        action = msg.get("action")
                        response = {"status": "ok", "action": action}

                        if action == "materialize_segment_day":
                            segment_id = str(msg.get("segment_id", "SEG_001")).strip().upper()
                            day = int(msg.get("day", 1))
                            try:
                                state = materialize_segment_day(segment_id, day)
                                response.update({
                                    "segment_id": segment_id,
                                    "day": day,
                                    "deterioration_state": state["deterioration_state"],
                                    "severity_value": state["renderer_condition_score"],
                                    "camera_pose": state["camera_pose"],
                                    "total_defects": len(state["defects"])
                                })
                            except Exception as exc:
                                response["status"] = "error"
                                response["error"] = str(exc)

                        elif action == "generate":
                            defects = spawn_full_world(
                                road_health=msg.get("road_health", "Moderate Deterioration (Grade C)"),
                                lighting_preset=msg.get("lighting", "Clear Noon (70° Sun)"),
                                size_profile=msg.get("sizing", "Multi-Scale Organic (Mixed)"),
                                density_per_100m2=float(msg.get("density", 3.5)),
                                water_ratio=msg.get("water", "Mixed Wet/Dry"),
                                random_seed=int(time.time()) % 10000
                            )
                            response["total_defects"] = len(defects)

                        elif action in {"capture", "inspection_capture"}:
                            segment_id = str(msg.get("segment_id", _CURRENT_SELECTED_STATE.get("segment_id", "SEG_001"))).strip().upper()
                            day = int(msg.get("day", _CURRENT_SELECTED_STATE.get("day", 1)))
                            try:
                                response.update(_ipc_inspection_capture(day, segment_id))
                            except Exception as exc:
                                response["status"] = "error"
                                response["error"] = str(exc)

                        elif action == "teleport":
                            teleport_drone_camera(msg.get("viewpoint", ""))

                        elif action == "lighting":
                            apply_environment_lighting(msg.get("preset", "Clear Noon (70° Sun)"))

                        elif action == "temporal_open":
                            day = int(msg.get("day", 1))
                            manifest_path = str(
                                Path(TEMPORAL_MANIFEST_ROOT) / f"day_{day:02d}.json"
                            )
                            try:
                                scene = materialize_temporal_manifest(manifest_path)
                                response["day"] = scene["day"]
                                response["segments"] = [
                                    {
                                        "segment_id": s["road_segment_id"],
                                        "condition_score": s.get("renderer_condition_score"),
                                        "camera_pose": s["camera_pose"],
                                        "world_location_cm": s["world_location_cm"],
                                        "persistent_location": s["persistent_location"],
                                    }
                                    for s in scene["segments"]
                                ]
                                response["camera_config"] = scene["camera_config"]
                                response["total_defects"] = len(scene["defects"])
                            except Exception as exc:
                                response["status"] = "error"
                                response["error"] = str(exc)

                        elif action == "temporal_goto_segment":
                            segment_id = str(msg.get("segment_id", "")).strip()
                            try:
                                segments = _TEMPORAL_STATE.get("segments", {})
                                if not segments or segment_id not in segments:
                                    materialize_segment_day(segment_id, _CURRENT_SELECTED_STATE.get("day", 1))
                                    segments = _TEMPORAL_STATE.get("segments", {})
                                pose = segments[segment_id]["camera_pose"]
                                loc = unreal.Vector(
                                    float(pose["x_cm"]),
                                    float(pose["y_cm"]),
                                    float(pose["z_cm"]),
                                )
                                rot = unreal.Rotator(
                                    roll=float(pose.get("roll_deg", 0.0)),
                                    pitch=float(pose["pitch_deg"]),
                                    yaw=float(pose["yaw_deg"]),
                                )
                                unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)
                                response["segment_id"] = segment_id
                                response["camera_pose"] = pose
                            except Exception as exc:
                                response["status"] = "error"
                                response["error"] = str(exc)

                        resp_bytes = (json.dumps(response) + "\n").encode("utf-8")
                        sock.sendall(resp_bytes)
                except Exception as ex:
                    pass
    except Exception:
        pass


def launch_external_gui():
    """Launches the PySide6 Studio GUI process using carla_env or system Python."""
    carla_python = "/home/nitin-nandakumar/Downloads/roadsentinel/carla_env/bin/python3"
    sys_python = "/usr/bin/python3"
    chosen_python = carla_python if os.path.exists(carla_python) else sys_python

    gui_script = "/home/nitin-nandakumar/Downloads/roadsentinel/env/scripts/rs_interactive_studio.py"
    if os.path.exists(gui_script):
        try:
            subprocess.Popen([chosen_python, gui_script])
            print(f"[✓] Launched RoadSentinel Studio GUI with {chosen_python}")
        except Exception as e:
            print(f"[!] Could not launch GUI process: {e}")


# ==============================================================================
# 8. Main Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RoadSentinel 3D Studio & Defect Generator")
    parser.add_argument("--headless", action="store_true", help="Run procedural scatter without launching GUI")
    parser.add_argument("--road-health", type=str, default="Moderate Deterioration (Grade C)")
    parser.add_argument("--lighting", type=str, default="Clear Noon (70° Sun)")
    parser.add_argument("--density", type=float, default=3.5)
    args = parser.parse_args()

    # 1. Generate initial baseline world
    defects = spawn_full_world(
        road_health=args.road_health,
        lighting_preset=args.lighting,
        density_per_100m2=args.density
    )

    if HAS_UNREAL:
        # Default starting camera: Low-angle close-up of pothole cluster
        teleport_drone_camera("🔍 Low-Angle Pothole Inspection (30° Close-Up)")
        # 2. Start IPC server for interactive GUI control
        start_ipc_server()
        if not args.headless:
            # 3. Launch interactive PySide6 Studio GUI window
            launch_external_gui()
            unreal.log("=======================================================================")
            unreal.log(" [✓] RoadSentinel 3D Studio GUI opened!")
            unreal.log("     - Use dropdowns to tweak Lighting, Road Health, & Sizing.")
            unreal.log("     - Press 'C' anytime to capture high-res drone photos.")
            unreal.log("=======================================================================")
    else:
        print(f"Generated {len(defects)} defects in standalone mode.")


if __name__ == "__main__":
    main()
