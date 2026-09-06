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
                          actor.actor_has_tag(unreal.Name("RS_Defect"))):
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
        marking_thick_cm = 0.8

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
        for offset_cm in [-12.0, 12.0]:
            ylw = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(12500.0, offset_cm, 0.4),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
            )
            if ylw and cube_mesh:
                ylw.tags.append(unreal.Name("RS_Marking"))
                yc = ylw.static_mesh_component
                if yc:
                    yc.set_static_mesh(cube_mesh)
                    yc.set_world_scale3d(unreal.Vector(250.0, 0.12, marking_thick_cm / 100.0))
                    if mat_yellow:
                        yc.set_material(0, mat_yellow)

        # Outer white fog lines
        for fog_offset_cm in [-750.0, 750.0]:
            fog = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor,
                unreal.Vector(12500.0, fog_offset_cm, 0.4),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
            )
            if fog and cube_mesh:
                fog.tags.append(unreal.Name("RS_Marking"))
                fc = fog.static_mesh_component
                if fc:
                    fc.set_static_mesh(cube_mesh)
                    fc.set_world_scale3d(unreal.Vector(250.0, 0.15, marking_thick_cm / 100.0))
                    if mat_white:
                        fc.set_material(0, mat_white)

        # Dashed white lane lines
        for x_m in range(6, 246, 12):
            for lane_offset_cm in [-375.0, 375.0]:
                dash = unreal.EditorLevelLibrary.spawn_actor_from_class(
                    unreal.StaticMeshActor,
                    unreal.Vector(x_m * 100.0, lane_offset_cm, 0.4),
                    unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
                )
                if dash and cube_mesh:
                    dash.tags.append(unreal.Name("RS_Marking"))
                    dc = dash.static_mesh_component
                    if dc:
                        dc.set_static_mesh(cube_mesh)
                        dc.set_world_scale3d(unreal.Vector(3.0, 0.15, marking_thick_cm / 100.0))
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
                unreal.Vector(cx_cm, cy_cm, 0.4),
                unreal.Rotator(roll=0.0, pitch=0.0, yaw=yaw_deg)
            )
            if c_ylw and cube_mesh:
                c_ylw.tags.append(unreal.Name("RS_Marking"))
                cyc = c_ylw.static_mesh_component
                if cyc:
                    cyc.set_static_mesh(cube_mesh)
                    cyc.set_world_scale3d(unreal.Vector(5.5, 0.20, marking_thick_cm / 100.0))
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
# 6. Drone Camera Navigation & Photo Capture ('C')
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
    """Captures a photo from the current drone viewport camera and logs telemetry."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
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

        # Execute high resolution screenshot
        cmd = f"HighResShot 1920x1080 filename=\"{target_path}\""
        unreal.SystemLibrary.execute_console_command(None, cmd)
        unreal.SystemLibrary.execute_console_command(None, "Shot")

        # Copy newest screenshot if saved to standard Linux Screenshots folder
        ue_saved_dir = os.path.join(WORKSPACE_ROOT, "RoadSentinelSim", "Saved", "Screenshots", "Linux")
        if os.path.exists(ue_saved_dir):
            files = [os.path.join(ue_saved_dir, f) for f in os.listdir(ue_saved_dir) if f.endswith(".png")]
            if files:
                newest = max(files, key=os.path.getmtime)
                if time.time() - os.path.getmtime(newest) < 4.0:
                    shutil.copy2(newest, target_path)

    # Append to capture log
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
# 7. Unreal Engine IPC Server (Receives commands from Studio GUI)
# ==============================================================================

import socket
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

                        if action == "generate":
                            defects = spawn_full_world(
                                road_health=msg.get("road_health", "Moderate Deterioration (Grade C)"),
                                lighting_preset=msg.get("lighting", "Clear Noon (70° Sun)"),
                                size_profile=msg.get("sizing", "Multi-Scale Organic (Mixed)"),
                                density_per_100m2=float(msg.get("density", 3.5)),
                                water_ratio=msg.get("water", "Mixed Wet/Dry"),
                                random_seed=int(time.time()) % 10000
                            )
                            response["total_defects"] = len(defects)

                        elif action == "capture":
                            out_path = capture_drone_photo()
                            response["path"] = out_path

                        elif action == "teleport":
                            teleport_drone_camera(msg.get("viewpoint", ""))

                        elif action == "lighting":
                            apply_environment_lighting(msg.get("preset", "Clear Noon (70° Sun)"))

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

