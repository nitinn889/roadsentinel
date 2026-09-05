#!/usr/bin/env python3
"""
render_pipeline.py
==================
Automated CARLA-to-Blender co-simulation rendering pipeline for RoadSentinel.
Generates photorealistic aerial road inspection datasets with physical geometric depth
for SAM2 and DINOv2 segmentation.

Usage:
  blender -b -P render_pipeline.py
  or: blender -P render_pipeline.py
"""

import csv
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

try:
    import bpy
    import mathutils
    _HAS_BPY = True
except ImportError:
    _HAS_BPY = False
    print("[ERROR] render_pipeline.py must be executed inside Blender: blender -b -P render_pipeline.py")


def generate_pothole_displacement_map(filepath: str, width: int = 1024, height: int = 1024) -> str:
    """Generate a high-contrast PBR pothole heightmap image for physical mesh displacement."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Base asphalt road height = 240 (~0.94)
    img_data = np.full((height, width), 240, dtype=np.uint8)

    # Micro-roughness asphalt texture noise
    noise = np.random.normal(0, 3.5, (height, width)).astype(np.int16)
    img_data = np.clip(img_data.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Define procedural pothole cavities along corridor (normalized u, v)
    potholes = [
        {"center": (0.50, 0.15), "rx": 0.08, "ry": 0.06, "depth": 210, "roughness": 0.02},
        {"center": (0.46, 0.35), "rx": 0.11, "ry": 0.08, "depth": 225, "roughness": 0.03},
        {"center": (0.54, 0.55), "rx": 0.07, "ry": 0.09, "depth": 195, "roughness": 0.02},
        {"center": (0.49, 0.75), "rx": 0.10, "ry": 0.07, "depth": 220, "roughness": 0.025},
        {"center": (0.52, 0.90), "rx": 0.08, "ry": 0.08, "depth": 200, "roughness": 0.02},
    ]

    xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

    for p in potholes:
        cx, cy = p["center"]
        rx, ry = p["rx"], p["ry"]
        depth_val = p["depth"]

        dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        n_border = np.random.normal(0, p["roughness"], (height, width))
        dist_noisy = dist + n_border

        # Bowl-shaped cavity depression profile with sharp rim
        depth_profile = np.clip(1.0 - (np.clip(dist_noisy, 0.0, 1.0) / 1.0) ** 1.8, 0.0, 1.0)

        depression = (depth_profile * depth_val).astype(np.uint8)
        img_data = np.clip(img_data.astype(np.int16) - depression.astype(np.int16), 0, 255).astype(np.uint8)

    Image.fromarray(img_data).save(filepath)
    print(f"[Blender Pipeline] High-contrast displacement map saved to: {filepath}")
    return filepath


def load_drone_trajectory() -> list[dict[str, float]]:
    """Locate and read drone_trajectory.csv."""
    search_paths = [
        Path("drone_trajectory.csv"),
        Path("env/output/drone_trajectory.csv"),
        Path(__file__).parent / "drone_trajectory.csv",
        Path(__file__).parent / "env" / "output" / "drone_trajectory.csv",
    ]

    found_path = None
    for p in search_paths:
        if p.is_file():
            found_path = p
            break

    if not found_path:
        raise FileNotFoundError("drone_trajectory.csv not found. Run drone_sim.py first to generate telemetry trajectory.")

    rows = []
    with open(found_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "frame": int(r["frame"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
                "pitch": float(r["pitch"]),
                "yaw": float(r["yaw"]),
                "roll": float(r["roll"]),
            })
    print(f"[Blender Pipeline] Loaded {len(rows)} keyframes from {found_path}")
    return rows


def setup_blender_scene(trajectory: list[dict[str, float]]) -> None:
    """Construct 3D scene, camera keyframe animation, displacement road surface, and low-angle sun light."""
    if not _HAS_BPY:
        return

    # 1. Clean existing scene data safely using pure Data API
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam, do_unlink=True)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)

    scene = bpy.context.scene

    # Create Collection
    if "RoadSentinel_Scene" not in bpy.data.collections:
        coll = bpy.data.collections.new("RoadSentinel_Scene")
        scene.collection.children.link(coll)
    else:
        coll = bpy.data.collections["RoadSentinel_Scene"]

    # 2. Setup Drone Camera & Keyframe Animation
    cam_data = bpy.data.cameras.new("DroneCamera")
    cam_data.angle = math.radians(60.0)  # 60° horizontal FOV survey lens
    cam_data.clip_start = 0.1
    cam_data.clip_end = 500.0

    cam_obj = bpy.data.objects.new("DroneCamera", cam_data)
    coll.objects.link(cam_obj)
    scene.camera = cam_obj

    print("[Blender Pipeline] Keyframing camera trajectory...")
    for row in trajectory:
        frame_num = row["frame"] + 1  # 1-indexed Blender frames

        # Coordinate transformation: CARLA (left-handed) -> Blender (right-handed)
        # x_blender = x_carla, y_blender = -y_carla, z_blender = z_carla
        x_b = row["x"]
        y_b = -row["y"]
        z_b = row["z"]

        cam_obj.location = (x_b, y_b, z_b)

        # Euler angle conversion
        # Pitch = -90° (nadir looking down) maps to rot_x = 0.0 in Blender
        rot_x = math.radians(row["pitch"] + 90.0)
        rot_y = math.radians(row["roll"])
        rot_z = math.radians(-row["yaw"])

        cam_obj.rotation_euler = (rot_x, rot_y, rot_z)

        cam_obj.keyframe_insert(data_path="location", frame=frame_num)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame_num)

    scene.frame_start = 1
    scene.frame_end = max(1, len(trajectory))

    # 3. Construct 3D Road Plane Mesh (Pure Data API - 100% Context Independent)
    ys = [-r["y"] for r in trajectory]
    xs = [r["x"] for r in trajectory]
    min_y, max_y = min(ys) - 50.0, max(ys) + 50.0
    min_x, max_x = min(xs) - 50.0, max(xs) + 50.0
    center_y = (min_y + max_y) / 2.0
    center_x = (min_x + max_x) / 2.0
    length_y = max(100.0, max_y - min_y)
    length_x = max(100.0, max_x - min_x)

    half_x = max(40.0, length_x / 2.0)
    half_y = max(50.0, length_y / 2.0)

    mesh = bpy.data.meshes.new("AsphaltRoadMesh")
    verts = [
        (center_x - half_x, center_y - half_y, 0.0),
        (center_x + half_x, center_y - half_y, 0.0),
        (center_x + half_x, center_y + half_y, 0.0),
        (center_x - half_x, center_y + half_y, 0.0),
    ]
    faces = [(0, 1, 2, 3)]
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for j, loop_idx in enumerate(poly.loop_indices):
            uv_layer.data[loop_idx].uv = uvs[j]

    road_obj = bpy.data.objects.new("AsphaltRoadPlane", mesh)
    coll.objects.link(road_obj)

    # Add Subdivision Surface modifier for micro-geometry subdivision
    subsurf = road_obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.subdivision_type = 'SIMPLE'
    subsurf.levels = 6
    subsurf.render_levels = 6

    # 4. Construct PBR Asphalt Material with Physical Displacement and Bump
    disp_path = str(Path("dataset/pothole_displacement.png").resolve())
    if not os.path.isfile(disp_path):
        generate_pothole_displacement_map(disp_path)

    mat = bpy.data.materials.new("AsphaltPotholeMaterial")
    mat.use_nodes = True

    # Set displacement method (Blender 5.0+ and 3.x/4.x compatible)
    if hasattr(mat, "displacement_method"):
        mat.displacement_method = 'BOTH'
    elif hasattr(mat, "cycles") and hasattr(mat.cycles, "displacement_method"):
        mat.cycles.displacement_method = 'BOTH'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create Material Nodes
    node_out = nodes.new('ShaderNodeOutputMaterial')
    node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    node_disp = nodes.new('ShaderNodeDisplacement')
    node_tex = nodes.new('ShaderNodeTexImage')

    # Configure BSDF (Dark Asphalt Pavement)
    if 'Base Color' in node_bsdf.inputs:
        node_bsdf.inputs['Base Color'].default_value = (0.07, 0.07, 0.08, 1.0)
    if 'Roughness' in node_bsdf.inputs:
        node_bsdf.inputs['Roughness'].default_value = 0.85

    # Load Displacement Image Texture
    img = bpy.data.images.load(disp_path)
    img.colorspace_settings.name = 'Non-Color'
    node_tex.image = img

    # Configure Displacement Node
    node_disp.inputs['Height'].default_value = 1.0
    node_disp.inputs['Midlevel'].default_value = 0.94  # 0.94 is ground level
    node_disp.inputs['Scale'].default_value = 0.20     # 20cm max cavity depth

    # Connect Nodes
    links.new(node_tex.outputs['Color'], node_disp.inputs['Height'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    links.new(node_disp.outputs['Displacement'], node_out.inputs['Displacement'])

    road_obj.data.materials.append(mat)

    # 5. Directional Sun Lighting at Low Angle (15-25 degrees for deep physical shadows)
    light_data = bpy.data.lights.new(name="SunLight", type='SUN')
    light_data.energy = 4.5
    light_data.angle = math.radians(1.0)  # Sharp shadow edges

    light_obj = bpy.data.objects.new(name="SunLight", object_data=light_data)
    coll.objects.link(light_obj)

    # Sun elevation = 20 degrees above horizon -> Pitch = 70 degrees from zenith
    light_obj.rotation_euler = (math.radians(70.0), math.radians(15.0), math.radians(45.0))

    # 6. Configure Cycles Render Engine & GPU / OptiX Denoising
    scene.render.engine = 'CYCLES'

    # Configure GPU compute devices safely
    try:
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        gpu_enabled = False
        for dev_type in ['OPTIX', 'CUDA', 'HIP']:
            try:
                cycles_prefs.compute_device_type = dev_type
                cycles_prefs.get_devices()
                devices = cycles_prefs.devices
                if devices:
                    for d in devices:
                        d.use = True
                    scene.cycles.device = 'GPU'
                    gpu_enabled = True
                    print(f"[Blender Pipeline] GPU Compute enabled ({dev_type}).")
                    break
            except Exception:
                pass

        if not gpu_enabled:
            scene.cycles.device = 'CPU'
            print("[Blender Pipeline] GPU compute unavailable. Falling back to CPU render.")
    except Exception as exc:
        scene.cycles.device = 'CPU'
        print(f"[Blender Pipeline] Cycles preferences setup ({exc}). Using CPU render.")

    # Samples & Denoising
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = 'OPTIX'
    except Exception:
        try:
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        except Exception:
            pass

    # Output settings
    out_dir = Path("./dataset/rendered_frames").resolve()
    os.makedirs(out_dir, exist_ok=True)

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720

    print(f"[Blender Pipeline] Setup complete. Output destination: {out_dir}/frame_####.png")


def main():
    print("=" * 68)
    print("      ROADSENTINEL CARLA-TO-BLENDER CO-SIMULATION PIPELINE v3.0")
    print("=" * 68)

    trajectory = load_drone_trajectory()
    setup_blender_scene(trajectory)

    scene = bpy.context.scene
    out_dir = Path("./dataset/rendered_frames").resolve()
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Blender Pipeline] Rendering {len(trajectory)} frames to {out_dir}...")
    for row in trajectory:
        frame_num = row["frame"] + 1
        scene.frame_set(frame_num)
        frame_path = str(out_dir / f"frame_{frame_num:04d}.png")
        scene.render.filepath = frame_path
        print(f"[Blender Pipeline] Rendering frame {frame_num}/{len(trajectory)} -> {frame_path}")
        bpy.ops.render.render(write_still=True)

    print("=" * 68)
    print("[Blender Pipeline] Render completed successfully!")
    print("=" * 68)


if __name__ == "__main__":
    main()
