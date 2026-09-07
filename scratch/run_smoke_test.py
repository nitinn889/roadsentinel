import os
import sys
import time
import json
from pathlib import Path

import unreal

print("=" * 70)
print(">>> ROAD SENTINEL SMOKE TEST: DAY 1 - SEG_001 SCENECAPTURE2D <<<")
print("=" * 70)

# 1. Load SimBlank level
map_path = "/Game/SimBlank/Levels/SimBlank"
print(f"Loading map: {map_path} ...")
unreal.EditorLoadingAndSavingUtils.load_map(map_path)
world = unreal.EditorLevelLibrary.get_editor_world()
if not world:
    raise RuntimeError("Failed to obtain Editor World")
print(f"Current world: {world.get_path_name()}")

# 2. Add python paths
repo_root = Path("/home/nitin-nandakumar/Downloads/roadsentinel")
python_dir = repo_root / "RoadSentinelSim" / "Content" / "Python"
scripts_dir = repo_root / "env" / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))
if str(python_dir) in sys.path:
    sys.path.remove(str(python_dir))
sys.path.insert(0, str(python_dir))

if "rs_phase2_surface_defects" in sys.modules:
    del sys.modules["rs_phase2_surface_defects"]
import rs_phase2_surface_defects as phase2
print(f"Loaded phase2 from: {phase2.__file__}")
print(f"Has materialize_temporal_manifest: {hasattr(phase2, 'materialize_temporal_manifest')}")
import rs_inspection_capture as ic

# 3. Materialise Day 1
manifest_path = repo_root / "env" / "output" / "temporal_20_day" / "manifests" / "day_01.json"
gt_path = repo_root / "env" / "output" / "smoke_test" / "day_01_gt.json"
print(f"Materialising Day 1 from {manifest_path} ...")
scene = phase2.materialize_temporal_manifest(str(manifest_path), str(gt_path))
print(f"Materialised Day 1: {len(scene['segments'])} segments, {len(scene['defects'])} defects.")

# 4. Move to SEG_001's fixed downward camera pose
seg1 = next(s for s in scene["segments"] if s["road_segment_id"] == "SEG_001")
camera_pose = seg1["camera_pose"]
world_location_cm = seg1["world_location_cm"]
persistent_location = seg1["persistent_location"]
condition_score = seg1.get("renderer_condition_score") or 0.0
camera_config = scene["camera_config"]

print(f"SEG_001 Camera Pose: {camera_pose}")
loc = unreal.Vector(float(camera_pose["x_cm"]), float(camera_pose["y_cm"]), float(camera_pose["z_cm"]))
rot = unreal.Rotator(roll=float(camera_pose.get("roll_deg", 0.0)), pitch=float(camera_pose["pitch_deg"]), yaw=float(camera_pose["yaw_deg"]))

# Update viewport camera to match
unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)

# 5. Output directory: env/output/manual_inspections/day_01/SEG_001
out_dir = repo_root / "env" / "output" / "manual_inspections" / "day_01" / "SEG_001"
out_dir.mkdir(parents=True, exist_ok=True)
raw_path = out_dir / "raw.png"
if raw_path.exists():
    raw_path.unlink()

# 6. Create Render Target
target_w = int(camera_config.get("width", 1920))
target_h = int(camera_config.get("height", 1080))
render_format = getattr(unreal.TextureRenderTargetFormat, "RTF_RGBA8_SRGB", unreal.TextureRenderTargetFormat.RTF_RGBA8)
target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)
if not target:
    raise RuntimeError("Failed to create RenderTarget2D")

# 7. Spawn SceneCapture2D actor
cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
if not cam:
    raise RuntimeError("Failed to spawn SceneCapture2D actor")

try:
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", target)
    comp.set_editor_property("fov_angle", float(camera_config.get("fov_deg", 90.0)))
    comp.set_editor_property("capture_every_frame", False)
    comp.set_editor_property("capture_on_movement", False)
    comp.set_editor_property("always_persist_rendering_state", True)
    comp.set_editor_property("inherit_main_view_camera_post_process_settings", True)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)

    pps = comp.get_editor_property("post_process_settings")
    pps.set_editor_property("override_auto_exposure_bias", True)
    pps.set_editor_property("auto_exposure_bias", 2.6)
    comp.set_editor_property("post_process_settings", pps)
    comp.set_editor_property("post_process_blend_weight", 1.0)

    # Let lighting and materials stream
    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()

    # Export to PNG
    print(f"Exporting Render Target to {raw_path} ...")
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), raw_path.name)

    # Wait for export
    deadline = time.time() + 15.0
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
        time.sleep(0.1)
    else:
        raise RuntimeError("SceneCapture2D PNG export timed out")

    print(f"PNG export confirmed: {actual_path} ({actual_path.stat().st_size} bytes)")

finally:
    unreal.EditorLevelLibrary.destroy_actor(cam)

# 8. Validate PNG
val = ic.validate_png(actual_path, expected_width=target_w, expected_height=target_h)
print(f"Validation result: {val}")

# 9. Write Metadata
meta_path = ic.write_metadata(
    output_dir=out_dir,
    day=1,
    segment_id="SEG_001",
    camera_pose=camera_pose,
    world_coordinates={
        "x_cm": world_location_cm.get("x", 0.0),
        "y_cm": world_location_cm.get("y", 0.0),
        "z_cm": world_location_cm.get("z", 0.0),
        "along_m": persistent_location.get("along_m", 0.0),
        "across_m": persistent_location.get("across_m", 0.0),
    },
    condition_score=condition_score,
    extra={
        "capture_day": 1,
        "total_defects_day": len(scene["defects"]),
    }
)
print(f"Metadata written: {meta_path}")
print(f"Smoke test execution complete!")
