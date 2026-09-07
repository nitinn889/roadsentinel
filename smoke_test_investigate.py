import os
import sys
import time
from pathlib import Path
import json

import unreal

print("=" * 60)
print(">>> SMOKE TEST INVESTIGATION: RoadSentinelSim SceneCapture2D <<<")
print("=" * 60)

# 1. Load SimBlank level
map_path = "/Game/SimBlank/Levels/SimBlank"
print(f"Loading map: {map_path} ...")
try:
    unreal.EditorLoadingAndSavingUtils.load_map(map_path)
    print(f"Loaded map successfully!")
except Exception as e:
    print(f"Error loading map: {e}")

world = unreal.EditorLevelLibrary.get_editor_world()
print(f"Current world: {world.get_path_name() if world else 'None'}")

# 2. Add Phase 2 script path
script_dir = "/home/nitin-nandakumar/Downloads/roadsentinel/RoadSentinelSim/Content/Python"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import rs_phase2_surface_defects as phase2

# 3. Materialize Day 1
manifest_path = "/home/nitin-nandakumar/Downloads/roadsentinel/env/output/temporal_20_day/manifests/day_01.json"
gt_path = "/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test/day_01_gt.json"
Path(gt_path).parent.mkdir(parents=True, exist_ok=True)

print(f"Materializing Day 1 from: {manifest_path} ...")
scene = phase2.materialize_temporal_manifest(manifest_path, gt_path)
print(f"Scene materialized! Segments: {len(scene['segments'])}, Defects: {len(scene['defects'])}")

# Find SEG_001
seg1 = next(s for s in scene["segments"] if s["road_segment_id"] == "SEG_001")
camera_pose = seg1["camera_pose"]
print(f"SEG_001 camera pose: {camera_pose}")

# 4. Check lighting actors
all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
print(f"Total level actors: {len(all_actors)}")
for a in all_actors:
    cname = a.get_class().get_name()
    if any(k in cname.lower() or k in a.get_name().lower() for k in ["sun", "light", "fog", "postprocess"]):
        print(f"  Light/Atmosphere Actor: {a.get_name()} ({cname})")

# 5. Setup SceneCapture2D with various exposure / capture settings to see what produces genuine render
out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test")
out_dir.mkdir(parents=True, exist_ok=True)

camera_config = scene["camera_config"]
target_w = int(camera_config["width"])
target_h = int(camera_config["height"])

render_format = getattr(
    unreal.TextureRenderTargetFormat,
    "RTF_RGBA8_SRGB",
    unreal.TextureRenderTargetFormat.RTF_RGBA8,
)
try:
    target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)
    print(f"Created render target: {target}")
except Exception as e:
    print(f"Failed to create render target: {e}")
    target = None

loc = unreal.Vector(float(camera_pose["x_cm"]), float(camera_pose["y_cm"]), float(camera_pose["z_cm"]))
rot = unreal.Rotator(roll=0.0, pitch=float(camera_pose["pitch_deg"]), yaw=float(camera_pose["yaw_deg"]))

# Move editor viewport camera as well
unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)

def wait_for_png(expected_path: Path, timeout_s: float = 10.0) -> Path:
    candidates = (expected_path, expected_path.with_name(expected_path.name + ".png"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 0:
                if candidate != expected_path:
                    candidate.replace(expected_path)
                return expected_path
        time.sleep(0.1)
    print(f"Warning: File not found within {timeout_s}s: {expected_path}")
    return expected_path

# Test 1: Standard SceneCapture2D (as previously configured)
try:
    cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", target)
    comp.set_editor_property("fov_angle", float(camera_config["fov_deg"]))
    comp.set_editor_property("capture_every_frame", False)
    comp.set_editor_property("capture_on_movement", False)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)

    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()

    raw_png_1 = out_dir / "test1_scenecapture_standard.png"
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), raw_png_1.name)
    wait_for_png(raw_png_1)
    print(f"Exported Test 1 to {raw_png_1} (exists: {raw_png_1.exists()}, size: {raw_png_1.stat().st_size if raw_png_1.exists() else 0})")
except Exception as e:
    print(f"Test 1 failed: {e}")

# Test 2: SceneCapture2D with auto-exposure settings explicitly enabled
try:
    pps = comp.get_editor_property("post_process_settings")
    try:
        pps.set_editor_property("override_auto_exposure_method", True)
        pps.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_HISTOGRAM)
    except Exception as ex1:
        print(f"Auto-exposure method setting error: {ex1}")
    try:
        pps.set_editor_property("override_auto_exposure_bias", True)
        pps.set_editor_property("auto_exposure_bias", 0.0)
    except Exception as ex2:
        print(f"Auto-exposure bias setting error: {ex2}")
    try:
        pps.set_editor_property("override_auto_exposure_min_brightness", True)
        pps.set_editor_property("auto_exposure_min_brightness", 0.03)
        pps.set_editor_property("override_auto_exposure_max_brightness", True)
        pps.set_editor_property("auto_exposure_max_brightness", 10.0)
    except Exception as ex3:
        print(f"Auto-exposure min/max setting error: {ex3}")
    comp.set_editor_property("post_process_settings", pps)

    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()

    raw_png_2 = out_dir / "test2_scenecapture_autoexposure.png"
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), raw_png_2.name)
    wait_for_png(raw_png_2)
    print(f"Exported Test 2 to {raw_png_2} (exists: {raw_png_2.exists()}, size: {raw_png_2.stat().st_size if raw_png_2.exists() else 0})")
except Exception as e:
    print(f"Test 2 failed: {e}")

# Test 3: HighResShot from Viewport
try:
    shot_path = str(out_dir / "test3_viewport_highresshot.png")
    cmd = f"HighResShot 1920x1080 filename=\"{shot_path}\""
    unreal.SystemLibrary.execute_console_command(world, cmd)
    print(f"Executed Test 3 console command: {cmd}")
except Exception as e:
    print(f"Test 3 failed: {e}")

try:
    time.sleep(1.0)
    if 'cam' in locals() and cam:
        unreal.EditorLevelLibrary.destroy_actor(cam)
except Exception as e:
    print(f"Cleanup error: {e}")

print(">>> SMOKE TEST INVESTIGATION COMPLETE <<<")
