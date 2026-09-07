import sys
import time
from pathlib import Path
import unreal

map_path = "/Game/SimBlank/Levels/SimBlank"
unreal.EditorLoadingAndSavingUtils.load_map(map_path)
world = unreal.EditorLevelLibrary.get_editor_world()

python_dir = "/home/nitin-nandakumar/Downloads/roadsentinel/RoadSentinelSim/Content/Python"
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
import rs_phase2_surface_defects as phase2

manifest_path = "/home/nitin-nandakumar/Downloads/roadsentinel/env/output/temporal_20_day/manifests/day_01.json"
gt_path = "/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test/day_01_gt.json"
scene = phase2.materialize_temporal_manifest(manifest_path, gt_path)

loc = unreal.Vector(3500.0, -135.0, 2500.0)
rot = unreal.Rotator(0, -89.0, 0)
out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/scratch/gamma_test")
out_dir.mkdir(parents=True, exist_ok=True)

# Test 1: RTF_RGBA8 (Linear, not SRGB) with SCS_FINAL_COLOR_LDR
target_linear = unreal.RenderingLibrary.create_render_target2d(world, 1920, 1080, unreal.TextureRenderTargetFormat.RTF_RGBA8)
cam1 = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp1 = cam1.get_component_by_class(unreal.SceneCaptureComponent2D)
comp1.set_editor_property("texture_target", target_linear)
comp1.set_editor_property("fov_angle", 90.0)
comp1.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp1.set_editor_property("always_persist_rendering_state", True)
comp1.set_editor_property("inherit_main_view_camera_post_process_settings", True)
pps1 = comp1.get_editor_property("post_process_settings")
pps1.set_editor_property("override_auto_exposure_bias", True)
pps1.set_editor_property("auto_exposure_bias", 2.6)
comp1.set_editor_property("post_process_settings", pps1)

comp1.capture_scene()
time.sleep(0.3)
comp1.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target_linear, str(out_dir), "test_linear_rt.png")

# Test 2: Viewport screenshot as baseline
unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)
shot_path = str(out_dir / "test_viewport_baseline.png")
unreal.SystemLibrary.execute_console_command(world, f"HighResShot 1920x1080 filename=\"{shot_path}\"")

time.sleep(1.0)
unreal.EditorLevelLibrary.destroy_actor(cam1)
print("Gamma test completed!")
