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
out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/scratch")

target_w, target_h = 1920, 1080
render_format = getattr(unreal.TextureRenderTargetFormat, "RTF_RGBA8_SRGB", unreal.TextureRenderTargetFormat.RTF_RGBA8)
target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
comp.set_editor_property("texture_target", target)
comp.set_editor_property("fov_angle", 90.0)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("always_persist_rendering_state", True)
comp.set_editor_property("inherit_main_view_camera_post_process_settings", True)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)

# Try PRM_RENDER_SCENE_PRIMITIVES
for mode_name in dir(unreal.SceneCapturePrimitiveRenderMode):
    if not mode_name.startswith("_"):
        val = getattr(unreal.SceneCapturePrimitiveRenderMode, mode_name)
        print(f"SceneCapturePrimitiveRenderMode: {mode_name} = {val}")

try:
    comp.set_editor_property("primitive_render_mode", unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES)
    print("Set primitive_render_mode to PRM_RENDER_SCENE_PRIMITIVES")
except Exception as e:
    print(f"Failed to set PRM_RENDER_SCENE_PRIMITIVES: {e}")

pps = comp.get_editor_property("post_process_settings")
pps.set_editor_property("override_auto_exposure_bias", True)
pps.set_editor_property("auto_exposure_bias", 2.6)
comp.set_editor_property("post_process_settings", pps)

comp.capture_scene()
time.sleep(0.3)
comp.capture_scene()

sc_path = out_dir / "test_prm.png"
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), sc_path.name)
print("Saved test_prm.png")

unreal.EditorLevelLibrary.destroy_actor(cam)
