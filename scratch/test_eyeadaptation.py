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
out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/scratch/flag_test")
out_dir.mkdir(parents=True, exist_ok=True)

target = unreal.RenderingLibrary.create_render_target2d(world, 1920, 1080, unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
comp.set_editor_property("texture_target", target)
comp.set_editor_property("fov_angle", 90.0)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("always_persist_rendering_state", True)
comp.set_editor_property("inherit_main_view_camera_post_process_settings", True)
comp.set_editor_property("capture_every_frame", True) # Let it run frames for eye adaptation

# Add EyeAdaptation to show_flag_settings
flag_eye = unreal.EngineShowFlagsSetting()
flag_eye.set_editor_property("enabled", True)
flag_eye.set_editor_property("show_flag_name", "EyeAdaptation")
comp.set_editor_property("show_flag_settings", [flag_eye])

# Let it capture a few frames to adapt
for _ in range(5):
    comp.capture_scene()
    time.sleep(0.1)

unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "sc_eyeadaptation.png")
print("Exported sc_eyeadaptation.png")
unreal.EditorLevelLibrary.destroy_actor(cam)
