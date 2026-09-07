import os
import sys
import time
from pathlib import Path
import unreal

print("=" * 60)
print(">>> TEST SOLUTION FOR SCENECAPTURE2D EXPOSURE <<<")
print("=" * 60)

map_path = "/Game/SimBlank/Levels/SimBlank"
unreal.EditorLoadingAndSavingUtils.load_map(map_path)
world = unreal.EditorLevelLibrary.get_editor_world()

script_dir = "/home/nitin-nandakumar/Downloads/roadsentinel/RoadSentinelSim/Content/Python"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import rs_phase2_surface_defects as phase2

manifest_path = "/home/nitin-nandakumar/Downloads/roadsentinel/env/output/temporal_20_day/manifests/day_01.json"
gt_path = "/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test/day_01_gt.json"
scene = phase2.materialize_temporal_manifest(manifest_path, gt_path)

seg1 = next(s for s in scene["segments"] if s["road_segment_id"] == "SEG_001")
camera_pose = seg1["camera_pose"]
loc = unreal.Vector(float(camera_pose["x_cm"]), float(camera_pose["y_cm"]), float(camera_pose["z_cm"]))
rot = unreal.Rotator(roll=0.0, pitch=float(camera_pose["pitch_deg"]), yaw=float(camera_pose["yaw_deg"]))

out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test/solution_test")
out_dir.mkdir(parents=True, exist_ok=True)

target_w, target_h = 1920, 1080
render_format = getattr(unreal.TextureRenderTargetFormat, "RTF_RGBA8_SRGB", unreal.TextureRenderTargetFormat.RTF_RGBA8)
target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)

# Set viewport camera pose as well
unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)

# Test Method 1: inherit_main_view_camera_post_process_settings = True
print("--- Testing Method 1: inherit_main_view_camera_post_process_settings ---")
cam1 = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp1 = cam1.get_component_by_class(unreal.SceneCaptureComponent2D)
comp1.set_editor_property("texture_target", target)
comp1.set_editor_property("fov_angle", 90.0)
comp1.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp1.set_editor_property("always_persist_rendering_state", True)
comp1.set_editor_property("inherit_main_view_camera_post_process_settings", True)
comp1.set_editor_property("capture_every_frame", False)
comp1.capture_scene()
time.sleep(0.3)
comp1.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "method1_inherit_viewport.png")
unreal.EditorLevelLibrary.destroy_actor(cam1)

# Test Method 2: Physical camera exposure settings (ISO 100, f/11, 1/1000s = EV 16)
print("--- Testing Method 2: Physical Camera Exposure ---")
cam2 = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp2 = cam2.get_component_by_class(unreal.SceneCaptureComponent2D)
comp2.set_editor_property("texture_target", target)
comp2.set_editor_property("fov_angle", 90.0)
comp2.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
pps2 = comp2.get_editor_property("post_process_settings")
pps2.set_editor_property("override_auto_exposure_method", True)
pps2.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
pps2.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
pps2.set_editor_property("auto_exposure_apply_physical_camera_exposure", True)
pps2.set_editor_property("override_camera_iso", True)
pps2.set_editor_property("camera_iso", 100.0)
pps2.set_editor_property("override_camera_shutter_speed", True)
pps2.set_editor_property("camera_shutter_speed", 1000.0)
pps2.set_editor_property("override_depth_of_field_fstop", True)
pps2.set_editor_property("depth_of_field_fstop", 8.0)
comp2.set_editor_property("post_process_settings", pps2)
comp2.set_editor_property("post_process_blend_weight", 1.0)
comp2.capture_scene()
time.sleep(0.3)
comp2.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "method2_physical_camera.png")
unreal.EditorLevelLibrary.destroy_actor(cam2)

# Test Method 3: Lighting calibrated to standard unitless / SceneCapture scale
# (Where sun_lux is scaled by 10,000 so 100,000 lux -> 10.0, which fits standard LDR)
print("--- Testing Method 3: Calibrated Sun Lux with raised markings ---")
for a in unreal.EditorLevelLibrary.get_all_level_actors():
    dlc = a.get_component_by_class(unreal.DirectionalLightComponent)
    if dlc:
        dlc.set_editor_property("intensity", 10.0)
    slc = a.get_component_by_class(unreal.SkyLightComponent)
    if slc:
        slc.set_editor_property("intensity", 1.5)

# Also check raising markings by 2.0 cm so there is zero Z-fighting
for a in unreal.EditorLevelLibrary.get_all_level_actors():
    if a.actor_has_tag(unreal.Name("RS_Marking")):
        loc_m = a.get_actor_location()
        a.set_actor_location(unreal.Vector(loc_m.x, loc_m.y, 2.0), False, False)

cam3 = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp3 = cam3.get_component_by_class(unreal.SceneCaptureComponent2D)
comp3.set_editor_property("texture_target", target)
comp3.set_editor_property("fov_angle", 90.0)
comp3.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp3.capture_scene()
time.sleep(0.3)
comp3.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "method3_calibrated_sun_raised_markings.png")
unreal.EditorLevelLibrary.destroy_actor(cam3)

print(">>> TEST SOLUTION COMPLETE <<<")
