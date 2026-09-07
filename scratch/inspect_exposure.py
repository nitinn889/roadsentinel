import os
import sys
import time
from pathlib import Path
import unreal

print("=" * 60)
print(">>> INSPECT EXPOSURE & SCENECAPTURE2D <<<")
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

# Find PostProcessVolume_1
pp_vol = None
for a in unreal.EditorLevelLibrary.get_all_level_actors():
    if "PostProcessVolume" in a.get_name() or a.get_class().get_name() == "PostProcessVolume":
        pp_vol = a
        print(f"Found PostProcessVolume: {a.get_name()}")
        break

if pp_vol:
    s = pp_vol.get_editor_property("settings")
    for prop in dir(s):
        if "exposure" in prop.lower() or "ev" in prop.lower() or "bright" in prop.lower():
            try:
                val = s.get_editor_property(prop)
                print(f"  PP Setting: {prop} = {val}")
            except Exception:
                pass

# Now create SceneCapture2D and test different exposure setups
out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test")
target_w, target_h = 1920, 1080
render_format = getattr(unreal.TextureRenderTargetFormat, "RTF_RGBA8_SRGB", unreal.TextureRenderTargetFormat.RTF_RGBA8)
target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
comp.set_editor_property("texture_target", target)
comp.set_editor_property("fov_angle", 90.0)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("capture_on_movement", False)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)

# Experiment A: Copy post_process_settings from PostProcessVolume_1
if pp_vol:
    print("Testing Experiment A: Copy PP volume settings to SceneCapture2D...")
    comp.set_editor_property("post_process_settings", pp_vol.get_editor_property("settings"))
    comp.set_editor_property("post_process_blend_weight", 1.0)
    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "exp_a_copy_pp.png")
    print("Exported exp_a_copy_pp.png")

# Experiment B: Manual Exposure EV100 (e.g. EV100 = 10, 12, 14)
print("Testing Experiment B: Manual Exposure EV100...")
pps = comp.get_editor_property("post_process_settings")
try:
    pps.set_editor_property("override_auto_exposure_method", True)
    pps.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    pps.set_editor_property("override_auto_exposure_bias", True)
    # In UE5 manual exposure, EV100 = -bias or bias
    pps.set_editor_property("auto_exposure_bias", 12.0)
    comp.set_editor_property("post_process_settings", pps)
    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "exp_b_manual_ev12.png")
    print("Exported exp_b_manual_ev12.png")
except Exception as e:
    print(f"Exp B failed: {e}")

# Experiment C: Check show_flags
try:
    print("Testing Experiment C: Enable EyeAdaptation show flag...")
    # SceneCaptureComponent2D show flags: show_flags.eye_adaptation
    comp.set_editor_property("post_process_settings", pp_vol.get_editor_property("settings") if pp_vol else pps)
    # In UE, show_flag_settings exists
    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "exp_c.png")
except Exception as e:
    print(f"Exp C failed: {e}")

# Experiment D: Lower Sun light intensity in scene to standard non-EV lux (e.g. 50 lux)
print("Testing Experiment D: Adjusting Sun light to match standard LDR rendering...")
for a in unreal.EditorLevelLibrary.get_all_level_actors():
    dlc = a.get_component_by_class(unreal.DirectionalLightComponent)
    if dlc:
        print(f"Found DirectionalLight on {a.get_name()}, intensity was: {dlc.get_editor_property('intensity')}")
        dlc.set_editor_property("intensity", 10.0)
    slc = a.get_component_by_class(unreal.SkyLightComponent)
    if slc:
        print(f"Found SkyLight on {a.get_name()}, intensity was: {slc.get_editor_property('intensity')}")
        slc.set_editor_property("intensity", 1.0)

# Reset PP settings on comp to default
comp.set_editor_property("post_process_blend_weight", 0.0)
comp.capture_scene()
time.sleep(0.3)
comp.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "exp_d_lower_sun.png")
print("Exported exp_d_lower_sun.png")

unreal.EditorLevelLibrary.destroy_actor(cam)
print(">>> INSPECT EXPOSURE COMPLETE <<<")
