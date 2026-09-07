import os
import sys
import time
from pathlib import Path
import unreal

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

out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test/bias_test")
out_dir.mkdir(parents=True, exist_ok=True)

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

# Test negative biases with AEM_HISTOGRAM and AEM_MANUAL
for method_name, method in [("manual", unreal.AutoExposureMethod.AEM_MANUAL), ("histogram", unreal.AutoExposureMethod.AEM_HISTOGRAM)]:
    for bias in [-14.0, -10.0, -6.0, -3.0, 0.0]:
        pps = comp.get_editor_property("post_process_settings")
        pps.set_editor_property("override_auto_exposure_method", True)
        pps.set_editor_property("auto_exposure_method", method)
        pps.set_editor_property("override_auto_exposure_bias", True)
        pps.set_editor_property("auto_exposure_bias", bias)
        comp.set_editor_property("post_process_settings", pps)
        comp.set_editor_property("post_process_blend_weight", 1.0)

        comp.capture_scene()
        time.sleep(0.15)
        comp.capture_scene()
        fname = f"{method_name}_bias_{bias:+.1f}.png"
        unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), fname)
        print(f"Exported {fname}")

unreal.EditorLevelLibrary.destroy_actor(cam)
print(">>> BIAS TEST COMPLETE <<<")
