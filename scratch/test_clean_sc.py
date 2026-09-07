import os
import sys
import time
from pathlib import Path
import unreal

print("=" * 60)
print(">>> TEST CLEAN SCENECAPTURE2D WITH ROAD OFFSET <<<")
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

all_actors = unreal.EditorLevelLibrary.get_all_level_actors()

# Lower road slabs slightly by 2cm so markings (at z=0.4) are 2.4cm above road!
for a in all_actors:
    if a.actor_has_tag(unreal.Name("RS_Road")):
        loc_r = a.get_actor_location()
        a.set_actor_location(unreal.Vector(loc_r.x, loc_r.y, loc_r.z - 2.0), False, False)
        print(f"Lowered road {a.get_name()} by 2cm: new Z = {loc_r.z - 2.0}")

seg1 = next(s for s in scene["segments"] if s["road_segment_id"] == "SEG_001")
camera_pose = seg1["camera_pose"]
loc = unreal.Vector(float(camera_pose["x_cm"]), float(camera_pose["y_cm"]), float(camera_pose["z_cm"]))
rot = unreal.Rotator(roll=0.0, pitch=float(camera_pose["pitch_deg"]), yaw=float(camera_pose["yaw_deg"]))

out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test")
out_dir.mkdir(parents=True, exist_ok=True)

target_w, target_h = 1920, 1080
render_format = getattr(unreal.TextureRenderTargetFormat, "RTF_RGBA8_SRGB", unreal.TextureRenderTargetFormat.RTF_RGBA8)
target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)

unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
comp.set_editor_property("texture_target", target)
comp.set_editor_property("fov_angle", 90.0)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("always_persist_rendering_state", True)
comp.set_editor_property("inherit_main_view_camera_post_process_settings", True)
comp.set_editor_property("capture_every_frame", False)

pps = comp.get_editor_property("post_process_settings")
pps.set_editor_property("override_auto_exposure_bias", True)
pps.set_editor_property("auto_exposure_bias", 2.2)
comp.set_editor_property("post_process_settings", pps)
comp.set_editor_property("post_process_blend_weight", 1.0)

comp.capture_scene()
time.sleep(0.3)
comp.capture_scene()

fname = "scenecapture_road_lowered.png"
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), fname)
print(f"Exported {fname}")

unreal.EditorLevelLibrary.destroy_actor(cam)
print(">>> CLEAN SCENECAPTURE2D COMPLETE <<<")
