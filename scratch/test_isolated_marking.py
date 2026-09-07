import os
import sys
import time
from pathlib import Path
import unreal

print("=" * 60)
print(">>> TEST ISOLATED MARKING ACTOR IN SCENECAPTURE2D <<<")
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
yellow_marking = next(a for a in all_actors if a.get_name() == "StaticMeshActor_3")
print(f"Found yellow marking: {yellow_marking.get_name()}")
print(f"  Location: {yellow_marking.get_actor_location()}")
print(f"  Scale: {yellow_marking.get_actor_scale3d()}")
print(f"  Material: {yellow_marking.static_mesh_component.get_material(0).get_name()}")
print(f"  Hidden in game: {yellow_marking.is_hidden_ed()}")

seg1 = next(s for s in scene["segments"] if s["road_segment_id"] == "SEG_001")
camera_pose = seg1["camera_pose"]
loc = unreal.Vector(float(camera_pose["x_cm"]), float(camera_pose["y_cm"]), float(camera_pose["z_cm"]))
rot = unreal.Rotator(roll=0.0, pitch=float(camera_pose["pitch_deg"]), yaw=float(camera_pose["yaw_deg"]))

out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test/marking_test")
out_dir.mkdir(parents=True, exist_ok=True)

target_w, target_h = 1920, 1080
render_format = getattr(unreal.TextureRenderTargetFormat, "RTF_RGBA8_SRGB", unreal.TextureRenderTargetFormat.RTF_RGBA8)
target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
comp.set_editor_property("texture_target", target)
comp.set_editor_property("fov_angle", 90.0)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("always_persist_rendering_state", True)
comp.set_editor_property("inherit_main_view_camera_post_process_settings", True)
pps = comp.get_editor_property("post_process_settings")
pps.set_editor_property("override_auto_exposure_bias", True)
pps.set_editor_property("auto_exposure_bias", 2.2)
comp.set_editor_property("post_process_settings", pps)

# Test 1: Capture ONLY the yellow marking actor
print("Testing Test 1: Show ONLY yellow marking...")
comp.set_editor_property("primitive_render_mode", unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST)
comp.set_editor_property("show_only_actors", [yellow_marking])
comp.capture_scene()
time.sleep(0.2)
comp.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "only_yellow_marking.png")

# Test 2: Show road + yellow marking
print("Testing Test 2: Show road + yellow marking...")
road_actor = next(a for a in all_actors if a.get_name() == "StaticMeshActor_2")
comp.set_editor_property("show_only_actors", [yellow_marking, road_actor])
comp.capture_scene()
time.sleep(0.2)
comp.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "road_and_marking.png")

# Test 3: What if road is slightly lowered by 1 cm (e.g. z = -11 instead of -10)?
print("Testing Test 3: Road lowered by 1 cm...")
road_loc = road_actor.get_actor_location()
# Change road component location z
road_actor.static_mesh_component.set_world_location(unreal.Vector(road_loc.x, road_loc.y, road_loc.z - 1.0), False, False)
comp.set_editor_property("primitive_render_mode", unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES)
comp.capture_scene()
time.sleep(0.2)
comp.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "road_lowered_1cm.png")

unreal.EditorLevelLibrary.destroy_actor(cam)
print(">>> ISOLATED MARKING TEST COMPLETE <<<")
