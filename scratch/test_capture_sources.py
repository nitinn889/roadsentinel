import os
import sys
import time
from pathlib import Path
import unreal

print("=" * 60)
print(">>> INSPECT SCENECAPTURE COMPONENT & ACTORS <<<")
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

# Let's inspect all actors tagged RS_Marking
all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
markings = [a for a in all_actors if a.actor_has_tag(unreal.Name("RS_Marking"))]
print(f"Total RS_Marking actors in level: {len(markings)}")
for m in markings[:5]:
    loc = m.get_actor_location()
    scale = m.get_actor_scale3d()
    mat = m.static_mesh_component.get_material(0) if m.static_mesh_component else None
    print(f"  Marking: {m.get_name()}, Loc: {loc}, Scale: {scale}, Mat: {mat.get_name() if mat else 'None'}")

roads = [a for a in all_actors if a.actor_has_tag(unreal.Name("RS_Road"))]
print(f"Total RS_Road actors in level: {len(roads)}")
for r in roads[:2]:
    loc = r.get_actor_location()
    scale = r.get_actor_scale3d()
    mat = r.static_mesh_component.get_material(0) if r.static_mesh_component else None
    print(f"  Road: {r.get_name()}, Loc: {loc}, Scale: {scale}, Mat: {mat.get_name() if mat else 'None'}")

seg1 = next(s for s in scene["segments"] if s["road_segment_id"] == "SEG_001")
camera_pose = seg1["camera_pose"]
loc = unreal.Vector(float(camera_pose["x_cm"]), float(camera_pose["y_cm"]), float(camera_pose["z_cm"]))
rot = unreal.Rotator(roll=0.0, pitch=float(camera_pose["pitch_deg"]), yaw=float(camera_pose["yaw_deg"]))

out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/env/output/smoke_test/inspect_capture")
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

# Print all capture sources available
for cs in dir(unreal.SceneCaptureSource):
    if cs.startswith("SCS_"):
        print(f"Capture source option: {cs}")

# Test 1: SCS_BASE_COLOR (Unlit albedo)
print("Testing SCS_BASE_COLOR...")
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_BASE_COLOR)
comp.capture_scene()
time.sleep(0.3)
comp.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "source_base_color.png")

# Test 2: SCS_SCENE_COLOR_HDR
print("Testing SCS_SCENE_COLOR_HDR...")
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR)
comp.capture_scene()
time.sleep(0.3)
comp.capture_scene()
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "source_scene_color_hdr.png")

# Test 3: SCS_FINAL_COLOR_HDR
print("Testing SCS_FINAL_COLOR_HDR...")
try:
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR)
    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "source_final_color_hdr.png")
except Exception as e:
    print(f"SCS_FINAL_COLOR_HDR failed: {e}")

# Test 4: SCS_FINAL_TONE_CURVE_HDR
print("Testing SCS_FINAL_TONE_CURVE_HDR...")
try:
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_TONE_CURVE_HDR)
    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "source_final_tone_curve_hdr.png")
except Exception as e:
    print(f"SCS_FINAL_TONE_CURVE_HDR failed: {e}")

unreal.EditorLevelLibrary.destroy_actor(cam)
print(">>> INSPECTION COMPLETE <<<")
