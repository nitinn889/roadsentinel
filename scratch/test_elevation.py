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

all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
# Elevate all RS_Marking actors to Z = 3.0 cm with thickness 4.0 cm (scale Z = 0.04)
# So bottom is at 1.0 cm, top is at 5.0 cm -> 100% clear of road surface at Z=0.0!
for a in all_actors:
    if a and a.actor_has_tag(unreal.Name("RS_Marking")):
        loc = a.get_actor_location()
        scale = a.get_actor_scale3d()
        a.set_actor_location(unreal.Vector(loc.x, loc.y, 3.0), False, False)
        a.set_actor_scale3d(unreal.Vector(scale.x, scale.y, 0.04))

loc = unreal.Vector(3500.0, -135.0, 2500.0)
rot = unreal.Rotator(0, -89.0, 0)
out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/scratch/elevation_test")
out_dir.mkdir(parents=True, exist_ok=True)

target_w, target_h = 1920, 1080
render_format = getattr(unreal.TextureRenderTargetFormat, "RTF_RGBA8_SRGB", unreal.TextureRenderTargetFormat.RTF_RGBA8)
target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)

# Test with different exposure biases: 3.0 and 3.6
for bias in [3.0, 3.6]:
    cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", target)
    comp.set_editor_property("fov_angle", 90.0)
    comp.set_editor_property("capture_every_frame", False)
    comp.set_editor_property("always_persist_rendering_state", True)
    comp.set_editor_property("inherit_main_view_camera_post_process_settings", True)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)

    pps = comp.get_editor_property("post_process_settings")
    pps.set_editor_property("override_auto_exposure_bias", True)
    pps.set_editor_property("auto_exposure_bias", bias)
    comp.set_editor_property("post_process_settings", pps)

    comp.capture_scene()
    time.sleep(0.3)
    comp.capture_scene()

    fname = f"elevated_bias_{bias}.png"
    unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), fname)
    print(f"Exported {fname}")
    unreal.EditorLevelLibrary.destroy_actor(cam)

print("Elevation test complete.")
