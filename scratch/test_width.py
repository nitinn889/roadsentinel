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

cube_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube")
mat_yellow = unreal.EditorAssetLibrary.load_asset("/Game/RS_Roads/Materials/M_RS_Marking_Yellow")
mat_white = unreal.EditorAssetLibrary.load_asset("/Game/RS_Roads/Materials/M_RS_Marking_White")

# Adjust all markings to 30cm width, 4cm thickness, elevated to Z=2.5cm
all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
for a in all_actors:
    if a and a.actor_has_tag(unreal.Name("RS_Marking")):
        loc = a.get_actor_location()
        scale = a.get_actor_scale3d()
        # If it's a yellow line (scale.x > 100)
        if scale.x > 100.0:
            a.set_actor_location(unreal.Vector(loc.x, loc.y, 2.5), False, False)
            a.set_actor_scale3d(unreal.Vector(scale.x, 0.30, 0.04))
        else: # dashed or fog line
            a.set_actor_location(unreal.Vector(loc.x, loc.y, 2.5), False, False)
            a.set_actor_scale3d(unreal.Vector(scale.x, max(0.25, scale.y), 0.04))

loc = unreal.Vector(3500.0, -135.0, 2500.0)
rot = unreal.Rotator(0, -89.0, 0)
out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/scratch/width_test")
out_dir.mkdir(parents=True, exist_ok=True)

target = unreal.RenderingLibrary.create_render_target2d(world, 1920, 1080, unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
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
pps.set_editor_property("auto_exposure_bias", 3.0)
comp.set_editor_property("post_process_settings", pps)

comp.capture_scene()
time.sleep(0.3)
comp.capture_scene()

unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "sc_wide_markings.png")
print("Exported sc_wide_markings.png")
unreal.EditorLevelLibrary.destroy_actor(cam)
