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

# Update materials for Yellow and White markings to have Emissive Color
mel = unreal.MaterialEditingLibrary
mat_yellow = unreal.EditorAssetLibrary.load_asset("/Game/RS_Roads/Materials/M_RS_Marking_Yellow")
if mat_yellow:
    # Connect a VectorParameter to Emissive Color
    em = mel.create_material_expression(mat_yellow, unreal.MaterialExpressionVectorParameter, -400, 400)
    em.set_editor_property("parameter_name", "EmissiveColor")
    em.set_editor_property("default_value", unreal.LinearColor(0.95, 0.70, 0.04, 1.0))
    mel.connect_material_property(em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat_yellow)
    unreal.EditorAssetLibrary.save_loaded_asset(mat_yellow)
    print("Added Emissive to M_RS_Marking_Yellow")

mat_white = unreal.EditorAssetLibrary.load_asset("/Game/RS_Roads/Materials/M_RS_Marking_White")
if mat_white:
    em = mel.create_material_expression(mat_white, unreal.MaterialExpressionVectorParameter, -400, 400)
    em.set_editor_property("parameter_name", "EmissiveColor")
    em.set_editor_property("default_value", unreal.LinearColor(0.90, 0.90, 0.90, 1.0))
    mel.connect_material_property(em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat_white)
    unreal.EditorAssetLibrary.save_loaded_asset(mat_white)
    print("Added Emissive to M_RS_Marking_White")

# Adjust all markings to 25cm width, 4cm thickness, elevated to Z=2.0cm
all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
for a in all_actors:
    if a and a.actor_has_tag(unreal.Name("RS_Marking")):
        loc = a.get_actor_location()
        scale = a.get_actor_scale3d()
        if scale.x > 100.0: # yellow line or continuous fog line
            a.set_actor_location(unreal.Vector(loc.x, loc.y, 2.0), False, False)
            a.set_actor_scale3d(unreal.Vector(scale.x, max(0.20, scale.y), 0.04))
        else: # dashed lines
            a.set_actor_location(unreal.Vector(loc.x, loc.y, 2.0), False, False)
            a.set_actor_scale3d(unreal.Vector(scale.x, max(0.20, scale.y), 0.04))

loc = unreal.Vector(3500.0, -135.0, 2500.0)
rot = unreal.Rotator(0, -89.0, 0)
out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/scratch/emissive_test")
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
pps.set_editor_property("auto_exposure_bias", 2.6)
comp.set_editor_property("post_process_settings", pps)

comp.capture_scene()
time.sleep(0.3)
comp.capture_scene()

unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), "sc_emissive_markings.png")
print("Exported sc_emissive_markings.png")
unreal.EditorLevelLibrary.destroy_actor(cam)
