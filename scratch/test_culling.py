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

# Disable distance culling via console
unreal.SystemLibrary.execute_console_command(world, "r.ViewDistanceScale 10")
unreal.SystemLibrary.execute_console_command(world, "r.ForceLOD 0")

all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
for a in all_actors:
    if a and a.actor_has_tag(unreal.Name("RS_Marking")):
        mc = a.static_mesh_component
        if mc:
            try:
                mc.set_editor_property("never_distance_cull", True)
            except Exception:
                pass
            try:
                mc.set_editor_property("bounds_scale", 10.0)
            except Exception:
                pass

# Spawn an unmistakable test marker right under the camera
cube_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube")
test_cube = unreal.EditorLevelLibrary.spawn_actor_from_class(
    unreal.StaticMeshActor,
    unreal.Vector(3500.0, 0.0, 5.0),
    unreal.Rotator(0, 0, 0)
)
if test_cube:
    tc = test_cube.static_mesh_component
    tc.set_static_mesh(cube_mesh)
    tc.set_world_scale3d(unreal.Vector(5.0, 1.0, 0.1)) # 5m long x 1m wide x 10cm tall
    mat_yellow = unreal.EditorAssetLibrary.load_asset("/Game/RS_Roads/Materials/M_RS_Marking_Yellow")
    if mat_yellow:
        tc.set_material(0, mat_yellow)
    print("Spawned test marker under camera!")

loc = unreal.Vector(3500.0, -135.0, 2500.0)
rot = unreal.Rotator(0, -89.0, 0)

out_dir = Path("/home/nitin-nandakumar/Downloads/roadsentinel/scratch")
target_w, target_h = 1920, 1080
render_format = getattr(unreal.TextureRenderTargetFormat, "RTF_RGBA8_SRGB", unreal.TextureRenderTargetFormat.RTF_RGBA8)
target = unreal.RenderingLibrary.create_render_target2d(world, target_w, target_h, render_format)

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
comp.set_editor_property("texture_target", target)
comp.set_editor_property("fov_angle", 90.0)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("always_persist_rendering_state", True)
comp.set_editor_property("inherit_main_view_camera_post_process_settings", True)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("lod_distance_factor", 0.01)

pps = comp.get_editor_property("post_process_settings")
pps.set_editor_property("override_auto_exposure_bias", True)
pps.set_editor_property("auto_exposure_bias", 2.6)
comp.set_editor_property("post_process_settings", pps)

comp.capture_scene()
time.sleep(0.3)
comp.capture_scene()

sc_path = out_dir / "cull_test.png"
unreal.RenderingLibrary.export_render_target(world, target, str(out_dir), sc_path.name)
print("Saved cull_test.png")

unreal.EditorLevelLibrary.destroy_actor(cam)
if test_cube:
    unreal.EditorLevelLibrary.destroy_actor(test_cube)
