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
markings = [a for a in all_actors if a and a.actor_has_tag(unreal.Name("RS_Marking"))]
roads = [a for a in all_actors if a and a.actor_has_tag(unreal.Name("RS_Road"))]
print(f"Road actors found: {len(roads)}")
print(f"Marking actors found: {len(markings)}")

for i, m in enumerate(markings[:5]):
    loc = m.get_actor_location()
    scale = m.get_actor_scale3d()
    comp = m.static_mesh_component
    mesh = comp.static_mesh if comp else None
    mat = comp.get_material(0) if comp else None
    print(f"Marking {i}: loc=({loc.x}, {loc.y}, {loc.z}), scale=({scale.x}, {scale.y}, {scale.z}), mesh={mesh.get_name() if mesh else 'None'}, mat={mat.get_name() if mat else 'None'}")

if roads:
    r = roads[0]
    loc = r.get_actor_location()
    scale = r.get_actor_scale3d()
    comp = r.static_mesh_component
    mesh = comp.static_mesh if comp else None
    mat = comp.get_material(0) if comp else None
    print(f"Road: loc=({loc.x}, {loc.y}, {loc.z}), scale=({scale.x}, {scale.y}, {scale.z}), mesh={mesh.get_name() if mesh else 'None'}, mat={mat.get_name() if mat else 'None'}")

# Now test SceneCaptureComponent2D
cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(3500.0, -135.0, 2500.0), unreal.Rotator(0, -89.0, 0))
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
print("SceneCaptureComponent2D properties:")
for prop in ["primitive_render_mode", "capture_source", "lod_distance_factor", "max_view_distance_override"]:
    try:
        val = comp.get_editor_property(prop)
        print(f"  {prop} = {val}")
    except Exception as e:
        print(f"  {prop}: {e}")

unreal.EditorLevelLibrary.destroy_actor(cam)
print("Diagnosis complete.")
