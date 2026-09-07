import os
import sys
import unreal

map_path = "/Game/SimBlank/Levels/SimBlank"
unreal.EditorLoadingAndSavingUtils.load_map(map_path)
world = unreal.EditorLevelLibrary.get_editor_world()

for a in unreal.EditorLevelLibrary.get_all_level_actors():
    cname = a.get_class().get_name()
    if cname == "StaticMeshActor":
        loc = a.get_actor_location()
        scale = a.get_actor_scale3d()
        mat = a.static_mesh_component.get_material(0) if a.static_mesh_component else None
        print(f"StaticMeshActor: {a.get_name()}, Loc: {loc}, Scale: {scale}, Mat: {mat.get_name() if mat else 'None'}")
