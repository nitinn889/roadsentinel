import os
import sys
import unreal

map_path = "/Game/SimBlank/Levels/SimBlank"
unreal.EditorLoadingAndSavingUtils.load_map(map_path)
world = unreal.EditorLevelLibrary.get_editor_world()

all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
print(f"Total default actors in SimBlank: {len(all_actors)}")
for a in all_actors:
    cname = a.get_class().get_name()
    if any(k in cname.lower() or k in a.get_name().lower() for k in ["sun", "sky", "light", "post", "fog", "camera"]):
        loc = a.get_actor_location()
        rot = a.get_actor_rotation()
        print(f"  Actor: {a.get_name()} ({cname}) at Loc={loc}, Rot={rot}")
        for comp in a.get_components_by_class(unreal.ActorComponent):
            comp_cname = comp.get_class().get_name()
            if any(k in comp_cname.lower() for k in ["light", "sky", "fog", "post"]):
                print(f"    Component: {comp.get_name()} ({comp_cname})")
                if hasattr(comp, "get_editor_property"):
                    try:
                        print(f"      intensity: {comp.get_editor_property('intensity')}")
                    except Exception:
                        pass
