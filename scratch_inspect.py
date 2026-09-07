import json
import unreal

def inspect_level():
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = subsystem.get_all_level_actors()
    actor_info = []
    
    dir_lights = []
    sky_lights = []
    post_process = []
    potholes = []
    triggers = []
    road_sections = []
    
    for a in actors:
        info = {
            "name": a.get_name(),
            "label": a.get_actor_label(),
            "class": a.get_class().get_name(),
            "location": [round(c, 2) for c in a.get_actor_location().to_tuple()],
            "rotation": [round(c, 2) for c in a.get_actor_rotation().to_tuple()],
            "tags": [str(t) for t in a.tags],
            "folder": str(a.get_folder_path())
        }
        actor_info.append(info)
        
        cname = a.get_class().get_name()
        if "DirectionalLight" in cname or "SunSky" in cname or "Light" in cname:
            dir_lights.append(info)
        if "SkyLight" in cname:
            sky_lights.append(info)
        if "PostProcess" in cname:
            post_process.append(info)
        if "Pothole" in info["label"] or "pothole" in info["label"].lower():
            potholes.append(info)
        if "Trigger" in cname or "trigger" in info["label"].lower():
            triggers.append(info)
        if "RoadSection" in info["label"]:
            road_sections.append(info)
            
    print("=== SUMMARY OF LEVEL ===")
    print(f"Total Actors: {len(actors)}")
    print(f"Road Sections: {len(road_sections)}")
    for r in road_sections:
        print(f"  {r['label']} @ {r['location']}")
    print(f"Triggers: {len(triggers)}")
    for t in triggers:
        print(f"  {t['label']} @ {t['location']} tags={t['tags']}")
    print(f"Directional / Sun Lights:")
    for l in dir_lights:
        print(f"  {l['label']} ({l['class']}) @ {l['location']} rot={l['rotation']}")
    print(f"Sky Lights:")
    for s in sky_lights:
        print(f"  {s['label']} ({s['class']})")
    print(f"Existing Pothole Actors: {len(potholes)}")
    for p in potholes[:10]:
        print(f"  {p['label']} @ {p['location']}")
    print(f"Post Process: {len(post_process)}")
    for pp in post_process:
        print(f"  {pp['label']}")

    # Also list assets in Content
    asset_reg = unreal.AssetRegistryHelpers.get_asset_registry()
    assets = asset_reg.get_assets_by_path("/Game", recursive=True)
    print(f"Total assets under /Game: {len(assets)}")
    mesh_assets = [a.package_name for a in assets if a.asset_class_path.asset_name == unreal.Name("StaticMesh")]
    print(f"StaticMeshes under /Game: {len(mesh_assets)}")
    for m in mesh_assets:
        print(f"  Mesh: {m}")
        
    mat_assets = [a.package_name for a in assets if a.asset_class_path.asset_name in (unreal.Name("Material"), unreal.Name("MaterialInstanceConstant"))]
    print(f"Materials under /Game: {len(mat_assets)}")
    for m in mat_assets:
        print(f"  Material: {m}")

inspect_level()
