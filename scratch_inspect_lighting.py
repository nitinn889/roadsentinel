import unreal

subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actors = subsystem.get_all_level_actors()

for a in actors:
    cname = a.get_class().get_name()
    label = a.get_actor_label()
    if "Sun" in label or "SunSky" in cname or "Light" in cname:
        print(f"Actor: {label} ({cname})")
        for comp in a.get_components_by_class(unreal.ActorComponent):
            print(f"  Component: {comp.get_name()} ({comp.get_class().get_name()})")
            if isinstance(comp, unreal.DirectionalLightComponent):
                print(f"    Intensity: {comp.get_editor_property('intensity')}")
                print(f"    LightColor: {comp.get_editor_property('light_color')}")
                print(f"    Rotation: {comp.get_editor_property('relative_rotation')}")
            elif isinstance(comp, unreal.SkyLightComponent):
                print(f"    Intensity: {comp.get_editor_property('intensity')}")
                print(f"    LowerHemisphereColor: {comp.get_editor_property('lower_hemisphere_color')}")
                print(f"    RealTimeCapture: {comp.get_editor_property('real_time_capture')}")

print("\n--- Check materials ---")
for mat_name in ["M_Road_Seg1", "M_Road_Seg2", "M_Road_Seg3", "M_Road_Seg4", "M_Road_Seg5", "M_Water_Pothole", "M_Road_PotholeWater"]:
    for prefix in ["/Game/RS_Roads/Materials", "/Game/Materials", "/Game"]:
        path = f"{prefix}/{mat_name}"
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            print(f"Found material: {path}")

print("\n--- Check existing foliage / bush meshes ---")
reg = unreal.AssetRegistryHelpers.get_asset_registry()
engine_meshes = reg.get_assets_by_path("/Engine", recursive=True)
print(f"Engine assets found: {len(engine_meshes)}")
for m in engine_meshes:
    name = str(m.asset_name)
    if "bush" in name.lower() or "shrub" in name.lower() or "tree" in name.lower() or "foliage" in name.lower():
        print(f"Engine Foliage Mesh: {m.package_name}")

game_meshes = reg.get_assets_by_path("/Game", recursive=True)
for m in game_meshes:
    name = str(m.asset_name)
    if "bush" in name.lower() or "shrub" in name.lower():
        print(f"Game Foliage Mesh: {m.package_name}")
