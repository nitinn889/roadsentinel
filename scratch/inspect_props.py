import unreal

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0,0,0), unreal.Rotator(0,0,0))
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)

print("=== SceneCaptureComponent2D Properties & Methods ===")
props = dir(comp)
for p in sorted(props):
    if not p.startswith("_"):
        print(p)

unreal.EditorLevelLibrary.destroy_actor(cam)
