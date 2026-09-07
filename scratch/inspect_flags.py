import unreal

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0,0,0), unreal.Rotator(0,0,0))
comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)

try:
    flags = comp.get_editor_property("show_flag_settings")
    print(f"Current show_flag_settings: {flags}")
    print(f"Type: {type(flags)}")
except Exception as e:
    print(f"Error reading show_flag_settings: {e}")

try:
    setting_class = getattr(unreal, "EngineShowFlagsSetting", None)
    print(f"EngineShowFlagsSetting class: {setting_class}")
    if setting_class:
        s = setting_class()
        print(f"Fields: {dir(s)}")
except Exception as e:
    print(f"Error with setting class: {e}")

unreal.EditorLevelLibrary.destroy_actor(cam)
