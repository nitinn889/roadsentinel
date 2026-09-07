import unreal
print(">>> HELLO FROM UNREAL ENGINE PYTHON <<<")
world = unreal.EditorLevelLibrary.get_editor_world()
print(f">>> WORLD: {world.get_name() if world else 'None'} <<<")
all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
print(f">>> ACTORS IN LEVEL: {len(all_actors)} <<<")
for a in all_actors[:10]:
    print(f"    {a.get_name()} ({a.get_class().get_name()})")
