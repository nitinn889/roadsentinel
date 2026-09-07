import os
import sys
import time
from pathlib import Path
import unreal

print("=" * 60)
print(">>> INSPECT PPS PROPS & CAMERA EXPOSURE <<<")
print("=" * 60)

map_path = "/Game/SimBlank/Levels/SimBlank"
unreal.EditorLoadingAndSavingUtils.load_map(map_path)
world = unreal.EditorLevelLibrary.get_editor_world()

pps = unreal.PostProcessSettings()
props = [p for p in dir(pps) if not p.startswith("_")]
print(f"Total PPS properties: {len(props)}")

exposure_props = [p for p in props if any(k in p.lower() for k in ["exposure", "camera", "iso", "shutter", "fstop", "aperture", "bias"])]
print("Exposure-related PPS properties:")
for ep in sorted(exposure_props):
    try:
        val = pps.get_editor_property(ep)
        print(f"  {ep} = {val}")
    except Exception as e:
        print(f"  {ep} (error: {e})")

# Check SceneCaptureComponent2D properties
comp = unreal.SceneCaptureComponent2D()
comp_props = [p for p in dir(comp) if not p.startswith("_")]
comp_exposure = [p for p in comp_props if any(k in p.lower() for k in ["exposure", "persist", "view", "state", "flag", "source"])]
print("SceneCaptureComponent2D relevant properties:")
for cp in sorted(comp_exposure):
    try:
        val = comp.get_editor_property(cp)
        print(f"  {cp} = {val}")
    except Exception as e:
        print(f"  {cp} (error: {e})")

print(">>> INSPECTION COMPLETE <<<")
