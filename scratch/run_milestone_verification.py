import os
import sys
import time
import json
from pathlib import Path
import unreal

print("=" * 60)
print(">>> MILESTONE 1 VERIFICATION: SEG_001 (Day 1, Day 5, Day 10) <<<")
print("=" * 60)

map_path = "/Game/SimBlank/Levels/SimBlank"
unreal.EditorLoadingAndSavingUtils.load_map(map_path)
world = unreal.EditorLevelLibrary.get_editor_world()

script_dir = "/home/nitin-nandakumar/Downloads/roadsentinel/RoadSentinelSim/Content/Python"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import rs_phase2_surface_defects as phase2

# Test Milestone 1 states: Day 1, Day 5, Day 10 for SEG_001
for day in [1, 5, 10]:
    print(f"\n--- Testing SEG_001 Day {day:02d} ---")
    state = phase2.materialize_segment_day("SEG_001", day)
    print(f"  Materialised: {state['road_segment_id']} Day {state['day']} ({state['deterioration_state']})")
    print(f"  Defects count: {len(state['defects'])}, Severity: {state['renderer_condition_score']:.2f}")
    print(f"  Camera pose: {state['camera_pose']}")
    
    # Capture via _ipc_inspection_capture
    res = phase2._ipc_inspection_capture(day, "SEG_001")
    print(f"  Capture Result: {res}")
    
    # Verify image exists and size > 2KB
    img_path = Path(res["path"])
    assert img_path.is_file(), f"Capture file missing: {img_path}"
    assert img_path.stat().st_size > 2000, f"Capture file too small: {img_path.stat().st_size}"
    print(f"  [✓] Verified capture file: {img_path} ({img_path.stat().st_size} bytes)")

print("\n>>> ALL MILESTONE 1 CAPTURES VERIFIED SUCCESSFULLY <<<")
