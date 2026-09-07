import sys
import os
from pathlib import Path
import unreal

python_dir = Path(__file__).resolve().parent
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

try:
    import rs_phase2_surface_defects
    rs_phase2_surface_defects.start_ipc_server()
    unreal.log("[✓] RoadSentinel IPC Server initialized on editor startup (Port 8899)")
except Exception as e:
    unreal.log_error(f"[!] RoadSentinel init_unreal.py error: {e}")
