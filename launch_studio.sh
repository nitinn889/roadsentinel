#!/usr/bin/env bash
# ==============================================================================
# launch_studio.sh
# ----------------
# Launches the RoadSentinel 3D Studio & Drone Controller GUI.
# ==============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "======================================================================"
echo " RoadSentinel: 3D Highway Studio & Drone Flight Controller"
echo "======================================================================"

# The historical no-argument behaviour opens only the PySide controller.  The
# temporal mode below launches the actual RoadSentinelSim Unreal project and
# uses its SceneCapture2D renderer; it does not talk to CARLA.
if [ "${1:-}" = "--temporal-20-day" ]; then
    shift
    if [ -x "${PROJECT_DIR}/.venv/bin/python" ]; then
        TEMPORAL_PYTHON="${PROJECT_DIR}/.venv/bin/python"
    else
        TEMPORAL_PYTHON="python3"
    fi
    echo "[*] Starting the 20-day Unreal Engine capture and RoadSentinel pipeline..."
    exec "${TEMPORAL_PYTHON}" "${PROJECT_DIR}/run_20_day_simulation.py" --unreal-gui "$@"
fi

if [ -f "${PROJECT_DIR}/carla_env/bin/python3" ]; then
    PYTHON_EXE="${PROJECT_DIR}/carla_env/bin/python3"
else
    PYTHON_EXE="python3"
fi

"${PYTHON_EXE}" "${PROJECT_DIR}/env/scripts/rs_interactive_studio.py" "$@"
