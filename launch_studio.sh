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

if [ -f "${PROJECT_DIR}/carla_env/bin/python3" ]; then
    PYTHON_EXE="${PROJECT_DIR}/carla_env/bin/python3"
else
    PYTHON_EXE="python3"
fi

"${PYTHON_EXE}" "${PROJECT_DIR}/env/scripts/rs_interactive_studio.py" "$@"

