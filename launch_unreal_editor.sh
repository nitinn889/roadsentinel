#!/usr/bin/env bash
# ==============================================================================
# launch_unreal_editor.sh
# -----------------------
# Launches the Unreal Engine 5.8 Editor for the RoadSentinelSim project.
# Automatically verifies that /mnt/bigdata is mounted and starts the 3D viewport.
# ==============================================================================

set -e

PROJECT_DIR="/home/nitin-nandakumar/Downloads/roadsentinel"
UPROJECT="${PROJECT_DIR}/RoadSentinelSim/RoadSentinelSim.uproject"
UE_EDITOR="/mnt/bigdata/unreal_engine/Engine/Binaries/Linux/UnrealEditor"

echo "======================================================================"
echo " RoadSentinel: Unreal Engine 5.8 3D Environment Launcher"
echo "======================================================================"

# 1. Check if /mnt/bigdata is mounted
if [ ! -f "${UE_EDITOR}" ]; then
    echo "[!] /mnt/bigdata is not mounted or Unreal Engine 5.8 is not accessible."
    echo "[*] Mounting /dev/nvme0n1p3 to /mnt/bigdata..."
    sudo mkdir -p /mnt/bigdata
    sudo mount -t ntfs-3g -o remove_hiberfile,rw,uid=1000,gid=1000 /dev/nvme0n1p3 /mnt/bigdata
fi

# Verify again
if [ ! -f "${UE_EDITOR}" ]; then
    echo "[ERROR] Could not find UnrealEditor at ${UE_EDITOR} even after mount."
    exit 1
fi

echo "[✓] Unreal Engine 5.8 verified at: ${UE_EDITOR}"
echo "[✓] Project target: ${UPROJECT}"
echo "[*] Launching Unreal Engine 5.8 3D Editor GUI..."

# 2. Launch Unreal Engine Editor GUI
"${UE_EDITOR}" "${UPROJECT}" "$@"
