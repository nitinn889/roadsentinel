#!/usr/bin/env bash
# Launch script for RoadSentinel Driver Dashboard
# Launches ONLY the Driver User Dashboard. Does NOT start Unreal/CARLA.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " Starting RoadSentinel Driver Dashboard..."
echo " Port: 8502"
echo " Route: Continuous 2.4 km (SEG_001 - SEG_006)"
echo "=================================================="

exec .venv/bin/streamlit run driver_dashboard/app.py --server.port 8502
