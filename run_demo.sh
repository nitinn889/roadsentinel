#!/usr/bin/env bash
# ==============================================================================
# RoadSentinel — Master Execution Script
# ==============================================================================
# Activates the Python virtual environment and runs the streamlined
# orchestrator pipeline. CARLA simulation is fully decoupled — the user
# captures images separately and points this script at the output folder.
#
# Usage:
#   ./run_demo.sh                                  # Use default input dir
#   ./run_demo.sh --input-dir /path/to/images      # Custom image folder
#   ./run_demo.sh --port 9000                      # Custom dashboard port
#   ./run_demo.sh --no-server                      # Skip dashboard launch
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${CYAN}==================================================================${NC}"
echo -e "${BOLD}${CYAN}       ROADSENTINEL — MASTER EXECUTION DEMO RUNNER                ${NC}"
echo -e "${BOLD}${CYAN}  Image Ingestion → Analysis → Government Dashboard              ${NC}"
echo -e "${BOLD}${CYAN}==================================================================${NC}"
echo ""

# ---------------------------------------------------------------------------
# Locate Python Environment
# ---------------------------------------------------------------------------
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

if [ -f "$PYTHON_BIN" ]; then
    echo -e "${GREEN}[✓] Python environment:${NC} $SCRIPT_DIR/.venv"
else
    echo -e "${YELLOW}[!] .venv not found — falling back to system python3.${NC}"
    PYTHON_BIN="python3"
fi

# ---------------------------------------------------------------------------
# Launch Orchestrator (all CLI args forwarded)
# ---------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}Starting RoadSentinel Orchestrator...${NC}"
echo -e "${DIM}  Python: ${PYTHON_BIN}${NC}"
echo -e "${DIM}  Args:   ${*:-<none>}${NC}"
echo ""

"$PYTHON_BIN" "$SCRIPT_DIR/orchestrator.py" "$@"
