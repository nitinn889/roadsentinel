#!/usr/bin/env python3
"""
RoadSentinel Simulation Interactive Launcher
=============================================
Provides a pre-CARLA interactive terminal UI (TUI) with dropdown-style selectors
for configuring simulation scenarios, weather, camera altitude, drone speed,
water conditions, procedural seed, flight duration, and rendering mode.

Handles:
- Pre-CARLA TUI menu displayed BEFORE CARLA opens
- Input validation and session configuration persistence
- Docker container detection and automatic startup (carlasim/carla:0.9.16)
- CARLA server readiness polling on port 2000
- Launching the procedural road flight simulation
- Fallback to standalone procedural simulation if needed
- Full non-interactive CLI argument support for automated experiments
"""

import argparse
import curses
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "env" / "output"

SCENARIO_OPTIONS = ["poor", "healthy", "moderate", "critical"]
WEATHER_OPTIONS = ["post_rain", "clear", "overcast", "wet", "rain", "low_light", "sunset", "early_morning"]
ALTITUDE_OPTIONS = [100.0, 80.0, 90.0, 120.0]
SPEED_OPTIONS = [30.0, 20.0, 10.0]
WATER_OPTIONS = ["Automatic by Scenario", "Mostly Dry", "Mixed", "Mostly Water-Filled"]
SEED_OPTIONS = ["42", "Random", "100", "2026", "Custom..."]
DURATION_OPTIONS = [60.0, 30.0, 120.0, 0.0]  # 0.0 = Continuous / infinite
RENDERING_OPTIONS = ["GUI", "Headless"]

WATER_RATIO_MAP: Dict[str, Optional[float]] = {
    "Automatic by Scenario": None,
    "Mostly Dry": 0.10,
    "Mixed": 0.50,
    "Mostly Water-Filled": 0.85,
}


def find_carla_python() -> str:
    """Locates the Python binary that contains the CARLA package."""
    carla_env = WORKSPACE_DIR / "carla_env" / "bin" / "python"
    if carla_env.exists() and os.access(carla_env, os.X_OK):
        return str(carla_env)
    venv_py = WORKSPACE_DIR / ".venv" / "bin" / "python"
    if venv_py.exists() and os.access(venv_py, os.X_OK):
        return str(venv_py)
    return sys.executable


def is_port_open(host: str = "127.0.0.1", port: int = 2000, timeout: float = 1.0) -> bool:
    """Checks if a TCP port is actively accepting connections."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        res = sock.connect_ex((host, port))
        return res == 0
    except Exception:
        return False
    finally:
        sock.close()


def check_carla_running(host: str = "127.0.0.1", port: int = 2000, timeout: float = 1.5) -> bool:
    """Verifies whether CARLA RPC server is reachable."""
    return is_port_open(host, port, timeout=timeout)


def wait_for_carla_ready(host: str = "127.0.0.1", port: int = 2000, max_wait_sec: float = 35.0, poll_interval: float = 1.0) -> bool:
    """Polls CARLA port until ready or timeout expires."""
    print(f"[RoadSentinel] Waiting for CARLA server on {host}:{port}...", end="", flush=True)
    start = time.time()
    while (time.time() - start) < max_wait_sec:
        if is_port_open(host, port, timeout=poll_interval):
            print(" Connected!")
            return True
        print(".", end="", flush=True)
        time.sleep(poll_interval)
    print(" Timed out.")
    return False


def start_carla_docker(headless: bool = False, host: str = "127.0.0.1", port: int = 2000) -> bool:
    """
    Attempts to start the CARLA 0.9.16 Docker container if not already running.
    Preserves NVIDIA GPU runtime and OpenGL GUI rendering.
    """
    if check_carla_running(host, port):
        print(f"[RoadSentinel] CARLA 0.9.16 is already active on {host}:{port}.")
        return True

    # Check if docker CLI is available
    if not shutil.which("docker"):
        print("[RoadSentinel] Warning: 'docker' CLI not found. Unable to auto-start container.")
        return False

    # Check if a container named carla_sim exists (running or stopped)
    try:
        inspect = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=^carla_sim$", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, check=True
        )
        output = inspect.stdout.strip()
        if "carla_sim" in output:
            if "Up" in output:
                print("[RoadSentinel] Container 'carla_sim' is running. Polling port readiness...")
                return wait_for_carla_ready(host, port, max_wait_sec=20.0)
            else:
                print("[RoadSentinel] Existing 'carla_sim' container stopped. Starting it...")
                subprocess.run(["docker", "start", "carla_sim"], check=True)
                return wait_for_carla_ready(host, port, max_wait_sec=30.0)
    except Exception as exc:
        print(f"[RoadSentinel] Docker inspect error: {exc}")

    # If no existing container, launch fresh docker container
    print("[RoadSentinel] Launching fresh CARLA 0.9.16 Docker container...")
    display = os.environ.get("DISPLAY", ":0")
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000")

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", "carla_sim",
        "--runtime=nvidia",
        "--gpus", "all",
        "--net=host",
        "-e", f"DISPLAY={display}",
        "-e", f"XDG_RUNTIME_DIR={xdg_runtime}",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
        "-v", f"{xdg_runtime}:{xdg_runtime}",
        "carlasim/carla:0.9.16",
        "bash", "CarlaUE4.sh", "-opengl", "-quality-level=Low", "-nosound"
    ]
    if headless:
        cmd.extend(["-RenderOffScreen"])

    try:
        subprocess.run(cmd, check=True)
        return wait_for_carla_ready(host, port, max_wait_sec=35.0)
    except Exception as exc:
        print(f"[RoadSentinel] Failed to launch CARLA docker container: {exc}")
        return False


def validate_configuration(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates configuration parameters, normalizes strings, and enforces bounds."""
    validated = dict(cfg)

    scenario = str(validated.get("scenario", "poor")).lower().strip()
    if scenario not in SCENARIO_OPTIONS:
        raise ValueError(f"Invalid scenario '{scenario}'. Allowed: {SCENARIO_OPTIONS}")
    validated["scenario"] = scenario

    weather = str(validated.get("weather", "post_rain")).lower().strip().replace("-", "_")
    if weather not in WEATHER_OPTIONS:
        raise ValueError(f"Invalid weather '{weather}'. Allowed: {WEATHER_OPTIONS}")
    validated["weather"] = weather

    alt = float(validated.get("altitude", 100.0))
    if alt <= 0.0 or alt > 500.0:
        raise ValueError(f"Altitude {alt}m out of bounds (0, 500]")
    validated["altitude"] = alt

    speed = float(validated.get("speed", 30.0))
    if speed <= 0.0 or speed > 200.0:
        raise ValueError(f"Speed {speed} km/h out of bounds (0, 200]")
    validated["speed"] = speed

    water = validated.get("water", "Automatic by Scenario")
    if water not in WATER_OPTIONS and not isinstance(water, (int, float)):
        raise ValueError(f"Invalid water setting '{water}'. Allowed: {WATER_OPTIONS}")
    validated["water"] = water

    seed_val = validated.get("seed", 42)
    if seed_val is not None and str(seed_val).lower() != "random":
        try:
            validated["seed"] = int(seed_val)
        except ValueError:
            raise ValueError(f"Seed '{seed_val}' must be an integer or 'Random'")
    else:
        validated["seed"] = None

    dur = float(validated.get("duration", 60.0))
    if dur < 0:
        raise ValueError(f"Duration cannot be negative: {dur}")
    validated["duration"] = dur

    rend = str(validated.get("rendering", "GUI")).strip()
    if rend.lower() in ("headless", "true", "1"):
        validated["rendering"] = "Headless"
        validated["headless"] = True
    else:
        validated["rendering"] = "GUI"
        validated["headless"] = False

    return validated


def save_session_metadata(cfg: Dict[str, Any], output_dir: Path) -> Path:
    """Saves session configuration to output directory for traceability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "session_metadata.json"
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": cfg,
        "environment": {
            "workspace": str(WORKSPACE_DIR),
            "python_executable": sys.executable,
        }
    }
    with open(meta_path, "w") as f:
        json.dump(session_data, f, indent=2)
    return meta_path


def build_drone_cmd(cfg: Dict[str, Any], output_dir: Path, standalone: bool = False) -> List[str]:
    """Constructs the command arguments to invoke env/drone_sim.py."""
    python_bin = find_carla_python()
    script_path = str(WORKSPACE_DIR / "env" / "drone_sim.py")

    cmd = [
        python_bin, script_path,
        "--scenario", cfg["scenario"],
        "--weather", cfg["weather"],
        "--altitude", str(cfg["altitude"]),
        "--speed", str(cfg["speed"]),
        "--duration", str(cfg["duration"]),
        "--output-dir", str(output_dir),
    ]

    if cfg.get("seed") is not None:
        cmd.extend(["--seed", str(cfg["seed"])])

    water_str = cfg.get("water", "Automatic by Scenario")
    if water_str in WATER_RATIO_MAP and WATER_RATIO_MAP[water_str] is not None:
        cmd.extend(["--water-ratio", str(WATER_RATIO_MAP[water_str])])
    elif isinstance(water_str, (int, float)):
        cmd.extend(["--water-ratio", str(water_str)])

    if cfg.get("auto_fly", True):
        cmd.append("--auto-fly")

    if cfg.get("headless", False):
        cmd.append("--headless")

    if standalone:
        cmd.append("--standalone")

    return cmd


class TUIConfigMenu:
    """
    Curses-based interactive configuration menu providing dropdown-style selection
    BEFORE CARLA is launched or opened.
    """

    def __init__(self):
        self.scenario_idx = 0       # Default: "poor"
        self.weather_idx = 0        # Default: "post_rain"
        self.altitude_idx = 0       # Default: 100.0
        self.speed_idx = 0          # Default: 30.0
        self.water_idx = 0          # Default: "Automatic by Scenario"
        self.seed_idx = 0           # Default: "42"
        self.custom_seed: int = 42
        self.duration_idx = 0       # Default: 60.0
        self.rendering_idx = 0      # Default: "GUI"

        self.current_field = 0
        self.total_fields = 9       # 8 config items + 1 start button

    def get_config_dict(self) -> Dict[str, Any]:
        seed_opt = SEED_OPTIONS[self.seed_idx]
        if seed_opt == "Random":
            seed_val = None
        elif seed_opt == "Custom...":
            seed_val = self.custom_seed
        else:
            seed_val = int(seed_opt)

        return {
            "scenario": SCENARIO_OPTIONS[self.scenario_idx],
            "weather": WEATHER_OPTIONS[self.weather_idx],
            "altitude": ALTITUDE_OPTIONS[self.altitude_idx],
            "speed": SPEED_OPTIONS[self.speed_idx],
            "water": WATER_OPTIONS[self.water_idx],
            "seed": seed_val,
            "duration": DURATION_OPTIONS[self.duration_idx],
            "rendering": RENDERING_OPTIONS[self.rendering_idx],
        }

    def run(self, stdscr) -> Optional[Dict[str, Any]]:
        curses.curs_set(0)
        stdscr.clear()
        stdscr.keypad(True)

        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)     # Titles & Accents
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)  # Selected Item Highlight
            curses.init_pair(3, curses.COLOR_GREEN, -1)    # Values / Success
            curses.init_pair(4, curses.COLOR_YELLOW, -1)   # Buttons / Warnings
            curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Launch Button Highlight

        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()

            if h < 24 or w < 65:
                msg = "Please enlarge terminal window (min 65x24)."
                stdscr.addstr(max(0, h // 2), max(0, (w - len(msg)) // 2), msg)
                stdscr.refresh()
                key = stdscr.getch()
                if key in (ord('q'), ord('Q'), 27):
                    return None
                continue

            # Header
            title_1 = "=========================================================="
            title_2 = "           ROADSENTINEL SIMULATOR CONFIGURATION           "
            title_3 = "=========================================================="
            stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(1, max(0, (w - len(title_1)) // 2), title_1)
            stdscr.addstr(2, max(0, (w - len(title_2)) // 2), title_2)
            stdscr.addstr(3, max(0, (w - len(title_3)) // 2), title_3)
            stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

            # Instructions
            hint = "[UP/DOWN] Navigate   [LEFT/RIGHT] Cycle Options   [ENTER] Select / Start"
            stdscr.addstr(5, max(0, (w - len(hint)) // 2), hint, curses.A_DIM)

            # Menu fields
            dur_label = f"{int(DURATION_OPTIONS[self.duration_idx])} sec" if DURATION_OPTIONS[self.duration_idx] > 0 else "Continuous (ESC)"
            seed_label = str(self.custom_seed) if SEED_OPTIONS[self.seed_idx] == "Custom..." else SEED_OPTIONS[self.seed_idx]

            fields = [
                ("Scenario", SCENARIO_OPTIONS[self.scenario_idx].upper()),
                ("Weather", WEATHER_OPTIONS[self.weather_idx].upper().replace("_", " ")),
                ("Altitude", f"{int(ALTITUDE_OPTIONS[self.altitude_idx])} m"),
                ("Drone Speed", f"{int(SPEED_OPTIONS[self.speed_idx])} km/h"),
                ("Pothole Water", WATER_OPTIONS[self.water_idx]),
                ("Seed", seed_label),
                ("Duration", dur_label),
                ("Rendering", RENDERING_OPTIONS[self.rendering_idx]),
            ]

            start_y = 7
            label_col = max(4, (w - 55) // 2)
            val_col = label_col + 24

            for idx, (label, val) in enumerate(fields):
                y = start_y + idx
                is_selected = (self.current_field == idx)
                if is_selected:
                    stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                    stdscr.addstr(y, label_col, f" > {label:<18} : < {val:<18} > ")
                    stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
                else:
                    stdscr.addstr(y, label_col, f"   {label:<18} : ")
                    stdscr.attron(curses.color_pair(3))
                    stdscr.addstr(y, val_col, f"  {val}")
                    stdscr.attroff(curses.color_pair(3))

            # Separator
            sep_y = start_y + len(fields) + 1
            stdscr.addstr(sep_y, label_col, "-" * 54, curses.A_DIM)

            # Start Simulation Button
            button_y = sep_y + 2
            btn_text = "[ START SIMULATION ]"
            btn_x = max(0, (w - len(btn_text)) // 2)
            if self.current_field == 8:
                stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
                stdscr.addstr(button_y, btn_x, f" >>> {btn_text} <<< ")
                stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
            else:
                stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
                stdscr.addstr(button_y, btn_x, btn_text)
                stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)

            exit_hint = "Press 'Q' or ESC to Cancel"
            stdscr.addstr(button_y + 2, max(0, (w - len(exit_hint)) // 2), exit_hint, curses.A_DIM)

            stdscr.refresh()

            # Handle Keys
            key = stdscr.getch()
            if key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                return None

            if key in (curses.KEY_UP, ord('k')):
                self.current_field = (self.current_field - 1) % self.total_fields
            elif key in (curses.KEY_DOWN, ord('j')):
                self.current_field = (self.current_field + 1) % self.total_fields
            elif key in (curses.KEY_LEFT, ord('h')):
                self._cycle_current_field(delta=-1)
            elif key in (curses.KEY_RIGHT, ord('l'), ord(' ')):
                self._cycle_current_field(delta=1)
            elif key in (curses.KEY_ENTER, 10, 13):
                if self.current_field == 8:
                    return self.get_config_dict()
                elif self.current_field == 5 and SEED_OPTIONS[self.seed_idx] == "Custom...":
                    self._prompt_custom_seed(stdscr, w, button_y + 1)
                else:
                    self._cycle_current_field(delta=1)

    def _cycle_current_field(self, delta: int):
        if self.current_field == 0:
            self.scenario_idx = (self.scenario_idx + delta) % len(SCENARIO_OPTIONS)
        elif self.current_field == 1:
            self.weather_idx = (self.weather_idx + delta) % len(WEATHER_OPTIONS)
        elif self.current_field == 2:
            self.altitude_idx = (self.altitude_idx + delta) % len(ALTITUDE_OPTIONS)
        elif self.current_field == 3:
            self.speed_idx = (self.speed_idx + delta) % len(SPEED_OPTIONS)
        elif self.current_field == 4:
            self.water_idx = (self.water_idx + delta) % len(WATER_OPTIONS)
        elif self.current_field == 5:
            self.seed_idx = (self.seed_idx + delta) % len(SEED_OPTIONS)
        elif self.current_field == 6:
            self.duration_idx = (self.duration_idx + delta) % len(DURATION_OPTIONS)
        elif self.current_field == 7:
            self.rendering_idx = (self.rendering_idx + delta) % len(RENDERING_OPTIONS)

    def _prompt_custom_seed(self, stdscr, w: int, y: int):
        curses.echo()
        curses.curs_set(1)
        prompt = "Enter custom integer seed: "
        x = max(2, (w - 35) // 2)
        stdscr.addstr(y, x, prompt)
        stdscr.clrtoeol()
        stdscr.refresh()
        val_bytes = stdscr.getstr(y, x + len(prompt), 10)
        curses.noecho()
        curses.curs_set(0)
        try:
            self.custom_seed = int(val_bytes.decode("utf-8").strip())
        except ValueError:
            pass


def parse_cli_args() -> argparse.Namespace:
    """Parses optional CLI arguments for headless/batch execution bypassing the TUI."""
    parser = argparse.ArgumentParser(description="RoadSentinel Interactive & Automated Simulation Launcher")
    parser.add_argument("--non-interactive", action="store_true", help="Bypass curses TUI and run directly from CLI args")
    parser.add_argument("--scenario", type=str, default=None, choices=SCENARIO_OPTIONS, help="Road degradation scenario")
    parser.add_argument("--weather", type=str, default=None, help="CARLA weather preset")
    parser.add_argument("--altitude", type=float, default=None, help="Drone altitude in meters")
    parser.add_argument("--speed", type=float, default=None, help="Drone speed in km/h")
    parser.add_argument("--water", type=str, default=None, help="Water condition (or ratio 0.0-1.0)")
    parser.add_argument("--seed", type=str, default=None, help="Procedural seed integer or 'random'")
    parser.add_argument("--duration", type=float, default=None, help="Flight duration in seconds")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without window")
    parser.add_argument("--standalone", action="store_true", help="Force standalone simulation engine without CARLA")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Destination output directory")
    parser.add_argument("--no-docker", action="store_true", help="Do not attempt to auto-launch CARLA docker container")
    return parser.parse_args()


def print_session_summary(cfg: Dict[str, Any], output_dir: Path):
    """Prints a clean, formatted session summary table upon simulation completion."""
    img_dir = output_dir / "images"
    num_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    gt_file = output_dir / "ground_truth.json"
    meta_csv = output_dir / "metadata.csv"

    print("\n" + "=" * 64)
    print("           ROADSENTINEL SIMULATION SESSION COMPLETED           ")
    print("=" * 64)
    print(f"  Scenario:          {cfg['scenario'].upper()}")
    print(f"  Weather:           {cfg['weather'].upper()}")
    print(f"  Altitude:          {cfg['altitude']:.1f} m")
    print(f"  Speed:             {cfg['speed']:.1f} km/h")
    print(f"  Water Condition:   {cfg['water']}")
    print(f"  Procedural Seed:   {cfg['seed']}")
    print(f"  Frames Captured:   {num_images}")
    print(f"  Images Directory:  {img_dir}")
    print(f"  Metadata CSV:      {meta_csv} (exists: {meta_csv.exists()})")
    print(f"  Ground Truth JSON: {gt_file} (exists: {gt_file.exists()})")
    print("=" * 64)
    print("\nNext steps to run RoadSentinel ML pipeline & evaluation:")
    print("  1. Run ML Pipeline:")
    print("     ./.venv/bin/python orchestrator.py --input-dir ./env/output --no-server")
    print("  2. Evaluate vs Ground Truth:")
    print("     ./.venv/bin/python env/evaluate_simulation.py --ground-truth ./env/output/ground_truth.json --predictions road_health_pipeline/output/analytics_demo/result.json\n")


def main():
    cli_args = parse_cli_args()
    output_dir = Path(cli_args.output_dir).resolve()

    # Determine whether to run interactive TUI or direct CLI mode
    use_tui = not cli_args.non_interactive and sys.stdin.isatty() and os.environ.get("TERM")

    selected_cfg: Dict[str, Any] = {}

    if use_tui and not any([cli_args.scenario, cli_args.weather, cli_args.altitude, cli_args.speed]):
        menu = TUIConfigMenu()
        res = curses.wrapper(menu.run)
        if res is None:
            print("[RoadSentinel] Simulation launch aborted by user.")
            sys.exit(0)
        selected_cfg = res
    else:
        # CLI fallback
        selected_cfg = {
            "scenario": cli_args.scenario or "poor",
            "weather": cli_args.weather or "post_rain",
            "altitude": cli_args.altitude if cli_args.altitude is not None else 100.0,
            "speed": cli_args.speed if cli_args.speed is not None else 30.0,
            "water": cli_args.water or "Automatic by Scenario",
            "seed": cli_args.seed if cli_args.seed is not None else "42",
            "duration": cli_args.duration if cli_args.duration is not None else 60.0,
            "rendering": "Headless" if cli_args.headless else "GUI",
        }

    # Validate and normalize configuration
    cfg = validate_configuration(selected_cfg)

    # Print chosen setup
    print("\n" + "-" * 56)
    print("  Selected RoadSentinel Configuration:")
    print(f"    Scenario:   {cfg['scenario'].upper()}")
    print(f"    Weather:    {cfg['weather'].upper()}")
    print(f"    Altitude:   {cfg['altitude']} m")
    print(f"    Speed:      {cfg['speed']} km/h")
    print(f"    Water:      {cfg['water']}")
    print(f"    Seed:       {cfg['seed']}")
    print(f"    Duration:   {cfg['duration']} s")
    print(f"    Rendering:  {cfg['rendering']}")
    print("-" * 56 + "\n")

    # Persist session metadata BEFORE simulation launches
    save_session_metadata(cfg, output_dir)

    # Manage CARLA Docker container if not in standalone mode
    is_standalone = cli_args.standalone
    if not is_standalone:
        is_running = check_carla_running(host="127.0.0.1", port=2000)
        if not is_running and not cli_args.no_docker:
            print("[RoadSentinel] CARLA server not detected. Attempting Docker startup...")
            is_running = start_carla_docker(headless=cfg["headless"])
            if not is_running:
                print("[RoadSentinel] CARLA could not be reached. Falling back to Standalone Simulation Engine.")
                is_standalone = True
        elif is_running:
            print("[RoadSentinel] Active CARLA server detected on 127.0.0.1:2000. Ready!")

    # Build execution command
    cmd = build_drone_cmd(cfg, output_dir, standalone=is_standalone)
    print(f"[RoadSentinel] Executing drone flight simulation:\n  {' '.join(cmd)}\n")

    # Execute simulation process
    try:
        proc = subprocess.run(cmd, cwd=str(WORKSPACE_DIR))
        if proc.returncode != 0:
            print(f"[RoadSentinel] Drone simulation exited with return code {proc.returncode}")
    except KeyboardInterrupt:
        print("\n[RoadSentinel] Simulation interrupted by user.")

    # Display final session summary
    print_session_summary(cfg, output_dir)


if __name__ == "__main__":
    main()
