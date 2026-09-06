#!/usr/bin/env python3
"""generate_road_scenarios.py
----------------------------
Batch Dataset Generation Utility for the RoadSentinel Simulation Testbed.

Generates structured, reproducible test datasets across combinations of:
- Road-health scenarios: healthy, moderate, poor, critical
- Weather & lighting presets: clear, overcast, wet, rain, low_light, sunset, etc.
- Random seeds: deterministic procedural defects across runs
- Configurable durations, survey altitudes, and speeds

For each simulation run, creates an isolated folder containing:
  - images/: high-resolution aerial RGB frames with ~70% overlap
  - metadata.csv: per-capture drone telemetry (lat, lon, alt, GSD, speed, timestamp)
  - geo.txt: OpenDroneMap/GIS georeference records
  - ground_truth.json: detailed ground truth specifications for every generated defect
  - run_config.json: exact parameters and environment metadata for reproducibility

Also compiles a master `dataset_manifest.json` indexing all runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure parent directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config

DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "output" / "datasets"


def parse_args():
    parser = argparse.ArgumentParser(
        description="RoadSentinel Batch Road Degradation & Weather Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["healthy", "moderate", "poor", "critical"],
        choices=["healthy", "moderate", "poor", "critical", "all"],
        help="Road degradation scenario levels to generate",
    )
    parser.add_argument(
        "--weathers",
        nargs="+",
        default=["clear", "wet", "overcast"],
        help="Weather presets (e.g., clear, overcast, wet, rain, low_light, sunset) or 'all'",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 101],
        help="List of integer random seeds for reproducible procedural generation",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Flight duration in seconds per simulation run (e.g. 8s produces ~10 overlapping frames)",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=config.ALTITUDE_M,
        help="Drone survey altitude in meters (default: 30m)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=config.SPEED_KMPH,
        help="Drone flight speed in km/h (default: 30 km/h)",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Root directory for generated datasets",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Force standalone procedural simulation mode (no CARLA server needed)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum total simulation runs to execute (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--num-defects",
        type=int,
        default=None,
        help="Override defect count per corridor",
    )
    parser.add_argument(
        "--water-ratio",
        type=float,
        default=None,
        help="Override fraction of water-filled potholes [0.0 - 1.0]",
    )
    return parser.parse_args()


def run_single_simulation(
    scenario: str,
    weather: str,
    seed: int,
    duration: float,
    altitude: float,
    speed: float,
    run_dir: Path,
    standalone: bool = False,
    num_defects: Optional[int] = None,
    water_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute a single drone simulation flight run as an isolated subprocess."""
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "drone_sim.py"),
        "--scenario", scenario,
        "--weather", weather,
        "--seed", str(seed),
        "--duration", str(duration),
        "--altitude", str(altitude),
        "--speed", str(speed),
        "--output-dir", str(run_dir),
        "--headless",
        "--auto-fly",
    ]
    if standalone:
        cmd.append("--standalone")
    if num_defects is not None:
        cmd.extend(["--num-defects", str(num_defects)])
    if water_ratio is not None:
        cmd.extend(["--water-ratio", str(water_ratio)])

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPT_DIR) + ":" + env.get("PYTHONPATH", "")

    print(f"\n[DatasetGen] Launching run: {scenario.upper()} | {weather.upper()} | seed={seed}")
    print(f"             Command: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env, capture_output=True, text=True)
    wall_time = time.time() - t0

    if proc.returncode != 0:
        print(f"[DatasetGen] ERROR in run {run_dir.name}:\n{proc.stderr}")
        success = False
    else:
        success = True

    # Read generated ground truth if present
    gt_file = run_dir / "ground_truth.json"
    num_defects_gen = 0
    num_water_gen = 0
    if gt_file.is_file():
        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                num_defects_gen = len(gt_data.get("defects", []))
                num_water_gen = sum(
                    1 for d in gt_data.get("defects", [])
                    if d.get("water_state", {}).get("is_water_filled", False)
                )
        except Exception:
            pass

    # Count captured images
    captured_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    run_record = {
        "run_id": run_dir.name,
        "scenario": scenario,
        "weather": weather,
        "seed": seed,
        "duration_s": duration,
        "altitude_m": altitude,
        "speed_kmph": speed,
        "output_dir": str(run_dir),
        "success": success,
        "wall_time_s": round(wall_time, 2),
        "images_captured": len(captured_images),
        "total_defects": num_defects_gen,
        "water_filled_defects": num_water_gen,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save run configuration
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    status_str = "SUCCESS" if success else "FAILED"
    print(f"[DatasetGen] [{status_str}] Captured {len(captured_images)} photos, {num_defects_gen} defects ({num_water_gen} water) in {wall_time:.1f}s.")
    return run_record


def main():
    args = parse_args()
    base_out = Path(args.output_base_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    scenarios = (
        ["healthy", "moderate", "poor", "critical"]
        if "all" in args.scenarios
        else args.scenarios
    )
    weathers = (
        list(config.WEATHER_PRESETS.keys())
        if "all" in args.weathers
        else args.weathers
    )
    seeds = args.seeds

    runs_to_execute = []
    for sc in scenarios:
        for w in weathers:
            for s in seeds:
                runs_to_execute.append((sc, w, s))

    if args.max_runs and len(runs_to_execute) > args.max_runs:
        print(f"[DatasetGen] Limiting execution to first {args.max_runs} of {len(runs_to_execute)} planned runs.")
        runs_to_execute = runs_to_execute[:args.max_runs]

    print("=" * 72)
    print("      ROADSENTINEL PROCEDURAL DATASET GENERATION MATRIX")
    print("=" * 72)
    print(f"  Target Base Directory : {base_out}")
    print(f"  Total Runs Planned    : {len(runs_to_execute)}")
    print(f"  Scenarios             : {', '.join(scenarios)}")
    print(f"  Weather Conditions    : {', '.join(weathers)}")
    print(f"  Random Seeds          : {seeds}")
    print(f"  Duration per Run      : {args.duration} s")
    print(f"  Altitude / Speed      : {args.altitude} m / {args.speed} km/h")
    print(f"  Standalone Mode       : {args.standalone}")
    print("=" * 72)

    manifest: List[Dict[str, Any]] = []
    total_t0 = time.time()

    for idx, (scenario, weather, seed) in enumerate(runs_to_execute, start=1):
        run_name = f"{scenario}_{weather}_seed{seed}"
        run_dir = base_out / run_name
        print(f"\n--- Progress: [{idx}/{len(runs_to_execute)}] {run_name} ---")

        run_info = run_single_simulation(
            scenario=scenario,
            weather=weather,
            seed=seed,
            duration=args.duration,
            altitude=args.altitude,
            speed=args.speed,
            run_dir=run_dir,
            standalone=args.standalone,
            num_defects=args.num_defects,
            water_ratio=args.water_ratio,
        )
        manifest.append(run_info)

    # Compile master manifest
    manifest_path = base_out / "dataset_manifest.json"
    manifest_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runs": len(manifest),
        "successful_runs": sum(1 for r in manifest if r["success"]),
        "total_images_generated": sum(r["images_captured"] for r in manifest),
        "total_defects_generated": sum(r["total_defects"] for r in manifest),
        "total_wall_time_s": round(time.time() - total_t0, 2),
        "runs": manifest,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    print("\n" + "=" * 72)
    print("      DATASET GENERATION BATCH COMPLETE")
    print("=" * 72)
    print(f"  Total Runs Completed  : {manifest_payload['successful_runs']} / {manifest_payload['total_runs']}")
    print(f"  Total Images Captured : {manifest_payload['total_images_generated']}")
    print(f"  Total Defects Spawned : {manifest_payload['total_defects_generated']}")
    print(f"  Total Elapsed Time    : {manifest_payload['total_wall_time_s']:.1f} s")
    print(f"  Master Manifest Saved : {manifest_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
