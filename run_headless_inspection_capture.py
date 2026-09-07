#!/usr/bin/env python3
"""
run_headless_inspection_capture.py
----------------------------------
Host-side orchestrator for the automated headless RoadSentinel inspection
capture.  Generates a capture plan (20 days × 6 segments × random weather),
launches Unreal Engine headlessly, and validates all outputs.

Usage:
    python3 run_headless_inspection_capture.py [--gui] [--seed 42]

    --gui    Show the Unreal Editor window while rendering (default: offscreen)
    --seed   Random seed for weather assignment (default: 2026)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT
OUTPUT_ROOT = ROOT / "env" / "output"
MANUAL_INSPECTIONS_ROOT = OUTPUT_ROOT / "manual_inspections"
MANIFESTS_ROOT = OUTPUT_ROOT / "temporal_20_day" / "manifests"

UNREAL_PROJECT = ROOT / "RoadSentinelSim" / "RoadSentinelSim.uproject"
UNREAL_EDITOR = Path("/mnt/bigdata/unreal_engine/Engine/Binaries/Linux/UnrealEditor")
UNREAL_LAUNCHER = ROOT / "launch_unreal_editor.sh"
CAPTURE_WORKER = ROOT / "RoadSentinelSim" / "Content" / "Python" / "rs_headless_capture_worker.py"

SEGMENT_IDS = [f"SEG_{n:03d}" for n in range(1, 7)]

WEATHER_PRESETS = [
    "Clear Noon (70° Sun)",
    "Golden Hour Sunset",
    "Overcast Day",
    "Heavy Rain & Wet Road",
    "Dense Atmospheric Fog",
    "Night Highway with Lamps",
]

PLAN_SCHEMA = "RoadSentinelInspectionCapturePlan/v1"


def build_capture_plan(seed: int = 2026) -> Dict[str, Any]:
    """Generate the full 20-day × 6-segment capture plan with random weather.

    Weather is assigned randomly per (day, segment) pair using a deterministic
    seed so that re-running with the same seed produces identical assignments.

    Returns the plan dict (also written to disk by the caller).
    """
    rng = random.Random(seed)

    captures: List[Dict[str, Any]] = []
    weather_summary: Dict[str, int] = {w: 0 for w in WEATHER_PRESETS}

    for day in range(1, 21):
        for seg_id in SEGMENT_IDS:
            weather = rng.choice(WEATHER_PRESETS)
            weather_summary[weather] += 1
            captures.append({
                "day": day,
                "segment_id": seg_id,
                "weather": weather,
            })

    completion_marker = str(MANUAL_INSPECTIONS_ROOT / "capture_complete.json")

    plan = {
        "schema_version": PLAN_SCHEMA,
        "seed": seed,
        "total_captures": len(captures),
        "days": 20,
        "segments": SEGMENT_IDS,
        "weather_presets": WEATHER_PRESETS,
        "weather_distribution": weather_summary,
        "completion_marker": completion_marker,
        "captures": captures,
    }
    return plan


def verify_prerequisites() -> None:
    """Check that all required files exist before launching Unreal."""
    errors = []

    if not UNREAL_PROJECT.is_file():
        errors.append(f"Unreal project not found: {UNREAL_PROJECT}")

    if not UNREAL_EDITOR.is_file() and not UNREAL_LAUNCHER.is_file():
        errors.append(
            f"Unreal Editor not found at {UNREAL_EDITOR} and launcher not at {UNREAL_LAUNCHER}"
        )

    if not CAPTURE_WORKER.is_file():
        errors.append(f"Capture worker script not found: {CAPTURE_WORKER}")

    for day in range(1, 21):
        manifest = MANIFESTS_ROOT / f"day_{day:02d}.json"
        if not manifest.is_file():
            errors.append(f"Missing manifest: {manifest}")

    if errors:
        print("=" * 70)
        print("PREREQUISITE CHECK FAILED")
        print("=" * 70)
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


def print_plan_summary(plan: Dict[str, Any]) -> None:
    """Print a human-readable summary of the capture plan."""
    print()
    print("=" * 70)
    print("  RoadSentinel — Automated Headless Inspection Capture")
    print("=" * 70)
    print()
    print(f"  Days:            1 – 20")
    print(f"  Segments:        {', '.join(SEGMENT_IDS)}")
    print(f"  Total Captures:  {plan['total_captures']}")
    print(f"  Random Seed:     {plan['seed']}")
    print()
    print("  Weather Distribution:")
    for weather, count in plan["weather_distribution"].items():
        bar = "█" * count
        print(f"    {weather:32s} {count:3d}  {bar}")
    print()
    print(f"  Output:  {MANUAL_INSPECTIONS_ROOT}/")
    print(f"           day_XX/SEG_XXX/raw.png + metadata.json")
    print()

    # Show weather assignment table for first 3 days as preview
    print("  Weather Preview (first 3 days):")
    print(f"  {'Day':>5}  {'Segment':>8}  Weather")
    print(f"  {'---':>5}  {'-------':>8}  -------")
    for entry in plan["captures"][:18]:
        day = entry["day"]
        if day > 3:
            break
        print(f"  {day:5d}  {entry['segment_id']:>8}  {entry['weather']}")
    print(f"  ... ({plan['total_captures'] - 18} more captures)")
    print()


def run_unreal_headless(plan_path: Path, gui: bool = False) -> None:
    """Launch Unreal Engine to execute the capture worker script."""
    # Determine the launcher
    if UNREAL_LAUNCHER.is_file():
        launcher = str(UNREAL_LAUNCHER)
    elif UNREAL_EDITOR.is_file():
        launcher = str(UNREAL_EDITOR)
    else:
        raise FileNotFoundError("No Unreal Engine launcher found")

    cmd = [
        launcher,
        f"-ExecutePythonScript={CAPTURE_WORKER}",
        "-NoSound",
        "-NoSplash",
        "-Unattended",
    ]
    if not gui:
        cmd.append("-RenderOffscreen")

    env = os.environ.copy()
    env["ROADSENTINEL_INSPECTION_CAPTURE_PLAN"] = str(plan_path)

    mode = "GUI" if gui else "offscreen (headless)"
    print(f"  [*] Launching Unreal Engine in {mode} mode...")
    print(f"  [*] Command: {' '.join(cmd[:3])} ...")
    print(f"  [*] This will take approximately 15–40 minutes for 120 captures.")
    print()

    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n  [✗] Unreal Engine exited with code {result.returncode} after {elapsed:.0f}s")
    else:
        print(f"\n  [✓] Unreal Engine finished in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


def validate_outputs(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Check the completion marker and validate all expected outputs."""
    marker_path = Path(plan["completion_marker"])
    if not marker_path.is_file():
        print("  [✗] Completion marker not found — Unreal may have crashed.")
        return {"status": "incomplete", "missing": plan["total_captures"]}

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") == "error":
        print(f"  [✗] Capture failed with error: {marker.get('error')}")
        return marker

    # Verify every expected file exists
    missing_files = []
    valid_captures = 0
    total_size_bytes = 0

    for entry in plan["captures"]:
        day = entry["day"]
        seg = entry["segment_id"]
        raw_path = MANUAL_INSPECTIONS_ROOT / f"day_{day:02d}" / seg / "raw.png"
        meta_path = MANUAL_INSPECTIONS_ROOT / f"day_{day:02d}" / seg / "metadata.json"

        if raw_path.is_file() and meta_path.is_file():
            valid_captures += 1
            total_size_bytes += raw_path.stat().st_size
        else:
            missing_files.append(f"day_{day:02d}/{seg}")

    total_mb = total_size_bytes / (1024 * 1024)
    print()
    print("  ═══════════════════════════════════════════════════════")
    print("  CAPTURE VALIDATION RESULTS")
    print("  ═══════════════════════════════════════════════════════")
    print(f"  Total expected:    {plan['total_captures']}")
    print(f"  Successfully saved: {valid_captures}")
    print(f"  Missing:           {len(missing_files)}")
    print(f"  Total disk usage:  {total_mb:.1f} MB")
    print(f"  Avg per capture:   {total_mb / max(valid_captures, 1) * 1024:.0f} KB")

    if marker.get("results"):
        warnings = sum(1 for r in marker["results"] if r.get("status") == "warning")
        errors = sum(1 for r in marker["results"] if r.get("status") == "error")
        if warnings:
            print(f"  Validation warnings: {warnings} (may be blank or low-quality)")
        if errors:
            print(f"  Capture errors:      {errors}")

    if missing_files:
        print(f"\n  Missing captures:")
        for m in missing_files[:10]:
            print(f"    ✗ {m}")
        if len(missing_files) > 10:
            print(f"    ... and {len(missing_files) - 10} more")

    print()
    return {
        "status": "complete" if not missing_files else "partial",
        "valid_captures": valid_captures,
        "missing": len(missing_files),
        "total_size_mb": round(total_mb, 1),
    }


def write_dataset_summary(plan: Dict[str, Any]) -> Path:
    """Write a CSV summary of all captures for downstream ML pipelines."""
    import csv

    summary_path = MANUAL_INSPECTIONS_ROOT / "dataset_summary.csv"
    rows = []

    for entry in plan["captures"]:
        day = entry["day"]
        seg = entry["segment_id"]
        raw_path = MANUAL_INSPECTIONS_ROOT / f"day_{day:02d}" / seg / "raw.png"
        meta_path = MANUAL_INSPECTIONS_ROOT / f"day_{day:02d}" / seg / "metadata.json"

        row = {
            "day": day,
            "segment_id": seg,
            "weather": entry["weather"],
            "image_path": str(raw_path),
            "exists": raw_path.is_file(),
            "file_size_bytes": raw_path.stat().st_size if raw_path.is_file() else 0,
        }

        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                row["condition_score"] = meta.get("condition_score", "")
                row["along_m"] = meta.get("world_coordinates", {}).get("along_m", "")
                row["across_m"] = meta.get("world_coordinates", {}).get("across_m", "")
            except Exception:
                pass

        rows.append(row)

    fieldnames = [
        "day", "segment_id", "weather", "image_path", "exists",
        "file_size_bytes", "condition_score", "along_m", "across_m",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [✓] Dataset summary: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Automated headless RoadSentinel inspection capture: "
                    "20 days × 6 segments × random weather"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Show the Unreal Editor window while rendering (default: headless)",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed for weather assignment (default: 2026)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate the capture plan and print summary, but don't launch Unreal",
    )
    args = parser.parse_args()

    # 1. Verify prerequisites
    print("\n  [*] Checking prerequisites...")
    verify_prerequisites()
    print("  [✓] All prerequisites OK\n")

    # 2. Generate capture plan
    plan = build_capture_plan(seed=args.seed)
    plan_path = MANUAL_INSPECTIONS_ROOT / "capture_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"  [✓] Capture plan written: {plan_path}")

    print_plan_summary(plan)

    if args.dry_run:
        print("  [*] Dry run — not launching Unreal Engine.")
        print(f"  [*] Plan saved to: {plan_path}")
        return

    # 3. Clean previous completion marker
    marker_path = Path(plan["completion_marker"])
    if marker_path.is_file():
        marker_path.unlink()

    # 4. Launch Unreal Engine
    run_unreal_headless(plan_path, gui=args.gui)

    # 5. Validate outputs
    validation = validate_outputs(plan)

    # 6. Write dataset summary CSV
    write_dataset_summary(plan)

    # 7. Final status
    print()
    if validation.get("status") == "complete":
        print("  ✅ ALL 120 CAPTURES COMPLETE — Dataset ready for ML pipeline!")
    elif validation.get("valid_captures", 0) > 0:
        print(f"  ⚠  Partial: {validation['valid_captures']}/120 captures succeeded.")
    else:
        print("  ❌ No captures were produced. Check Unreal Engine logs.")
    print()


if __name__ == "__main__":
    main()
