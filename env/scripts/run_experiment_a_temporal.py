#!/usr/bin/env python3
"""run_experiment_a_temporal.py
--------------------------------
RoadSentinel Phase 3 — Experiment A Temporal Response Analysis.

Applies the existing conservative defect-matching and temporal tracking
pipeline (from xgboost/temporal/) over the 8 verified same-camera candidate
subsequences from Experiment A (Phase 2 DINOv2 + SAM2 frozen perception outputs).

ABSOLUTE RESTRICTIONS:
- NEVER touches Unreal Engine / CARLA / simulation.
- FILE ACCESS ONLY over frozen Phase-2 outputs in integration/experiment_a/perception/.
- NO model tuning, NO retraining, NO threshold changes.
- NO YOLO, NO XGBoost forecasting (Model V2 is strictly avoided).
- NO cross-camera matching across transitions.
- Evaluates 8 same-camera sequences; SEG_003 Day 10 excluded (missing sidecar).
- Results framed as MODEL-OBSERVED CHANGE ACROSS SIMULATED INSPECTION STATES.

Outputs:
- integration/experiment_a/temporal/<sequence_id>/
    - daily_summary.csv
    - daily_summary.json
    - defect_tracks.json
    - temporal_events.json
    - progression_summary.json
- integration/experiment_a/temporal_summary.csv
- integration/experiment_a/temporal/plots/*.png
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_ROOT = ROOT / "xgboost/temporal"
SAM2_DINO_ROOT = ROOT / "sam2_dino"

for p in (str(ROOT), str(TEMPORAL_ROOT), str(SAM2_DINO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from defect_matching import MatchingConfig
from feature_contract import load_feature_record
from load_sequence import DailyRecord, missing_days
from progression import build_daily_summary, build_tracks, progression_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("phase3_temporal")

CANDIDATE_SEQUENCES = [
    {
        "sequence_id": "SEG_001_D03_D10",
        "segment_id": "SEG_001",
        "start_day": 3,
        "end_day": 10,
        "camera_preset": "🌄 Highway Curve Vantage Overlook",
        "temporal_label": "SIMULATED TEMPORAL PROGRESSION (MULTI-STATE)",
        "notes": "Longest same-camera multi-condition sequence (8 states); captures Day 06-08 deterioration and Day 09-10 sunset shadow suppression.",
    },
    {
        "sequence_id": "SEG_002_D01_D02",
        "segment_id": "SEG_002",
        "start_day": 1,
        "end_day": 2,
        "camera_preset": "🔍 Low-Angle Pothole Inspection (30° Close-Up)",
        "temporal_label": "PAIRWISE SIMULATED CHANGE",
        "notes": "Pristine Grade A pair; grazing viewpoint magnifies coarse aggregate; 2 defects persist across days.",
    },
    {
        "sequence_id": "SEG_002_D04_D05",
        "segment_id": "SEG_002",
        "start_day": 4,
        "end_day": 5,
        "camera_preset": "💧 Waterlogged Pothole Macro View",
        "temporal_label": "PAIRWISE SIMULATED CHANGE",
        "notes": "Moderate Grade C pair; lighting shifts from Clear Noon to Overcast, increasing severity from 0.3687 to 0.7078.",
    },
    {
        "sequence_id": "SEG_002_D06_D07",
        "segment_id": "SEG_002",
        "start_day": 6,
        "end_day": 7,
        "camera_preset": "🌄 Highway Curve Vantage Overlook",
        "temporal_label": "PAIRWISE SIMULATED CHANGE",
        "notes": "Overlook pair transitioning from Grade C (Clear Noon) to Grade D (Clear Noon).",
    },
    {
        "sequence_id": "SEG_003_D01_D07",
        "segment_id": "SEG_003",
        "start_day": 1,
        "end_day": 7,
        "camera_preset": "🌄 Highway Curve Vantage Overlook",
        "temporal_label": "MODEL-RESPONSE STABILITY ON PRISTINE SIMULATION STATES",
        "notes": "Pristine Grade A stability benchmark (7 states, constant camera/lighting); exact same boundary candidate tracked across all 7 days.",
    },
    {
        "sequence_id": "SEG_003_D08_D09",
        "segment_id": "SEG_003",
        "start_day": 8,
        "end_day": 9,
        "camera_preset": "🔭 Overhead Drone Survey (SAM 2 Top-Down)",
        "temporal_label": "PAIRWISE SIMULATED CHANGE",
        "notes": "Drone survey pair transitioning from Grade C to Grade D; 1 track persists with 0.9412 mask IoU.",
    },
    {
        "sequence_id": "SEG_004_D01_D05",
        "segment_id": "SEG_004",
        "start_day": 1,
        "end_day": 5,
        "camera_preset": "🔭 Overhead Drone Survey (SAM 2 Top-Down)",
        "temporal_label": "SIMULATED TEMPORAL PROGRESSION (DRONE SUBSET)",
        "notes": "5-day top-down drone progression; severity scales monotonically from 0.0000 (Grade C/B) to 0.7302 (Grade D Overcast).",
    },
    {
        "sequence_id": "SEG_004_D06_D10",
        "segment_id": "SEG_004",
        "start_day": 6,
        "end_day": 10,
        "camera_preset": "🌄 Highway Curve Vantage Overlook",
        "temporal_label": "SIMULATED TEMPORAL PROGRESSION (OVERLOOK SUBSET)",
        "notes": "5-day overlook progression; high severity on Day 06-08 (up to 0.8134 in heavy rain), followed by road-marking suppression drop on Day 09-10.",
    },
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_subsequence_records(
    segment_id: str, start_day: int, end_day: int
) -> List[DailyRecord]:
    """Load DailyRecords from Phase-2 outputs, resolving relative mask paths."""
    records: List[DailyRecord] = []
    perception_dir = ROOT / "integration/experiment_a/perception" / segment_id

    for day in range(start_day, end_day + 1):
        feat_path = perception_dir / f"day_{day:02d}" / "features.json"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing Phase-2 feature record: {feat_path}")

        raw_record = load_feature_record(feat_path)
        record = dict(raw_record)
        record["defects"] = [dict(d) for d in record.get("defects", [])]

        # Resolve relative mask references
        for defect in record["defects"]:
            mask = defect.get("mask_path")
            if mask and not Path(mask).is_absolute():
                defect["mask_path"] = str((feat_path.parent / mask).resolve())

        records.append(DailyRecord(record=record, source_path=feat_path))

    return records


def analyze_sequence(
    seq_cfg: Dict[str, Any], output_root: Path
) -> Dict[str, Any]:
    """Execute conservative defect matching and progression analytics for one subsequence."""
    seq_id = seq_cfg["sequence_id"]
    seg_id = seq_cfg["segment_id"]
    start_day = seq_cfg["start_day"]
    end_day = seq_cfg["end_day"]
    cam_preset = seq_cfg["camera_preset"]
    label = seq_cfg["temporal_label"]
    notes = seq_cfg["notes"]

    log.info("Analyzing %s (%s, Days %02d-%02d)", seq_id, seg_id, start_day, end_day)
    records = load_subsequence_records(seg_id, start_day, end_day)

    # Core temporal analytics
    daily = build_daily_summary(records)
    tracks, events = build_tracks(records, config=MatchingConfig())
    summary = progression_summary(records, tracks, events)

    # Compute descriptive metrics
    severities = [float(r.record["current_severity"] or 0.0) for r in records]
    defect_counts = [int(r.record["defect_count"]) for r in records]
    areas = [float(r.record["defect_area_ratio"] or 0.0) for r in records]

    track_lengths = [len(t["days_seen"]) for t in tracks.values()]
    mean_track_len = float(np.mean(track_lengths)) if track_lengths else 0.0
    max_track_len = max(track_lengths, default=0)

    event_counter = Counter(e["event"] for e in events)
    new_defects = event_counter.get("NEW_DEFECT", 0)
    matched_events = (
        event_counter.get("MATCHED_EXISTING", 0)
        + event_counter.get("OBSERVED_AREA_INCREASED", 0)
        + event_counter.get("OBSERVED_AREA_DECREASED", 0)
    )
    not_observed = event_counter.get("NOT_OBSERVED", 0)

    # Save per-sequence outputs
    target_dir = output_root / seq_id
    target_dir.mkdir(parents=True, exist_ok=True)

    fields = list(daily[0].keys())
    with (target_dir / "daily_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(daily)

    _write_json(target_dir / "daily_summary.json", daily)
    _write_json(
        target_dir / "defect_tracks.json",
        {
            "sequence_id": seq_id,
            "segment_id": seg_id,
            "camera_preset": cam_preset,
            "tracks": list(tracks.values()),
        },
    )
    _write_json(
        target_dir / "temporal_events.json",
        {
            "sequence_id": seq_id,
            "segment_id": seg_id,
            "camera_preset": cam_preset,
            "events": events,
        },
    )
    _write_json(
        target_dir / "progression_summary.json",
        {
            "sequence_id": seq_id,
            **summary,
            "camera_preset": cam_preset,
            "temporal_label": label,
            "unique_tracks": len(tracks),
            "mean_track_length": round(mean_track_len, 2),
            "max_track_length": max_track_len,
            "new_defect_events": new_defects,
            "matched_events": matched_events,
            "not_observed_events": not_observed,
            "mean_severity": round(float(np.mean(severities)), 4),
            "mean_defect_count": round(float(np.mean(defect_counts)), 2),
            "mean_defect_area_ratio": round(float(np.mean(areas)), 6),
            "notes": notes,
        },
    )

    log.info(
        "  Finished %s: %d tracks, %d matched events, sev delta: %.4f",
        seq_id,
        len(tracks),
        matched_events,
        summary["severity_delta"] or 0.0,
    )

    sev_delta = summary["severity_delta"]
    sev_change_str = f"{sev_delta:+.4f}" if sev_delta is not None else "0.0000"

    return {
        "sequence_id": seq_id,
        "segment_id": seg_id,
        "start_day": start_day,
        "end_day": end_day,
        "camera_preset": cam_preset,
        "num_states": len(records),
        "initial_severity": round(severities[0], 4),
        "final_severity": round(severities[-1], 4),
        "severity_change": sev_change_str,
        "mean_severity": round(float(np.mean(severities)), 4),
        "mean_defect_count": round(float(np.mean(defect_counts)), 2),
        "unique_tracks": len(tracks),
        "mean_track_length": round(mean_track_len, 2),
        "max_track_length": max_track_len,
        "new_defect_events": new_defects,
        "matched_events": matched_events,
        "not_observed_events": not_observed,
        "mean_defect_area_ratio": round(float(np.mean(areas)), 6),
        "temporal_label": label,
        "notes": notes,
    }


def generate_research_plots(output_dir: Path) -> List[Path]:
    """Generate research-ready plots for key temporal sequences."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated_plots: List[Path] = []

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    # 1. SEG_001 Day 03-10: Severity vs Simulated Day
    seg001_path = output_dir / "SEG_001_D03_D10/daily_summary.json"
    if seg001_path.exists():
        data = json.loads(seg001_path.read_text(encoding="utf-8"))
        days = [d["day"] for d in data]
        sev = [d["current_severity"] or 0.0 for d in data]
        counts = [d["defect_count"] for d in data]
        areas = [d["defect_area_ratio"] or 0.0 for d in data]

        # Plot 1: Severity
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
        ax.plot(days, sev, marker="o", color="#d9534f", linewidth=2.5, markersize=8, label="Model-Observed Severity")
        ax.set_title("SEG_001 (Days 03–10): Model-Observed Severity Across Simulated States\n(Camera: Highway Curve Vantage Overlook)", fontsize=12, pad=12, weight="bold")
        ax.set_xlabel("Simulated Inspection Day", fontsize=11, labelpad=8)
        ax.set_ylabel("Current Severity [0, 1]", fontsize=11, labelpad=8)
        ax.set_ylim(-0.05, 1.0)
        ax.set_xticks(days)
        ax.grid(True, linestyle="--", alpha=0.6)
        # Annotate key days
        ax.annotate("Day 06: Overcast Peak\n(sev=0.7285)", xy=(6, 0.7285), xytext=(5.2, 0.85),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2), fontsize=9, weight="bold")
        ax.annotate("Day 10: Sunset Glare\n(Shadow Suppression -> 0.0)", xy=(10, 0.0), xytext=(7.8, 0.20),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2), fontsize=9, weight="bold")
        p1 = plots_dir / "seg001_d03_d10_severity.png"
        fig.tight_layout()
        fig.savefig(p1)
        plt.close(fig)
        generated_plots.append(p1)

        # Plot 2: Defect Count
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
        ax.bar(days, counts, color="#337ab7", alpha=0.85, width=0.55, edgecolor="#23527c", linewidth=1.2)
        ax.set_title("SEG_001 (Days 03–10): Detected Defect Count Across Simulated States\n(Camera: Highway Curve Vantage Overlook)", fontsize=12, pad=12, weight="bold")
        ax.set_xlabel("Simulated Inspection Day", fontsize=11, labelpad=8)
        ax.set_ylabel("Defect Count", fontsize=11, labelpad=8)
        ax.set_ylim(0, max(counts) + 2)
        ax.set_xticks(days)
        for d, c in zip(days, counts):
            ax.text(d, c + 0.25, str(c), ha="center", va="bottom", fontsize=10, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        p2 = plots_dir / "seg001_d03_d10_defect_count.png"
        fig.tight_layout()
        fig.savefig(p2)
        plt.close(fig)
        generated_plots.append(p2)

        # Plot 3: Defect Area Ratio
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
        ax.plot(days, [a * 100 for a in areas], marker="s", color="#f0ad4e", linewidth=2.5, markersize=8)
        ax.set_title("SEG_001 (Days 03–10): Defect Surface Area Ratio Across Simulated States\n(Camera: Highway Curve Vantage Overlook)", fontsize=12, pad=12, weight="bold")
        ax.set_xlabel("Simulated Inspection Day", fontsize=11, labelpad=8)
        ax.set_ylabel("Defect Area Ratio (% of Road)", fontsize=11, labelpad=8)
        ax.set_xticks(days)
        ax.grid(True, linestyle="--", alpha=0.6)
        p3 = plots_dir / "seg001_d03_d10_defect_area.png"
        fig.tight_layout()
        fig.savefig(p3)
        plt.close(fig)
        generated_plots.append(p3)

    # 4 & 5. SEG_003 Day 01-07: Pristine Grade A Stability
    seg003_path = output_dir / "SEG_003_D01_D07/daily_summary.json"
    if seg003_path.exists():
        data = json.loads(seg003_path.read_text(encoding="utf-8"))
        days = [d["day"] for d in data]
        sev = [d["current_severity"] or 0.0 for d in data]
        counts = [d["defect_count"] for d in data]

        # Plot 4: Severity Stability
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
        ax.plot(days, sev, marker="o", color="#5cb85c", linewidth=2.5, markersize=8, label="Model Severity")
        ax.axhline(float(np.mean(sev)), color="#4cae4c", linestyle="--", alpha=0.7, label=f"Mean Severity ({np.mean(sev):.4f})")
        ax.set_title("SEG_003 (Days 01–07): Pristine Grade A Response Stability\n(Constant Camera: Highway Curve Vantage Overlook | Clear Noon)", fontsize=12, pad=12, weight="bold")
        ax.set_xlabel("Simulated Inspection Day", fontsize=11, labelpad=8)
        ax.set_ylabel("Current Severity [0, 1]", fontsize=11, labelpad=8)
        ax.set_ylim(0.20, 0.30)
        ax.set_xticks(days)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)
        ax.text(0.04, 0.12, "Persistent Road-Curb False Alarm\nTrack Length: 7 days | Mask IoU: 0.85–0.96\nSeverity Spread: ±0.0028",
                transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f5e9", edgecolor="#81c784", alpha=0.9))
        p4 = plots_dir / "seg003_d01_d07_severity_stability.png"
        fig.tight_layout()
        fig.savefig(p4)
        plt.close(fig)
        generated_plots.append(p4)

        # Plot 5: Defect Count Stability
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
        ax.bar(days, counts, color="#5cb85c", alpha=0.85, width=0.5, edgecolor="#4cae4c", linewidth=1.2)
        ax.set_title("SEG_003 (Days 01–07): Defect Count Stability Across Pristine States\n(Invariant: Exactly 1 Detected Region Per Day Across 7 Days)", fontsize=12, pad=12, weight="bold")
        ax.set_xlabel("Simulated Inspection Day", fontsize=11, labelpad=8)
        ax.set_ylabel("Defect Count", fontsize=11, labelpad=8)
        ax.set_ylim(0, 3)
        ax.set_xticks(days)
        for d, c in zip(days, counts):
            ax.text(d, c + 0.1, f"{c} (DEFECT_001)", ha="center", va="bottom", fontsize=10, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        p5 = plots_dir / "seg003_d01_d07_defect_count_stability.png"
        fig.tight_layout()
        fig.savefig(p5)
        plt.close(fig)
        generated_plots.append(p5)

    # 6. SEG_004 Day 01-05: Overhead Drone Survey Response
    seg004_d01_path = output_dir / "SEG_004_D01_D05/daily_summary.json"
    if seg004_d01_path.exists():
        data = json.loads(seg004_d01_path.read_text(encoding="utf-8"))
        days = [d["day"] for d in data]
        sev = [d["current_severity"] or 0.0 for d in data]
        counts = [d["defect_count"] for d in data]

        fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=200)
        color = "#e67e22"
        ax1.set_xlabel("Simulated Inspection Day", fontsize=11, labelpad=8)
        ax1.set_ylabel("Current Severity [0, 1]", color=color, fontsize=11, labelpad=8)
        line1 = ax1.plot(days, sev, color=color, marker="o", linewidth=2.5, markersize=8, label="Severity")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(-0.05, 1.0)
        ax1.set_xticks(days)

        ax2 = ax1.twinx()
        color2 = "#2980b9"
        ax2.set_ylabel("Defect Count", color=color2, fontsize=11, labelpad=8)
        line2 = ax2.plot(days, counts, color=color2, marker="s", linestyle="--", linewidth=2, markersize=7, label="Defect Count")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(-0.2, max(counts) + 2)

        plt.title("SEG_004 (Days 01–05): Temporal Response under Overhead Drone Survey\n(Monotonic Progression: Grade C/B -> Grade D Overcast)", fontsize=12, pad=12, weight="bold")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        p6 = plots_dir / "seg004_d01_d05_temporal_response.png"
        fig.tight_layout()
        fig.savefig(p6)
        plt.close(fig)
        generated_plots.append(p6)

    # 7. SEG_004 Day 06-10: Highway Curve Vantage Overlook Response
    seg004_d06_path = output_dir / "SEG_004_D06_D10/daily_summary.json"
    if seg004_d06_path.exists():
        data = json.loads(seg004_d06_path.read_text(encoding="utf-8"))
        days = [d["day"] for d in data]
        sev = [d["current_severity"] or 0.0 for d in data]
        counts = [d["defect_count"] for d in data]

        fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=200)
        color = "#c0392b"
        ax1.set_xlabel("Simulated Inspection Day", fontsize=11, labelpad=8)
        ax1.set_ylabel("Current Severity [0, 1]", color=color, fontsize=11, labelpad=8)
        line1 = ax1.plot(days, sev, color=color, marker="o", linewidth=2.5, markersize=8, label="Severity")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(-0.05, 1.0)
        ax1.set_xticks(days)

        ax2 = ax1.twinx()
        color2 = "#8e44ad"
        ax2.set_ylabel("Defect Count", color=color2, fontsize=11, labelpad=8)
        line2 = ax2.plot(days, counts, color=color2, marker="^", linestyle="--", linewidth=2, markersize=7, label="Defect Count")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(-0.2, max(counts) + 3)

        # Annotations
        ax1.annotate("Heavy Rain Peak\n(sev=0.8134, count=12)", xy=(8, 0.8134), xytext=(6.5, 0.90),
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2), fontsize=9, weight="bold")
        ax1.annotate("Sunset & Shadow Glare\n(Marking Suppressor Drop -> 0)", xy=(9, 0.0), xytext=(8.2, 0.35),
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2), fontsize=9, weight="bold")

        plt.title("SEG_004 (Days 06–10): Temporal Response under Highway Curve Overlook\n(Heavy Rain Detection Surge followed by Glare Suppression)", fontsize=12, pad=12, weight="bold")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center left", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        p7 = plots_dir / "seg004_d06_d10_temporal_response.png"
        fig.tight_layout()
        fig.savefig(p7)
        plt.close(fig)
        generated_plots.append(p7)

    log.info("Generated %d research plots in %s", len(generated_plots), plots_dir)
    return generated_plots


def main() -> int:
    output_root = ROOT / "integration/experiment_a/temporal"
    master_csv_path = ROOT / "integration/experiment_a/temporal_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for seq_cfg in CANDIDATE_SEQUENCES:
        row = analyze_sequence(seq_cfg, output_root)
        summary_rows.append(row)

    # Write master temporal summary CSV
    fieldnames = [
        "sequence_id",
        "segment_id",
        "start_day",
        "end_day",
        "camera_preset",
        "num_states",
        "initial_severity",
        "final_severity",
        "severity_change",
        "mean_severity",
        "mean_defect_count",
        "unique_tracks",
        "mean_track_length",
        "max_track_length",
        "new_defect_events",
        "matched_events",
        "not_observed_events",
        "mean_defect_area_ratio",
        "temporal_label",
        "notes",
    ]
    with open(master_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("Saved master temporal summary to %s", master_csv_path)

    # Generate research plots
    generate_research_plots(output_root)

    print(json.dumps({
        "status": "COMPLETED",
        "sequences_analyzed": len(summary_rows),
        "output_directory": str(output_root),
        "master_summary_csv": str(master_csv_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
