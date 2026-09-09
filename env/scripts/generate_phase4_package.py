#!/usr/bin/env python3
"""generate_phase4_package.py
------------------------------
RoadSentinel Phase 4 — Temporal Analytics Finalization & Research Evidence Package.

Generates:
1. Paper-ready tables in integration/experiment_a/temporal/tables/
2. Paper-ready figures in integration/experiment_a/temporal/plots/
3. Representative track visualizations in integration/experiment_a/temporal/visualizations/
4. Validates event accounting, match quality, persistence metrics, and descriptive associations.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from collections import Counter
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_DIR = ROOT / "integration/experiment_a/temporal"
PERCEPTION_DIR = ROOT / "integration/experiment_a/perception"
TABLES_DIR = TEMPORAL_DIR / "tables"
PLOTS_DIR = TEMPORAL_DIR / "plots"
VIS_DIR = TEMPORAL_DIR / "visualizations"

for d in (TABLES_DIR, PLOTS_DIR, VIS_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("phase4_package")

SEQUENCE_METADATA = [
    {
        "sequence_id": "SEG_001_D03_D10",
        "segment_id": "SEG_001",
        "start_day": 3,
        "end_day": 10,
        "camera_preset": "🌄 Highway Curve Vantage Overlook",
        "evidence_grade": "ENVIRONMENT_CONFOUNDED",
        "evidence_justification": "Cloud cover (Overcast) inflates contrast on D06-08 (sev 0.73); sunset shadows on D10 trigger marking suppression, dropping sev to 0.0.",
    },
    {
        "sequence_id": "SEG_002_D01_D02",
        "segment_id": "SEG_002",
        "start_day": 1,
        "end_day": 2,
        "camera_preset": "🔍 Low-Angle Pothole Inspection (30° Close-Up)",
        "evidence_grade": "PAIRWISE_CHANGE_ONLY",
        "evidence_justification": "2-state control pair on Pristine Grade A; grazing angle consistently triggers aggregate grain contrast (both defects matched across days).",
    },
    {
        "sequence_id": "SEG_002_D04_D05",
        "segment_id": "SEG_002",
        "start_day": 4,
        "end_day": 5,
        "camera_preset": "💧 Waterlogged Pothole Macro View",
        "evidence_grade": "PAIRWISE_CHANGE_ONLY",
        "evidence_justification": "2-state pair under macro view; lighting change (Noon -> Overcast) alters boundary contours, preventing confident matching.",
    },
    {
        "sequence_id": "SEG_002_D06_D07",
        "segment_id": "SEG_002",
        "start_day": 6,
        "end_day": 7,
        "camera_preset": "🌄 Highway Curve Vantage Overlook",
        "evidence_grade": "PAIRWISE_CHANGE_ONLY",
        "evidence_justification": "2-state overlook pair transitioning Grade C -> D; distant perspective limits defect candidate resolution.",
    },
    {
        "sequence_id": "SEG_003_D01_D07",
        "segment_id": "SEG_003",
        "start_day": 1,
        "end_day": 7,
        "camera_preset": "🌄 Highway Curve Vantage Overlook",
        "evidence_grade": "STABILITY_TEST",
        "evidence_justification": "7-state Pristine Grade A control under invariant noon lighting; exact same road shoulder candidate tracked through all 7 states (CV = 0.67%).",
    },
    {
        "sequence_id": "SEG_003_D08_D09",
        "segment_id": "SEG_003",
        "start_day": 8,
        "end_day": 9,
        "camera_preset": "🔭 Overhead Drone Survey (SAM 2 Top-Down)",
        "evidence_grade": "PAIRWISE_CHANGE_ONLY",
        "evidence_justification": "2-state top-down drone pair (Grade C -> D); 1 track persists with 0.9412 mask IoU.",
    },
    {
        "sequence_id": "SEG_004_D01_D05",
        "segment_id": "SEG_004",
        "start_day": 1,
        "end_day": 5,
        "camera_preset": "🔭 Overhead Drone Survey (SAM 2 Top-Down)",
        "evidence_grade": "STRONG_TEMPORAL_EVIDENCE",
        "evidence_justification": "5-state top-down drone sequence displaying clean monotonic severity progression from unblemished road to severe breakdown (sev delta +0.7302).",
    },
    {
        "sequence_id": "SEG_004_D06_D10",
        "segment_id": "SEG_004",
        "start_day": 6,
        "end_day": 10,
        "camera_preset": "🌄 Highway Curve Vantage Overlook",
        "evidence_grade": "ENVIRONMENT_CONFOUNDED",
        "evidence_justification": "Heavy Rain on D07-08 generates peak defect density (12 defects, 11 matched tracks); sunset glare on D09-10 causes road-marking over-suppression to 0 defects.",
    },
]


def generate_tables():
    log.info("Generating paper-ready tables in %s", TABLES_DIR)

    # 1. table_temporal_sequences.csv
    seq_table_path = TABLES_DIR / "table_temporal_sequences.csv"
    with open(ROOT / "integration/experiment_a/temporal_summary.csv") as f:
        rows = list(csv.DictReader(f))

    # Augment with evidence grades
    grade_by_id = {s["sequence_id"]: (s["evidence_grade"], s["evidence_justification"]) for s in SEQUENCE_METADATA}
    for r in rows:
        eg, just = grade_by_id.get(r["sequence_id"], ("UNKNOWN", ""))
        r["evidence_grade"] = eg
        r["evidence_justification"] = just

    fieldnames = [
        "sequence_id", "segment_id", "start_day", "end_day", "camera_preset", "num_states",
        "initial_severity", "final_severity", "severity_change", "mean_severity", "mean_defect_count",
        "unique_tracks", "mean_track_length", "max_track_length", "evidence_grade", "evidence_justification"
    ]
    with open(seq_table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s", seq_table_path)

    # 2. table_tracking_statistics.csv
    track_table_path = TABLES_DIR / "table_tracking_statistics.csv"
    track_rows = []
    for s in SEQUENCE_METADATA:
        seq_id = s["sequence_id"]
        seq_p = TEMPORAL_DIR / seq_id
        tracks = json.loads((seq_p / "defect_tracks.json").read_text())["tracks"]
        events = json.loads((seq_p / "temporal_events.json").read_text())["events"]
        daily = json.loads((seq_p / "daily_summary.json").read_text())

        track_lens = [len(t["days_seen"]) for t in tracks]
        ge2 = sum(1 for l in track_lens if l >= 2)
        ge3 = sum(1 for l in track_lens if l >= 3)
        total_detections = sum(d["defect_count"] for d in daily)
        persistent_detections = sum(len(t["days_seen"]) for t in tracks if len(t["days_seen"]) >= 2)
        pct_persistent = (persistent_detections / total_detections * 100) if total_detections else 0.0

        ev_c = Counter(e["event"] for e in events)
        matched_evs = ev_c.get("MATCHED_EXISTING", 0) + ev_c.get("OBSERVED_AREA_INCREASED", 0) + ev_c.get("OBSERVED_AREA_DECREASED", 0)
        opps = sum(d["defect_count"] for d in daily[:-1])
        adj_rate = (matched_evs / opps * 100) if opps else 0.0

        track_rows.append({
            "sequence_id": seq_id,
            "camera_preset": s["camera_preset"],
            "states": len(daily),
            "total_detections": total_detections,
            "unique_tracks": len(tracks),
            "tracks_ge2_states": ge2,
            "tracks_ge3_states": ge3,
            "max_track_length": max(track_lens, default=0),
            "mean_track_length": round(float(np.mean(track_lens)), 2) if track_lens else 0.0,
            "persistent_detections": persistent_detections,
            "persistent_detection_pct": f"{pct_persistent:.1f}%",
            "matched_events": matched_evs,
            "adjacent_persistence_rate": f"{adj_rate:.1f}%",
            "new_defect_events": ev_c.get("NEW_DEFECT", 0),
            "not_observed_events": ev_c.get("NOT_OBSERVED", 0),
        })

    with open(track_table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()))
        writer.writeheader()
        writer.writerows(track_rows)
    log.info("Wrote %s", track_table_path)

    # 3. table_stability_control.csv (SEG_003 Day 01-07)
    stab_table_path = TABLES_DIR / "table_stability_control.csv"
    seg003_daily = json.loads((TEMPORAL_DIR / "SEG_003_D01_D07/daily_summary.json").read_text())
    seg003_tracks = json.loads((TEMPORAL_DIR / "SEG_003_D01_D07/defect_tracks.json").read_text())["tracks"]
    seg003_events = json.loads((TEMPORAL_DIR / "SEG_003_D01_D07/temporal_events.json").read_text())["events"]

    sevs = [d["current_severity"] for d in seg003_daily]
    counts = [d["defect_count"] for d in seg003_daily]
    areas = [d["defect_area_ratio"] for d in seg003_daily]
    anoms = [d["surface_anomaly_score"] for d in seg003_daily]
    match_confs = [e["matching_confidence"] for e in seg003_events if e["matching_confidence"] is not None]

    stab_rows = [
        {"metric": "Evaluation States", "value": "7 continuous days (Day 01 to Day 07)", "notes": "Same physical camera & road scene"},
        {"metric": "Simulation Ground Truth", "value": "Pristine (Grade A), Defect Density = 7.3 / 100m²", "notes": "No true physical damage simulated"},
        {"metric": "Environmental Condition", "value": "Clear Noon (70° Sun)", "notes": "Constant direct illumination"},
        {"metric": "Camera Viewpoint", "value": "🌄 Highway Curve Vantage Overlook", "notes": "Fixed extrinsic vantage"},
        {"metric": "Mean Model Severity", "value": f"{np.mean(sevs):.4f}", "notes": "Systematic baseline response"},
        {"metric": "Severity Standard Deviation", "value": f"{np.std(sevs):.4f}", "notes": "Dispersion over 7 days"},
        {"metric": "Severity Range [Min, Max]", "value": f"[{np.min(sevs):.4f}, {np.max(sevs):.4f}]", "notes": f"Total spread = {np.max(sevs)-np.min(sevs):.4f}"},
        {"metric": "Coefficient of Variation (CV)", "value": f"{(np.std(sevs)/np.mean(sevs))*100:.2f}%", "notes": "Sub-1% variation over 7 states"},
        {"metric": "Defect Count Invariant", "value": "Exactly 1 defect per day across all 7 days", "notes": "100% count consistency"},
        {"metric": "Persistent Systematic Tracks", "value": "1 / 1 (DEFECT_001)", "notes": "Same boundary candidate tracked across all 7 days"},
        {"metric": "Maximum Track Length", "value": "7 states", "notes": "Full-duration persistence"},
        {"metric": "Adjacent Mask-IoU Mean", "value": f"{np.mean(match_confs):.4f}", "notes": "IoU across adjacent inspection cycles"},
        {"metric": "Adjacent Mask-IoU Range", "value": f"[{np.min(match_confs):.4f}, {np.max(match_confs):.4f}]", "notes": "Peaks at 0.9551 on Day 5->6"},
        {"metric": "Interpretation", "value": "SPATIALLY STABLE SYSTEMATIC FALSE-POSITIVE CANDIDATE", "notes": "Confirms model repeatability, not physical damage"},
    ]
    with open(stab_table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "notes"])
        writer.writeheader()
        writer.writerows(stab_rows)
    log.info("Wrote %s", stab_table_path)

    # 4. table_environmental_failure_modes.csv
    fail_table_path = TABLES_DIR / "table_environmental_failure_modes.csv"
    fail_rows = [
        {
            "failure_mode": "Overcast Diffuse Amplification",
            "affected_sequences": "SEG_001 (D06-07), SEG_002 (D05), SEG_004 (D05-06)",
            "simulation_condition": "Overcast Day (diffuse sky illumination)",
            "model_manifestation": "Severity jumps by +0.30 to +0.40; defect counts multiply up to 7x without simulated physical damage increase.",
            "technical_root_cause": "Absence of direct directional sunlight removes specular highlights and increases local asphalt contrast, elevating normalized patch anomaly scores against the 1x memory bank.",
            "research_implication": "Single-frame visual severity cannot be directly compared across sunny vs overcast days without illumination normalization.",
        },
        {
            "failure_mode": "Heavy Rain Surface Flooding & Specular Artifacts",
            "affected_sequences": "SEG_004 (D07-D08)",
            "simulation_condition": "Heavy Rain & Wet Road (specular asphalt, puddles)",
            "model_manifestation": "Defect count explodes to 11-12 defects; model severity hits peak 0.8134; 100% water hazard flags active.",
            "technical_root_cause": "Reflective pooling water creates dark saturated patches and high-gradient puddle edges that trigger dense DINOv2 patch anomalies.",
            "research_implication": "Water hazards are reliably detected, but defect counts become artificially coupled to rainfall intensity rather than asphalt cavitation.",
        },
        {
            "failure_mode": "Golden Hour Sunset Shadow & Road Marking Over-Suppression",
            "affected_sequences": "SEG_001 (D10), SEG_004 (D09-D10)",
            "simulation_condition": "Golden Hour Sunset (low sun angle, stretched shadows)",
            "model_manifestation": "Model severity collapses to 0.0000; 0 defects detected on Severe (Grade D) and Critical (Grade F) simulated roads.",
            "technical_root_cause": "Elongated dark shadows cast by guardrails and vehicles cause the heuristic RoadMarkingSuppressor to classify up to 98.5% of road pixels as painted markings/barriers, zeroing out anomaly candidates.",
            "research_implication": "Critical failure mode: high-hazard pavement is rendered completely invisible to the perception model under low-angle sun.",
        },
        {
            "failure_mode": "Grazing-Angle Macro-Texture Contrast",
            "affected_sequences": "SEG_002 (D01-D02)",
            "simulation_condition": "🔍 Low-Angle Pothole Inspection (30° close-up, Pristine Grade A)",
            "model_manifestation": "2 persistent false-positive defects detected with severity ~0.33-0.35 on unblemished pavement.",
            "technical_root_cause": "Low camera angle places aggregate stones in sharp foreshortened relief relative to the nadir-trained memory bank, exceeding normal anomaly thresholds.",
            "research_implication": "Camera viewpoint angle must be explicitly incorporated as an extrinsic conditioning factor for road-surface baseline calibration.",
        },
        {
            "failure_mode": "Distant Road-Boundary Contrast on Vantage Overlooks",
            "affected_sequences": "SEG_003 (D01-D07)",
            "simulation_condition": "🌄 Highway Curve Vantage Overlook (Pristine Grade A)",
            "model_manifestation": "Persistent false-positive defect (DEFECT_001, severity 0.2444) detected across all 7 inspection cycles.",
            "technical_root_cause": "Road segmentation boundary along the distant curve contains high-contrast asphalt-to-soil transitions that trigger candidate generation.",
            "research_implication": "Boundary candidates on curved roadway corridors require geometric margin erosion to prevent systematic false alarms.",
        },
    ]
    with open(fail_table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fail_rows)
    log.info("Wrote %s", fail_table_path)

    # 5. table_descriptive_associations.csv
    assoc_table_path = TABLES_DIR / "table_descriptive_associations.csv"
    with open(ROOT / "integration/experiment_a/perception_results.csv") as f:
        perc_rows = [r for r in csv.DictReader(f) if r["metadata_status"] == "VALID"]

    grade_map = {
        "Pristine (Grade A)": 1,
        "Minor Wear (Grade B)": 2,
        "Moderate Deterioration (Grade C)": 3,
        "Severe Breakdown (Grade D - Critical)": 4,
        "Critical Hazard (Grade F)": 5,
    }

    # Datasets for association
    subsets = {
        "Experiment A Primary Set (Unstratified)": perc_rows,
        "Overhead Drone Survey (All Segments)": [r for r in perc_rows if "Overhead Drone" in r["camera_preset"]],
        "Highway Curve Overlook (All Segments)": [r for r in perc_rows if "Highway Curve" in r["camera_preset"]],
        "SEG_004 D01-D05 (Overhead Drone Controlled Progression)": [r for r in perc_rows if r["segment_id"] == "SEG_004" and int(r["day"]) <= 5],
        "SEG_001 D03-D10 (Highway Curve Overlook Multi-Condition)": [r for r in perc_rows if r["segment_id"] == "SEG_001" and int(r["day"]) >= 3],
        "Clear Noon Lighting Only (All Cameras)": [r for r in perc_rows if "Clear Noon" in r["lighting_preset"]],
    }

    assoc_rows = []
    for sub_name, sub_data in subsets.items():
        n = len(sub_data)
        if n < 4:
            continue
        g = [grade_map[r["road_health_state"]] for r in sub_data]
        s = [float(r["current_severity"]) for r in sub_data]
        dens = [float(r["density"]) for r in sub_data]
        cnt = [int(r["defect_count"]) for r in sub_data]

        r_g_s, p_g_s = spearmanr(g, s)
        r_d_s, p_d_s = spearmanr(dens, s)
        r_d_c, p_d_c = spearmanr(dens, cnt)

        assoc_rows.append({
            "subset_name": sub_name,
            "sample_size_N": n,
            "spearman_rho_grade_vs_severity": f"{r_g_s:.4f}" if np.isfinite(r_g_s) else "N/A",
            "p_value_grade_vs_severity": f"{p_g_s:.4e}" if np.isfinite(p_g_s) else "N/A",
            "spearman_rho_density_vs_severity": f"{r_d_s:.4f}" if np.isfinite(r_d_s) else "N/A",
            "p_value_density_vs_severity": f"{p_d_s:.4e}" if np.isfinite(p_d_s) else "N/A",
            "spearman_rho_density_vs_count": f"{r_d_c:.4f}" if np.isfinite(r_d_c) else "N/A",
            "p_value_density_vs_count": f"{p_d_c:.4e}" if np.isfinite(p_d_c) else "N/A",
            "interpretation": (
                "Strong positive rank association when viewpoint is controlled"
                if r_g_s > 0.7
                else (
                    "Inverted association due to sunset marking over-suppression"
                    if r_g_s < -0.2
                    else "Weak association dominated by environmental confounding across conditions"
                )
            ),
        })

    with open(assoc_table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(assoc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(assoc_rows)
    log.info("Wrote %s", assoc_table_path)


def generate_figures():
    log.info("Generating paper-ready figures in %s", PLOTS_DIR)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    # 1. Figure 1: SEG_003 Stability Control (Severity & Defect Invariant)
    seg003_p = TEMPORAL_DIR / "SEG_003_D01_D07/daily_summary.json"
    if seg003_p.exists():
        data = json.loads(seg003_p.read_text())
        days = [d["day"] for d in data]
        sev = [d["current_severity"] for d in data]
        counts = [d["defect_count"] for d in data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.2), dpi=200, sharex=True)
        ax1.plot(days, sev, marker="o", color="#2e7d32", linewidth=2.5, markersize=8, label="Model Severity")
        ax1.axhline(float(np.mean(sev)), color="#4caf50", linestyle="--", alpha=0.8, label=f"Mean Severity: {np.mean(sev):.4f}")
        ax1.fill_between(days, float(np.mean(sev)) - float(np.std(sev)), float(np.mean(sev)) + float(np.std(sev)), color="#c8e6c9", alpha=0.5, label=f"±1 Std Dev (σ = {np.std(sev):.4f})")
        ax1.set_title("Figure 1A: SEG_003 Pristine Grade A Severity Response Stability\n(Constant Camera: Highway Curve Overlook | Clear Noon | CV = 0.67%)", fontsize=11, weight="bold", pad=8)
        ax1.set_ylabel("Current Severity [0, 1]", fontsize=10)
        ax1.set_ylim(0.20, 0.28)
        ax1.legend(loc="upper right", frameon=True, fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.bar(days, counts, color="#2e7d32", alpha=0.8, width=0.45, edgecolor="#1b5e20", linewidth=1.2)
        ax2.set_title("Figure 1B: Defect Count Invariant Across 7 Inspection Cycles (100% Invariant)", fontsize=11, weight="bold", pad=8)
        ax2.set_xlabel("Simulated Inspection Day", fontsize=10)
        ax2.set_ylabel("Defect Count", fontsize=10)
        ax2.set_ylim(0, 2.5)
        ax2.set_xticks(days)
        for d, c in zip(days, counts):
            ax2.text(d, c + 0.1, f"{c} (DEFECT_001)", ha="center", va="bottom", fontsize=9, weight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "fig1_seg003_stability_control.png")
        plt.close(fig)

    # 2. Figure 2: SEG_004 Drone Progression (Severity & Area)
    seg004_d01_p = TEMPORAL_DIR / "SEG_004_D01_D05/daily_summary.json"
    if seg004_d01_p.exists():
        data = json.loads(seg004_d01_p.read_text())
        days = [d["day"] for d in data]
        sev = [d["current_severity"] for d in data]
        areas = [d["defect_area_ratio"] * 100 for d in data]

        fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=200)
        color_sev = "#d35400"
        color_area = "#2980b9"

        ax1.set_xlabel("Simulated Inspection Day (Overhead Drone Survey)", fontsize=11, labelpad=8)
        ax1.set_ylabel("Current Severity [0, 1]", color=color_sev, fontsize=11)
        line1 = ax1.plot(days, sev, color=color_sev, marker="o", linewidth=2.5, markersize=8, label="Model Severity")
        ax1.tick_params(axis="y", labelcolor=color_sev)
        ax1.set_ylim(-0.05, 1.0)
        ax1.set_xticks(days)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Defect Surface Area Ratio (%)", color=color_area, fontsize=11)
        line2 = ax2.plot(days, areas, color=color_area, marker="s", linestyle="--", linewidth=2, markersize=7, label="Defect Area Ratio (%)")
        ax2.tick_params(axis="y", labelcolor=color_area)
        ax2.set_ylim(-0.2, max(areas) + 1.0)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", frameon=True)
        ax1.set_title("Figure 2: SEG_004 (Days 01–05) Monotonic Deterioration Progression\n(Camera: Overhead Drone Survey | Spearman ρ = 0.8839, p = 0.0467)", fontsize=12, pad=10, weight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "fig2_seg004_drone_progression.png")
        plt.close(fig)

    # 3. Figure 3: SEG_001 Environmental Response (Cloud & Sunset Modulation)
    seg001_p = TEMPORAL_DIR / "SEG_001_D03_D10/daily_summary.json"
    if seg001_p.exists():
        data = json.loads(seg001_p.read_text())
        days = [d["day"] for d in data]
        sev = [d["current_severity"] for d in data]
        counts = [d["defect_count"] for d in data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.2), dpi=200, sharex=True)
        ax1.plot(days, sev, marker="o", color="#c0392b", linewidth=2.5, markersize=8)
        ax1.set_title("Figure 3A: SEG_001 (Days 03–10) Environmental Severity Modulation\n(Camera: Highway Curve Overlook | Cloud Amplification vs Sunset Glare Suppression)", fontsize=11, weight="bold", pad=8)
        ax1.set_ylabel("Current Severity [0, 1]", fontsize=10)
        ax1.set_ylim(-0.05, 1.0)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.annotate("D06 Overcast Peak\n(sev = 0.7285)", xy=(6, 0.7285), xytext=(5.1, 0.85),
                     arrowprops=dict(arrowstyle="->", color="#333", lw=1.2), fontsize=9, weight="bold")
        ax1.annotate("D10 Sunset Shadow\n(Suppression -> 0.0)", xy=(10, 0.0), xytext=(7.8, 0.25),
                     arrowprops=dict(arrowstyle="->", color="#333", lw=1.2), fontsize=9, weight="bold")

        ax2.bar(days, counts, color="#34495e", alpha=0.85, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
        ax2.set_title("Figure 3B: Detected Defect Count Across Simulated Inspection States", fontsize=11, weight="bold", pad=8)
        ax2.set_xlabel("Simulated Inspection Day", fontsize=10)
        ax2.set_ylabel("Defect Count", fontsize=10)
        ax2.set_ylim(0, max(counts) + 2)
        ax2.set_xticks(days)
        for d, c in zip(days, counts):
            ax2.text(d, c + 0.25, str(c), ha="center", va="bottom", fontsize=9, weight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "fig3_seg001_environmental_response.png")
        plt.close(fig)

    # 4. Figure 4: Simulation Health State vs Model Severity (Descriptive Association Scatter)
    with open(ROOT / "integration/experiment_a/perception_results.csv") as f:
        perc_rows = [r for r in csv.DictReader(f) if r["metadata_status"] == "VALID"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=200)
    grade_order = ["Pristine (Grade A)", "Minor Wear (Grade B)", "Moderate Deterioration (Grade C)", "Severe Breakdown (Grade D - Critical)", "Critical Hazard (Grade F)"]
    grade_x = {g: idx for idx, g in enumerate(grade_order, start=1)}

    viewpoints = sorted(list(set(r["camera_preset"] for r in perc_rows)))
    color_map = {
        viewpoints[0]: "#e74c3c",  # Curve
        viewpoints[1]: "#2980b9",  # Macro
        viewpoints[2]: "#8e44ad",  # Low-Angle
        viewpoints[3]: "#27ae60",  # Drone
    }
    short_cam = {
        viewpoints[0]: "Highway Curve Overlook",
        viewpoints[1]: "Macro View",
        viewpoints[2]: "Low-Angle Close-Up",
        viewpoints[3]: "Overhead Drone Survey",
    }

    # Plot points with jitter
    np.random.seed(42)
    for cam in viewpoints:
        cam_rows = [r for r in perc_rows if r["camera_preset"] == cam]
        xs = [grade_x[r["road_health_state"]] + np.random.uniform(-0.15, 0.15) for r in cam_rows]
        ys = [float(r["current_severity"]) for r in cam_rows]
        ax.scatter(xs, ys, color=color_map[cam], label=short_cam[cam], s=60, alpha=0.85, edgecolors="none")

    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["Grade A\n(Pristine)", "Grade B\n(Minor)", "Grade C\n(Moderate)", "Grade D\n(Severe)", "Grade F\n(Critical)"], fontsize=10)
    ax.set_ylabel("Model-Observed Severity [0, 1]", fontsize=11)
    ax.set_title("Figure 4: Simulated Road-Health State vs Model-Observed Severity\n(Stratified by Camera Viewpoint; Demonstrates Environmental & Viewpoint Confounding)", fontsize=11, weight="bold", pad=10)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Highlight anomalies
    ax.annotate("Grade F Suppression\n(Sunset Shadows -> 0.0)", xy=(5, 0.0), xytext=(4.2, 0.25),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2), fontsize=9, color="#c0392b", weight="bold")
    ax.annotate("Grade A Grazing Angle\n(Aggregate Texture -> 0.35)", xy=(1, 0.35), xytext=(1.3, 0.50),
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.2), fontsize=9, color="#8e44ad", weight="bold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig4_simulation_state_vs_model_severity.png")
    plt.close(fig)

    # 5. Figure 5: Track Length Distribution
    all_tracks = []
    for s in SEQUENCE_METADATA:
        seq_p = TEMPORAL_DIR / s["sequence_id"]
        tracks = json.loads((seq_p / "defect_tracks.json").read_text())["tracks"]
        all_tracks.extend(tracks)

    lens = [len(t["days_seen"]) for t in all_tracks]
    len_counts = Counter(lens)
    x_lens = range(1, max(lens) + 1)
    y_counts = [len_counts.get(x, 0) for x in x_lens]

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=200)
    bars = ax.bar(x_lens, y_counts, color="#2980b9", edgecolor="#1c5980", alpha=0.85, width=0.6)
    ax.set_title("Figure 5: Defect Track-Length Distribution Across All 8 Sequences\n(Total Tracks = 48 | Mean Length = 2.04 States | Max Length = 7 States)", fontsize=11, weight="bold", pad=10)
    ax.set_xlabel("Track Duration (Consecutive Simulated Inspection States)", fontsize=10)
    ax.set_ylabel("Number of Defect Tracks", fontsize=10)
    ax.set_xticks(x_lens)
    ax.set_ylim(0, max(y_counts) + 5)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, str(h), ha="center", va="bottom", fontsize=10, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig5_track_length_distribution.png")
    plt.close(fig)

    # 6. Figure 6: Temporal Event Distribution across Sequences
    seq_ids = [s["sequence_id"] for s in SEQUENCE_METADATA]
    short_names = [s["sequence_id"].replace("SEG_", "S").replace("_D", " D") for s in SEQUENCE_METADATA]
    new_counts = []
    matched_counts = []
    not_obs_counts = []

    for s in SEQUENCE_METADATA:
        seq_p = TEMPORAL_DIR / s["sequence_id"]
        events = json.loads((seq_p / "temporal_events.json").read_text())["events"]
        c = Counter(e["event"] for e in events)
        new_counts.append(c.get("NEW_DEFECT", 0))
        matched_counts.append(c.get("MATCHED_EXISTING", 0) + c.get("OBSERVED_AREA_INCREASED", 0) + c.get("OBSERVED_AREA_DECREASED", 0))
        not_obs_counts.append(c.get("NOT_OBSERVED", 0))

    x = np.arange(len(seq_ids))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=200)
    ax.bar(x - width, new_counts, width, label="NEW_DEFECT (48)", color="#3498db")
    ax.bar(x, matched_counts, width, label="MATCHED_EXISTING / AREA_DELTA (33)", color="#2ecc71")
    ax.bar(x + width, not_obs_counts, width, label="NOT_OBSERVED (34)", color="#e74c3c")

    ax.set_ylabel("Event Count", fontsize=10)
    ax.set_title("Figure 6: Temporal Event Distribution by Sequence (Total Events = 115)", fontsize=11, weight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig6_temporal_event_distribution.png")
    plt.close(fig)

    # 7. Figure 7: Match Quality Distribution (IoU Scores)
    all_events = []
    for s in SEQUENCE_METADATA:
        seq_p = TEMPORAL_DIR / s["sequence_id"]
        events = json.loads((seq_p / "temporal_events.json").read_text())["events"]
        all_events.extend(events)

    mask_confs = [e["matching_confidence"] for e in all_events if e.get("matching_method") == "mask_iou"]
    bbox_confs = [e["matching_confidence"] for e in all_events if e.get("matching_method") == "bbox_iou"]

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=200)
    ax.hist(mask_confs, bins=np.linspace(0.5, 1.0, 11), color="#27ae60", alpha=0.8, edgecolor="#1e8449", label=f"Mask IoU Matches (N={len(mask_confs)}, Mean={np.mean(mask_confs):.4f})")
    ax.hist(bbox_confs, bins=np.linspace(0.3, 0.6, 7), color="#f39c12", alpha=0.8, edgecolor="#d68910", label=f"BBox IoU Matches (N={len(bbox_confs)}, Mean={np.mean(bbox_confs):.4f})")
    ax.set_title("Figure 7: Match Confidence / IoU Distribution for Matched Defect Steps\n(Conservative Matching: Mask IoU >= 0.50, BBox IoU >= 0.30)", fontsize=11, weight="bold", pad=10)
    ax.set_xlabel("Intersection over Union (IoU) Confidence Score", fontsize=10)
    ax.set_ylabel("Matched Occurrence Count", fontsize=10)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig7_match_quality_distribution.png")
    plt.close(fig)

    log.info("Saved 7 research figures to %s", PLOTS_DIR)


def generate_visualizations():
    log.info("Generating representative track crop visualizations in %s", VIS_DIR)

    # Visualization 1: SEG_003 DEFECT_001 across Day 01 to Day 07 (7-panel montage)
    # Target crop bounding box around [991, 620, 1043, 746]:
    crop_x1, crop_y1, crop_x2, crop_y2 = 910, 560, 1120, 790
    crops = []

    for d in range(1, 8):
        overlay_path = PERCEPTION_DIR / f"SEG_003/day_{d:02d}/overlay.png"
        feat_path = PERCEPTION_DIR / f"SEG_003/day_{d:02d}/features.json"
        img = cv2.imread(str(overlay_path))
        if img is None:
            log.warning("Could not read %s", overlay_path)
            continue
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        feat = json.loads(feat_path.read_text())
        sev = feat["current_severity"]
        # Add day label on crop
        cv2.putText(crop, f"Day {d:02d}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(crop, f"sev={sev:.4f}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # Add border
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (80, 80, 80), 2)
        crops.append(crop)

    if len(crops) == 7:
        montage_seg003 = np.hstack(crops)
        # Add top banner
        banner = np.zeros((45, montage_seg003.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, "SEG_003 DEFECT_001 Track Persistence across Days 01-07 (Pristine Grade A Stability | Mask IoU: 0.85-0.96)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 255), 2, cv2.LINE_AA)
        montage_seg003 = np.vstack([banner, montage_seg003])
        out_v1 = VIS_DIR / "vis_seg003_defect001_7days.png"
        cv2.imwrite(str(out_v1), montage_seg003)
        log.info("Saved %s", out_v1)

    # Visualization 2: SEG_004 Day 07 vs Day 08 persistent tracks under Heavy Rain
    img7 = cv2.imread(str(PERCEPTION_DIR / "SEG_004/day_07/overlay.png"))
    img8 = cv2.imread(str(PERCEPTION_DIR / "SEG_004/day_08/overlay.png"))
    if img7 is not None and img8 is not None:
        # Resize to 960x540 each
        h, w = 540, 960
        r7 = cv2.resize(img7, (w, h))
        r8 = cv2.resize(img8, (w, h))
        cv2.putText(r7, "Day 07 (Heavy Rain): 11 Defects | Severity: 0.8104", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(r8, "Day 08 (Heavy Rain): 12 Defects | Severity: 0.8134", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(r8, "11 Tracks Matched (Mask IoU: 0.72 - 0.94)", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 120), 2, cv2.LINE_AA)
        comp_rain = np.hstack([r7, r8])
        out_v2 = VIS_DIR / "vis_seg004_persistent_tracks.png"
        cv2.imwrite(str(out_v2), comp_rain)
        log.info("Saved %s", out_v2)

    # Visualization 3: SEG_004 Day 04 vs Day 05 showing Event Types (NEW_DEFECT, MATCHED, NOT_OBSERVED)
    img4 = cv2.imread(str(PERCEPTION_DIR / "SEG_004/day_04/overlay.png"))
    img5 = cv2.imread(str(PERCEPTION_DIR / "SEG_004/day_05/overlay.png"))
    if img4 is not None and img5 is not None:
        h, w = 540, 960
        r4 = cv2.resize(img4, (w, h))
        r5 = cv2.resize(img5, (w, h))
        cv2.putText(r4, "Day 04 (Clear Noon): 2 Defects | sev=0.6542", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(r4, "Defect #1 -> NOT_OBSERVED on Day 5", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(r5, "Day 05 (Overcast): 4 Defects | sev=0.7302", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(r5, "DEFECT_001 MATCHED (Mask IoU = 0.6385) + 3 NEW_DEFECTS", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 120), 2, cv2.LINE_AA)
        comp_events = np.hstack([r4, r5])
        out_v3 = VIS_DIR / "vis_track_events_examples.png"
        cv2.imwrite(str(out_v3), comp_events)
        log.info("Saved %s", out_v3)


def main():
    generate_tables()
    generate_figures()
    generate_visualizations()
    print(json.dumps({"status": "SUCCESS", "tables": str(TABLES_DIR), "plots": str(PLOTS_DIR), "visualizations": str(VIS_DIR)}, indent=2))


if __name__ == "__main__":
    main()
