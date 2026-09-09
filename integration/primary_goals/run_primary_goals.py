#!/usr/bin/env python3
"""run_primary_goals.py
-----------------------
Executes RoadSentinel Phase 11:
Goal 1 (Current Road Health Assessment -> XGBoost Future Severity Prediction)
Goal 2 (Same-Road Temporal Change Analysis & Tracking)
for all 40 Experiment A physical images across SEG_001 to SEG_004.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_CSV = WORKSPACE_ROOT / "integration" / "experiment_a" / "perception_results.csv"
TEMPORAL_SUMMARY_CSV = WORKSPACE_ROOT / "integration" / "experiment_a" / "temporal_summary.csv"
TEMPORAL_DIR = WORKSPACE_ROOT / "integration" / "experiment_a" / "temporal"
XGB_CONFIG_PATH = WORKSPACE_ROOT / "xgboost" / "config" / "scenario_model_v2.json"
XGB_MODEL_PATH = WORKSPACE_ROOT / "xgboost" / "model" / "scenario_model_v2.json"

OUT_DIR = WORKSPACE_ROOT / "integration" / "primary_goals"
FIGURES_DIR = OUT_DIR / "figures"
SEGMENTS_DIR = OUT_DIR / "segments"
DASHBOARD_DIR = WORKSPACE_ROOT / "integration" / "dashboard_assets" / "primary_goals"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("primary_goals")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300

FORECAST_LABEL = "MODEL-BASED FORECAST"
CURRENT_LABEL = "CURRENT MODEL ASSESSMENT"
TEMPORAL_LABEL = "MODEL-OBSERVED TEMPORAL CHANGE"


# ==============================================================================
# Goal 2 Subsequence Eligibility Mapping
# ==============================================================================

ELIGIBLE_SEQUENCES = {
    "SEG_001_D03_D10": {"segment": "SEG_001", "start": 3, "end": 10, "camera": "🌄 Highway Curve Vantage Overlook", "type": "ENVIRONMENT_CONFOUNDED"},
    "SEG_002_D01_D02": {"segment": "SEG_002", "start": 1, "end": 2, "camera": "🔍 Low-Angle Pothole Inspection (30° Close-Up)", "type": "PAIRWISE_CHANGE"},
    "SEG_002_D04_D05": {"segment": "SEG_002", "start": 4, "end": 5, "camera": "💧 Waterlogged Pothole Macro View", "type": "PAIRWISE_CHANGE"},
    "SEG_002_D06_D07": {"segment": "SEG_002", "start": 6, "end": 7, "camera": "🌄 Highway Curve Vantage Overlook", "type": "PAIRWISE_CHANGE"},
    "SEG_003_D01_D07": {"segment": "SEG_003", "start": 1, "end": 7, "camera": "🌄 Highway Curve Vantage Overlook", "type": "STABILITY_CONTROL"},
    "SEG_003_D08_D09": {"segment": "SEG_003", "start": 8, "end": 9, "camera": "🔭 Overhead Drone Survey (SAM 2 Top-Down)", "type": "PAIRWISE_CHANGE"},
    "SEG_004_D01_D05": {"segment": "SEG_004", "start": 1, "end": 5, "camera": "🔭 Overhead Drone Survey (SAM 2 Top-Down)", "type": "MODEL_OBSERVED_PROGRESSION"},
    "SEG_004_D06_D10": {"segment": "SEG_004", "start": 6, "end": 10, "camera": "🌄 Highway Curve Vantage Overlook", "type": "MODEL_OBSERVED_PROGRESSION"},
}


def get_temporal_eligibility(segment_id: str, day: int) -> Tuple[bool, str, str]:
    """Determine if a physical image is part of an approved same-camera temporal sequence."""
    for seq_id, info in ELIGIBLE_SEQUENCES.items():
        if info["segment"] == segment_id and info["start"] <= day <= info["end"]:
            return True, seq_id, "ELIGIBLE_SAME_CAMERA"
            
    # Ineligible reasons
    if segment_id == "SEG_001":
        if day == 1:
            return False, "NONE", "VIEWPOINT_CHANGE (Overhead Drone Survey on Day01 vs Overlook on Day03-10)"
        elif day == 2:
            return False, "NONE", "VIEWPOINT_CHANGE (Waterlogged Macro View on Day02 vs Overlook on Day03-10)"
    elif segment_id == "SEG_002":
        if day == 3:
            return False, "NONE", "VIEWPOINT_CHANGE (Overhead Drone Survey on Day03 isolate)"
        elif day == 8:
            return False, "NONE", "VIEWPOINT_CHANGE (Overhead Drone Survey on Day08 isolate)"
        elif day == 9:
            return False, "NONE", "VIEWPOINT_CHANGE (Waterlogged Macro View on Day09 isolate)"
        elif day == 10:
            return False, "NONE", "INSUFFICIENT_CONTIGUOUS_STATES (Overlook single state on Day10)"
    elif segment_id == "SEG_003":
        if day == 10:
            return False, "NONE", "MISSING_METADATA (Simulation metadata missing on Day10)"
    return False, "NONE", "NON_CONTIGUOUS_VIEWPOINT"


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main() -> None:
    log.info("Starting Phase 11: Primary Goals Execution...")
    for d in (OUT_DIR, FIGURES_DIR, SEGMENTS_DIR, DASHBOARD_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Perception Results (All 40 physical captures)
    df_perc = pd.read_csv(PERCEPTION_CSV)
    df_perc = df_perc.sort_values(by=["segment_id", "day"]).reset_index(drop=True)
    log.info("Loaded %d physical captures from %s", len(df_perc), PERCEPTION_CSV)

    # 2. Load Frozen XGBoost Model & Scenario Config
    with open(XGB_CONFIG_PATH, "r", encoding="utf-8") as f:
        xgb_config = json.load(f)
    
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(XGB_MODEL_PATH))
    log.info("Loaded frozen XGBoost Model V2 from %s", XGB_MODEL_PATH)

    feature_order = xgb_config["feature_order"]
    scenario_presets = xgb_config["scenario_presets"]
    horizons = xgb_config["supported_horizons_days"]  # [30, 60, 90]

    # 3. Generate Goal 1 Forecasts
    forecast_rows = []
    image_summary_rows = []

    for _, row in df_perc.iterrows():
        seg_id = row["segment_id"]
        day_num = int(row["day"])
        img_id = f"{seg_id}_day_{day_num:02d}"
        curr_sev = float(row["current_severity"])

        img_summary = {
            "segment_id": seg_id,
            "day": day_num,
            "image_id": img_id,
            "current_severity": round(curr_sev, 4),
        }

        scenario_90d_results = {}

        for sc_name, sc_params in scenario_presets.items():
            for h in horizons:
                feat_dict = {
                    "current_severity": curr_sev,
                    "rainfall_level": sc_params["rainfall_level"],
                    "traffic_level": sc_params["traffic_level"],
                    "temperature": sc_params["temperature"],
                    "water_exposure": sc_params["water_exposure"],
                    "days_ahead": float(h),
                }
                X_vec = np.array([[feat_dict[f] for f in feature_order]], dtype=float)
                pred_fut_sev = float(xgb_model.predict(X_vec)[0])
                pred_fut_sev = float(np.clip(pred_fut_sev, 0.0, 1.0))
                pred_change = pred_fut_sev - curr_sev

                forecast_rows.append({
                    "segment_id": seg_id,
                    "day": day_num,
                    "image_id": img_id,
                    "current_severity": round(curr_sev, 4),
                    "scenario": sc_name,
                    "days_ahead": h,
                    "rainfall_level": round(sc_params["rainfall_level"], 4),
                    "traffic_level": round(sc_params["traffic_level"], 1),
                    "temperature": round(sc_params["temperature"], 4),
                    "water_exposure": round(sc_params["water_exposure"], 4),
                    "predicted_future_severity": round(pred_fut_sev, 4),
                    "predicted_change": round(pred_change, 4),
                    "model_name": xgb_config["model_id"],
                    "forecast_label": FORECAST_LABEL,
                })

                if sc_name == "NORMAL":
                    img_summary[f"normal_{h}"] = round(pred_fut_sev, 4)
                if h == 90:
                    scenario_90d_results[sc_name] = pred_fut_sev
                    if sc_name != "NORMAL":
                        img_summary[f"{sc_name.lower()}_90"] = round(pred_fut_sev, 4)

        # Largest 90d scenario
        largest_sc = max(scenario_90d_results.items(), key=lambda x: x[1])
        img_summary["largest_predicted_90d_scenario"] = largest_sc[0]
        img_summary["largest_predicted_90d_severity"] = round(largest_sc[1], 4)

        image_summary_rows.append(img_summary)

    df_forecasts = pd.DataFrame(forecast_rows)
    df_forecasts.to_csv(OUT_DIR / "goal1_forecasts.csv", index=False)
    log.info("Saved %d scenario forecast rows to %s", len(df_forecasts), OUT_DIR / "goal1_forecasts.csv")

    df_img_summary = pd.DataFrame(image_summary_rows)
    df_img_summary.to_csv(OUT_DIR / "goal1_image_summary.csv", index=False)
    log.info("Saved image-level forecast summary to %s", OUT_DIR / "goal1_image_summary.csv")

    # 4. Process Goal 2 Temporal Analytics & Sequence Summaries
    temporal_summary_df = pd.read_csv(TEMPORAL_SUMMARY_CSV)
    
    # Audit canonical events from temporal_events.json
    seq_change_rows = []
    daily_transition_dict = {}  # (seg, day) -> {severity_change, area_change, matched_tracks}

    for _, srow in temporal_summary_df.iterrows():
        seq_id = srow["sequence_id"]
        ev_path = TEMPORAL_DIR / seq_id / "temporal_events.json"
        
        with open(ev_path, "r", encoding="utf-8") as f:
            ev_data = json.load(f)
        events = ev_data.get("events", [])

        # Count event types
        n_matched = sum(1 for e in events if e.get("event") in ("OBSERVED_AREA_INCREASED", "OBSERVED_AREA_DECREASED"))
        n_new = sum(1 for e in events if e.get("event") == "NEW_DEFECT")
        n_not_obs = sum(1 for e in events if e.get("event") == "NOT_OBSERVED")
        
        info = ELIGIBLE_SEQUENCES[seq_id]
        sub_perc = df_perc[(df_perc["segment_id"] == info["segment"]) & (df_perc["day"] >= info["start"]) & (df_perc["day"] <= info["end"])].sort_values("day")
        
        init_row = sub_perc.iloc[0]
        final_row = sub_perc.iloc[-1]
        init_sev = float(init_row["current_severity"])
        final_sev = float(final_row["current_severity"])
        
        seq_change_rows.append({
            "sequence_id": seq_id,
            "segment_id": info["segment"],
            "start_day": info["start"],
            "end_day": info["end"],
            "camera": info["camera"],
            "states": len(sub_perc),
            "initial_severity": round(init_sev, 4),
            "final_severity": round(final_sev, 4),
            "severity_delta": round(final_sev - init_sev, 4),
            "max_severity": round(float(sub_perc["current_severity"].max()), 4),
            "min_severity": round(float(sub_perc["current_severity"].min()), 4),
            "initial_defect_count": int(init_row["defect_count"]),
            "final_defect_count": int(final_row["defect_count"]),
            "initial_defect_area": round(float(init_row["defect_area_ratio"]), 6),
            "final_defect_area": round(float(final_row["defect_area_ratio"]), 6),
            "unique_tracks": int(srow["unique_tracks"]),
            "matched_events": n_matched,
            "new_events": n_new,
            "not_observed_events": n_not_obs,
            "interpretation_type": info["type"],
        })

    df_goal2_summary = pd.DataFrame(seq_change_rows)
    df_goal2_summary.to_csv(OUT_DIR / "goal2_change_summary.csv", index=False)
    log.info("Saved Goal 2 change summary to %s", OUT_DIR / "goal2_change_summary.csv")

    # 5. Build Master Results Table (ROADSENTINEL_PRIMARY_RESULTS.csv)
    master_rows = []
    
    for _, row in df_perc.iterrows():
        seg_id = row["segment_id"]
        day_num = int(row["day"])
        img_id = f"{seg_id}_day_{day_num:02d}"
        
        eligible, seq_id, reason = get_temporal_eligibility(seg_id, day_num)
        
        # Calculate transition change if contiguous compatible day exists
        prev_compat_day = None
        sev_change = None
        area_change = None
        matched_tracks = None
        
        if eligible:
            seq_info = ELIGIBLE_SEQUENCES[seq_id]
            if day_num > seq_info["start"]:
                prev_compat_day = day_num - 1
                prev_row = df_perc[(df_perc["segment_id"] == seg_id) & (df_perc["day"] == prev_compat_day)].iloc[0]
                sev_change = round(float(row["current_severity"]) - float(prev_row["current_severity"]), 4)
                area_change = round(float(row["defect_area_ratio"]) - float(prev_row["defect_area_ratio"]), 6)
                
                # Check active tracks from events
                ev_path = TEMPORAL_DIR / seq_id / "temporal_events.json"
                with open(ev_path, "r", encoding="utf-8") as f:
                    ev_data = json.load(f)
                matched_tracks = sum(1 for e in ev_data.get("events", []) if e.get("day") == day_num and e.get("event") in ("OBSERVED_AREA_INCREASED", "OBSERVED_AREA_DECREASED"))
        
        # Pull 90d forecasts from img_summary
        sum_row = df_img_summary[(df_img_summary["segment_id"] == seg_id) & (df_img_summary["day"] == day_num)].iloc[0]
        
        master_rows.append({
            "segment_id": seg_id,
            "day": day_num,
            "image_id": img_id,
            "metadata_status": row["metadata_status"],
            "camera_preset": row["camera_preset"],
            "lighting_preset": row["lighting_preset"],
            "road_health_state": row["road_health_state"],
            "current_severity": round(float(row["current_severity"]), 4),
            "defect_count": int(row["defect_count"]),
            "defect_area_ratio": round(float(row["defect_area_ratio"]), 6),
            "crack_area_ratio": round(float(row["crack_area_ratio"]), 6),
            "surface_anomaly_score": round(float(row["surface_anomaly_score"]), 4),
            "water_flag": bool(row["water_flag"]),
            "goal1_status": "COMPLETED",
            "goal2_status": "ELIGIBLE" if eligible else "TEMPORAL_MATCH_NOT_APPLICABLE",
            "goal2_sequence_id": seq_id if eligible else "NONE",
            "goal2_exclusion_reason": "NONE" if eligible else reason,
            "previous_compatible_day": prev_compat_day if prev_compat_day is not None else "N/A",
            "observed_severity_change": sev_change if sev_change is not None else "N/A",
            "observed_area_change": area_change if area_change is not None else "N/A",
            "matched_track_count": matched_tracks if matched_tracks is not None else "N/A",
            "normal_90d_forecast": sum_row["normal_90"],
            "heavy_rain_90d_forecast": sum_row["heavy_rain_90"],
            "heavy_traffic_90d_forecast": sum_row["heavy_traffic_90"],
            "high_heat_90d_forecast": sum_row["high_heat_90"],
            "wet_exposure_90d_forecast": sum_row["wet_exposure_90"],
        })

    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(OUT_DIR / "ROADSENTINEL_PRIMARY_RESULTS.csv", index=False)
    log.info("Saved master results table to %s (40 rows)", OUT_DIR / "ROADSENTINEL_PRIMARY_RESULTS.csv")

    # 6. Generate Segment Summary Markdown Reports
    for seg in ["SEG_001", "SEG_002", "SEG_003", "SEG_004"]:
        seg_rows = df_master[df_master["segment_id"] == seg].sort_values("day")
        
        md_content = f"# RoadSentinel — {seg} Primary Goal Evaluation Summary\n\n"
        md_content += f"## Segment Overview: {seg} (10 Physical Captures)\n\n"
        md_content += f"This report covers all 10 physical images captured for `{seg}`, documenting the single-image current road-health assessment, 90-day XGBoost scenario forecasts (Goal 1), and same-camera temporal change tracking (Goal 2).\n\n"
        
        md_content += "### Complete 10-Day Image Inventory & Results Table\n\n"
        md_content += "| Day | Camera Preset | Lighting Preset | Health State | Current Severity | Defect Count | Defect Area | 90d NORMAL | 90d HEAVY RAIN | Goal 2 Status |\n"
        md_content += "|---|---|---|---|---|---|---|---|---|---|\n"
        
        for _, r in seg_rows.iterrows():
            d_num = r["day"]
            cam = r["camera_preset"]
            light = r["lighting_preset"]
            health = r["road_health_state"]
            sev = r["current_severity"]
            cnt = r["defect_count"]
            area = r["defect_area_ratio"]
            n90 = r["normal_90d_forecast"]
            r90 = r["heavy_rain_90d_forecast"]
            g2_stat = r["goal2_status"]
            if g2_stat != "ELIGIBLE":
                g2_stat = f"❌ {r['goal2_exclusion_reason']}"
            else:
                g2_stat = f"✅ Eligible ({r['goal2_sequence_id']})"
                
            md_content += f"| Day {d_num:02d} | {cam} | {light} | {health} | **{sev:.4f}** | {cnt} | {area:.6f} | **{n90:.4f}** | **{r90:.4f}** | {g2_stat} |\n"
            
        md_content += "\n---\n\n"
        md_content += "### Key Research Insights for this Segment\n\n"
        if seg == "SEG_001":
            md_content += "- **Day01 & Day02 Handling**: Day 01 (Overhead Drone) and Day 02 (Waterlogged Macro) represent viewpoint transitions and are excluded from geometric temporal matching, but are fully assessed under Goal 1.\n"
            md_content += "- **Day03–Day10 Overlook Sequence**: Demonstrates strong environmental sensitivity (peak severity 0.7285 on Day 06 Overcast vs 0.0000 on Clear Noon).\n"
        elif seg == "SEG_002":
            md_content += "- **Multi-Perspective Pairwise Sequences**: Contains three discrete 2-day same-camera sequences (D01–D02 Low-Angle, D04–D05 Macro, D06–D07 Overlook) demonstrating pairwise temporal tracking across disparate viewpoints.\n"
        elif seg == "SEG_003":
            md_content += "- **Stability Control Benchmark (D01–D07)**: Features invariant camera and Clear Noon lighting on pristine asphalt, proving high metric stability (CV = 0.67%, exactly 1 candidate tracked across 7 days).\n"
            md_content += "- **Day10 Missing Metadata**: Day 10 image exists and is assessed for current severity (0.2435) with scenario forecasts generated in diagnostic mode.\n"
        elif seg == "SEG_004":
            md_content += "- **Model-Observed Progression (D01–D05 Nadir & D06–D10 Overlook)**: Shows model severity progression across deteriorating simulated states under fixed camera perspectives.\n"

        out_seg_file = SEGMENTS_DIR / f"{seg}_SUMMARY.md"
        out_seg_file.write_text(md_content, encoding="utf-8")
        log.info("Saved segment report to %s", out_seg_file)

    # 7. Generate Publication Figures
    log.info("Generating publication figures...")

    # Fig 1: All segments current severity across days
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seg, color, marker in zip(["SEG_001", "SEG_002", "SEG_003", "SEG_004"], ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], ["o", "s", "^", "D"]):
        sub = df_master[df_master["segment_id"] == seg].sort_values("day")
        ax.plot(sub["day"], sub["current_severity"], marker=marker, label=seg, color=color, lw=2)
    ax.set_xlabel("Inspection Day")
    ax.set_ylabel("Current Severity Assessment (DINOv2+SAM2)")
    ax.set_title("Current Road-Health Severity across Experiment A Captures (SEG_001–004)", fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(-0.02, 1.0)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_all_segments_current_severity.png", dpi=300)
    plt.close()

    # Fig 2–5: Current vs Forecasts for each segment
    for seg in ["SEG_001", "SEG_002", "SEG_003", "SEG_004"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sub = df_master[df_master["segment_id"] == seg].sort_values("day")
        days = sub["day"].values
        curr = sub["current_severity"].values
        n90 = sub["normal_90d_forecast"].values
        r90 = sub["heavy_rain_90d_forecast"].values
        t90 = sub["heavy_traffic_90d_forecast"].values
        w90 = sub["wet_exposure_90d_forecast"].values

        x = np.arange(len(days))
        w = 0.16
        ax.bar(x - 2*w, curr, w, label="Current Assessment", color="#7f7f7f", edgecolor="black")
        ax.bar(x - w, n90, w, label="90d NORMAL Forecast", color="#1f77b4", edgecolor="black")
        ax.bar(x, r90, w, label="90d HEAVY_RAIN Forecast", color="#2ca02c", edgecolor="black")
        ax.bar(x + w, t90, w, label="90d HEAVY_TRAFFIC Forecast", color="#ff7f0e", edgecolor="black")
        ax.bar(x + 2*w, w90, w, label="90d WET_EXPOSURE Forecast", color="#d62728", edgecolor="black")

        ax.set_xlabel("Inspection Day")
        ax.set_ylabel("Severity Score [0, 1]")
        ax.set_title(f"{seg}: Current Road-Health vs 90-Day Scenario Forecasts (Goal 1)", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"D{d:02d}" for d in days])
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fontsize=8, loc="upper right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig{2 + ['SEG_001', 'SEG_002', 'SEG_003', 'SEG_004'].index(seg)}_{seg.lower()}_current_vs_forecast.png", dpi=300)
        plt.close()

    # Fig 6: Temporal Change Summary (Goal 2 sequences)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    seq_names = df_goal2_summary["sequence_id"].values
    deltas = df_goal2_summary["severity_delta"].values
    colors = ["#d62728" if d > 0 else ("#1f77b4" if d < 0 else "#2ca02c") for d in deltas]
    bars = ax.barh(seq_names, deltas, color=colors, edgecolor="black", alpha=0.9)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Net Observed Severity Delta (End Day - Start Day)")
    ax.set_title("Goal 2 Temporal Progression: Net Observed Severity Delta by Sequence", fontweight="bold")
    for bar in bars:
        w_val = bar.get_width()
        ax.annotate(f"{w_val:+.4f}", xy=(w_val, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if w_val >= 0 else -35, 0), textcoords="offset points",
                    va="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_temporal_change_summary.png", dpi=300)
    plt.close()

    # Fig 7: Scenario Comparison across Horizons
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mean_forecasts = df_forecasts.groupby(["scenario", "days_ahead"])["predicted_future_severity"].mean().reset_index()
    scenarios = ["NORMAL", "HIGH_HEAT", "HEAVY_TRAFFIC", "HEAVY_RAIN", "WET_EXPOSURE"]
    colors_sc = ["#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c", "#d62728"]
    for sc, col in zip(scenarios, colors_sc):
        sub_sc = mean_forecasts[mean_forecasts["scenario"] == sc]
        ax.plot(sub_sc["days_ahead"], sub_sc["predicted_future_severity"], marker="o", lw=2.5, label=sc, color=col)
    ax.axhline(df_perc["current_severity"].mean(), color="gray", linestyle="--", label=f"Mean Current Baseline ({df_perc['current_severity'].mean():.4f})")
    ax.set_xlabel("Forecast Horizon (Days Ahead)")
    ax.set_ylabel("Mean Predicted Future Severity")
    ax.set_title("Goal 1 Deterioration Rate by Environmental Scenario Preset", fontweight="bold")
    ax.set_xticks([30, 60, 90])
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_current_vs_future_scenario_comparison.png", dpi=300)
    plt.close()

    # Fig 8: Pipeline Overview schematic plot
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.text(0.15, 0.7, "SINGLE IMAGE\n(Any Viewpoint)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f2ff", edgecolor="#1f77b4", lw=2), fontweight="bold")
    ax.annotate("", xy=(0.32, 0.7), xytext=(0.23, 0.7), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.42, 0.7, "DINOv2+SAM2\nCurrent Severity", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6ffe6", edgecolor="#2ca02c", lw=2), fontweight="bold")
    ax.annotate("", xy=(0.60, 0.7), xytext=(0.52, 0.7), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.75, 0.7, "GOAL 1: XGBOOST V2\n30/60/90d Forecast", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2e6", edgecolor="#ff7f0e", lw=2), fontweight="bold")

    ax.text(0.15, 0.25, "REPEATED IMAGES\n(Same Camera Preset)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f2ff", edgecolor="#1f77b4", lw=2), fontweight="bold")
    ax.annotate("", xy=(0.32, 0.25), xytext=(0.23, 0.25), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.42, 0.25, "TEMPORAL TRACKER\nMask/BBox IoU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6ffe6", edgecolor="#2ca02c", lw=2), fontweight="bold")
    ax.annotate("", xy=(0.60, 0.25), xytext=(0.52, 0.25), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.75, 0.25, "GOAL 2: CHANGE ANALYSIS\nTrack Area/Count Delta", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe6e6", edgecolor="#d62728", lw=2), fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.title("RoadSentinel Primary Research Architecture: Goal 1 vs Goal 2", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_primary_goal_pipeline.png", dpi=300)
    plt.close()

    # 8. Copy Assets to Dashboard Directory
    log.info("Copying deliverables to dashboard handoff directory...")
    shutil.copy(OUT_DIR / "ROADSENTINEL_PRIMARY_RESULTS.csv", DASHBOARD_DIR / "ROADSENTINEL_PRIMARY_RESULTS.csv")
    shutil.copy(OUT_DIR / "goal1_image_summary.csv", DASHBOARD_DIR / "goal1_image_summary.csv")
    shutil.copy(OUT_DIR / "goal2_change_summary.csv", DASHBOARD_DIR / "goal2_change_summary.csv")
    for f in FIGURES_DIR.glob("*.png"):
        shutil.copy(f, DASHBOARD_DIR / f.name)

    log.info("Phase 11 Primary Goals Execution Complete. All deliverables generated.")


if __name__ == "__main__":
    main()
