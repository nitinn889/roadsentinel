"""Generate Publication-Quality Figures for RoadSentinel Phase 13 Decision Engine.

Generates:
- fig1_decision_architecture.png: End-to-end evidence-based decision architecture.
- fig2_experiment_a_decision_distribution.png: Decision distribution across 40 physical captures.
- fig3_segment_decision_timelines.png: Multi-segment timeline swimlanes (SEG_001 to SEG_004).
- fig4_severity_temporal_forecast_example.png: Multi-stream evidence comparison (SEG_004 vs SEG_001).
- fig5_reliability_vs_decision.png: Current severity vs perception reliability decision mapping.
- fig6_cross_domain_routing.png: Zero-shot cross-domain protection and quarantine efficacy.
- fig7_scenario_priority_effect.png: 90-day scenario forecast deltas across environmental conditions.
- fig8_decision_case_studies.png: Qualitative case study matrix across operational scenarios.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_phase13_figures")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = WORKSPACE_ROOT / "decision_engine/tables"
DECISIONS_CSV_PATH = WORKSPACE_ROOT / "decision_engine/ROAD_HEALTH_DECISIONS.csv"
FIGURES_DIR = WORKSPACE_ROOT / "decision_engine/figures"
DASHBOARD_ASSETS_DIR = WORKSPACE_ROOT / "integration/dashboard_assets/decision_engine"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300

TIER_COLORS = {
    "AUTOMATED_ACCEPT": "#16a34a",     # Green
    "MONITOR": "#2563eb",              # Blue
    "REINSPECT": "#f59e0b",            # Amber
    "PRIORITY_REVIEW": "#dc2626",      # Red
    "DOMAIN_ESCALATION": "#7c3aed",    # Purple
}


import matplotlib.patches as mpatches

def plot_fig1_architecture():
    """Fig 1: End-to-end evidence-based decision architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # Draw Architecture Pipeline Boxes
    boxes = [
        {"title": "1. VISUAL INPUT", "text": "Road Image Capture\n+ Ingest Metadata Sidecar", "x": 0.05, "y": 0.5, "w": 0.15, "h": 0.35, "color": "#e2e8f0", "edge": "#64748b"},
        {"title": "2. DOMAIN GATE", "text": "DINOv2 Foundation Embedding\nkNN Distance vs Training ODD\n(Threshold p99 = 0.4491)", "x": 0.24, "y": 0.5, "w": 0.16, "h": 0.35, "color": "#ede9fe", "edge": "#7c3aed"},
        {"title": "3. PERCEPTION CORE", "text": "YOLOv8n Damage Detection\n+ DINOv2+SAM2 Severity\n(Current Condition Features)", "x": 0.44, "y": 0.5, "w": 0.16, "h": 0.35, "color": "#dbeafe", "edge": "#2563eb"},
        {"title": "4. RELIABILITY FILTER", "text": "Model B Confidence Calibration\nHIGH (≥0.85) / MED / LOW (<0.60)\n(Quarantines Uncertainty)", "x": 0.64, "y": 0.5, "w": 0.16, "h": 0.35, "color": "#fef3c7", "edge": "#f59e0b"},
        {"title": "5. DECISION LAYER", "text": "Hierarchical Deterministic Policy\n+ Temporal Trends & Forecasts\n→ Action Tiers (1 to 5)", "x": 0.84, "y": 0.5, "w": 0.14, "h": 0.35, "color": "#fee2e2", "edge": "#dc2626"},
    ]

    for b in boxes:
        rect = mpatches.FancyBboxPatch((b["x"], b["y"] - b["h"]/2), b["w"], b["h"], facecolor=b["color"],
                                       edgecolor=b["edge"], linewidth=2, transform=ax.transAxes, zorder=2,
                                       boxstyle="round,pad=0.02")
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]/2 - 0.06, b["title"], ha="center", va="top",
                fontsize=11, fontweight="bold", color=b["edge"], transform=ax.transAxes, zorder=3)
        ax.text(b["x"] + b["w"]/2, b["y"] - 0.03, b["text"], ha="center", va="center",
                fontsize=9, color="#1e293b", transform=ax.transAxes, zorder=3)

    # Connecting Arrows
    for i in range(len(boxes) - 1):
        x_start = boxes[i]["x"] + boxes[i]["w"]
        x_end = boxes[i+1]["x"]
        ax.annotate("", xy=(x_end, 0.5), xytext=(x_start, 0.5), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=2.5), zorder=4)

    # Action Tiers Banner
    tiers_text = "Action Tiers:  [1] DOMAIN_ESCALATION    [2] REINSPECT    [3] PRIORITY_REVIEW    [4] MONITOR    [5] AUTOMATED_ACCEPT"
    ax.text(0.5, 0.15, tiers_text, ha="center", va="center", fontsize=11, fontweight="bold",
            color="#0f172a", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f1f5f9", edgecolor="#94a3b8", lw=1.5))

    plt.suptitle("Fig 1: RoadSentinel Multi-Stream Reliability-Aware Decision Architecture", fontsize=14, fontweight="bold", y=0.92)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_decision_architecture.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 1")


def plot_fig2_decision_distribution(df_dec: pd.DataFrame):
    """Fig 2: Decision distribution across 40 physical captures."""
    fig, ax = plt.subplots(figsize=(9, 5))

    counts = df_dec["decision"].value_counts()
    tiers = ["MONITOR", "REINSPECT", "PRIORITY_REVIEW", "AUTOMATED_ACCEPT", "DOMAIN_ESCALATION"]
    vals = [counts.get(t, 0) for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    bars = ax.bar(tiers, vals, color=colors, alpha=0.85, width=0.55, edgecolor="#1e293b", linewidth=1.2)
    ax.set_title("Fig 2: Experiment A Prototype Decision Tier Distribution (N=40)", fontweight="bold", pad=12)
    ax.set_ylabel("Number of Captures")
    ax.set_ylim(0, max(vals) + 5)

    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.6, f"{val}\n({val/40*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_experiment_a_decision_distribution.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 2")


def plot_fig3_segment_timelines(df_dec: pd.DataFrame):
    """Fig 3: Timeline swimlanes across SEG_001 to SEG_004."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)

    tier_map = {"AUTOMATED_ACCEPT": 1, "MONITOR": 2, "REINSPECT": 3, "PRIORITY_REVIEW": 4, "DOMAIN_ESCALATION": 5}
    segments = ["SEG_001", "SEG_002", "SEG_003", "SEG_004"]
    titles = [
        "SEG_001: Environmental Modulation Sequence (Overcast inflation & sunset glare suppression)",
        "SEG_002: Multi-Viewpoint Mixed Capture Sequence (Macro views & highway curve overlooks)",
        "SEG_003: Pristine Grade A Invariant Control (D01-07) + SAM2 Drone Pair (D08-09) + Missing Meta (D10)",
        "SEG_004: Monotonic Deterioration Progression (D01-05) + Heavy Rain Escalation (D06-10)",
    ]

    days = np.arange(1, 11)

    for idx, (seg_id, title) in enumerate(zip(segments, titles)):
        ax = axes[idx]
        sub = df_dec[df_dec["segment"] == seg_id].sort_values(by="day")
        
        dec_names = sub["decision"].values
        tier_vals = [tier_map[d] for d in dec_names]
        colors = [TIER_COLORS[d] for d in dec_names]

        ax.scatter(days, tier_vals, c=colors, s=180, edgecolors="#1e293b", linewidth=1.5, zorder=3)
        ax.plot(days, tier_vals, color="#94a3b8", linestyle="--", alpha=0.7, zorder=2)

        for d, t_val, dec in zip(days, tier_vals, dec_names):
            short_lbl = {"MONITOR": "MON", "REINSPECT": "REINSP", "PRIORITY_REVIEW": "PRIO", "AUTOMATED_ACCEPT": "ACCEPT"}.get(dec, dec)
            ax.text(d, t_val + 0.35, short_lbl, ha="center", va="bottom", fontsize=8, fontweight="bold", color=TIER_COLORS[dec])

        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["Accept", "Monitor", "Reinspect", "Priority"], fontsize=8)
        ax.set_ylim(0.5, 4.8)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xticks(days)
    axes[-1].set_xticklabels([f"Day {d:02d}" for d in days], fontsize=10)
    axes[-1].set_xlabel("Simulated Inspection Day")

    plt.suptitle("Fig 3: Segment Decision Timelines across Days 01–10", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_segment_decision_timelines.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 3")


def plot_fig4_evidence_streams(df_dec: pd.DataFrame):
    """Fig 4: Multi-stream evidence comparison (SEG_004 vs SEG_001)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: SEG_004 Clean Monotonic Progression
    ax1 = axes[0]
    seg4 = df_dec[df_dec["segment"] == "SEG_004"].sort_values(by="day")
    days = seg4["day"].values
    sev4 = seg4["current_severity"].values
    wet4 = seg4["wet_exposure_90d"].values
    norm4 = seg4["normal_90d"].values

    ax1.plot(days, sev4, "o-", color="#dc2626", linewidth=2.5, label="Current Model Severity", markersize=6)
    ax1.plot(days, wet4, "^--", color="#2563eb", linewidth=2, label="Wet Exposure 90d Forecast", markersize=6)
    ax1.plot(days, norm4, "s:", color="#16a34a", linewidth=1.5, label="Normal 90d Forecast", markersize=5)
    ax1.axvspan(1, 5, color="#dcfce7", alpha=0.5, label="Clean Monotonic Phase (D01-D05)")
    ax1.axvspan(6, 10, color="#fee2e2", alpha=0.5, label="Weather Shift Phase (D06-D10)")
    ax1.set_title("Panel A: SEG_004 Severity & Forecast Progression", fontweight="bold")
    ax1.set_xlabel("Inspection Day")
    ax1.set_ylabel("Severity Metric (0.0 to 1.0)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    # Panel B: SEG_001 Environmental Modulation
    ax2 = axes[1]
    seg1 = df_dec[df_dec["segment"] == "SEG_001"].sort_values(by="day")
    sev1 = seg1["current_severity"].values
    wet1 = seg1["wet_exposure_90d"].values

    ax2.plot(days, sev1, "o-", color="#dc2626", linewidth=2.5, label="Current Model Severity", markersize=6)
    ax2.plot(days, wet1, "^--", color="#2563eb", linewidth=2, label="Wet Exposure 90d Forecast", markersize=6)
    ax2.axvspan(6, 8, color="#fef3c7", alpha=0.6, label="Overcast Contrast Inflation (D06-D08)")
    ax2.axvspan(9, 10, color="#f1f5f9", alpha=0.6, label="Sunset Shadow Over-Suppression (D09-D10)")
    ax2.set_title("Panel B: SEG_001 Environmental Confound Modulation", fontweight="bold")
    ax2.set_xlabel("Inspection Day")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left", frameon=True, fontsize=8)

    plt.suptitle("Fig 4: Model Assessment vs Future Forecast across Road Sequences", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_severity_temporal_forecast_example.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 4")


def plot_fig5_severity_vs_reliability(df_dec: pd.DataFrame):
    """Fig 5: Current severity vs perception reliability decision mapping."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for dec in ["MONITOR", "REINSPECT", "PRIORITY_REVIEW", "AUTOMATED_ACCEPT"]:
        sub = df_dec[df_dec["decision"] == dec]
        ax.scatter(sub["current_severity"], sub["reliability_score"],
                   c=TIER_COLORS[dec], label=f"{dec} (N={len(sub)})", s=90, edgecolors="#1e293b", linewidth=1.2, alpha=0.9)

    ax.axhline(0.85, color="#16a34a", linestyle="--", alpha=0.7, label="HIGH Reliability Threshold (0.85)")
    ax.axhline(0.60, color="#dc2626", linestyle="--", alpha=0.7, label="LOW Reliability Threshold (0.60)")
    ax.axvline(0.50, color="#94a3b8", linestyle=":", alpha=0.7, label="High Severity Threshold (0.50)")
    ax.axvline(0.20, color="#94a3b8", linestyle=":", alpha=0.7, label="Moderate Severity Threshold (0.20)")

    ax.set_title("Fig 5: Decoupled Current Severity vs Perception Reliability", fontweight="bold", pad=12)
    ax.set_xlabel("Current Model Severity Score (0.0 to 1.0)")
    ax.set_ylabel("Perception Reliability Score (0.0 to 1.0)")
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_reliability_vs_decision.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 5")


def plot_fig6_cross_domain_quarantine():
    """Fig 6: Cross-domain India quarantine routing sankey/flow chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    systems = ["1. Standalone YOLO", "2. Severity Only", "3. Confidence Only", "4. Full RoadSentinel Gate"]
    unsafe_rates = [97.67, 97.67, 85.71, 0.0]
    quarantine_eff = [0.0, 0.0, 91.81, 100.0]

    x = np.arange(len(systems))
    width = 0.35

    ax.bar(x - width/2, unsafe_rates, width, label="Unsafe Failure Rate Among Accepted (%)", color="#dc2626", alpha=0.85)
    ax.bar(x + width/2, quarantine_eff, width, label="Failure Quarantine Efficacy (%)", color="#16a34a", alpha=0.85)

    for i in range(len(systems)):
        ax.text(x[i] - width/2, unsafe_rates[i] + 1.5, f"{unsafe_rates[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i] + width/2, quarantine_eff[i] + 1.5, f"{quarantine_eff[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Fig 6: Cross-Domain Protection & Failure Quarantine on RDD2022 India (N=300)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)")
    ax.legend(loc="upper center", frameon=True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_cross_domain_routing.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 6")


def plot_fig7_scenario_sensitivity(df_dec: pd.DataFrame):
    """Fig 7: 90-day scenario forecast deltas across environmental conditions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = ["NORMAL", "HIGH_HEAT", "HEAVY_TRAFFIC", "HEAVY_RAIN", "WET_EXPOSURE"]
    delta_cols = [
        df_dec["normal_90d"] - df_dec["current_severity"],
        df_dec["high_heat_90d"] - df_dec["current_severity"],
        df_dec["heavy_traffic_90d"] - df_dec["current_severity"],
        df_dec["heavy_rain_90d"] - df_dec["current_severity"],
        df_dec["wet_exposure_90d"] - df_dec["current_severity"],
    ]

    mean_deltas = [float(np.mean(d)) for d in delta_cols]
    std_deltas = [float(np.std(d)) for d in delta_cols]
    colors = ["#64748b", "#f59e0b", "#d97706", "#0284c7", "#2563eb"]

    bars = ax.bar(scenarios, mean_deltas, yerr=std_deltas, color=colors, alpha=0.85, capsize=4, width=0.55, edgecolor="#1e293b")
    ax.set_title("Fig 7: Scenario-Conditioned 90-Day Deterioration Sensitivity (N=40 Captures)", fontweight="bold", pad=12)
    ax.set_ylabel("Mean Predicted 90-Day Severity Delta")
    ax.set_ylim(0, max(mean_deltas) + 0.08)

    for bar, val in zip(bars, mean_deltas):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"+{val:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_scenario_priority_effect.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 7")


def plot_fig8_case_studies():
    """Fig 8: Qualitative case study matrix across operational scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    cases = [
        {"title": "Case 1: Pristine Road Control", "sub": "SEG_003 Day 01 (Severity = 0.2472)", "tier": "MONITOR", "col": "#2563eb",
         "text": "• Grade A road shoulder\n• Perception Reliability: HIGH (0.88)\n• Temporal Trend: Stable\n• Decision: MONITOR"},
        {"title": "Case 2: Confident Severe Distress", "sub": "SEG_004 Day 05 (Severity = 0.7302)", "tier": "PRIORITY_REVIEW", "col": "#dc2626",
         "text": "• Severe breakdown state\n• Persistent defect tracks confirmed\n• Wet Forecast: +0.0312\n• Decision: PRIORITY_REVIEW"},
        {"title": "Case 3: Rapid Temporal Growth", "sub": "SEG_004 Day 04 (Severity = 0.3150)", "tier": "PRIORITY_REVIEW", "col": "#dc2626",
         "text": "• Sudden defect onset (+0.3150)\n• High Perception Reliability (0.88)\n• Escalates from Monitor\n• Decision: PRIORITY_REVIEW"},
        {"title": "Case 4: Environmental Confound", "sub": "SEG_001 Day 06 (Severity = 0.7141)", "tier": "PRIORITY_REVIEW", "col": "#dc2626",
         "text": "• Overcast cloud contrast inflation\n• High apparent model distress\n• Flags lighting shift limitation\n• Decision: PRIORITY_REVIEW"},
        {"title": "Case 5: Cross-Domain Shift", "sub": "RDD2022 India Frame 05", "tier": "DOMAIN_ESCALATION", "col": "#7c3aed",
         "text": "• Raw DINO distance = 0.8530 > 0.4491\n• Zero-shot dashcam shift\n• Auto-acceptance blocked\n• Decision: DOMAIN_ESCALATION"},
        {"title": "Case 6: Missing Metadata Diagnostic", "sub": "SEG_003 Day 10 (Diagnostic Frame)", "tier": "MONITOR", "col": "#2563eb",
         "text": "• Sidecar camera preset missing\n• Fallback physics applied\n• Documents metadata limitation\n• Decision: MONITOR"},
    ]

    for idx, c in enumerate(cases):
        r, col = idx // 3, idx % 3
        ax = axes[r, col]
        ax.set_title(c["title"], fontweight="bold", color=c["col"], fontsize=11)
        ax.text(0.5, 0.8, c["sub"], ha="center", fontsize=9, fontweight="bold", color="#475569")
        ax.text(0.5, 0.45, c["text"], ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor=c["col"], lw=1.5))
        ax.text(0.5, 0.12, f"TIER: {c['tier']}", ha="center", fontsize=10, fontweight="bold",
                color="white", bbox=dict(boxstyle="square,pad=0.3", facecolor=c["col"], edgecolor="none"))
        ax.axis("off")

    plt.suptitle("Fig 8: RoadSentinel Qualitative Decision Engine Case Matrix", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_decision_case_studies.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 8")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    df_dec = pd.read_csv(DECISIONS_CSV_PATH)
    log.info("Loaded decision records for figure generation (N=%d)", len(df_dec))

    plot_fig1_architecture()
    plot_fig2_decision_distribution(df_dec)
    plot_fig3_segment_timelines(df_dec)
    plot_fig4_evidence_streams(df_dec)
    plot_fig5_severity_vs_reliability(df_dec)
    plot_fig6_cross_domain_quarantine()
    plot_fig7_scenario_sensitivity(df_dec)
    plot_fig8_case_studies()

    log.info("All 8 Phase 13 figures successfully generated.")


if __name__ == "__main__":
    main()
