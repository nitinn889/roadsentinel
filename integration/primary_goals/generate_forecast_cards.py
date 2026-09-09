#!/usr/bin/env python3
"""generate_forecast_cards.py
----------------------------
Generates visual forecast cards for representative images from each segment (SEG_001 to SEG_004).
Shows original image, current road-health assessment, and 90-day scenario forecasts.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MASTER_CSV = WORKSPACE_ROOT / "integration" / "primary_goals" / "ROADSENTINEL_PRIMARY_RESULTS.csv"
OUT_FIGURES_DIR = WORKSPACE_ROOT / "integration" / "primary_goals" / "figures"
DASHBOARD_DIR = WORKSPACE_ROOT / "integration" / "dashboard_assets" / "primary_goals"

# Representative images to showcase
REP_IMAGES = {
    "SEG_001": {"day": 6, "desc": "Day 06 (Overlook, Peak Environmental Severity)"},
    "SEG_002": {"day": 4, "desc": "Day 04 (Macro Pothole View)"},
    "SEG_003": {"day": 1, "desc": "Day 01 (Overlook, Stability Baseline)"},
    "SEG_004": {"day": 5, "desc": "Day 05 (Nadir Drone, Severe Breakdown)"},
}


def main() -> None:
    df_master = pd.read_csv(MASTER_CSV)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (seg, info) in enumerate(REP_IMAGES.items()):
        ax = axes[idx]
        d_num = info["day"]
        row = df_master[(df_master["segment_id"] == seg) & (df_master["day"] == d_num)].iloc[0]
        
        # Load image
        img_path = WORKSPACE_ROOT / "env" / "output" / "temporal_segments" / seg / f"day_{d_num:02d}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
        ax.imshow(img_rgb)
        ax.axis("off")
        
        # Format text card
        card_text = (
            f"=== {seg} {info['desc']} ===\n"
            f"• Camera: {row['camera_preset']}\n"
            f"• Health State: {row['road_health_state']}\n"
            f"• CURRENT SEVERITY: {row['current_severity']:.4f}\n"
            f"• Defect Count: {row['defect_count']} | Area Ratio: {row['defect_area_ratio']:.6f}\n"
            f"----------------------------------------\n"
            f"90-DAY SCENARIO FORECASTS (XGBoost V2):\n"
            f"  • NORMAL:         {row['normal_90d_forecast']:.4f} ({row['normal_90d_forecast'] - row['current_severity']:+.4f})\n"
            f"  • HIGH HEAT:      {row['high_heat_90d_forecast']:.4f} ({row['high_heat_90d_forecast'] - row['current_severity']:+.4f})\n"
            f"  • HEAVY TRAFFIC:  {row['heavy_traffic_90d_forecast']:.4f} ({row['heavy_traffic_90d_forecast'] - row['current_severity']:+.4f})\n"
            f"  • HEAVY RAIN:     {row['heavy_rain_90d_forecast']:.4f} ({row['heavy_rain_90d_forecast'] - row['current_severity']:+.4f})\n"
            f"  • WET EXPOSURE:   {row['wet_exposure_90d_forecast']:.4f} ({row['wet_exposure_90d_forecast'] - row['current_severity']:+.4f})"
        )
        
        ax.text(
            0.02, 0.04, card_text, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", fontfamily="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="black", alpha=0.82, edgecolor="#00b4d8", lw=2)
        )

    plt.suptitle("RoadSentinel Primary Goal 1: Representative Segment Forecast Cards (90-Day Scenarios)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_fig = OUT_FIGURES_DIR / "fig9_representative_forecast_cards.png"
    plt.savefig(out_fig, dpi=300)
    plt.savefig(DASHBOARD_DIR / "fig9_representative_forecast_cards.png", dpi=300)
    plt.close()
    print(f"Saved representative forecast cards to {out_fig}")


if __name__ == "__main__":
    main()
