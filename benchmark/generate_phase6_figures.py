#!/usr/bin/env python3
"""Generate paper-ready publication figures for RoadSentinel Phase 6.

Generates 7 focused, high-contrast, publication-quality figures:
1. fig1_precision_recall_f1.png (Binary P/R/F1 comparison)
2. fig2_latency_throughput_log.png (Latency & throughput with log scale)
3. fig3_matched_bbox_iou.png (Mean and median matched bbox IoU)
4. fig4_yolo_class_wise_metrics.png (Class-wise P, R, F1 for YOLO semantic classes)
5. fig5_dino_failure_mode_breakdown.png (Detection vs Localization misses, FP size distribution)
6. fig6_iou_sensitivity_comparison.png (IoU 0.50 vs 0.25 sensitivity for both models)
7. fig7_qualitative_panel_montage.png (Montage of key qualitative panels)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmark/final_comparison/figures"
PANELS_DIR = ROOT / "benchmark/final_comparison/panels"
SUMMARY_JSON = ROOT / "benchmark/final_comparison/benchmark_summary.json"


def set_plot_style():
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def generate_all_figures():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    # --- Figure 1: Precision, Recall, F1 Comparison ---
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
    metrics = ["Precision", "Recall", "F1 Score"]
    yolo_vals = [0.6568, 0.7736, 0.7104]
    dino_vals = [0.0221, 0.0337, 0.0267]
    x = np.arange(len(metrics))
    width = 0.32

    rects1 = ax.bar(x - width / 2, yolo_vals, width, label="Supervised YOLOv8n", color="#1f77b4", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width / 2, dino_vals, width, label="Zero-Shot DINOv2 + SAM2", color="#e377c2", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.set_title("Common Validation Benchmark (IoU ≥ 0.50, N=480 Images)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.90)
    ax.legend(loc="upper right")

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    f1_path = OUT_DIR / "fig1_precision_recall_f1.png"
    plt.savefig(f1_path)
    plt.close()
    print("Saved:", f1_path)

    # --- Figure 2: Latency & Throughput (Log Scale) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.5), dpi=300)
    models = ["YOLOv8n", "DINOv2+SAM2"]
    latencies = [3.62, 263.15]
    fps_vals = [276.33, 3.80]
    colors = ["#1f77b4", "#e377c2"]

    # Latency
    bars1 = ax1.bar(models, latencies, color=colors, edgecolor="black", width=0.45)
    ax1.set_ylabel("Inference Latency (ms) [Log Scale]")
    ax1.set_yscale("log")
    ax1.set_ylim(1, 1000)
    ax1.set_title("Per-Image GPU Latency (RTX 5060)")
    ax1.axhline(33.3, color="gray", linestyle=":", alpha=0.7, label="Real-time 30 FPS (33 ms)")
    ax1.legend(loc="upper left")
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f} ms", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Throughput
    bars2 = ax2.bar(models, fps_vals, color=colors, edgecolor="black", width=0.45)
    ax2.set_ylabel("Throughput (Frames Per Second)")
    ax2.set_title("Processing Throughput (FPS)")
    ax2.set_ylim(0, 320)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f} FPS", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Computational Efficiency Comparison (72.7× Latency Ratio)", y=1.02)
    plt.tight_layout()
    f2_path = OUT_DIR / "fig2_latency_throughput_log.png"
    plt.savefig(f2_path)
    plt.close()
    print("Saved:", f2_path)

    # --- Figure 3: Matched Bounding-Box IoU ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    cats = ["Mean Matched IoU", "Median Matched IoU"]
    y_ious = [0.8007, 0.8154]
    d_ious = [0.6533, 0.6480]
    x = np.arange(len(cats))
    width = 0.32

    r1 = ax.bar(x - width / 2, y_ious, width, label="YOLOv8n (Native Boxes)", color="#1f77b4", edgecolor="black")
    r2 = ax.bar(x + width / 2, d_ious, width, label="DINOv2+SAM2 (Mask-Derived Boxes)", color="#e377c2", edgecolor="black")

    ax.set_ylabel("Intersection over Union (IoU)")
    ax.set_title("Geometric Box Alignment on True Positives (IoU ≥ 0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.50, color="red", linestyle="--", alpha=0.6, label="Evaluation Threshold (0.50)")
    ax.legend(loc="lower left")

    for r in r1 + r2:
        h = r.get_height()
        ax.annotate(f"{h:.4f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    f3_path = OUT_DIR / "fig3_matched_bbox_iou.png"
    plt.savefig(f3_path)
    plt.close()
    print("Saved:", f3_path)

    # --- Figure 4: YOLO Native Class-Wise Performance ---
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    classes = ["D00 (Long.)\nN=266", "D10 (Trans.)\nN=256", "D20 (Alligator)\nN=58", "D40 (Pothole)*\nN=15 (Low)", "Repair\nN=147"]
    precisions = [0.5926, 0.7057, 0.4600, 0.7000, 0.6575]
    recalls = [0.7820, 0.7773, 0.3966, 0.4667, 0.8095]
    f1_scores = [0.6742, 0.7398, 0.4259, 0.5600, 0.7256]

    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, precisions, width, label="Precision", color="#2ca02c", edgecolor="black")
    ax.bar(x, recalls, width, label="Recall", color="#ff7f0e", edgecolor="black")
    ax.bar(x + width, f1_scores, width, label="F1 Score", color="#1f77b4", edgecolor="black")

    ax.set_ylabel("Metric Score")
    ax.set_title("Supervised YOLOv8n Class-Wise Performance (IoU ≥ 0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")

    plt.tight_layout()
    f4_path = OUT_DIR / "fig4_yolo_class_wise_metrics.png"
    plt.savefig(f4_path)
    plt.close()
    print("Saved:", f4_path)

    # --- Figure 5: DINO/SAM FP & FN Failure Breakdown ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), dpi=300)

    # Left: False Negatives Breakdown (717 missed GT boxes)
    fn_labels = ["Detection Failures\n(No overlap / Diluted)\n389 (54.3%)", "Localization Failures\n(Partial overlap IoU<0.50)\n328 (45.7%)"]
    fn_sizes = [389, 328]
    fn_colors = ["#d62728", "#ff9896"]
    ax1.pie(fn_sizes, labels=fn_labels, colors=fn_colors, autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "black", "linewidth": 0.8})
    ax1.set_title("Missed Defect Breakdown (FN = 717)")

    # Right: False Positives by Region Area (1104 FP boxes)
    fp_categories = ["Small Noise\n(<500 px²)\n1.4%", "Texture Patches\n(500–5000 px²)\n50.7%", "Broad Anomalies\n(≥5000 px²)\n47.9%"]
    fp_counts = [15, 560, 529]
    ax2.bar(fp_categories, fp_counts, color=["#aec7e8", "#9467bd", "#8c564b"], edgecolor="black", width=0.55)
    ax2.set_ylabel("Number of False Positive Boxes")
    ax2.set_title("False Alarm Area Distribution (FP = 1,104)")
    ax2.set_ylim(0, 650)
    for i, v in enumerate(fp_counts):
        ax2.annotate(str(v), xy=(i, v), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("DINOv2 + SAM2 Failure Mode Quantitative Characterization", y=1.02)
    plt.tight_layout()
    f5_path = OUT_DIR / "fig5_dino_failure_mode_breakdown.png"
    plt.savefig(f5_path)
    plt.close()
    print("Saved:", f5_path)

    # --- Figure 6: IoU 0.50 vs 0.25 Sensitivity ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.5), dpi=300)

    # TP comparison
    models = ["YOLOv8n", "DINOv2+SAM2"]
    x = np.arange(len(models))
    width = 0.35
    tp_50 = [574, 25]
    tp_25 = [610, 74]

    rects_tp1 = ax1.bar(x - width / 2, tp_50, width, label="Primary IoU ≥ 0.50", color="#1f77b4", edgecolor="black")
    rects_tp2 = ax1.bar(x + width / 2, tp_25, width, label="Secondary IoU ≥ 0.25", color="#2ca02c", edgecolor="black")
    ax1.set_ylabel("True Positive Boxes (Out of 742 GT)")
    ax1.set_title("True Positive Sensitivity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 750)
    ax1.legend(loc="upper right")
    for r in rects_tp1:
        ax1.annotate(str(r.get_height()), xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for r in rects_tp2:
        ax1.annotate(str(r.get_height()), xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # F1 comparison
    f1_50 = [0.7104, 0.0267]
    f1_25 = [0.7550, 0.0791]

    rects_f1 = ax2.bar(x - width / 2, f1_50, width, label="Primary IoU ≥ 0.50", color="#1f77b4", edgecolor="black")
    rects_f2 = ax2.bar(x + width / 2, f1_25, width, label="Secondary IoU ≥ 0.25", color="#2ca02c", edgecolor="black")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Score Sensitivity")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 0.90)
    ax2.legend(loc="upper right")
    for r in rects_f1:
        ax2.annotate(f"{r.get_height():.4f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for r in rects_f2:
        ax2.annotate(f"{r.get_height():.4f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Impact of Threshold Relaxation on Thin-Crack Localization", y=1.02)
    plt.tight_layout()
    f6_path = OUT_DIR / "fig6_iou_sensitivity_comparison.png"
    plt.savefig(f6_path)
    plt.close()
    print("Saved:", f6_path)

    # --- Figure 7: Montage of Selected Qualitative Panels ---
    panel_files = [
        PANELS_DIR / "A_BOTH_CORRECT_China_Drone_001063.jpg",
        PANELS_DIR / "B_YOLO_ONLY_SUCCESS_China_Drone_000010.jpg",
        PANELS_DIR / "C_DINO_SAM_ONLY_SUCCESS_China_Drone_002162.jpg",
        PANELS_DIR / "D_BOTH_FAIL_China_Drone_000033.jpg",
    ]
    loaded_imgs = [cv2.imread(str(p)) for p in panel_files if p.exists()]
    if len(loaded_imgs) == 4:
        # Resize all to same width if necessary
        target_w = 1200
        resized = []
        for im in loaded_imgs:
            h, w = im.shape[:2]
            scaled_h = int(h * target_w / w)
            resized.append(cv2.resize(im, (target_w, scaled_h)))
        montage = np.vstack(resized)
        f7_path = OUT_DIR / "fig7_qualitative_panel_montage.png"
        cv2.imwrite(str(f7_path), montage)
        print("Saved montage:", f7_path)


if __name__ == "__main__":
    generate_all_figures()
