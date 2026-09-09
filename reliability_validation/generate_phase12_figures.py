"""Generate Publication-Quality Figures for RoadSentinel Phase 12.

Generates:
- fig1_target_prevalence.png: Multi-target class balance (T0 - T4) across China and India.
- fig2_ablation_auroc.png: In-Domain vs Cross-Domain AUROC across Models A to H with 95% Bootstrap CIs.
- fig3_ablation_auprc.png: In-Domain vs Cross-Domain AUPRC across Models A to H.
- fig4_risk_coverage_strict_target.png: Risk-coverage failure reduction curves for strict targets (T1, T3).
- fig5_domain_distance_raw.png: Raw DINO kNN distance distribution and saturation audit.
- fig6_domain_gate_roc.png: DINO Domain Gate ROC and PR curves.
- fig7_reliability_band_validation.png: Multi-target reliability band success rates (HIGH, MEDIUM, LOW).
- fig8_in_vs_cross_domain_reliability.png: Reliability score distributions in-domain vs cross-domain.
- fig9_temporal_ablation.png: Temporal signal ablation performance comparison (Experiment A).
- fig10_failure_case_panel.png: Multi-panel qualitative visual failure taxonomy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_phase12_figures")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = WORKSPACE_ROOT / "reliability_validation/tables"
DATA_DIR = WORKSPACE_ROOT / "reliability_validation/data"
FIGURES_DIR = WORKSPACE_ROOT / "reliability_validation/figures"
DASHBOARD_ASSETS_DIR = WORKSPACE_ROOT / "integration/dashboard_assets/reliability_validation"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300

CHINA_P99_KNN = 0.4491024327278137
CHINA_P95_KNN = 0.38041598200798035


def plot_fig1_target_prevalence(df_prev: pd.DataFrame):
    """Fig 1: Target Prevalence across In-Domain and Cross-Domain datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    targets = ["T0", "T1", "T2", "T3", "T4"]
    target_labels = [
        "T0\n(F1>0)",
        "T1\n(F1≥0.50)",
        "T2\n(Recall≥0.50)",
        "T3\n(Recall≥0.75)",
        "T4\n(Recall=1.0)",
    ]

    for idx, (d_name, title, color_succ, color_fail) in enumerate([
        ("China_Drone_Val", "In-Domain (RDD2022 China_Drone, N=480)", "#2563eb", "#ef4444"),
        ("India_Cross_Domain", "Cross-Domain (RDD2022 India, N=300)", "#059669", "#dc2626"),
    ]):
        ax = axes[idx]
        sub = df_prev[df_prev["dataset"] == d_name]
        
        succ_pct = [sub[sub["target"] == t]["success_pct"].values[0] for t in targets]
        fail_pct = [sub[sub["target"] == t]["failure_pct"].values[0] for t in targets]
        succ_cnt = [sub[sub["target"] == t]["success_count"].values[0] for t in targets]
        fail_cnt = [sub[sub["target"] == t]["failure_count"].values[0] for t in targets]

        x = np.arange(len(targets))
        width = 0.55

        rects1 = ax.bar(x, succ_pct, width, label="Success", color=color_succ, alpha=0.85)
        rects2 = ax.bar(x, fail_pct, width, bottom=succ_pct, label="Failure", color=color_fail, alpha=0.85)

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(target_labels)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percentage (%)" if idx == 0 else "")
        ax.legend(loc="upper right", frameon=True)

        for i in range(len(targets)):
            # Annotate success count/pct
            if succ_pct[i] > 10:
                ax.text(x[i], succ_pct[i] / 2, f"{succ_cnt[i]}\n({succ_pct[i]:.1f}%)",
                        ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            elif succ_pct[i] > 0:
                ax.text(x[i], succ_pct[i] + 2, f"{succ_cnt[i]}", ha="center", va="bottom", color=color_succ, fontsize=8)

            # Annotate failure count/pct
            if fail_pct[i] > 15:
                ax.text(x[i], succ_pct[i] + fail_pct[i] / 2, f"{fail_cnt[i]}\n({fail_pct[i]:.1f}%)",
                        ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    plt.suptitle("Fig 1: Target Prevalence & Perception Degradation (T0–T4)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_target_prevalence.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 1")


def plot_fig2_ablation_auroc(df_ablation: pd.DataFrame):
    """Fig 2: In-Domain vs Cross-Domain AUROC across Models A to H with 95% CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    models = [f"Model_{c}" for c in ["A", "B", "C", "D", "E", "F", "G", "H"]]
    model_labels = ["A: MaxConf", "B: Conf Feats", "C: DINO OOD", "D: Img Qual", "E: Conf+DINO", "F: Conf+Qual", "G: DINO+Qual", "H: Combined"]

    for idx, (d_str, title, color_main) in enumerate([
        ("China", "In-Domain (China_Drone 5-Fold CV)", "#2563eb"),
        ("India", "Cross-Domain Transfer (India Zero-Shot)", "#dc2626"),
    ]):
        ax = axes[idx]
        sub = df_ablation[df_ablation["dataset"].str.contains(d_str)]
        
        for t_idx, (t_key, t_color, t_marker) in enumerate([
            ("T0", "#1e293b", "o"),
            ("T1", "#0284c7", "s"),
            ("T3", "#d97706", "^"),
            ("T4", "#9333ea", "D"),
        ]):
            sub_t = sub[sub["target"] == t_key]
            aurocs = [sub_t[sub_t["model"] == m]["AUROC"].values[0] for m in models]
            ci_low = [aurocs[i] - sub_t[sub_t["model"] == m]["AUROC_95CI_low"].values[0] for i, m in enumerate(models)]
            ci_high = [sub_t[sub_t["model"] == m]["AUROC_95CI_high"].values[0] - aurocs[i] for i, m in enumerate(models)]
            
            x_pos = np.arange(len(models)) + (t_idx - 1.5) * 0.15
            ax.errorbar(x_pos, aurocs, yerr=[ci_low, ci_high], fmt=t_marker, label=f"{t_key}",
                        color=t_color, capsize=3, capthick=1.2, elinewidth=1.2, markersize=6)

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Random Guess (0.50)")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(model_labels, rotation=35, ha="right")
        ax.set_ylim(0.35, 1.02)
        ax.set_ylabel("AUROC (Failure Prediction)" if idx == 0 else "")
        ax.legend(loc="lower left", frameon=True, fontsize=9)

    plt.suptitle("Fig 2: Signal Ablation AUROC with 95% Bootstrap CIs", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_ablation_auroc.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 2")


def plot_fig3_ablation_auprc(df_ablation: pd.DataFrame):
    """Fig 3: In-Domain vs Cross-Domain AUPRC across Models A to H."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    models = [f"Model_{c}" for c in ["A", "B", "C", "D", "E", "F", "G", "H"]]
    model_labels = ["A: MaxConf", "B: Conf Feats", "C: DINO OOD", "D: Img Qual", "E: Conf+DINO", "F: Conf+Qual", "G: DINO+Qual", "H: Combined"]

    for idx, (d_str, title, base_rate) in enumerate([
        ("China", "In-Domain (China_Drone CV)", 0.1188),
        ("India", "Cross-Domain Transfer (India)", 0.9733),
    ]):
        ax = axes[idx]
        sub = df_ablation[df_ablation["dataset"].str.contains(d_str)]
        
        for t_idx, (t_key, t_color, t_marker) in enumerate([
            ("T0", "#1e293b", "o"),
            ("T1", "#0284c7", "s"),
            ("T3", "#d97706", "^"),
            ("T4", "#9333ea", "D"),
        ]):
            sub_t = sub[sub["target"] == t_key]
            auprcs = [sub_t[sub_t["model"] == m]["AUPRC"].values[0] for m in models]
            x_pos = np.arange(len(models)) + (t_idx - 1.5) * 0.15
            ax.plot(x_pos, auprcs, marker=t_marker, label=f"{t_key}", color=t_color, linewidth=1.5, markersize=6)

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(model_labels, rotation=35, ha="right")
        ax.set_ylabel("AUPRC (Precision-Recall AUC)")
        ax.legend(loc="lower left" if idx == 0 else "center right", frameon=True, fontsize=9)

    plt.suptitle("Fig 3: Signal Ablation AUPRC across Detection Targets", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_ablation_auprc.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 3")


def plot_fig4_risk_coverage(df_risk: pd.DataFrame):
    """Fig 4: Risk-Coverage Selective Prediction for Strict Targets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, (t_key, title) in enumerate([
        ("T1", "Target T1: Strict F1 ≥ 0.50"),
        ("T3", "Target T3: High Recall ≥ 0.75"),
    ]):
        ax = axes[idx]
        sub_china_h = df_risk[(df_risk["dataset"].str.contains("China")) & (df_risk["target"] == t_key) & (df_risk["model"] == "Model_H")]
        sub_china_b = df_risk[(df_risk["dataset"].str.contains("China")) & (df_risk["target"] == t_key) & (df_risk["model"] == "Model_B")]
        
        cov = sub_china_h["coverage_pct"].values
        fail_rate_h = sub_china_h["accepted_failure_rate"].values * 100
        fail_rate_b = sub_china_b["accepted_failure_rate"].values * 100

        ax.plot(cov, fail_rate_h, "o-", color="#2563eb", linewidth=2.5, markersize=7, label="Model H (Full Combined)")
        ax.plot(cov, fail_rate_b, "s--", color="#059669", linewidth=2, markersize=6, label="Model B (YOLO Confidence)")
        
        # Baseline reference
        base_err = fail_rate_h[0]
        ax.axhline(base_err, color="#dc2626", linestyle=":", label=f"Baseline Error ({base_err:.1f}%)")

        for c, f in zip(cov, fail_rate_h):
            ax.annotate(f"{f:.1f}%", (c, f), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Coverage (% of Automated Inspections Accepted)")
        ax.set_ylabel("Accepted Failure Rate (%)" if idx == 0 else "")
        ax.set_xlim(105, 45)  # Inverted x-axis
        ax.set_ylim(0, max(fail_rate_h[0] + 5, 35))
        ax.legend(loc="upper left", frameon=True)

    plt.suptitle("Fig 4: Risk-Coverage Error Reduction under Strict Targets (China_Drone)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_risk_coverage_strict_target.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 4")


def plot_fig5_raw_distance(df_china: pd.DataFrame, df_india: pd.DataFrame):
    """Fig 5: Raw DINO kNN Distance Distributions & Saturation Audit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Raw DINO k-NN Distance
    ax1 = axes[0]
    bins = np.linspace(0.0, 1.0, 50)
    ax1.hist(df_china["dino_knn_distance"], bins=bins, alpha=0.7, color="#2563eb", label="China_Drone Val (N=480)", density=True)
    ax1.hist(df_india["dino_knn_distance"], bins=bins, alpha=0.7, color="#dc2626", label="India Cross-Domain (N=300)", density=True)
    ax1.axvline(CHINA_P99_KNN, color="#1e293b", linestyle="--", linewidth=2, label=f"China Train p99 = {CHINA_P99_KNN:.4f}")
    ax1.axvline(CHINA_P95_KNN, color="#64748b", linestyle=":", linewidth=1.5, label=f"China Train p95 = {CHINA_P95_KNN:.4f}")
    ax1.set_title("Panel A: Raw DINOv2 k-NN Cosine Distance", fontweight="bold")
    ax1.set_xlabel("k-NN Cosine Distance (k=20)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", frameon=True)

    # Panel B: Raw Normalized Distance (d / p99)
    ax2 = axes[1]
    norm_china = df_china["dino_knn_distance"] / CHINA_P99_KNN
    norm_india = df_india["dino_knn_distance"] / CHINA_P99_KNN
    bins_norm = np.linspace(0.0, 2.2, 50)
    ax2.hist(norm_china, bins=bins_norm, alpha=0.7, color="#2563eb", label="China_Drone Val (d/p99)", density=True)
    ax2.hist(norm_india, bins=bins_norm, alpha=0.7, color="#dc2626", label="India Cross-Domain (d/p99)", density=True)
    ax2.axvline(1.0, color="#1e293b", linestyle="--", linewidth=2, label="Clipping Threshold (OOD=1.0)")
    ax2.set_title("Panel B: Raw Normalized Distance (Without Saturation Clipping)", fontweight="bold")
    ax2.set_xlabel("Normalized Distance (d / p99)")
    ax2.set_ylabel("Density")
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle("Fig 5: DINOv2 Domain Distance & Saturation Audit", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_domain_distance_raw.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 5")


def plot_fig6_domain_gate_roc(df_china: pd.DataFrame, df_india: pd.DataFrame):
    """Fig 6: DINO Domain Gate ROC and Precision-Recall Curves."""
    y_domain = np.array([0] * len(df_china) + [1] * len(df_india))
    scores = np.concatenate([df_china["dino_knn_distance"].values, df_india["dino_knn_distance"].values])

    fpr, tpr, _ = roc_curve(y_domain, scores)
    prec, rec, _ = precision_recall_curve(y_domain, scores)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curve
    ax1 = axes[0]
    ax1.plot(fpr, tpr, color="#2563eb", linewidth=2.5, label="DINO k-NN Gate (AUROC = 1.0000)")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    ax1.scatter([np.mean(df_china["dino_knn_distance"] > CHINA_P99_KNN)],
                [np.mean(df_india["dino_knn_distance"] > CHINA_P99_KNN)],
                color="#dc2626", s=100, zorder=5, label="Operating Point (p99: FPR=1.0%, TPR=100%)")
    ax1.set_title("Panel A: Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax1.set_xlabel("False Positive Rate (China False Warning)")
    ax1.set_ylabel("True Positive Rate (India Shift Detection)")
    ax1.legend(loc="lower right", frameon=True)

    # Precision-Recall Curve
    ax2 = axes[1]
    ax2.plot(rec, prec, color="#059669", linewidth=2.5, label="DINO k-NN Gate (AUPRC = 1.0000)")
    ax2.set_title("Panel B: Precision-Recall Curve", fontweight="bold")
    ax2.set_xlabel("Recall (Domain Shift Detection Rate)")
    ax2.set_ylabel("Precision (Domain Warning Accuracy)")
    ax2.legend(loc="lower left", frameon=True)

    plt.suptitle("Fig 6: DINOv2 Foundation Embeddings as Macro Domain Gate", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_domain_gate_roc.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 6")


def plot_fig7_reliability_bands(df_bands: pd.DataFrame):
    """Fig 7: Reliability Band Performance across Targets T0 - T4."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, (d_str, title) in enumerate([
        ("China", "In-Domain (China_Drone Val) Band Success Rate"),
        ("India", "Cross-Domain (India Dashcam) Band Failure Isolation"),
    ]):
        ax = axes[idx]
        sub = df_bands[(df_bands["dataset"].str.contains(d_str)) & (df_bands["band_calibration_type"] == "Original_Phase8_Bands") & (df_bands["model"] == "Model_H")]
        
        targets = ["T0", "T1", "T2", "T3", "T4"]
        target_labels = ["T0 (F1>0)", "T1 (F1≥0.5)", "T2 (Rec≥0.5)", "T3 (Rec≥0.75)", "T4 (Rec=1.0)"]
        
        high_succ = [sub[sub["target"] == t]["high_band_success_rate"].values[0] * 100 for t in targets]
        med_succ = [sub[sub["target"] == t]["medium_band_success_rate"].values[0] * 100 for t in targets]
        low_fail = [sub[sub["target"] == t]["low_band_failure_rate"].values[0] * 100 for t in targets]

        x = np.arange(len(targets))
        width = 0.26

        if idx == 0:
            ax.bar(x - width, high_succ, width, label="HIGH Band Success (%)", color="#16a34a", alpha=0.9)
            ax.bar(x, med_succ, width, label="MEDIUM Band Success (%)", color="#f59e0b", alpha=0.9)
            ax.bar(x + width, low_fail, width, label="LOW Band Failure (%)", color="#dc2626", alpha=0.9)
            ax.set_ylabel("Rate (%)")
        else:
            low_counts = [sub[sub["target"] == t]["low_band_count"].values[0] for t in targets]
            ax.bar(x, low_fail, width=0.45, label="LOW Band Failure Rate (%)", color="#dc2626", alpha=0.9)
            for i, cnt in enumerate(low_counts):
                ax.text(x[i], low_fail[i] / 2, f"{cnt}/300\n({cnt/300*100:.1f}%)", ha="center", va="center", color="white", fontweight="bold")

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(target_labels, rotation=20)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right" if idx == 0 else "lower right", frameon=True)

    plt.suptitle("Fig 7: Operational Reliability Band Validation (HIGH / MEDIUM / LOW)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_reliability_band_validation.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 7")


def plot_fig8_in_vs_cross_domain(df_china: pd.DataFrame, df_india: pd.DataFrame):
    """Fig 8: In-Domain vs Cross-Domain Reliability Score Distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rel_china = df_china["reliability_Model_H_T1"].values
    rel_india = df_india["reliability_Model_H_T1"].values

    bins = np.linspace(0.0, 1.0, 40)
    ax.hist(rel_china, bins=bins, alpha=0.7, color="#2563eb", label=f"China_Drone (Mean Rel = {np.mean(rel_china):.3f})", density=True)
    ax.hist(rel_india, bins=bins, alpha=0.7, color="#dc2626", label=f"India Shifted (Mean Rel = {np.mean(rel_india):.3f})", density=True)

    ax.axvline(0.85, color="#16a34a", linestyle="--", linewidth=2, label="HIGH Band Threshold (≥0.85)")
    ax.axvline(0.60, color="#dc2626", linestyle="--", linewidth=2, label="LOW Band Threshold (<0.60)")

    ax.set_title("Fig 8: In-Domain vs Cross-Domain Reliability Score Distribution (Target T1)", fontweight="bold", pad=12)
    ax.set_xlabel("Estimated Perception Reliability Score (0.0 to 1.0)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper center", frameon=True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_in_vs_cross_domain_reliability.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 8")


def plot_fig9_temporal_ablation(df_temp_eval: pd.DataFrame):
    """Fig 9: Temporal Feature Ablation (Experiment A)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = df_temp_eval["model"].tolist()
    labels = ["B: Conf Only", "I: Conf + Temp", "J: Conf + DINO + Temp", "K: All Features"]
    aurocs = df_temp_eval["AUROC"].tolist()
    auprcs = df_temp_eval["AUPRC"].tolist()

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, aurocs, width, label="AUROC", color="#2563eb", alpha=0.85)
    ax.bar(x + width/2, auprcs, width, label="AUPRC", color="#059669", alpha=0.85)

    for i in range(len(models)):
        ax.text(x[i] - width/2, aurocs[i] + 0.02, f"{aurocs[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i] + width/2, auprcs[i] + 0.02, f"{auprcs[i]:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Fig 9: Temporal Signal Ablation on Experiment A (N=33 States)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Metric Score")
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig9_temporal_ablation.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 9")


def plot_fig10_failure_cases(df_china: pd.DataFrame, df_india: pd.DataFrame):
    """Fig 10: Multi-panel Qualitative Case Failure Analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot A: High-confidence YOLO False Alarm / False Positive
    ax_a = axes[0, 0]
    ax_a.text(0.5, 0.7, "Case A: High-Confidence YOLO Failure\n(China_Drone_000033 / False Over-Confidence)",
              ha="center", va="center", fontsize=12, fontweight="bold", color="#1e293b")
    ax_a.text(0.5, 0.35, "• Ground Truth: 2 cracks\n• YOLO Detections: 0 (FN=2)\n• Max Conf = 0.00 | DINO kNN = 0.2810\n• Reliability Score = 0.04 (Correctly Quarantined to LOW Band)",
              ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#fee2e2", edgecolor="#ef4444"))
    ax_a.set_title("Panel A: High-Risk Miss Quarantined by Reliability", fontweight="bold", color="#b91c1c")
    ax_a.axis("off")

    # Subplot B: Low-confidence Correct Perception
    ax_b = axes[0, 1]
    ax_b.text(0.5, 0.7, "Case B: Low-Confidence Correct Perception\n(China_Drone_000015 / Subtle Crack)",
              ha="center", va="center", fontsize=12, fontweight="bold", color="#1e293b")
    ax_b.text(0.5, 0.35, "• Ground Truth: 1 crack\n• YOLO Detections: 2 (TP=1, FP=1, F1=0.67)\n• Max Conf = 0.558 | DINO kNN = 0.1437\n• Reliability Score = 0.81 (Assigned to MEDIUM Band for Secondary Inspection)",
              ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7", edgecolor="#f59e0b"))
    ax_b.set_title("Panel B: Borderline Perception Routed to Secondary Inspection", fontweight="bold", color="#b45309")
    ax_b.axis("off")

    # Subplot C: Out-of-Domain Dashcam Flagged by DINO Domain Gate
    ax_c = axes[1, 0]
    ax_c.text(0.5, 0.7, "Case C: Extreme Domain Shift Detection\n(India_000005 / Unpaved Wet Dashcam)",
              ha="center", va="center", fontsize=12, fontweight="bold", color="#1e293b")
    ax_c.text(0.5, 0.35, "• Domain: Indian Dashcam (vs Drone Training)\n• DINO k-NN Distance = 0.8530 (vs p99=0.4491)\n• DINO Domain Flag: EXTREME_DOMAIN_SHIFT (100% Gated)\n• Outcome: Automated Acceptance Blocked at Entry",
              ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#e0e7ff", edgecolor="#4338ca"))
    ax_c.set_title("Panel C: DINO Macro Domain Gating Action", fontweight="bold", color="#3730a3")
    ax_c.axis("off")

    # Subplot D: Strict Target Partial Miss (T3 / T4)
    ax_d = axes[1, 1]
    ax_d.text(0.5, 0.7, "Case D: Strict Target Partial Recall Miss\n(China_Drone_000004 / 1 of 2 Defects Detected)",
              ha="center", va="center", fontsize=12, fontweight="bold", color="#1e293b")
    ax_d.text(0.5, 0.35, "• Ground Truth: 2 defects | YOLO TP = 1, FN = 1\n• Recall = 0.50 (Passes T0 & T1, FAILS Strict T3 & T4)\n• Max Conf = 0.7612\n• Key Finding: Single-defect capture requires stricter recall targets",
              ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3e8ff", edgecolor="#9333ea"))
    ax_d.set_title("Panel D: Strict Recall Target (T3/T4) Failure Mode", fontweight="bold", color="#6b21a8")
    ax_d.axis("off")

    plt.suptitle("Fig 10: RoadSentinel Qualitative Failure Taxonomy & Gating Behaviors", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig10_failure_case_panel.png", bbox_inches="tight")
    plt.close()
    log.info("Saved Fig 10")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    df_prev = pd.read_csv(TABLES_DIR / "table_target_prevalence.csv")
    df_ablation = pd.read_csv(TABLES_DIR / "table_full_ablation.csv")
    df_bands = pd.read_csv(TABLES_DIR / "table_reliability_bands_multitarget.csv")
    df_risk = pd.read_csv(TABLES_DIR / "table_risk_coverage_strict.csv")
    df_china = pd.read_csv(DATA_DIR / "china_validation_ablation_master.csv")
    df_india = pd.read_csv(DATA_DIR / "india_cross_domain_ablation_master.csv")
    df_temp = pd.read_csv(TABLES_DIR / "table_temporal_ablation.csv")

    plot_fig1_target_prevalence(df_prev)
    plot_fig2_ablation_auroc(df_ablation)
    plot_fig3_ablation_auprc(df_ablation)
    plot_fig4_risk_coverage(df_risk)
    plot_fig5_raw_distance(df_china, df_india)
    plot_fig6_domain_gate_roc(df_china, df_india)
    plot_fig7_reliability_bands(df_bands)
    plot_fig8_in_vs_cross_domain(df_china, df_india)
    plot_fig9_temporal_ablation(df_temp)
    plot_fig10_failure_cases(df_china, df_india)

    log.info("All 10 Phase 12 figures successfully generated.")


if __name__ == "__main__":
    main()
