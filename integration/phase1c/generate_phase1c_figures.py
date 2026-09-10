#!/usr/bin/env python3
"""Phase 1C: Publication-Quality Figure Generation.

Generates the 5 required visual figures for Prompt 1C at 300 DPI:
1. figures/phase1c/model_comparison.png
2. figures/phase1c/observed_vs_predicted_change.png
3. figures/phase1c/error_by_horizon.png
4. figures/phase1c/site_win_loss.png
5. figures/phase1c/feature_importance.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("phase1c_figures")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts" / "phase1c"
FIGURES_DIR = WORKSPACE_ROOT / "figures" / "phase1c"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic palette
NAVY = "#1B365D"
TEAL = "#008080"
AMBER = "#D97706"
CRIMSON = "#DC2626"
SLATE = "#475569"
LIGHT_BG = "#F8FAFC"
GRID_COLOR = "#E2E8F0"

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def plot_model_comparison(comp_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    models = comp_df["model_id"].tolist()
    labels = ["M0\nPersistence", "M1\nOLS", "M2\nXGB Scenario", "M3\nXGB Temporal", "M4\nXGB Combined"]
    colors = [SLATE, AMBER, "#6366F1", "#0D9488", NAVY]
    
    # Panel 1: Future Severity MAE
    axes[0].bar(labels, comp_df["mae"], color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Future Severity MAE (Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_ylabel("Mean Absolute Error (Severity [0,1])", fontsize=10)
    axes[0].set_ylim(0.045, 0.065)
    for i, v in enumerate(comp_df["mae"]):
        axes[0].text(i, v + 0.0006, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
        
    # Panel 2: R^2 Score
    axes[1].bar(labels, comp_df["r2_score"], color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Future Severity R² Score (Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_ylabel("R² Coefficient of Determination", fontsize=10)
    axes[1].set_ylim(0.89, 0.93)
    for i, v in enumerate(comp_df["r2_score"]):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
        
    # Panel 3: MAE Skill vs Persistence
    skills = comp_df["mae_skill_vs_persistence"] * 100.0
    bar_colors = [SLATE if s == 0 else (CRIMSON if s < 0 else TEAL) for s in skills]
    axes[2].bar(labels, skills, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)
    axes[2].axhline(0, color="black", linestyle="-", linewidth=1.0)
    axes[2].set_title("MAE Skill vs. Persistence (%)", fontsize=12, fontweight="bold", pad=10)
    axes[2].set_ylabel("Skill Score % (1 - MAE_model / MAE_M0)", fontsize=10)
    axes[2].set_ylim(-8, 8)
    for i, v in enumerate(skills):
        axes[2].text(i, v + (0.4 if v >= 0 else -0.9), f"{v:+.2f}%", ha="center", fontsize=9, fontweight="bold")
        
    plt.suptitle("RoadSentinel Phase 1C: Forecast Model Ablation (5-Fold Nested Grouped CV, N=113)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out_path)


def plot_observed_vs_predicted_change(oof_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    delta_true = oof_df["delta_severity_true"].values
    delta_pred = oof_df["m4_xgb_combined_pred_delta"].values
    horizons = oof_df["observed_days_ahead"].values
    
    sc = ax.scatter(
        delta_true,
        delta_pred,
        c=horizons,
        cmap="viridis",
        s=45,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Horizon Duration (Days)", fontsize=10)
    
    # Reference lines
    lims = [-0.15, 0.35]
    ax.plot(lims, lims, "k--", alpha=0.6, label="Ideal Prediction (y = x)")
    ax.axhline(0, color=CRIMSON, linestyle=":", alpha=0.7, label="Persistence Baseline (Δy = 0)")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed Severity Change (Δy = y_future - y_current)", fontsize=11)
    ax.set_ylabel("M4 Predicted Severity Change (Δy_pred)", fontsize=11)
    ax.set_title("M4 Combined XGBoost: Observed vs. Predicted Deterioration Change", fontsize=12, fontweight="bold", pad=12)
    
    # Annotation box
    textstr = (
        "Evaluation: 5-Fold Nested Grouped CV\n"
        "Sample Size: N = 113 pairs (23 sites)\n"
        "Direction Accuracy: 65.18%\n"
        "Spearman Rank Corr: ρ = 0.2181\n"
        "Change MAE: 0.0561"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#CBD5E1")
    ax.text(0.04, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)
    
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    out_path = FIGURES_DIR / "observed_vs_predicted_change.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out_path)


def plot_error_by_horizon(skill_json_path: Path) -> None:
    with open(skill_json_path) as f:
        data = json.load(f)
    h_data = data["performance_by_horizon"]
    
    categories = ["Short Horizon\n(< 365 days)\nN=43", "Medium Horizon\n(365–550 days)\nN=35", "Long Horizon\n(> 550 days)\nN=35"]
    keys = ["Short_Horizon_lt_365d", "Medium_Horizon_365_550d", "Long_Horizon_gt_550d"]
    
    m0_vals = [h_data[k]["m0_mae"] for k in keys]
    m4_vals = [h_data[k]["m4_mae"] for k in keys]
    skills = [h_data[k]["m4_skill_vs_m0"] * 100.0 for k in keys]
    
    x = np.arange(len(categories))
    width = 0.32
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300, gridspec_kw={"width_ratios": [1.5, 1]})
    
    # Subplot 1: MAE comparison
    rects1 = ax1.bar(x - width/2, m0_vals, width, label="M0 Persistence", color=SLATE, edgecolor="black", linewidth=0.7)
    rects2 = ax1.bar(x + width/2, m4_vals, width, label="M4 XGBoost Combined", color=NAVY, edgecolor="black", linewidth=0.7)
    
    ax1.set_ylabel("Mean Absolute Error (Severity [0,1])", fontsize=10)
    ax1.set_title("Forecast MAE by Supported Horizon Interval", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylim(0, 0.085)
    ax1.legend(framealpha=0.95)
    
    for r in rects1:
        ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.0015, f"{r.get_height():.4f}", ha="center", fontsize=8.5, fontweight="bold")
    for r in rects2:
        ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.0015, f"{r.get_height():.4f}", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)
        
    # Subplot 2: Skill percentage
    skill_colors = [CRIMSON if s < 0 else TEAL for s in skills]
    rects3 = ax2.bar(categories, skills, width=0.45, color=skill_colors, edgecolor="black", linewidth=0.7)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.set_ylabel("M4 Skill vs Persistence (%)", fontsize=10)
    ax2.set_title("M4 Relative Improvement by Horizon", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylim(-8, 14)
    
    for r in rects3:
        h = r.get_height()
        ax2.text(r.get_x() + r.get_width()/2, h + (0.6 if h >= 0 else -1.2), f"{h:+.2f}%", ha="center", fontsize=9, fontweight="bold")
        
    plt.suptitle("Horizon-Stratified Validation: Persistence Dominates Short Intervals; XGBoost Wins Multi-Year Horizons", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "error_by_horizon.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out_path)


def plot_site_win_loss(site_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Sort sites by delta MAE
    df_sorted = site_df.sort_values("paired_delta_mae_m4_vs_m0", ascending=False).reset_index(drop=True)
    
    deltas = df_sorted["paired_delta_mae_m4_vs_m0"].values
    sites = df_sorted["site_id"].tolist()
    is_audit = (df_sorted["stratum"] == "Previously_Observed_Audit_Site").values
    
    colors = [TEAL if d < 0 else CRIMSON for d in deltas]
    
    bars = ax.barh(np.arange(len(sites)), deltas, color=colors, edgecolor="black", linewidth=0.6, height=0.65)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
    
    ax.set_yticks(np.arange(len(sites)))
    # Add an asterisk for previously observed audit sites
    site_labels = [f"{s} (Audit)*" if a else s for s, a in zip(sites, is_audit)]
    ax.set_yticklabels(site_labels, fontsize=9)
    
    ax.set_xlabel("Paired ΔMAE (M4 Error - M0 Persistence Error)\n[ < 0 : M4 Wins | > 0 : Persistence Wins ]", fontsize=10)
    ax.set_title("Per-Site Paired Forecasting Win/Loss Comparison (N=23 Highway Sections)", fontsize=12, fontweight="bold", pad=12)
    
    # Value annotations
    for i, d in enumerate(deltas):
        offset = 0.0015 if d >= 0 else -0.0015
        ha = "left" if d >= 0 else "right"
        ax.text(d + offset, i, f"{d:+.4f}", va="center", ha=ha, fontsize=8, fontweight="bold")
        
    # Summary box
    wins = (deltas < 0).sum()
    textstr = (
        f"Overall Site Win Rate: {wins}/23 ({wins/23*100:.1f}%)\n"
        f"Audit Set (6 sites)*: 2/6 (33.3%)\n"
        f"Unseen Sites (17 sites): 10/17 (58.8%)\n"
        f"Site-Clustered 95% CI: [-0.0106, +0.0043]"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#CBD5E1")
    ax.text(0.04, 0.15, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="bottom", bbox=props)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "site_win_loss.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out_path)


def plot_feature_importance(manifest_path: Path) -> None:
    raw_df = pd.read_csv(manifest_path)
    
    all_iris = np.concatenate([raw_df["current_iri_m_per_km"].values, raw_df["future_iri_m_per_km"].values])
    q_low = float(np.quantile(all_iris, 0.01))
    q_high = float(np.quantile(all_iris, 0.99))
    scale_range = q_high - q_low
    
    df = raw_df.copy()
    df["current_severity"] = np.clip((df["current_iri_m_per_km"] - q_low) / scale_range, 0.0, 1.0)
    df["future_severity"] = np.clip((df["future_iri_m_per_km"] - q_low) / scale_range, 0.0, 1.0)
    df["delta_severity"] = df["future_severity"] - df["current_severity"]
    
    df["prev_severity"] = np.clip((df["prev_iri_m_per_km"] - q_low) / scale_range, 0.0, 1.0)
    df["second_prev_severity"] = np.clip((df["second_prev_iri_m_per_km"] - q_low) / scale_range, 0.0, 1.0)
    df["rolling_severity_mean"] = np.clip((df["rolling_iri_mean"] - q_low) / scale_range, 0.0, 1.0)
    df["most_recent_severity_change"] = df["most_recent_iri_change"] / scale_range
    df["annualized_severity_slope"] = df["annualized_iri_slope"] / scale_range
    df["rolling_severity_std"] = df["rolling_iri_std"] / scale_range
    
    feature_cols = [
        "current_severity",
        "rainfall_level",
        "traffic_level",
        "temperature",
        "water_exposure",
        "days_ahead",
        "num_prior_inspections",
        "time_since_first_observation_days",
        "time_since_prev_inspection_days",
        "prev_severity",
        "second_prev_severity",
        "most_recent_severity_change",
        "annualized_severity_slope",
        "rolling_severity_mean",
        "rolling_severity_std",
        "same_construction_no",
    ]
    
    X = df[feature_cols].values
    y = df["delta_severity"].values
    
    # Train reference model
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        reg_lambda=5.0,
        subsample=0.85,
        n_jobs=1,
    )
    model.fit(X, y)
    base_pred = model.predict(X)
    base_mae = float(mean_absolute_error(y, base_pred))
    
    # Permutation importance (increase in MAE)
    rng = np.random.default_rng(42)
    perm_importances = []
    for j in range(len(feature_cols)):
        maes = []
        for _ in range(10):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            pred_perm = model.predict(X_perm)
            maes.append(float(mean_absolute_error(y, pred_perm)) - base_mae)
        perm_importances.append(float(np.mean(maes)))
        
    # XGBoost gain importance
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type="gain")
    gain_importances = [score_dict.get(f"f{i}", 0.0) for i in range(len(feature_cols))]
    total_gain = sum(gain_importances)
    gain_pcts = [g / total_gain * 100.0 if total_gain > 0 else 0.0 for g in gain_importances]
    
    # Sort by permutation importance
    sorted_idx = np.argsort(perm_importances)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
    
    y_pos = np.arange(len(feature_cols))
    ax1.barh(y_pos, [perm_importances[i] for i in sorted_idx], color=NAVY, edgecolor="black", linewidth=0.6, height=0.65)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([feature_cols[i] for i in sorted_idx], fontsize=9)
    ax1.set_xlabel("Permutation Importance (Increase in MAE upon Shuffling)", fontsize=10)
    ax1.set_title("Permutation Feature Importance (Predictive Value)", fontsize=11, fontweight="bold", pad=10)
    
    ax2.barh(y_pos, [gain_pcts[i] for i in sorted_idx], color=TEAL, edgecolor="black", linewidth=0.6, height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Tree Gain Importance (% of Total Split Gain)", fontsize=10)
    ax2.set_title("XGBoost Split Gain Distribution (%)\n[Model Behavior Weighting, NOT Physical Causality]", fontsize=11, fontweight="bold", pad=10)
    
    plt.suptitle("M4 Forecaster Feature Explainability: Distinguishing Predictive Value from Causal Claims", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out_path)


def main() -> None:
    comp_df = pd.read_csv(ARTIFACTS_DIR / "model_comparison.csv")
    oof_df = pd.read_csv(ARTIFACTS_DIR / "out_of_fold_predictions.csv")
    site_df = pd.read_csv(ARTIFACTS_DIR / "paired_site_metrics.csv")
    skill_json_path = ARTIFACTS_DIR / "forecast_skill.json"
    manifest_path = ARTIFACTS_DIR / "temporal_feature_manifest.csv"
    
    plot_model_comparison(comp_df)
    plot_observed_vs_predicted_change(oof_df)
    plot_error_by_horizon(skill_json_path)
    plot_site_win_loss(site_df)
    plot_feature_importance(manifest_path)
    log.info("All 5 Phase 1C figures successfully generated.")


if __name__ == "__main__":
    main()
