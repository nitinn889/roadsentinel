#!/usr/bin/env python3
"""RoadSentinel Phase 1B: Experiment 7 - Controlled Corruption Robustness Validation.

Tests 7 controlled corruption types across 3 severity levels (Mild, Moderate, Severe):
1. Brightness reduction (0.75, 0.50, 0.25)
2. Contrast reduction (0.75, 0.50, 0.25)
3. Gaussian blur (k=5, 11, 21)
4. Motion blur (k=7, 15, 25)
5. Image compression (JPEG quality 70, 40, 15)
6. Synthetic rain / streak noise (50, 150, 300 streaks)
7. Partial occlusion (5%, 15%, 30% area)

Evaluates:
- YOLO F1
- Domain Score (mean d_kNN vs training reference)
- Domain Escalation Rate (fraction exceeding p99 = 0.4491)
- Mean Reliability Score
- Unsafe Acceptance Rate (fraction of detection failures auto-accepted)

Critical Scope Requirement:
- Save transformed data only in temporary/runtime cache. Never overwrite source files.
- Labeled strictly as 'controlled corruption robustness', NOT real-world adverse weather.

Outputs:
- artifacts/phase1b/corruption_robustness.csv
- figures/phase1b/corruption_robustness.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase1b_corruptions")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts" / "phase1b"
FIGURES_DIR = WORKSPACE_ROOT / "figures" / "phase1b"
BENCHMARK_SUMMARY_PATH = WORKSPACE_ROOT / "benchmark" / "final_comparison" / "benchmark_summary.json"
VAL_IMAGES_DIR = WORKSPACE_ROOT / "yolo" / "data" / "rdd2022" / "images" / "val"
YOLO_WEIGHTS_PATH = WORKSPACE_ROOT / "yolo" / "weights" / "best.pt"
REF_BANK_PATH = WORKSPACE_ROOT / "cross_domain" / "results" / "train_domain_reference_embeddings.npz"

P99_THRESHOLD = 0.4491
RELIABILITY_HIGH_THRESH = 0.85
N_EVAL_SAMPLES = 50 # Deterministic stratified sample of 50 images across validation set
RANDOM_SEED = 42

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300


def bbox_iou(b1: List[float], b2: List[float]) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    ia = iw * ih
    ua = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - ia
    return ia / ua if ua > 0 else 0.0


# -----------------------------------------------------------------------------
# Controlled Corruption Functions
# -----------------------------------------------------------------------------
def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img * factor, 0, 255).astype(np.uint8)


def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    mean = np.mean(img)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)


def apply_gaussian_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_motion_blur(img: np.ndarray, size: int) -> np.ndarray:
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel /= size
    return cv2.filter2D(img, -1, kernel)


def apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, 1)


def apply_rain(img: np.ndarray, n_streaks: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w, _ = out.shape
    for _ in range(n_streaks):
        x = rng.integers(0, max(1, w - 10))
        y = rng.integers(0, max(1, h - 25))
        cv2.line(out, (x, y), (x + 3, y + 20), (200, 200, 200), 1)
    return out


def apply_occlusion(img: np.ndarray, pct: float) -> np.ndarray:
    out = img.copy()
    h, w, _ = out.shape
    occ_h, occ_w = int(h * np.sqrt(pct)), int(w * np.sqrt(pct))
    y1, x1 = (h - occ_h) // 2, (w - occ_w) // 2
    out[y1:y1 + occ_h, x1:x1 + occ_w] = 0
    return out


CORRUPTIONS = [
    ("clean_baseline", 0, "Clean Baseline (No Corruption)", lambda img: img),
    # 1. Brightness
    ("brightness", 1, "Mild Brightness (0.75x)", lambda img: apply_brightness(img, 0.75)),
    ("brightness", 2, "Moderate Brightness (0.50x)", lambda img: apply_brightness(img, 0.50)),
    ("brightness", 3, "Severe Brightness (0.25x)", lambda img: apply_brightness(img, 0.25)),
    # 2. Contrast
    ("contrast", 1, "Mild Contrast (0.75x)", lambda img: apply_contrast(img, 0.75)),
    ("contrast", 2, "Moderate Contrast (0.50x)", lambda img: apply_contrast(img, 0.50)),
    ("contrast", 3, "Severe Contrast (0.25x)", lambda img: apply_contrast(img, 0.25)),
    # 3. Gaussian Blur
    ("gaussian_blur", 1, "Mild Gaussian Blur (k=5)", lambda img: apply_gaussian_blur(img, 5, 1.0)),
    ("gaussian_blur", 2, "Moderate Gaussian Blur (k=11)", lambda img: apply_gaussian_blur(img, 11, 2.5)),
    ("gaussian_blur", 3, "Severe Gaussian Blur (k=21)", lambda img: apply_gaussian_blur(img, 21, 5.0)),
    # 4. Motion Blur
    ("motion_blur", 1, "Mild Motion Blur (k=7)", lambda img: apply_motion_blur(img, 7)),
    ("motion_blur", 2, "Moderate Motion Blur (k=15)", lambda img: apply_motion_blur(img, 15)),
    ("motion_blur", 3, "Severe Motion Blur (k=25)", lambda img: apply_motion_blur(img, 25)),
    # 5. JPEG Compression
    ("jpeg_compression", 1, "Mild JPEG (Quality 70)", lambda img: apply_jpeg(img, 70)),
    ("jpeg_compression", 2, "Moderate JPEG (Quality 40)", lambda img: apply_jpeg(img, 40)),
    ("jpeg_compression", 3, "Severe JPEG (Quality 15)", lambda img: apply_jpeg(img, 15)),
    # 6. Synthetic Rain
    ("synthetic_rain", 1, "Mild Rain (50 streaks)", lambda img: apply_rain(img, 50)),
    ("synthetic_rain", 2, "Moderate Rain (150 streaks)", lambda img: apply_rain(img, 150)),
    ("synthetic_rain", 3, "Severe Rain (300 streaks)", lambda img: apply_rain(img, 300)),
    # 7. Partial Occlusion
    ("partial_occlusion", 1, "Mild Occlusion (5% area)", lambda img: apply_occlusion(img, 0.05)),
    ("partial_occlusion", 2, "Moderate Occlusion (15% area)", lambda img: apply_occlusion(img, 0.15)),
    ("partial_occlusion", 3, "Severe Occlusion (30% area)", lambda img: apply_occlusion(img, 0.30)),
]


def run_corruption_audit() -> None:
    log.info("Starting Experiment 7: Controlled Corruption Robustness Validation...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load ground truth for China validation set
    with open(BENCHMARK_SUMMARY_PATH) as f:
        bench_data = json.load(f)
    gt_map = {r["image_id"]: [g["bbox"] for g in r["raw_gt"]] for r in bench_data["per_image_records"]}

    # Sample deterministic subset of 50 images
    all_img_files = sorted(list(VAL_IMAGES_DIR.glob("*.jpg")) + list(VAL_IMAGES_DIR.glob("*.png")))
    step = len(all_img_files) // N_EVAL_SAMPLES
    selected_files = [all_img_files[i * step] for i in range(N_EVAL_SAMPLES)]
    log.info("Selected %d representative validation images for corruption evaluation.", len(selected_files))

    # 2. Load models (YOLO & DINOv2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading YOLOv8n from %s onto %s...", YOLO_WEIGHTS_PATH, device)
    yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))

    log.info("Loading DINOv2 ViT-S/14 onto %s...", device)
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    dino_model.eval()

    dino_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 3. Load DINOv2 reference bank
    ref_data = np.load(REF_BANK_PATH)
    ref_embeds = ref_data["embeds"]
    ref_embeds = ref_embeds / np.linalg.norm(ref_embeds, axis=1, keepdims=True)

    corruption_results: List[Dict[str, Any]] = []

    # Iterate over all 22 conditions
    for c_type, sev, label, transform_fn in CORRUPTIONS:
        log.info("Evaluating condition: %s (Severity %d)...", c_type, sev)

        tps, fps, fns = 0, 0, 0
        domain_dists: List[float] = []
        reliability_scores: List[float] = []
        unsafe_accepts = 0
        total_eval_images = 0

        for f_path in selected_files:
            img_id = f_path.stem
            gt_boxes = gt_map.get(img_id, [])
            img_bgr = cv2.imread(str(f_path))
            if img_bgr is None:
                continue
            total_eval_images += 1

            # Apply corruption transformation
            trans_bgr = transform_fn(img_bgr)
            trans_rgb = cv2.cvtColor(trans_bgr, cv2.COLOR_BGR2RGB)

            # --- YOLO Inference ---
            results = yolo_model(trans_bgr, conf=0.25, iou=0.50, verbose=False)
            pred_boxes = []
            pred_confs = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                for b, c in zip(boxes_xyxy, confs):
                    pred_boxes.append(b.tolist())
                    pred_confs.append(float(c))

            # Match against GT
            candidates = []
            for pi, b1 in enumerate(pred_boxes):
                for gi, b2 in enumerate(gt_boxes):
                    v = bbox_iou(b1, b2)
                    if v >= 0.50:
                        candidates.append((v, pi, gi))
            candidates.sort(key=lambda x: x[0], reverse=True)

            matched_p, matched_g = set(), set()
            for v, pi, gi in candidates:
                if pi not in matched_p and gi not in matched_g:
                    matched_p.add(pi)
                    matched_g.add(gi)

            img_tp = len(matched_p)
            img_fp = len(pred_boxes) - img_tp
            img_fn = len(gt_boxes) - img_tp
            tps += img_tp
            fps += img_fp
            fns += img_fn

            img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0.0
            img_rec = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0.0
            img_f1 = (2 * img_prec * img_rec) / (img_prec + img_rec) if (img_prec + img_rec) > 0 else 0.0
            is_failure = img_f1 < 0.50

            # --- DINOv2 Feature Distance ---
            t_tensor = dino_transform(trans_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = dino_model(t_tensor).cpu().numpy().squeeze(0)
            feat_norm = feat / np.linalg.norm(feat)

            # Cosine distance to top 20 reference embeddings
            sims = np.dot(ref_embeds, feat_norm)
            top20_sims = np.partition(sims, -20)[-20:]
            d_knn = float(1.0 - np.mean(top20_sims))
            domain_dists.append(d_knn)

            # --- Reliability Score (Confidence-based heuristic model) ---
            max_c = max(pred_confs) if len(pred_confs) > 0 else 0.0
            mean_c = float(np.mean(pred_confs)) if len(pred_confs) > 0 else 0.0
            # Calibrated logistic score based on confidence
            rel_score = float(1.0 / (1.0 + np.exp(-(3.5 * mean_c + 2.0 * max_c - 2.8))))
            reliability_scores.append(rel_score)

            # Unsafe Accept check: failure exists, but domain pass AND reliability pass
            if is_failure and (d_knn <= P99_THRESHOLD) and (rel_score >= RELIABILITY_HIGH_THRESH):
                unsafe_accepts += 1

        # Summary statistics for condition
        prec = tps / (tps + fps) if (tps + fps) > 0 else 0.0
        rec = tps / (tps + fns) if (tps + fns) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        mean_d = float(np.mean(domain_dists))
        dom_esc_rate = float(np.mean([d > P99_THRESHOLD for d in domain_dists]) * 100.0)
        mean_rel = float(np.mean(reliability_scores))
        unsafe_accept_rate = (unsafe_accepts / total_eval_images) * 100.0

        corruption_results.append({
            "corruption_type": c_type,
            "severity_level": sev,
            "condition_name": label,
            "sample_size": total_eval_images,
            "yolo_precision": round(prec, 4),
            "yolo_recall": round(rec, 4),
            "yolo_f1": round(f1, 4),
            "mean_domain_distance": round(mean_d, 4),
            "domain_escalation_pct": round(dom_esc_rate, 2),
            "mean_reliability_score": round(mean_rel, 4),
            "unsafe_automated_accept_rate_pct": round(unsafe_accept_rate, 2),
        })

    df_out = pd.DataFrame(corruption_results)
    out_csv = ARTIFACTS_DIR / "corruption_robustness.csv"
    df_out.to_csv(out_csv, index=False)
    log.info("Saved corruption robustness table to %s (%d conditions)", out_csv, len(df_out))

    # =========================================================================
    # Figure: Corruption Robustness Comparison
    # =========================================================================
    log.info("Plotting corruption_robustness.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    corr_types = ["brightness", "contrast", "gaussian_blur", "motion_blur", "jpeg_compression", "synthetic_rain", "partial_occlusion"]
    x = np.arange(len(corr_types))
    w = 0.25

    # Panel 1: YOLO F1 across severities
    f1_sev1 = [df_out[(df_out["corruption_type"] == c) & (df_out["severity_level"] == 1)]["yolo_f1"].values[0] for c in corr_types]
    f1_sev2 = [df_out[(df_out["corruption_type"] == c) & (df_out["severity_level"] == 2)]["yolo_f1"].values[0] for c in corr_types]
    f1_sev3 = [df_out[(df_out["corruption_type"] == c) & (df_out["severity_level"] == 3)]["yolo_f1"].values[0] for c in corr_types]
    baseline_f1 = df_out[df_out["corruption_type"] == "clean_baseline"]["yolo_f1"].values[0]

    ax1.axhline(baseline_f1, color="#10b981", linestyle="--", lw=1.8, label=f"Clean Baseline F1 ({baseline_f1:.3f})")
    ax1.bar(x - w, f1_sev1, width=w, label="Severity 1 (Mild)", color="#93c5fd", edgecolor="black")
    ax1.bar(x, f1_sev2, width=w, label="Severity 2 (Moderate)", color="#3b82f6", edgecolor="black")
    ax1.bar(x + w, f1_sev3, width=w, label="Severity 3 (Severe)", color="#1d4ed8", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", " ").title() for c in corr_types], rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("YOLO Detection F1 Score", fontsize=10, fontweight="bold")
    ax1.set_title("Perception Degradation Under Controlled Corruptions", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.0, 0.85)
    ax1.legend(loc="upper right", frameon=True)

    # Panel 2: Domain Escalation Rate (%) across severities
    esc_sev1 = [df_out[(df_out["corruption_type"] == c) & (df_out["severity_level"] == 1)]["domain_escalation_pct"].values[0] for c in corr_types]
    esc_sev2 = [df_out[(df_out["corruption_type"] == c) & (df_out["severity_level"] == 2)]["domain_escalation_pct"].values[0] for c in corr_types]
    esc_sev3 = [df_out[(df_out["corruption_type"] == c) & (df_out["severity_level"] == 3)]["domain_escalation_pct"].values[0] for c in corr_types]

    ax2.bar(x - w, esc_sev1, width=w, label="Severity 1 (Mild)", color="#fde047", edgecolor="black")
    ax2.bar(x, esc_sev2, width=w, label="Severity 2 (Moderate)", color="#f59e0b", edgecolor="black")
    ax2.bar(x + w, esc_sev3, width=w, label="Severity 3 (Severe)", color="#b45309", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", " ").title() for c in corr_types], rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Domain Escalation Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_title(r"DINOv2 Safety Escalation ($d > p_{99}$)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "corruption_robustness.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    log.info("Saved corruption_robustness.png to %s", fig_path)


if __name__ == "__main__":
    run_corruption_audit()
