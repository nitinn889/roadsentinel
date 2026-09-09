"""Build Training-Domain Reference & Reliability Dataset for RoadSentinel Phase 8.

Extracts DINOv2 (ViT-S/14) global CLS embeddings from the 1,921 YOLO training images,
establishes the non-parametric k-NN training-domain reference, extracts validation
features, image condition metrics, and YOLO confidence statistics, and compiles the
complete 480-image reliability dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_reliability_dataset")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_IMG_DIR = WORKSPACE_ROOT / "yolo/data/rdd2022/images/train"
VAL_IMG_DIR = WORKSPACE_ROOT / "yolo/data/rdd2022/images/val"
BENCHMARK_JSON = WORKSPACE_ROOT / "benchmark/final_comparison/benchmark_summary.json"
OUT_DATA_DIR = WORKSPACE_ROOT / "reliability/data"
OUT_DOMAIN_DIR = WORKSPACE_ROOT / "reliability/domain"

BATCH_SIZE = 32
K_NEIGHBORS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        with Image.open(path).convert("RGB") as img:
            tensor = self.transform(img)
        return tensor, path.stem


def extract_dinov2_embeddings(
    image_paths: List[Path], model: torch.nn.Module, transform: Any, desc: str = ""
) -> Tuple[np.ndarray, List[str]]:
    """Extract L2-normalized 384-dimensional DINOv2 CLS token embeddings."""
    dataset = SimpleImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    all_embeds = []
    all_names = []

    log.info("Extracting DINOv2 embeddings for %d images (%s)...", len(image_paths), desc)
    model.eval()
    with torch.no_grad():
        for i, (imgs, names) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            # CLS token output (B, 384)
            embeds = model(imgs)
            # L2 normalize
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.cpu().numpy())
            all_names.extend(names)
            if (i + 1) % 15 == 0 or (i + 1) == len(loader):
                log.info("  Processed %d / %d batches", i + 1, len(loader))

    all_embeds_np = np.concatenate(all_embeds, axis=0)
    return all_embeds_np, all_names


def compute_image_condition_features(image_path: Path) -> Dict[str, float]:
    """Extract brightness, contrast, sharpness, and edge density using OpenCV."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "edge_density": 0.0,
        }
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Brightness: mean intensity
    brightness = float(np.mean(gray))
    # Contrast: std deviation of intensities
    contrast = float(np.std(gray))
    # Sharpness: variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian))
    # Edge density: Canny edge fraction
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1]))

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "edge_density": round(edge_density, 5),
    }


def main():
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DOMAIN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load DINOv2 model
    log.info("Loading DINOv2 ViT-S/14 backbone onto %s...", DEVICE)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE)
    model.eval()

    dinov2_transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 2. Extract Training-Domain Embeddings (1,921 images)
    train_paths = sorted(list(TRAIN_IMG_DIR.glob("*.jpg")) + list(TRAIN_IMG_DIR.glob("*.png")))
    log.info("Found %d training images.", len(train_paths))
    train_embeds, train_names = extract_dinov2_embeddings(train_paths, model, dinov2_transform, "YOLO Training Set")

    # Compute training domain centroid
    train_centroid = np.mean(train_embeds, axis=0)
    train_centroid = train_centroid / np.linalg.norm(train_centroid)

    # Compute training internal reference scale (leave-one-out cosine distance)
    # Cosine distance = 1 - dot product for L2 normalized vectors
    train_sims = np.dot(train_embeds, train_embeds.T)
    # Exclude self-similarity along diagonal
    np.fill_diagonal(train_sims, -1.0)
    # Top-K highest similarities (smallest distances)
    topk_sims = np.partition(train_sims, -K_NEIGHBORS, axis=1)[:, -K_NEIGHBORS:]
    train_knn_dists = 1.0 - np.mean(topk_sims, axis=1)

    dist_min = float(np.min(train_knn_dists))
    dist_median = float(np.median(train_knn_dists))
    dist_p95 = float(np.percentile(train_knn_dists, 95))
    dist_p99 = float(np.percentile(train_knn_dists, 99))
    dist_max = float(np.max(train_knn_dists))

    log.info("Training Domain Reference Distance Distribution (k=%d):", K_NEIGHBORS)
    log.info("  Min: %.4f | Median: %.4f | p95: %.4f | p99: %.4f | Max: %.4f",
             dist_min, dist_median, dist_p95, dist_p99, dist_max)

    # Save training domain summary
    domain_meta = {
        "model_backbone": "dinov2_vits14",
        "embedding_dim": 384,
        "input_resolution": [518, 518],
        "training_image_count": len(train_paths),
        "k_neighbors": K_NEIGHBORS,
        "knn_distance_distribution": {
            "min": dist_min,
            "median": dist_median,
            "mean": float(np.mean(train_knn_dists)),
            "std": float(np.std(train_knn_dists)),
            "p95": dist_p95,
            "p99": dist_p99,
            "max": dist_max,
        },
        "normalization_formula": "dino_domain_score = clip(1.0 - (d_knn - p05) / (p95 - p05), 0, 1)",
    }
    (OUT_DOMAIN_DIR / "domain_reference_metadata.json").write_text(json.dumps(domain_meta, indent=2), encoding="utf-8")

    # 3. Extract Validation Embeddings (480 images)
    bench_data = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    val_recs = bench_data["per_image_records"]

    val_paths = []
    val_id_to_rec = {}
    for r in val_recs:
        img_id = r["image_id"]
        p = VAL_IMG_DIR / f"{img_id}.jpg"
        if not p.exists():
            p = WORKSPACE_ROOT / f"benchmark/common_candidate/images/{img_id}.jpg"
        if not p.exists():
            p = WORKSPACE_ROOT / f"benchmark/common_candidate/images/{img_id}.png"
        val_paths.append(p)
        val_id_to_rec[img_id] = r

    val_embeds, val_names = extract_dinov2_embeddings(val_paths, model, dinov2_transform, "Validation Set")

    # 4. Compute Distance & Familiarity Metrics for Validation Images
    # Val similarity to all training embeddings: (480, 1921)
    val_train_sims = np.dot(val_embeds, train_embeds.T)
    # Top-K nearest
    val_topk_sims = np.partition(val_train_sims, -K_NEIGHBORS, axis=1)[:, -K_NEIGHBORS:]
    val_knn_dists = 1.0 - np.mean(val_topk_sims, axis=1)
    val_min_dists = 1.0 - np.max(val_train_sims, axis=1)
    val_centroid_dists = 1.0 - np.dot(val_embeds, train_centroid)

    # Familiarity scaling: relative to training reference distribution
    # Using training min and 99th percentile for robust normalization
    ref_min = dist_min
    ref_max = dist_p99
    val_domain_scores = np.clip(1.0 - (val_knn_dists - ref_min) / (ref_max - ref_min), 0.0, 1.0)
    val_ood_scores = 1.0 - val_domain_scores

    # 5. Compile Complete Reliability Dataset
    rows = []
    log.info("Compiling reliability dataset with visual condition and confidence features...")

    for idx, (img_id, p) in enumerate(zip(val_names, val_paths)):
        rec = val_id_to_rec[img_id]
        raw_yolo = rec.get("raw_yolo", [])
        
        # Ground Truth & Outcome Labels
        gt_count = int(rec["gt_count"])
        yolo_pred_count = int(rec["yolo_pred_count"])
        yolo_tp = int(rec["yolo_tp"])
        yolo_fp = int(rec["yolo_fp"])
        yolo_fn = int(rec["yolo_fn"])
        yolo_precision = float(rec["yolo_precision"])
        yolo_recall = float(rec["yolo_recall"])
        yolo_f1 = float(rec["yolo_f1"])

        # Target definitions:
        # Primary: YOLO_FAILURE = 1 if F1 == 0 (failed to localize any GT box with IoU >= 0.50), else 0
        yolo_failure = 1 if yolo_f1 == 0.0 else 0
        yolo_success = 1 - yolo_failure
        # Strict alternative: YOLO_STRICT_FAILURE = 1 if F1 < 0.50
        yolo_strict_failure = 1 if yolo_f1 < 0.50 else 0

        # YOLO Confidence Features
        confs = [float(d["confidence"]) for d in raw_yolo] if raw_yolo else []
        mean_conf = float(np.mean(confs)) if confs else 0.0
        max_conf = float(np.max(confs)) if confs else 0.0
        min_conf = float(np.min(confs)) if confs else 0.0
        std_conf = float(np.std(confs)) if len(confs) > 1 else 0.0
        num_low_conf = sum(1 for c in confs if c < 0.40)
        num_high_conf = sum(1 for c in confs if c >= 0.60)

        # Image Condition Features
        cond_feats = compute_image_condition_features(p)

        row = {
            "image_id": img_id,
            # Ground truth / YOLO Performance (Targets)
            "gt_count": gt_count,
            "yolo_pred_count": yolo_pred_count,
            "yolo_tp": yolo_tp,
            "yolo_fp": yolo_fp,
            "yolo_fn": yolo_fn,
            "yolo_precision": round(yolo_precision, 4),
            "yolo_recall": round(yolo_recall, 4),
            "yolo_f1": round(yolo_f1, 4),
            "yolo_failure": yolo_failure,
            "yolo_success": yolo_success,
            "yolo_strict_failure": yolo_strict_failure,
            # YOLO Confidence Features
            "mean_confidence": round(mean_conf, 4),
            "max_confidence": round(max_conf, 4),
            "min_confidence": round(min_conf, 4),
            "std_confidence": round(std_conf, 4),
            "num_low_conf": num_low_conf,
            "num_high_conf": num_high_conf,
            # DINOv2 Domain Features
            "dino_knn_distance": round(float(val_knn_dists[idx]), 4),
            "dino_min_distance": round(float(val_min_dists[idx]), 4),
            "dino_centroid_distance": round(float(val_centroid_dists[idx]), 4),
            "dino_domain_score": round(float(val_domain_scores[idx]), 4),
            "dino_ood_score": round(float(val_ood_scores[idx]), 4),
            # Image Condition Features
            "brightness": cond_feats["brightness"],
            "contrast": cond_feats["contrast"],
            "sharpness": cond_feats["sharpness"],
            "edge_density": cond_feats["edge_density"],
        }
        rows.append(row)

    df_rel = pd.DataFrame(rows)
    out_csv = OUT_DATA_DIR / "reliability_dataset.csv"
    df_rel.to_csv(out_csv, index=False)
    log.info("Saved reliability dataset to %s (Shape: %s)", out_csv, df_rel.shape)
    log.info("Class Distribution: Failures (F1=0) = %d (%.1f%%) | Successes = %d (%.1f%%)",
             df_rel["yolo_failure"].sum(), df_rel["yolo_failure"].mean() * 100,
             df_rel["yolo_success"].sum(), df_rel["yolo_success"].mean() * 100)

    # 6. Evaluate Diagnostic Domain-Shift Images
    diag_samples = [
        ("India_005086", WORKSPACE_ROOT / "yolo/test_conditions/cross_domain/India_005086.jpg", "Forward dashcam view (cross-domain)"),
        ("0454", WORKSPACE_ROOT / "yolo/test_conditions/cross_domain/0454.png", "Pothole dataset probe"),
        ("original_healthy", WORKSPACE_ROOT / "yolo/test_conditions/cross_domain/original_healthy.jpg", "Pristine synthetic road baseline"),
    ]
    diag_paths = [p for _, p, _ in diag_samples if p.exists()]
    diag_records = {}

    if diag_paths:
        diag_embeds, diag_names = extract_dinov2_embeddings(diag_paths, model, dinov2_transform, "Diagnostic Cross-Domain Probes")
        diag_train_sims = np.dot(diag_embeds, train_embeds.T)
        diag_topk_sims = np.partition(diag_train_sims, -K_NEIGHBORS, axis=1)[:, -K_NEIGHBORS:]
        diag_knn_dists = 1.0 - np.mean(diag_topk_sims, axis=1)
        diag_domain_scores = np.clip(1.0 - (diag_knn_dists - ref_min) / (ref_max - ref_min), 0.0, 1.0)
        diag_ood_scores = 1.0 - diag_domain_scores

        for i, (tag, p, desc) in enumerate(diag_samples):
            if p.exists():
                diag_records[tag] = {
                    "description": desc,
                    "file": str(p),
                    "dino_knn_distance": round(float(diag_knn_dists[i]), 4),
                    "dino_domain_score": round(float(diag_domain_scores[i]), 4),
                    "dino_ood_score": round(float(diag_ood_scores[i]), 4),
                }
                log.info("Diagnostic [%s]: kNN Dist = %.4f | Domain Score = %.4f | OOD Score = %.4f (%s)",
                         tag, diag_knn_dists[i], diag_domain_scores[i], diag_ood_scores[i], desc)

    (OUT_DOMAIN_DIR / "diagnostic_domain_scores.json").write_text(json.dumps(diag_records, indent=2), encoding="utf-8")
    log.info("Completed domain reference and dataset extraction.")


if __name__ == "__main__":
    main()
