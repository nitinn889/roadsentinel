"""Build a multi-domain healthy-road distribution bank for RoadSentinel.

Ingests healthy road patches from both real-world road surveys and simulated
CARLA/Unreal road models to eliminate the domain gap.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import find_images, load_rgb, save_json
from inference.sam2_mask import RoadMasker
from inference.dinov2_embed import Dinov2Embedder
from inference.road_distribution import RoadDistributionModel
from memory_bank.coreset import random_presample, k_center_greedy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build_distribution_bank")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build multi-domain healthy road distribution bank.")
    ap.add_argument(
        "--real-healthy-dir",
        type=Path,
        default=ROOT.parent / "RoadSentinel_datasets" / "rdd2022_full" / "China_Drone" / "China_Drone" / "train" / "images",
    )
    ap.add_argument(
        "--sim-healthy-dir",
        type=Path,
        default=ROOT.parent / "dataset" / "rendered_frames",
    )
    ap.add_argument("--output-dir", type=Path, default=CONFIG.output_dir / "distribution_bank")
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--max-images-per-domain", type=int, default=40)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading SAM2 on %s", args.device)
    masker = RoadMasker(device=args.device)
    log.info("Loading DINOv2 on %s", args.device)
    embedder = Dinov2Embedder.from_config(device=args.device)

    dist_model = RoadDistributionModel(embed_dim=384)
    all_coreset_chunks: list[np.ndarray] = []

    # 1. Process Real Healthy Road Domain
    real_images = find_images(args.real_healthy_dir)[: args.max_images_per_domain]
    if real_images:
        log.info("Extracting patch embeddings for REAL domain (%d images)...", len(real_images))
        real_patches = []
        for p in real_images:
            try:
                rgb = load_rgb(p)
                mask = masker.get_road_mask(rgb)
                emb, _ = embedder.extract_road_patch_embeddings(rgb, mask)
                if len(emb):
                    real_patches.append(emb)
            except Exception as e:
                log.warning("Skipping %s: %s", p.name, e)
        if real_patches:
            real_all = np.concatenate(real_patches, axis=0).astype(np.float32)
            dist_model.add_domain_cluster("real_asphalt", real_all)
            # Sample coreset
            pool = random_presample(real_all, min(len(real_all), 10000), CONFIG.seed)
            n_sel = min(3000, len(pool))
            idx = k_center_greedy(pool, n_sel, seed=CONFIG.seed)
            all_coreset_chunks.append(pool[idx])

    # 2. Process Simulated Healthy Road Domain
    sim_images = find_images(args.sim_healthy_dir)[: args.max_images_per_domain]
    if sim_images:
        log.info("Extracting patch embeddings for SIM domain (%d images)...", len(sim_images))
        sim_patches = []
        for p in sim_images:
            try:
                rgb = load_rgb(p)
                mask = masker.get_road_mask(rgb)
                emb, _ = embedder.extract_road_patch_embeddings(rgb, mask)
                if len(emb):
                    sim_patches.append(emb)
            except Exception as e:
                log.warning("Skipping %s: %s", p.name, e)
        if sim_patches:
            sim_all = np.concatenate(sim_patches, axis=0).astype(np.float32)
            dist_model.add_domain_cluster("sim_asphalt", sim_all)
            pool = random_presample(sim_all, min(len(sim_all), 10000), CONFIG.seed)
            n_sel = min(3000, len(pool))
            idx = k_center_greedy(pool, n_sel, seed=CONFIG.seed)
            all_coreset_chunks.append(pool[idx])

    # Save distribution model
    dist_model.save(out / "distribution_model.json")

    # Combine coresets into FAISS index
    if all_coreset_chunks:
        unified_coreset = np.concatenate(all_coreset_chunks, axis=0).astype(np.float32)
        faiss.normalize_L2(unified_coreset)
        index = faiss.IndexFlatIP(unified_coreset.shape[1])
        index.add(unified_coreset)
        faiss.write_index(index, str(out / "index.faiss"))
        np.save(out / "embeddings.npy", unified_coreset)

        metadata = {
            "embedding_dim": int(unified_coreset.shape[1]),
            "num_memory_embeddings": int(len(unified_coreset)),
            "model": CONFIG.dinov2_model_name,
            "patch_size": CONFIG.patch_size,
            "clusters": [c.name for c in dist_model.clusters],
            "real_images": len(real_images),
            "sim_images": len(sim_images),
            "faiss_metric": "inner_product_after_L2_normalization",
        }
        save_json(metadata, out / "metadata.json")
        log.info("Multi-domain distribution bank successfully created at %s", out)


if __name__ == "__main__":
    main()
