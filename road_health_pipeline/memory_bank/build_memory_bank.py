from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json
import sys

import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import find_images, load_rgb, save_json
from inference.sam2_mask import RoadMasker
from inference.dinov2_embed import Dinov2Embedder
from memory_bank.coreset import random_presample, k_center_greedy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build_memory_bank")


def require_cuda(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "ROADSENTINEL_DEVICE requests CUDA, but this PyTorch installation has no CUDA support. "
            "Install a CUDA-enabled PyTorch build on the GPU machine; do not silently fall back to CPU."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy-dir", type=Path, default=CONFIG.healthy_roads_dir)
    ap.add_argument("--output-dir", type=Path, default=CONFIG.memory_bank_dir)
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--coreset-pool", type=int, default=CONFIG.coreset_presample_max)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    require_cuda(args.device)
    images = find_images(args.healthy_dir)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found under {args.healthy_dir}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    log.info("Found %d healthy-road candidate images", len(images))
    log.info("Loading SAM2 on %s", args.device)
    masker = RoadMasker(device=args.device)
    log.info("Loading DINOv2 %s on %s", CONFIG.dinov2_model_name, args.device)
    embedder = Dinov2Embedder(device=args.device)

    chunks: list[np.ndarray] = []
    total = 0
    for i, path in enumerate(images, 1):
        try:
            rgb = load_rgb(path)
            road_mask = masker.get_road_mask(rgb)
            emb, _coords = embedder.extract_road_patch_embeddings(rgb, road_mask)
            if len(emb):
                chunks.append(emb.astype(np.float32, copy=False))
                total += len(emb)
        except Exception as exc:
            log.exception("Skipping %s: %s", path, exc)
        if i % 100 == 0 or i == len(images):
            log.info("Processed %d/%d images; collected %d patch embeddings", i, len(images), total)

    if not chunks:
        raise RuntimeError("No patch embeddings were produced")

    all_embeddings = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    np.save(out / "all_embeddings.npy", all_embeddings)

    n_select = min(CONFIG.coreset_max_points, max(1, int(len(all_embeddings) * CONFIG.coreset_ratio)))
    pool = random_presample(all_embeddings, args.coreset_pool, CONFIG.seed)
    log.info("Coreset: N=%d, presample=%d, final=%d", len(all_embeddings), len(pool), n_select)
    n_select = min(n_select, len(pool))
    idx = k_center_greedy(pool, n_select, seed=CONFIG.seed, block_size=CONFIG.coreset_block_size)
    coreset = pool[idx].astype(np.float32)

    faiss.normalize_L2(coreset)
    index = faiss.IndexFlatIP(coreset.shape[1])
    index.add(coreset)
    faiss.write_index(index, str(out / "index.faiss"))
    np.save(out / "embeddings.npy", coreset)

    # Keep calibration data deliberately simple; inference computes an image-level score from patch NN distances.
    metadata = {
        "embedding_dim": int(coreset.shape[1]),
        "num_source_images": len(images),
        "num_source_patch_embeddings": int(len(all_embeddings)),
        "num_memory_embeddings": int(len(coreset)),
        "model": CONFIG.dinov2_model_name,
        "patch_size": CONFIG.patch_size,
        "sam2_checkpoint": str(CONFIG.sam2_checkpoint),
        "sam2_model_cfg": CONFIG.sam2_model_cfg,
        "healthy_dataset_definition": "Images without XML damage annotations; not guaranteed perfectly healthy.",
        "faiss_metric": "inner_product_after_L2_normalization",
        "seed": CONFIG.seed,
    }
    save_json(metadata, out / "metadata.json")
    log.info("Memory bank created at %s", out)


if __name__ == "__main__":
    main()
