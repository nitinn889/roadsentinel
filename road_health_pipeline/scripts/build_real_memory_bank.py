#!/usr/bin/env python3
"""Build a production DINOv2 healthy-road memory bank from real datasets.

Default source:
  RoadSentinel_datasets/rdd2022/healthy_road/
  (or any directory containing clean, defect-free road images)

Output:
  output/real_memory_bank/
    ├── index.faiss
    ├── embeddings.npy
    └── metadata.json

Usage:
  python scripts/build_real_memory_bank.py [--healthy-dir PATH] [--output-dir PATH] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import find_images, load_rgb, utc_iso
from inference.dinov2_embed import Dinov2Embedder
from inference.sam2_mask import RoadMasker
from memory_bank.coreset import k_center_greedy, random_presample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_real_memory_bank")

DEFAULT_HEALTHY_DIR = ROOT.parent / "RoadSentinel_datasets" / "rdd2022" / "healthy_road"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "real_memory_bank"


def parse_args():
    ap = argparse.ArgumentParser(description="Build real healthy-road memory bank")
    ap.add_argument(
        "--healthy-dir",
        type=Path,
        default=DEFAULT_HEALTHY_DIR,
        help=f"Directory of healthy road images (default: {DEFAULT_HEALTHY_DIR})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    ap.add_argument(
        "--coreset-pool",
        type=int,
        default=50000,
        help="Max candidate embeddings to pool before k-center greedy selection",
    )
    ap.add_argument(
        "--coreset-final",
        type=int,
        default=10000,
        help="Final number of memory bank vectors to retain (default: 10000)",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional max images to process",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available; falling back to CPU")
        args.device = "cpu"

    log.info("=" * 65)
    log.info("RoadSentinel: Building Real DINOv2 Healthy Road Memory Bank")
    log.info("=" * 65)
    log.info("Healthy images dir : %s", args.healthy_dir)
    log.info("Output directory   : %s", args.output_dir)
    log.info("Device             : %s", args.device)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        log.info("GPU                : %s", torch.cuda.get_device_name(0))

    if not args.healthy_dir.exists():
        raise FileNotFoundError(f"Healthy images directory not found: {args.healthy_dir}")

    images = find_images(args.healthy_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.healthy_dir}")
    if args.max_images:
        images = images[: args.max_images]

    log.info("Found %d healthy road images to process", len(images))

    # Initialize models
    log.info("Loading SAM2 for road surface extraction...")
    masker = RoadMasker(device=args.device)

    log.info("Loading DINOv2 (%s)...", CONFIG.dinov2_model_name)
    embedder = Dinov2Embedder.from_config(device=args.device)

    # Process images and extract road patch features
    chunks: list[np.ndarray] = []
    total_patches = 0
    t0 = time.monotonic()

    for idx, img_path in enumerate(images, 1):
        try:
            rgb = load_rgb(img_path)
            road_mask = masker.get_road_mask(rgb)
            emb, _ = embedder.extract_road_patch_embeddings(rgb, road_mask)
            if len(emb) > 0:
                chunks.append(emb.astype(np.float32, copy=False))
                total_patches += len(emb)
        except Exception as e:
            log.warning("Failed on %s: %s", img_path.name, e)

        if idx % 10 == 0 or idx == len(images):
            elapsed = time.monotonic() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(images) - idx) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] %.1f img/s (ETA %.0fs) | Extracted %d total patch tokens",
                idx,
                len(images),
                rate,
                eta,
                total_patches,
            )

    if not chunks:
        raise RuntimeError("No patch embeddings were extracted from the input images.")

    all_emb = np.vstack(chunks)
    log.info("Extracted %d raw patch embeddings (dim=%d)", all_emb.shape[0], all_emb.shape[1])

    # Coreset selection to construct an efficient, non-redundant memory bank
    log.info("Performing coreset selection (pool=%d -> final=%d)...", args.coreset_pool, args.coreset_final)
    pool = random_presample(all_emb, max_points=args.coreset_pool, seed=args.seed)
    n_select = min(args.coreset_final, len(pool))
    idx = k_center_greedy(pool, n_select=n_select, seed=args.seed)
    bank_vectors = pool[idx].astype(np.float32)

    # Normalize vectors for Cosine Similarity search with FAISS Inner Product
    faiss.normalize_L2(bank_vectors)

    # Build FAISS Flat Inner Product index
    dim = bank_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(bank_vectors)

    # Save artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_file = args.output_dir / "index.faiss"
    emb_file = args.output_dir / "embeddings.npy"
    meta_file = args.output_dir / "metadata.json"

    faiss.write_index(index, str(index_file))
    np.save(str(emb_file), bank_vectors)

    metadata = {
        "embedding_dim": int(dim),
        "num_source_images": len(images),
        "num_source_patch_embeddings": int(total_patches),
        "num_memory_embeddings": int(bank_vectors.shape[0]),
        "model": CONFIG.dinov2_model_name,
        "patch_size": int(CONFIG.patch_size),
        "sam2_checkpoint": str(CONFIG.sam2_checkpoint),
        "healthy_dataset_source": str(args.healthy_dir),
        "faiss_metric": "inner_product_after_L2_normalization",
        "created_at": utc_iso(),
        "seed": args.seed,
    }
    meta_file.write_text(json.dumps(metadata, indent=2))

    log.info("=" * 65)
    log.info("✅ Successfully created Real Memory Bank:")
    log.info("   • Index:      %s (%d vectors)", index_file, index.ntotal)
    log.info("   • Embeddings: %s", emb_file)
    log.info("   • Metadata:   %s", meta_file)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
