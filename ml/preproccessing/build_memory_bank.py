"""
STEP 1 (offline, one-time): build the healthy-road memory bank.

Run this on a machine with a GPU, not on the Pi. It walks the healthy-roads
dataset, segments the road with SAM2, extracts DINOv2 patch embeddings for the
road region, reduces everything to a compact coreset, and writes out the
artifacts you'll later copy to the Pi:

  output/memory_bank/
    embeddings.npy      -- (N, C) float32, the coreset memory bank
    index.faiss          -- FAISS flat L2 index over embeddings.npy, for fast NN lookup
    metadata.json         -- config used, counts, embedding dim, per-source stats

Usage:
    python build_memory_bank.py
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from coreset import k_center_greedy
from dinov2_embed import Dinov2Embedder
from sam2_mask import RoadMasker

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def main():
    t0 = time.time()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(config.HEALTHY_ROADS_DIR)
    if not image_paths:
        raise SystemExit(f"No images found under {config.HEALTHY_ROADS_DIR}")
    print(f"Found {len(image_paths)} healthy-road images.")

    masker = RoadMasker(device=config.DEVICE)
    embedder = Dinov2Embedder(device=config.DEVICE)

    all_embeddings = []
    per_image_counts = {}
    skipped = []

    for path in tqdm(image_paths, desc="Extracting road patch embeddings"):
        try:
            image_rgb = load_rgb(path)
            road_mask = masker.get_road_mask(image_rgb)

            if road_mask.sum() == 0:
                skipped.append(str(path))
                continue

            embeddings, _coords = embedder.extract_road_patch_embeddings(image_rgb, road_mask)
            if len(embeddings) == 0:
                skipped.append(str(path))
                continue

            all_embeddings.append(embeddings)
            per_image_counts[str(path)] = int(len(embeddings))
        except Exception as e:
            print(f"  [warn] failed on {path}: {e}")
            skipped.append(str(path))

    if not all_embeddings:
        raise SystemExit("No embeddings extracted -- check SAM2 masking and ROI settings.")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Extracted {all_embeddings.shape[0]} total road-patch embeddings "
          f"(dim={all_embeddings.shape[1]}) from {len(per_image_counts)} images.")
    if skipped:
        print(f"Skipped {len(skipped)} images (no road mask found or error).")

    n_select = min(
        config.CORESET_MAX_POINTS,
        max(1, int(all_embeddings.shape[0] * config.CORESET_RATIO)),
    )
    print(f"Running coreset subsampling -> keeping {n_select} points...")
    idx = k_center_greedy(all_embeddings, n_select, seed=config.SEED)
    coreset_embeddings = all_embeddings[idx].astype(np.float32)

    # Build a flat L2 FAISS index. Flat = exact search; fine at this scale (<=20k points)
    # and simplest to run on the Pi later with no extra index-tuning.
    dim = coreset_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(coreset_embeddings)

    np.save(config.OUTPUT_DIR / "embeddings.npy", coreset_embeddings)
    faiss.write_index(index, str(config.OUTPUT_DIR / "index.faiss"))

    metadata = {
        "dinov2_model": config.DINOV2_MODEL_NAME,
        "patch_size": config.PATCH_SIZE,
        "dinov2_input_size": config.DINOV2_INPUT_SIZE,
        "roi_box_fractions": config.ROI_BOX_FRACTIONS,
        "embedding_dim": dim,
        "n_source_images": len(per_image_counts),
        "n_skipped_images": len(skipped),
        "n_total_patch_embeddings": int(all_embeddings.shape[0]),
        "n_coreset_points": int(coreset_embeddings.shape[0]),
        "coreset_ratio_config": config.CORESET_RATIO,
        "coreset_max_points_config": config.CORESET_MAX_POINTS,
        "build_time_seconds": round(time.time() - t0, 1),
    }
    with open(config.OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone in {metadata['build_time_seconds']}s.")
    print(f"Memory bank written to: {config.OUTPUT_DIR.resolve()}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
