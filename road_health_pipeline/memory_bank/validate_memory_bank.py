from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CONFIG


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate a RoadSentinel healthy-road memory bank."
    )
    ap.add_argument("--dir", type=Path, default=CONFIG.memory_bank_dir)
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress JSON output; exit 0 on success, non-zero on failure.",
    )
    args = ap.parse_args()
    required = [args.dir / "embeddings.npy", args.dir / "index.faiss", args.dir / "metadata.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("INVALID: missing " + ", ".join(missing))
    emb = np.load(required[0], mmap_mode="r")
    index = faiss.read_index(str(required[1]))
    meta = json.loads(required[2].read_text())
    assert emb.ndim == 2, f"embeddings.npy must be 2-D, got {emb.ndim}-D"
    assert emb.shape[0] == index.ntotal, (
        f"embeddings.npy rows ({emb.shape[0]}) != faiss ntotal ({index.ntotal})"
    )
    assert emb.shape[1] == index.d, (
        f"embeddings.npy dim ({emb.shape[1]}) != faiss dim ({index.d})"
    )
    assert meta["num_memory_embeddings"] == index.ntotal, (
        f"metadata num_memory_embeddings ({meta['num_memory_embeddings']}) "
        f"!= faiss ntotal ({index.ntotal})"
    )
    assert np.isfinite(emb[: min(1000, len(emb))]).all(), "Non-finite values in embeddings.npy"
    result = {"valid": True, "shape": list(emb.shape), "faiss_ntotal": index.ntotal, "metadata": meta}
    if not args.quiet:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
