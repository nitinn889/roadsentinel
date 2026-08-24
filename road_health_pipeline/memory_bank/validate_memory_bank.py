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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=CONFIG.memory_bank_dir)
    args = ap.parse_args()
    required = [args.dir / "embeddings.npy", args.dir / "index.faiss", args.dir / "metadata.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("INVALID: missing " + ", ".join(missing))
    emb = np.load(required[0], mmap_mode="r")
    index = faiss.read_index(str(required[1]))
    meta = json.loads(required[2].read_text())
    assert emb.ndim == 2
    assert emb.shape[0] == index.ntotal
    assert emb.shape[1] == index.d
    assert meta["num_memory_embeddings"] == index.ntotal
    assert np.isfinite(emb[: min(1000, len(emb))]).all()
    print(json.dumps({"valid": True, "shape": list(emb.shape), "faiss_ntotal": index.ntotal, "metadata": meta}, indent=2))


if __name__ == "__main__":
    main()
