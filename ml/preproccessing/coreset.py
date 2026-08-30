"""
Greedy k-center coreset subsampling (the same idea PatchCore uses).

Why: if you keep every road patch embedding from every healthy-road image, the
memory bank can easily reach millions of vectors -- too big to ship to and search
on an 8GB Pi. Coreset subsampling picks a small, spread-out subset that still
covers the diversity of "normal road" well, by repeatedly picking whichever
remaining point is farthest from everything already picked.

This is O(n * k) with numpy, which is fine up to a few hundred thousand points.
If your dataset produces millions of patch embeddings, subsample randomly down to
~200k first, then run this on top.
"""

import numpy as np


def k_center_greedy(embeddings: np.ndarray, n_select: int, seed: int = 42) -> np.ndarray:
    """
    embeddings: (N, C) float32 array
    n_select: number of points to keep
    Returns: indices (n_select,) into `embeddings` of the selected coreset
    """
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    if n_select >= n:
        return np.arange(n)

    selected = [int(rng.integers(0, n))]
    # distance from every point to the nearest already-selected point
    min_dist = np.linalg.norm(embeddings - embeddings[selected[0]], axis=1)

    for _ in range(n_select - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        new_dist = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return np.array(selected, dtype=np.int64)
