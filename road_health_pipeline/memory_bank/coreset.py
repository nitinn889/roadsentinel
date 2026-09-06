from __future__ import annotations

import math
import numpy as np


def random_presample(x: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    if len(x) <= max_points:
        return np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return np.asarray(x[idx], dtype=np.float32)


def _pairwise_min_sqdist_blockwise(points: np.ndarray, centers: np.ndarray, block_size: int) -> np.ndarray:
    """Return min squared L2 distance to any center without materializing NxK."""
    mins = np.full(len(points), np.inf, dtype=np.float32)
    c = centers.astype(np.float32, copy=False)
    c_norm = (c * c).sum(axis=1)
    for start in range(0, len(points), block_size):
        stop = min(start + block_size, len(points))
        p = points[start:stop].astype(np.float32, copy=False)
        p_norm = (p * p).sum(axis=1, keepdims=True)
        d = p_norm + c_norm[None, :] - 2.0 * (p @ c.T)
        mins[start:stop] = np.minimum(mins[start:stop], d.min(axis=1))
    return mins


def k_center_greedy(points: np.ndarray, n_select: int, seed: int = 42,
                     block_size: int = 8192) -> np.ndarray:
    """Memory-conscious farthest-point/k-center selection on a presampled pool.

    Complexity is O(N*K) but never creates the full NxK distance matrix.
    Intended for a bounded presample, not the original multi-million point set.
    """
    points = np.asarray(points, dtype=np.float32)
    n_select = min(int(n_select), len(points))
    if n_select <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = np.empty(n_select, dtype=np.int64)
    selected[0] = rng.integers(0, len(points))

    min_dist = np.sum((points - points[selected[0]]) ** 2, axis=1).astype(np.float32)
    min_dist[selected[0]] = -np.inf

    for i in range(1, n_select):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx
        if i == n_select - 1:
            break
        center = points[next_idx]
        for start in range(0, len(points), block_size):
            stop = min(start + block_size, len(points))
            p = points[start:stop]
            d = np.sum((p - center) ** 2, axis=1)
            min_dist[start:stop] = np.minimum(min_dist[start:stop], d)
        min_dist[selected[: i + 1]] = -np.inf
    return selected
