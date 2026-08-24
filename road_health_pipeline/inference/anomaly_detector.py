from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import faiss

from config import CONFIG


class AnomalyDetector:
    def __init__(self, memory_bank_dir: Path = CONFIG.memory_bank_dir):
        self.memory_bank_dir = Path(memory_bank_dir)
        self.index = faiss.read_index(str(self.memory_bank_dir / "index.faiss"))
        self.embeddings = np.load(self.memory_bank_dir / "embeddings.npy", mmap_mode="r")
        self.metadata = json.loads((self.memory_bank_dir / "metadata.json").read_text())
        if self.embeddings.shape[1] != self.index.d:
            raise ValueError("FAISS dimension does not match embeddings.npy")
        self.threshold = None

    def score_patches(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.empty(0, dtype=np.float32)
        q = embeddings.astype(np.float32, copy=True)
        faiss.normalize_L2(q)
        similarity, _ = self.index.search(q, CONFIG.knn_k)
        # Convert cosine similarity to anomaly-like distance. Higher = more unusual.
        score = 1.0 - similarity.mean(axis=1)
        return score.astype(np.float32)

    def build_anomaly_map(self, patch_coords: np.ndarray, scores: np.ndarray, image_shape: tuple[int, int], grid_size: int) -> np.ndarray:
        h, w = image_shape
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        for (r, c), s in zip(patch_coords, scores):
            grid[int(r), int(c)] = max(grid[int(r), int(c)], float(s))
        import cv2
        amap = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.maximum(amap, 0)

    def summarize(self, scores: np.ndarray) -> tuple[float, float]:
        if len(scores) == 0:
            return 0.0, 0.0
        image_score = float(np.percentile(scores, CONFIG.anomaly_percentile))
        threshold = float(np.percentile(scores, 75))
        return image_score, threshold
