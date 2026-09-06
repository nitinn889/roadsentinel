"""
anomaly_detector.py

Trains an anomaly detector using ONLY images of healthy, undamaged road — no
labeled pothole dataset required. Any new road patch whose DINOv2 features
deviate significantly from what "healthy road" looks like gets a high anomaly
score. This is what later catches potholes, cracks, AND water patches, all
with the same mechanism.

METHOD CHOSEN: a k-nearest-neighbor "memory bank" approach, in the spirit of
PatchCore (Roth et al., 2022). Why this fits our situation well:
  - We likely only have a small number of healthy-road images to start with —
    this method needs no training/backprop at all, just collecting reference
    patch features, so it works fine even with a modest dataset.
  - It's simple to reason about and debug: the anomaly score for a new patch
    is literally "how far is this from the closest thing I've seen in healthy
    road" — easy to sanity-check by eye.
  - It naturally produces a CONTINUOUS score (distance), not just a binary
    flag — which we specifically want, since later in the project we reuse
    this same continuous score as a general "road health" gauge, not just a
    pothole/not-pothole switch.

Alternative considered: fitting a per-patch-position Gaussian (mean + covariance)
and using Mahalanobis distance. This assumes patch statistics are roughly
Gaussian and that the SAME patch position across images is comparable, which
doesn't hold well here since road images won't be perfectly aligned frame to
frame. The memory-bank/kNN approach avoids that assumption entirely.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from dinov2_features import DinoV2FeatureExtractor


class RoadAnomalyDetector:
    def __init__(self, feature_extractor: DinoV2FeatureExtractor, k_neighbors: int = 3):
        self.extractor = feature_extractor
        self.k_neighbors = k_neighbors
        self.memory_bank: np.ndarray | None = None  # shape (N, hidden_dim)
        self.nn_index: NearestNeighbors | None = None

    # ---------- Building the healthy-road reference ----------

    def build_reference(self, healthy_image_paths: list[Path], max_patches_per_image: int | None = None):
        """
        Extracts dense DINOv2 features from every healthy-road image and stores
        all patch feature vectors as the memory bank. Optionally subsamples
        patches per image to keep the memory bank a manageable size if you have
        a lot of healthy images.
        """
        all_patches = []
        for path in healthy_image_paths:
            image = Image.open(path).convert("RGB")
            features, grid_size = self.extractor.extract_dense_features(image)
            grid_h, grid_w, hidden_dim = features.shape
            flat = features.reshape(-1, hidden_dim)

            if max_patches_per_image is not None and flat.shape[0] > max_patches_per_image:
                idx = np.random.choice(flat.shape[0], max_patches_per_image, replace=False)
                flat = flat[idx]

            all_patches.append(flat)
            print(f"  [reference] {path.name}: grid {grid_size}, {flat.shape[0]} patches added")

        self.memory_bank = np.concatenate(all_patches, axis=0)
        self._fit_nn_index()
        print(f"[RoadAnomalyDetector] Memory bank built: {self.memory_bank.shape[0]} reference patches "
              f"(dim={self.memory_bank.shape[1]})")

    def _fit_nn_index(self):
        n_neighbors = min(self.k_neighbors, self.memory_bank.shape[0])
        self.nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.nn_index.fit(self.memory_bank)

    # ---------- Save / load, so we don't rebuild the reference every run ----------

    def save_reference(self, path: Path):
        np.save(path, self.memory_bank)
        print(f"[RoadAnomalyDetector] Saved reference memory bank to {path}")

    def load_reference(self, path: Path):
        self.memory_bank = np.load(path)
        self._fit_nn_index()
        print(f"[RoadAnomalyDetector] Loaded reference memory bank from {path} "
              f"({self.memory_bank.shape[0]} patches)")

    # ---------- Scoring a new image ----------

    def score_image(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          - patch_scores: (grid_h, grid_w) anomaly score per patch (mean distance
            to k nearest neighbors in the healthy-road memory bank)
          - heatmap: the same scores upsampled (bicubic) to the original image's
            pixel resolution, normalized to 0-1, ready to overlay/visualize.
        """
        if self.memory_bank is None or self.nn_index is None:
            raise RuntimeError("No reference loaded — call build_reference() or load_reference() first.")

        features, (grid_h, grid_w) = self.extractor.extract_dense_features(image)
        hidden_dim = features.shape[-1]
        flat = features.reshape(-1, hidden_dim)

        distances, _ = self.nn_index.kneighbors(flat)  # (num_patches, k)
        patch_scores = distances.mean(axis=1).reshape(grid_h, grid_w)

        # Upsample the coarse patch-grid score map to full image resolution so it
        # can be overlaid pixel-for-pixel on the original photo.
        img_w, img_h = image.size
        heatmap = cv2.resize(patch_scores.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_CUBIC)

        # Normalize to 0-1 for easy visualization/thresholding. Note: this is a
        # PER-IMAGE normalization for display purposes only — for the health-
        # trend tracking used later in the project, the RAW (un-normalized)
        # patch_scores should be stored instead, since normalizing per-image
        # would make scores incomparable across different visits/photos.
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return patch_scores, heatmap_norm


def overlay_heatmap_on_image(image: Image.Image, heatmap_norm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends a normalized (0-1) heatmap onto the original image using a JET colormap."""
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return blended


if __name__ == "__main__":
    # Smoke test using synthetic images (no real road photos available yet in
    # this environment) — this only validates that shapes/save/load/scoring
    # logic all run correctly end to end, not real detection accuracy.
    extractor = DinoV2FeatureExtractor()
    detector = RoadAnomalyDetector(extractor, k_neighbors=3)

    tmp_dir = Path("/tmp/dummy_healthy_road")
    tmp_dir.mkdir(exist_ok=True)
    for i in range(3):
        img = Image.new("RGB", (518, 518), color=(100 + i * 5, 100, 90))
        img.save(tmp_dir / f"healthy_{i}.jpg")

    healthy_paths = list(tmp_dir.glob("*.jpg"))
    detector.build_reference(healthy_paths)

    ref_path = Path("/tmp/reference_memory_bank.npy")
    detector.save_reference(ref_path)

    detector2 = RoadAnomalyDetector(extractor, k_neighbors=3)
    detector2.load_reference(ref_path)

    test_img = Image.new("RGB", (518, 518), color=(100, 100, 90))
    patch_scores, heatmap = detector2.score_image(test_img)
    print(f"patch_scores shape: {patch_scores.shape}, heatmap shape: {heatmap.shape}")
    print(f"heatmap min/max: {heatmap.min():.3f}/{heatmap.max():.3f}")
