"""Generate synthetic test fixtures for RoadSentinel unit tests.

Run once (from road_health_pipeline/):
    python tests/make_fixtures.py

Produces:
    tests/fixtures/healthy_mock/road_00.png … road_04.png
    tests/fixtures/pothole_mock/pothole_00.png

These are small synthetic images committed to git so that tests can run
without any real dataset.  They contain:
  - Realistic-ish road textures (gray gradient + Gaussian noise + grid lines)
  - A synthetic pothole (dark oval + subtle shadow + texture variation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEALTHY_DIR = ROOT / "tests" / "fixtures" / "healthy_mock"
POTHOLE_DIR = ROOT / "tests" / "fixtures" / "pothole_mock"

W, H = 512, 512


def _road_base(rng: np.random.Generator) -> np.ndarray:
    """Asphalt-like gray gradient + Perlin-ish noise."""
    # Base gray with slight vertical gradient (brighter at centre)
    base = np.full((H, W, 3), 105, dtype=np.float32)
    grad = np.linspace(0, 20, H, dtype=np.float32)[:, None]
    base[:, :, 0] += grad
    base[:, :, 1] += grad
    base[:, :, 2] += grad

    # Large-scale noise (road texture variation)
    noise_large = rng.normal(0, 8, (H // 8, W // 8)).astype(np.float32)
    noise_large = cv2.resize(noise_large, (W, H), interpolation=cv2.INTER_CUBIC)
    # Fine-grain noise
    noise_fine = rng.normal(0, 3, (H, W)).astype(np.float32)

    for ch in range(3):
        base[:, :, ch] += noise_large + noise_fine

    return np.clip(base, 0, 255).astype(np.uint8)


def _add_lane_markings(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a dashed centre line and edge lines."""
    img = img.copy()
    cx = W // 2
    # Centre dashed line
    for y in range(0, H, 60):
        if rng.random() > 0.3:
            cv2.line(img, (cx, y), (cx, min(y + 40, H - 1)), (210, 210, 200), 3)
    # Edge lines
    cv2.line(img, (30, 0), (30, H - 1), (200, 200, 190), 2)
    cv2.line(img, (W - 30, 0), (W - 30, H - 1), (200, 200, 190), 2)
    return img


def _add_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a soft shadow region (simulates tree shadow etc.)."""
    img = img.astype(np.float32)
    x0 = int(rng.integers(0, W // 2))
    x1 = x0 + int(rng.integers(80, 200))
    shadow_mask = np.zeros((H, W), dtype=np.float32)
    shadow_mask[:, x0:x1] = rng.uniform(0.12, 0.25)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (61, 61), 20)
    for ch in range(3):
        img[:, :, ch] -= img[:, :, ch] * shadow_mask
    return np.clip(img, 0, 255).astype(np.uint8)


def make_healthy(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = _road_base(rng)
    img = _add_lane_markings(img, rng)
    if rng.random() > 0.5:
        img = _add_shadow(img, rng)
    return img


def make_pothole(seed: int = 999) -> np.ndarray:
    """Road with a synthetic pothole: dark irregular oval + shadow halo."""
    rng = np.random.default_rng(seed)
    img = _road_base(rng).astype(np.float32)
    img = _add_lane_markings(img.astype(np.uint8), rng).astype(np.float32)

    # Pothole centre and size
    cx, cy = int(rng.integers(W // 4, 3 * W // 4)), int(rng.integers(H // 4, 3 * H // 4))
    rx, ry = int(rng.integers(25, 55)), int(rng.integers(20, 45))

    # Create irregular oval mask by warping with noise
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    # Add irregular boundary by adding noise to the distance field
    boundary_noise = cv2.GaussianBlur(rng.normal(0, 0.15, (H, W)).astype(np.float32), (21, 21), 8)
    dist_noisy = dist + boundary_noise

    pothole_mask = dist_noisy < 1.0

    # Shadow halo around the pothole
    halo_mask = (dist_noisy < 1.4) & ~pothole_mask
    halo_factor = np.where(halo_mask, 0.88, 1.0)

    # Pothole interior: darker, rougher texture
    pothole_interior = rng.normal(50, 12, (H, W)).astype(np.float32)
    for ch in range(3):
        img[:, :, ch] = np.where(
            pothole_mask,
            np.clip(pothole_interior + rng.normal(0, 3, (H, W)), 20, 80),
            img[:, :, ch] * halo_factor,
        )

    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    HEALTHY_DIR.mkdir(parents=True, exist_ok=True)
    POTHOLE_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        img = make_healthy(seed=i * 17 + 42)
        path = HEALTHY_DIR / f"road_{i:02d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  Written {path}")

    pth = POTHOLE_DIR / "pothole_00.png"
    img = make_pothole(seed=999)
    cv2.imwrite(str(pth), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  Written {pth}")

    print("Fixtures generated.")


if __name__ == "__main__":
    sys.exit(main())
