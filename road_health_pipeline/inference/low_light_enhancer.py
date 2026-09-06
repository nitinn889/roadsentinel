"""
low_light_enhancer.py
---------------------
RoadSentinel: Adaptive Low-Light Restoration & Dynamic Range Preprocessing.

Enhances underexposed, night, and dusk road drone captures so that vision
transformers (DINOv2) and foundation segmentation models (SAM 2) receive
balanced contrast, visible asphalt aggregate textures, and distinct crater rims.
"""

from __future__ import annotations

import cv2
import numpy as np


class LowLightEnhancer:
    """Adaptive low-light illumination normalizer and contrast enhancer."""

    def __init__(
        self,
        low_light_mean_threshold: float = 65.0,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
        gamma: float = 0.55,
        apply_bilateral: bool = True,
    ) -> None:
        self.threshold = low_light_mean_threshold
        self.clip_limit = clahe_clip_limit
        self.tile_grid = clahe_tile_grid_size
        self.gamma = gamma
        self.apply_bilateral = apply_bilateral

    def is_low_light(self, rgb: np.ndarray) -> bool:
        """Determines if the image has low average luminance."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray)) < self.threshold

    def enhance(self, rgb: np.ndarray, force: bool = False) -> tuple[np.ndarray, bool]:
        """
        Enhances the input RGB image if it is underexposed/low-light.

        Returns
        -------
        (enhanced_rgb, was_enhanced)
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mean_val = float(np.mean(gray))

        if not force and mean_val >= self.threshold:
            return rgb.copy(), False

        # 1. LAB Color Space CLAHE (Contrast Limited Adaptive Histogram Equalization) on L-channel
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid
        )
        l_clahe = clahe.apply(l_channel)

        # 2. Adaptive Gamma Correction for shadow lifting
        # More aggressive gamma for extreme darkness (mean < 30)
        eff_gamma = self.gamma if mean_val > 30.0 else max(0.40, self.gamma * 0.85)
        inv_gamma = 1.0 / eff_gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        l_lifted = cv2.LUT(l_clahe, table)

        # Recombine LAB channels
        lab_enhanced = cv2.merge((l_lifted, a_channel, b_channel))
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # 3. Bilateral Edge-Preserving Denoising (smoothes low-light sensor noise while keeping asphalt edges sharp)
        if self.apply_bilateral:
            rgb_enhanced = cv2.bilateralFilter(rgb_enhanced, d=5, sigmaColor=35, sigmaSpace=35)

        return rgb_enhanced, True
