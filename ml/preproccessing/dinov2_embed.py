"""
DINOv2 patch-token feature extraction.

DINOv2 already splits its input into 14x14-pixel patches internally and returns one
embedding vector per patch (the "patch tokens"). That's the local feature granularity
we want for anomaly detection -- there's no need to manually crop the image into tiles
first. We just need to:
  1. resize the image to a fixed size that's a multiple of 14,
  2. normalize with ImageNet stats,
  3. run it through DINOv2,
  4. keep only the patch tokens whose corresponding image region is road (per SAM2 mask).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Dinov2Embedder:
    def __init__(self, device: str = config.DEVICE):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", config.DINOV2_MODEL_NAME)
        self.model.eval().to(device)
        self.grid = config.DINOV2_INPUT_SIZE // config.PATCH_SIZE  # patches per side

    @torch.no_grad()
    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            image_rgb,
            (config.DINOV2_INPUT_SIZE, config.DINOV2_INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t.to(self.device)

    def _resize_mask_to_patch_grid(self, mask: np.ndarray) -> np.ndarray:
        # Downsample the pixel-level road mask to one bool per DINOv2 patch.
        # A patch counts as "road" if the majority of its pixels are road.
        mask_f = mask.astype(np.float32)
        small = cv2.resize(mask_f, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
        return small > 0.5

    @torch.no_grad()
    def extract_road_patch_embeddings(self, image_rgb: np.ndarray, road_mask: np.ndarray):
        """
        Returns:
          embeddings: (N_road_patches, C) float32 numpy array
          patch_coords: (N_road_patches, 2) int array of (row, col) in the patch grid,
                        kept so we can later map an anomaly back to a pixel region.
        """
        x = self._preprocess(image_rgb)
        out = self.model.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"][0]  # (grid*grid, C)
        patch_tokens = patch_tokens.reshape(self.grid, self.grid, -1)  # (grid, grid, C)

        patch_mask = self._resize_mask_to_patch_grid(road_mask)  # (grid, grid) bool
        rows, cols = np.where(patch_mask)

        if len(rows) == 0:
            return np.empty((0, patch_tokens.shape[-1]), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

        selected = patch_tokens[rows, cols, :].cpu().numpy().astype(np.float32)
        coords = np.stack([rows, cols], axis=1).astype(np.int32)
        return selected, coords
