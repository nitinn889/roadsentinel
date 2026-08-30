"""DINOv2 feature extraction for RoadSentinel.

Model reference
---------------
Model        : dinov2_vits14  (ViT-Small, patch size 14)
Architecture : Vision Transformer (ViT-S/14), facebookresearch/dinov2
Feature dim  : 384
Patch size   : 14 px
Input size   : 518 px  (configurable; must be divisible by patch_size)
Patch grid   : 518 // 14 = 37  →  37 × 37 = 1369 patch tokens per image
Normalisation: ImageNet mean/std  (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)

Patch-to-pixel correspondence
------------------------------
Token at grid position (r, c) covers the following pixel rectangle in the
resized 518×518 input:

    y_start = r * patch_size
    y_end   = y_start + patch_size
    x_start = c * patch_size
    x_end   = x_start + patch_size

After the anomaly map is upsampled back to original image dimensions via
cv2.resize, spatial correspondence is preserved by the bilinear interpolation.

Token layout in DINOv2 forward_features output
------------------------------------------------
out["x_norm_patchtokens"]  shape (1, grid*grid, dim)  — patch tokens (normalised)
out["x_norm_clstoken"]     shape (1, dim)             — CLS global token (normalised)
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import CONFIG

log = logging.getLogger(__name__)

# Module-level singleton cache keyed by (model_name, device_str)
_INSTANCES: dict[tuple[str, str], "Dinov2Embedder"] = {}


class Dinov2Embedder:
    """Load DINOv2 once and reuse it for many images.

    Preferred usage — load once, process many images:

        embedder = Dinov2Embedder.from_config()   # singleton per (model, device)
        grid = embedder.extract_patch_grid(rgb)   # shape (37, 37, 384)
        cls  = embedder.extract_cls_token(rgb)    # shape (384,)

    The constructor is still public for cases where an independent instance is
    needed (e.g. tests that patch the model object).
    """

    def __init__(
        self,
        device: str = CONFIG.device,
        model_name: str = CONFIG.dinov2_model_name,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning(
                "CUDA requested for DINOv2 but not available; falling back to CPU."
            )
            device = "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.patch_size: int = CONFIG.patch_size
        self.input_size: int = CONFIG.dinov2_input_size
        self.grid_size: int = self.input_size // self.patch_size  # 37 for vits14@518

        log.info(
            "Loading DINOv2 %s on %s (grid %dx%d, dim=%s)",
            model_name,
            self.device,
            self.grid_size,
            self.grid_size,
            "384",
        )
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        self.model.eval().to(self.device)

        self.tf = transforms.Compose(
            [
                transforms.Resize(
                    (self.input_size, self.input_size), antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        device: str = CONFIG.device,
        model_name: str = CONFIG.dinov2_model_name,
    ) -> "Dinov2Embedder":
        """Return a cached instance for the given (model_name, device) pair.

        Calling this multiple times with the same arguments will reuse the
        already-loaded model, avoiding repeated hub downloads and GPU memory
        allocations.
        """
        # Normalise device string so "cuda" and "cuda:0" hash the same way.
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        key = (model_name, device)
        if key not in _INSTANCES:
            _INSTANCES[key] = cls(device=device, model_name=model_name)
        return _INSTANCES[key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Convert HxWx3 uint8 RGB array to a normalised (1, 3, H, W) tensor."""
        pil = Image.fromarray(image_rgb)
        return self.tf(pil).unsqueeze(0).to(self.device)

    def _autocast(self):
        """Return appropriate autocast context (CUDA fp16 or CPU no-op)."""
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _forward_features(self, x: torch.Tensor) -> dict:
        with torch.no_grad(), self._autocast():
            return self.model.forward_features(x)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_patch_grid(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract the full patch-token grid for an image.

        Parameters
        ----------
        image_rgb:
            HxWx3 uint8 RGB image.

        Returns
        -------
        np.ndarray
            Shape ``(grid_size, grid_size, embed_dim)`` float32 array.
            For dinov2_vits14 with input_size=518: ``(37, 37, 384)``.

            Token at ``[r, c]`` corresponds to the patch covering pixels
            ``[r*14:(r+1)*14, c*14:(c+1)*14]`` in the 518×518 resized input.
        """
        x = self._prepare_tensor(image_rgb)
        out = self._forward_features(x)
        tokens = out["x_norm_patchtokens"]  # (1, grid*grid, dim)
        g = self.grid_size
        return tokens.reshape(1, g, g, -1)[0].float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_cls_token(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract the global CLS token for an image.

        Returns
        -------
        np.ndarray
            Shape ``(embed_dim,)`` float32.  Represents the global image-level
            representation from the [CLS] token after the final transformer block.
        """
        x = self._prepare_tensor(image_rgb)
        out = self._forward_features(x)
        return out["x_norm_clstoken"][0].float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_road_patch_embeddings(
        self,
        image_rgb: np.ndarray,
        road_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract patch embeddings for pixels inside ``road_mask``.

        Parameters
        ----------
        image_rgb:
            HxWx3 uint8 RGB image.
        road_mask:
            HxW boolean or float32 mask.  Patches with more than
            ``CONFIG.road_patch_fraction`` of their area inside the mask are kept.

        Returns
        -------
        selected : np.ndarray
            Shape ``(N, embed_dim)`` float32 — embeddings for road patches.
        coords : np.ndarray
            Shape ``(N, 2)`` int16 — (row, col) grid indices for each selected patch.
            Use these to map back to pixel coordinates via ``coord * patch_size``.
        """
        x = self._prepare_tensor(image_rgb)
        out = self._forward_features(x)
        tokens = out["x_norm_patchtokens"]  # (1, grid*grid, dim)
        g = self.grid_size
        emb = tokens.reshape(1, g, g, -1)[0]  # (g, g, dim)

        mask_t = torch.from_numpy(road_mask.astype(np.float32))[None, None]
        mask_grid = F.interpolate(mask_t, size=(g, g), mode="area")[0, 0]
        keep = mask_grid > CONFIG.road_patch_fraction

        coords = torch.nonzero(keep, as_tuple=False).cpu().numpy().astype(np.int16)
        selected = emb[keep].float().cpu().numpy().astype(np.float32)
        return selected, coords
