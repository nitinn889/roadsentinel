"""
dinov2_features.py

Loads a frozen, pretrained DINOv2 model and extracts DENSE PATCH-LEVEL features
from road images (not just a single global embedding) so we can later localize
WHERE an anomaly is, not just whether one exists in the image.

DINOv2 splits an image into a grid of fixed-size patches (patch_size=14 for the
standard models) and outputs one feature vector per patch, plus one extra "CLS"
token representing the whole image. We strip the CLS token and reshape the
remaining patch tokens into a 2D grid of feature vectors matching the image's
spatial layout.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# facebook/dinov2-small is the right pick here: DINOv2 comes in small/base/large/
# giant variants. For near-real-time use on a laptop GPU (and eventual edge
# deployment on Jetson/Pi), "small" gives strong general-purpose features at a
# fraction of the compute/memory cost of base or large. We can upgrade to
# dinov2-base later if accuracy turns out to be the bottleneck rather than speed.
DEFAULT_MODEL_NAME = "facebook/dinov2-small"


class DinoV2FeatureExtractor:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DinoV2FeatureExtractor] Loading '{model_name}' on device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # frozen, inference-only — we never train DINOv2 itself

        # DINOv2's patch size (14x14 pixels per patch) — needed to compute the
        # resulting patch grid dimensions from the input image size.
        self.patch_size = self.model.config.patch_size

    @torch.no_grad()
    def extract_dense_features(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Given a PIL image, returns:
          - patch_features: np.ndarray of shape (grid_h, grid_w, hidden_dim)
          - grid_size: (grid_h, grid_w) — how many patches tall/wide the image was
            split into, needed later to upsample scores back to full image size.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"]  # shape (1, 3, H, W)

        _, _, H, W = pixel_values.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state shape: (1, num_patches + 1, hidden_dim)
        # token 0 is the CLS token (whole-image summary) — we drop it since we
        # want per-patch, spatially-localized features instead.
        hidden_states = outputs.last_hidden_state[0]  # (num_patches + 1, hidden_dim)
        patch_tokens = hidden_states[1:]  # drop CLS -> (num_patches, hidden_dim)

        num_patches, hidden_dim = patch_tokens.shape
        expected = grid_h * grid_w
        if num_patches != expected:
            # Some DINOv2 configs add register tokens; trim/pad defensively so
            # this doesn't silently break on a slightly different model variant.
            patch_tokens = patch_tokens[:expected]

        patch_features = patch_tokens.reshape(grid_h, grid_w, hidden_dim)
        return patch_features.cpu().numpy(), (grid_h, grid_w)


if __name__ == "__main__":
    # Quick smoke test with a synthetic image, just to confirm shapes are sane.
    extractor = DinoV2FeatureExtractor()
    dummy_image = Image.new("RGB", (518, 518), color=(120, 120, 120))
    features, grid_size = extractor.extract_dense_features(dummy_image)
    print(f"Patch feature grid shape: {features.shape} (grid_size={grid_size})")
