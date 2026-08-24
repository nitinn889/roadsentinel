from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import CONFIG


class Dinov2Embedder:
    def __init__(self, device: str = CONFIG.device, model_name: str = CONFIG.dinov2_model_name):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DINOv2 requested on CUDA, but torch.cuda.is_available() is False")
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.patch_size = CONFIG.patch_size
        self.input_size = CONFIG.dinov2_input_size
        self.tf = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def extract_road_patch_embeddings(self, image_rgb: np.ndarray, road_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pil = Image.fromarray(image_rgb)
        x = self.tf(pil).unsqueeze(0).to(self.device)
        amp = torch.autocast(device_type="cuda", dtype=torch.float16) if self.device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with amp:
            out = self.model.forward_features(x)
        tokens = out["x_norm_patchtokens"]
        # DINOv2 ViT-S/14 with 518 input -> 37x37 tokens.
        grid = int(self.input_size // self.patch_size)
        emb = tokens.reshape(1, grid, grid, -1)[0]
        mask_t = torch.from_numpy(road_mask.astype(np.float32))[None, None]
        mask_grid = F.interpolate(mask_t, size=(grid, grid), mode="area")[0, 0]
        keep = mask_grid > CONFIG.road_patch_fraction
        coords = torch.nonzero(keep, as_tuple=False).cpu().numpy().astype(np.int16)
        selected = emb[keep].float().cpu().numpy().astype(np.float32)
        return selected, coords
