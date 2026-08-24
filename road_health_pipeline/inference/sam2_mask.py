from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from config import CONFIG


class RoadMasker:
    def __init__(self, device: str = CONFIG.device, camera_mode: str = CONFIG.camera_mode):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM2 requested on CUDA, but torch.cuda.is_available() is False")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise ImportError("Install SAM2 from the official Meta repository before using RoadMasker") from exc
        if not CONFIG.sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {CONFIG.sam2_checkpoint}")
        model = build_sam2(CONFIG.sam2_model_cfg, str(CONFIG.sam2_checkpoint), device=device)
        self.predictor = SAM2ImagePredictor(model)
        self.device = device
        self.camera_mode = camera_mode

    def _roi_box(self, w: int, h: int) -> np.ndarray:
        if self.camera_mode == "nadir":
            f = CONFIG.nadir_roi_box_fractions
        elif self.camera_mode == "forward":
            f = CONFIG.forward_roi_box_fractions
        else:
            raise ValueError(f"Unknown camera mode: {self.camera_mode}")
        x0f, y0f, x1f, y1f = f
        return np.array([x0f * w, y0f * h, x1f * w, y1f * h], dtype=np.float32)

    @staticmethod
    def _choose_road_mask(masks: np.ndarray, scores: np.ndarray, h: int, w: int) -> np.ndarray:
        valid = []
        center = np.array([w / 2, h / 2])
        for i, m in enumerate(masks):
            area_frac = float(m.mean())
            ys, xs = np.nonzero(m)
            if len(xs) < 1:
                continue
            centroid = np.array([xs.mean(), ys.mean()])
            center_dist = np.linalg.norm((centroid - center) / np.array([w, h]))
            # Prefer large, central masks while still respecting SAM score.
            quality = float(scores[i]) + 0.35 * min(area_frac / 0.65, 1.0) - 0.25 * center_dist
            valid.append((quality, i))
        if not valid:
            return masks[int(np.argmax(scores))].astype(bool)
        return masks[max(valid)[1]].astype(bool)

    def get_road_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        box = self._roi_box(w, h)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") else torch.autocast(device_type="cpu", enabled=False)
        with torch.inference_mode(), autocast:
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(box=box[None, :], multimask_output=True)
        return self._choose_road_mask(masks, scores, h, w)

    def refine_box(self, image_rgb: np.ndarray, box_xyxy: list[float]) -> np.ndarray:
        """Prompt SAM2 with a candidate pothole box for local refinement."""
        box = np.asarray(box_xyxy, dtype=np.float32)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") else torch.autocast(device_type="cpu", enabled=False)
        with torch.inference_mode(), autocast:
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(box=box[None, :], multimask_output=True)
        return masks[int(np.argmax(scores))].astype(bool)
