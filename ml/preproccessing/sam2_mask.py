"""
Road segmentation using SAM2.

We don't want the person prompting SAM2 by hand for thousands of images, so this
module auto-generates a box prompt from a heuristic ROI (bottom-of-frame region,
see config.ROI_BOX_FRACTIONS) and asks SAM2 to segment within it.

If your camera is fixed-mount, replace `get_road_mask` with a single static mask
loaded from disk instead of running SAM2 per-image -- it'll be both faster and
more consistent.
"""

import numpy as np
import torch

import config


class RoadMasker:
    def __init__(self, device: str = config.DEVICE):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(
            config.SAM2_MODEL_CFG,
            str(config.SAM2_CHECKPOINT),
            device=device,
        )
        self.predictor = SAM2ImagePredictor(sam2_model)

    def _roi_box(self, w: int, h: int) -> np.ndarray:
        x0f, y0f, x1f, y1f = config.ROI_BOX_FRACTIONS
        return np.array([x0f * w, y0f * h, x1f * w, y1f * h])

    def get_road_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        image_rgb: HxWx3 uint8 array.
        Returns: HxW bool mask, True where the pixel is road surface.
        """
        h, w = image_rgb.shape[:2]
        box = self._roi_box(w, h)

        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
        # Pick SAM2's most confident mask for this box prompt.
        best = masks[int(np.argmax(scores))]
        return best.astype(bool)
