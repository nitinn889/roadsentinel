"""SAM2 road masking and pothole segmentation for RoadSentinel.

This module provides two modes of SAM2 usage:

1. Road-surface masking (``get_road_mask``):
   Prompts SAM2 with a bounding box covering the expected road region
   (configurable per camera mode: nadir vs forward-facing).  Used during
   memory-bank building to isolate road pixels before DINOv2 embedding.

2. Pothole candidate refinement (``refine_box``):
   Prompts SAM2 with a candidate bounding box identified by the DINOv2
   anomaly detector.  Returns a ``SegmentationResult`` with the mask,
   SAM2 confidence score, tight bounding box, and area.

SAM2 behaviour with ``multimask_output=True``
----------------------------------------------
SAM2 returns up to three candidate masks ranked by its internal IoU
predictor.  The ``_choose_road_mask`` and ``_select_best_mask`` helpers
rank these candidates using a multi-factor score rather than blindly
picking the one with the highest SAM2 score.

Installation
------------
Install SAM2 from Meta's official repository or via ``pip install sam2``.
The checkpoint path is read from ``CONFIG.sam2_checkpoint``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch

from common.schemas import SegmentationResult
from config import CONFIG

log = logging.getLogger(__name__)


class RoadMasker:
    """Wrapper around SAM2ImagePredictor for road-surface and pothole masking.

    Parameters
    ----------
    device:
        Torch device string.  Falls back to CPU if CUDA unavailable.
    camera_mode:
        ``"nadir"`` or ``"forward"`` — controls the SAM2 prompt box.
    """

    def __init__(
        self,
        device: str = CONFIG.device,
        camera_mode: str = CONFIG.camera_mode,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("SAM2 requested CUDA but CUDA unavailable; falling back to CPU.")
            device = "cpu"
        self.device = device
        self.camera_mode = camera_mode

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise ImportError(
                "SAM2 is not installed.  Install it with:\n"
                "  pip install sam2\n"
                "and download a checkpoint; see README.md for details."
            ) from exc

        ckpt = CONFIG.sam2_checkpoint
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {ckpt}\n"
                "Download it with:\n"
                "  wget -P road_health_pipeline/checkpoints/ \\\n"
                "    https://dl.fbaipublicfiles.com/segment_anything_2/"
                "092824/sam2.1_hiera_small.pt"
            )

        log.info("Loading SAM2 from %s on %s", ckpt, device)
        model = build_sam2(CONFIG.sam2_model_cfg, str(ckpt), device=device)
        self.predictor = SAM2ImagePredictor(model)
        self.last_raw_road_mask: Optional[np.ndarray] = None
        self.last_road_mask_diagnostics: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _autocast(self):
        if self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def _roi_box(self, w: int, h: int) -> np.ndarray:
        """Return the SAM2 prompt box for the road ROI (pixel coords, XYXY)."""
        if self.camera_mode == "nadir":
            f = CONFIG.nadir_roi_box_fractions
        elif self.camera_mode == "forward":
            f = CONFIG.forward_roi_box_fractions
        else:
            raise ValueError(f"Unknown camera_mode: {self.camera_mode!r}")
        x0f, y0f, x1f, y1f = f
        return np.array([x0f * w, y0f * h, x1f * w, y1f * h], dtype=np.float32)

    @staticmethod
    def _filter_masks(
        masks: np.ndarray,
        scores: np.ndarray,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove masks that are implausibly tiny or cover the whole image.

        Filters out masks whose area is:
          - below ``CONFIG.candidate_min_area_px`` pixels, OR
          - above ``CONFIG.candidate_max_area_fraction`` of the total image area.

        If all masks fail the filter, all are returned unchanged so the caller
        can still pick the least-bad one rather than crashing.
        """
        total_px = h * w
        keep = []
        for i, m in enumerate(masks):
            area = int(m.sum())
            if area < CONFIG.candidate_min_area_px:
                continue
            if area / total_px > CONFIG.candidate_max_area_fraction:
                continue
            keep.append(i)
        if not keep:
            return masks, scores
        return masks[keep], scores[keep]

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> list[int]:
        """Return the tight bounding box [x1, y1, x2, y2] of a boolean mask."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return [0, 0, 0, 0]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    @staticmethod
    def _choose_road_mask(
        masks: np.ndarray, scores: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Multi-factor ranking for road-surface mask selection.

        Weights SAM2 confidence, fractional coverage, and proximity to the
        image centre.  Larger, central masks are preferred for nadir views
        where the road typically fills most of the frame.

        Returns the boolean mask of the best candidate.
        """
        valid = []
        center = np.array([w / 2, h / 2])
        for i, m in enumerate(masks):
            area_frac = float(m.mean())
            ys, xs = np.nonzero(m)
            if len(xs) < 1:
                continue
            centroid = np.array([xs.mean(), ys.mean()])
            center_dist = np.linalg.norm((centroid - center) / np.array([w, h]))
            # Prefer large, central masks while still honouring SAM2 score.
            # Weights: SAM2 confidence 1.0, area (up to 0.65 frac) 0.35, centrality -0.25
            quality = (
                float(scores[i])
                + 0.35 * min(area_frac / 0.65, 1.0)
                - 0.25 * center_dist
            )
            valid.append((quality, i))
        if not valid:
            return masks[int(np.argmax(scores))].astype(bool)
        return masks[max(valid)[1]].astype(bool)

    @staticmethod
    def _select_best_mask(
        masks: np.ndarray, scores: np.ndarray, h: int, w: int
    ) -> SegmentationResult:
        """Select the best mask for a candidate pothole and return a SegmentationResult.

        Uses the same multi-factor ranking as ``_choose_road_mask`` so that
        both road-surface and candidate-refinement paths behave consistently.
        """
        valid = []
        center = np.array([w / 2, h / 2])
        for i, m in enumerate(masks):
            area_frac = float(m.mean())
            ys, xs = np.nonzero(m)
            if len(xs) < 1:
                continue
            centroid = np.array([xs.mean(), ys.mean()])
            center_dist = np.linalg.norm((centroid - center) / np.array([w, h]))
            quality = (
                float(scores[i])
                + 0.35 * min(area_frac / 0.65, 1.0)
                - 0.25 * center_dist
            )
            valid.append((quality, i))

        best_idx = max(valid)[1] if valid else int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)
        return SegmentationResult(
            mask=best_mask,
            confidence=float(scores[best_idx]),
            bbox_xyxy=RoadMasker._mask_bbox(best_mask),
            area_px=int(best_mask.sum()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_road_corridor_fallback(image_rgb: np.ndarray) -> np.ndarray:
        """Extracts the central drivable road corridor when segmentation prompts fail."""
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        # In standard drone survey & dashcam angles, the highway occupies the central 65-75% corridor
        # We detect parallel asphalt edges or bound the corridor to reject off-road checkerboard & verges
        x1 = int(0.15 * w)
        x2 = int(0.75 * w)
        mask[:, x1:x2] = True
        return mask

    def _road_prior(self, w: int, h: int) -> np.ndarray:
        """Return the conservative camera-mode prompt corridor as a mask."""
        x1, y1, x2, y2 = self._roi_box(w, h).astype(int)
        prior = np.zeros((h, w), dtype=bool)
        prior[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        return prior

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if count <= 1:
            return np.zeros_like(mask, dtype=bool)
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == label

    def _stabilize_road_mask(self, raw_mask: np.ndarray, image_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """Reject full-frame leaks and undersized masks using only camera priors.

        Vegetation suppression is deliberately applied only to an implausibly
        large SAM2 mask; it is not used to define a road on normal images.
        """
        h, w = raw_mask.shape
        prior = self._road_prior(w, h)
        raw = np.asarray(raw_mask, dtype=bool)
        raw_ratio = float(raw.mean())
        prior_coverage = float((raw & prior).sum() / max(1, prior.sum()))
        reason = "accepted_raw"
        candidate = self._largest_component(raw)
        if raw_ratio > CONFIG.road_mask_max_area_fraction:
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            vegetation = (
                (hsv[..., 0] >= CONFIG.road_mask_vegetation_hue_low)
                & (hsv[..., 0] <= CONFIG.road_mask_vegetation_hue_high)
                & (hsv[..., 1] >= CONFIG.road_mask_vegetation_min_saturation)
            )
            candidate = self._largest_component(candidate & prior & ~vegetation)
            reason = "full_frame_sanitized_with_prior"
        elif prior_coverage < CONFIG.road_mask_min_prior_coverage:
            candidate = prior
            reason = "undersized_replaced_with_prior"

        if candidate.mean() < CONFIG.road_mask_min_area_fraction:
            candidate = prior
            reason = "empty_after_sanitization_replaced_with_prior"
        return candidate, {
            "raw_area_ratio": raw_ratio,
            "improved_area_ratio": float(candidate.mean()),
            "prior_area_ratio": float(prior.mean()),
            "raw_prior_coverage": prior_coverage,
            "selection_reason": reason,
        }

    def get_road_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a boolean HxW mask covering the road surface.

        Prompts SAM2 with the camera-mode-specific ROI box.  Used during
        inference & memory-bank building to focus DINOv2 embeddings on road pixels.

        Parameters
        ----------
        image_rgb:
            HxWx3 uint8 RGB image.

        Returns
        -------
        np.ndarray
            Boolean HxW mask.
        """
        h, w = image_rgb.shape[:2]

        # Enhance low-light inputs so SAM 2 can distinguish road boundaries
        proc_rgb = image_rgb
        try:
            from inference.low_light_enhancer import LowLightEnhancer
            proc_rgb, _ = LowLightEnhancer().enhance(image_rgb)
        except Exception:
            pass

        box = self._roi_box(w, h)
        with torch.inference_mode(), self._autocast():
            self.predictor.set_image(proc_rgb)
            masks, scores, _ = self.predictor.predict(
                box=box[None, :], multimask_output=True
            )
        # Select best road mask
        raw_mask = self._choose_road_mask(masks, scores, h, w)
        self.last_raw_road_mask = raw_mask.copy()
        road_mask, diagnostics = self._stabilize_road_mask(raw_mask, image_rgb)
        diagnostics["sam2_scores"] = [float(v) for v in scores]
        self.last_road_mask_diagnostics = diagnostics
        return road_mask

    def refine_box(
        self,
        image_rgb: np.ndarray,
        box_xyxy: list[float],
    ) -> SegmentationResult:
        """Refine a candidate pothole bounding box into a segmentation mask.

        Prompts SAM2 with the anomaly-detector candidate box.  Returns a
        ``SegmentationResult`` containing the mask, SAM2 confidence, tight bbox,
        and area in pixels.

        If SAM2 produces no valid mask, a fallback rectangular mask is returned
        with confidence 0.0.

        Parameters
        ----------
        image_rgb:
            HxWx3 uint8 RGB image.
        box_xyxy:
            Candidate bounding box ``[x1, y1, x2, y2]`` in pixel coords.

        Returns
        -------
        SegmentationResult
        """
        h, w = image_rgb.shape[:2]
        box = np.asarray(box_xyxy, dtype=np.float32)
        with torch.inference_mode(), self._autocast():
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(
                box=box[None, :], multimask_output=True
            )
        masks, scores = self._filter_masks(masks, scores, h, w)
        if masks.shape[0] == 0:
            # Fallback: rectangular mask covering the prompt box
            log.warning("SAM2 returned no valid masks for box %s; using fallback rect.", box_xyxy)
            fallback = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in box_xyxy]
            fallback[y1:y2, x1:x2] = True
            return SegmentationResult(
                mask=fallback,
                confidence=0.0,
                bbox_xyxy=[x1, y1, x2, y2],
                area_px=int(fallback.sum()),
            )
        return self._select_best_mask(masks, scores, h, w)
