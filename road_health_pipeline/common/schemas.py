from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional
import json

import numpy as np


@dataclass
class Telemetry:
    timestamp: str
    latitude: Optional[float]
    longitude: Optional[float]
    altitude_m: Optional[float]
    heading_deg: Optional[float]
    frame_id: Optional[int] = None
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    world_z: Optional[float] = None
    speed_mps: Optional[float] = None


@dataclass
class SegmentationResult:
    """Structured output from a single SAM2 segmentation call.

    Attributes
    ----------
    mask:
        Boolean HxW array indicating the segmented region.
    confidence:
        SAM2 IoU score for this mask (higher = more confident).
    bbox_xyxy:
        Tight bounding box [x1, y1, x2, y2] around the mask (integer pixel coords).
    area_px:
        Number of True pixels in the mask.
    """

    mask: np.ndarray
    confidence: float
    bbox_xyxy: list[int]
    area_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy,
            "area_px": self.area_px,
            # mask is not serialised here — use mask.tolist() explicitly if needed
        }


@dataclass
class CandidateRegion:
    """An anomaly candidate produced by PotholeLocalizer.

    This is typed so that all downstream consumers (run_inference, tests) share
    a stable contract rather than relying on ad-hoc dict keys.

    Attributes
    ----------
    mask:
        Boolean HxW candidate mask (may come from the anomaly map connected
        component or from SAM2 refinement).
    bbox_xyxy:
        Bounding box [x1, y1, x2, y2] of the connected component (integer coords).
    anomaly_score:
        Mean DINOv2 anomaly score across pixels in the anomaly-map region.
    pothole_confidence:
        Heuristic confidence score [0, 1].  Not a trained classifier output.
    sam2_result:
        SAM2 segmentation result if refinement was applied, else None.
    """

    mask: np.ndarray
    bbox_xyxy: list[int]
    anomaly_score: float
    pothole_confidence: float
    sam2_result: Optional[SegmentationResult] = None


@dataclass
class PotholeRecord:
    pothole_id: str
    timestamp: str
    latitude: Optional[float]
    longitude: Optional[float]
    altitude_m: Optional[float]
    area_m2: Optional[float]
    estimated_depth_m: Optional[float]
    anomaly_score: float
    pothole_confidence: float
    severity_score: float
    water_flag: bool
    water_confidence: float
    source_image: str
    mask_area_px: int
    bbox_xyxy: list[int]
    depth_source: str = "unavailable"
    notes: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    image_path: str
    timestamp: str
    frame_id: Optional[int]
    telemetry: dict[str, Any]
    image_shape: list[int]
    anomaly_threshold: float
    anomaly_score: float
    potholes: list[PotholeRecord]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["potholes"] = [asdict(p) for p in self.potholes]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
