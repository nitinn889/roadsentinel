from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional
import json

import numpy as np


class DefectType(str, Enum):
    POTHOLE = "pothole"
    WATER_FILLED_POTHOLE = "water_filled_pothole"
    CRACK_OR_DAMAGE = "crack_or_damage"
    UNKNOWN_ANOMALY = "unknown_road_anomaly"


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
        }


@dataclass
class CandidateRegion:
    """An anomaly candidate produced by PotholeLocalizer."""

    mask: np.ndarray
    bbox_xyxy: list[int]
    anomaly_score: float
    pothole_confidence: float
    sam2_result: Optional[SegmentationResult] = None
    defect_type: str = DefectType.UNKNOWN_ANOMALY.value
    shape_circularity: float = 0.0
    aspect_ratio: float = 1.0
    surrounding_damage: float = 0.0


@dataclass
class SeverityBreakdown:
    area: Optional[float] = None
    depth: Optional[float] = None
    shape: Optional[float] = None
    water: Optional[float] = None
    surrounding_damage: Optional[float] = None
    confidence: Optional[float] = None

    def to_dict(self) -> dict[str, Optional[float]]:
        return asdict(self)


@dataclass
class SeverityResult:
    severity: str  # "low", "medium", "high", "critical"
    severity_score: float  # continuous [0, 1]
    severity_components: dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoadHealthResult:
    road_health_score: float  # 0 to 100
    condition_class: str  # "good", "fair", "poor", "critical"
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    deterioration_probability: Optional[float] = None
    pothole_formation_probability: Optional[float] = None
    prediction_horizon_days: int = 30
    progression_direction: Optional[str] = "stable"  # "improving", "stable", "degrading", "critical"
    feature_importances: Optional[dict[str, float]] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    defect_type: str = DefectType.POTHOLE.value
    road_segment_id: Optional[str] = None
    crack_or_damage_extent: Optional[float] = None
    shape_circularity: Optional[float] = None
    aspect_ratio: Optional[float] = None
    severity_breakdown: Optional[dict[str, Optional[float]]] = None
    depth_source: str = "unavailable"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentSummary:
    road_segment_id: str
    total_defects: int
    total_potholes: int
    total_damaged_area_m2: Optional[float]
    avg_severity: float
    max_severity: float
    has_water_hazard: bool
    water_hazard_count: int
    road_health: RoadHealthResult
    prediction: Optional[PredictionResult] = None
    inspection_timestamp: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    detections: list[PotholeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["road_health"] = self.road_health.to_dict() if self.road_health else None
        d["prediction"] = self.prediction.to_dict() if self.prediction else None
        d["detections"] = [det.to_dict() if hasattr(det, "to_dict") else asdict(det) for det in self.detections]
        return d


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
    road_segment_id: Optional[str] = None
    road_health: Optional[RoadHealthResult] = None
    prediction: Optional[PredictionResult] = None
    segment_summary: Optional[dict[str, Any]] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["potholes"] = [p.to_dict() if hasattr(p, "to_dict") else asdict(p) for p in self.potholes]
        if self.road_health is not None:
            d["road_health"] = self.road_health.to_dict() if hasattr(self.road_health, "to_dict") else asdict(self.road_health)
        if self.prediction is not None:
            d["prediction"] = self.prediction.to_dict() if hasattr(self.prediction, "to_dict") else asdict(self.prediction)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

