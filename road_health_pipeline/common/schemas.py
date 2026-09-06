from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
import json

import numpy as np


class DefectType(str, Enum):
    POTHOLE = "pothole"
    WATER_FILLED_POTHOLE = "water_filled_pothole"
    CRACK_OR_DAMAGE = "crack_or_damage"
    CRACK = "crack"
    SURFACE_WEAR = "surface_wear"
    ROAD_ANOMALY = "road_anomaly"
    UNKNOWN_ANOMALY = "unknown_road_anomaly"


@dataclass
class Telemetry:
    timestamp: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None
    frame_id: Optional[int] = None
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    world_z: Optional[float] = None
    speed_mps: Optional[float] = None


@dataclass
class SegmentationResult:
    """Structured output from a single SAM2 segmentation call."""
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
    """Transparent severity estimation breakdown for an individual defect."""
    severity: str = "low"  # "low", "medium", "high", "critical"
    severity_score: float = 0.0  # continuous or 0-100 scale
    severity_components: dict[str, Optional[float]] = field(default_factory=dict)
    area: Optional[float] = None
    depth: Optional[float] = None
    shape: Optional[float] = None
    water: Optional[float] = None
    surrounding_damage: Optional[float] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        if not self.severity_components:
            self.severity_components = {
                "area": self.area,
                "depth": self.depth,
                "water": self.water,
                "surrounding_damage": self.surrounding_damage,
                "shape": self.shape,
                "confidence": self.confidence,
            }
        else:
            if self.area is None and "area" in self.severity_components:
                self.area = self.severity_components["area"]
            if self.depth is None and "depth" in self.severity_components:
                self.depth = self.severity_components["depth"]
            if self.water is None and "water" in self.severity_components:
                self.water = self.severity_components["water"]
            if self.surrounding_damage is None and "surrounding_damage" in self.severity_components:
                self.surrounding_damage = self.severity_components["surrounding_damage"]
            if self.shape is None and "shape" in self.severity_components:
                self.shape = self.severity_components["shape"]
            if self.confidence is None and "confidence" in self.severity_components:
                self.confidence = self.severity_components["confidence"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "severity_score": round(self.severity_score, 4) if isinstance(self.severity_score, float) else self.severity_score,
            "severity_components": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.severity_components.items()
            },
            "area": self.area,
            "depth": self.depth,
            "shape": self.shape,
            "water": self.water,
            "surrounding_damage": self.surrounding_damage,
            "confidence": self.confidence,
        }


@dataclass
class SeverityResult:
    severity: str  # "low", "medium", "high", "critical"
    severity_score: float  # continuous [0, 1] or 0-100
    severity_components: dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefectMeasurement:
    """Individual defect measurement with full traceability and physical metrics."""
    defect_id: str
    defect_type: str  # from DefectType
    confidence: float
    bbox: list[int]  # [x1, y1, x2, y2]
    mask_area_pixels: int
    estimated_area_m2: Optional[float]
    estimated_depth_m: Optional[float]
    is_water_filled: bool
    water_confidence: float
    crack_or_damage_extent: Optional[float]
    road_segment_id: str
    timestamp: str
    latitude: Optional[float]
    longitude: Optional[float]
    severity: SeverityBreakdown
    depth_source: str = "unavailable"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "defect_type": self.defect_type,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "mask_area_pixels": self.mask_area_pixels,
            "estimated_area_m2": round(self.estimated_area_m2, 4) if self.estimated_area_m2 is not None else None,
            "estimated_depth_m": round(self.estimated_depth_m, 4) if self.estimated_depth_m is not None else None,
            "is_water_filled": self.is_water_filled,
            "water_confidence": round(self.water_confidence, 4),
            "crack_or_damage_extent": round(self.crack_or_damage_extent, 4) if self.crack_or_damage_extent is not None else None,
            "road_segment_id": self.road_segment_id,
            "timestamp": self.timestamp,
            "latitude": round(self.latitude, 7) if self.latitude is not None else None,
            "longitude": round(self.longitude, 7) if self.longitude is not None else None,
            "severity": self.severity.to_dict() if hasattr(self.severity, "to_dict") else self.severity,
            "depth_source": self.depth_source,
            "notes": self.notes,
        }


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
class RoadHealthResult:
    road_health_score: float  # 0 to 100
    condition_class: str  # "good", "fair", "poor", "critical"
    components: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "road_health_score": round(self.road_health_score, 2),
            "condition_class": self.condition_class,
            "components": {k: round(v, 2) for k, v in self.components.items()},
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
        }


# RoadHealthScore alias/dataclass for compatibility
@dataclass
class RoadHealthScore:
    road_health_score: float
    condition_class: str  # "Good", "Fair", "Poor", "Critical"
    components: dict[str, float] = field(default_factory=lambda: {
        "pothole_penalty": 0.0,
        "crack_penalty": 0.0,
        "water_penalty": 0.0,
        "surface_penalty": 0.0,
    })
    confidence: float = 1.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "road_health_score": round(self.road_health_score, 2),
            "condition_class": self.condition_class,
            "components": {k: round(v, 2) for k, v in self.components.items()},
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
        }


@dataclass
class PredictionResult:
    """Temporal deterioration and pothole formation prediction result."""
    deterioration_probability: Optional[float] = None
    pothole_formation_probability: Optional[float] = None
    prediction_horizon_days: Optional[int] = 30
    progression_direction: Optional[str] = "stable"  # "improving", "stable", "degrading", "critical"
    progression_trend: Optional[str] = None  # "stable", "deteriorating", "rapidly_deteriorating"
    scientific_status: str = "CARLA-SYNTHETIC ONLY"
    feature_importances: Optional[dict[str, float]] = None
    features_used: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterioration_probability": round(self.deterioration_probability, 4) if self.deterioration_probability is not None else None,
            "pothole_formation_probability": round(self.pothole_formation_probability, 4) if self.pothole_formation_probability is not None else None,
            "prediction_horizon_days": self.prediction_horizon_days,
            "progression_direction": self.progression_direction,
            "progression_trend": self.progression_trend or self.progression_direction,
            "scientific_status": self.scientific_status,
            "feature_importances": self.feature_importances,
            "features_used": self.features_used,
            "notes": self.notes,
        }


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
    road_health: Union[RoadHealthResult, RoadHealthScore]
    prediction: Optional[PredictionResult] = None
    inspection_timestamp: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    detections: list[PotholeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["road_health"] = self.road_health.to_dict() if hasattr(self.road_health, "to_dict") else self.road_health
        d["prediction"] = self.prediction.to_dict() if hasattr(self.prediction, "to_dict") else self.prediction
        d["detections"] = [det.to_dict() if hasattr(det, "to_dict") else asdict(det) for det in self.detections]
        return d


@dataclass
class RoadSegmentAggregate:
    """Aggregated segment record retaining full traceability to individual defect measurements."""
    road_segment_id: str
    inspection_timestamp: str
    latitude: Optional[float]
    longitude: Optional[float]
    total_defects: int
    total_potholes: int
    total_water_potholes: int
    total_cracks: int
    total_damaged_area_m2: float
    avg_severity_score: float
    max_severity_score: float
    water_hazard_count: int
    crack_damage_extent_m2: float
    road_health: Union[RoadHealthScore, RoadHealthResult]
    prediction: PredictionResult
    detections: list[DefectMeasurement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "road_segment_id": self.road_segment_id,
            "inspection_timestamp": self.inspection_timestamp,
            "geolocation": {
                "lat": round(self.latitude, 7) if self.latitude is not None else None,
                "lon": round(self.longitude, 7) if self.longitude is not None else None,
            },
            "total_defects": self.total_defects,
            "total_potholes": self.total_potholes,
            "total_water_potholes": self.total_water_potholes,
            "total_cracks": self.total_cracks,
            "total_damaged_area_m2": round(self.total_damaged_area_m2, 4),
            "avg_severity_score": round(self.avg_severity_score, 2),
            "max_severity_score": round(self.max_severity_score, 2),
            "water_hazard_count": self.water_hazard_count,
            "crack_damage_extent_m2": round(self.crack_damage_extent_m2, 4),
            "road_health": self.road_health.to_dict() if hasattr(self.road_health, "to_dict") else self.road_health,
            "prediction": self.prediction.to_dict() if hasattr(self.prediction, "to_dict") else self.prediction,
            "detections": [d.to_dict() if hasattr(d, "to_dict") else asdict(d) for d in self.detections],
        }


@dataclass
class InferenceResult:
    """Stable JSON output schema complying with Section 14 and backward compatibility."""
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    timestamp: str = ""
    road_segment_id: Optional[str] = None
    geolocation: dict[str, Optional[float]] = field(default_factory=dict)
    image_shape: list[int] = field(default_factory=list)
    anomaly_threshold: float = 0.0
    anomaly_score: float = 0.0
    detections: list[DefectMeasurement] = field(default_factory=list)
    potholes: list[PotholeRecord] = field(default_factory=list)
    road_health: Optional[Union[RoadHealthScore, RoadHealthResult]] = None
    prediction: Optional[PredictionResult] = None
    segment_summary: Optional[dict[str, Any]] = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    frame_id: Optional[int] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id or (Path(self.image_path).name if self.image_path else "IMG"),
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "road_segment_id": self.road_segment_id,
            "geolocation": self.geolocation or {
                "lat": self.telemetry.get("latitude") if self.telemetry else None,
                "lon": self.telemetry.get("longitude") if self.telemetry else None,
            },
            "image_shape": self.image_shape,
            "anomaly_threshold": round(self.anomaly_threshold, 4) if self.anomaly_threshold is not None else 0.0,
            "anomaly_score": round(self.anomaly_score, 4) if self.anomaly_score is not None else 0.0,
            "detections": [d.to_dict() if hasattr(d, "to_dict") else asdict(d) for d in self.detections],
            "potholes": [p.to_dict() if hasattr(p, "to_dict") else asdict(p) for p in self.potholes],
            "road_health": self.road_health.to_dict() if hasattr(self.road_health, "to_dict") else (asdict(self.road_health) if self.road_health else None),
            "prediction": self.prediction.to_dict() if hasattr(self.prediction, "to_dict") else (asdict(self.prediction) if self.prediction else None),
            "segment_summary": self.segment_summary,
            "telemetry": self.telemetry,
            "frame_id": self.frame_id,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


