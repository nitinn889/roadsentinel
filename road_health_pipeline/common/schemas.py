from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional
import json


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
