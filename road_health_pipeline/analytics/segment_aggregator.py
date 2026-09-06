"""Road-Segment Spatial & Temporal Aggregator for RoadSentinel.

Aggregates multiple frame-level inspection detections and telemetries into stable
road segment records while retaining full backward traceability to individual defects.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Sequence, Union
import numpy as np

from common.schemas import (
    InferenceResult,
    PotholeRecord,
    RoadHealthResult,
    SegmentSummary,
)
from analytics.road_health import calculate_road_health_score


def generate_spatial_segment_id(lat: Optional[float], lon: Optional[float], precision: int = 3) -> str:
    """Generate a stable spatial segment ID from GPS coordinates if ID is missing.

    A precision of 3 decimal places corresponds to ~110m cells, which represents
    a standard urban block / road segment.
    """
    if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
        return "segment_unknown"
    lat_bin = round(lat, precision)
    lon_bin = round(lon, precision)
    return f"seg_{lat_bin:.3f}_{lon_bin:.3f}"


class RoadSegmentAggregator:
    """Aggregates frame detections into segment-level health summaries."""

    def __init__(self, default_segment_id: Optional[str] = None):
        self.default_segment_id = default_segment_id

    def aggregate_records(
        self,
        records: Sequence[PotholeRecord],
        segment_id: Optional[str] = None,
        timestamp: str = "",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        surface_anomaly_mean: float = 0.0,
    ) -> SegmentSummary:
        """Aggregate a collection of pothole records for a single road segment."""
        # Determine segment ID
        if segment_id:
            seg_id = segment_id
        elif self.default_segment_id:
            seg_id = self.default_segment_id
        else:
            seg_id = generate_spatial_segment_id(latitude, longitude)

        total_defects = len(records)
        total_potholes = 0
        water_hazard_count = 0
        severities: list[float] = []
        total_area_m2: Optional[float] = 0.0
        has_any_area = False
        lats: list[float] = []
        lons: list[float] = []

        for r in records:
            if r.defect_type in {"pothole", "water_filled_pothole"}:
                total_potholes += 1
            if r.water_flag:
                water_hazard_count += 1
            severities.append(float(r.severity_score))
            if r.area_m2 is not None and np.isfinite(r.area_m2):
                total_area_m2 += float(r.area_m2)
                has_any_area = True
            if r.latitude is not None and np.isfinite(r.latitude):
                lats.append(r.latitude)
            if r.longitude is not None and np.isfinite(r.longitude):
                lons.append(r.longitude)

        avg_sev = float(np.mean(severities)) if severities else 0.0
        max_sev = float(np.max(severities)) if severities else 0.0
        damaged_area_m2 = round(total_area_m2, 4) if has_any_area else None

        # Compute centroid lat/lon if not supplied
        centroid_lat = latitude if latitude is not None else (float(np.mean(lats)) if lats else None)
        centroid_lon = longitude if longitude is not None else (float(np.mean(lons)) if lons else None)

        # Compute segment health score
        health_res = calculate_road_health_score(
            potholes=records,
            total_crack_area_m2=0.0,
            surface_anomaly_mean=surface_anomaly_mean,
        )

        return SegmentSummary(
            road_segment_id=seg_id,
            total_defects=total_defects,
            total_potholes=total_potholes,
            total_damaged_area_m2=damaged_area_m2,
            avg_severity=round(avg_sev, 4),
            max_severity=round(max_sev, 4),
            has_water_hazard=water_hazard_count > 0,
            water_hazard_count=water_hazard_count,
            road_health=health_res,
            prediction=None,
            inspection_timestamp=timestamp,
            latitude=round(centroid_lat, 6) if centroid_lat is not None else None,
            longitude=round(centroid_lon, 6) if centroid_lon is not None else None,
            detections=list(records),
        )

    def aggregate_inferences(
        self,
        inferences: Sequence[InferenceResult],
    ) -> dict[str, SegmentSummary]:
        """Group a batch of inference results by segment ID and return summary per segment."""
        grouped: dict[str, list[PotholeRecord]] = defaultdict(list)
        timestamps: dict[str, str] = {}
        lats: dict[str, list[float]] = defaultdict(list)
        lons: dict[str, list[float]] = defaultdict(list)
        anomaly_scores: dict[str, list[float]] = defaultdict(list)

        for inf in inferences:
            t = inf.telemetry or {}
            lat = t.get("latitude")
            lon = t.get("longitude")
            seg_id = inf.road_segment_id or generate_spatial_segment_id(lat, lon)
            
            for p in inf.potholes:
                p.road_segment_id = seg_id
                grouped[seg_id].append(p)
            
            timestamps[seg_id] = inf.timestamp
            if lat is not None and np.isfinite(lat):
                lats[seg_id].append(lat)
            if lon is not None and np.isfinite(lon):
                lons[seg_id].append(lon)
            anomaly_scores[seg_id].append(inf.anomaly_score)

        summaries: dict[str, SegmentSummary] = {}
        for seg_id, records in grouped.items():
            mean_lat = float(np.mean(lats[seg_id])) if lats[seg_id] else None
            mean_lon = float(np.mean(lons[seg_id])) if lons[seg_id] else None
            mean_amap = float(np.mean(anomaly_scores[seg_id])) if anomaly_scores[seg_id] else 0.0

            summary = self.aggregate_records(
                records=records,
                segment_id=seg_id,
                timestamp=timestamps.get(seg_id, ""),
                latitude=mean_lat,
                longitude=mean_lon,
                surface_anomaly_mean=mean_amap,
            )
            summaries[seg_id] = summary

        return summaries
