"""Unit tests for road segment aggregation."""

import pytest
from analytics.segment_aggregator import RoadSegmentAggregator, generate_spatial_segment_id
from common.schemas import InferenceResult, PotholeRecord, Telemetry


def make_sample_record(pid, lat=13.0827, lon=80.2707, sev=0.6, water=False, area=0.8):
    return PotholeRecord(
        pothole_id=pid,
        timestamp="2026-08-30T12:00:00Z",
        latitude=lat,
        longitude=lon,
        altitude_m=30.0,
        area_m2=area,
        estimated_depth_m=0.07,
        anomaly_score=0.75,
        pothole_confidence=0.85,
        severity_score=sev,
        water_flag=water,
        water_confidence=0.8 if water else 0.0,
        source_image="img.jpg",
        mask_area_px=1500,
        bbox_xyxy=[10, 10, 80, 80],
    )


def test_spatial_segment_id_generation():
    seg1 = generate_spatial_segment_id(13.08271, 80.27071)
    seg2 = generate_spatial_segment_id(13.08274, 80.27069)
    assert seg1 == seg2
    assert seg1 == "seg_13.083_80.271"
    assert generate_spatial_segment_id(None, None) == "segment_unknown"


def test_aggregate_records_traceability():
    agg = RoadSegmentAggregator()
    recs = [
        make_sample_record("p1", sev=0.4, water=False, area=0.5),
        make_sample_record("p2", sev=0.8, water=True, area=1.2),
    ]
    summary = agg.aggregate_records(recs, segment_id="seg_alpha")
    assert summary.road_segment_id == "seg_alpha"
    assert summary.total_defects == 2
    assert summary.total_potholes == 2
    assert summary.total_damaged_area_m2 == 1.7
    assert summary.max_severity == 0.8
    assert summary.has_water_hazard is True
    assert summary.water_hazard_count == 1
    assert len(summary.detections) == 2
    assert summary.detections[0].pothole_id == "p1"
    assert summary.detections[1].pothole_id == "p2"


def test_aggregate_inferences_grouping():
    agg = RoadSegmentAggregator()
    inf1 = InferenceResult(
        image_path="img1.jpg",
        timestamp="2026-08-30T10:00:00Z",
        frame_id=1,
        telemetry={"latitude": 13.082, "longitude": 80.270},
        image_shape=[720, 1280, 3],
        anomaly_threshold=0.5,
        anomaly_score=0.4,
        potholes=[make_sample_record("p1", lat=13.082, lon=80.270)],
        road_segment_id="seg_A",
    )
    inf2 = InferenceResult(
        image_path="img2.jpg",
        timestamp="2026-08-30T10:05:00Z",
        frame_id=2,
        telemetry={"latitude": 13.095, "longitude": 80.280},
        image_shape=[720, 1280, 3],
        anomaly_threshold=0.5,
        anomaly_score=0.3,
        potholes=[make_sample_record("p2", lat=13.095, lon=80.280)],
        road_segment_id="seg_B",
    )

    summaries = agg.aggregate_inferences([inf1, inf2])
    assert "seg_A" in summaries
    assert "seg_B" in summaries
    assert summaries["seg_A"].total_defects == 1
    assert summaries["seg_B"].total_defects == 1
