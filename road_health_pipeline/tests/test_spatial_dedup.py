import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.server import SpatialDeduplicator, DEDUP_RADIUS_M


def test_spatial_deduplication_3m_radius():
    """Verify that detections within 3.0m are grouped, bboxes merged, and work order not duplicated."""
    dedup = SpatialDeduplicator()
    assert DEDUP_RADIUS_M == 3.0

    img_a = b"dummy_jpeg_bytes_a"
    img_b = b"dummy_jpeg_bytes_b_higher_confidence"

    det_a = {
        "pothole_id": "pot-001",
        "latitude": 13.082700,
        "longitude": 80.270700,
        "confidence": 0.80,
        "pothole_confidence": 0.80,
        "severity_score": 0.72,
        "bbox_xyxy": [100, 100, 200, 200],
        "area_m2": 0.50,
        "estimated_depth_m": 0.08,
    }

    # Ingest detection A -> new cluster
    res_a = dedup.ingest(det_a, image_bytes=img_a)
    assert res_a["status"] == "new"
    cid = res_a["cluster_id"]
    assert dedup.cluster_count() == 1
    assert len(dedup.get_all_work_orders()) == 1

    # Overlapping detection B: ~1.2 meters away (delta ~0.000008 deg)
    det_b = {
        "pothole_id": "pot-002",
        "latitude": 13.082708,
        "longitude": 80.270708,
        "confidence": 0.95,
        "pothole_confidence": 0.95,
        "severity_score": 0.88,
        "bbox_xyxy": [110, 115, 220, 215],
        "area_m2": 0.65,
        "estimated_depth_m": 0.10,
    }

    # Ingest detection B -> deduplicated
    res_b = dedup.ingest(det_b, image_bytes=img_b)
    assert res_b["status"] == "deduplicated"
    assert res_b["cluster_id"] == cid
    assert res_b["distance_m"] < 3.0

    # Verify cluster count did NOT increase
    assert dedup.cluster_count() == 1

    # Verify canonical detection selected higher confidence (0.95 > 0.80)
    canonical = res_b["canonical"]
    assert float(canonical.get("pothole_confidence") or canonical.get("confidence")) == 0.95
    assert dedup.get_patch_image(cid) == img_b

    # Verify bounding boxes were merged
    assert len(canonical["all_bboxes"]) == 2
    assert [100, 100, 200, 200] in canonical["all_bboxes"]
    assert [110, 115, 220, 215] in canonical["all_bboxes"]

    # Verify Work Order was NOT duplicated
    work_orders = dedup.get_all_work_orders()
    assert len(work_orders) == 1, "Work order count must remain 1 after merging duplicate detection!"


def test_distinct_clusters_beyond_3m():
    """Verify that detections > 3.0m apart create distinct clusters and work orders."""
    dedup = SpatialDeduplicator()

    det1 = {
        "pothole_id": "pothole-loc1",
        "latitude": 13.082700,
        "longitude": 80.270700,
        "confidence": 0.88,
        "severity_score": 0.75,
        "bbox_xyxy": [50, 50, 150, 150],
    }

    # ~45 meters away
    det2 = {
        "pothole_id": "pothole-loc2",
        "latitude": 13.083100,
        "longitude": 80.271100,
        "confidence": 0.90,
        "severity_score": 0.82,
        "bbox_xyxy": [80, 80, 180, 180],
    }

    r1 = dedup.ingest(det1)
    r2 = dedup.ingest(det2)

    assert r1["status"] == "new"
    assert r2["status"] == "new"
    assert r1["cluster_id"] != r2["cluster_id"]
    assert dedup.cluster_count() == 2
    assert len(dedup.get_all_work_orders()) == 2
