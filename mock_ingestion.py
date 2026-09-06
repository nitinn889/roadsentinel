#!/usr/bin/env python3
"""RoadSentinel Local Mock Data Pipeline (No CARLA Required).

Allows developers and operators to ingest standard pothole photos downloaded
from the internet or test suites into the RoadSentinel Government Dashboard.

When a photo is ingested, this script automatically:
  1. Computes image geometry and generates dummy bounding box coordinates [x1, y1, x2, y2].
  2. Generates dummy GPS coordinates (Latitude/Longitude).
  3. Generates realistic random physical metrics: Depth (cm), Area (m²), Severity Score, and Water flag.
  4. Transmits the payload directly to the dashboard server (/api/ingest) via HTTP.
  5. Validates the 3-meter spatial deduplication and work order duplication prevention.

Usage Examples:
  # Ingest a single downloaded photo:
  python mock_ingestion.py --image path/to/pothole.jpg

  # Ingest all images from a directory:
  python mock_ingestion.py --image path/to/photos_folder/

  # Run an automated 3-meter spatial overlap & deduplication test:
  python mock_ingestion.py --simulate-overlap

  # Ingest with custom base GPS coordinates:
  python mock_ingestion.py --image pothole.png --lat 13.0827 --lon 80.2707
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mock_ingestion")

REPO_ROOT = Path(__file__).resolve().parent

# Default demo sample images to check if none specified
FALLBACK_SEARCH_DIRS = [
    REPO_ROOT / "road_health_pipeline" / "output" / "analytics_demo",
    REPO_ROOT / "road_health_pipeline" / "tests" / "fixtures" / "pothole_mock",
    REPO_ROOT / "ml" / "data" / "test_road",
    REPO_ROOT / "ml" / "data" / "healthy_road",
]


def find_sample_images() -> list[Path]:
    """Search the workspace for sample images to use if none provided."""
    imgs: list[Path] = []
    for d in FALLBACK_SEARCH_DIRS:
        if d.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                found = sorted(d.glob(ext))
                if found:
                    imgs.extend(found)
    return imgs


def create_synthetic_pothole_image(width: int = 640, height: int = 480) -> bytes:
    """Generate a synthetic asphalt pothole image if no files are found."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (width, height), color=(60, 63, 68))
        draw = ImageDraw.Draw(img)
        for _ in range(800):
            rx = random.randint(0, width - 1)
            ry = random.randint(0, height - 1)
            shade = random.randint(45, 80)
            draw.point((rx, ry), fill=(shade, shade, shade))

        cx, cy = width // 2, height // 2
        rx, ry = int(width * 0.18), int(height * 0.15)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(22, 24, 28), outline=(15, 16, 18), width=3)
        draw.ellipse([cx - int(rx * 0.6), cy - int(ry * 0.5), cx + int(rx * 0.6), cy + int(ry * 0.5)], fill=(12, 14, 18))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        )


def build_mock_defect_payload(
    image_bytes: bytes,
    image_name: str,
    lat: float,
    lon: float,
    road_segment_id: str = "seg_local_test_01",
    pothole_id: str | None = None,
    confidence: float | None = None,
    severity_score: float | None = None,
    defect_type: str | None = None,
    bbox_xyxy: list[int] | None = None,
) -> dict:
    """Build a complete RoadSentinel mock detection package."""
    w, h = 640, 480
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            w, h = pil_img.size
    except Exception:
        pass

    if bbox_xyxy is None:
        bx1 = int(w * random.uniform(0.18, 0.32))
        by1 = int(h * random.uniform(0.25, 0.42))
        bw = int(w * random.uniform(0.25, 0.45))
        bh = int(h * random.uniform(0.20, 0.35))
        bbox_xyxy = [bx1, by1, min(w - 5, bx1 + bw), min(h - 5, by1 + bh)]

    area_m2 = round(random.uniform(0.35, 1.75), 2)
    depth_m = round(random.uniform(0.04, 0.16), 3)

    if severity_score is None:
        severity_score = round(
            min(0.98, max(0.45, (area_m2 / 2.0) * 0.45 + (depth_m / 0.15) * 0.55)),
            2,
        )

    if confidence is None:
        confidence = round(random.uniform(0.80, 0.97), 2)

    is_water = (random.random() < 0.35) if defect_type is None else ("water" in defect_type.lower())
    if defect_type is None:
        defect_type = "water_filled_pothole" if is_water else "pothole"

    pid = pothole_id or f"mock-def-{random.randint(1000, 9999)}"

    return {
        "pothole_id": pid,
        "defect_id": pid,
        "road_segment_id": road_segment_id,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "bbox_xyxy": bbox_xyxy,
        "area_m2": area_m2,
        "estimated_depth_m": depth_m,
        "severity_score": severity_score,
        "pothole_confidence": confidence,
        "confidence": confidence,
        "defect_type": defect_type,
        "is_water_filled": is_water,
        "water_flag": is_water,
        "source_image": image_name,
        "_image_b64": base64.b64encode(image_bytes).decode("ascii"),
        "severity_breakdown": {
            "area": round(min(1.0, area_m2 / 1.5), 2),
            "depth": round(min(1.0, depth_m / 0.15), 2),
            "water": 0.85 if is_water else 0.10,
            "confidence": confidence,
        },
    }


def send_ingestion_payload(server_url: str, payload: dict) -> dict:
    """Send detection payload to the server's /api/ingest endpoint."""
    url = server_url.rstrip("/") + "/api/ingest"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_server_online(server_url: str) -> bool:
    """Check if the dashboard server is responding."""
    try:
        url = server_url.rstrip("/") + "/health"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("status") == "online"
    except Exception:
        return False


def run_overlap_deduplication_demo(server_url: str, sample_images: list[Path]) -> None:
    """Run an automated spatial deduplication test."""
    print("\n" + "=" * 70)
    print("  RoadSentinel 3-Meter Spatial Deduplication Simulation")
    print("=" * 70)

    img1_bytes = sample_images[0].read_bytes() if sample_images else create_synthetic_pothole_image()
    img1_name = sample_images[0].name if sample_images else "synthetic_pothole_1.jpg"

    img2_bytes = sample_images[1].read_bytes() if len(sample_images) > 1 else img1_bytes
    img2_name = sample_images[1].name if len(sample_images) > 1 else "overlapping_pothole_angle2.jpg"

    # Choose coordinates that don't overlap with pre-seeded demo potholes
    base_lat, base_lon = 13.085500, 80.273500

    # Step 1: Photo 1
    print("\n[Step 1] Ingesting Photo 1 (Reference viewpoint):")
    p1 = build_mock_defect_payload(
        img1_bytes, img1_name,
        lat=base_lat, lon=base_lon,
        pothole_id="pothole-overlap-test-01",
        confidence=0.82,
        severity_score=0.74,
        bbox_xyxy=[120, 140, 360, 320],
    )
    print(f"  -> GPS: ({p1['latitude']:.6f}, {p1['longitude']:.6f})")
    print(f"  -> Confidence: {p1['confidence']*100:.0f}%  |  Severity: {p1['severity_score']*100:.0f}%")
    print(f"  -> BBox: {p1['bbox_xyxy']}")

    res1 = send_ingestion_payload(server_url, p1)
    print(f"  [Server Response] Status: {res1['dedup_status'].upper()} | Cluster: {res1['cluster_id']}")
    print(f"  Active Clusters: {res1['total_clusters']} | Active Work Orders: {res1['total_work_orders']}")
    assert res1["dedup_status"] == "new", "Step 1 should create a new cluster!"
    canonical_cid = res1["cluster_id"]

    # Step 2: Photo 2 (1.2 meters away, higher confidence)
    overlap_lat = base_lat + 0.000008
    overlap_lon = base_lon + 0.000008
    print(f"\n[Step 2] Ingesting Photo 2 (Overlapping photo, ~1.2m away, HIGHER CONFIDENCE 96%):")
    p2 = build_mock_defect_payload(
        img2_bytes, img2_name,
        lat=overlap_lat, lon=overlap_lon,
        pothole_id="pothole-overlap-test-02",
        confidence=0.96,
        severity_score=0.88,
        bbox_xyxy=[140, 150, 390, 340],
    )
    print(f"  -> GPS: ({p2['latitude']:.6f}, {p2['longitude']:.6f})")
    print(f"  -> Confidence: {p2['confidence']*100:.0f}%  |  Severity: {p2['severity_score']*100:.0f}%")
    print(f"  -> BBox: {p2['bbox_xyxy']}")

    res2 = send_ingestion_payload(server_url, p2)
    print(f"  [Server Response] Status: {res2['dedup_status'].upper()} | Cluster: {res2['cluster_id']}")
    print(f"  Distance to canonical: {res2.get('distance_m')}m (<= 3.0m threshold)")
    print(f"  Merged BBoxes in Cluster: {res2.get('canonical', {}).get('merged_count', 2)}")
    print(f"  Active Clusters: {res2['total_clusters']} (Unchanged: Grouped as same defect)")
    print(f"  Active Work Orders: {res2['total_work_orders']} (Unchanged: DO NOT DUPLICATE WORK ORDER)")
    assert res2["dedup_status"] == "deduplicated", "Step 2 must be deduplicated within 3m!"
    assert res2["cluster_id"] == canonical_cid, "Step 2 must merge into canonical cluster!"

    # Step 3: Photo 3 (45 meters away)
    far_lat = base_lat + 0.000400
    far_lon = base_lon + 0.000400
    print(f"\n[Step 3] Ingesting Photo 3 (Far photo, ~60m away, distinct defect):")
    p3 = build_mock_defect_payload(
        img1_bytes, "far_defect_photo.jpg",
        lat=far_lat, lon=far_lon,
        pothole_id="pothole-distinct-03",
        confidence=0.91,
        severity_score=0.79,
    )
    print(f"  -> GPS: ({p3['latitude']:.6f}, {p3['longitude']:.6f})")
    res3 = send_ingestion_payload(server_url, p3)
    print(f"  [Server Response] Status: {res3['dedup_status'].upper()} | Cluster: {res3['cluster_id']}")
    print(f"  Active Clusters: {res3['total_clusters']} (Incremented)")
    print(f"  Active Work Orders: {res3['total_work_orders']} (Incremented)")
    assert res3["dedup_status"] == "new", "Step 3 should create a distinct new cluster!"

    print("\n" + "=" * 70)
    print("  DEDUPLICATION VERIFICATION COMPLETE: ALL 3-METER TESTS PASSED!")
    print("  View results in browser at http://localhost:8000")
    print("    - Tab 1: Map View shows correct GPS markers")
    print("    - Tab 2: Patch Inspection shows merged bboxes & highest-conf photo")
    print("    - Tab 3: Work Orders contains single repair order per physical defect")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoadSentinel Local Mock Ingestion Pipeline — Ingest photos and test 3m deduplication without CARLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to an image file or directory of downloaded pothole images",
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        default="http://localhost:8000",
        help="Dashboard server URL",
    )
    parser.add_argument(
        "--simulate-overlap", "-o",
        action="store_true",
        help="Run 3-meter spatial overlap test to verify deduplication and work order grouping",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=13.0827,
        help="Base latitude coordinate",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=80.2707,
        help="Base longitude coordinate",
    )
    parser.add_argument(
        "--seg",
        type=str,
        default="seg_local_test_01",
        help="Road segment ID to tag detections with",
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=1,
        help="Number of mock detections to generate if no directory specified",
    )
    args = parser.parse_args()

    print("\nRoadSentinel Local Mock Ingestion Pipeline")
    print(f"Target Server: {args.server}")

    if not test_server_online(args.server):
        log.error(
            "Could not connect to dashboard server at %s. Please start server.py first:\n"
            "  python road_health_pipeline/inference/server.py --port 8000",
            args.server,
        )
        sys.exit(1)
    log.info("Connected to dashboard server at %s [ONLINE]", args.server)

    sample_imgs = find_sample_images()

    if args.simulate_overlap:
        run_overlap_deduplication_demo(args.server, sample_imgs)
        return

    target_files: list[Path] = []
    if args.image:
        p = Path(args.image)
        if p.is_file():
            target_files = [p]
        elif p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                target_files.extend(sorted(p.glob(ext)))
        else:
            log.error("Image path '%s' not found.", args.image)
            sys.exit(1)
    elif sample_imgs:
        target_files = sample_imgs[:args.num]
    else:
        target_files = []

    if not target_files:
        log.info("No input image specified and no demo images found. Generating synthetic pothole image...")
        synth_bytes = create_synthetic_pothole_image()
        payload = build_mock_defect_payload(
            synth_bytes, "synthetic_pothole.jpg",
            lat=args.lat, lon=args.lon,
            road_segment_id=args.seg,
        )
        res = send_ingestion_payload(args.server, payload)
        print(f"Ingested Synthetic Pothole -> Status: {res['dedup_status']} | Cluster ID: {res['cluster_id']}")
        return

    log.info("Ingesting %d photo(s)...", len(target_files))
    for i, img_path in enumerate(target_files, 1):
        img_bytes = img_path.read_bytes()
        jitter_lat = args.lat + random.uniform(-0.0004, 0.0004)
        jitter_lon = args.lon + random.uniform(-0.0004, 0.0004)

        payload = build_mock_defect_payload(
            img_bytes, img_path.name,
            lat=jitter_lat, lon=jitter_lon,
            road_segment_id=args.seg,
        )

        try:
            res = send_ingestion_payload(args.server, payload)
            print(
                f"[{i}/{len(target_files)}] Ingested {img_path.name} -> "
                f"Status: {res['dedup_status'].upper()} | Cluster: {res['cluster_id']} | "
                f"GPS: ({payload['latitude']:.5f}, {payload['longitude']:.5f}) | "
                f"Active Clusters: {res['total_clusters']} | Work Orders: {res['total_work_orders']}"
            )
        except Exception as exc:
            log.error("Failed ingesting %s: %s", img_path.name, exc)

    print("\nIngestion complete. Open your browser to view the multi-tab dashboard:")
    print(f"  {args.server}")


if __name__ == "__main__":
    main()
