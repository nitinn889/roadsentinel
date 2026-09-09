#!/usr/bin/env python3
"""audit_captures.py
------------------
Audits Experiment B captures (SEG_005 and SEG_006) for camera consistency,
lighting invariance, day sequence integrity, and repair intervention metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import cv2

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_SEGMENTS_ROOT = WORKSPACE_ROOT / "env" / "output" / "temporal_segments"
LEGACY_MANUAL_ROOT = WORKSPACE_ROOT / "env" / "output" / "manual_inspections"
LEGACY_20DAY_ROOT = WORKSPACE_ROOT / "env" / "output" / "temporal_20_day"


def audit_segment_in_temporal_segments(segment_id: str) -> Dict[str, Any]:
    seg_dir = TEMPORAL_SEGMENTS_ROOT / segment_id
    report: Dict[str, Any] = {
        "segment_id": segment_id,
        "source_directory": str(seg_dir),
        "directory_exists": seg_dir.exists(),
        "images_found": 0,
        "metadata_found": 0,
        "days_present": [],
        "camera_consistency": "UNKNOWN",
        "lighting_consistency": "UNKNOWN",
        "health_state_sequence": [],
        "repair_metadata_present": False,
        "status": "MISSING",
        "issues": [],
    }

    if not seg_dir.exists():
        report["issues"].append(f"Directory {seg_dir} does not exist.")
        return report

    cameras = set()
    lightings = set()
    for d in range(1, 11):
        day_str = f"day_{d:02d}"
        img_path = seg_dir / f"{day_str}.png"
        meta_path = seg_dir / f"{day_str}_metadata.json"

        img_ok = False
        if img_path.exists():
            report["images_found"] += 1
            img = cv2.imread(str(img_path))
            if img is not None:
                img_ok = True
                h, w, _ = img.shape
                if (h, w) != (1080, 1920):
                    report["issues"].append(f"Day {d:02d} image resolution {w}x{h} != 1920x1080")
            else:
                report["issues"].append(f"Day {d:02d} image {img_path} unreadable")

        if meta_path.exists():
            report["metadata_found"] += 1
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cam = meta.get("camera_preset") or meta.get("camera_mode")
                if cam:
                    cameras.add(cam)
                light = meta.get("lighting_preset") or meta.get("weather_preset")
                if light:
                    lightings.add(light)
                state = meta.get("road_health_state") or meta.get("health_state") or meta.get("condition")
                report["health_state_sequence"].append(f"Day{d:02d}:{state}")
                if d == 6 and ("REPAIR" in str(meta) or "Repair" in str(meta) or "Pristine" in str(state)):
                    report["repair_metadata_present"] = True
            except Exception as e:
                report["issues"].append(f"Day {d:02d} metadata parse error: {e}")

        if img_ok and meta_path.exists():
            report["days_present"].append(d)

    report["camera_consistency"] = "FIXED" if len(cameras) == 1 else ("MIXED" if len(cameras) > 1 else "MISSING")
    report["lighting_consistency"] = "FIXED" if len(lightings) == 1 else ("MIXED" if len(lightings) > 1 else "MISSING")
    report["status"] = "COMPLETE" if len(report["days_present"]) == 10 else f"PARTIAL_{len(report['days_present'])}/10"
    return report


def main() -> None:
    print("==================================================")
    print("ROADSENTINEL EXPERIMENT B CAPTURE AUDIT")
    print("==================================================")
    
    seg5_report = audit_segment_in_temporal_segments("SEG_005")
    seg6_report = audit_segment_in_temporal_segments("SEG_006")

    print("\n--- SEG_005 AUDIT ---")
    print(json.dumps(seg5_report, indent=2))

    print("\n--- SEG_006 AUDIT ---")
    print(json.dumps(seg6_report, indent=2))


if __name__ == "__main__":
    main()
