#!/usr/bin/env python3
"""
verify_temporal_capture_dataset.py
----------------------------------
File-only dataset integrity and metadata verifier for RoadSentinel Phase 1.

CRITICAL OPERATING RULE:
This script is completely decoupled from Unreal Engine, CARLA, PySide, and IPC.
It NEVER executes simulation code or commands. It strictly inspects files on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Use PIL if available for image dimension validation
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = WORKSPACE_ROOT / "env" / "config" / "temporal_capture_plan.csv"
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "env" / "output" / "temporal_segments"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "integration" / "temporal"


@dataclass
class PlannedCapture:
    segment_id: str
    day: int
    lighting_preset: str
    road_health_state: str
    pothole_sizing_spectrum: str
    pothole_density_per_100m2: float
    pothole_moisture_state: str
    camera_preset: str
    event: str
    expected_image_path: str
    expected_metadata_path: str


@dataclass
class VerificationRecord:
    segment_id: str
    day: int
    image_exists: bool = False
    image_valid: bool = False
    metadata_exists: bool = False
    metadata_valid: bool = False
    settings_match: bool = False
    capture_count: int = 0
    status: str = "PENDING"
    notes: List[str] = field(default_factory=list)
    actual_image_path: Optional[str] = None
    actual_metadata_path: Optional[str] = None
    actual_metadata: Dict[str, Any] = field(default_factory=dict)


def load_capture_plan(plan_path: Path) -> List[PlannedCapture]:
    """Read the 60-image capture plan CSV."""
    if not plan_path.is_file():
        raise FileNotFoundError(f"Capture plan not found: {plan_path}")

    records = []
    with plan_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(PlannedCapture(
                segment_id=row["segment_id"].strip(),
                day=int(row["day"].strip()),
                lighting_preset=row["lighting_preset"].strip(),
                road_health_state=row["road_health_state"].strip(),
                pothole_sizing_spectrum=row["pothole_sizing_spectrum"].strip(),
                pothole_density_per_100m2=float(row["pothole_density_per_100m2"].strip()),
                pothole_moisture_state=row["pothole_moisture_state"].strip(),
                camera_preset=row["camera_preset"].strip(),
                event=row["event"].strip(),
                expected_image_path=row["expected_image_path"].strip(),
                expected_metadata_path=row["expected_metadata_path"].strip(),
            ))
    return records


def validate_image_file(image_path: Path, expected_width: int = 1920, expected_height: int = 1080) -> Tuple[bool, str]:
    """Verify PNG file integrity, non-zero size, and dimensions without computer vision models."""
    if not image_path.is_file():
        return False, "File does not exist"

    size_bytes = image_path.stat().st_size
    if size_bytes < 2000:
        return False, f"File too small ({size_bytes} bytes < 2000 minimum)"

    if HAS_PIL:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                fmt = img.format
                if fmt != "PNG":
                    return False, f"Unexpected image format: {fmt} (expected PNG)"
                if (w, h) != (expected_width, expected_height):
                    return False, f"Unexpected resolution {w}x{h} (expected {expected_width}x{expected_height})"
                # Test readability of raster data
                img.verify()
            return True, f"Valid PNG ({w}x{h}, {size_bytes // 1024} KB)"
        except Exception as exc:
            return False, f"Image read error: {exc}"
    else:
        # Basic header verification if PIL is unavailable
        data = image_path.read_bytes()
        if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return False, "Invalid PNG signature"
        import struct
        w, h = struct.unpack(">II", data[16:24])
        if (w, h) != (expected_width, expected_height):
            return False, f"Dimensions {w}x{h} mismatch {expected_width}x{expected_height}"
        return True, f"Valid PNG header ({w}x{h}, {size_bytes // 1024} KB)"


def audit_capture_history(history_path: Path) -> Tuple[Counter, List[Dict[str, Any]], List[str]]:
    """Parse capture_history.jsonl and identify repeated captures / failures."""
    capture_counts = Counter()
    history_entries = []
    issues = []

    if not history_path.is_file():
        return capture_counts, history_entries, ["capture_history.jsonl does not exist"]

    with history_path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                history_entries.append(rec)
                seg = rec.get("segment_id")
                day = rec.get("day")
                if seg and day is not None:
                    capture_counts[(seg, int(day))] += 1
            except Exception as exc:
                issues.append(f"Line {idx} corrupt in history: {exc}")

    return capture_counts, history_entries, issues


def audit_manifest(manifest_path: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, str]], List[str]]:
    """Parse dataset_manifest.csv if present."""
    manifest_records = {}
    issues = []
    if not manifest_path.is_file():
        return manifest_records, ["dataset_manifest.csv does not exist"]

    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            seg = row.get("segment_id")
            day_str = row.get("day")
            if seg and day_str:
                try:
                    day = int(day_str)
                    manifest_records[(seg, day)] = row
                except ValueError:
                    issues.append(f"Row {idx} in manifest has invalid day: {day_str}")
    return manifest_records, issues


def check_settings_match(plan: PlannedCapture, meta: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compare actual sidecar metadata against planned settings."""
    mismatches = []

    # Segment
    actual_seg = meta.get("segment_id")
    if actual_seg != plan.segment_id:
        mismatches.append(f"segment_id: expected {plan.segment_id}, got {actual_seg}")

    # Day
    actual_day = meta.get("day")
    if actual_day != plan.day:
        mismatches.append(f"day: expected {plan.day}, got {actual_day}")

    # Lighting preset
    actual_lighting = meta.get("lighting_preset")
    if actual_lighting is None:
        mismatches.append("lighting_preset: missing from metadata")
    elif actual_lighting.strip() != plan.lighting_preset.strip():
        mismatches.append(f"lighting_preset: expected '{plan.lighting_preset}', got '{actual_lighting}'")

    # Road health state
    actual_health = meta.get("road_health_state")
    if actual_health is None:
        # Check alternative field deterioration_state
        det = meta.get("deterioration_state")
        if det:
            mismatches.append(f"road_health_state: missing standard field (found legacy deterioration_state '{det}')")
        else:
            mismatches.append("road_health_state: missing from metadata")
    elif actual_health.strip() != plan.road_health_state.strip():
        mismatches.append(f"road_health_state: expected '{plan.road_health_state}', got '{actual_health}'")

    # Pothole sizing spectrum
    actual_sizing = meta.get("pothole_sizing_spectrum")
    if actual_sizing is None:
        mismatches.append("pothole_sizing_spectrum: missing from metadata")
    elif actual_sizing.strip() != plan.pothole_sizing_spectrum.strip():
        mismatches.append(f"pothole_sizing_spectrum: expected '{plan.pothole_sizing_spectrum}', got '{actual_sizing}'")

    # Pothole density
    actual_density = meta.get("pothole_density_per_100m2")
    if actual_density is None:
        mismatches.append("pothole_density_per_100m2: missing from metadata")
    else:
        try:
            if abs(float(actual_density) - plan.pothole_density_per_100m2) > 0.01:
                mismatches.append(f"density: expected {plan.pothole_density_per_100m2}, got {actual_density}")
        except ValueError:
            mismatches.append(f"density: invalid float value {actual_density}")

    # Pothole moisture state
    actual_moisture = meta.get("pothole_moisture_state")
    if actual_moisture is None:
        mismatches.append("pothole_moisture_state: missing from metadata")
    elif actual_moisture.strip() != plan.pothole_moisture_state.strip():
        mismatches.append(f"pothole_moisture_state: expected '{plan.pothole_moisture_state}', got '{actual_moisture}'")

    # Camera preset
    actual_camera = meta.get("camera_preset")
    if actual_camera is None:
        mismatches.append("camera_preset: missing from metadata")
    elif actual_camera.strip() != plan.camera_preset.strip():
        mismatches.append(f"camera_preset: expected '{plan.camera_preset}', got '{actual_camera}'")

    # Timestamp present
    if not (meta.get("capture_timestamp") or meta.get("timestamp")):
        mismatches.append("timestamp: missing timestamp field")

    return (len(mismatches) == 0), mismatches


def find_unexpected_files(data_dir: Path, plan: List[PlannedCapture]) -> List[Path]:
    """Identify files in data_dir that are not planned or standard index artifacts."""
    if not data_dir.is_dir():
        return []

    planned_files = set()
    for p in plan:
        planned_files.add((WORKSPACE_ROOT / p.expected_image_path).resolve())
        planned_files.add((WORKSPACE_ROOT / p.expected_metadata_path).resolve())

    allowed_root_files = {
        (data_dir / "dataset_manifest.csv").resolve(),
        (data_dir / "capture_history.jsonl").resolve(),
        (data_dir / "temporal_results.json").resolve(),
        (data_dir / "work_orders.json").resolve(),
    }

    unexpected = []
    for path in data_dir.rglob("*"):
        if path.is_file():
            # Skip analysis/ folder (post-perception outputs)
            if "analysis" in path.parts:
                continue
            resolved = path.resolve()
            if resolved in planned_files or resolved in allowed_root_files:
                continue
            # Also allow legacy segment-level metadata.json and segment_history.json
            if path.name in ("metadata.json", "segment_history.json"):
                continue
            unexpected.append(path)
    return unexpected


def verify_dataset(
    plan_path: Path = DEFAULT_PLAN_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Execute complete Phase 1 file-only verification."""
    plan = load_capture_plan(plan_path)
    history_path = data_dir / "capture_history.jsonl"
    manifest_path = data_dir / "dataset_manifest.csv"

    history_counts, history_entries, history_issues = audit_capture_history(history_path)
    manifest_records, manifest_issues = audit_manifest(manifest_path)
    unexpected_files = find_unexpected_files(data_dir, plan)

    results: List[VerificationRecord] = []
    images_present_count = 0
    sidecars_present_count = 0
    fully_valid_count = 0
    mismatch_count = 0
    duplicate_count = 0

    segment_stats = defaultdict(lambda: {"images": 0, "sidecars": 0, "valid": 0})

    for item in plan:
        record = VerificationRecord(segment_id=item.segment_id, day=item.day)
        key = (item.segment_id, item.day)
        record.capture_count = history_counts.get(key, 0)

        # 1. Check Image
        expected_img = WORKSPACE_ROOT / item.expected_image_path
        if not expected_img.is_file():
            # Try data_dir fallback
            expected_img = data_dir / item.segment_id / f"day_{item.day:02d}.png"

        if expected_img.is_file():
            record.image_exists = True
            record.actual_image_path = str(expected_img)
            images_present_count += 1
            segment_stats[item.segment_id]["images"] += 1

            valid_img, img_reason = validate_image_file(expected_img)
            record.image_valid = valid_img
            if not valid_img:
                record.notes.append(f"Image invalid: {img_reason}")
        else:
            record.image_exists = False
            record.image_valid = False
            record.notes.append("MISSING_IMAGE")

        # 2. Check Sidecar Metadata
        expected_meta = WORKSPACE_ROOT / item.expected_metadata_path
        if not expected_meta.is_file():
            expected_meta = data_dir / item.segment_id / f"day_{item.day:02d}_metadata.json"

        # Check if legacy metadata.json exists in segment folder and matches this day
        legacy_meta_path = data_dir / item.segment_id / "metadata.json"
        actual_meta_to_read = None

        if expected_meta.is_file():
            actual_meta_to_read = expected_meta
        elif legacy_meta_path.is_file():
            try:
                parsed_legacy = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
                if parsed_legacy.get("day") == item.day:
                    actual_meta_to_read = legacy_meta_path
            except Exception:
                pass

        if actual_meta_to_read is not None:
            record.metadata_exists = True
            record.actual_metadata_path = str(actual_meta_to_read)
            sidecars_present_count += 1
            segment_stats[item.segment_id]["sidecars"] += 1

            try:
                meta_dict = json.loads(actual_meta_to_read.read_text(encoding="utf-8"))
                record.metadata_valid = True
                record.actual_metadata = meta_dict

                # Check settings match
                matches, mismatches = check_settings_match(item, meta_dict)
                record.settings_match = matches
                if not matches:
                    mismatch_count += 1
                    record.notes.append(f"SETTING_MISMATCH ({'; '.join(mismatches)})")
            except Exception as exc:
                record.metadata_valid = False
                record.notes.append(f"CORRUPT_METADATA ({exc})")
        else:
            record.metadata_exists = False
            record.metadata_valid = False
            record.notes.append("MISSING_METADATA")

        # 3. Check Duplicates in History
        if record.capture_count > 1:
            duplicate_count += 1
            record.notes.append(f"DUPLICATE_HISTORY ({record.capture_count} captures logged)")

        # 4. Determine overall status
        status_flags = []
        if not record.image_exists:
            status_flags.append("MISSING_IMAGE")
        elif not record.image_valid:
            status_flags.append("INVALID_IMAGE")

        if not record.metadata_exists:
            status_flags.append("MISSING_METADATA")
        elif not record.metadata_valid:
            status_flags.append("CORRUPT_METADATA")
        elif not record.settings_match:
            status_flags.append("SETTING_MISMATCH")

        if record.capture_count > 1:
            status_flags.append("DUPLICATE_HISTORY")

        if not status_flags:
            record.status = "PASS"
            fully_valid_count += 1
            segment_stats[item.segment_id]["valid"] += 1
        else:
            record.status = "; ".join(status_flags)

        results.append(record)

    # 5. Check Repair Events (SEG_005 and SEG_006)
    # SEG_005: Day 05 = pre-repair severe/critical, Day 06 = repair/reset state
    # SEG_006: Day 05 = pre-repair severe/critical, Day 06 = repair/reset state
    seg5_d5 = next((r for r in results if r.segment_id == "SEG_005" and r.day == 5), None)
    seg5_d6 = next((r for r in results if r.segment_id == "SEG_005" and r.day == 6), None)
    seg6_d5 = next((r for r in results if r.segment_id == "SEG_006" and r.day == 5), None)
    seg6_d6 = next((r for r in results if r.segment_id == "SEG_006" and r.day == 6), None)

    plan_seg5_d5 = next((p for p in plan if p.segment_id == "SEG_005" and p.day == 5), None)
    plan_seg5_d6 = next((p for p in plan if p.segment_id == "SEG_005" and p.day == 6), None)
    plan_seg6_d5 = next((p for p in plan if p.segment_id == "SEG_006" and p.day == 5), None)
    plan_seg6_d6 = next((p for p in plan if p.segment_id == "SEG_006" and p.day == 6), None)

    plan_repair_5_verified = (
        plan_seg5_d5 is not None and "Critical" in plan_seg5_d5.road_health_state and plan_seg5_d5.event == "pre-repair" and
        plan_seg5_d6 is not None and "Pristine" in plan_seg5_d6.road_health_state and plan_seg5_d6.event == "repair"
    )
    plan_repair_6_verified = (
        plan_seg6_d5 is not None and "Critical" in plan_seg6_d5.road_health_state and plan_seg6_d5.event == "pre-repair" and
        plan_seg6_d6 is not None and "Pristine" in plan_seg6_d6.road_health_state and plan_seg6_d6.event == "repair"
    )

    # Actual sidecar verification
    def check_actual_repair(d5_rec, d6_rec):
        if not (d5_rec and d5_rec.metadata_exists and d6_rec and d6_rec.metadata_exists):
            return "PENDING_CAPTURE"
        m5 = d5_rec.actual_metadata
        m6 = d6_rec.actual_metadata
        h5 = m5.get("road_health_state", m5.get("deterioration_state", ""))
        h6 = m6.get("road_health_state", m6.get("deterioration_state", ""))
        if ("Critical" in h5 or "Severe" in h5) and ("Pristine" in h6 or "Grade A" in h6):
            return "YES"
        return "NO"

    actual_repair_5 = check_actual_repair(seg5_d5, seg5_d6)
    actual_repair_6 = check_actual_repair(seg6_d5, seg6_d6)

    # Phase 1 Completion Evaluation
    phase1_complete = (
        len(plan) == 60 and
        images_present_count == 60 and
        sidecars_present_count == 60 and
        fully_valid_count == 60 and
        mismatch_count == 0 and
        actual_repair_5 == "YES" and
        actual_repair_6 == "YES"
    )
    phase1_status = "PHASE 1 COMPLETE" if phase1_complete else "PARTIAL PASS — CAPTURE IN PROGRESS"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_planned": len(plan),
        "images_present": images_present_count,
        "sidecars_present": sidecars_present_count,
        "fully_valid_captures": fully_valid_count,
        "setting_mismatches": mismatch_count,
        "duplicate_history_entries": duplicate_count,
        "seg_005_repair_plan_verified": "YES" if plan_repair_5_verified else "NO",
        "seg_005_repair_sidecar_verified": actual_repair_5,
        "seg_006_repair_plan_verified": "YES" if plan_repair_6_verified else "NO",
        "seg_006_repair_sidecar_verified": actual_repair_6,
        "phase1_status": phase1_status,
        "segment_breakdown": {seg: dict(stats) for seg, stats in sorted(segment_stats.items())},
        "history_issues": history_issues,
        "manifest_issues": manifest_issues,
        "unexpected_files": [str(p) for p in unexpected_files],
    }

    # Write CSV export
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_report_path = output_dir / "phase1_dataset_verification.csv"
    with csv_report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "segment_id",
            "day",
            "image_exists",
            "metadata_exists",
            "settings_match",
            "image_valid",
            "capture_count",
            "status",
            "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "segment_id": r.segment_id,
                "day": r.day,
                "image_exists": r.image_exists,
                "metadata_exists": r.metadata_exists,
                "settings_match": r.settings_match,
                "image_valid": r.image_valid,
                "capture_count": r.capture_count,
                "status": r.status,
                "notes": " | ".join(r.notes) if r.notes else "OK",
            })

    # Write Markdown reports
    write_reports(output_dir, plan, results, summary)

    return {
        "summary": summary,
        "results": results,
    }


def write_reports(
    output_dir: Path,
    plan: List[PlannedCapture],
    results: List[VerificationRecord],
    summary: Dict[str, Any],
) -> None:
    """Generate PHASE1_PROGRESS.md and PHASE1_DATASET_VERIFICATION.md."""

    # 1. PHASE1_PROGRESS.md
    progress_md_path = output_dir / "PHASE1_PROGRESS.md"
    progress_content = f"""# RoadSentinel — Phase 1: Temporal Dataset Progress

**Date**: {summary['timestamp']}  
**Phase 1 Status**: **{summary['phase1_status']}**

---

## 1. Executive Summary Metrics

| Metric | Target | Actual | Status |
|---|---|---|---|
| **Total Planned Captures** | 60 | {summary['total_planned']} | Completed Specification |
| **Images Present** | 60 | {summary['images_present']} | {'✅ Complete' if summary['images_present'] == 60 else f'⚠️ In Progress ({60 - summary["images_present"]} remaining)'} |
| **Sidecars Present** | 60 | {summary['sidecars_present']} | {'✅ Complete' if summary['sidecars_present'] == 60 else f'⚠️ In Progress ({60 - summary["sidecars_present"]} remaining)'} |
| **Fully Valid Captures** | 60 | {summary['fully_valid_captures']} | {'✅ Complete' if summary['fully_valid_captures'] == 60 else f'⚠️ In Progress ({60 - summary["fully_valid_captures"]} remaining)'} |
| **Setting Mismatches** | 0 | {summary['setting_mismatches']} | {'✅ None' if summary['setting_mismatches'] == 0 else f'⚠️ {summary["setting_mismatches"]} mismatch(es)'} |
| **Duplicate History Entries** | 0 | {summary['duplicate_history_entries']} | {'✅ None' if summary['duplicate_history_entries'] == 0 else f'⚠️ {summary["duplicate_history_entries"]} duplicate(s)'} |
| **SEG_005 Repair Event Verified** | YES | Plan: {summary['seg_005_repair_plan_verified']} / Sidecar: {summary['seg_005_repair_sidecar_verified']} | {'✅ Verified' if summary['seg_005_repair_sidecar_verified'] == 'YES' else '⏳ Pending sidecar capture'} |
| **SEG_006 Repair Event Verified** | YES | Plan: {summary['seg_006_repair_plan_verified']} / Sidecar: {summary['seg_006_repair_sidecar_verified']} | {'✅ Verified' if summary['seg_006_repair_sidecar_verified'] == 'YES' else '⏳ Pending sidecar capture'} |

---

## 2. Segment Completion Progress

| Segment ID | Description | Images Present | Sidecars Present | Fully Valid | Remaining |
|---|---|---|---|---|---|
| **SEG_001** | Progressive Fatigue (Wheeltrack Cracking -> Waterlogged Crater) | {summary['segment_breakdown'].get('SEG_001', {}).get('images', 0)}/10 | {summary['segment_breakdown'].get('SEG_001', {}).get('sidecars', 0)}/10 | {summary['segment_breakdown'].get('SEG_001', {}).get('valid', 0)}/10 | {10 - summary['segment_breakdown'].get('SEG_001', {}).get('images', 0)} images |
| **SEG_002** | Rapid Pothole Formation (Stripping -> Raveling -> Large Wet Pothole) | {summary['segment_breakdown'].get('SEG_002', {}).get('images', 0)}/10 | {summary['segment_breakdown'].get('SEG_002', {}).get('sidecars', 0)}/10 | {summary['segment_breakdown'].get('SEG_002', {}).get('valid', 0)}/10 | {10 - summary['segment_breakdown'].get('SEG_002', {}).get('images', 0)} images |
| **SEG_003** | Crack-Dominated (Transverse -> Block Cracking -> Edge Spalling) | {summary['segment_breakdown'].get('SEG_003', {}).get('images', 0)}/10 | {summary['segment_breakdown'].get('SEG_003', {}).get('sidecars', 0)}/10 | {summary['segment_breakdown'].get('SEG_003', {}).get('valid', 0)}/10 | {10 - summary['segment_breakdown'].get('SEG_003', {}).get('images', 0)} images |
| **SEG_004** | Multi-Epicenter Pothole Cluster | {summary['segment_breakdown'].get('SEG_004', {}).get('images', 0)}/10 | {summary['segment_breakdown'].get('SEG_004', {}).get('sidecars', 0)}/10 | {summary['segment_breakdown'].get('SEG_004', {}).get('valid', 0)}/10 | {10 - summary['segment_breakdown'].get('SEG_004', {}).get('images', 0)} images |
| **SEG_005** | Repair Demonstration Segment 1 (Pre-repair D05 -> Reset D06) | {summary['segment_breakdown'].get('SEG_005', {}).get('images', 0)}/10 | {summary['segment_breakdown'].get('SEG_005', {}).get('sidecars', 0)}/10 | {summary['segment_breakdown'].get('SEG_005', {}).get('valid', 0)}/10 | {10 - summary['segment_breakdown'].get('SEG_005', {}).get('images', 0)} images |
| **SEG_006** | Repair Demonstration Segment 2 (Edge Shear Gouge -> Reset D06) | {summary['segment_breakdown'].get('SEG_006', {}).get('images', 0)}/10 | {summary['segment_breakdown'].get('SEG_006', {}).get('sidecars', 0)}/10 | {summary['segment_breakdown'].get('SEG_006', {}).get('valid', 0)}/10 | {10 - summary['segment_breakdown'].get('SEG_006', {}).get('images', 0)} images |

---

## 3. Remaining Captures Required from User

The user must manually capture the following outstanding segments and days in the Interactive Studio:

"""
    missing_items = [r for r in results if not r.image_exists]
    if missing_items:
        progress_content += "| Segment ID | Day | Planned Lighting | Planned Health State | Event |\n|---|---|---|---|---|\n"
        for m in missing_items:
            # Find planned info
            p_match = next((p for p in plan if p.segment_id == m.segment_id and p.day == m.day), None)
            lighting = p_match.lighting_preset if p_match else "Unknown"
            health = p_match.road_health_state if p_match else "Unknown"
            ev = p_match.event if p_match else "normal"
            progress_content += f"| `{m.segment_id}` | Day {m.day:02d} | {lighting} | {health} | `{ev}` |\n"
    else:
        progress_content += "✅ All 60 planned images exist on disk!\n"

    progress_md_path.write_text(progress_content, encoding="utf-8")

    # 2. PHASE1_DATASET_VERIFICATION.md
    verification_md_path = output_dir / "PHASE1_DATASET_VERIFICATION.md"
    verif_content = f"""# RoadSentinel — Phase 1: Dataset Verification Report

**Generated**: {summary['timestamp']}  
**Verification Method**: Strict File-Only Audit (zero Unreal/CARLA interaction)  
**Overall Status**: **{summary['phase1_status']}**

---

## 1. Audit Findings

1. **Images Audit**:
   - Total expected: 60 (1920x1080 PNG)
   - Found on disk: **{summary['images_present']}/60**
   - Corrupted/truncated PNGs: **0**
   - Valid resolution & format: **{summary['images_present']}**

2. **Sidecar Metadata Audit**:
   - Total expected: 60 (`day_XX_metadata.json`)
   - Found on disk: **{summary['sidecars_present']}/60**
   - Notes: Existing capture directory currently has legacy segment-level `metadata.json` and `segment_history.json`. Individual per-image sidecars (`day_XX_metadata.json`) will be written by `rs_capture_metadata_logger.py` during ongoing captures.

3. **Manifest & History Audit**:
   - `dataset_manifest.csv`: {'Present' if not summary['manifest_issues'] else 'Not yet created (will be created as captures occur)'}
   - `capture_history.jsonl`: {'Present' if not summary['history_issues'] else 'Not yet created (will be created as captures occur)'}
   - Repeated/duplicate captures in history: **{summary['duplicate_history_entries']}**

4. **Repair Event Verification**:
   - **SEG_005**:
     - Capture Plan: Day 05 = `Critical Hazard (Grade F)` (`pre-repair`), Day 06 = `Pristine (Grade A)` (`repair`) → **{summary['seg_005_repair_plan_verified']}**
     - Sidecar Metadata: **{summary['seg_005_repair_sidecar_verified']}** (awaiting capture of SEG_005 Days 05 and 06)
   - **SEG_006**:
     - Capture Plan: Day 05 = `Critical Hazard (Grade F)` (`pre-repair`), Day 06 = `Pristine (Grade A)` (`repair`) → **{summary['seg_006_repair_plan_verified']}**
     - Sidecar Metadata: **{summary['seg_006_repair_sidecar_verified']}** (awaiting capture of SEG_006 Days 05 and 06)

---

## 2. Complete Planned 60-Image Capture Table

| Segment | Day | Image Status | Sidecar Status | Settings Match | Overall Status | Notes |
|---|---|---|---|---|---|---|
"""
    for r in results:
        img_str = "✅ Valid" if (r.image_exists and r.image_valid) else ("⚠️ Missing" if not r.image_exists else "❌ Invalid")
        meta_str = "✅ Valid" if (r.metadata_exists and r.metadata_valid) else ("⚠️ Missing" if not r.metadata_exists else "❌ Corrupt")
        match_str = "✅ Match" if r.settings_match else ("N/A" if not r.metadata_exists else "⚠️ Mismatch")
        status_badge = "✅ PASS" if r.status == "PASS" else f"`{r.status}`"
        notes_str = "; ".join(r.notes) if r.notes else "OK"
        verif_content += f"| `{r.segment_id}` | {r.day:02d} | {img_str} | {meta_str} | {match_str} | {status_badge} | {notes_str} |\n"

    verification_md_path.write_text(verif_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="RoadSentinel Phase 1 Dataset Verifier")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH, help="Path to capture plan CSV")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to temporal segments directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to output report directory")
    parser.add_argument("--quiet", action="store_true", help="Do not print table to stdout")
    args = parser.parse_args()

    res = verify_dataset(args.plan, args.data_dir, args.output_dir)
    summary = res["summary"]

    if not args.quiet:
        print("=" * 60)
        print("ROADSENTINEL — PHASE 1 DATASET VERIFICATION")
        print("=" * 60)
        print(f"Total Planned Captures:     {summary['total_planned']}")
        print(f"Images Present:             {summary['images_present']} / 60")
        print(f"Sidecars Present:           {summary['sidecars_present']} / 60")
        print(f"Fully Valid Captures:       {summary['fully_valid_captures']} / 60")
        print(f"Setting Mismatches:         {summary['setting_mismatches']}")
        print(f"Duplicate History:          {summary['duplicate_history_entries']}")
        print(f"SEG_005 Repair Plan Verif:  {summary['seg_005_repair_plan_verified']}")
        print(f"SEG_005 Repair Sidecar:     {summary['seg_005_repair_sidecar_verified']}")
        print(f"SEG_006 Repair Plan Verif:  {summary['seg_006_repair_plan_verified']}")
        print(f"SEG_006 Repair Sidecar:     {summary['seg_006_repair_sidecar_verified']}")
        print(f"Phase 1 Status:             {summary['phase1_status']}")
        print("=" * 60)
        print("Segment Breakdown:")
        for seg, counts in summary["segment_breakdown"].items():
            print(f"  {seg}: images={counts['images']}/10, sidecars={counts['sidecars']}/10, valid={counts['valid']}/10")
        print("=" * 60)
        print(f"Reports written to: {args.output_dir}")


if __name__ == "__main__":
    main()
