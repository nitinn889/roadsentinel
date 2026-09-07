"""
rs_inspection_capture.py
------------------------
RoadSentinel — User-Driven Inspection Capture Helper.

Provides:
  - build_output_dir()  : deterministic output path for any day/segment pair.
  - validate_png()      : reject blank, tiny, all-white, or all-black frames.
  - write_metadata()    : persist capture provenance as metadata.json.

This module is intentionally free of Unreal imports so that it can be unit-
tested in any Python environment.  The SceneCapture2D rendering itself lives
in rs_phase2_surface_defects.py which runs inside Unreal Editor.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_SEGMENTS_ROOT = WORKSPACE_ROOT / "env" / "output" / "temporal_segments"
MANUAL_INSPECTIONS_ROOT = TEMPORAL_SEGMENTS_ROOT
MANIFESTS_ROOT = WORKSPACE_ROOT / "env" / "output" / "temporal_20_day" / "manifests"

# Segments that exist in the system.
KNOWN_SEGMENT_IDS = [f"SEG_{n:03d}" for n in range(1, 7)]  # SEG_001 … SEG_006


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_output_dir(day: int, segment_id: str, root: Optional[Path] = None) -> Path:
    """Return the segment-wise directory for an inspection capture.

    Structure: env/output/temporal_segments/SEG_XXX/

    Args:
        day:        Inspection day (1–10).
        segment_id: Canonical segment identifier, e.g. "SEG_003".
        root:       Override root; defaults to workspace TEMPORAL_SEGMENTS_ROOT.

    Returns:
        A ``Path`` to the segment folder (e.g. env/output/temporal_segments/SEG_003/).
    """
    if not 1 <= day <= 20:
        raise ValueError(f"day must be 1–20, got {day!r}")
    segment_id = segment_id.strip().upper()
    if segment_id not in KNOWN_SEGMENT_IDS:
        raise ValueError(
            f"segment_id {segment_id!r} is not a known segment "
            f"({KNOWN_SEGMENT_IDS[0]}–{KNOWN_SEGMENT_IDS[-1]})"
        )
    base = root if root is not None else TEMPORAL_SEGMENTS_ROOT
    return base / segment_id


def build_image_path(day: int, segment_id: str, root: Optional[Path] = None) -> Path:
    """Return the deterministic image file path for day/segment.

    Example: env/output/temporal_segments/SEG_001/day_05.png
    """
    seg_dir = build_output_dir(day, segment_id, root)
    return seg_dir / f"day_{day:02d}.png"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(day: int, manifests_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load and return the parsed JSON for one temporal day manifest.

    Raises:
        FileNotFoundError: if the manifest file is absent.
        ValueError:        if the schema_version is wrong or day mismatches.
    """
    root = manifests_root if manifests_root is not None else MANIFESTS_ROOT
    path = root / f"day_{day:02d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Day {day} manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "RoadSentinelTemporalManifest/v1":
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema_version')!r}")
    if int(payload.get("day", 0)) != day:
        raise ValueError(
            f"Manifest day mismatch: file says day {payload.get('day')}, expected {day}"
        )
    return payload


def get_segment_info(manifest: Dict[str, Any], segment_id: str) -> Dict[str, Any]:
    """Return the segment entry for *segment_id* from a parsed manifest dict.

    Raises:
        KeyError: if segment_id is not in the manifest.
    """
    for seg in manifest.get("segments", []):
        if str(seg.get("road_segment_id", "")) == segment_id:
            return seg
    raise KeyError(f"segment_id {segment_id!r} not found in manifest for day {manifest.get('day')}")


# ---------------------------------------------------------------------------
# PNG validation
# ---------------------------------------------------------------------------

def _read_png_ihdr(data: bytes) -> tuple[int, int]:
    """Parse width and height from a PNG byte-string without PIL/numpy."""
    # PNG signature is 8 bytes; IHDR chunk is next 25 bytes.
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a valid PNG file")
    # IHDR data starts at offset 16 (4 len + 4 type + data[0:4]=width, data[4:8]=height)
    width = struct.unpack(">I", data[16:20])[0]
    height = struct.unpack(">I", data[20:24])[0]
    return width, height


def validate_png(
    path: Path,
    expected_width: int = 1920,
    expected_height: int = 1080,
    min_size_bytes: int = 2_000,
    near_uniform_tolerance: float = 0.98,
) -> dict:
    """Validate that *path* is a genuine UE road render.

    Checks performed:
      1. File exists and is non-empty (>= ``min_size_bytes``).
      2. Valid PNG header.
      3. Width/height match ``expected_width`` x ``expected_height``.
      4. Not nearly all-white or all-black (>= ``near_uniform_tolerance`` of
         pixels would need to be at the extreme to trigger rejection).

    The colour check uses only the PNG IDAT stream so it works without
    numpy/PIL.  It is approximate: it checks the frequency of byte value 0x00
    and 0xFF in the uncompressed scanline data.

    Returns:
        dict with keys: ``valid`` (bool), ``reason`` (str | None),
        ``file_size_bytes``, ``width``, ``height``.
    """
    result: Dict[str, Any] = {
        "valid": False,
        "reason": None,
        "file_size_bytes": 0,
        "width": 0,
        "height": 0,
    }

    if not path.is_file():
        result["reason"] = f"File does not exist: {path}"
        return result

    data = path.read_bytes()
    result["file_size_bytes"] = len(data)

    if len(data) < min_size_bytes:
        result["reason"] = (
            f"File too small ({len(data)} bytes < {min_size_bytes} minimum); "
            "likely an empty or failed capture"
        )
        return result

    try:
        w, h = _read_png_ihdr(data)
    except ValueError as exc:
        result["reason"] = f"Invalid PNG header: {exc}"
        return result

    result["width"] = w
    result["height"] = h

    if w != expected_width or h != expected_height:
        result["reason"] = (
            f"Unexpected image dimensions {w}x{h}; "
            f"expected {expected_width}x{expected_height}"
        )
        return result

    # Approximate colour uniformity check via IDAT decompression.
    try:
        idat_payload = bytearray()
        offset = 8
        while offset + 12 <= len(data):
            chunk_len = struct.unpack(">I", data[offset : offset + 4])[0]
            chunk_type = data[offset + 4 : offset + 8]
            chunk_data = data[offset + 8 : offset + 8 + chunk_len]
            if chunk_type == b"IDAT":
                idat_payload.extend(chunk_data)
            elif chunk_type == b"IEND":
                break
            offset += 12 + chunk_len

        if idat_payload:
            raw = zlib.decompress(bytes(idat_payload))
            total = len(raw)
            if total > 0:
                frac_zero = raw.count(0x00) / total
                frac_max = raw.count(0xFF) / total
                if frac_zero >= near_uniform_tolerance:
                    result["reason"] = (
                        f"Image appears all-black ({frac_zero:.1%} zero bytes); "
                        "SceneCapture2D may not have rendered"
                    )
                    return result
                if frac_max >= near_uniform_tolerance:
                    result["reason"] = (
                        f"Image appears all-white ({frac_max:.1%} 0xFF bytes); "
                        "render target may not have been populated"
                    )
                    return result
    except Exception:
        # If decompression fails (e.g. multi-chunk interlaced PNG), skip the colour check.
        pass

    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# Metadata writer
# ---------------------------------------------------------------------------

def write_metadata(
    output_dir: Path,
    day: int,
    segment_id: str,
    camera_pose: Dict[str, float],
    world_coordinates: Dict[str, float],
    condition_score: float,
    image_filename: str = "raw.png",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write inspection capture provenance to *output_dir/metadata.json*.

    Required fields in the written JSON:
      - day, segment_id, timestamp (ISO 8601 UTC)
      - camera_pose: {x_cm, y_cm, z_cm, pitch_deg, yaw_deg, roll_deg}
      - world_coordinates: {x_cm, y_cm, z_cm, along_m, across_m}
      - condition_score: float (renderer condition score from manifest)
      - image_path: absolute path to raw.png

    Args:
        output_dir:        Directory where raw.png was saved.
        day:               Inspection day (1–20).
        segment_id:        Road segment identifier.
        camera_pose:       Dict with UE camera location/rotation fields.
        world_coordinates: Dict with road world location fields.
        condition_score:   Deterioration condition score from the manifest.
        image_filename:    Name of the capture file inside output_dir.
        extra:             Optional additional fields merged into the record.

    Returns:
        Path to the written metadata.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / image_filename
    record = {
        "schema_version": "RoadSentinelInspectionMetadata/v1",
        "day": day,
        "segment_id": segment_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "camera_pose": {
            "x_cm": float(camera_pose.get("x_cm", 0.0)),
            "y_cm": float(camera_pose.get("y_cm", 0.0)),
            "z_cm": float(camera_pose.get("z_cm", 0.0)),
            "pitch_deg": float(camera_pose.get("pitch_deg", -89.0)),
            "yaw_deg": float(camera_pose.get("yaw_deg", 0.0)),
            "roll_deg": float(camera_pose.get("roll_deg", 0.0)),
        },
        "world_coordinates": {
            "x_cm": float(world_coordinates.get("x_cm", 0.0)),
            "y_cm": float(world_coordinates.get("y_cm", 0.0)),
            "z_cm": float(world_coordinates.get("z_cm", 0.0)),
            "along_m": float(world_coordinates.get("along_m", 0.0)),
            "across_m": float(world_coordinates.get("across_m", 0.0)),
        },
        "condition_score": float(condition_score),
        "image_path": str(image_path.resolve()),
        "capture_source": "RoadSentinelSim Unreal Engine SceneCapture2D",
        "project": "RoadSentinelSim.uproject",
    }
    if extra:
        record.update(extra)

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # Update segment_history.json
    history_path = output_dir / "segment_history.json"
    history_data = {"segment_id": segment_id, "history": []}
    if history_path.is_file():
        try:
            history_data = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    history_list = history_data.get("history", [])
    # Replace existing day record or append
    new_entry = {
        "segment_id": segment_id,
        "day": day,
        "deterioration_state": str(extra.get("deterioration_state", "observed_deterioration")) if extra else "observed_deterioration",
        "defect_types": extra.get("defect_types", []) if extra else [],
        "severity_value": float(condition_score),
        "camera_pose": record["camera_pose"],
        "image_filename": image_filename,
        "timestamp": record["timestamp"]
    }
    history_list = [x for x in history_list if int(x.get("day", -1)) != day]
    history_list.append(new_entry)
    history_list.sort(key=lambda x: int(x.get("day", 0)))
    history_data["history"] = history_list
    history_path.write_text(json.dumps(history_data, indent=2), encoding="utf-8")

    return meta_path
