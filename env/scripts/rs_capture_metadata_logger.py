#!/usr/bin/env python3
"""Passive sidecar, manifest, and audit logging for inspection captures.

This module has no Unreal, CARLA, IPC, Qt, rendering, or simulation dependencies.
It only serializes caller-provided GUI values to metadata files.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping


MANIFEST_FIELDS = (
    "segment_id",
    "day",
    "image_path",
    "metadata_path",
    "lighting_preset",
    "road_health_state",
    "pothole_sizing_spectrum",
    "pothole_density_per_100m2",
    "pothole_moisture_state",
    "camera_preset",
    "capture_timestamp",
)


def _display_path(path: Path, workspace_root: Path) -> str:
    """Prefer a stable workspace-relative POSIX path when possible."""
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _replace_text(path: Path, text: str) -> None:
    """Atomically replace one metadata index/sidecar file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def record_capture_attempt(
    selections: Mapping[str, Any],
    *,
    image_path: Path,
    output_root: Path,
    workspace_root: Path,
    capture_succeeded: bool,
    capture_status: str,
) -> dict[str, Any]:
    """Persist one passive capture audit record.

    Every invocation appends to capture_history.jsonl. Successful captures also
    replace the per-image sidecar and upsert one row in dataset_manifest.csv.
    """
    image_path = image_path.resolve()
    metadata_path = image_path.with_name(f"{image_path.stem}_metadata.json")
    record = dict(selections)
    record.update(
        {
            "image_filename": image_path.name,
            "image_path": _display_path(image_path, workspace_root),
            "metadata_path": _display_path(metadata_path, workspace_root),
            "capture_succeeded": bool(capture_succeeded),
            "capture_status": str(capture_status),
            "source": "RoadSentinel Interactive Studio",
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    history_path = output_root / "capture_history.jsonl"
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not capture_succeeded:
        return record

    _replace_text(metadata_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    manifest_path = output_root / "dataset_manifest.csv"
    rows: list[dict[str, str]] = []
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    manifest_row = {field: record.get(field, "") for field in MANIFEST_FIELDS}
    rows = [row for row in rows if row.get("image_path") != manifest_row["image_path"]]
    rows.append(manifest_row)

    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_path)
    return record
