#!/usr/bin/env python3
"""integration/comparison/run_comparison.py
-------------------------------------------
Run YOLO and DINOv2+SAM2 on the same set of images and record comparison
results for interface-compatibility verification.

Purpose:
    Establish that both pipelines can consume the same manifest/image set
    on the same hardware (RTX 5060) and produce structurally comparable
    per-image output records.

NOT a final research benchmark. Results here do NOT represent:
    - Equal training conditions
    - Real-weather robustness verification
    - Final YOLO-vs-DINO comparison
    - Ensemble or fusion metrics

Typical usage:
    python integration/comparison/run_comparison.py \\
        --manifest benchmark/common_candidate/manifest.csv \\
        --output integration/comparison \\
        --limit 4

Output:
    integration/comparison/
        yolo/          raw YOLO results.json per run
        sam2_dino/     raw Marion run_summary.json per run
        comparison_records.csv
        comparison_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Prefer the road_health venv for Marion, root .venv for YOLO
MARION_PYTHON = REPO_ROOT / "road_health_pipeline" / ".venv" / "bin" / "python"
YOLO_PYTHON   = REPO_ROOT / ".venv" / "bin" / "python"

YOLO_RUNNER   = REPO_ROOT / "yolo" / "run_yolo.py"
MARION_RUNNER = REPO_ROOT / "run_sam2_dino.py"   # root-level runner (stable)


def _resolve_python(candidate: Path) -> str:
    if candidate.exists():
        return str(candidate)
    # Fall back to whichever python3 is available
    import shutil
    p = shutil.which("python3")
    return p or "python3"


def run_yolo_on_image(image_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run YOLO on a single image; return parsed result dict."""
    yolo_py = _resolve_python(YOLO_PYTHON)
    cmd = [
        yolo_py, str(YOLO_RUNNER),
        "--input", str(image_path),
        "--output", str(output_dir),
        "--no-render",
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results_file = output_dir / "results.json"
        if proc.returncode == 0 and results_file.exists():
            data = json.loads(results_file.read_text())
            results = data.get("results", [{}])
            first = results[0] if results else {}
            return {
                "success": True,
                "defect_count": first.get("defect_count", 0),
                "detections": first.get("detections", []),
                "inference_ms": first.get("inference_ms", elapsed_ms),
                "error": None,
            }
        else:
            return {"success": False, "defect_count": 0, "detections": [],
                    "inference_ms": elapsed_ms, "error": proc.stderr.strip()[-400:]}
    except subprocess.TimeoutExpired:
        return {"success": False, "defect_count": 0, "detections": [],
                "inference_ms": -1, "error": "TIMEOUT"}


def run_marion_on_image(image_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run DINOv2+SAM2 on a single image; return parsed result dict."""
    marion_py = _resolve_python(MARION_PYTHON)
    cmd = [
        marion_py, str(MARION_RUNNER),
        "--input", str(image_path),
        "--output", str(output_dir),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        summary_file = output_dir / "run_summary.json"
        if proc.returncode == 0 and summary_file.exists():
            data = json.loads(summary_file.read_text())
            records = data.get("records", [{}])
            first = records[0] if records else {}
            return {
                "success": first.get("status") == "SUCCESS",
                "defect_count": first.get("defect_count", 0),
                "defect_boxes_xyxy": first.get("defect_boxes_xyxy", []),
                "current_severity": first.get("current_severity"),
                "inference_ms": first.get("inference_time_ms", elapsed_ms),
                "note": "bounding boxes derived from segmentation masks",
                "error": None,
            }
        else:
            return {"success": False, "defect_count": 0, "defect_boxes_xyxy": [],
                    "current_severity": None, "inference_ms": elapsed_ms,
                    "note": "bounding boxes derived from segmentation masks",
                    "error": proc.stderr.strip()[-400:]}
    except subprocess.TimeoutExpired:
        return {"success": False, "defect_count": 0, "defect_boxes_xyxy": [],
                "current_severity": None, "inference_ms": -1,
                "note": "bounding boxes derived from segmentation masks",
                "error": "TIMEOUT"}


def load_manifest_images(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_rel = row.get("image_path") or row.get("path") or row.get("image")
            if not img_rel:
                continue
            abs_path = (REPO_ROOT / img_rel).resolve()
            if not abs_path.exists():
                continue
            rows.append({
                "image_id": row.get("image_id", abs_path.stem),
                "original_filename": row.get("original_filename", abs_path.name),
                "image_path": str(abs_path),
                "condition": row.get("condition", "unknown"),
                "viewpoint": row.get("viewpoint", "unknown"),
                "gt_compatible": row.get("gt_compatible", "false"),
            })
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="Path to benchmark manifest CSV")
    g.add_argument("--image", help="Single image path")
    parser.add_argument("--output", default="integration/comparison", help="Output root")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    args = parser.parse_args()

    out_root = (REPO_ROOT / args.output).resolve()
    yolo_out = out_root / "yolo"
    marion_out = out_root / "sam2_dino"
    yolo_out.mkdir(parents=True, exist_ok=True)
    marion_out.mkdir(parents=True, exist_ok=True)

    # Collect images
    if args.manifest:
        manifest_path = (REPO_ROOT / args.manifest).resolve()
        images = load_manifest_images(manifest_path, args.limit)
    else:
        img = Path(args.image).resolve()
        images = [{"image_id": img.stem, "original_filename": img.name,
                   "image_path": str(img), "condition": "unknown",
                   "viewpoint": "unknown", "gt_compatible": "false"}]

    print(f"Processing {len(images)} image(s) through both pipelines on RTX 5060...")

    records = []
    for entry in images:
        img_path = Path(entry["image_path"])
        iid = entry["image_id"]
        print(f"\n  [{iid}]")

        # Run YOLO
        y_out = yolo_out / iid
        y_out.mkdir(parents=True, exist_ok=True)
        yolo_res = run_yolo_on_image(img_path, y_out)
        yolo_status = "SUCCESS" if yolo_res["success"] else "FAILED"
        print(f"    YOLO: {yolo_status} | defects={yolo_res['defect_count']} | {yolo_res['inference_ms']:.1f} ms")

        # Run Marion
        m_out = marion_out / iid
        m_out.mkdir(parents=True, exist_ok=True)
        marion_res = run_marion_on_image(img_path, m_out)
        marion_status = "SUCCESS" if marion_res["success"] else "FAILED"
        print(f"    Marion: {marion_status} | defects={marion_res['defect_count']} | {marion_res['inference_ms']:.1f} ms")

        records.append({
            "image_id": iid,
            "original_filename": entry["original_filename"],
            "condition": entry["condition"],
            "viewpoint": entry["viewpoint"],
            "gt_compatible": entry["gt_compatible"],
            "yolo_status": yolo_status,
            "yolo_defect_count": yolo_res["defect_count"],
            "yolo_inference_ms": round(yolo_res["inference_ms"], 2),
            "yolo_defect_classes": ";".join(d.get("class","?") for d in yolo_res.get("detections",[])),
            "marion_status": marion_status,
            "marion_defect_count": marion_res["defect_count"],
            "marion_inference_ms": round(marion_res["inference_ms"], 2),
            "marion_current_severity": marion_res.get("current_severity"),
            "marion_note": "bounding boxes derived from segmentation masks",
            "class_comparison_note": (
                "YOLO outputs named defect classes (D00/D10/D20/D40/Repair). "
                "Marion outputs road_defect (generic anomaly). "
                "Class-wise comparison is NOT valid without a shared taxonomy."
            ),
        })

    # Write CSV
    csv_path = out_root / "comparison_records.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nComparison records saved: {csv_path}")

    # Write summary JSON
    total = len(records)
    y_ok = sum(1 for r in records if r["yolo_status"] == "SUCCESS")
    m_ok = sum(1 for r in records if r["marion_status"] == "SUCCESS")
    summary = {
        "integration_run": "VRINDA1-5_MARION1-4_NITIN1-4_SYNC",
        "total_images": total,
        "yolo_success": y_ok,
        "yolo_failed": total - y_ok,
        "marion_success": m_ok,
        "marion_failed": total - m_ok,
        "interface_compatible": y_ok > 0 and m_ok > 0,
        "caveat": (
            "This is an interface compatibility check only. "
            "Results do NOT constitute a final research benchmark. "
            "Class-wise YOLO vs Marion comparison is NOT valid: YOLO uses D00/D10/D20/D40/Repair taxonomy; "
            "Marion uses generic road_defect anomaly segmentation."
        ),
        "records": records,
    }
    summary_path = out_root / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Comparison summary saved: {summary_path}")

    print(f"\nInterface compatibility: {'OK' if summary['interface_compatible'] else 'PARTIAL/FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
