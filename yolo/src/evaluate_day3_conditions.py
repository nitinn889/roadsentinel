#!/usr/bin/env python3
"""Evaluate fixed YOLO detections against geometry-preserving Day 3 labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def xywh_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    _, cx, cy, w, h = values
    return [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_conditions" / "manifest.csv")
    parser.add_argument("--results-root", type=Path, default=ROOT / "outputs" / "day3_conditions")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "day3_robustness_summary.csv")
    args = parser.parse_args()
    with args.manifest.open(newline="") as f:
        manifest = list(csv.DictReader(f))
    output_rows, aggregate = [], defaultdict(lambda: defaultdict(float))
    for item in manifest:
        condition, name = item["condition"], item["image_name"]
        result_path = args.results_root / condition / "results.json"
        result = json.loads(result_path.read_text())
        observed = next(x for x in result["results"] if x["image_name"] == name)
        width, height = observed["width"], observed["height"]
        gt = []
        for line in Path(item["annotation_path"]).read_text().splitlines():
            values = [float(v) for v in line.split()]
            gt.append((int(values[0]), xywh_to_xyxy(values, width, height)))
        preds = [(int(d["class_id"]), [float(v) for v in d["bbox"]], float(d["confidence"])) for d in observed["detections"]]
        matched, ious = set(), []
        tp = 0
        for cls, box, _ in preds:
            candidates = [(idx, iou(box, gt_box)) for idx, (gt_cls, gt_box) in enumerate(gt) if idx not in matched and cls == gt_cls]
            if candidates and max(candidates, key=lambda x: x[1])[1] >= 0.5:
                idx, score = max(candidates, key=lambda x: x[1]); matched.add(idx); tp += 1; ious.append(score)
        fp, fn = len(preds) - tp, len(gt) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {"image_name": name, "condition": condition, "ground_truth_available": "true", "tp": tp, "fp": fp, "fn": fn,
               "precision": f"{precision:.4f}", "recall": f"{recall:.4f}", "f1": f"{f1:.4f}",
               "iou": f"{sum(ious) / len(ious):.4f}" if ious else "", "num_detections": len(preds),
               "mean_confidence": f"{sum(x[2] for x in preds) / len(preds):.4f}" if preds else "",
               "inference_ms": observed["inference_ms"], "notes": "Synthetic photometric condition; IoU match uses same class and threshold 0.50."}
        output_rows.append(row)
        for key in ("tp", "fp", "fn"):
            aggregate[condition][key] += row[key]
    fields = list(output_rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
    for condition, values in sorted(aggregate.items()):
        tp, fp, fn = (values[k] for k in ("tp", "fp", "fn"))
        p, r = (tp / (tp + fp) if tp + fp else 0), (tp / (tp + fn) if tp + fn else 0)
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"{condition}: TP={int(tp)} FP={int(fp)} FN={int(fn)} P={p:.4f} R={r:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()
