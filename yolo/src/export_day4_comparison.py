#!/usr/bin/env python3
"""Export fixed-checkpoint Day 4 YOLO records and strict holdout metrics."""
from __future__ import annotations

import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "yolo/natural_condition_holdout/manifest.csv"
RESULTS = ROOT / "yolo/outputs/day4_natural_conditions/results.json"
WEIGHTS = ROOT / "yolo/weights/best.pt"
OUT = ROOT / "yolo/outputs/day4_comparison"
IOU_THRESHOLD = 0.5

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0

def main():
    manifest = {r["image_name"]: r for r in csv.DictReader(MANIFEST.open(newline=""))}
    results = json.loads(RESULTS.read_text())
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    rows, by_class = [], {}
    for item in results["results"]:
        m = manifest[item["image_name"]]
        for det in item["detections"]:
            rows.append({
                "image_id": Path(item["image_name"]).stem,
                "original_filename": item["image_name"],
                "model_name": "RoadSentinel YOLOv8n",
                "checkpoint_hash": digest,
                "class_id": det["class_id"], "class_name": det["class"],
                "confidence": det["confidence"], "bbox": json.dumps(det["bbox"]),
                "inference_ms": item["inference_ms"], "condition": m["condition"],
                "condition_provenance": m["condition_provenance"], "viewpoint": m["viewpoint"],
            })
        gt = []
        label_path = ROOT / m["annotation_path"].replace("\\", "/")
        for line in label_path.read_text().splitlines():
            cid, xc, yc, ww, hh = map(float, line.split()[:5])
            w, h = item["width"], item["height"]
            gt.append((int(cid), [(xc-ww/2)*w, (yc-hh/2)*h, (xc+ww/2)*w, (yc+hh/2)*h]))
        preds = [(d["class_id"], d["bbox"]) for d in item["detections"]]
        for cid in sorted(set([x[0] for x in gt] + [x[0] for x in preds])):
            g = [x[1] for x in gt if x[0] == cid]; p = [x[1] for x in preds if x[0] == cid]
            matched = set()
            matched_iou = 0.0
            tp = 0
            for pb in p:
                candidates = [(iou(pb, gb), j) for j, gb in enumerate(g) if j not in matched and iou(pb, gb) >= IOU_THRESHOLD]
                if candidates:
                    best_iou, j = max(candidates); matched.add(j); matched_iou += best_iou; tp += 1
            key = str(cid); acc = by_class.setdefault(key, [0, 0, 0, 0.0])
            acc[0] += tp; acc[1] += len(p)-tp; acc[2] += len(g)-tp
            acc[3] += matched_iou
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUT / "yolo_records.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    names = results["classes"]
    metrics = {}
    for cid, (tp, fp, fn, isum) in by_class.items():
        precision = tp/(tp+fp) if tp+fp else 0.0; recall = tp/(tp+fn) if tp+fn else 0.0
        metrics[names[cid]] = {"class_id": int(cid), "tp": tp, "fp": fp, "fn": fn,
          "precision": round(precision, 4), "recall": round(recall, 4),
          "f1": round(2*precision*recall/(precision+recall), 4) if precision+recall else 0.0,
          "mean_matched_iou": round(isum/tp, 4) if tp else None}
    (OUT / "metrics.json").write_text(json.dumps({"checkpoint_hash": digest, "iou_threshold": IOU_THRESHOLD,
        "condition_provenance_note": "Metrics use compatible YOLO ground-truth boxes; condition labels remain visual, not weather-verified.",
        "images": len(results["results"]), "metrics_by_class": metrics}, indent=2) + "\n")

if __name__ == "__main__": main()
