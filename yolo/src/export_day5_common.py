#!/usr/bin/env python3
"""Export common-candidate YOLO records and valid box metrics only."""
from __future__ import annotations

import csv, hashlib, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmark/common_candidate/manifest.csv"
RESULTS = ROOT / "yolo/outputs/day5_common_candidate/results.json"
WEIGHTS = ROOT / "yolo/weights/best.pt"
OUT = ROOT / "yolo/outputs/day5_common_candidate"

def iou(a, b):
    x1, y1, x2, y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0,a[2]-a[0]) * max(0,a[3]-a[1]); ab = max(0,b[2]-b[0]) * max(0,b[3]-b[1])
    return inter / (aa + ab - inter) if aa + ab - inter else 0.0

def main():
    manifest = {r["original_filename"]: r for r in csv.DictReader(MANIFEST.open(newline=""))}
    data = json.loads(RESULTS.read_text()); digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    names = data["classes"]; rows = []; stats = defaultdict(lambda: [0, 0, 0, 0.0]); condition_counts = defaultdict(lambda: [0, 0])
    for item in data["results"]:
        m = manifest[item["image_name"]]
        for d in item["detections"]:
            rows.append({"image_id": Path(item["image_name"]).stem, "original_filename": item["image_name"],
                "model_name": "RoadSentinel YOLOv8n", "checkpoint_hash": digest, "class_id": d["class_id"],
                "class_name": d["class"], "confidence": d["confidence"], "bbox": json.dumps(d["bbox"]),
                "inference_ms": item["inference_ms"], "condition": m["condition"],
                "condition_provenance": m["condition_provenance"], "viewpoint": m["viewpoint"]})
        if m["gt_compatible"].lower() != "true": continue
        labels = (ROOT / m["ground_truth_path"]).read_text().splitlines(); w, h = item["width"], item["height"]
        gt = []
        for line in labels:
            cid, xc, yc, ww, hh = map(float, line.split()[:5]); gt.append((int(cid), [(xc-ww/2)*w,(yc-hh/2)*h,(xc+ww/2)*w,(yc+hh/2)*h]))
        pred = [(d["class_id"], d["bbox"]) for d in item["detections"]]
        condition_counts[m["condition"]][0] += 1; condition_counts[m["condition"]][1] += len(gt)
        for cid in sorted(set(x[0] for x in gt) | set(x[0] for x in pred)):
            g = [x[1] for x in gt if x[0] == cid]; p = [x[1] for x in pred if x[0] == cid]; used = set(); tp = 0; matched_iou = 0.0
            for pb in p:
                choices = [(iou(pb, gb), j) for j, gb in enumerate(g) if j not in used and iou(pb, gb) >= 0.5]
                if choices:
                    best, j = max(choices); used.add(j); tp += 1; matched_iou += best
            s = stats[str(cid)]; s[0] += tp; s[1] += len(p)-tp; s[2] += len(g)-tp; s[3] += matched_iou
    fields = list(rows[0]);
    with (OUT / "yolo_records.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    metric_rows = {}
    for cid, (tp, fp, fn, isum) in stats.items():
        p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
        metric_rows[names[cid]] = {"class_id": int(cid), "tp": tp, "fp": fp, "fn": fn, "precision": round(p,4), "recall": round(r,4), "f1": round(2*p*r/(p+r),4) if p+r else 0.0, "mean_matched_iou": round(isum/tp,4) if tp else None}
    (OUT / "metrics.json").write_text(json.dumps({"checkpoint_hash": digest, "iou_threshold": 0.5,
        "labelled_images": sum(v[0] for v in condition_counts.values()), "unlabelled_reference_images": len(data["results"])-sum(v[0] for v in condition_counts.values()),
        "condition_counts": {k: {"images": v[0], "ground_truth_boxes": v[1]} for k,v in condition_counts.items()},
        "metrics_by_class": metric_rows, "mAP50": "not calculated: small development candidate and visual-only condition labels",
        "mAP50-95": "not calculated: small development candidate and visual-only condition labels"}, indent=2) + "\n")

if __name__ == "__main__": main()
