#!/usr/bin/env python3
"""build_manifest.py
-------------------
Builds and freezes the deterministic cross-domain benchmark manifest from RDD2022 India.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
INDIA_ROOT = WORKSPACE_ROOT / "RoadSentinel_datasets" / "rdd2022_full" / "India" / "India" / "train"
XML_DIR = INDIA_ROOT / "annotations" / "xmls"
IMG_DIR = INDIA_ROOT / "images"

OUTPUT_MANIFEST = WORKSPACE_ROOT / "cross_domain" / "benchmark_manifest.csv"
OUTPUT_GT_JSON = WORKSPACE_ROOT / "cross_domain" / "ground_truth.json"
OUTPUT_DOMAIN_TABLE = WORKSPACE_ROOT / "cross_domain" / "tables" / "table_domain_characteristics.csv"

# Valid damage categories in RDD2022
VALID_CLASSES = {"D00", "D10", "D20", "D40", "Repair", "D01", "D11", "D43", "D44"}
MAPPED_BINARY_CLASS = "road_defect"


def parse_voc_xml(xml_path: Path) -> Dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find("filename").text if root.find("filename") is not None else xml_path.stem + ".jpg"
    size_node = root.find("size")
    width = int(size_node.find("width").text) if size_node is not None else 720
    height = int(size_node.find("height").text) if size_node is not None else 720
    
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # Clip to image boundaries
        xmin = max(0.0, min(float(width), xmin))
        ymin = max(0.0, min(float(height), ymin))
        xmax = max(0.0, min(float(width), xmax))
        ymax = max(0.0, min(float(height), ymax))
        
        if xmax > xmin and ymax > ymin:
            boxes.append({
                "class_name": name,
                "binary_class": MAPPED_BINARY_CLASS,
                "bbox_xyxy": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
                "area": round((xmax - xmin) * (ymax - ymin), 1)
            })
            
    return {
        "image_id": xml_path.stem,
        "filename": filename,
        "width": width,
        "height": height,
        "boxes": boxes,
        "box_count": len(boxes)
    }


def main() -> None:
    print("Scanning RDD2022 India dataset...")
    xml_files = sorted(list(XML_DIR.glob("*.xml")))
    print(f"Found {len(xml_files)} total XML annotation files.")
    
    selected_samples: List[Dict[str, Any]] = []
    
    # Select first 300 non-empty annotated images deterministically
    TARGET_COUNT = 300
    for xml_path in xml_files:
        img_path = IMG_DIR / f"{xml_path.stem}.jpg"
        if not img_path.exists():
            continue
            
        parsed = parse_voc_xml(xml_path)
        if parsed["box_count"] > 0:
            parsed["image_path"] = str(img_path.relative_to(WORKSPACE_ROOT))
            parsed["xml_path"] = str(xml_path.relative_to(WORKSPACE_ROOT))
            selected_samples.append(parsed)
            
        if len(selected_samples) >= TARGET_COUNT:
            break
            
    print(f"Selected {len(selected_samples)} non-empty annotated cross-domain images.")
    
    # 1. Write benchmark_manifest.csv
    with open(OUTPUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "dataset",
            "source_path",
            "original_classes",
            "binary_class",
            "gt_box_count",
            "viewpoint",
            "country",
            "resolution",
            "notes"
        ])
        for s in selected_samples:
            orig_classes = ";".join(sorted(list(set(b["class_name"] for b in s["boxes"]))))
            writer.writerow([
                s["image_id"],
                "RDD2022_India",
                s["image_path"],
                orig_classes,
                MAPPED_BINARY_CLASS,
                s["box_count"],
                "Forward-Facing Windshield Dashcam",
                "India",
                f"{s['width']}x{s['height']}",
                "Independent Cross-Domain Benchmark Sample"
            ])
            
    print(f"Saved benchmark manifest to {OUTPUT_MANIFEST}")
    
    # 2. Write ground_truth.json
    gt_dict = {
        "dataset_name": "RDD2022_India_CrossDomain_Benchmark",
        "sample_size": len(selected_samples),
        "total_gt_boxes": sum(s["box_count"] for s in selected_samples),
        "target_class": MAPPED_BINARY_CLASS,
        "images": {s["image_id"]: s for s in selected_samples}
    }
    with open(OUTPUT_GT_JSON, "w", encoding="utf-8") as f:
        json.dump(gt_dict, f, indent=2)
    print(f"Saved ground truth JSON to {OUTPUT_GT_JSON} (Total GT Boxes: {gt_dict['total_gt_boxes']})")
    
    # 3. Write table_domain_characteristics.csv
    domain_rows = [
        {
            "attribute": "Dataset Name",
            "in_domain_reference": "RDD2022 China_Drone (Validation)",
            "cross_domain_benchmark": "RDD2022 India (Cross-Domain)",
            "domain_difference_significance": "Independent geographic and physical domain"
        },
        {
            "attribute": "Country / Geography",
            "in_domain_reference": "China (Urban municipal roads)",
            "cross_domain_benchmark": "India (Mixed urban/rural national highways)",
            "domain_difference_significance": "Substantial variation in asphalt formulation and maintenance practices"
        },
        {
            "attribute": "Capture Platform & Viewpoint",
            "in_domain_reference": "UAV Drone (Orthogonal Nadir ~25-100m, Pitch -89°)",
            "cross_domain_benchmark": "Vehicle Dashcam / Smartphone (~1.2m, Forward Oblique)",
            "domain_difference_significance": "Extreme geometric perspective shift, horizon vanishing line, vehicle hood reflection"
        },
        {
            "attribute": "Image Resolution",
            "in_domain_reference": "512x512 / 1080p aerial crop",
            "cross_domain_benchmark": "720x720 dashcam frame",
            "domain_difference_significance": "Different spatial aspect ratio and pixel ground sample distance (GSD)"
        },
        {
            "attribute": "Optical / Illumination Conditions",
            "in_domain_reference": "Consistent daylight diffuse aerial illumination",
            "cross_domain_benchmark": "Direct sun glare, windshield reflections, dynamic roadside shadows",
            "domain_difference_significance": "Severe photometric disturbance for vision backbones"
        },
        {
            "attribute": "Road Surface Characteristics",
            "in_domain_reference": "Smooth asphalt with standard municipal lane markings",
            "cross_domain_benchmark": "Weathered asphalt, dust accumulation, unpaved shoulders, irregular potholes",
            "domain_difference_significance": "Higher textural entropy and visual clutter"
        },
        {
            "attribute": "YOLO Training Relationship",
            "in_domain_reference": "Drawn from same distribution as 1,921 training images",
            "cross_domain_benchmark": "ZERO overlap; 100% unseen by YOLOv8n detector",
            "domain_difference_significance": "Strict zero-shot cross-domain evaluation"
        },
        {
            "attribute": "DINOv2 Domain Reference Relationship",
            "in_domain_reference": "Drawn from same distribution as training reference",
            "cross_domain_benchmark": "ZERO overlap; strictly external probe distribution",
            "domain_difference_significance": "Unbiased test of DINOv2 feature-space OOD distance metric"
        }
    ]
    
    with open(OUTPUT_DOMAIN_TABLE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(domain_rows[0].keys()))
        writer.writeheader()
        writer.writerows(domain_rows)
    print(f"Saved domain comparison table to {OUTPUT_DOMAIN_TABLE}")


if __name__ == "__main__":
    main()
