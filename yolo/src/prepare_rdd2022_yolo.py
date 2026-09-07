#!/usr/bin/env python3
"""Convert RDD2022 China_Drone dataset to YOLO format with deterministic train/val split.

Excludes benchmark images (China_Drone_001267) from train to prevent data leakage.
Creates symlinks for images to avoid copying gigabytes of files, and writes normalized
YOLO annotation txt files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

CLASS_TO_ID: Dict[str, int] = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
    "Repair": 4,
}

BENCHMARK_EXCLUSIONS = {
    "China_Drone_001267.jpg",
}


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[int, float, float, float, float]]]:
    """Parse a Pascal VOC XML file and return (width, height, list of [class_id, x, y, w, h])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)
    else:
        width, height = 512, 512

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_TO_ID:
            continue
        class_id = CLASS_TO_ID[name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Clip coordinates to image boundary
        xmin = max(0.0, min(float(width), xmin))
        xmax = max(0.0, min(float(width), xmax))
        ymin = max(0.0, min(float(height), ymin))
        ymax = max(0.0, min(float(height), ymax))

        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        norm_w = box_w / width
        norm_h = box_h / height

        # Clamp normalized values to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        boxes.append((class_id, x_center, y_center, norm_w, norm_h))

    return width, height, boxes


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    images_dir = source_root / "train" / "images"
    xmls_dir = source_root / "train" / "annotations" / "xmls"

    if not images_dir.exists():
        raise FileNotFoundError(f"Source images directory not found: {images_dir}")
    if not xmls_dir.exists():
        raise FileNotFoundError(f"Source XMLs directory not found: {xmls_dir}")

    all_images = sorted(images_dir.glob("*.jpg"))
    if not all_images:
        raise ValueError(f"No JPG images found in {images_dir}")

    print(f"Found {len(all_images)} source images in {images_dir}")

    # Set up deterministic split
    random.seed(seed)
    # Shuffle image list
    shuffled = list(all_images)
    random.shuffle(shuffled)

    # Force benchmark exclusions into val/test
    val_set = set()
    train_candidates = []
    for img in shuffled:
        if img.name in BENCHMARK_EXCLUSIONS:
            val_set.add(img)
        else:
            train_candidates.append(img)

    num_val_target = int(len(all_images) * val_ratio)
    needed_for_val = max(0, num_val_target - len(val_set))
    val_set.update(train_candidates[:needed_for_val])
    train_set = set(train_candidates[needed_for_val:])

    print(f"Train images: {len(train_set)}, Val images: {len(val_set)} (Benchmark exclusions: {BENCHMARK_EXCLUSIONS})")

    # Prepare directories
    for split in ["train", "val"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "train_images": len(train_set),
        "val_images": len(val_set),
        "class_counts_train": {cls: 0 for cls in CLASS_TO_ID},
        "class_counts_val": {cls: 0 for cls in CLASS_TO_ID},
        "id_to_class": {v: k for k, v in CLASS_TO_ID.items()},
    }

    # Process train and val splits
    for split_name, img_set in [("train", train_set), ("val", val_set)]:
        for img_path in img_set:
            stem = img_path.stem
            xml_path = xmls_dir / f"{stem}.xml"

            # 1. Symlink or copy image
            dest_img = output_root / "images" / split_name / img_path.name
            if dest_img.exists() or dest_img.is_symlink():
                dest_img.unlink()
            dest_img.symlink_to(img_path.resolve())

            # 2. Convert and write labels
            dest_label = output_root / "labels" / split_name / f"{stem}.txt"
            lines = []
            if xml_path.exists():
                _, _, boxes = parse_voc_xml(xml_path)
                for class_id, xc, yc, w, h in boxes:
                    lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    cls_name = stats["id_to_class"][class_id]
                    stats[f"class_counts_{split_name}"][cls_name] += 1

            dest_label.write_text("\n".join(lines) + ("\n" if lines else ""))

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("RoadSentinel_datasets/rdd2022_full/China_Drone/China_Drone"),
        help="Source directory for RDD2022 China_Drone subset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo/data/rdd2022"),
        help="Output directory for YOLO dataset",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    args = parser.parse_args()

    stats = prepare_dataset(
        source_root=args.source,
        output_root=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    stats_file = args.output / "dataset_summary.json"
    stats_file.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Dataset preparation complete. Summary saved to {stats_file}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
