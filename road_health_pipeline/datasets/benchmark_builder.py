"""Multi-domain validation benchmark and dataset partition manager.

Structures a unified benchmark containing:
  - Real-world and Unreal/CARLA simulated imagery
  - Healthy roads, dry potholes, water-filled potholes, cracks, lane markings, shadows, and asphalt patches
  - Deterministic splits: train (60%), val (20%), test (20%)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

log = logging.getLogger(__name__)


@dataclass
class BenchmarkItem:
    item_id: str
    filepath: str
    domain: str  # "real" or "sim"
    category: str  # "healthy_road", "pothole", "water_filled_pothole", "crack", "road_marking", "shadow", "patch"
    split: str  # "train", "val", "test"
    annotation_type: str = "image_level"  # "image_level", "bbox", "mask"
    metadata: Dict[str, Any] = None


class BenchmarkBuilder:
    """Discovers, categorizes, and partitions multi-domain datasets."""

    def __init__(self, workspace_root: Path = Path(__file__).resolve().parents[2]) -> None:
        self.workspace_root = workspace_root
        self.items: List[BenchmarkItem] = []

    def build_benchmark(
        self,
        output_manifest: Optional[Path] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Scan workspace datasets, categorize samples, and partition into train/val/test."""
        random.seed(seed)
        self.items.clear()

        # 1. Simulated CARLA / Blender Frames
        sim_frames_dir = self.workspace_root / "dataset" / "rendered_frames"
        if sim_frames_dir.is_dir():
            for f in sorted(sim_frames_dir.glob("*.png")):
                self.items.append(
                    BenchmarkItem(
                        item_id=f"sim_{f.stem}",
                        filepath=str(f.resolve()),
                        domain="sim",
                        category="pothole",  # Rendered frames contain procedural pothole cavities
                        split="",
                        annotation_type="procedural_depth",
                        metadata={"source": "carla_blender_cosim"},
                    )
                )

        # 2. Simulated Healthy Road Frames (e.g. initial clean flight frames)
        env_dir = self.workspace_root / "env"
        for f in env_dir.glob("test_*.jpg"):
            self.items.append(
                BenchmarkItem(
                    item_id=f"sim_healthy_{f.stem}",
                    filepath=str(f.resolve()),
                    domain="sim",
                    category="healthy_road",
                    split="",
                    annotation_type="image_level",
                    metadata={"source": "carla_env_capture"},
                )
            )

        # 3. Real-world Pothole-600
        p600_rgb = self.workspace_root / "RoadSentinel_datasets" / "pothole_600" / "Pothole" / "rgb"
        if p600_rgb.is_dir():
            for f in sorted(p600_rgb.glob("*.png"))[:100]:
                self.items.append(
                    BenchmarkItem(
                        item_id=f"p600_{f.stem}",
                        filepath=str(f.resolve()),
                        domain="real",
                        category="pothole",
                        split="",
                        annotation_type="pixel_mask",
                        metadata={"source": "pothole_600"},
                    )
                )

        # 4. Real-world Water-Filled Potholes
        water_dir = self.workspace_root / "RoadSentinel_datasets" / "water_filled_potholes"
        if water_dir.is_dir():
            for f in sorted(water_dir.glob("**/*.jpg"))[:60]:
                self.items.append(
                    BenchmarkItem(
                        item_id=f"water_{f.stem}",
                        filepath=str(f.resolve()),
                        domain="real",
                        category="water_filled_pothole",
                        split="",
                        annotation_type="pascal_voc",
                        metadata={"source": "water_filled_potholes"},
                    )
                )

        # 5. Real-world MWPD (Shadows, Weather Variations)
        mwpd_dir = self.workspace_root / "RoadSentinel_datasets" / "mwpd"
        if mwpd_dir.is_dir():
            for f in sorted(mwpd_dir.glob("**/*.jpg"))[:60]:
                cat = "shadow" if "shadow" in f.name.lower() else "pothole"
                self.items.append(
                    BenchmarkItem(
                        item_id=f"mwpd_{f.stem}",
                        filepath=str(f.resolve()),
                        domain="real",
                        category=cat,
                        split="",
                        annotation_type="yolo_bbox",
                        metadata={"source": "mwpd"},
                    )
                )

        # 6. Real-world RDD2022 (Healthy roads & Cracks)
        rdd_train_dir = (
            self.workspace_root
            / "RoadSentinel_datasets"
            / "rdd2022_full"
            / "China_Drone"
            / "China_Drone"
            / "train"
            / "images"
        )
        if rdd_train_dir.is_dir():
            for f in sorted(rdd_train_dir.glob("*.jpg"))[:120]:
                xml_f = f.parent.parent / "annotations" / "xmls" / f"{f.stem}.xml"
                cat = "crack" if xml_f.exists() else "healthy_road"
                self.items.append(
                    BenchmarkItem(
                        item_id=f"rdd_{f.stem}",
                        filepath=str(f.resolve()),
                        domain="real",
                        category=cat,
                        split="",
                        annotation_type="pascal_voc" if xml_f.exists() else "image_level",
                        metadata={"source": "rdd2022_china_drone"},
                    )
                )

        # Stratified partition by (domain, category)
        groups: Dict[Tuple[str, str], List[BenchmarkItem]] = {}
        for item in self.items:
            key = (item.domain, item.category)
            groups.setdefault(key, []).append(item)

        for key, group in groups.items():
            random.shuffle(group)
            n = len(group)
            n_train = max(1, int(0.60 * n))
            n_val = max(1, int(0.20 * n)) if n >= 3 else 0
            for i, item in enumerate(group):
                if i < n_train:
                    item.split = "train"
                elif i < n_train + n_val:
                    item.split = "val"
                else:
                    item.split = "test"

        summary = {
            "total_items": len(self.items),
            "splits": {
                "train": sum(1 for x in self.items if x.split == "train"),
                "val": sum(1 for x in self.items if x.split == "val"),
                "test": sum(1 for x in self.items if x.split == "test"),
            },
            "domains": {
                "real": sum(1 for x in self.items if x.domain == "real"),
                "sim": sum(1 for x in self.items if x.domain == "sim"),
            },
            "categories": {},
            "items": [asdict(x) for x in self.items],
        }
        for item in self.items:
            summary["categories"][item.category] = summary["categories"].get(item.category, 0) + 1

        if output_manifest is not None:
            output_manifest = Path(output_manifest)
            output_manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(output_manifest, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            log.info("Saved Benchmark Manifest to %s (%d items)", output_manifest, len(self.items))

        return summary


if __name__ == "__main__":
    builder = BenchmarkBuilder()
    manifest = builder.build_benchmark(
        output_manifest=Path(__file__).resolve().parents[1] / "output" / "benchmark_manifest.json"
    )
    print(f"Benchmark Built: {manifest['total_items']} items. Splits: {manifest['splits']}. Domains: {manifest['domains']}")
