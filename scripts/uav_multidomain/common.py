"""Shared, dependency-light helpers for Phase UAV-1 dataset audits."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readable_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def perceptual_hash(path: Path) -> int:
    """Return a deterministic 64-bit pHash based on the low-frequency DCT."""
    pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise ValueError(f"Unreadable image: {path}")
    small = cv2.resize(pixels, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    values = dct.flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(frozen=True)
class Box:
    class_id: int
    xc: float
    yc: float
    width: float
    height: float


def parse_yolo(path: Path, num_classes: int) -> list[Box]:
    boxes: list[Box] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric YOLO field") from exc
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{path}:{line_number}: class {class_id} outside [0,{num_classes - 1}]")
        xc, yc, width, height = values
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite coordinate")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized box {values}")
        if xc - width / 2 < -1e-6 or xc + width / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: horizontal bounds exceed image")
        if yc - height / 2 < -1e-6 or yc + height / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: vertical bounds exceed image")
        boxes.append(Box(class_id, xc, yc, width, height))
    return boxes


def parse_voc(path: Path, class_to_id: dict[str, int] | None = None) -> tuple[int, int, list[Box], list[str]]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{path}: missing <size>")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: invalid size {width}x{height}")
    boxes: list[Box] = []
    names: list[str] = []
    for index, obj in enumerate(root.findall("object"), 1):
        name = (obj.findtext("name") or "").strip()
        bbox = obj.find("bndbox")
        if not name or bbox is None:
            raise ValueError(f"{path}: object {index} missing name/bndbox")
        coords = [float(bbox.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
        xmin, ymin, xmax, ymax = coords
        if not all(np.isfinite(value) for value in coords):
            raise ValueError(f"{path}: object {index} has non-finite coordinate")
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError(f"{path}: object {index} box {coords} outside {width}x{height}")
        names.append(name)
        class_id = class_to_id.get(name, -1) if class_to_id else -1
        boxes.append(
            Box(
                class_id,
                (xmin + xmax) / (2 * width),
                (ymin + ymax) / (2 * height),
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            )
        )
    return width, height, boxes, names


def paired_label(image: Path, images_root: Path, labels_root: Path, suffix: str) -> Path:
    return labels_root / image.relative_to(images_root).with_suffix(suffix)


def bytes_in_tree(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def stable_sample_id(dataset: str, group_id: str, original_id: str) -> str:
    def clean(value: str) -> str:
        return "_".join(value.strip().lower().replace("-", "_").split())

    return f"{clean(dataset)}::{clean(group_id)}::{clean(original_id)}"


def ensure_unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"Duplicate {label}: {preview}")
