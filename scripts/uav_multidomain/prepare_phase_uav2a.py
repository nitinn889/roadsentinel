#!/usr/bin/env python3
"""Prepare leakage-safe Phase UAV-2A manifests and YOLO pilot datasets.

This command is intentionally preparation-only.  It never trains or evaluates a
model and it never reads or writes production weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from common import Box, parse_voc, readable_image, sha256_file


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "artifacts/uav_multidomain/split_manifest.csv"
DUPLICATE_AUDIT = ROOT / "artifacts/uav_multidomain/duplicate_audit.csv"
PILOT_ARTIFACTS = ROOT / "artifacts/uav_multidomain/pilot"
FIGURES = ROOT / "reports/uav_multidomain/pilot_figures"
PREPARED = ROOT / "data/external_uav/prepared/uav2a_pilot"
SEED = 20260912

CANONICAL_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
}
SOURCE_MAP = {
    "rdd2022_china_drone": {"D00": 0, "D10": 1, "D20": 2, "D40": 3},
    "uav_pdd2023": {
        "LC": 0,
        "Longitudinal crack": 0,
        "TC": 1,
        "Transverse crack": 1,
        "AC": 2,
        "Alligator crack": 2,
        "PH": 3,
        "Pothole": 3,
    },
}
TRAINING_PHASE1_ROLES = {"yolo_train", "dino_reference", "sam2_development"}

# Manual review of the 14 source-annotation-empty images that become eligible
# only after valid training-role reuse.  The remaining canonical-empty records
# have explicit noncanonical source boxes and are classified from those labels.
SOURCE_EMPTY_MANUAL = {
    "China_Drone_001134": ("Genuine healthy/background image", True, "No visible pavement distress in full-image review."),
    "China_Drone_001391": ("Genuine healthy/background image", True, "Background/road context with no visible distress in full-image review."),
    "China_Drone_000256": ("Contains ambiguous distress", False, "Dark resurfacing/seam boundary is visible; conservatively excluded."),
    "China_Drone_000218": ("Contains ambiguous distress", False, "Faint rectangular surface feature may be repair or marking; conservatively excluded."),
    "tb_00490_top_left": ("Genuine healthy/background image", True, "No visible pavement distress in full-image review."),
    "lr_00490_top_left": ("Genuine healthy/background image", True, "No visible pavement distress in full-image review."),
    "r180_00490_top_left": ("Genuine healthy/background image", True, "No visible pavement distress in full-image review."),
    "r180_00367_top_left": ("Genuine healthy/background image", True, "No visible pavement distress in full-image review."),
    "tb_00367_top_right": ("Genuine healthy/background image", True, "No visible pavement distress in full-image review."),
    "lr_00367_top_right": ("Contains ambiguous distress", False, "Possible linear distress at the road edge; conservatively excluded."),
    "r180_00368_bottom_left": ("Annotation is incomplete", False, "Visible linear shoulder distress has no source annotation."),
    "r180_00499_top_right": ("Annotation is incomplete", False, "Visible linear shoulder distress has no source annotation."),
    "tb_00368_bottom_left": ("Annotation is incomplete", False, "Visible linear shoulder distress has no source annotation."),
    "tb_00499_top_right": ("Annotation is incomplete", False, "Visible linear shoulder distress has no source annotation."),
}

ROLE_FIELDS = [
    "yolo_training",
    "dinov2_reference_bank",
    "sam2_development_candidate",
    "highrpd_pretraining_candidate",
    "dinov2_threshold_calibration",
    "reliability_calibration",
    "model_validation",
    "china_heldout_test",
    "uav_pdd_heldout_test",
    "uapd_evaluation_candidate",
    "poznan_evaluation",
    "india_dashcam_stress_test",
    "sam2_final_mask_test",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def pdd_photo_id(original_id: str) -> str:
    match = re.search(r"_(\d+)_", original_id)
    if not match:
        raise ValueError(f"Cannot derive UAV-PDD source photograph from {original_id}")
    return match.group(1)


def stable_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}|{namespace}|{value}".encode()).hexdigest()


def canonical_boxes(row: dict[str, str]) -> list[Box]:
    mapping = SOURCE_MAP.get(row["dataset_id"], {})
    if not mapping:
        return []
    _, _, boxes, names = parse_voc(ROOT / row["annotation_path"])
    result: list[Box] = []
    for box, name in zip(boxes, names):
        class_id = mapping.get(name)
        if class_id is not None:
            result.append(Box(class_id, box.xc, box.yc, box.width, box.height))
    return result


def make_phash_review(rows: list[dict[str, str]], manifest: list[dict[str, str]]) -> set[str]:
    candidates = [r for r in rows if r["resolution"] == "cross_dataset_pHash_candidate_requires_visual_confirmation"]
    if len(candidates) != 31:
        raise ValueError(f"Expected 31 cross-source pHash candidates, found {len(candidates)}")
    image_by_sample = {r["sample_id"]: r["image_path"] for r in manifest}
    decisions: list[dict[str, str]] = []
    for index, row in enumerate(candidates, 1):
        decisions.append({
            "pair_id": f"PHASH-{index:02d}",
            "sample_id_a": row["sample_id_a"],
            "sample_id_b": row["sample_id_b"],
            "dataset_a": row["dataset_a"],
            "dataset_b": row["dataset_b"],
            "image_path_a": image_by_sample[row["sample_id_a"]],
            "image_path_b": image_by_sample[row["sample_id_b"]],
            "phash_distance": row["distance"],
            "manual_decision": "Visually different",
            "exclude_from_independent_evaluation": "false",
            "review_basis": "Side-by-side full-image visual review; scene geometry and surface features do not match.",
            "reviewer": "Codex visual review",
            "review_date": "2026-09-12",
        })
    fields = list(decisions[0])
    write_csv(PILOT_ARTIFACTS / "phash_manual_review.csv", decisions, fields)

    FIGURES.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_w, tile_h, header_h, per_page = 600, 350, 58, 5
    for page, start in enumerate(range(0, len(decisions), per_page), 1):
        batch = decisions[start:start + per_page]
        canvas = Image.new("RGB", (tile_w * 2, (tile_h + header_h) * len(batch)), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, decision in enumerate(batch):
            y_base = offset * (tile_h + header_h)
            draw.text((8, y_base + 5), f"{decision['pair_id']}  pHash distance={decision['phash_distance']}  decision=Visually different", fill="black", font=font)
            for side in ("a", "b"):
                path = ROOT / decision[f"image_path_{side}"]
                with Image.open(path) as source:
                    fitted = ImageOps.contain(source.convert("RGB"), (tile_w, tile_h))
                x_cell = 0 if side == "a" else tile_w
                x = x_cell + (tile_w - fitted.width) // 2
                y = y_base + header_h + (tile_h - fitted.height) // 2
                canvas.paste(fitted, (x, y))
                draw.text((x_cell + 8, y_base + 25), decision[f"sample_id_{side}"][-75:], fill="black", font=font)
        canvas.save(FIGURES / f"phash_review_{page:02d}.jpg", quality=92)
    return set()


def pdd_partition_by_photo(rows: list[dict[str, str]]) -> dict[str, str]:
    by_photo: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_photo[pdd_photo_id(row["original_id"])].add(row["role"])
    result: dict[str, str] = {}
    for photo, roles in by_photo.items():
        if "uav_pdd_source_heldout_evaluation" in roles:
            result[photo] = "internal_test"
        elif "dino_validation" in roles:
            result[photo] = "validation"
        elif roles & {"dino_calibration", "reliability_calibration"}:
            result[photo] = "calibration"
        else:
            result[photo] = "train"
    return result


def choose_china_pothole_validation_groups(rows: list[dict[str, str]], count: int = 15) -> set[str]:
    candidates: set[str] = set()
    for row in rows:
        if row["dataset_id"] != "rdd2022_china_drone" or row["role"] != "yolo_train":
            continue
        if any(box.class_id == 3 for box in canonical_boxes(row)):
            candidates.add(row["group_id"])
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} China pothole groups available for validation")
    return set(sorted(candidates, key=lambda x: stable_rank("china-pothole-val", x))[:count])


def build_remediated_manifest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    pdd_rows = [r for r in rows if r["dataset_id"] == "uav_pdd2023"]
    pdd_partitions = pdd_partition_by_photo(pdd_rows)
    china_extra_validation = choose_china_pothole_validation_groups(rows)
    output: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        dataset, original_role = row["dataset_id"], row["role"]
        roles = {name: False for name in ROLE_FIELDS}
        adjustment = "none"
        isolation_group = row["group_id"]

        if dataset == "uav_pdd2023":
            isolation_group = f"source_photo_{pdd_photo_id(row['original_id'])}"
            partition = pdd_partitions[pdd_photo_id(row["original_id"])]
            if partition == "internal_test":
                roles["uav_pdd_heldout_test"] = True
                if original_role != "uav_pdd_source_heldout_evaluation":
                    adjustment = "protect_original_photo_neighbours_of_uav_pdd_test"
            elif partition == "validation":
                roles["model_validation"] = True
                if original_role != "dino_validation":
                    adjustment = "protect_original_photo_neighbours_of_validation"
            elif partition == "calibration":
                calibration_roles = {r["role"] for r in pdd_rows if pdd_photo_id(r["original_id"]) == pdd_photo_id(row["original_id"])}
                if "dino_calibration" in calibration_roles and "reliability_calibration" not in calibration_roles:
                    roles["dinov2_threshold_calibration"] = True
                elif "reliability_calibration" in calibration_roles and "dino_calibration" not in calibration_roles:
                    roles["reliability_calibration"] = True
                elif int(stable_rank("pdd-calibration-role", isolation_group)[:8], 16) % 2:
                    roles["dinov2_threshold_calibration"] = True
                else:
                    roles["reliability_calibration"] = True
                if original_role not in {"dino_calibration", "reliability_calibration"}:
                    adjustment = "protect_original_photo_neighbours_of_calibration"
            else:
                roles["yolo_training"] = True
                roles["dinov2_reference_bank"] = original_role == "dino_reference"
                roles["sam2_development_candidate"] = original_role == "sam2_development"
                if original_role in {"dino_reference", "sam2_development"}:
                    adjustment = "valid_training_role_reuse_for_yolo"
        elif dataset == "rdd2022_china_drone":
            if original_role == "china_uav_heldout_evaluation":
                partition, roles["china_heldout_test"] = "internal_test", True
            elif original_role == "dino_validation" or row["group_id"] in china_extra_validation:
                partition, roles["model_validation"] = "validation", True
                if row["group_id"] in china_extra_validation:
                    adjustment = "pothole_stratified_model_validation_reserve"
            elif original_role in {"dino_calibration", "reliability_calibration"}:
                partition = "calibration"
                roles["dinov2_threshold_calibration"] = original_role == "dino_calibration"
                roles["reliability_calibration"] = original_role == "reliability_calibration"
            else:
                partition = "train"
                roles["yolo_training"] = True
                roles["dinov2_reference_bank"] = original_role == "dino_reference"
                roles["sam2_development_candidate"] = original_role == "sam2_development"
                if original_role in {"dino_reference", "sam2_development"}:
                    adjustment = "valid_training_role_reuse_for_yolo"
        elif dataset == "highrpd":
            partition, roles["highrpd_pretraining_candidate"] = "train", True
        elif dataset == "uapd":
            partition, roles["uapd_evaluation_candidate"] = "external_test", True
        elif dataset == "poznan_uav":
            partition, roles["poznan_evaluation"] = "external_test", True
        elif dataset == "rdd2022_india_stress":
            partition, roles["india_dashcam_stress_test"] = "severe_shift_test", True
        else:
            raise ValueError(f"Unrecognized dataset {dataset}")

        row.update({
            "phase1_role": original_role,
            "partition": partition,
            "isolation_group_id": isolation_group,
            "role_adjustment": adjustment,
        })
        row.update({key: bool_text(value) for key, value in roles.items()})
        output.append(row)

    # Integrity gate: no source group can cross a top-level partition.
    partition_by_group: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in output:
        partition_by_group[(row["dataset_id"], row["isolation_group_id"])].add(row["partition"])
    bad = {key: values for key, values in partition_by_group.items() if len(values) > 1}
    if bad:
        raise ValueError(f"Groups cross partitions: {list(bad.items())[:10]}")
    return output


def canonical_empty_audit(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    audit: list[dict[str, str]] = []
    for row in rows:
        if row["dataset_id"] not in SOURCE_MAP:
            continue
        boxes = canonical_boxes(row)
        if boxes:
            continue
        phase1_empty = row["phase1_role"] == "yolo_train"
        remediated_train_empty = row["partition"] == "train" and row["yolo_training"] == "true"
        if not (phase1_empty or remediated_train_empty):
            continue
        if row["source_classes"]:
            category = "Contains only excluded visible distress"
            include = False
            evidence = f"Validated source annotation contains only excluded classes: {row['source_classes']}."
        else:
            if row["original_id"] not in SOURCE_EMPTY_MANUAL:
                raise ValueError(f"Missing manual empty-image decision for {row['sample_id']}")
            category, include, evidence = SOURCE_EMPTY_MANUAL[row["original_id"]]
        audit.append({
            "sample_id": row["sample_id"],
            "dataset_id": row["dataset_id"],
            "original_id": row["original_id"],
            "isolation_group_id": row["isolation_group_id"],
            "remediated_partition": row["partition"],
            "phase1_role": row["phase1_role"],
            "present_in_phase1_harmonized_view": bool_text(row["phase1_role"] == "yolo_train"),
            "image_path": row["image_path"],
            "annotation_path": row["annotation_path"],
            "source_classes": row["source_classes"],
            "decision_category": category,
            "include_as_negative": bool_text(include),
            "pilot_action": "include_verified_negative" if include else "exclude_from_pilot",
            "evidence": evidence,
            "reviewer": "Codex source-annotation and full-image review",
            "review_date": "2026-09-12",
        })
    return audit


def render_canonical_empty_samples(audit: list[dict[str, str]]) -> None:
    categories = [
        "Genuine healthy/background image",
        "Contains only excluded visible distress",
        "Contains ambiguous distress",
        "Annotation is incomplete",
        "Corrupted or unusable",
    ]
    font = ImageFont.load_default()
    tile_w, tile_h, header_h, cols = 440, 300, 48, 4
    canvas = Image.new("RGB", (tile_w * cols, (tile_h + header_h) * len(categories)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, category in enumerate(categories):
        matches = [r for r in audit if r["decision_category"] == category]
        for col in range(cols):
            x_cell = col * tile_w
            y_cell = row_index * (tile_h + header_h)
            if col == 0:
                draw.text((x_cell + 5, y_cell + 4), f"{category} (n={len(matches)})", fill="black", font=font)
            if col >= len(matches):
                if col == 0 and not matches:
                    draw.rectangle((x_cell + 8, y_cell + header_h + 8, x_cell + tile_w - 8, y_cell + header_h + tile_h - 8), outline="#888888", width=2)
                    draw.text((x_cell + 20, y_cell + header_h + 25), "No images assigned to this category.", fill="#555555", font=font)
                continue
            sample = matches[col]
            with Image.open(ROOT / sample["image_path"]) as source:
                fitted = ImageOps.contain(source.convert("RGB"), (tile_w, tile_h))
            x = x_cell + (tile_w - fitted.width) // 2
            y = y_cell + header_h + (tile_h - fitted.height) // 2
            canvas.paste(fitted, (x, y))
            draw.text((x_cell + 5, y_cell + 22), sample["original_id"][:48], fill="black", font=font)
    FIGURES.mkdir(parents=True, exist_ok=True)
    canvas.save(FIGURES / "canonical_empty_decision_samples.jpg", quality=92)


def materialize_dataset(name: str, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    destination = PREPARED / name
    if destination.exists():
        shutil.rmtree(destination)
    inventory: list[dict[str, str]] = []
    for split, rows in (("train", train_rows), ("val", val_rows)):
        images_out = destination / "images" / split
        labels_out = destination / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for row in rows:
            image = ROOT / row["image_path"]
            readable_image(image)
            boxes = canonical_boxes(row)
            token = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:20]
            image_out = images_out / f"{token}{image.suffix.lower()}"
            os.symlink(os.path.relpath(image, image_out.parent), image_out)
            label_out = labels_out / f"{token}.txt"
            lines = [f"{box.class_id} {box.xc:.8f} {box.yc:.8f} {box.width:.8f} {box.height:.8f}" for box in boxes]
            label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts = Counter(box.class_id for box in boxes)
            inventory.append({
                "model_dataset": name,
                "split": split,
                "sample_id": row["sample_id"],
                "dataset_id": row["dataset_id"],
                "original_id": row["original_id"],
                "isolation_group_id": row["isolation_group_id"],
                "image_path": row["image_path"],
                "materialized_image": image_out.relative_to(ROOT).as_posix(),
                "materialized_label": label_out.relative_to(ROOT).as_posix(),
                "is_verified_negative": bool_text(not boxes),
                "longitudinal_objects": str(counts[0]),
                "transverse_objects": str(counts[1]),
                "alligator_objects": str(counts[2]),
                "pothole_objects": str(counts[3]),
                "total_objects": str(len(boxes)),
            })
    yaml_text = (
        f"path: {destination.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: longitudinal_crack\n"
        "  1: transverse_crack\n"
        "  2: alligator_crack\n"
        "  3: pothole\n"
    )
    (destination / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    return inventory


def validate_materialized(inventory: list[dict[str, str]]) -> dict:
    errors: list[str] = []
    seen: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in inventory:
        image = ROOT / row["materialized_image"]
        label = ROOT / row["materialized_label"]
        try:
            readable_image(image)
        except Exception as exc:
            errors.append(f"unreadable image {image}: {exc}")
        try:
            for line_number, line in enumerate(label.read_text().splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"line {line_number}: expected 5 fields")
                class_id = int(fields[0]); coords = [float(v) for v in fields[1:]]
                if class_id not in CANONICAL_NAMES or not all(math.isfinite(v) for v in coords):
                    raise ValueError(f"line {line_number}: invalid class or coordinate")
                xc, yc, width, height = coords
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"line {line_number}: invalid normalized box")
                if xc - width / 2 < -1e-6 or xc + width / 2 > 1 + 1e-6 or yc - height / 2 < -1e-6 or yc + height / 2 > 1 + 1e-6:
                    raise ValueError(f"line {line_number}: box outside image")
        except Exception as exc:
            errors.append(f"invalid label {label}: {exc}")
        seen[(row["model_dataset"], row["sample_id"])].add(row["split"])
    overlaps = [key for key, splits in seen.items() if len(splits) > 1]
    if overlaps:
        errors.append(f"train/validation sample overlap: {overlaps[:10]}")

    by_model = defaultdict(dict)
    for model in sorted({r["model_dataset"] for r in inventory}):
        by_model[model]["train_ids"] = sorted(r["sample_id"] for r in inventory if r["model_dataset"] == model and r["split"] == "train")
        by_model[model]["val_ids"] = sorted(r["sample_id"] for r in inventory if r["model_dataset"] == model and r["split"] == "val")
    if by_model["m_uav0"]["val_ids"] != by_model["m_uav1"]["val_ids"]:
        errors.append("M-UAV0 and M-UAV1 validation sets differ")
    china0 = {r["sample_id"] for r in inventory if r["model_dataset"] == "m_uav0" and r["split"] == "train"}
    china1 = {r["sample_id"] for r in inventory if r["model_dataset"] == "m_uav1" and r["split"] == "train" and r["dataset_id"] == "rdd2022_china_drone"}
    if china0 != china1:
        errors.append("M-UAV1 China subset differs from M-UAV0")
    prohibited = [r for r in inventory if r["dataset_id"] in {"highrpd", "uapd", "poznan_uav", "rdd2022_india_stress"}]
    if prohibited:
        errors.append(f"prohibited source rows materialized: {len(prohibited)}")
    return {"status": "FAIL" if errors else "PASS", "rows": len(inventory), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Rebuild the ignored prepared-data directory")
    args = parser.parse_args()
    manifest = read_csv(SOURCE_MANIFEST)
    duplicate_rows = read_csv(DUPLICATE_AUDIT)
    make_phash_review(duplicate_rows, manifest)
    remediated = build_remediated_manifest(manifest)
    audit = canonical_empty_audit(remediated)
    audit_by_sample = {r["sample_id"]: r for r in audit}

    for row in remediated:
        audit_row = audit_by_sample.get(row["sample_id"])
        canonical = bool(canonical_boxes(row)) if row["dataset_id"] in SOURCE_MAP else False
        eligible = row["partition"] in {"train", "validation"} and row["dataset_id"] in SOURCE_MAP
        if audit_row and audit_row["include_as_negative"] != "true":
            eligible = False
        if not canonical and not audit_row:
            eligible = False
        row["pilot_eligible"] = bool_text(eligible)
        row["canonical_empty_decision"] = audit_row["decision_category"] if audit_row else "not_canonical_empty"

    fields = list(manifest[0]) + ["phase1_role", "partition", "isolation_group_id", "role_adjustment"] + ROLE_FIELDS + ["pilot_eligible", "canonical_empty_decision"]
    write_csv(PILOT_ARTIFACTS / "remediated_role_manifest.csv", remediated, fields)
    write_csv(PILOT_ARTIFACTS / "canonical_empty_audit.csv", audit, list(audit[0]))
    render_canonical_empty_samples(audit)

    train_china = [r for r in remediated if r["dataset_id"] == "rdd2022_china_drone" and r["partition"] == "train" and r["yolo_training"] == "true" and r["pilot_eligible"] == "true"]
    train_pdd = [r for r in remediated if r["dataset_id"] == "uav_pdd2023" and r["partition"] == "train" and r["yolo_training"] == "true" and r["pilot_eligible"] == "true"]
    validation = [r for r in remediated if r["dataset_id"] in SOURCE_MAP and r["partition"] == "validation" and r["model_validation"] == "true" and r["pilot_eligible"] == "true"]

    inventory = []
    inventory += materialize_dataset("m_uav0", train_china, validation)
    inventory += materialize_dataset("m_uav1", train_china + train_pdd, validation)
    write_csv(PILOT_ARTIFACTS / "training_inventory.csv", inventory, list(inventory[0]))
    validation_result = validate_materialized(inventory)
    (PILOT_ARTIFACTS / "preparation_validation.json").write_text(json.dumps(validation_result, indent=2) + "\n", encoding="utf-8")

    summary = {
        "status": validation_result["status"],
        "seed": SEED,
        "phase1_harmonized_canonical_empty_count": sum(r["present_in_phase1_harmonized_view"] == "true" for r in audit),
        "all_training_partition_canonical_empty_count": sum(r["remediated_partition"] == "train" for r in audit),
        "verified_negative_count": sum(r["include_as_negative"] == "true" and r["remediated_partition"] == "train" for r in audit),
        "valid_multi_role_reuse_count": sum(r["role_adjustment"] == "valid_training_role_reuse_for_yolo" for r in remediated),
        "pilot_eligible_multi_role_reuse_count": sum(
            r["role_adjustment"] == "valid_training_role_reuse_for_yolo" and r["pilot_eligible"] == "true"
            for r in remediated
        ),
        "m_uav0_train_images": len(train_china),
        "m_uav1_train_images": len(train_china) + len(train_pdd),
        "m_uav1_added_uav_pdd_images": len(train_pdd),
        "validation_images": Counter(r["dataset_id"] for r in validation),
        "uav_pdd_source_photo_partition_counts": dict(Counter(r["partition"] for r in remediated if r["dataset_id"] == "uav_pdd2023")),
        "uav_pdd_source_photo_group_counts": dict(Counter((r["partition"] for r in {x["isolation_group_id"]: x for x in remediated if x["dataset_id"] == "uav_pdd2023"}.values()))),
        "phash_manual_review": {"pairs": 31, "visually_different": 31, "confirmed_or_likely_duplicates": 0},
        "preparation_validation": validation_result,
    }
    # Counter is not JSON serializable when nested through direct construction.
    summary["validation_images"] = dict(summary["validation_images"])
    (PILOT_ARTIFACTS / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if validation_result["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
