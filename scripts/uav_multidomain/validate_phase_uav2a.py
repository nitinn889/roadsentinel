#!/usr/bin/env python3
"""Independent integrity gate for Phase UAV-2A artifacts and frozen assets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/uav_multidomain/pilot"
REQUIRED = [
    "remediated_role_manifest.csv", "canonical_empty_audit.csv", "phash_manual_review.csv",
    "training_inventory.csv", "pilot_runs.csv", "per_source_metrics.csv", "per_class_metrics.csv",
    "per_image_predictions.csv", "bootstrap_comparison.json", "go_no_go_decision.json",
]
FROZEN = {
    "yolo/weights/best.pt": "ddbea38144425aad288f05d90f07a38bcac35eeb467767643c707eb8e2fb5dcf",
    "road_health_pipeline/output/real_memory_bank/embeddings.npy": "e8d16c41e3090bb60e0f5474774c9f9d8a80c6ffcd124dd69e8a2a4e54a0236d",
    "road_health_pipeline/output/real_memory_bank/index.faiss": "d12a8120a2f9ff36f5ec5c981164b586b06e3477e0e7711adb8f6fb34c7bc1c2",
    "xgboost/model/scenario_model_v2.json": "613227f98a8c5b45fb9909ab9f3b36ba12b219f68dbd2998fd26db6d77c7e679",
    "integration/CANONICAL_RESEARCH_METRICS.json": "e1acc85a0a95c511d60f77afb1c46ef8adbc4c81904f7aa701fc321fcf84764b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    errors, checks = [], {}
    missing = [name for name in REQUIRED if not (ARTIFACTS / name).exists()]
    checks["required_artifacts"] = {"pass": not missing, "missing": missing}
    errors.extend(f"missing required artifact: {name}" for name in missing)
    if missing:
        raise SystemExit(json.dumps({"status": "FAIL", "errors": errors}, indent=2))

    inventory = pd.read_csv(ARTIFACTS / "training_inventory.csv")
    manifest = pd.read_csv(ARTIFACTS / "remediated_role_manifest.csv")
    empty = pd.read_csv(ARTIFACTS / "canonical_empty_audit.csv")
    phash = pd.read_csv(ARTIFACTS / "phash_manual_review.csv")
    runs = pd.read_csv(ARTIFACTS / "pilot_runs.csv")
    per_source = pd.read_csv(ARTIFACTS / "per_source_metrics.csv")
    per_class = pd.read_csv(ARTIFACTS / "per_class_metrics.csv")
    per_image = pd.read_csv(ARTIFACTS / "per_image_predictions.csv")

    # Comparable training arguments, excluding only dataset/run destinations.
    configs = {}
    for model in ("m_uav0", "m_uav1"):
        payload = json.loads((ARTIFACTS / f"training_config_pilot_{model}.json").read_text())
        args = dict(payload["training_arguments"])
        for key in ("data", "project", "name"):
            args.pop(key)
        configs[model] = args
    comparable = configs["m_uav0"] == configs["m_uav1"]
    checks["identical_training_settings"] = comparable
    if not comparable:
        errors.append("pilot training arguments differ beyond data/run destination")

    val0 = set(inventory[(inventory.model_dataset == "m_uav0") & (inventory.split == "val")].sample_id)
    val1 = set(inventory[(inventory.model_dataset == "m_uav1") & (inventory.split == "val")].sample_id)
    checks["identical_validation_ids"] = {"pass": val0 == val1, "count": len(val0)}
    if val0 != val1:
        errors.append("validation sample IDs differ")
    china0 = set(inventory[(inventory.model_dataset == "m_uav0") & (inventory.split == "train")].sample_id)
    china1 = set(inventory[(inventory.model_dataset == "m_uav1") & (inventory.split == "train") & (inventory.dataset_id == "rdd2022_china_drone")].sample_id)
    checks["identical_china_training_ids"] = {"pass": china0 == china1, "count": len(china0)}
    if china0 != china1:
        errors.append("China training subsets differ")

    prohibited_sources = {"highrpd", "uapd", "poznan_uav", "rdd2022_india_stress"}
    prohibited = inventory[inventory.dataset_id.isin(prohibited_sources)]
    checks["no_prohibited_sources"] = {"pass": prohibited.empty, "rows": len(prohibited)}
    if not prohibited.empty:
        errors.append("prohibited source entered pilot materialization")

    # Training and validation isolation at the strongest remediated group key.
    group_splits: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in inventory.itertuples():
        group_splits[(row.model_dataset, row.dataset_id, row.isolation_group_id)].add(row.split)
    crossing = [key for key, splits in group_splits.items() if len(splits) > 1]
    checks["no_train_validation_group_crossing"] = {"pass": not crossing, "crossing_groups": crossing[:20]}
    if crossing:
        errors.append("training/validation isolation group crossing")

    # No calibration/internal/external/severe-shift row is materialized.
    pilot_samples = set(inventory.sample_id)
    protected = manifest[manifest.partition.isin(["calibration", "internal_test", "external_test", "severe_shift_test"])]
    leaked = sorted(pilot_samples.intersection(protected.sample_id))
    checks["no_protected_partition_samples"] = {"pass": not leaked, "leaked": leaked[:20]}
    if leaked:
        errors.append("protected partition sample entered training or validation")

    negatives = inventory[(inventory.split == "train") & (inventory.total_objects == 0)]
    allowed = set(empty[(empty.include_as_negative == True) & (empty.decision_category == "Genuine healthy/background image")].sample_id)
    bad_negatives = sorted(set(negatives.sample_id) - allowed)
    checks["canonical_empty_policy"] = {"pass": not bad_negatives, "verified_negative_rows": len(negatives), "bad_samples": bad_negatives}
    if bad_negatives:
        errors.append("unapproved canonical-empty image treated as a negative")

    valid_phash_decisions = {"Confirmed duplicate", "Likely near-duplicate", "Visually different", "Unresolved"}
    phash_ok = len(phash) == 31 and set(phash.manual_decision).issubset(valid_phash_decisions)
    checks["phash_review"] = {"pass": phash_ok, "pairs": len(phash), "decision_counts": phash.manual_decision.value_counts().to_dict()}
    if not phash_ok:
        errors.append("pHash review incomplete or invalid")
    uapd_candidates = manifest[(manifest.dataset_id == "uapd") & (manifest.uapd_evaluation_candidate == True)]
    checks["uapd_independent_candidates"] = {"pass": len(uapd_candidates) == 749, "count": len(uapd_candidates), "known_china_overlap_excluded": 2401}
    if len(uapd_candidates) != 749:
        errors.append("UAPD independent-candidate count changed")

    pilots = runs[(runs.stage == "pilot") & (runs.status == "PASS")]
    checks["pilot_runs_pass"] = {"pass": set(pilots.model_id) == {"m_uav0", "m_uav1"}, "rows": len(pilots)}
    smokes = runs[(runs.stage == "smoke") & (runs.status == "PASS")]
    checks["smoke_runs_pass_and_separate"] = {"pass": set(smokes.model_id) == {"m_uav0", "m_uav1"}, "rows": len(smokes)}
    if not checks["pilot_runs_pass"]["pass"] or not checks["smoke_runs_pass_and_separate"]["pass"]:
        errors.append("smoke or pilot run records incomplete")

    expected_source_rows = {(model, source) for model in ("m_uav0", "m_uav1") for source in ("rdd2022_china_drone", "uav_pdd2023")}
    found_source_rows = set(zip(per_source.model_id, per_source.source))
    expected_class_rows = {(model, source, cls) for model, source in expected_source_rows for cls in range(4)}
    found_class_rows = set(zip(per_class.model_id, per_class.source, per_class.class_id))
    expected_image_rows = len(val0) * 2
    checks["metric_coverage"] = {
        "pass": found_source_rows == expected_source_rows and found_class_rows == expected_class_rows and len(per_image) == expected_image_rows,
        "source_rows": len(found_source_rows), "class_rows": len(found_class_rows), "image_rows": len(per_image),
    }
    if not checks["metric_coverage"]["pass"]:
        errors.append("metric artifact coverage is incomplete")

    hashes = {}
    for relative, expected in FROZEN.items():
        actual = sha256(ROOT / relative)
        hashes[relative] = {"pass": actual == expected, "expected_sha256": expected, "actual_sha256": actual}
        if actual != expected:
            errors.append(f"frozen artifact changed: {relative}")
    checks["frozen_artifact_hashes"] = hashes

    prep = json.loads((ARTIFACTS / "preparation_validation.json").read_text())
    checks["preparation_validation"] = prep
    if prep.get("status") != "PASS":
        errors.append("preparation validation did not pass")
    verdict = json.loads((ARTIFACTS / "go_no_go_decision.json").read_text())
    checks["decision"] = verdict["decision"]

    result = {"status": "FAIL" if errors else "PASS", "errors": errors, "checks": checks}
    (ARTIFACTS / "integrity_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
