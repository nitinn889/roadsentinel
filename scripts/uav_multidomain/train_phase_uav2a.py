#!/usr/bin/env python3
"""Run the two leakage-safe Phase UAV-2A YOLOv8n smoke or pilot jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/uav_multidomain/pilot"
PREPARED = ROOT / "data/external_uav/prepared/uav2a_pilot"
RUN_ROOT = ROOT / "outputs/uav2a_pilot"
BASE_MODEL = ROOT / "yolov8n.pt"
SEED = 20260912
MODEL_NAMES = {"m_uav0": "M-UAV0", "m_uav1": "M-UAV1"}

COMMON_CONFIG = {
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 4,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "cos_lr": True,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "deterministic": True,
    "amp": True,
    "cache": False,
    "plots": True,
    "save": True,
    "val": True,
    "seed": SEED,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run this GPU-only pilot")
    device_name = torch.cuda.get_device_name(0)
    if "RTX 5060" not in device_name:
        raise RuntimeError(f"Expected RTX 5060, detected {device_name}; refusing to run")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    import ultralytics

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_available": True,
        "gpu": device_name,
        "nvidia_smi": query,
        "ultralytics": ultralytics.__version__,
        "base_model": str(BASE_MODEL.relative_to(ROOT)),
    }


def inventory_counts(model: str) -> tuple[int, int]:
    inventory = pd.read_csv(ARTIFACTS / "training_inventory.csv")
    subset = inventory[(inventory["model_dataset"] == model) & (inventory["split"] == "train")]
    return len(subset), subset["isolation_group_id"].nunique()


def run_record_path() -> Path:
    return ARTIFACTS / "pilot_runs.csv"


def write_run_record(record: dict) -> None:
    path = run_record_path()
    rows: list[dict] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [r for r in rows if not (r["stage"] == record["stage"] and r["model_id"] == record["model_id"])]
    rows.append(record)
    rows.sort(key=lambda r: (r["stage"], r["model_id"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pilot_smokes_passed() -> bool:
    path = run_record_path()
    if not path.exists():
        return False
    rows = pd.read_csv(path).fillna("")
    passed = rows[(rows["stage"] == "smoke") & (rows["status"] == "PASS")]
    return set(passed["model_id"]) == set(MODEL_NAMES)


def parse_results(save_dir: Path) -> dict:
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        raise RuntimeError(f"Training results missing: {results_csv}")
    frame = pd.read_csv(results_csv)
    frame.columns = [c.strip() for c in frame.columns]
    loss_columns = [c for c in frame.columns if c.startswith("train/") and c.endswith("loss")]
    if not loss_columns:
        raise RuntimeError("No training loss columns in results.csv")
    losses = frame[loss_columns].apply(pd.to_numeric, errors="coerce")
    first_loss = float(losses.iloc[0].sum())
    final_loss = float(losses.iloc[-1].sum())
    finite = bool(losses.notna().all().all())
    map_column = next((c for c in frame.columns if "mAP50-95" in c), None)
    if map_column is None:
        raise RuntimeError("Validation mAP50-95 column missing from results.csv")
    maps = pd.to_numeric(frame[map_column], errors="coerce")
    best_epoch = int(maps.idxmax()) + 1
    return {
        "epochs_completed": len(frame),
        "loss_columns": loss_columns,
        "first_total_train_loss": first_loss,
        "final_total_train_loss": final_loss,
        "loss_finite": finite,
        "loss_stable": finite and final_loss <= first_loss * 1.10,
        "best_epoch": best_epoch,
        "best_map50_95": float(maps.max()),
        "final_map50_95": float(maps.iloc[-1]),
    }


def train_one(model_id: str, stage: str) -> dict:
    env = environment()
    epochs = 2 if stage == "smoke" else 40
    patience = 2 if stage == "smoke" else 10
    close_mosaic = 0 if stage == "smoke" else 10
    data_yaml = PREPARED / model_id / "dataset.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"Prepared dataset missing: {data_yaml}")
    if not BASE_MODEL.exists():
        raise RuntimeError(f"Pretrained initialization missing: {BASE_MODEL}")
    if stage == "pilot" and not pilot_smokes_passed():
        raise RuntimeError("Both smoke tests must pass before pilot training")

    project = RUN_ROOT / stage
    run_name = model_id
    save_dir = project / run_name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    project.mkdir(parents=True, exist_ok=True)

    train_images, train_groups = inventory_counts(model_id)
    config = dict(COMMON_CONFIG)
    config.update({
        "data": str(data_yaml),
        "epochs": epochs,
        "patience": patience,
        "close_mosaic": close_mosaic,
        "project": str(project),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
    })

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = utc_now()
    start_s = time.perf_counter()
    model = YOLO(str(BASE_MODEL))
    result = model.train(**config)
    duration = time.perf_counter() - start_s
    peak_gib = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    actual_save_dir = Path(result.save_dir)
    parsed = parse_results(actual_save_dir)
    best = actual_save_dir / "weights/best.pt"
    last = actual_save_dir / "weights/last.pt"
    checkpoints_ok = best.exists() and last.exists()
    status = "PASS" if parsed["loss_stable"] and checkpoints_ok and parsed["epochs_completed"] > 0 else "FAIL"
    batch = COMMON_CONFIG["batch"]
    steps_per_epoch = math.ceil(train_images / batch)
    record = {
        "stage": stage,
        "model_id": model_id,
        "display_name": MODEL_NAMES[model_id],
        "status": status,
        "started_utc": started,
        "completed_utc": utc_now(),
        "duration_seconds": f"{duration:.3f}",
        "epochs_requested": str(epochs),
        "epochs_completed": str(parsed["epochs_completed"]),
        "best_epoch": str(parsed["best_epoch"]),
        "train_images": str(train_images),
        "train_groups": str(train_groups),
        "batch_size": str(batch),
        "steps_per_epoch": str(steps_per_epoch),
        "optimization_steps": str(steps_per_epoch * parsed["epochs_completed"]),
        "first_total_train_loss": f"{parsed['first_total_train_loss']:.8f}",
        "final_total_train_loss": f"{parsed['final_total_train_loss']:.8f}",
        "loss_finite": str(parsed["loss_finite"]).lower(),
        "loss_stable": str(parsed["loss_stable"]).lower(),
        "validation_executed": "true",
        "checkpoints_written": str(checkpoints_ok).lower(),
        "best_map50_95": f"{parsed['best_map50_95']:.8f}",
        "peak_gpu_memory_gib": f"{peak_gib:.4f}",
        "device": env["gpu"],
        "pytorch": env["pytorch"],
        "cuda": env["pytorch_cuda"],
        "ultralytics": env["ultralytics"],
        "driver": env["nvidia_smi"].split(",")[1].strip(),
        "data_yaml": data_yaml.relative_to(ROOT).as_posix(),
        "run_dir": actual_save_dir.relative_to(ROOT).as_posix(),
        "best_weights": best.relative_to(ROOT).as_posix(),
        "last_weights": last.relative_to(ROOT).as_posix(),
    }
    write_run_record(record)
    config_artifact = {
        "stage": stage,
        "model_id": model_id,
        "initialization": str(BASE_MODEL.relative_to(ROOT)),
        "environment": env,
        "training_arguments": config,
        "evaluation_settings": {"confidence": 0.25, "iou_match": 0.50, "nms_iou": 0.70},
    }
    (ARTIFACTS / f"training_config_{stage}_{model_id}.json").write_text(json.dumps(config_artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if status != "PASS":
        raise RuntimeError(f"{stage} {model_id} failed post-training checks")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--model", choices=("m_uav0", "m_uav1", "all"), default="all")
    args = parser.parse_args()
    models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    environment_data = environment()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "environment.json").write_text(json.dumps(environment_data, indent=2) + "\n", encoding="utf-8")
    for model_id in models:
        train_one(model_id, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
