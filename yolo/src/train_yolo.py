#!/usr/bin/env python3
"""Train YOLOv8n road-defect detector on RDD2022 for RoadSentinel."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
YOLO_ROOT = HERE.parent
REPO_ROOT = YOLO_ROOT.parent


def train(
    data_yaml: Path,
    base_model: str = "yolov8n.pt",
    epochs: int = 3,
    imgsz: int = 512,
    batch_size: int = 32,
    device: str = "0",
    patience: int = 7,
    seed: int = 42,
    project: Path = YOLO_ROOT / "outputs" / "training",
    name: str = "rdd2022_baseline",
    weights_dir: Path = YOLO_ROOT / "weights",
) -> dict:
    start_time = time.time()
    project.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model} (COCO pretrained initialization)")
    model = YOLO(base_model)

    print(
        f"Starting training on device={device}, epochs={epochs}, imgsz={imgsz}, "
        f"batch={batch_size}, patience={patience}, seed={seed}..."
    )
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        workers=4,
        patience=patience,
        seed=seed,
        plots=True,
        save=True,
        val=True,
    )

    elapsed_s = time.time() - start_time
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project / name

    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    dest_best = weights_dir / "best.pt"
    dest_last = weights_dir / "last.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, dest_best)
        print(f"Copied best weights to {dest_best} ({dest_best.stat().st_size} bytes)")
    if last_pt.exists():
        shutil.copy2(last_pt, dest_last)
        print(f"Copied last weights to {dest_last} ({dest_last.stat().st_size} bytes)")

    # Extract validation metrics from training results if available
    metrics = {}
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            metrics[k] = float(v) if isinstance(v, (int, float)) else str(v)

    summary = {
        "status": "COMPLETED",
        "model": "yolov8n",
        "initialization": "yolov8n.pt (COCO pretrained)",
        "fine_tuned_weights_path": str(dest_best),
        "save_dir": str(save_dir),
        "data_yaml": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": device,
        "patience": patience,
        "seed": seed,
        "duration_seconds": round(elapsed_s, 2),
        "metrics": metrics,
    }

    summary_file = save_dir / "training_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Training summary saved to {summary_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=YOLO_ROOT / "configs" / "data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base pretrained model checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID or 'cpu'")
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Early-stopping patience in epochs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument(
        "--name",
        type=str,
        default="rdd2022_baseline",
        help="Run directory name within the training project",
    )
    args = parser.parse_args()

    summary = train(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        name=args.name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
