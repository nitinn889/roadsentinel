#!/usr/bin/env python3
"""Standard standalone YOLO road-defect inference runner for RoadSentinel.

Processes an image or a folder of images, saves annotated visual overlays,
and emits structured, machine-readable JSON results.
Does NOT depend on DINOv2, SAM2, or XGBoost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights" / "best.pt"
DEFAULT_OUTPUT = HERE / "outputs" / "inference"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Visual annotation colors per class (BGR)
CLASS_COLORS = {
    "D00": (0, 140, 255),    # Orange: Longitudinal Crack
    "D10": (0, 255, 255),    # Yellow: Transverse Crack
    "D20": (255, 0, 255),    # Magenta: Alligator Crack
    "D40": (0, 0, 255),      # Red: Pothole
    "Repair": (0, 255, 0),   # Green: Repair/Patch
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(
    image: cv2.typing.MatLike,
    detections: List[Dict[str, Any]],
) -> cv2.typing.MatLike:
    """Draw bounding boxes and class labels on image."""
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_name = det["class"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        # Draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Draw label background and text
        label = f"{cls_name} {conf:.2f}"
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            canvas,
            (x1, max(0, y1 - h - baseline - 4)),
            (x1 + w + 4, max(h + baseline + 4, y1)),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, max(y1 - 4, h + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255) if sum(color) < 400 else (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_inference(
    input_path: Path,
    output_dir: Path,
    weights_path: Path = DEFAULT_WEIGHTS,
    conf_threshold: float = 0.25,
    device: Optional[str] = None,
    save_annotated: bool = True,
) -> Dict[str, Any]:
    """Run YOLO road defect inference over an image file or directory of images."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found at: {weights_path}. "
            "Please train the model first or supply valid weights via --weights."
        )

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Loading YOLO defect model from {weights_path} (device: {device})...")
    model = YOLO(str(weights_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / "annotated"
    if save_annotated:
        images_output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not image_files:
        raise ValueError(f"No image files found at: {input_path}")

    print(f"Running inference on {len(image_files)} image(s)...")

    results_list = []
    total_detections = 0

    for img_file in image_files:
        raw_bgr = cv2.imread(str(img_file))
        if raw_bgr is None:
            print(f"Warning: Could not read image {img_file}; skipping.")
            continue

        h, w = raw_bgr.shape[:2]
        t0 = time.perf_counter()
        preds = model.predict(
            source=raw_bgr,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict[str, Any]] = []
        if preds and len(preds) > 0:
            boxes = preds[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()

                for box, score, cls_id in zip(xyxy, confs, cls_ids):
                    cid = int(cls_id)
                    cname = model.names.get(cid, str(cid))
                    b_int = [int(round(v)) for v in box]
                    detections.append({
                        "class": cname,
                        "class_id": cid,
                        "confidence": float(round(score, 4)),
                        "bbox": b_int,
                    })

        total_detections += len(detections)

        # Save annotated image
        annotated_filename = None
        if save_annotated:
            annotated_bgr = draw_detections(raw_bgr, detections)
            annotated_filename = f"annotated_{img_file.name}"
            out_img_path = images_output_dir / annotated_filename
            cv2.imwrite(str(out_img_path), annotated_bgr)

        results_list.append({
            "image_name": img_file.name,
            "image_path": str(img_file.resolve()),
            "width": w,
            "height": h,
            "inference_ms": round(infer_ms, 2),
            "defect_count": len(detections),
            "annotated_image": str(images_output_dir / annotated_filename) if annotated_filename else None,
            "detections": detections,
        })

    summary = {
        "model_weights": str(weights_path.resolve()),
        "classes": model.names,
        "confidence_threshold": conf_threshold,
        "device": device,
        "total_images": len(results_list),
        "total_defects_detected": total_detections,
        "results": results_list,
    }

    results_file = output_dir / "results.json"
    results_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Inference complete: {len(results_list)} images processed, {total_detections} defects detected.")
    print(f"Summary JSON saved to: {results_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RoadSentinel YOLO road defect detector on an image or folder."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an input image or directory of images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory to save annotated images and results.json",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained YOLO weights file (default: yolo/weights/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('0', '1', 'cpu', etc.)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Do not save visual annotated image overlays",
    )

    args = parser.parse_args()
    run_inference(
        input_path=args.input,
        output_dir=args.output,
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        save_annotated=not args.no_render,
    )


if __name__ == "__main__":
    main()
