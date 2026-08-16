"""
test_part1_pipeline.py

End-to-end test for Part 1: build (or load) the healthy-road reference, score
a test image, overlay the anomaly heatmap, threshold it into candidate boxes,
and save the visualized output.

Usage:
    python test_part1_pipeline.py

Expects:
    ml/data/healthy_road/   -> a handful of images of clean, undamaged road
    ml/data/test_road/      -> one or more images to test detection on
Outputs:
    ml/outputs/<test_image_name>_heatmap.jpg   -> heatmap overlay
    ml/outputs/<test_image_name>_boxes.jpg     -> candidate boxes drawn on image
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dinov2_features import DinoV2FeatureExtractor
from anomaly_detector import RoadAnomalyDetector, overlay_heatmap_on_image
from thresholding import heatmap_to_candidate_boxes

BASE_DIR = Path(__file__).resolve().parent.parent  # -> ml/
HEALTHY_DIR = BASE_DIR / "data" / "healthy_road"
TEST_DIR = BASE_DIR / "data" / "test_road"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REFERENCE_PATH = MODELS_DIR / "healthy_road_reference.npy"


def get_or_build_reference(extractor: DinoV2FeatureExtractor) -> RoadAnomalyDetector:
    detector = RoadAnomalyDetector(extractor, k_neighbors=3)

    if REFERENCE_PATH.exists():
        detector.load_reference(REFERENCE_PATH)
        return detector

    healthy_paths = list(HEALTHY_DIR.glob("*.jpg")) + list(HEALTHY_DIR.glob("*.png"))
    if not healthy_paths:
        raise FileNotFoundError(
            f"No healthy-road images found in {HEALTHY_DIR}. "
            f"Add a few clean-road photos there before running this script."
        )

    print(f"No saved reference found — building one from {len(healthy_paths)} healthy-road image(s)...")
    detector.build_reference(healthy_paths)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    detector.save_reference(REFERENCE_PATH)
    return detector


def draw_boxes(image: Image.Image, boxes: list[dict]) -> np.ndarray:
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for b in boxes:
        x, y, w, h = b["bbox"]
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img_bgr, f"{b['score']:.2f}", (x, max(y - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img_bgr


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    extractor = DinoV2FeatureExtractor()
    detector = get_or_build_reference(extractor)

    test_paths = list(TEST_DIR.glob("*.jpg")) + list(TEST_DIR.glob("*.png"))
    if not test_paths:
        raise FileNotFoundError(
            f"No test images found in {TEST_DIR}. Add a road photo to test detection on."
        )

    for test_path in test_paths:
        print(f"\nProcessing {test_path.name} ...")
        image = Image.open(test_path).convert("RGB")

        patch_scores, heatmap_norm = detector.score_image(image)

        overlay = overlay_heatmap_on_image(image, heatmap_norm)
        heatmap_out_path = OUTPUTS_DIR / f"{test_path.stem}_heatmap.jpg"
        cv2.imwrite(str(heatmap_out_path), overlay)
        print(f"  Saved heatmap overlay -> {heatmap_out_path}")

        boxes = heatmap_to_candidate_boxes(heatmap_norm, threshold=0.6, min_area_px=200)
        print(f"  Found {len(boxes)} candidate anomaly region(s)")

        boxed_image = draw_boxes(image, boxes)
        boxes_out_path = OUTPUTS_DIR / f"{test_path.stem}_boxes.jpg"
        cv2.imwrite(str(boxes_out_path), boxed_image)
        print(f"  Saved candidate boxes -> {boxes_out_path}")


if __name__ == "__main__":
    main()
