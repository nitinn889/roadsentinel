"""
thresholding.py

Converts a raw anomaly heatmap into a list of candidate bounding boxes —
this becomes the input "prompt" for SAM2 in Part 2 of the pipeline (SAM2
gets pointed at these rough regions rather than scanning the whole image
blindly).
"""

import cv2
import numpy as np


def heatmap_to_candidate_boxes(
    heatmap_norm: np.ndarray,
    threshold: float = 0.6,
    min_area_px: int = 200,
) -> list[dict]:
    """
    Args:
      heatmap_norm: normalized (0-1) anomaly heatmap, full image resolution.
      threshold: patches/pixels above this normalized score are considered
        "anomalous". Adjustable — lower catches more (including faint/early
        damage), higher is stricter and reduces false positives.
      min_area_px: filters out tiny noise regions (a handful of stray pixels)
        that aren't large enough to plausibly be real road damage.

    Returns: list of dicts, each: {"bbox": (x, y, w, h), "score": float},
      where "score" is the mean heatmap value inside that region (useful later
      as an initial confidence signal before SAM2 even runs).
    """
    binary_mask = (heatmap_norm >= threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    candidates = []
    for label_id in range(1, num_labels):  # skip label 0 = background
        x, y, w, h, area = stats[label_id]
        if area < min_area_px:
            continue

        region_mask = labels[y:y + h, x:x + w] == label_id
        region_heat = heatmap_norm[y:y + h, x:x + w]
        mean_score = float(region_heat[region_mask].mean())

        candidates.append({"bbox": (int(x), int(y), int(w), int(h)), "score": mean_score})

    # Strongest anomalies first — useful once we start feeding these into SAM2,
    # so the most likely real detections get processed/reviewed first.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


if __name__ == "__main__":
    # Smoke test: a synthetic heatmap with one obvious "hot" region.
    dummy_heatmap = np.zeros((300, 300), dtype=np.float32)
    dummy_heatmap[100:160, 120:200] = 0.9  # simulated anomalous patch
    dummy_heatmap[10:20, 10:15] = 0.65     # small region, should be filtered by min_area_px

    boxes = heatmap_to_candidate_boxes(dummy_heatmap, threshold=0.6, min_area_px=200)
    print(f"Found {len(boxes)} candidate region(s):")
    for b in boxes:
        print(f"  bbox={b['bbox']}, score={b['score']:.3f}")
