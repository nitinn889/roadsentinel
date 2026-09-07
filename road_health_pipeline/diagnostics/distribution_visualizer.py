"""Observability and visual diagnostics for the RoadSentinel ML pipeline.

Visualizes:
1. Intermediate pipeline outputs (6-panel diagnostic board):
   - Input RGB image
   - Road surface mask
   - Raw DINOv2 patch anomaly heatmap
   - Local contrast & spatial consistency filter map
   - SAM2 refined defect segmentation contours
   - Final hierarchical assessment overlay with road health score

2. Anomaly score distributions:
   - Real-world vs simulated healthy road distributions
   - Separation margins and cumulative distribution functions (CDFs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class DistributionVisualizer:
    """Diagnostic tool to render intermediate pipeline stages and distribution plots."""

    @staticmethod
    def render_diagnostic_board(
        rgb_image: np.ndarray,
        road_mask: Optional[np.ndarray],
        raw_anomaly_map: np.ndarray,
        filtered_anomaly_map: np.ndarray,
        candidates: List[Any],
        hierarchical_results: List[Any],
        road_health_score: float = 100.0,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Create a comprehensive 6-panel diagnostic visualization board.

        Layout: 2 rows x 3 columns
        [1. RGB Input]            [2. Road Mask]         [3. Raw DINOv2 Heatmap]
        [4. Filtered Contrast]    [5. SAM2 Contours]     [6. Hierarchical Decision & HUD]
        """
        h, w = rgb_image.shape[:2]
        target_w, target_h = 640, 360

        def prep(img: np.ndarray) -> np.ndarray:
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Panel 1: Input RGB
        p1 = cv2.cvtColor(prep(rgb_image), cv2.COLOR_RGB2BGR)
        cv2.putText(p1, "1. Input RGB Frame", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Panel 2: Road Surface Mask
        if road_mask is not None and np.any(road_mask):
            mask_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            overlay = mask_rgb.copy()
            overlay[road_mask] = (0, 200, 0)  # Green tint for drivable road
            p2 = cv2.addWeighted(mask_rgb, 0.6, overlay, 0.4, 0)
        else:
            p2 = np.full((h, w, 3), 40, dtype=np.uint8)
        p2 = prep(p2)
        cv2.putText(p2, "2. Segmented Road Surface", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Panel 3: Raw DINOv2 Anomaly Map
        raw_norm = cv2.normalize(raw_anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p3 = cv2.applyColorMap(raw_norm, cv2.COLORMAP_JET)
        p3 = prep(p3)
        cv2.putText(p3, "3. Raw DINOv2 Heatmap", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Panel 4: Filtered Contrast & Spatial Consistency Map
        filt_norm = cv2.normalize(filtered_anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p4 = cv2.applyColorMap(filt_norm, cv2.COLORMAP_INFERNO)
        p4 = prep(p4)
        cv2.putText(p4, "4. Spatial & Contrast Filter", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Panel 5: SAM2 Refined Defect Masks
        p5 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        mask_layer = np.zeros_like(p5)
        for cand in candidates:
            m = getattr(cand, "mask", None)
            if m is not None and np.any(m):
                mask_layer[m] = (0, 0, 255)  # Red for defect
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(p5, contours, -1, (255, 255, 255), 2)
        p5 = cv2.addWeighted(p5, 0.75, mask_layer, 0.25, 0)
        p5 = prep(p5)
        cv2.putText(p5, "5. SAM2 Defect Contours", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Panel 6: Final Hierarchical Assessment & HUD
        p6 = cv2.cvtColor(prep(rgb_image), cv2.COLOR_RGB2BGR)
        scale_x = target_w / w
        scale_y = target_h / h

        for h_res, cand in zip(hierarchical_results, candidates):
            bx = getattr(cand, "bbox_xyxy", [0, 0, 0, 0])
            x1 = int(bx[0] * scale_x)
            y1 = int(bx[1] * scale_y)
            x2 = int(bx[2] * scale_x)
            y2 = int(bx[3] * scale_y)

            final_type = getattr(h_res, "final_type", "pothole")
            dtype_str = final_type.value if hasattr(final_type, "value") else str(final_type)
            conf = getattr(h_res, "calibrated_confidence", 0.5)

            if "pothole" in dtype_str.lower():
                color = (0, 70, 255) if "water" not in dtype_str.lower() else (255, 180, 0)
            elif "crack" in dtype_str.lower():
                color = (0, 220, 255)
            else:
                color = (180, 180, 180)

            cv2.rectangle(p6, (x1, y1), (x2, y2), color, 2)
            tag = f"{dtype_str.split('_')[0].upper()} {conf:.2f}"
            cv2.putText(p6, tag, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Road Health HUD
        health_color = (0, 220, 0) if road_health_score >= 80 else ((0, 180, 255) if road_health_score >= 60 else (0, 0, 255))
        hud_text = f"Road Health: {road_health_score:.1f}/100"
        cv2.rectangle(p6, (10, 10), (280, 45), (20, 20, 20), -1)
        cv2.putText(p6, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, health_color, 2)

        cv2.putText(p6, "6. Hierarchical Assessment HUD", (15, target_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Assemble 2x3 grid
        row1 = np.hstack([p1, p2, p3])
        row2 = np.hstack([p4, p5, p6])
        board = np.vstack([row1, row2])

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), board)
            log.info("Saved 6-panel diagnostic board to %s", output_path)

        return board

    @staticmethod
    def render_distribution_histogram(
        real_healthy_scores: np.ndarray,
        sim_healthy_scores: np.ndarray,
        defect_scores: np.ndarray,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Render OpenCV-based score distribution histogram comparing real, sim, and defects."""
        plot_w, plot_h = 800, 400
        canvas = np.full((plot_h, plot_w, 3), 24, dtype=np.uint8)

        bins = np.linspace(0.0, 1.0, 41)

        def get_hist(arr: np.ndarray) -> np.ndarray:
            if len(arr) == 0:
                return np.zeros(40, dtype=np.float32)
            h, _ = np.histogram(arr, bins=bins, density=True)
            return h

        h_real = get_hist(real_healthy_scores)
        h_sim = get_hist(sim_healthy_scores)
        h_defect = get_hist(defect_scores)

        max_density = max(1e-4, float(np.max([h_real.max(), h_sim.max(), h_defect.max()])))

        # Coordinate transform
        def pt(b_idx: int, val: float) -> Tuple[int, int]:
            x = int(60 + (b_idx / 40.0) * (plot_w - 100))
            y = int(plot_h - 50 - (val / max_density) * (plot_h - 100))
            return x, y

        # Draw grid
        cv2.line(canvas, (60, plot_h - 50), (plot_w - 40, plot_h - 50), (120, 120, 120), 1)
        cv2.line(canvas, (60, 40), (60, plot_h - 50), (120, 120, 120), 1)

        # Plot curves
        for i in range(39):
            # Real Healthy: Green
            cv2.line(canvas, pt(i, h_real[i]), pt(i + 1, h_real[i + 1]), (0, 220, 100), 2)
            # Sim Healthy: Blue
            cv2.line(canvas, pt(i, h_sim[i]), pt(i + 1, h_sim[i + 1]), (255, 160, 0), 2)
            # Defect: Red
            cv2.line(canvas, pt(i, h_defect[i]), pt(i + 1, h_defect[i + 1]), (0, 60, 255), 2)

        # Legend & Labels
        cv2.putText(canvas, "DINOv2 Score Distribution Inspector", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Real Healthy Road", (plot_w - 240, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 1)
        cv2.putText(canvas, "Sim Healthy Road", (plot_w - 240, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1)
        cv2.putText(canvas, "Defect (Pothole/Crack)", (plot_w - 240, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1)

        cv2.putText(canvas, "Anomaly Score (0 -> 1)", (plot_w // 2 - 80, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), canvas)
            log.info("Saved score distribution histogram to %s", output_path)

        return canvas
