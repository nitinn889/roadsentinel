# RoadSentinel Phase 7 Dashboard Assets — Perception Module

This directory contains the frozen, paper-ready perception benchmark artifacts and assets produced by Phase 5 and Phase 6 for consumption by the Phase 7 Interactive Dashboard.

## Files & Assets

1. **`perception_handoff_manifest.json`**:
   - Master machine-readable manifest with frozen configuration flags, primary metrics, secondary sensitivity, model architecture specs, and relative asset paths.

2. **Tables**:
   - `FINAL_PERCEPTION_TABLE.csv`: High-level scientific comparison table with methodology types, strengths, and limitations.
   - `common_binary_metrics.csv`: Primary binary localization metrics at IoU ≥ 0.50 (TP, FP, FN, P, R, F1, IoU, latency, FPS).
   - `common_binary_sensitivity_iou25.csv`: Secondary sensitivity metrics at IoU ≥ 0.25.
   - `yolo_semantic_metrics.csv`: Detailed 5-class semantic evaluation for supervised YOLOv8n (D00, D10, D20, D40, Repair).
   - `panel_examples.csv`: Manifest of 8 curated qualitative visual panels for dashboard case-study tabs.

3. **Research Figures** (located in `benchmark/final_comparison/figures/`):
   - `fig1_precision_recall_f1.png`: P/R/F1 comparison bar chart.
   - `fig2_latency_throughput_log.png`: Log-scale latency and FPS comparison.
   - `fig3_matched_bbox_iou.png`: Matched bounding-box IoU comparison.
   - `fig4_yolo_class_wise_metrics.png`: YOLO semantic class metrics.
   - `fig5_dino_failure_mode_breakdown.png`: DINO/SAM false positive and false negative breakdowns.
   - `fig6_iou_sensitivity_comparison.png`: IoU 0.50 vs 0.25 sensitivity.
   - `fig7_qualitative_panel_montage.png`: Qualitative 4-panel visual montage.

4. **Visual Panel Figures** (located in `benchmark/final_comparison/panels/`):
   - 3-panel comparative figures (GT vs. YOLO vs. DINO/SAM) across representative success, failure, and cross-domain categories.
