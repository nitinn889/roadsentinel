#!/bin/bash
cd /home/nitin-nandakumar/Downloads/roadsentinel/road_health_pipeline
source .venv/bin/activate

echo "========================================"
echo " Evaluating pothole600_test (Segmentation)"
echo "========================================"
python scripts/evaluate_real_dataset.py --mode segmentation --subset pothole600_test --device cuda

echo "========================================"
echo " Evaluating pothole600_val (Segmentation)"
echo "========================================"
python scripts/evaluate_real_dataset.py --mode segmentation --subset pothole600_val --device cuda

echo "========================================"
echo " Evaluating china_drone (RDD XML)"
echo "========================================"
python scripts/evaluate_real_dataset.py --mode rdd_xml --subset china_drone --device cuda

echo "========================================"
echo " Evaluating india (RDD XML)"
echo "========================================"
python scripts/evaluate_real_dataset.py --mode rdd_xml --subset india --device cuda

echo "Full Dataset Evaluation Complete!"

echo "========================================"
echo " Generating VLM Work Orders (Critical / High Defects)"
echo "========================================"
python vlm_work_order_gen.py

