from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import pytest

from vlm_work_order_gen import (
    construct_work_order_prompt,
    crop_defect_region,
    generate_fallback_work_order,
    generate_vlm_work_orders,
    locate_defect_image,
)


def test_construct_work_order_prompt():
    prompt = construct_work_order_prompt(
        defect_class="water_filled_pothole",
        severity_tier="critical",
        severity_score=0.92,
        area_m2=1.25,
        estimated_depth_m=0.15,
        is_water_filled=True,
        road_segment_id="seg_test_1001",
        pothole_id="pothole-001",
    )
    assert "seg_test_1001" in prompt
    assert "water_filled_pothole" in prompt
    assert "CRITICAL" in prompt
    assert "1.25 m²" in prompt
    assert "15.0 cm" in prompt
    assert "Hazardous Standing Water Present" in prompt
    assert "3-sentence" in prompt


def test_generate_fallback_work_order_critical_water():
    res = generate_fallback_work_order(
        defect_class="water_filled_pothole",
        severity_tier="critical",
        severity_score=0.95,
        area_m2=1.5,
        estimated_depth_m=0.18,
        is_water_filled=True,
        road_segment_id="seg_test_water",
        pothole_id="pothole-water-01",
    )
    text = res["work_order_text"]
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    assert len(sentences) >= 3
    assert "URGENT" in text
    assert "seg_test_water" in text
    assert "submersible pump" in text.lower() or "dewater" in text.lower()
    assert "MUTCD" in text
    assert len(res["required_materials"]) >= 3
    assert len(res["required_equipment"]) >= 3


def test_generate_fallback_work_order_high_dry():
    res = generate_fallback_work_order(
        defect_class="pothole",
        severity_tier="high",
        severity_score=0.72,
        area_m2=0.6,
        estimated_depth_m=0.08,
        is_water_filled=False,
        road_segment_id="seg_test_dry",
        pothole_id="pothole-dry-01",
    )
    text = res["work_order_text"]
    assert "seg_test_dry" in text
    assert "tack coat" in text.lower() or "asphalt" in text.lower()
    assert res["estimated_crew_size"] >= 2


def test_crop_defect_region():
    # Create test 100x100 RGB image
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    bbox = [20, 30, 60, 70]
    cropped = crop_defect_region(img, bbox, padding_fraction=0.10)
    assert cropped.width > 40
    assert cropped.height > 40
    assert cropped.width <= 100
    assert cropped.height <= 100


def test_generate_vlm_work_orders_filters_severities():
    sample_result = {
        "road_segment_id": "seg_mixed_defects",
        "total_defects": 3,
        "max_severity": 0.91,
        "detections": [
            {
                "pothole_id": "def-low",
                "defect_type": "surface_wear",
                "severity_score": 0.20,
                "severity_tier": "low",
                "area_m2": 0.1,
                "bbox_xyxy": [10, 10, 20, 20],
            },
            {
                "pothole_id": "def-high",
                "defect_type": "alligator_crack",
                "severity_score": 0.75,
                "severity_tier": "high",
                "area_m2": 0.8,
                "bbox_xyxy": [50, 50, 150, 150],
            },
            {
                "pothole_id": "def-critical",
                "defect_type": "water_filled_pothole",
                "severity_score": 0.91,
                "severity_tier": "critical",
                "area_m2": 1.4,
                "water_flag": True,
                "bbox_xyxy": [200, 200, 350, 350],
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        res_file = tmp_path / "result.json"
        res_file.write_text(json.dumps(sample_result), encoding="utf-8")
        out_file = tmp_path / "work_orders.json"

        work_orders_doc = generate_vlm_work_orders(
            result_data=res_file,
            output_path=out_file,
        )

        assert out_file.exists()
        saved = json.loads(out_file.read_text(encoding="utf-8"))

        # Should only have 2 work orders (high and critical), low excluded
        assert len(saved["work_orders"]) == 2
        pothole_ids = [wo["pothole_id"] for wo in saved["work_orders"]]
        assert "def-high" in pothole_ids
        assert "def-critical" in pothole_ids
        assert "def-low" not in pothole_ids

        assert "seg_mixed_defects" in saved["by_segment"]
        assert saved["by_segment"]["seg_mixed_defects"]["highest_severity_tier"] == "critical"


def test_cli_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        res_file = tmp_path / "result.json"
        sample_result = {
            "road_segment_id": "seg_cli_test",
            "detections": [
                {
                    "pothole_id": "cli-pothole-01",
                    "defect_type": "pothole",
                    "severity_score": 0.88,
                    "area_m2": 0.9,
                    "water_flag": False,
                }
            ],
        }
        res_file.write_text(json.dumps(sample_result), encoding="utf-8")
        out_file = tmp_path / "work_orders.json"

        script_path = Path(__file__).resolve().parents[1] / "vlm_work_order_gen.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--result-json",
            str(res_file),
            "--output",
            str(out_file),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert proc.returncode == 0
        assert out_file.exists()
        saved = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(saved["work_orders"]) == 1
        assert saved["road_segment_id"] == "seg_cli_test"
