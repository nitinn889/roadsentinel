"""Unit tests for defect severity model."""

import pytest
from analytics.severity import calculate_defect_severity, classify_severity_label
from common.schemas import SeverityResult


def test_severity_classification_labels():
    assert classify_severity_label(0.90) == "critical"
    assert classify_severity_label(0.70) == "high"
    assert classify_severity_label(0.50) == "medium"
    assert classify_severity_label(0.20) == "low"


def test_severity_no_depth_no_area():
    """Missing depth or area should dynamically rebalance available weights."""
    res = calculate_defect_severity(
        confidence=0.8,
        area_m2=None,
        depth_m=None,
        is_water_filled=False,
    )
    assert isinstance(res, SeverityResult)
    assert 0.0 <= res.severity_score <= 1.0
    assert res.severity_components["area"] is None
    assert res.severity_components["depth"] is None
    assert res.severity_components["confidence"] == 0.8


def test_severity_water_filled_boost():
    """Water-filled pothole should have heightened hazard severity."""
    dry_res = calculate_defect_severity(
        confidence=0.8,
        area_m2=0.5,
        depth_m=0.05,
        is_water_filled=False,
    )
    wet_res = calculate_defect_severity(
        confidence=0.8,
        area_m2=0.5,
        depth_m=0.05,
        is_water_filled=True,
        water_confidence=0.9,
    )
    assert wet_res.severity_score > dry_res.severity_score
    assert wet_res.severity_components["water"] == 1.0


def test_severity_critical_large_deep():
    """Very large, deep pothole should score critical."""
    res = calculate_defect_severity(
        confidence=0.95,
        area_m2=2.5,
        depth_m=0.20,
        is_water_filled=True,
        surrounding_damage=0.8,
    )
    assert res.severity == "critical"
    assert res.severity_score >= 0.85
