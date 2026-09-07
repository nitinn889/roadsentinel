"""Regression coverage for hierarchical water/marking classification."""

import numpy as np

from inference.hierarchical_classifier import HierarchicalClassifier


def test_water_sheen_detection_does_not_reference_undefined_marking_state():
    rgb = np.full((80, 80, 3), 95, dtype=np.uint8)
    # Smooth, dark and blue-tinted candidate: a water-sheen candidate.
    rgb[25:55, 25:55] = (35, 40, 52)
    mask = np.zeros((80, 80), dtype=bool)
    mask[25:55, 25:55] = True

    is_water, confidence = HierarchicalClassifier._detect_water_sheen(rgb, mask)

    assert isinstance(is_water, bool)
    assert 0.0 <= confidence <= 1.0
