"""Defect severity estimation model for RoadSentinel.

Calculates an explainable, multi-factor severity score in [0.0, 1.0] and qualitative
rating based on physical measurements (area, depth, shape, water presence, surrounding damage).
All component weights and normalization constants are configured in `config.py`.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from config import CONFIG
from common.schemas import SeverityBreakdown, SeverityResult


def classify_severity_label(score: float) -> str:
    """Map continuous severity score [0, 1] to qualitative classification."""
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def calculate_defect_severity(
    confidence: float,
    area_m2: Optional[float] = None,
    depth_m: Optional[float] = None,
    is_water_filled: bool = False,
    water_confidence: float = 0.0,
    surrounding_damage: float = 0.0,
    shape_circularity: Optional[float] = None,
    weight_conf: Optional[float] = None,
    weight_area: Optional[float] = None,
    weight_depth: Optional[float] = None,
    weight_water: Optional[float] = None,
    weight_damage: Optional[float] = None,
) -> SeverityResult:
    """Compute transparent multi-factor defect severity.

    Parameters
    ----------
    confidence:
        Detection confidence [0, 1].
    area_m2:
        Estimated surface area in m^2 (if available).
    depth_m:
        Estimated depth in metres (if available).
    is_water_filled:
        Whether the defect contains pooled water.
    water_confidence:
        Confidence score of water pooling heuristic [0, 1].
    surrounding_damage:
        Severity indicator of surrounding cracks or surface fatigue [0, 1].
    shape_circularity:
        Circularity of defect mask [0, 1] (1.0 = perfect circle).
    weight_*:
        Optional overrides for config weights.

    Returns
    -------
    SeverityResult
        Contains severity label ("low"|"medium"|"high"|"critical"),
        continuous score in [0, 1], and per-factor component breakdown.
    """
    w_conf = weight_conf if weight_conf is not None else CONFIG.severity_weight_confidence
    w_area = weight_area if weight_area is not None else CONFIG.severity_weight_area
    w_depth = weight_depth if weight_depth is not None else CONFIG.severity_weight_depth
    w_water = weight_water if weight_water is not None else CONFIG.severity_weight_water
    w_damage = weight_damage if weight_damage is not None else CONFIG.severity_weight_damage

    # Normalize individual components to [0, 1]
    c_conf = float(np.clip(confidence, 0.0, 1.0))
    
    # Area component: scaled relative to configurable nominal large pothole area (e.g. 2.0 m^2)
    c_area: Optional[float] = None
    if area_m2 is not None and np.isfinite(area_m2):
        c_area = float(np.clip(max(0.0, area_m2) / max(0.01, CONFIG.severity_area_norm_m2), 0.0, 1.0))

    # Depth component: scaled relative to nominal severe depth (e.g. 0.15 m = 15 cm)
    c_depth: Optional[float] = None
    if depth_m is not None and np.isfinite(depth_m):
        c_depth = float(np.clip(max(0.0, depth_m) / max(0.01, CONFIG.severity_depth_norm_m), 0.0, 1.0))

    # Water component: pooled water hides depth and creates severe hydroplaning hazard
    c_water = float(np.clip(max(float(is_water_filled), water_confidence), 0.0, 1.0))

    # Surrounding damage / fatigue cracking
    c_damage = float(np.clip(surrounding_damage, 0.0, 1.0))

    # Calculate weighted continuous score
    # If depth or area is missing, rebalance available weights dynamically so score remains in [0, 1]
    active_weights = []
    active_values = []

    # Confidence is always active
    active_weights.append(w_conf)
    active_values.append(c_conf)

    if c_area is not None:
        active_weights.append(w_area)
        active_values.append(c_area)
    
    if c_depth is not None:
        active_weights.append(w_depth)
        active_values.append(c_depth)

    active_weights.append(w_water)
    active_values.append(c_water)

    active_weights.append(w_damage)
    active_values.append(c_damage)

    total_weight = sum(active_weights)
    if total_weight > 0:
        raw_score = sum(w * v for w, v in zip(active_weights, active_values)) / total_weight
    else:
        raw_score = c_conf

    # Additional hazard boost for confirmed water-filled defects with high confidence
    if is_water_filled and raw_score < 0.90:
        raw_score = min(1.0, raw_score + 0.10)

    severity_score = float(np.clip(raw_score, 0.0, 1.0))
    label = classify_severity_label(severity_score)

    breakdown = SeverityBreakdown(
        area=round(c_area, 4) if c_area is not None else None,
        depth=round(c_depth, 4) if c_depth is not None else None,
        shape=round(float(shape_circularity), 4) if shape_circularity is not None else None,
        water=round(c_water, 4),
        surrounding_damage=round(c_damage, 4),
        confidence=round(c_conf, 4),
    )

    return SeverityResult(
        severity=label,
        severity_score=round(severity_score, 4),
        severity_components=breakdown.to_dict(),
    )
