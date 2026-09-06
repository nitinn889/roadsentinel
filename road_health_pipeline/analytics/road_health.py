"""Road Health Scoring (0–100 Index) for RoadSentinel.

Computes a transparent, reproducible, and explainable 0–100 road segment health score
where 100 is pristine/healthy road and 0 is severely hazardous/failed pavement.
Penalties are explicitly broken down across pothole count/severity, cracking,
water pooling hazards, and overall surface roughness.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union
import numpy as np

from config import CONFIG
from common.schemas import PotholeRecord, RoadHealthResult


def classify_road_condition(score: float) -> str:
    """Classify 0-100 score into standard condition rating."""
    if score >= CONFIG.health_good_threshold:
        return "good"
    if score >= CONFIG.health_fair_threshold:
        return "fair"
    if score >= CONFIG.health_poor_threshold:
        return "poor"
    return "critical"


def calculate_road_health_score(
    potholes: Sequence[Union[PotholeRecord, dict]],
    total_crack_area_m2: float = 0.0,
    surface_anomaly_mean: float = 0.0,
    road_area_m2: Optional[float] = None,
) -> RoadHealthResult:
    """Calculate the 0–100 road health index for a road segment or frame.

    Parameters
    ----------
    potholes:
        Sequence of detected pothole records or dicts.
    total_crack_area_m2:
        Total surface area in m^2 classified as cracking or surface damage.
    surface_anomaly_mean:
        Mean background anomaly score across road surface [0, 1].
    road_area_m2:
        Optional estimated total road segment area in m^2.

    Returns
    -------
    RoadHealthResult
        Contains road_health_score in [0, 100], condition_class, and per-penalty breakdown.
    """
    n_potholes = len(potholes)
    
    # Extract severities and water flags
    severities: list[float] = []
    water_flags: list[bool] = []
    total_pothole_area_m2 = 0.0

    for p in potholes:
        if isinstance(p, dict):
            sev = float(p.get("severity_score", 0.5))
            water = bool(p.get("water_flag", False))
            area = p.get("area_m2")
        else:
            sev = float(p.severity_score)
            water = bool(p.water_flag)
            area = p.area_m2
        
        severities.append(sev)
        water_flags.append(water)
        if area is not None and np.isfinite(area):
            total_pothole_area_m2 += max(0.0, float(area))

    # 1. Pothole Penalty (Max 25 points default)
    # Scaled by count relative to max cap (e.g. 5 potholes)
    count_factor = min(1.0, n_potholes / max(1, CONFIG.health_max_pothole_penalty_count))
    pothole_penalty = float(CONFIG.health_pothole_weight * count_factor)

    # 2. Pothole Severity Penalty (Max 30 points default)
    # Weighted by maximum and mean severity of present potholes
    if n_potholes > 0:
        max_sev = max(severities)
        avg_sev = float(np.mean(severities))
        sev_factor = 0.65 * max_sev + 0.35 * avg_sev
        severity_penalty = float(CONFIG.health_severity_weight * np.clip(sev_factor, 0.0, 1.0))
    else:
        severity_penalty = 0.0

    # 3. Crack / Surface Damage Penalty (Max 20 points default)
    # Scaled by damage area in m^2 or crack extent
    crack_scale_m2 = 5.0 if road_area_m2 is None else max(1.0, road_area_m2 * 0.10)
    crack_factor = min(1.0, total_crack_area_m2 / crack_scale_m2)
    crack_penalty = float(CONFIG.health_crack_weight * crack_factor)

    # 4. Water Pooling Hazard Penalty (Max 15 points default)
    # Each water-filled defect carries heightened hydroplaning and concealed hazard risk
    water_count = sum(1 for w in water_flags if w)
    if water_count > 0:
        water_factor = min(1.0, water_count / 2.0)
        water_penalty = float(CONFIG.health_water_weight * water_factor)
    else:
        water_penalty = 0.0

    # 5. Background Surface Roughness Penalty (Max 10 points default)
    surface_factor = float(np.clip(surface_anomaly_mean * 2.0, 0.0, 1.0))
    surface_penalty = float(CONFIG.health_surface_weight * surface_factor)

    # Compute net score
    total_penalty = pothole_penalty + severity_penalty + crack_penalty + water_penalty + surface_penalty
    raw_score = CONFIG.health_base_score - total_penalty
    final_score = float(np.clip(raw_score, 0.0, 100.0))
    condition = classify_road_condition(final_score)

    components = {
        "pothole_count_penalty": round(pothole_penalty, 2),
        "pothole_severity_penalty": round(severity_penalty, 2),
        "crack_penalty": round(crack_penalty, 2),
        "water_hazard_penalty": round(water_penalty, 2),
        "surface_roughness_penalty": round(surface_penalty, 2),
        "total_penalty": round(total_penalty, 2),
    }

    return RoadHealthResult(
        road_health_score=round(final_score, 2),
        condition_class=condition,
        components=components,
    )
