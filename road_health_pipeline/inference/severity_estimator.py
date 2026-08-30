from __future__ import annotations

from typing import Optional
import numpy as np

from common.schemas import SeverityBreakdown
from config import CONFIG, SeverityWeights


class SeverityEstimator:
    """Computes transparent, explainable severity scores for individual road defects.

    Features:
    - Area magnitude (m^2 or relative pixel fraction)
    - Estimated depth (metres) — gracefully re-weights if depth is unavailable
    - Water hazard status & confidence
    - Surrounding cracking / damage extent
    - Confidence normalization
    """

    def __init__(self, weights: Optional[SeverityWeights] = None):
        self.cfg = weights or CONFIG.severity

    def compute_severity(self,
                         area_m2: Optional[float],
                         depth_m: Optional[float],
                         is_water_filled: bool,
                         water_confidence: float = 0.0,
                         crack_or_damage_extent: Optional[float] = None,
                         confidence: float = 1.0,
                         mask_area_px: Optional[int] = None) -> SeverityBreakdown:
        """Calculates defect severity score (0-100) and returns detailed breakdown.

        When depth_m is None (RGB-only without metric depth sensor/model),
        the available weights are dynamically normalized to maintain scientific honesty.
        """
        # 1. Area Component (0-100)
        s_area: Optional[float] = None
        if area_m2 is not None and area_m2 > 0:
            # Scaled linearly up to area_high_m2
            s_area = float(np.clip((area_m2 / max(0.01, self.cfg.area_high_m2)) * 100.0, 0.0, 100.0))
        elif mask_area_px is not None and mask_area_px > 0:
            # Fallback to pixel area scaling if physical altitude is unknown
            # 50,000 px on a 1280x720 frame is ~5.4% of screen
            s_area = float(np.clip((mask_area_px / 30_000.0) * 100.0, 0.0, 100.0))

        # 2. Depth Component (0-100)
        s_depth: Optional[float] = None
        if depth_m is not None and depth_m > 0 and np.isfinite(depth_m):
            s_depth = float(np.clip((depth_m / max(0.005, self.cfg.depth_high_m)) * 100.0, 0.0, 100.0))

        # 3. Water Hazard Component (0-100)
        s_water: Optional[float] = 0.0
        if is_water_filled:
            s_water = float(np.clip(water_confidence * 100.0 + self.cfg.water_hazard_bonus, 0.0, 100.0))

        # 4. Surrounding Damage Component (0-100)
        s_damage: Optional[float] = 0.0
        if crack_or_damage_extent is not None and crack_or_damage_extent > 0:
            s_damage = float(np.clip((crack_or_damage_extent / 1.0) * 100.0, 0.0, 100.0))

        # Weighted combination over available components
        active_weights = []
        active_scores = []

        if s_area is not None:
            active_weights.append(self.cfg.weight_area)
            active_scores.append(s_area)

        if s_depth is not None:
            active_weights.append(self.cfg.weight_depth)
            active_scores.append(s_depth)

        if s_water is not None:
            active_weights.append(self.cfg.weight_water)
            active_scores.append(s_water)

        if s_damage is not None:
            active_weights.append(self.cfg.weight_surrounding_damage)
            active_scores.append(s_damage)

        if not active_weights:
            total_score = 0.0
        else:
            w_sum = sum(active_weights)
            weighted_score = sum(w * s for w, s in zip(active_weights, active_scores)) / w_sum
            # Scale slightly by detection confidence
            conf_clamped = float(np.clip(confidence, 0.2, 1.0))
            total_score = float(np.clip(weighted_score * (0.5 + 0.5 * conf_clamped), 0.0, 100.0))

        # Determine qualitative severity class
        if total_score >= self.cfg.threshold_high:
            severity_class = "critical"
        elif total_score >= self.cfg.threshold_medium:
            severity_class = "high"
        elif total_score >= self.cfg.threshold_low:
            severity_class = "medium"
        else:
            severity_class = "low"

        return SeverityBreakdown(
            severity=severity_class,
            severity_score=round(total_score, 2),
            severity_components={
                "area": round(s_area, 2) if s_area is not None else None,
                "depth": round(s_depth, 2) if s_depth is not None else None,
                "water": round(s_water, 2) if s_water is not None else None,
                "surrounding_damage": round(s_damage, 2) if s_damage is not None else None,
            }
        )
