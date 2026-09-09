"""Vehicle Movement Simulation and 400m Look-Ahead Health Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from config import (
    DEFAULT_SPEED_KMH,
    LOOKAHEAD_DISTANCE_M,
    MAX_SPEED_KMH,
    MIN_SPEED_KMH,
    MISSING_DATA_SEGMENTS,
    SEGMENT_BOUNDARIES,
    SEGMENT_LENGTH_M,
    TOTAL_ROAD_LENGTH_M,
)
from data_loader import load_primary_results_data
from route import get_segment_at_distance


def compute_road_health_score(severity: float) -> float:
    """Convert model-derived severity [0, 1] to user-facing Road Health Score [0, 100].

    Formula: health_score = 100 * (1 - severity)
    Example: severity 0.35 -> health_score = 65 / 100.
    """
    sev = max(0.0, min(1.0, float(severity)))
    return round(100.0 * (1.0 - sev), 1)


def compute_400m_lookahead_health(
    vehicle_dist_m: float, day: int, primary_data: Dict[int, Dict[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Calculate MODEL-DERIVED ROAD HEALTH for the 400m window ahead of the vehicle.

    If look-ahead is completely inside one segment, uses that segment severity.
    If look-ahead crosses segment boundaries, calculates a distance-weighted average.
    For missing segments (e.g. SEG_005/006), marks data_pending = True.
    """
    v_start = max(0.0, min(TOTAL_ROAD_LENGTH_M, vehicle_dist_m))
    v_end = min(TOTAL_ROAD_LENGTH_M, v_start + LOOKAHEAD_DISTANCE_M)
    window_length = v_end - v_start

    if window_length <= 1e-6:
        # At the very end of the road
        last_seg = get_segment_at_distance(TOTAL_ROAD_LENGTH_M)
        seg_info = primary_data.get(day, {}).get(last_seg, {})
        if not seg_info.get("has_data") or last_seg in MISSING_DATA_SEGMENTS:
            return {"health_score": None, "weighted_severity": None, "data_pending": True, "status": "DATA PENDING"}
        sev = seg_info.get("current_severity", 0.0)
        return {
            "health_score": compute_road_health_score(sev),
            "weighted_severity": round(sev, 4),
            "data_pending": False,
            "status": "VALID",
        }

    # Find segment overlap lengths in interval [v_start, v_end]
    total_weighted_severity = 0.0
    valid_length = 0.0
    has_missing_segment = False

    for seg_id, (s_start, s_end) in SEGMENT_BOUNDARIES.items():
        overlap_start = max(v_start, s_start)
        overlap_end = min(v_end, s_end)
        overlap_len = max(0.0, overlap_end - overlap_start)

        if overlap_len > 0:
            seg_info = primary_data.get(day, {}).get(seg_id, {})
            if not seg_info.get("has_data") or seg_id in MISSING_DATA_SEGMENTS:
                has_missing_segment = True
            else:
                sev = seg_info.get("current_severity", 0.0)
                total_weighted_severity += overlap_len * sev
                valid_length += overlap_len

    if valid_length <= 1e-6:
        return {
            "health_score": None,
            "weighted_severity": None,
            "data_pending": True,
            "status": "ROAD HEALTH DATA PENDING",
        }

    # Compute weighted severity over valid window portion
    weighted_sev = total_weighted_severity / valid_length
    health = compute_road_health_score(weighted_sev)

    status_str = "VALID"
    if has_missing_segment:
        status_str = "PARTIAL (DATA PENDING FOR AHEAD)"

    return {
        "health_score": health,
        "weighted_severity": round(weighted_sev, 4),
        "data_pending": has_missing_segment and valid_length < window_length,
        "status": status_str,
        "valid_coverage_m": round(valid_length, 1),
    }


class SimulationState:
    """Class representing the vehicle simulation state."""

    def __init__(self, day: int = 5, speed_kmh: float = DEFAULT_SPEED_KMH) -> None:
        self.day = day
        self.speed_kmh = max(MIN_SPEED_KMH, min(MAX_SPEED_KMH, float(speed_kmh)))
        self.distance_m = 0.0
        self.is_running = False
        self.elapsed_sec = 0.0
        self.fired_alerts: set[tuple[str, str]] = set()
        self.alert_history: list[dict[str, Any]] = []

    def start(self) -> None:
        self.is_running = True

    def pause(self) -> None:
        self.is_running = False

    def reset(self) -> None:
        self.distance_m = 0.0
        self.is_running = False
        self.elapsed_sec = 0.0
        self.fired_alerts.clear()
        self.alert_history.clear()

    def set_speed(self, speed_kmh: float) -> None:
        self.speed_kmh = max(MIN_SPEED_KMH, min(MAX_SPEED_KMH, float(speed_kmh)))

    def set_day(self, day: int) -> None:
        if 1 <= day <= 10 and day != self.day:
            self.day = day
            self.fired_alerts.clear()

    def update(self, dt_seconds: float) -> None:
        if not self.is_running or dt_seconds <= 0:
            return
        speed_mps = self.speed_kmh / 3.6
        dist_inc = speed_mps * dt_seconds
        self.distance_m = min(TOTAL_ROAD_LENGTH_M, self.distance_m + dist_inc)
        self.elapsed_sec += dt_seconds
        if self.distance_m >= TOTAL_ROAD_LENGTH_M:
            self.is_running = False
