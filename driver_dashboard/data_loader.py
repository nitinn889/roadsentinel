"""Data Loader for RoadSentinel Driver Dashboard.

Ingests existing canonical CSVs and temporal segment JSONs without altering
scientific results or fabricating missing segment inspections.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import MISSING_DATA_SEGMENTS, SEGMENT_NAMES, WORKSPACE_ROOT

PRIMARY_RESULTS_CSV = WORKSPACE_ROOT / "integration/primary_goals/ROADSENTINEL_PRIMARY_RESULTS.csv"
DECISIONS_CSV = WORKSPACE_ROOT / "decision_engine/ROAD_HEALTH_DECISIONS.csv"
TEMPORAL_SEGMENTS_DIR = WORKSPACE_ROOT / "env/output/temporal_segments"


def load_primary_results_data() -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Load primary results per day and segment.

    Structure:
        data[day][segment_id] = {
            "current_severity": float,
            "defect_count": int,
            "defect_area_ratio": float,
            "surface_anomaly_score": float,
            "water_flag": bool,
            "road_health_state": str,
            "has_data": bool
        }
    """
    data: Dict[int, Dict[str, Dict[str, Any]]] = {
        day: {seg: {"has_data": False, "severity": None, "defect_count": 0} for seg in SEGMENT_NAMES}
        for day in range(1, 11)
    }

    if PRIMARY_RESULTS_CSV.exists():
        with open(PRIMARY_RESULTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    seg_id = row.get("segment_id", "").strip()
                    day = int(row.get("day", 0))
                    if not (1 <= day <= 10) or seg_id not in SEGMENT_NAMES:
                        continue

                    # If status is MISSING, leave has_data as False
                    goal1_status = row.get("goal1_status", "").upper()
                    if goal1_status == "MISSING" or seg_id in MISSING_DATA_SEGMENTS:
                        continue

                    sev = float(row.get("current_severity", 0.0))
                    def_cnt = int(float(row.get("defect_count", 0)))
                    def_area = float(row.get("defect_area_ratio", 0.0))
                    anomaly = float(row.get("surface_anomaly_score", 0.0))
                    water = row.get("water_flag", "False").lower() == "true"
                    health_state = row.get("road_health_state", "Normal")

                    data[day][seg_id] = {
                        "segment_id": seg_id,
                        "day": day,
                        "current_severity": round(sev, 4),
                        "defect_count": def_cnt,
                        "defect_area_ratio": def_area,
                        "surface_anomaly_score": anomaly,
                        "water_flag": water,
                        "road_health_state": health_state,
                        "has_data": True,
                    }
                except Exception:
                    continue

    return data


def load_temporal_defect_history(segment_id: str) -> Dict[int, Dict[str, Any]]:
    """Load defect history from env/output/temporal_segments/SEG_XXX/segment_history.json."""
    history_file = TEMPORAL_SEGMENTS_DIR / segment_id / "segment_history.json"
    day_defects: Dict[int, Dict[str, Any]] = {}

    if history_file.exists():
        try:
            content = json.loads(history_file.read_text(encoding="utf-8"))
            for entry in content.get("history", []):
                day = entry.get("day")
                if isinstance(day, int) and 1 <= day <= 10:
                    day_defects[day] = entry
        except Exception:
            pass

    return day_defects


def get_segment_day_info(
    segment_id: str,
    day: int,
    primary_data: Optional[Dict[int, Dict[str, Any]]] = None,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """Get segment status and metrics for a specific day."""
    if segment_id in MISSING_DATA_SEGMENTS:
        return {
            "segment_id": segment_id,
            "day": day,
            "has_data": False,
            "data_pending": True,
            "current_severity": None,
            "defect_count": 0,
            "status_text": "DATA UNAVAILABLE / PENDING INSPECTION",
        }

    all_data = primary_data if primary_data is not None else load_primary_results_data()
    seg_info = all_data.get(day, {}).get(segment_id, {"has_data": False})
    if not seg_info.get("has_data"):
        return {
            "segment_id": segment_id,
            "day": day,
            "has_data": False,
            "data_pending": True,
            "current_severity": None,
            "defect_count": 0,
            "status_text": "DATA UNAVAILABLE / PENDING INSPECTION",
        }

    res = dict(seg_info)
    res["data_pending"] = False
    res["status_text"] = "INSPECTED"
    return res
