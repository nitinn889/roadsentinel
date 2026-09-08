"""Observed segment changes and conservative defect-track events."""

from __future__ import annotations

from typing import Any

if __package__:
    from .defect_matching import MatchingConfig, match_adjacent
    from .load_sequence import DailyRecord, missing_days
else:
    from defect_matching import MatchingConfig, match_adjacent
    from load_sequence import DailyRecord, missing_days

MEASURES = ("current_severity", "crack_area_ratio", "defect_area_ratio", "defect_count", "surface_anomaly_score")


def _delta(current: Any, previous: Any) -> float | None:
    return None if current is None or previous is None else float(current) - float(previous)


def build_daily_summary(records: list[DailyRecord]) -> list[dict[str, Any]]:
    first = records[0].record if records else {}
    previous: dict[str, Any] | None = None
    rows = []
    for item in records:
        record = item.record
        row = {"segment_id": record["segment_id"], "day": record["day"], "original_filename": record["original_filename"], "condition": record["condition"], "water_flag": record["water_flag"]}
        for name in MEASURES:
            row[name] = record[name]
            row[f"{name}_delta"] = _delta(record[name], previous[name]) if previous else None
            row[f"{name}_change_from_day1"] = _delta(record[name], first[name])
        rows.append(row); previous = record
    return rows


def build_tracks(records: list[DailyRecord], config: MatchingConfig = MatchingConfig()) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tracks: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    previous_defects: list[dict[str, Any]] = []
    previous_track_ids: list[str] = []
    next_id = 1
    for item in records:
        day, defects = item.day, item.record["defects"]
        matches = match_adjacent(previous_defects, defects, config)
        by_current = {match["current_index"]: match for match in matches}
        matched_previous = {match["previous_index"] for match in matches}
        current_track_ids: list[str] = []
        for index, defect in enumerate(defects):
            match = by_current.get(index)
            if match:
                track_id = previous_track_ids[match["previous_index"]]
                event_type = "MATCHED_EXISTING"
                old = previous_defects[match["previous_index"]]
                area_delta = _delta(defect.get("area_ratio"), old.get("area_ratio"))
                if area_delta is not None and area_delta > 0:
                    event_type = "OBSERVED_AREA_INCREASED"
                elif area_delta is not None and area_delta < 0:
                    event_type = "OBSERVED_AREA_DECREASED"
                events.append({"day": day, "track_id": track_id, "event": event_type, "matching_method": match["method"], "matching_confidence": match["confidence"], "area_delta": area_delta})
            else:
                track_id = f"DEFECT_{next_id:03d}"; next_id += 1
                tracks[track_id] = {"track_id": track_id, "first_seen_day": day, "last_seen_day": day, "days_seen": [], "bbox_history": [], "centroid_history": [], "area_history": [], "type_history": [], "matching_history": []}
                events.append({"day": day, "track_id": track_id, "event": "NEW_DEFECT", "matching_method": None, "matching_confidence": None, "area_delta": None})
            track = tracks[track_id]; track["last_seen_day"] = day; track["days_seen"].append(day)
            track["bbox_history"].append({"day": day, "bbox": defect.get("bbox")})
            track["centroid_history"].append({"day": day, "centroid_x": defect.get("centroid_x"), "centroid_y": defect.get("centroid_y")})
            track["area_history"].append({"day": day, "area_ratio": defect.get("area_ratio")})
            track["type_history"].append({"day": day, "defect_type": defect.get("defect_type")})
            if match: track["matching_history"].append({"day": day, "method": match["method"], "confidence": match["confidence"]})
            current_track_ids.append(track_id)
        for old_index, track_id in enumerate(previous_track_ids):
            if old_index not in matched_previous:
                events.append({"day": day, "track_id": track_id, "event": "NOT_OBSERVED", "matching_method": None, "matching_confidence": None, "area_delta": None})
        previous_defects, previous_track_ids = defects, current_track_ids
    return tracks, events


def progression_summary(records: list[DailyRecord], tracks: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = records[0].record, records[-1].record
    def change(name: str) -> float | None: return _delta(last[name], first[name])
    water_days = [item.day for item in records if item.record["water_flag"] is True]
    return {"segment_id": first["segment_id"], "first_day": first["day"], "last_day": last["day"], "days_available": [item.day for item in records], "missing_days": missing_days(records), "severity_start": first["current_severity"], "severity_end": last["current_severity"], "severity_delta": change("current_severity"), "max_severity": max((float(x.record["current_severity"]) for x in records if x.record["current_severity"] is not None), default=None), "defect_area_start": first["defect_area_ratio"], "defect_area_end": last["defect_area_ratio"], "defect_area_delta": change("defect_area_ratio"), "crack_area_start": first["crack_area_ratio"], "crack_area_end": last["crack_area_ratio"], "crack_area_delta": change("crack_area_ratio"), "defect_count_start": first["defect_count"], "defect_count_end": last["defect_count"], "water_first_seen_day": water_days[0] if water_days else None, "num_tracks": len(tracks), "num_new_defects": sum(event["event"] == "NEW_DEFECT" for event in events), "summary_status": "OBSERVED TEMPORAL CHANGE"}
