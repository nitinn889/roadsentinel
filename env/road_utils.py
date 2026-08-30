"""
road_utils.py
-------------
Point (6) of the spec asks to be able to "control the road" from the
keyboard. Implemented here as: scan the loaded map for long, straight
stretches (the National-Highway-like sections), build a list of
candidate flight-start waypoints, and let the operator cycle through
them at runtime with N (next) / P (previous) - see drone_controller.py.

Straightness is judged by how little the road heading changes over a
lookahead distance; this is done programmatically against whatever map
is loaded rather than hardcoding road IDs, so it keeps working if you
switch CARLA_MAP in config.py (e.g. to the rural Town07).
"""

from typing import List
import config


def _heading_delta_deg(yaw_a: float, yaw_b: float) -> float:
    d = (yaw_b - yaw_a) % 360.0
    if d > 180.0:
        d -= 360.0
    return abs(d)


def find_straight_segments(carla_map) -> List["carla.Waypoint"]:
    """
    Returns a list of waypoints, each the start of a straight run at least
    ROAD_SEGMENT_LOOKAHEAD_M long. Falls back to all driving-lane
    waypoints if nothing qualifies (e.g. an unfamiliar/very curvy map).
    """
    all_wps = carla_map.generate_waypoints(config.ROAD_WAYPOINT_SPACING_M)
    driving_wps = [wp for wp in all_wps if wp.lane_type.name == "Driving"]

    straight_starts = []
    step = config.ROAD_WAYPOINT_SPACING_M
    lookahead_steps = max(1, int(config.ROAD_SEGMENT_LOOKAHEAD_M / step))

    for wp in driving_wps:
        cursor = wp
        ok = True
        start_yaw = wp.transform.rotation.yaw
        for _ in range(lookahead_steps):
            nxts = cursor.next(step)
            if not nxts:
                ok = False
                break
            cursor = nxts[0]
            if _heading_delta_deg(start_yaw, cursor.transform.rotation.yaw) > config.ROAD_SEGMENT_MAX_HEADING_CHANGE_DEG:
                ok = False
                break
        if ok:
            straight_starts.append(wp)

    if not straight_starts:
        return driving_wps[:20] if driving_wps else all_wps[:20]

    # Thin out near-duplicate points (every ~150m instead of every waypoint)
    thinned = straight_starts[::max(1, lookahead_steps)]
    return thinned if thinned else straight_starts
