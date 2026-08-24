from __future__ import annotations

from dataclasses import asdict
from common.schemas import Telemetry
from common.io_utils import utc_iso


def telemetry_from_carla(snapshot, transform, speed_mps: float = 0.0, frame_id: int | None = None) -> Telemetry:
    loc = transform.location
    rot = transform.rotation
    return Telemetry(
        timestamp=utc_iso(),
        latitude=None,
        longitude=None,
        altitude_m=float(loc.z),
        heading_deg=float(rot.yaw),
        frame_id=frame_id,
        world_x=float(loc.x),
        world_y=float(loc.y),
        world_z=float(loc.z),
        speed_mps=float(speed_mps),
    )
