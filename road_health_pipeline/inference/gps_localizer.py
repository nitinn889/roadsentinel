from __future__ import annotations

import math
from typing import Optional

from common.schemas import Telemetry
from config import CONFIG


class GPSLocalizer:
    def __init__(self, origin_lat: float = CONFIG.carla_origin_lat, origin_lon: float = CONFIG.carla_origin_lon):
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)

    def carla_world_to_gps(self, x_m: float, y_m: float) -> tuple[float, float]:
        meters_per_deg_lat = 111_320.0
        lat = self.origin_lat + y_m / meters_per_deg_lat
        lon_scale = meters_per_deg_lat * max(0.1, math.cos(math.radians(self.origin_lat)))
        lon = self.origin_lon + x_m / lon_scale
        return lat, lon

    def attach(self, telemetry: Telemetry) -> Telemetry:
        if telemetry.latitude is None or telemetry.longitude is None:
            if telemetry.world_x is not None and telemetry.world_y is not None:
                telemetry.latitude, telemetry.longitude = self.carla_world_to_gps(telemetry.world_x, telemetry.world_y)
        return telemetry


def telemetry_from_dict(d: dict) -> Telemetry:
    return Telemetry(
        timestamp=str(d.get("timestamp")),
        latitude=d.get("latitude"),
        longitude=d.get("longitude"),
        altitude_m=d.get("altitude_m"),
        heading_deg=d.get("heading_deg"),
        frame_id=d.get("frame_id"),
        world_x=d.get("world_x"),
        world_y=d.get("world_y"),
        world_z=d.get("world_z"),
        speed_mps=d.get("speed_mps"),
    )
