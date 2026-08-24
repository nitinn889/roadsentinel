from __future__ import annotations

import cv2
import numpy as np

from common.geometry import polygon_area_m2
from config import CONFIG


def estimate_area_m2(mask: np.ndarray, altitude_m: float | None,
                     horizontal_fov_deg: float = CONFIG.horizontal_fov_deg) -> float | None:
    if altitude_m is None or altitude_m <= 0:
        return None
    return polygon_area_m2(mask, float(altitude_m), horizontal_fov_deg)
