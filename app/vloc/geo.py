from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ENU:
    east_m: float
    north_m: float


def ll_to_enu(lat: float, lng: float, lat0: float, lng0: float) -> ENU:
    lat0_rad = math.radians(lat0)
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(lat0_rad)
    north = (lat - lat0) * m_per_deg_lat
    east = (lng - lng0) * m_per_deg_lng
    return ENU(east_m=float(east), north_m=float(north))


def enu_to_ll(enu: ENU, lat0: float, lng0: float) -> Tuple[float, float]:
    lat0_rad = math.radians(lat0)
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(lat0_rad)
    lat = lat0 + (enu.north_m / m_per_deg_lat)
    lng = lng0 + (enu.east_m / m_per_deg_lng)
    return float(lat), float(lng)
