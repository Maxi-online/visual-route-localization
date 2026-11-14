from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ControlPoint:
    time_sec: float
    lat: float
    lng: float


LINE_RE = re.compile(
    r"^\s*(?P<mm>\d{1,2}):(?P<ss>\d{2})\s*-\s*(?P<lat>[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<lng>[+-]?\d+(?:\.\d+)?)\s*$"
)


def parse_points_txt(path: Path) -> List[ControlPoint]:
    text = path.read_text(encoding="utf-8")
    points: List[ControlPoint] = []
    last_time_sec: float | None = None
    for i, raw in enumerate(text.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        m = LINE_RE.match(raw)
        if not m:
            raise ValueError(f"I didn't recognize the line.{i}: {raw!r}")
        mm = int(m.group("mm"))
        ss = int(m.group("ss"))
        t = float(mm * 60 + ss)
        if last_time_sec is not None:
            if t < last_time_sec:
                raise ValueError('The time at points.txt must go unforgivable.')
            if abs(t - last_time_sec) <= 1e-9:
                continue
        points.append(ControlPoint(time_sec=t, lat=float(m.group("lat")), lng=float(m.group("lng"))))
        last_time_sec = t
    if len(points) < 2:
        raise ValueError('We need at least two control points.')
    return points


def interpolate_track(frame_times_sec: np.ndarray, points: List[ControlPoint]) -> Tuple[np.ndarray, np.ndarray]:
    t_cp = np.array([p.time_sec for p in points], dtype=np.float64)
    lat_cp = np.array([p.lat for p in points], dtype=np.float64)
    lng_cp = np.array([p.lng for p in points], dtype=np.float64)
    t = np.clip(frame_times_sec.astype(np.float64), float(t_cp[0]), float(t_cp[-1]))
    lat = np.interp(t, t_cp, lat_cp).astype(np.float64)
    lng = np.interp(t, t_cp, lng_cp).astype(np.float64)
    return lat, lng
