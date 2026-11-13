from __future__ import annotations

import math
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_points(path: pathlib.Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(
            r"^(?P<mm>\d+):(?P<ss>\d+)\s*-\s*(?P<lat>[-+]?\d+(?:\.\d+)?),\s*(?P<lng>[-+]?\d+(?:\.\d+)?)$",
            line,
        )
        if match is None:
            continue
        mm = int(match.group("mm"))
        ss = int(match.group("ss"))
        lat = float(match.group("lat"))
        lng = float(match.group("lng"))
        rows.append((mm * 60 + ss, lat, lng))
    return rows


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


ref_01 = _parse_points(ROOT / "inputs" / "points" / "ref_01.txt")
ref_02 = _parse_points(ROOT / "inputs" / "points" / "ref_02.txt")

assert len(ref_01) >= 10, "Слишком мало контрольных точек в ref_01.txt"
assert len(ref_02) >= 10, "Слишком мало контрольных точек в ref_02.txt"

# Для каждой точки маршрута 02 ищем ближайшую точку маршрута 01.
min_dists = []
for _, lat2, lng2 in ref_02:
    best = min(_haversine_m(lat2, lng2, lat1, lng1) for _, lat1, lng1 in ref_01)
    min_dists.append(best)

min_dists_sorted = sorted(min_dists)
median_min_dist = min_dists_sorted[len(min_dists_sorted) // 2]
mean_min_dist = sum(min_dists) / len(min_dists)
max_min_dist = max(min_dists)

# Базовая проверка: 02 не должен совпадать с 01, иначе смещение в runtime
# невозможно валидировать как отдельную траекторию.
assert median_min_dist >= 80.0, (
    "Маршрут ref_02 слишком близок к ref_01 (median min dist < 80 м), "
    "невозможно честно валидировать боковое смещение."
)
assert mean_min_dist >= 120.0, (
    "Среднее минимальное расстояние между ref_02 и ref_01 слишком мало; "
    "ожидается заметно отличающаяся траектория."
)
assert max_min_dist >= 300.0, (
    "Маршруты отличаются недостаточно по длине/геометрии для проверок "
    "смещения на дальних участках."
)

print(
    "PASS route offset baseline: "
    f"mean={mean_min_dist:.2f}m median={median_min_dist:.2f}m max={max_min_dist:.2f}m"
)
