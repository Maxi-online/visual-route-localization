from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EwmaSmoother:
    base_alpha: float = 0.25
    min_alpha: float = 0.05
    max_alpha: float = 0.60

    _lat: Optional[float] = None
    _lng: Optional[float] = None

    def reset(self) -> None:
        self._lat = None
        self._lng = None

    def update(self, lat: float, lng: float, confidence: float) -> Tuple[float, float]:
        conf = max(0.0, min(100.0, float(confidence))) / 100.0
        alpha = self.base_alpha * (0.3 + 0.7 * conf)
        alpha = max(self.min_alpha, min(self.max_alpha, alpha))
        if self._lat is None or self._lng is None:
            self._lat, self._lng = float(lat), float(lng)
            return self._lat, self._lng
        self._lat = alpha * float(lat) + (1.0 - alpha) * self._lat
        self._lng = alpha * float(lng) + (1.0 - alpha) * self._lng
        return self._lat, self._lng
