from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Frame:
    bgr: np.ndarray
    ts_ns: int
    source_ts_ms: float


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None

    def put(self, bgr: np.ndarray, source_ts_ms: float = 0.0) -> None:
        with self._lock:
            self._frame = Frame(bgr=bgr, ts_ns=time.time_ns(), source_ts_ms=float(source_ts_ms))

    def get(self) -> Optional[Frame]:
        with self._lock:
            return self._frame
