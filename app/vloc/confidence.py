from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceEstimator:
    def compute_visual(
        self,
        sim: float,
        inlier_ratio: float,
        reproj_rmse: float,
        matches_cnt: int,
    ) -> float:
        _ = sim
        _ = reproj_rmse
        _ = matches_cnt
        return 100.0 * max(0.0, min(1.0, float(inlier_ratio)))
