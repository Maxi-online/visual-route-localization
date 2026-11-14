from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import sqlite3
from math import cos, radians, sqrt

from app import db as db_mod
from app.vloc.confidence import ConfidenceEstimator
from app.vloc.geo import ENU, enu_to_ll, ll_to_enu
from app.vloc.matching import LightGlueMatcher
from app.vloc.retrieval import Retriever
from app.vloc.smoothing import EwmaSmoother
from app.vloc.vision import GlobalEmbedder

CachedLocalFeatures = Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]


def _center_xy(w: int, h: int) -> np.ndarray:
    return np.array([float(w) * 0.5, float(h) * 0.5], dtype=np.float64)


def _ground_anchor_xy(w: int, h: int) -> np.ndarray:
    'Base point in the lower part of the frame which is more sensitive to topography.'
    return np.array([float(w) * 0.5, float(h) * 0.72], dtype=np.float64)


def _ground_anchor_grid(w: int, h: int) -> np.ndarray:
    'Line-point grid at bottom frame for steady displacement.'
    xs = (0.25, 0.5, 0.75)
    ys = (0.72, 0.82, 0.90)
    pts = []
    for y in ys:
        for x in xs:
            pts.append([float(w) * float(x), float(h) * float(y)])
    return np.array(pts, dtype=np.float64)


def _map_point(H: np.ndarray, pt_xy: np.ndarray) -> np.ndarray:
    pt = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H.astype(np.float64)).reshape(-1, 2)
    return out[0].astype(np.float64)


@dataclass
class Localizer:
    """Localizes the runtime frame relative to the indexed route.

    The class combines retrieval, local features comparison, evaluation
    Conference, time lock and geographical smoothing. On the way out
    Only the coordinates confirmed by the full pypline are returned
    Localization.
    """

    conn: sqlite3.Connection
    retriever: Retriever
    embedder: GlobalEmbedder
    matcher: LightGlueMatcher
    confidence: ConfidenceEstimator
    smoother: EwmaSmoother
    top_k: int = 10

    min_inlier_ratio: float = 0.14
    min_matches_cnt: int = 12
    max_reproj_rmse: float = 10.0
    min_inlier_ratio_local: float = 0.10
    min_matches_cnt_local: int = 8
    max_reproj_rmse_local: float = 12.0
    ambiguity_conf_band: float = 5.0
    ambiguity_max_geo_m: float = 5_000.0
    temporal_tolerance_ms: float = 12_000.0
    temporal_scale: float = 8.0
    max_temporal_jump_ms: float = 120_000.0
    max_homography_offset_m: float = 600.0
    homography_anchor_min_valid: int = 2
    lateral_component_gain: float = 1.2
    along_component_gain: float = 0.15
    local_back_ms: float = 20_000.0
    local_forward_ms: float = 25_000.0
    local_limit: int = 300
    reacquire_after_misses: int = 2

    feature_cache_size: int = 128
    feature_torch_cache_size: int = 48

    _feature_cache: OrderedDict[int, CachedLocalFeatures] = field(init=False, repr=False)
    _feature_torch_cache: OrderedDict[int, Any] = field(init=False, repr=False)
    _meta_cache: dict[int, dict[str, Any]] = field(init=False, repr=False)
    _px_to_enu_cache: dict[int, Optional[dict[str, Any]]] = field(init=False, repr=False)
    _last_ref_ts_ms: Optional[float] = field(default=None, init=False, repr=False)
    _last_source_ts_ms: Optional[float] = field(default=None, init=False, repr=False)
    _last_video_id: Optional[int] = field(default=None, init=False, repr=False)
    _miss_streak: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        'Initiating runtime caches of metadata and reference data.'
        self._feature_cache = OrderedDict()
        self._feature_torch_cache = OrderedDict()
        self._meta_cache = {}
        self._px_to_enu_cache = {}
        self._last_ref_ts_ms = None
        self._last_source_ts_ms = None
        self._last_video_id = None
        self._miss_streak = 0

    def reset_lock(self) -> None:
        'Maintains API compatibility and drops the smoother state.'
        self.smoother.reset()
        self._last_ref_ts_ms = None
        self._last_source_ts_ms = None
        self._last_video_id = None
        self._miss_streak = 0

    def _get_keyframe_meta_cached(self, keyframe_id: int) -> dict[str, Any]:
        'Returns the metadata keyframe-a from the cache or SQLite.'
        keyframe_id = int(keyframe_id)
        cached = self._meta_cache.get(keyframe_id)
        if cached is not None:
            return cached
        row = db_mod.fetch_keyframe_meta(self.conn, keyframe_id)
        data = {k: row[k] for k in row.keys()}
        self._meta_cache[keyframe_id] = data
        return data

    def _get_px_to_enu_cached(self, keyframe_id: int) -> Optional[dict[str, Any]]:
        'Returns pixel conversion from cache to ENU for reference keyframe-a.'
        keyframe_id = int(keyframe_id)
        if keyframe_id in self._px_to_enu_cache:
            return self._px_to_enu_cache[keyframe_id]
        row = db_mod.fetch_px_to_enu(self.conn, keyframe_id)
        data = None if row is None else {k: row[k] for k in row.keys()}
        self._px_to_enu_cache[keyframe_id] = data
        return data

    def _get_local_features_np(self, keyframe_id: int) -> CachedLocalFeatures:
        'Uploads local defaults in NumPy format with LRU-cashing.'
        keyframe_id = int(keyframe_id)
        cached = self._feature_cache.get(keyframe_id)
        if cached is not None:
            self._feature_cache.move_to_end(keyframe_id)
            return cached

        feat_row = db_mod.fetch_local_features(self.conn, keyframe_id)
        cached = (
            np.frombuffer(feat_row["kpts"], dtype=np.float32).reshape(-1, 2),
            np.frombuffer(feat_row["desc"], dtype=np.float32).reshape(-1, 256),
            np.frombuffer(feat_row["scores"], dtype=np.float32).reshape(-1),
            int(feat_row["image_w"]),
            int(feat_row["image_h"]),
        )
        self._feature_cache[keyframe_id] = cached
        self._feature_cache.move_to_end(keyframe_id)
        if len(self._feature_cache) > int(self.feature_cache_size):
            self._feature_cache.popitem(last=False)
        return cached

    def _get_local_features_torch(self, keyframe_id: int) -> Any:
        'Uploads local signs of the standard as the tags on the matcher-a device.'
        keyframe_id = int(keyframe_id)
        cached = self._feature_torch_cache.get(keyframe_id)
        if cached is not None:
            self._feature_torch_cache.move_to_end(keyframe_id)
            return cached

        kpts, desc, scores, _, _ = self._get_local_features_np(keyframe_id)
        cached = self.matcher.numpy_to_torch((kpts, desc, scores), self.matcher.device)
        self._feature_torch_cache[keyframe_id] = cached
        self._feature_torch_cache.move_to_end(keyframe_id)
        if len(self._feature_torch_cache) > int(self.feature_torch_cache_size):
            self._feature_torch_cache.popitem(last=False)
        return cached

    def _apply_homography_geo_correction(
        self,
        H_cur_to_ref: np.ndarray,
        ref_keyframe_id: int,
        cur_w: int,
        cur_h: int,
    ) -> Optional[Tuple[float, float]]:
        'Assesss lateral displacement along the lower bar grid.'
        try:
            ref = self._get_keyframe_meta_cached(int(ref_keyframe_id))
            ref_lat = float(ref["lat"])
            ref_lng = float(ref["lng"])

            _, _, _, ref_w, ref_h = self._get_local_features_np(int(ref_keyframe_id))
            a_row = self._get_px_to_enu_cached(int(ref_keyframe_id))
            if a_row is None or int(a_row.get("valid", 0)) != 1:
                return None

            A = np.array(
                [
                    [float(a_row["a11"]), float(a_row["a12"])],
                    [float(a_row["a21"]), float(a_row["a22"])],
                ],
                dtype=np.float64,
            )

            cur_grid = _ground_anchor_grid(cur_w, cur_h)
            ref_grid = _ground_anchor_grid(ref_w, ref_h)

            cur_pts = cur_grid.reshape(-1, 1, 2).astype(np.float32)
            mapped = cv2.perspectiveTransform(cur_pts, H_cur_to_ref.astype(np.float64)).reshape(-1, 2).astype(np.float64)
            dpx = ref_grid - mapped

            finite_mask = np.isfinite(dpx).all(axis=1)
            if int(finite_mask.sum()) < int(self.homography_anchor_min_valid):
                return None

            dpx_valid = dpx[finite_mask]
            denu_all = (A @ dpx_valid.T).T
            denu_med = np.median(denu_all, axis=0).astype(np.float64)
            east_m = float(denu_med[0])
            north_m = float(denu_med[1])

            # Project movement to the local base of the route (tangens/normal)),
            # to serve mainly the transverse component, not the longitudinal drift.
            try:
                ref_meta = self._get_keyframe_meta_cached(int(ref_keyframe_id))
                video_id = int(ref_meta["video_id"])
                ref_ts_ms = float(ref_meta["ts_ms"])
                basis_rows = db_mod.fetch_keyframes_by_video_ts_window(
                    self.conn,
                    video_id=video_id,
                    ts_center_ms=ref_ts_ms,
                    back_ms=8_000.0,
                    forward_ms=8_000.0,
                    limit=24,
                )
                if len(basis_rows) >= 2:
                    # Take the earliest and the latest point in the window for a sustainable direction.
                    rows_sorted = sorted(basis_rows, key=lambda r: float(r["ts_ms"]))
                    p0 = rows_sorted[0]
                    p1 = rows_sorted[-1]
                    if float(p1["ts_ms"]) > float(p0["ts_ms"]):
                        enu_01 = ll_to_enu(
                            float(p1["lat"]),
                            float(p1["lng"]),
                            float(p0["lat"]),
                            float(p0["lng"]),
                        )
                        t = np.array([float(enu_01.east_m), float(enu_01.north_m)], dtype=np.float64)
                        t_norm = float(np.linalg.norm(t))
                        if t_norm > 1e-6:
                            t_u = t / t_norm
                            n_u = np.array([-t_u[1], t_u[0]], dtype=np.float64)
                            v = np.array([east_m, north_m], dtype=np.float64)
                            along = float(np.dot(v, t_u))
                            lateral = float(np.dot(v, n_u))
                            v_proj = (
                                float(self.along_component_gain) * along * t_u
                                + float(self.lateral_component_gain) * lateral * n_u
                            )
                            east_m = float(v_proj[0])
                            north_m = float(v_proj[1])
            except Exception:
                pass

            return enu_to_ll(ENU(east_m=east_m, north_m=north_m), ref_lat, ref_lng)
        except Exception:
            return None

    def _score_candidates(
        self,
        bgr: np.ndarray,
        candidates: List[Tuple[int, float]],
        source: str,
        ts_ms: float,
        quality_mode: str = "global",
    ) -> dict:
        """Assesses retrieval candidates and collects the final result of localization.

        Args:
            bgr: Current frame in BGR format.
            candidates: vapours ``(keyframe_id, similarity)`' after retrieval.
            source: Logical name of source for output metadata.
            ts_ms: Temporary mark of the current frame.

        Returns:
            A dictionary of localization results with coordinates, conference and
            Information on the reference keyframe-e.
        """
        cur_h, cur_w = int(bgr.shape[0]), int(bgr.shape[1])

        def _zero(reason: str) -> dict:
            return {
                "ts_ms": float(ts_ms),
                "lat": 0.0,
                "lng": 0.0,
                "confidence": 0.0,
                "source": source,
                "ref_keyframe_id": None,
                "quality_mode": str(quality_mode),
                "reject_reason": str(reason),
            }

        if not candidates:
            return _zero("no_candidates")

        cur_feats_torch = self.matcher.extract_torch(bgr)
        scored: List[Tuple[int, float, float, float, float, float, float, int, np.ndarray]] = []

        for keyframe_id, sim in candidates:
            try:
                ref_feats_torch = self._get_local_features_torch(int(keyframe_id))

                H, inlier_ratio, reproj_rmse, matches_cnt = self.matcher.match_from_torch(
                    cur_feats_torch,
                    ref_feats_torch,
                )
                if H is None or matches_cnt <= 0:
                    continue

                conf = self.confidence.compute_visual(
                    sim=float(sim),
                    inlier_ratio=float(inlier_ratio),
                    reproj_rmse=float(reproj_rmse),
                    matches_cnt=int(matches_cnt),
                )
                if conf <= 0.0:
                    continue

                ref_meta = self._get_keyframe_meta_cached(int(keyframe_id))
                ref_ts_ms = float(ref_meta["ts_ms"])
                lat = float(ref_meta["lat"])
                lng = float(ref_meta["lng"])
                scored.append(
                    (
                        int(keyframe_id),
                        ref_ts_ms,
                        lat,
                        lng,
                        conf,
                        float(inlier_ratio),
                        float(reproj_rmse),
                        int(matches_cnt),
                        H.astype(np.float64),
                    )
                )
            except Exception:
                continue

        if not scored:
            return _zero("no_valid_matches")

        scored.sort(key=lambda s: float(s[4]), reverse=True)
        best_entry = scored[0]
        best_ref_id = int(best_entry[0])
        best_ref_ts_ms = float(best_entry[1])
        best_conf = float(best_entry[4])

        lat_raw = float(best_entry[2])
        lng_raw = float(best_entry[3])
        conf_raw = float(best_conf)
        best_inlier_ratio = float(best_entry[5])
        best_reproj_rmse = float(best_entry[6])
        best_matches_cnt = int(best_entry[7])
        best_h = best_entry[8]

        if str(quality_mode) == "local":
            min_inlier = float(self.min_inlier_ratio_local)
            min_matches = int(self.min_matches_cnt_local)
            max_rmse = float(self.max_reproj_rmse_local)
        else:
            min_inlier = float(self.min_inlier_ratio)
            min_matches = int(self.min_matches_cnt)
            max_rmse = float(self.max_reproj_rmse)

        quality_ok = (
            best_inlier_ratio >= min_inlier
            and best_matches_cnt >= min_matches
            and best_reproj_rmse <= max_rmse
        )
        if not quality_ok:
            return _zero("quality_gate")

        near_best = [entry for entry in scored if float(entry[4]) >= (best_conf - float(self.ambiguity_conf_band))]
        if len(near_best) > 1:
            ref_lat = lat_raw
            ref_lng = lng_raw
            m_per_deg_lat = 111_320.0
            m_per_deg_lng = 111_320.0 * cos(radians(ref_lat))
            max_dist_m = 0.0
            for entry in near_best[1:]:
                dlat = float(entry[2]) - ref_lat
                dlng = float(entry[3]) - ref_lng
                dn = dlat * m_per_deg_lat
                de = dlng * m_per_deg_lng
                dist_m = sqrt(dn * dn + de * de)
                if dist_m > max_dist_m:
                    max_dist_m = dist_m
            if max_dist_m > float(self.ambiguity_max_geo_m):
                return _zero("ambiguity_gate")

        if self._last_ref_ts_ms is not None and self._last_source_ts_ms is not None:
            dt_source_ms = max(0.0, float(ts_ms) - float(self._last_source_ts_ms))
            expected_ref_ts_ms = float(self._last_ref_ts_ms) + dt_source_ms
            temporal_delta_ms = abs(float(best_ref_ts_ms) - expected_ref_ts_ms)
            allowed_delta_ms = min(
                float(self.max_temporal_jump_ms),
                float(self.temporal_tolerance_ms) + dt_source_ms * float(self.temporal_scale),
            )
            if temporal_delta_ms > allowed_delta_ms:
                return _zero("temporal_gate")

        corrected = self._apply_homography_geo_correction(
            H_cur_to_ref=best_h,
            ref_keyframe_id=best_ref_id,
            cur_w=cur_w,
            cur_h=cur_h,
        )
        corr_dist_m = 0.0
        corrected_applied = False
        if corrected is not None:
            corr_lat, corr_lng = float(corrected[0]), float(corrected[1])
            m_per_deg_lat = 111_320.0
            m_per_deg_lng = 111_320.0 * cos(radians(lat_raw))
            dn = (corr_lat - lat_raw) * m_per_deg_lat
            de = (corr_lng - lng_raw) * m_per_deg_lng
            corr_dist_m = sqrt(dn * dn + de * de)
            if corr_dist_m <= float(self.max_homography_offset_m):
                lat_raw = corr_lat
                lng_raw = corr_lng
                corrected_applied = True

        self._last_ref_ts_ms = float(best_ref_ts_ms)
        self._last_source_ts_ms = float(ts_ms)
        try:
            self._last_video_id = int(self._get_keyframe_meta_cached(best_ref_id)["video_id"])
        except Exception:
            self._last_video_id = None
        self._miss_streak = 0

        return {
            "ts_ms": float(ts_ms),
            "lat": float(lat_raw),
            "lng": float(lng_raw),
            "confidence": float(conf_raw),
            "source": source,
            "ref_keyframe_id": best_ref_id,
            "quality_mode": str(quality_mode),
            "reject_reason": "",
            "corrected_applied": bool(corrected_applied),
            "correction_m": float(corr_dist_m),
            "best_inlier_ratio": float(best_inlier_ratio),
            "best_reproj_rmse": float(best_reproj_rmse),
            "best_matches_cnt": int(best_matches_cnt),
        }

    def _build_local_candidates(self, q: np.ndarray, ts_ms: float) -> List[Tuple[int, float]]:
        'Returns candidates from the adjacent temporary window of previous hit.'
        if self._last_ref_ts_ms is None or self._last_source_ts_ms is None or self._last_video_id is None:
            return []
        dt_source_ms = max(0.0, float(ts_ms) - float(self._last_source_ts_ms))
        expected_ref_ts_ms = float(self._last_ref_ts_ms) + dt_source_ms
        rows = db_mod.fetch_keyframes_by_video_ts_window(
            self.conn,
            video_id=int(self._last_video_id),
            ts_center_ms=float(expected_ref_ts_ms),
            back_ms=float(self.local_back_ms),
            forward_ms=float(self.local_forward_ms),
            limit=int(self.local_limit),
        )
        if not rows:
            return []
        cand_ids, cand_sims = self.retriever.score_ids(q, [int(row["id"]) for row in rows])
        candidates = list(zip(cand_ids.tolist(), cand_sims.tolist()))
        candidates.sort(key=lambda x: float(x[1]), reverse=True)
        return candidates[: max(int(self.top_k), 12)]

    def localize_bgr(self, bgr: np.ndarray, ts_ms: float, source: str) -> dict:
        """Runs a full localization pike for one BGR frame.

        Args:
            bgr: OpenCV BGR input frame.
            ts_ms: Temporary mark linked to frame.
            source: A person &apos; s name of the source during the output.

        Returns:
            The final serealized dictionary of the localization result.
        """
        if self.retriever.ids.size == 0:
            return {
                "ts_ms": float(ts_ms),
                "lat": 0.0,
                "lng": 0.0,
                "confidence": 0.0,
                "source": source,
                "ref_keyframe_id": None,
                "quality_mode": "global",
                "reject_reason": "empty_retriever",
            }

        q = self.embedder.encode_bgr(bgr)
        local_candidates = self._build_local_candidates(q=q, ts_ms=ts_ms)
        if local_candidates and self._miss_streak < int(self.reacquire_after_misses):
            local_res = self._score_candidates(
                bgr=bgr,
                candidates=local_candidates,
                source=source,
                ts_ms=ts_ms,
                quality_mode="local",
            )
            if float(local_res.get("confidence", 0.0)) > 0.0:
                return local_res
            self._miss_streak += 1
            return local_res

        cand_ids, cand_sims = self.retriever.search(q, max(int(self.top_k), 12))
        candidates = list(zip(cand_ids.tolist(), cand_sims.tolist()))
        global_res = self._score_candidates(
            bgr=bgr,
            candidates=candidates,
            source=source,
            ts_ms=ts_ms,
            quality_mode="global",
        )
        if float(global_res.get("confidence", 0.0)) <= 0.0:
            self._miss_streak += 1
        return global_res
