from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import sqlite3

from app import db as db_mod
from app.vloc.geo import ll_to_enu
from app.vloc.matching import LightGlueMatcher
from app.vloc.points import interpolate_track, parse_points_txt
from app.vloc.vision import GlobalEmbedder


@dataclass
class IndexedKeyframe:
    keyframe_id: int
    lat: float
    lng: float
    image_path: str
    w: int
    h: int
    feats_np: Tuple[np.ndarray, np.ndarray, np.ndarray]


def _video_info(video_path: Path) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open the video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0.0 or frame_count <= 0:
        raise RuntimeError('Could not read fps/frame_count.')
    return fps, frame_count, w, h


def _sample_frame_indices(src_fps: float, frame_count: int, sample_fps: float) -> np.ndarray:
    if float(sample_fps) <= 0.0 or float(sample_fps) >= float(src_fps):
        return np.arange(frame_count, dtype=np.int64)
    duration_sec = frame_count / src_fps
    step = 1.0 / float(sample_fps)
    times = np.arange(0.0, max(0.0, duration_sec - 1e-6), step, dtype=np.float64)
    idx = np.round(times * src_fps).astype(np.int64)
    idx = np.clip(idx, 0, frame_count - 1)
    return np.unique(idx)


def _rotation(theta: float) -> np.ndarray:
    c = float(math.cos(theta))
    s = float(math.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _fit_a_from_pairs(dpx_list: List[np.ndarray], denu_list: List[np.ndarray]) -> Tuple[np.ndarray, bool]:
    if not dpx_list or len(dpx_list) != len(denu_list):
        return np.eye(2, dtype=np.float64), False
    dpx = np.vstack(dpx_list).astype(np.float64)
    denu = np.vstack(denu_list).astype(np.float64)
    if dpx.shape[0] >= 2 and np.linalg.matrix_rank(dpx) >= 2:
        a_t, *_ = np.linalg.lstsq(dpx, denu, rcond=None)
        A = a_t.T.astype(np.float64)

        # Refinement: Dumping of gross emissions from residues.
        pred = (A @ dpx.T).T
        err = np.linalg.norm(pred - denu, axis=1)
        if err.size >= 6:
            q70 = float(np.percentile(err, 70))
            thr = max(1.0, q70 * 1.6)
            inlier_mask = err <= thr
            if int(inlier_mask.sum()) >= 3 and np.linalg.matrix_rank(dpx[inlier_mask]) >= 2:
                a_t_refined, *_ = np.linalg.lstsq(dpx[inlier_mask], denu[inlier_mask], rcond=None)
                A = a_t_refined.T.astype(np.float64)

        # Stabilize anithropy to avoid smashing the side component.
        u, s, vt = np.linalg.svd(A)
        s0 = float(max(s[0], 1e-6))
        s1 = float(max(s[1], 1e-6))
        max_anisotropy = 4.0
        if s0 / s1 > max_anisotropy:
            s1 = s0 / max_anisotropy
        # Physically plausible range of scale: Not allowed
        # Extreme m/px which makes correction unstable.
        min_scale = 0.03
        max_scale = 20.0
        s0 = min(max(s0, min_scale), max_scale)
        s1 = min(max(s1, min_scale), max_scale)
        A = (u @ np.diag([s0, s1]) @ vt).astype(np.float64)
        return A, True
    vpx = dpx[0]
    venu = denu[0]
    npx = float(np.linalg.norm(vpx))
    neu = float(np.linalg.norm(venu))
    if npx < 1e-6 or neu < 1e-6:
        return np.eye(2, dtype=np.float64), False
    scale = neu / npx
    theta = float(math.atan2(venu[1], venu[0]) - math.atan2(vpx[1], vpx[0]))
    return scale * _rotation(theta), True


def _map_center_to_ref(H: np.ndarray, neigh_w: int, neigh_h: int) -> np.ndarray:
    pt = np.array([[[float(neigh_w) * 0.5, float(neigh_h) * 0.5]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H.astype(np.float64)).reshape(-1, 2)
    return out[0].astype(np.float64)


@dataclass
class Indexer:
    conn: sqlite3.Connection
    frames_dir: Path
    device: str = "cuda"
    sample_fps: float = 0.0
    max_keyframes: int = 0
    a_min_inlier_ratio: float = 0.18
    a_max_rmse: float = 10.0
    a_neighbor_radius: int = 90
    a_neighbor_stride: int = 18
    a_min_baseline_m: float = 1.0
    a_max_neighbors_per_ref: int = 6
    progress_cb: Optional[Callable[[int, int, int, float], None]] = None

    def __post_init__(self) -> None:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = GlobalEmbedder(device=self.device)
        self.matcher = LightGlueMatcher(device=self.device, max_keypoints=512)

    def index_video(self, video_path: Path, points_path: Path) -> None:
        fps, frame_count, _, _ = _video_info(video_path)
        points = parse_points_txt(points_path)
        video_id = db_mod.upsert_video(self.conn, str(video_path), fps=fps, note="")
        frame_idxs = _sample_frame_indices(fps, frame_count, self.sample_fps)
        if self.max_keyframes > 0:
            frame_idxs = frame_idxs[: self.max_keyframes]
        times_sec = frame_idxs.astype(np.float64) / fps
        lat, lng = interpolate_track(times_sec, points)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open the video: {video_path}")
        video_out_dir = self.frames_dir / f"video_{video_id:03d}"
        video_out_dir.mkdir(parents=True, exist_ok=True)
        inserted: List[IndexedKeyframe] = []
        total = len(frame_idxs.tolist())
        for i, frame_idx in enumerate(frame_idxs.tolist()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Unable to read frame {frame_idx}")
            ts_ms = float(times_sec[i] * 1000.0)
            img_path = video_out_dir / f"frame_{int(frame_idx):09d}.jpg"
            if not cv2.imwrite(str(img_path), frame):
                raise RuntimeError(f"Unable to save {img_path}")
            keyframe_id = db_mod.insert_keyframe(
                self.conn,
                video_id=video_id,
                frame_idx=int(frame_idx),
                ts_ms=ts_ms,
                lat=float(lat[i]),
                lng=float(lng[i]),
                image_path=str(img_path.as_posix()),
            )
            emb = self.embedder.encode_bgr(frame)
            db_mod.upsert_embedding(self.conn, keyframe_id=keyframe_id, emb=emb.tobytes(), dim=int(emb.shape[0]))
            feats_np = self.matcher.extract_numpy(frame)
            kpts, desc, scores = feats_np
            db_mod.upsert_local_features(
                self.conn,
                keyframe_id=keyframe_id,
                kpts=kpts.tobytes(),
                desc=desc.tobytes(),
                scores=scores.tobytes(),
                image_w=int(frame.shape[1]),
                image_h=int(frame.shape[0]),
            )
            inserted.append(IndexedKeyframe(int(keyframe_id), float(lat[i]), float(lng[i]), str(img_path.as_posix()), int(frame.shape[1]), int(frame.shape[0]), feats_np))
            if self.progress_cb is not None:
                self.progress_cb(i + 1, total, int(frame_idx), ts_ms)
        cap.release()
        self._compute_px_to_enu(inserted)

    def _compute_px_to_enu(self, inserted: List[IndexedKeyframe]) -> None:
        if len(inserted) < 3:
            for kf in inserted:
                db_mod.upsert_px_to_enu(self.conn, kf.keyframe_id, 1.0, 0.0, 0.0, 1.0, False)
            return
        for i, ref in enumerate(inserted):
            ref_feats_t = self.matcher.numpy_to_torch(ref.feats_np, self.device)
            dpx_list: List[np.ndarray] = []
            denu_list: List[np.ndarray] = []
            stride = max(1, int(self.a_neighbor_stride))
            radius = max(stride, int(self.a_neighbor_radius))
            offsets = []
            step = stride
            while step <= radius:
                offsets.append(-step)
                offsets.append(step)
                step += stride
            candidate_js = []
            for off in offsets:
                j = i + off
                if 0 <= j < len(inserted):
                    candidate_js.append(j)
            used = 0
            for j in candidate_js:
                if j == i:
                    continue
                if used >= int(self.a_max_neighbors_per_ref):
                    break
                neigh = inserted[j]
                neigh_feats_t = self.matcher.numpy_to_torch(neigh.feats_np, self.device)
                H, inlier_ratio, rmse, matches_cnt = self.matcher.match_from_torch(neigh_feats_t, ref_feats_t)
                if H is None or matches_cnt < 20 or inlier_ratio < self.a_min_inlier_ratio or rmse > self.a_max_rmse:
                    continue
                mapped_center = _map_center_to_ref(H, neigh.w, neigh.h)
                ref_center = np.array([ref.w * 0.5, ref.h * 0.5], dtype=np.float64)
                dpx = (mapped_center - ref_center).astype(np.float64)
                enu = ll_to_enu(neigh.lat, neigh.lng, ref.lat, ref.lng)
                denu = np.array([enu.east_m, enu.north_m], dtype=np.float64)
                if float(np.linalg.norm(dpx)) < 1e-3 or float(np.linalg.norm(denu)) < 1e-3:
                    continue
                if float(np.linalg.norm(denu)) < float(self.a_min_baseline_m):
                    continue
                dpx_list.append(dpx)
                denu_list.append(denu)
                used += 1
            A, valid = _fit_a_from_pairs(dpx_list, denu_list)
            db_mod.upsert_px_to_enu(self.conn, ref.keyframe_id, float(A[0, 0]), float(A[0, 1]), float(A[1, 0]), float(A[1, 1]), bool(valid))
