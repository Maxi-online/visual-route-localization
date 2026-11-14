from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import re

import numpy as np

from app import db as db_mod
from app.settings import Settings
from app.vloc.indexer import Indexer
from app.vloc.points import parse_points_txt


def rebuild_index(db_path: Path, index_path: Path) -> None:
    """Retrieval index from embeddings stored in SQLite.

    Args:
        db_path: Path to SQLite Base with VPR embedding-something.
        index_path: Target route for FAISS index or numpy fallsback file.
    """
    conn = db_mod.connect(db_path)
    db_mod.migrate(conn)
    ids, dim, blobs = db_mod.fetch_embedding_matrix(conn)
    embs = np.vstack([np.frombuffer(b, dtype=np.float32).reshape(1, dim) for b in blobs]).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    try:
        import faiss  # type: ignore
    except Exception:
        out = index_path.with_suffix(".embeddings.npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, {"ids": np.array(ids, dtype=np.int64), "embs": embs})
        return
    index = faiss.IndexFlatIP(int(dim))
    index.add(embs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def _normalize_local_path(source_value: str) -> Optional[Path]:
    """Normalizes the local route received from the user.

    Args:
        source_value: Raw source value from API or UI.

    Returns:
        Permitted local route or ``None`' if source is empty
        or is a remote UrL.
    """
    value = str(source_value).strip()
    if not value or "://" in value:
        return None
    try:
        path = Path(value).expanduser().resolve()
    except Exception:
        return None
    return path


def _copy_if_needed(src: Path, dst: Path) -> Path:
    'Copys file only if source and purpose differ.'
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if src_resolved == dst_resolved:
        return dst_resolved
    dst_resolved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_resolved, dst_resolved)
    return dst_resolved


def _sanitize_storage_name(name: str, fallback: str, suffix: str) -> str:
    raw = Path(str(name).strip() or fallback).name
    stem = Path(raw).stem or Path(fallback).stem or "reference"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "reference"
    ext = Path(raw).suffix or suffix
    return f"{safe_stem}{ext}"


def _capture_source_to_video(settings: Settings, source_value: str, out_path: Path, duration_sec: float) -> Path:
    """Seizes remote source into local reference video file.

    Args:
        settings: FFmpeg and FFprobe application settings.
        source_value: Stream or source string identifier.
        out_path: Target path to save reference video.
        duration_sec: Length of capture in seconds.

    Returns:
        A way to a captured or recoded video file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if duration_sec > 0:
        cmd += ["-t", f"{float(duration_sec):.3f}"]
    cmd += [
        "-i",
        source_value,
        "-an",
        "-c:v",
        "copy",
        str(out_path),
    ]
    first = subprocess.run(cmd, capture_output=True, text=True)
    if first.returncode == 0 and out_path.exists():
        return out_path

    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if duration_sec > 0:
        cmd += ["-t", f"{float(duration_sec):.3f}"]
    cmd += [
        "-i",
        source_value,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    second = subprocess.run(cmd, capture_output=True, text=True)
    if second.returncode != 0 or not out_path.exists():
        err = second.stderr.strip() or first.stderr.strip() or "ffmpeg capture failed"
        raise RuntimeError(f"Could not prepare source for indexation: {err}")
    return out_path


@dataclass
class IndexJobRequest:
    'The parameters of one indexing task launched from UI.'

    source_type: str
    source_value: str
    points_path: Path
    uploaded_video_path: Optional[Path]
    sample_fps: float
    max_keyframes: int
    temp_root: Optional[Path] = None


class IndexBuildManager:
    'Coordinates background re-injection tasks launched through API.'

    def __init__(
        self,
        settings: Settings,
        conn,
        stop_runtime: Callable[[], None],
        reload_localizer: Callable[[], None],
    ) -> None:
        """It is initiated by the background index manager.

        Args:
            settings: Global application settings.
            conn: General SQLite connection.
            stop_runtime: Callback to stop the current runtime session.
            reload_localizer: Callback to reboot retrieval in memory.
        """
        self.settings = settings
        self.conn = conn
        self._stop_runtime = stop_runtime
        self._reload_localizer = reload_localizer
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._status: dict[str, object] = {
            "state": "idle",
            "detail": "The index hasn't started yet.",
            "error": None,
            "started_at": None,
            "finished_at": None,
            "keyframes": db_mod.count_keyframes(conn),
            "source_type": None,
            "source_value": None,
        }

    def status(self) -> dict[str, object]:
        'Returns a copy of the current indexation status.'
        with self._lock:
            return dict(self._status)

    def _set_status(self, **kwargs: object) -> None:
        'Atomarno updates the internal indexation state.'
        with self._lock:
            self._status.update(kwargs)

    def start(self, request: IndexJobRequest) -> dict[str, object]:
        """Starts a new background indexing task.

        Args:
            request: Full description of the indexing task from the API layer.

        Returns:
            Status image just initialized.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError('The indexing is already in progress.')
            self._status = {
                "state": "running",
                "detail": 'Preparation of indexation...',
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
                "keyframes": db_mod.count_keyframes(self.conn),
                "source_type": request.source_type,
                "source_value": request.source_value,
            }
            self._thread = threading.Thread(target=self._run_job, args=(request,), daemon=True)
            self._thread.start()
            return dict(self._status)

    def _materialize_reference_video(self, request: IndexJobRequest, work_dir: Path) -> Path:
        """Prepares a reference video used for indexing.

        Args:
            request: User &apos; s request with source type and routes.
            work_dir: Temporary directory for intermediate files.

        Returns:
            Path to reference video inside the project's work directorate.
        """
        if request.uploaded_video_path is not None:
            target_name = _sanitize_storage_name(
                request.uploaded_video_path.name,
                self.settings.reference_video_name,
                request.uploaded_video_path.suffix or ".mp4",
            )
            target_video = (self.settings.videos_dir / target_name).resolve()
            _copy_if_needed(request.uploaded_video_path, target_video)
            return target_video

        source_value = str(request.source_value).strip()
        if not source_value:
            raise RuntimeError('No reference video source specified.')

        local_path = _normalize_local_path(source_value)
        if request.source_type in {"video", "path"} and local_path is not None and local_path.exists():
            target_name = _sanitize_storage_name(
                local_path.name,
                self.settings.reference_video_name,
                local_path.suffix or ".mp4",
            )
            target_video = (self.settings.videos_dir / target_name).resolve()
            _copy_if_needed(local_path, target_video)
            return target_video

        if request.source_type == "websocket":
            raise RuntimeError('Indexing from WebSocket to MVP is not yet supported. Use a file or RTSP.')

        if request.source_type == "webrtc":
            raise RuntimeError('Direct WebRTC for indexing is not yet supported. Use a video file or RTSP.')

        duration_sec = 0.0
        points = parse_points_txt(request.points_path)
        if points:
            duration_sec = float(points[-1].time_sec) + 1.0
        captured = work_dir / "captured_reference.mp4"
        captured_video = _capture_source_to_video(self.settings, source_value, captured, duration_sec)
        target_name = _sanitize_storage_name(
            request.points_path.stem + ".mp4",
            self.settings.reference_video_name,
            ".mp4",
        )
        target_video = (self.settings.videos_dir / target_name).resolve()
        return _copy_if_needed(captured_video, target_video)

    def _materialize_reference_points(self, request: IndexJobRequest, target_video: Path) -> Path:
        target_name = _sanitize_storage_name(
            request.points_path.name,
            target_video.stem + ".txt",
            request.points_path.suffix or ".txt",
        )
        if Path(target_name).suffix.lower() != ".txt":
            target_name = f"{Path(target_name).stem}.txt"
        target_points = (self.settings.points_dir / target_name).resolve()
        return _copy_if_needed(request.points_path, target_points)

    def _run_job(self, request: IndexJobRequest) -> None:
        'Runs a full-time re-injection pike in the background stream.'
        work_dir = Path(tempfile.mkdtemp(prefix="vloc-index-"))
        try:
            self._set_status(detail='Stopping ringtime before re-indicating...')
            self._stop_runtime()

            self._set_status(detail='Preparation of a reference video...')
            reference_video = self._materialize_reference_video(request, work_dir)
            reference_points = self._materialize_reference_points(request, reference_video)

            self._set_status(detail='Update the selected masterpiece data...')
            old_video_id = db_mod.delete_video_by_path(self.conn, str(reference_video))
            if old_video_id is not None:
                old_frames_dir = self.settings.frames_dir / f"video_{int(old_video_id):03d}"
                if old_frames_dir.exists():
                    shutil.rmtree(old_frames_dir)

            self._set_status(detail='Standard indexation...')

            def on_progress(done: int, total: int, frame_idx: int, ts_ms: float) -> None:
                self._set_status(
                    detail=f"Personnel indexing: {done}/{total}, frame={frame_idx}, ts_ms={ts_ms:.0f}",
                    keyframes=db_mod.count_keyframes(self.conn),
                )

            indexer = Indexer(
                conn=self.conn,
                frames_dir=self.settings.frames_dir.resolve(),
                device=self.settings.device,
                sample_fps=float(request.sample_fps),
                max_keyframes=int(request.max_keyframes),
                progress_cb=on_progress,
            )
            indexer.index_video(reference_video, reference_points)

            self._set_status(detail='Retrieval index assembly...')
            rebuild_index(self.settings.db_path.resolve(), self.settings.faiss_index_path.resolve())

            self._set_status(detail='Localizer in memory...')
            self._reload_localizer()

            self._set_status(
                state="completed",
                detail='Indexation complete.',
                error=None,
                finished_at=time.time(),
                keyframes=db_mod.count_keyframes(self.conn),
                source_value=str(reference_video),
            )
        except Exception as exc:
            self._set_status(
                state="failed",
                detail='Indexing ended with a mistake.',
                error=str(exc),
                finished_at=time.time(),
                keyframes=db_mod.count_keyframes(self.conn),
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            if request.temp_root is not None:
                shutil.rmtree(request.temp_root, ignore_errors=True)
