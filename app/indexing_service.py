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
    """Пересобирает retrieval-индекс из embedding-ов, сохраненных в SQLite.

    Args:
        db_path: Путь к SQLite-базе с VPR embedding-ами.
        index_path: Целевой путь для FAISS-индекса или numpy fallback-файла.
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
    """Нормализует локальный путь, полученный от пользователя.

    Args:
        source_value: Сырое значение источника из API или UI.

    Returns:
        Разрешенный локальный путь или ``None``, если источник пустой
        или является удаленным URL.
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
    """Копирует файл только если источник и назначение различаются."""
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
    """Захватывает удаленный источник в локальный файл эталонного видео.

    Args:
        settings: Настройки приложения с путями к FFmpeg и FFprobe.
        source_value: URL потока или строковый идентификатор источника.
        out_path: Целевой путь для сохранения эталонного видео.
        duration_sec: Длительность захвата в секундах.

    Returns:
        Путь к захваченному или перекодированному видеофайлу.
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
        raise RuntimeError(f"Не удалось подготовить источник для индексации: {err}")
    return out_path


@dataclass
class IndexJobRequest:
    """Параметры одной задачи индексации, запущенной из UI."""

    source_type: str
    source_value: str
    points_path: Path
    uploaded_video_path: Optional[Path]
    sample_fps: float
    max_keyframes: int
    temp_root: Optional[Path] = None


class IndexBuildManager:
    """Координирует фоновые задачи переиндексации, запущенные через API."""

    def __init__(
        self,
        settings: Settings,
        conn,
        stop_runtime: Callable[[], None],
        reload_localizer: Callable[[], None],
    ) -> None:
        """Инициализирует менеджер фоновой индексации.

        Args:
            settings: Глобальные настройки приложения.
            conn: Общее SQLite-соединение.
            stop_runtime: Callback для остановки текущего runtime-сеанса.
            reload_localizer: Callback для перезагрузки retrieval-состояния в памяти.
        """
        self.settings = settings
        self.conn = conn
        self._stop_runtime = stop_runtime
        self._reload_localizer = reload_localizer
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._status: dict[str, object] = {
            "state": "idle",
            "detail": "Индексация еще не запускалась.",
            "error": None,
            "started_at": None,
            "finished_at": None,
            "keyframes": db_mod.count_keyframes(conn),
            "source_type": None,
            "source_value": None,
        }

    def status(self) -> dict[str, object]:
        """Возвращает копию текущего состояния индексации."""
        with self._lock:
            return dict(self._status)

    def _set_status(self, **kwargs: object) -> None:
        """Атомарно обновляет внутреннее состояние индексации."""
        with self._lock:
            self._status.update(kwargs)

    def start(self, request: IndexJobRequest) -> dict[str, object]:
        """Запускает новую фоновую задачу индексации.

        Args:
            request: Полное описание задачи индексации из API-слоя.

        Returns:
            Снимок состояния только что инициализированной задачи.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Индексация уже выполняется.")
            self._status = {
                "state": "running",
                "detail": "Подготовка индексации...",
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
        """Подготавливает эталонное видео, используемое для индексации.

        Args:
            request: Пользовательский запрос с типом источника и путями.
            work_dir: Временная директория для промежуточных файлов.

        Returns:
            Путь к эталонному видео внутри рабочей директории проекта.
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
            raise RuntimeError("Не указан источник эталонного видео.")

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
            raise RuntimeError("Индексация из WebSocket в MVP пока не поддержана. Используйте файл или RTSP.")

        if request.source_type == "webrtc":
            raise RuntimeError("Прямой WebRTC для индексации пока не поддержан. Используйте видеофайл или RTSP.")

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
        """Выполняет полный пайплайн переиндексации в фоновом потоке."""
        work_dir = Path(tempfile.mkdtemp(prefix="vloc-index-"))
        try:
            self._set_status(detail="Остановка runtime перед переиндексацией...")
            self._stop_runtime()

            self._set_status(detail="Подготовка эталонного видео...")
            reference_video = self._materialize_reference_video(request, work_dir)
            reference_points = self._materialize_reference_points(request, reference_video)

            self._set_status(detail="Обновление данных выбранного эталона...")
            old_video_id = db_mod.delete_video_by_path(self.conn, str(reference_video))
            if old_video_id is not None:
                old_frames_dir = self.settings.frames_dir / f"video_{int(old_video_id):03d}"
                if old_frames_dir.exists():
                    shutil.rmtree(old_frames_dir)

            self._set_status(detail="Индексация эталона...")

            def on_progress(done: int, total: int, frame_idx: int, ts_ms: float) -> None:
                self._set_status(
                    detail=f"Индексация кадров: {done}/{total}, frame={frame_idx}, ts_ms={ts_ms:.0f}",
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

            self._set_status(detail="Сборка retrieval-индекса...")
            rebuild_index(self.settings.db_path.resolve(), self.settings.faiss_index_path.resolve())

            self._set_status(detail="Обновление localizer в памяти...")
            self._reload_localizer()

            self._set_status(
                state="completed",
                detail="Индексация завершена.",
                error=None,
                finished_at=time.time(),
                keyframes=db_mod.count_keyframes(self.conn),
                source_value=str(reference_video),
            )
        except Exception as exc:
            self._set_status(
                state="failed",
                detail="Индексация завершилась с ошибкой.",
                error=str(exc),
                finished_at=time.time(),
                keyframes=db_mod.count_keyframes(self.conn),
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            if request.temp_root is not None:
                shutil.rmtree(request.temp_root, ignore_errors=True)
