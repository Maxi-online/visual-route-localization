from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import sqlite3

from app.settings import Settings
from app.stream.latest_frame import LatestFrameBuffer
from app.stream.nvdec_ffmpeg import start_ffmpeg
from app.vloc.localizer import Localizer


@dataclass
class RuntimeConfig:
    """Runtime-localization configuration.

    Attributes:
        source_type: Source type, e.g.``video`` or``websocket``.
        stream_url: Absolute path to file or URL video stream.
        target_fps: Desirable inlet decoded frequency.
        localize_fps: Target localization frequency from UI.
        resize_width: Width of frame after resuscitation.
        top_k: Number of best retrieval candidates for localization.
        source_token: WebSocket intest mode for Browser Bridge.
    """

    source_type: str
    stream_url: str
    target_fps: float
    localize_fps: float
    resize_width: int
    top_k: int
    source_token: str = ""


class RuntimeLocalizer:
    'Manage decoded, buffered and runtime-localized.'

    def __init__(self, settings: Settings, conn: sqlite3.Connection, localizer: Localizer) -> None:
        """The focal point for runtime-localization is initiated.

        Args:
            settings: Global application settings and file paths.
            conn: SQLite connection used by the runtime subsystem.
            localizer: Trained localizer for human resources processing.
        """
        self.settings = settings
        self.conn = conn
        self.localizer = localizer
        self.buffer = LatestFrameBuffer()
        self._lifecycle_lock = threading.Lock()

        self._decoder_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cfg: Optional[RuntimeConfig] = None
        self._ffmpeg_proc = None
        self._out_w = 0
        self._out_h = 0
        self._run_started_ts_ns = 0
        self._stream_duration_ms: Optional[float] = None
        self._source_fps = 0.0
        self._last_source_ts_ms = 0.0
        self._active_source_token = ""
        self._runtime_session_id = ""

        self._running = False
        self._last_error: Optional[str] = None
        self._decoded_frames = 0
        self._localized_frames = 0
        self._last_result: Optional[dict[str, Any]] = None

        self.on_result = None

    def _publish_result(self, res: dict[str, Any]) -> None:
        res["runtime_session_id"] = self._runtime_session_id
        self._localized_frames += 1
        self._last_result = res
        if self.on_result is not None:
            self.on_result(res)

    def _same_config(self, cfg: RuntimeConfig) -> bool:
        'Checks the match of a new start request with the current configuration.'
        cur = self._cfg
        if cur is None:
            return False

        def norm_url(value: str) -> str:
            s = str(value).strip()
            if "://" in s:
                return s
            try:
                return str(Path(s).resolve())
            except Exception:
                return s

        return (
            str(cur.source_type) == str(cfg.source_type)
            and
            norm_url(cur.stream_url) == norm_url(cfg.stream_url)
            and float(cur.target_fps) == float(cfg.target_fps)
            and float(cur.localize_fps) == float(cfg.localize_fps)
            and int(cur.resize_width) == int(cfg.resize_width)
            and int(cur.top_k) == int(cfg.top_k)
            and str(cur.source_token) == str(cfg.source_token)
        )

    def start(self, cfg: RuntimeConfig) -> bool:
        """Starts runtime currents with a shift configuration.

        Args:
            cfg: Runtime parameters from API or UI.

        Returns:
            ``True``, if a new one has been launchedruntime-session, otherwise.``False``,
            If the same set of parameters is already active.
        """
        with self._lifecycle_lock:
            if self._running and self._same_config(cfg):
                return False
            self._stop_locked()
            self.localizer.reset_lock()
            self.buffer = LatestFrameBuffer()
            self._stop = threading.Event()
            self._cfg = cfg
            self.localizer.top_k = int(cfg.top_k)
            self._last_error = None
            self._decoded_frames = 0
            self._localized_frames = 0
            self._last_result = None
            self._run_started_ts_ns = time.time_ns()
            self._stream_duration_ms = None
            self._source_fps = 0.0
            self._last_source_ts_ms = 0.0
            self._active_source_token = str(cfg.source_token)
            self._runtime_session_id = f"run-{time.time_ns()}"
            self._running = True

            if str(cfg.source_type) != "websocket":
                self._decoder_thread = threading.Thread(target=self._decoder_loop, daemon=True)
                self._decoder_thread.start()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            return True

    def stop(self) -> None:
        'Stops the decoder stream and the localization stream.'
        with self._lifecycle_lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        'Stops internal flows and external processes under lifecloth-lock.'
        self._stop.set()
        self._running = False
        self._active_source_token = ""
        self._runtime_session_id = ""
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.kill()
            except Exception:
                pass
            self._ffmpeg_proc = None
        if self._decoder_thread is not None:
            self._decoder_thread.join(timeout=2.0)
            self._decoder_thread = None
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def status(self) -> dict[str, Any]:
        'Returns the current runtime state for the API response.'
        cfg = self._cfg
        return {
            "running": bool(self._running),
            "stream_url": None if cfg is None else cfg.stream_url,
            "source_type": None if cfg is None else cfg.source_type,
            "target_fps": None if cfg is None else cfg.target_fps,
            "localize_fps": None if cfg is None else cfg.localize_fps,
            "resize_width": None if cfg is None else cfg.resize_width,
            "top_k": None if cfg is None else cfg.top_k,
            "websocket_active": bool(cfg is not None and str(cfg.source_type) == "websocket"),
            "decoded_frames": int(self._decoded_frames),
            "localized_frames": int(self._localized_frames),
            "has_frame": self.buffer.get() is not None,
            "last_error": self._last_error,
            "last_result": self._last_result,
            "runtime_session_id": self._runtime_session_id,
        }

    def _set_error(self, message: str) -> None:
        'Maintains the fatal runtime error and translates it into a stoped state.'
        self._last_error = message
        self._running = False
        self._active_source_token = ""

    def accepts_websocket_token(self, token: str) -> bool:
        'Checks the intest mode current for the websocket runtime.'
        cfg = self._cfg
        return bool(
            self._running
            and cfg is not None
            and str(cfg.source_type) == "websocket"
            and str(token) == str(self._active_source_token)
        )

    def push_frame_bytes(self, payload: bytes, source_ts_ms: Optional[float] = None) -> bool:
        """Decodes and puts a webcket entry frame into the buffer.

        Args:
            payload: Coded bytes of images obtained from an intest bridge.
            source_ts_ms: Non-mandatory temporary source mark in milliseconds.

        Returns:
            ``True``, if the frame was accepted inlatest-frame buffer, other``False``.
        """
        cfg = self._cfg
        if not self._running or cfg is None or str(cfg.source_type) != "websocket":
            return False
        img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self._last_error = 'Could not decode the websocket frame.'
            return False
        self._decoded_frames += 1
        if source_ts_ms is None:
            fps_for_ts = float(cfg.target_fps) if float(cfg.target_fps) > 0.0 else max(1.0, float(self._source_fps) or 30.0)
            source_ts_ms = (float(self._decoded_frames - 1) / fps_for_ts) * 1000.0
        source_ts_ms = float(source_ts_ms)
        if source_ts_ms + 250.0 < self._last_source_ts_ms:
            self.localizer.reset_lock()
            self._last_result = None
        self._last_source_ts_ms = source_ts_ms
        self.buffer.put(img, source_ts_ms=source_ts_ms)
        self._last_error = None
        return True

    def _decoder_loop(self) -> None:
        'Reads FFmpeg footage and publishes it in the back of the last frame.'
        assert self._cfg is not None
        cfg = self._cfg

        try:
            proc, out_w, out_h = start_ffmpeg(
                self.settings.ffmpeg_bin,
                self.settings.ffprobe_bin,
                cfg.stream_url,
                cfg.target_fps,
                cfg.resize_width,
            )
            self._ffmpeg_proc = proc
            self._out_w = out_w
            self._out_h = out_h
            if "://" not in str(cfg.stream_url):
                from app.stream.nvdec_ffmpeg import probe_stream

                info = probe_stream(self.settings.ffprobe_bin, cfg.stream_url)
                if info is not None and float(info.duration_sec) > 0.0:
                    self._stream_duration_ms = float(info.duration_sec) * 1000.0
                if info is not None and float(info.fps) > 0.0:
                    self._source_fps = float(info.fps)
        except Exception as exc:
            self._set_error(f"GPU-only mode: FFmpeg/NVDEC did not start: {exc}")
            return

        frame_bytes = self._out_w * self._out_h * 3
        proc = self._ffmpeg_proc
        if proc is None:
            self._set_error('FFmpeg process not initialized.')
            return

        stdout = proc.stdout
        if stdout is None:
            self._set_error('FFmpeg stdout is not available.')
            return

        while not self._stop.is_set():
            raw = stdout.read(frame_bytes)
            if raw is None or len(raw) != frame_bytes:
                if proc.poll() is not None:
                    self._set_error('The FFmpeg has ended or the flow has ended.')
                    return
                time.sleep(0.005)
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self._out_h, self._out_w, 3)
            self._decoded_frames += 1
            fps_for_ts = float(cfg.target_fps) if float(cfg.target_fps) > 0.0 else max(1.0, float(self._source_fps) or 30.0)
            source_ts_ms = (float(self._decoded_frames - 1) / fps_for_ts) * 1000.0
            if self._stream_duration_ms is not None and self._stream_duration_ms > 1.0:
                source_ts_ms = source_ts_ms % self._stream_duration_ms
                if source_ts_ms + 250.0 < self._last_source_ts_ms:
                    self.localizer.reset_lock()
                    self._last_result = None
                self._last_source_ts_ms = source_ts_ms
            self.buffer.put(frame, source_ts_ms=source_ts_ms)

    def _worker_loop(self) -> None:
        """Conducts localization for each new frame from the buffer.

        Flow only handles fresh human resources and does not publish synthetics
        coordinates between confirmed localizations.
        """
        cfg = self._cfg
        if cfg is None:
            return

        last_frame_ts_ns = 0
        while not self._stop.is_set():
            fr = self.buffer.get()
            if fr is None:
                time.sleep(0.005)
                continue
            if fr.ts_ns == last_frame_ts_ns:
                time.sleep(0.002)
                continue
            last_frame_ts_ns = fr.ts_ns

            try:
                res = self.localizer.localize_bgr(
                    fr.bgr,
                    ts_ms=float(fr.source_ts_ms),
                    source="runtime",
                )
                self._publish_result(res)
            except Exception as exc:
                self._set_error(f"Error of localization: {exc}")
                return
