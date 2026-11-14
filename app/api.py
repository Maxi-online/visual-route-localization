from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from app import db as db_mod
from app.indexing_service import IndexBuildManager, IndexJobRequest
from app.runtime import RuntimeConfig, RuntimeLocalizer
from app.settings import Settings
from app.schemas import LocalizationResult, RuntimeFrameMessage, RuntimeStartRequest
from app.vloc.localizer import Localizer


BUILD_VERSION = "vloc-mvp"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


class ResultBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._last: Optional[Dict[str, Any]] = None

    def publish(self, msg: Dict[str, Any]) -> None:
        self._last = msg
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def reset(self) -> None:
        self._last = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stream(self, request: Request) -> AsyncIterator[str]:
        try:
            if self._last is not None:
                yield f"data: {json.dumps(self._last, ensure_ascii=False)}\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            return


def _collect_video_sources(folder: Path) -> list[str]:
    result: list[str] = []
    if not folder.exists():
        return result
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            result.append(str(p.resolve()))
    return result


def _candidate_local_paths(value: str, settings: Settings) -> list[Path]:
    raw_value = str(value).strip()
    if not raw_value or "://" in raw_value:
        return []

    candidates: list[Path] = []
    normalized = raw_value.replace("\\", "/")
    normalized_lower = normalized.lower()

    try:
        candidates.append(Path(raw_value).expanduser())
    except Exception:
        pass

    if normalized.startswith("/app/") or normalized.startswith("/host-home/"):
        candidates.append(Path(normalized))

    inputs_marker = "/inputs/"
    if inputs_marker in normalized_lower:
        rel_after_inputs = normalized[normalized_lower.rfind(inputs_marker) + len(inputs_marker):].lstrip("/")
        if rel_after_inputs:
            candidates.append(settings.videos_dir.parent / Path(rel_after_inputs))

    parts = normalized.split("/")
    user_idx = next((idx for idx, part in enumerate(parts) if part.lower() == "users"), -1)
    if user_idx >= 0 and len(parts) > user_idx + 2:
        rel_after_home = [segment for segment in parts[user_idx + 2:] if segment]
        if rel_after_home:
            candidates.append(Path("/host-home").joinpath(*rel_after_home))

    return candidates


def _resolve_local_path(value: str, settings: Settings) -> Path | None:
    seen: set[str] = set()
    for candidate in _candidate_local_paths(value, settings):
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
    return None


def _list_runtime_sources(settings: Settings) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in _collect_video_sources(settings.videos_dir) + _collect_video_sources(settings.live_dir):
        if value in seen:
            continue
        if not Path(value).exists():
            continue
        seen.add(value)
        result.append({"type": "video", "value": value, "label": value})
    return result


def _list_index_sources(settings: Settings) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for value in _collect_video_sources(settings.videos_dir):
        result.append({"type": "video", "value": value, "label": value})
    return result


def _preview_stream(runtime: RuntimeLocalizer, request: Request):
    async def gen():
        last_ts = 0
        try:
            while True:
                if await request.is_disconnected():
                    return
                fr = runtime.buffer.get()
                if fr is None:
                    await asyncio.sleep(0.05)
                    continue
                if fr.ts_ns == last_ts:
                    await asyncio.sleep(0.03)
                    continue
                last_ts = fr.ts_ns
                ok, jpeg = cv2.imencode(".jpg", fr.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                data = jpeg.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode("ascii") + b"\r\n\r\n"
                    + data + b"\r\n"
                )
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            return

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


async def _save_upload(upload: UploadFile, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = await upload.read()
    out_path.write_bytes(data)
    return out_path


def _build_ws_url(request: Request, token: str) -> str:
    scheme = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}/api/v1/runtime/ws/{token}"


def build_router(
    localizer: Localizer,
    runtime: RuntimeLocalizer,
    bus: ResultBus,
    conn,
    settings: Settings,
    index_manager: IndexBuildManager,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "version": BUILD_VERSION}

    @router.get("/version")
    def version() -> Dict[str, str]:
        return {"version": BUILD_VERSION}

    @router.get("/localization/stream")
    async def localization_stream(request: Request) -> StreamingResponse:
        return StreamingResponse(bus.stream(request), media_type="text/event-stream")

    @router.get("/runtime/status")
    def runtime_status() -> JSONResponse:
        return JSONResponse(runtime.status())

    @router.get("/runtime/sources")
    def runtime_sources() -> JSONResponse:
        return JSONResponse(
            {
                "sources": _list_runtime_sources(settings),
                "supported_types": ["video", "rtsp", "websocket", "webrtc"],
                "note": 'WebRTC on MVP works via client Bridge on WebSocket.',
            }
        )

    @router.get("/index/sources")
    def index_sources() -> JSONResponse:
        return JSONResponse(
            {
                "sources": _list_index_sources(settings),
                "supported_types": ["video", "rtsp", "websocket", "webrtc"],
                "note": 'For MVP, UI indexing supports a video file or RTSP. WebSocket/WebRTC remains the next stage.',
            }
        )

    @router.get("/index/status")
    def index_status() -> JSONResponse:
        return JSONResponse(index_manager.status())

    @router.get("/index/routes")
    def index_routes() -> JSONResponse:
        videos = db_mod.list_videos(conn)
        routes: list[dict[str, Any]] = []
        for v in videos:
            points = db_mod.fetch_route_points(conn, int(v["id"]), step=10)
            routes.append(
                {
                    "video_id": int(v["id"]),
                    "path": str(v["path"]),
                    "points": [
                        {"lat": float(r["lat"]), "lng": float(r["lng"]), "ts_ms": float(r["ts_ms"])}
                        for r in points
                    ],
                }
            )
        return JSONResponse({"videos": routes, "keyframes": db_mod.count_keyframes(conn)})

    @router.get("/runtime/frame.mjpg")
    def runtime_frame_mjpg(request: Request) -> StreamingResponse:
        return _preview_stream(runtime, request)

    @router.get("/runtime/preview")
    def runtime_preview(request: Request) -> StreamingResponse:
        return _preview_stream(runtime, request)

    @router.get("/runtime/preview.mjpg")
    def runtime_preview_mjpg(request: Request) -> StreamingResponse:
        return _preview_stream(runtime, request)

    @router.post("/localize/frame", response_model=LocalizationResult)
    async def localize_frame(frame: UploadFile = File(...)) -> LocalizationResult:
        data = await frame.read()
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail='Could not decode the image.')
        res = localizer.localize_bgr(img, ts_ms=0.0, source="frame_api")
        bus.publish(res)
        return LocalizationResult(**res)

    @router.post("/index/upload-and-build")
    async def index_upload_and_build(
        source_type: str = Form("video"),
        source_value: str = Form(""),
        sample_fps: float = Form(0.0),
        max_keyframes: int = Form(0),
        points_value: str = Form(""),
        points_file: UploadFile | None = File(default=None),
        video_file: UploadFile | None = File(default=None),
    ) -> JSONResponse:
        if source_type == "webrtc":
            raise HTTPException(
                status_code=400,
                detail='Direct WebRTC for indexing is not yet supported. Use a video file or RTSP.',
            )

        if source_type == "websocket":
            raise HTTPException(
                status_code=400,
                detail='Indexing from WebSocket to MVP is not yet supported. Use a video file or RTSP.',
            )

        if video_file is None and not str(source_value).strip():
            raise HTTPException(status_code=400, detail='You need to choose a video file or indicate the path/URL of the source.')
        if points_file is None and not str(points_value).strip():
            raise HTTPException(
                status_code=400,
                detail='You need to download points.txt or indicate the path to the coordinate file.',
            )
        if video_file is None and str(source_type).strip().lower() == "video":
            local_video_path = _resolve_local_path(source_value, settings)
            if local_video_path is None or not local_video_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail='Unable to open reference video on a given path. Use the path from the project folder, the path from the user &apos; s home directory, or download the file via UI.',
                )

        temp_dir = Path(tempfile.mkdtemp(prefix="vloc-index-upload-"))
        if points_file is not None:
            points_name = points_file.filename or "reference_points.txt"
            points_path = await _save_upload(points_file, temp_dir / Path(points_name).name)
        else:
            local_points_path = _resolve_local_path(points_value, settings)
            if local_points_path is None or not local_points_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail='Unable to open points.txt along the specified path. Use the path from the project folder, the path from the user &apos; s home directory, or download the file via UI.',
                )
            points_path = local_points_path

        uploaded_video_path: Optional[Path] = None
        if video_file is not None:
            video_name = video_file.filename or "reference_video.mp4"
            uploaded_video_path = await _save_upload(video_file, temp_dir / Path(video_name).name)

        try:
            status = index_manager.start(
                IndexJobRequest(
                    source_type=str(source_type).strip().lower(),
                    source_value=str(source_value).strip(),
                    points_path=points_path,
                    uploaded_video_path=uploaded_video_path,
                    sample_fps=float(sample_fps),
                    max_keyframes=int(max_keyframes),
                    temp_root=temp_dir,
                )
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"status": "started", "job": status, "version": BUILD_VERSION})

    @router.post("/runtime/start")
    async def runtime_start(request: Request) -> JSONResponse:
        content_type = (request.headers.get("content-type") or "").lower()
        video_file: UploadFile | None = None

        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            maybe_file = form.get("video_file")
            if isinstance(maybe_file, UploadFile) or (
                hasattr(maybe_file, "filename") and hasattr(maybe_file, "read")
            ):
                video_file = maybe_file
            req = RuntimeStartRequest(
                source_type=str(form.get("source_type", "video") or "video"),
                stream_url=str(form.get("stream_url", "") or ""),
                target_fps=float(form.get("target_fps", 0.0) or 0.0),
                localize_fps=float(form.get("localize_fps", 0.0) or 0.0),
                resize_width=int(form.get("resize_width", 832) or 832),
                top_k=int(form.get("top_k", 16) or 16),
            )
        else:
            req = RuntimeStartRequest(**(await request.json()))

        source_type = str(req.source_type).strip().lower() or "video"
        if source_type == "webrtc":
            raise HTTPException(
                status_code=400,
                detail='Direct WebRTC for Backend not yet supported. Use the client WebRTC -> WebSocket Bridge.',
            )

        stream_url = str(req.stream_url).strip()
        source_token = ""
        ingest_url: Optional[str] = None

        if source_type == "websocket":
            source_token = uuid.uuid4().hex
            if not stream_url:
                stream_url = "websocket://ingest"
            ingest_url = _build_ws_url(request, source_token)
        else:
            if video_file is not None:
                upload_name = Path(video_file.filename or "runtime_upload.mp4").name
                upload_path = settings.live_dir / f"runtime_upload_{uuid.uuid4().hex}_{upload_name}"
                saved_path = await _save_upload(video_file, upload_path)
                stream_url = str(saved_path.resolve())
            elif not stream_url:
                sources = _list_runtime_sources(settings)
                if not sources:
                    raise HTTPException(
                        status_code=400,
                        detail='No Runtime source found. Show the path to the video, download the file or select the available source.',
                    )
                stream_url = str(sources[0]["value"])
            elif source_type == "video":
                local_video_path = _resolve_local_path(stream_url, settings)
                if local_video_path is None or not local_video_path.exists():
                    raise HTTPException(
                        status_code=400,
                        detail='Unable to open ringtime video along the specified path. Use the path from the project folder, the path from the user &apos; s home directory, or download the file via UI.',
                    )
                stream_url = str(local_video_path)

        cfg = RuntimeConfig(
            source_type=source_type,
            stream_url=stream_url,
            target_fps=float(req.target_fps),
            localize_fps=float(req.localize_fps),
            resize_width=int(req.resize_width),
            top_k=int(req.top_k),
            source_token=source_token,
        )
        bus.reset()
        started = runtime.start(cfg)
        payload: dict[str, Any] = {
            "status": "started" if started else "already_running",
            "stream_url": stream_url,
            "source_type": source_type,
            "version": BUILD_VERSION,
        }
        payload["runtime_session_id"] = runtime.status().get("runtime_session_id", "")
        if ingest_url is not None:
            payload["ingest_url"] = ingest_url
        return JSONResponse(payload)

    @router.post("/runtime/stop")
    async def runtime_stop() -> Dict[str, str]:
        runtime.stop()
        bus.reset()
        return {"status": "stopped", "version": BUILD_VERSION}

    @router.websocket("/runtime/ws/{token}")
    async def runtime_ws_ingest(websocket: WebSocket, token: str) -> None:
        if not runtime.accepts_websocket_token(token):
            await websocket.close(code=1008, reason="Runtime websocket source is not active.")
            return

        await websocket.accept()
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return

                if message.get("bytes") is not None:
                    runtime.push_frame_bytes(message["bytes"])
                    continue

                if message.get("text") is None:
                    continue

                raw_text = message["text"]
                data = json.loads(raw_text)
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                frame = RuntimeFrameMessage(**data)
                encoded = str(frame.image_base64)
                if "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                payload = base64.b64decode(encoded)
                runtime.push_frame_bytes(payload, source_ts_ms=frame.source_ts_ms)
        except WebSocketDisconnect:
            return
        except Exception:
            await websocket.close(code=1011, reason="Runtime websocket ingest failed.")

    return router
