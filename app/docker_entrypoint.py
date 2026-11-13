from __future__ import annotations

import os
from pathlib import Path


def _ensure_runtime_dirs() -> None:
    db_path = Path(os.getenv("VLOC_DB_PATH", "/app/data/vpr.sqlite"))
    frames_dir = Path(os.getenv("VLOC_FRAMES_DIR", "/app/data/frames"))
    videos_dir = Path(os.getenv("VLOC_VIDEOS_DIR", "/app/inputs/videos"))
    points_dir = Path(os.getenv("VLOC_POINTS_DIR", "/app/inputs/points"))
    live_dir = Path(os.getenv("VLOC_LIVE_DIR", "/app/inputs/live"))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)
    live_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    _ensure_runtime_dirs()

    host = os.getenv("VLOC_HOST", "0.0.0.0")
    port = os.getenv("VLOC_PORT", "8002")
    frontend_url = f"http://localhost:{port}/"

    print("=" * 72, flush=True)
    print("VLOC MVP (Visual Localization) Docker container is ready.", flush=True)
    print("1. Откройте браузер на этом компьютере.", flush=True)
    print(f"2. Перейдите по адресу: {frontend_url}", flush=True)
    print("3. Выберите эталонное видео, points.txt и runtime-видео прямо в интерфейсе.", flush=True)
    print("4. Можно выбрать файлы прямо в UI или указать путь к доступному локальному файлу.", flush=True)
    print("5. Для runtime можно сразу использовать видеофайл, RTSP или WebSocket-источник.", flush=True)
    print("=" * 72, flush=True)

    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "app.main:app",
            "--host",
            host,
            "--port",
            port,
        ],
    )


if __name__ == "__main__":
    main()
