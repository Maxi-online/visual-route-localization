from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLOC_", env_file=".env", extra="ignore")

    db_path: Path = Path("data/vpr.sqlite")
    faiss_index_path: Path = Path("data/faiss.index")
    frames_dir: Path = Path("data/frames")
    videos_dir: Path = Path("inputs/videos")
    points_dir: Path = Path("inputs/points")
    live_dir: Path = Path("inputs/live")

    reference_video_name: str = "ref_01.mp4"
    reference_points_name: str = "ref_01.txt"
    runtime_video_name: str = "ref_02.mp4"

    # GPU-only режим
    device: str = "cuda"
    sp_max_keypoints: int = 512

    runtime_target_fps: float = 20.0
    runtime_localize_fps: float = 8.0
    runtime_resize_width: int = 832
    runtime_top_k: int = 8

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
