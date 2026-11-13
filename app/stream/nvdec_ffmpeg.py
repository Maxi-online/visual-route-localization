from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class StreamInfo:
    width: int
    height: int
    fps: float
    duration_sec: float


def probe_stream(ffprobe_bin: str, url: str) -> Optional[StreamInfo]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        url,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=8)
        data = json.loads(p.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None
        s = streams[0]
        w = int(s.get("width", 0))
        h = int(s.get("height", 0))
        afr = str(s.get("avg_frame_rate", "0/1"))
        num, den = afr.split("/")
        fps = float(num) / float(den) if float(den) != 0.0 else 0.0
        if w <= 0 or h <= 0:
            return None
        if fps <= 0.0:
            fps = 30.0
        duration_sec = 0.0
        fmt = data.get("format", {})
        try:
            duration_sec = float(fmt.get("duration", 0.0))
        except Exception:
            duration_sec = 0.0
        return StreamInfo(width=w, height=h, fps=fps, duration_sec=duration_sec)
    except Exception:
        return None


def build_ffmpeg_cmd(
    ffmpeg_bin: str,
    url: str,
    width: int,
    height: int,
    target_fps: float,
    resize_width: int,
) -> Tuple[list[str], int, int]:
    scale_w = max(int(width), int(resize_width))
    in_ar = float(width) / float(height)
    out_h = int(round(scale_w / in_ar))
    if out_h % 2 == 1:
        out_h += 1

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
    ]

    if os.path.isfile(url):
        cmd += ["-stream_loop", "-1", "-re"]
    elif url.lower().startswith("rtsp://"):
        cmd += ["-rtsp_transport", "udp"]

    filter_steps = [f"scale_cuda={scale_w}:{out_h}", "hwdownload", "format=nv12", "format=bgr24"]
    if float(target_fps) > 0.0:
        filter_steps.insert(0, f"fps={float(target_fps):.3f}")

    cmd += [
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        url,
        "-vf",
        ",".join(filter_steps),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    return cmd, scale_w, out_h


def start_ffmpeg(
    ffmpeg_bin: str,
    ffprobe_bin: str,
    url: str,
    target_fps: float,
    resize_width: int,
) -> tuple[subprocess.Popen[bytes], int, int]:
    info = probe_stream(ffprobe_bin, url)
    if info is None:
        raise RuntimeError("Не удалось определить размер потока через ffprobe.")

    cmd, out_w, out_h = build_ffmpeg_cmd(
        ffmpeg_bin=ffmpeg_bin,
        url=url,
        width=info.width,
        height=info.height,
        target_fps=target_fps,
        resize_width=resize_width,
    )

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.stdout is None:
        raise RuntimeError("FFmpeg stdout is None.")
    return p, out_w, out_h
