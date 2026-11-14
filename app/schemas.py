from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RuntimeStartRequest(BaseModel):
    source_type: Literal["video", "rtsp", "websocket", "webrtc"] = Field(
        "video",
        description='Runtime source type',
    )
    stream_url: str = Field("", description='RTSP/DP URL or file path')
    target_fps: float = Field(0.0, ge=0.0, le=60.0)
    localize_fps: float = Field(0.0, ge=0.0, le=60.0)
    resize_width: int = Field(832, ge=320, le=3840)
    top_k: int = Field(16, ge=1, le=50)


class LocalizationResult(BaseModel):
    ts_ms: float
    lat: float
    lng: float
    confidence: float = Field(ge=0.0, le=100.0)
    source: str
    ref_keyframe_id: int | None = None


class RuntimeFrameMessage(BaseModel):
    source_ts_ms: float | None = Field(default=None, ge=0.0)
    image_base64: str = Field(description="JPEG/PNG frame encoded as base64")
