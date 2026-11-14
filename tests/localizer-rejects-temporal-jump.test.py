from __future__ import annotations

import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cv2_stub = types.ModuleType("cv2")
cv2_stub.perspectiveTransform = lambda pt, H: pt
sys.modules.setdefault("cv2", cv2_stub)

matching_stub = types.ModuleType("app.vloc.matching")


class LightGlueMatcher:
    pass


matching_stub.LightGlueMatcher = LightGlueMatcher
sys.modules.setdefault("app.vloc.matching", matching_stub)

vision_stub = types.ModuleType("app.vloc.vision")


class GlobalEmbedder:
    pass


vision_stub.GlobalEmbedder = GlobalEmbedder
sys.modules.setdefault("app.vloc.vision", vision_stub)

retrieval_stub = types.ModuleType("app.vloc.retrieval")


class Retriever:
    pass


retrieval_stub.Retriever = Retriever
sys.modules.setdefault("app.vloc.retrieval", retrieval_stub)

from app.vloc.confidence import ConfidenceEstimator
from app.vloc.localizer import Localizer
from app.vloc.smoothing import EwmaSmoother


class DummyMatcher:
    device = "cpu"

    def extract_torch(self, bgr):
        return {"shape": bgr.shape}

    def match_from_torch(self, cur_feats_torch, ref_feats_torch):
        return np.eye(3, dtype=np.float32), 0.86, 1.0, 80


localizer = Localizer(
    conn=None,
    retriever=Retriever(),
    embedder=GlobalEmbedder(),
    matcher=DummyMatcher(),
    confidence=ConfidenceEstimator(),
    smoother=EwmaSmoother(),
)
localizer.temporal_tolerance_ms = 1_800.0
localizer.temporal_scale = 2.5
localizer.max_temporal_jump_ms = 8_000.0

localizer._get_local_features_torch = lambda keyframe_id: {"keyframe_id": int(keyframe_id)}

localizer._last_ref_ts_ms = 12_000.0
localizer._last_source_ts_ms = 6_000.0

localizer._get_keyframe_meta_cached = lambda keyframe_id: {
    "ts_ms": 80_000.0,  # It's too far from the expected time neighbor.
    "lat": 48.3900,
    "lng": 37.3300,
    "video_id": 1,
}

result = localizer._score_candidates(
    bgr=np.zeros((8, 8, 3), dtype=np.uint8),
    candidates=[(9000, 0.99)],
    source="runtime",
    ts_ms=6_200.0,
)

assert result["confidence"] == 0.0, result
assert result["lat"] == 0.0, result
assert result["lng"] == 0.0, result
assert result["ref_keyframe_id"] is None, result

print("PASS localizer rejects temporal jumps to non-neighbor keyframes")
