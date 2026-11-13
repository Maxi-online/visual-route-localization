from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class Retriever:
    ids: np.ndarray
    embs: np.ndarray
    use_faiss: bool = False

    def __post_init__(self) -> None:
        self.index = None
        self._id_to_index: dict[int, int] = {}
        if self.embs.ndim != 2:
            self.embs = np.zeros((0, 1), dtype=np.float32)
        if self.ids.size == 0 or self.embs.shape[0] == 0:
            self.use_faiss = False
            return
        self._id_to_index = {int(keyframe_id): idx for idx, keyframe_id in enumerate(self.ids.tolist())}
        if not self.use_faiss:
            return
        try:
            import faiss  # type: ignore
        except Exception:
            self.use_faiss = False
            return
        d = int(self.embs.shape[1])
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embs.astype(np.float32))

    def search(self, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.ids.size == 0 or self.embs.shape[0] == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = q.astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-12)
        q = q.reshape(1, -1)
        top_k = int(max(1, min(top_k, self.embs.shape[0])))

        if self.use_faiss and self.index is not None:
            scores, idx = self.index.search(q, top_k)
            return self.ids[idx.ravel()], scores.ravel().astype(np.float32)

        scores = (self.embs @ q.ravel()).astype(np.float32)
        idx = np.argsort(-scores)[:top_k]
        return self.ids[idx], scores[idx]

    def score_ids(self, q: np.ndarray, candidate_ids: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        if self.ids.size == 0 or self.embs.shape[0] == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        requested_ids: list[int] = []
        requested_idx: list[int] = []
        for candidate_id in candidate_ids:
            keyframe_id = int(candidate_id)
            emb_idx = self._id_to_index.get(keyframe_id)
            if emb_idx is None:
                continue
            requested_ids.append(keyframe_id)
            requested_idx.append(emb_idx)

        if not requested_idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = q.astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-12)
        scores = (self.embs[np.array(requested_idx, dtype=np.int64)] @ q.ravel()).astype(np.float32)
        return np.array(requested_ids, dtype=np.int64), scores
