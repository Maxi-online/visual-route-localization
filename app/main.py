from __future__ import annotations

import asyncio

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import db as db_mod
from app.api import ResultBus, build_router
from app.indexing_service import IndexBuildManager
from app.runtime import RuntimeLocalizer
from app.settings import Settings
from app.vloc.confidence import ConfidenceEstimator
from app.vloc.localizer import Localizer
from app.vloc.matching import LightGlueMatcher
from app.vloc.retrieval import Retriever
from app.vloc.smoothing import EwmaSmoother
from app.vloc.vision import GlobalEmbedder


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


def load_retriever(conn):
    try:
        ids, dim, blobs = db_mod.fetch_embedding_matrix(conn)
    except Exception:
        ids, dim, blobs = [], 0, []

    if not ids:
        return Retriever(ids=np.array([], dtype=np.int64), embs=np.zeros((0, 1), dtype=np.float32), use_faiss=False)

    embs = np.vstack(
        [np.frombuffer(b, dtype=np.float32).reshape(1, dim) for b in blobs]
    ).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    use_faiss = False
    try:
        import faiss  # type: ignore
        use_faiss = True
    except Exception:
        use_faiss = False

    return Retriever(ids=np.array(ids, dtype=np.int64), embs=embs, use_faiss=use_faiss)


settings = Settings()
conn = db_mod.connect(settings.db_path)
db_mod.migrate(conn)

retriever = load_retriever(conn)
embedder = GlobalEmbedder(device=settings.device)
matcher = LightGlueMatcher(device=settings.device, max_keypoints=settings.sp_max_keypoints)
confidence = ConfidenceEstimator()
smoother = EwmaSmoother()

localizer = Localizer(
    conn=conn,
    retriever=retriever,
    embedder=embedder,
    matcher=matcher,
    confidence=confidence,
    smoother=smoother,
    top_k=settings.runtime_top_k,
)

bus = ResultBus()
runtime = RuntimeLocalizer(settings=settings, conn=conn, localizer=localizer)
runtime.on_result = bus.publish


def reload_runtime_index_state() -> None:
    localizer.retriever = load_retriever(conn)
    localizer.reset_lock()
    runtime.localizer = localizer


index_manager = IndexBuildManager(
    settings=settings,
    conn=conn,
    stop_runtime=runtime.stop,
    reload_localizer=reload_runtime_index_state,
)

app = FastAPI(title="VLOC MVP (Visual Localization)")


@app.on_event("startup")
async def setup_asyncio_exception_handler() -> None:
    loop = asyncio.get_running_loop()
    default_handler = loop.get_exception_handler()

    def handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
            return
        if default_handler is not None:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


app.include_router(build_router(localizer, runtime, bus, conn, settings, index_manager))
app.mount("/", NoCacheStaticFiles(directory="web", html=True), name="web")
