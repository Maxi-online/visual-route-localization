from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app import db as db_mod


def parse_args() -> argparse.Namespace:
    'Reviews the command line arguments for the re-assembly of the retrieval index.'
    p = argparse.ArgumentParser(description='Retrieval index collection from SQLite.')
    p.add_argument("--db", type=str, default="data/vpr.sqlite")
    p.add_argument("--index", type=str, default="data/faiss.index")
    return p.parse_args()


def main() -> None:
    'Retrieval index from already indexed embeddings.'
    args = parse_args()
    conn = db_mod.connect(Path(args.db))
    db_mod.migrate(conn)
    ids, dim, blobs = db_mod.fetch_embedding_matrix(conn)
    embs = np.vstack([np.frombuffer(b, dtype=np.float32).reshape(1, dim) for b in blobs]).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    try:
        import faiss  # type: ignore
    except Exception:  # noqa: BLE001
        out = Path(args.index).with_suffix('.embeddings.npy')
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, {'ids': np.array(ids, dtype=np.int64), 'embs': embs})
        print(f'FAISS not installed. Saved numpy embeddings: {out}')
        return
    index = faiss.IndexFlatIP(int(dim))
    index.add(embs)
    out_path = Path(args.index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    print(f'OK: {out_path}')


if __name__ == '__main__':
    main()
