from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from app import db as db_mod
from app.settings import Settings
from app.vloc.indexer import Indexer


def rebuild_index(db_path: Path, index_path: Path) -> None:
    'Collects retrieval index from embeddings stored in SQLite.'
    conn = db_mod.connect(db_path)
    db_mod.migrate(conn)
    ids, dim, blobs = db_mod.fetch_embedding_matrix(conn)
    embs = np.vstack([np.frombuffer(b, dtype=np.float32).reshape(1, dim) for b in blobs]).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    try:
        import faiss  # type: ignore
    except Exception:  # noqa: BLE001
        out = index_path.with_suffix('.embeddings.npy')
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, {'ids': np.array(ids, dtype=np.int64), 'embs': embs})
        print(f'FAISS not installed. Saved numpy embeddings: {out}')
        return
    index = faiss.IndexFlatIP(int(dim))
    index.add(embs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f'OK: {index_path}')


def parse_args() -> argparse.Namespace:
    'Reviews command line arguments for CLI standard indexation.'
    p = argparse.ArgumentParser(description='Indexing of ref_01 reference route only and assembly of retrieval index.')
    p.add_argument('--db', type=str, default='data/vpr.sqlite')
    p.add_argument('--index', type=str, default='data/faiss.index')
    p.add_argument('--frames-dir', type=str, default='data/frames')
    p.add_argument('--videos-dir', type=str, default='inputs/videos')
    p.add_argument('--points-dir', type=str, default='inputs/points')
    p.add_argument('--video-name', type=str, default='')
    p.add_argument('--points-name', type=str, default='')
    p.add_argument('--sample-fps', type=float, default=0.0)
    p.add_argument('--max-keyframes', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main() -> None:
    'Performs a complete standardization cycle for the standard project path.'
    args = parse_args()
    settings = Settings()
    db_path = Path(args.db).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve()
    frames_dir = Path(args.frames_dir).expanduser().resolve()
    videos_dir = Path(args.videos_dir).expanduser().resolve()
    points_dir = Path(args.points_dir).expanduser().resolve()
    if not videos_dir.exists():
        raise FileNotFoundError(f'Videos folder not found: {videos_dir}')
    if not points_dir.exists():
        raise FileNotFoundError(f'The point folder is not found: {points_dir}')

    video_name = str(args.video_name).strip() or settings.reference_video_name
    points_name = str(args.points_name).strip() or settings.reference_points_name
    video_path = (videos_dir / video_name).resolve()
    points_path = (points_dir / points_name).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f'No reference video for indexing found: {video_path}')
    if not points_path.exists():
        raise FileNotFoundError(f'No coordinate file for indexation found: {points_path}')

    conn = db_mod.connect(db_path)
    db_mod.migrate(conn)
    old_video_id = db_mod.delete_video_by_path(conn, str(video_path))
    if old_video_id is not None:
        old_frames_dir = frames_dir / f"video_{int(old_video_id):03d}"
        if old_frames_dir.exists():
            shutil.rmtree(old_frames_dir)
    indexer = Indexer(conn=conn, frames_dir=frames_dir, device=str(args.device), sample_fps=float(args.sample_fps), max_keyframes=int(args.max_keyframes))
    print(f'Standard indexation: {video_path.name} + {points_path.name}')
    indexer.index_video(video_path, points_path)
    print('Retrieval index assembly...')
    rebuild_index(db_path, index_path)
    print('Ready.')


if __name__ == '__main__':
    main()
