from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            fps REAL NOT NULL,
            note TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS keyframes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            frame_idx INTEGER NOT NULL,
            ts_ms REAL NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            image_path TEXT NOT NULL,
            FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_keyframes_video_ts ON keyframes(video_id, ts_ms);

        CREATE TABLE IF NOT EXISTS vpr_embeddings (
            keyframe_id INTEGER PRIMARY KEY,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY(keyframe_id) REFERENCES keyframes(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS local_features (
            keyframe_id INTEGER PRIMARY KEY,
            kpts BLOB NOT NULL,
            desc BLOB NOT NULL,
            scores BLOB NOT NULL,
            image_w INTEGER NOT NULL,
            image_h INTEGER NOT NULL,
            FOREIGN KEY(keyframe_id) REFERENCES keyframes(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS px_to_enu (
            keyframe_id INTEGER PRIMARY KEY,
            a11 REAL NOT NULL,
            a12 REAL NOT NULL,
            a21 REAL NOT NULL,
            a22 REAL NOT NULL,
            valid INTEGER NOT NULL,
            FOREIGN KEY(keyframe_id) REFERENCES keyframes(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def upsert_video(conn: sqlite3.Connection, path: str, fps: float, note: str = "") -> int:
    conn.execute(
        "INSERT INTO videos(path, fps, note) VALUES(?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET fps=excluded.fps, note=excluded.note;",
        (path, float(fps), note),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM videos WHERE path=?;", (path,)).fetchone()
    if row is None:
        raise RuntimeError("Не удалось получить video_id после upsert.")
    return int(row["id"])


def clear_index_data(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM videos;")
    conn.commit()


def fetch_video_by_path(conn: sqlite3.Connection, path: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT id, path, fps, note FROM videos WHERE path=?;",
        (str(path),),
    ).fetchone()


def delete_video_by_path(conn: sqlite3.Connection, path: str) -> Optional[int]:
    row = fetch_video_by_path(conn, path)
    if row is None:
        return None
    video_id = int(row["id"])
    conn.execute("DELETE FROM videos WHERE id=?;", (video_id,))
    conn.commit()
    return video_id


def insert_keyframe(
    conn: sqlite3.Connection,
    video_id: int,
    frame_idx: int,
    ts_ms: float,
    lat: float,
    lng: float,
    image_path: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO keyframes(video_id, frame_idx, ts_ms, lat, lng, image_path) VALUES(?,?,?,?,?,?);",
        (video_id, int(frame_idx), float(ts_ms), float(lat), float(lng), image_path),
    )
    conn.commit()
    return int(cur.lastrowid)


def upsert_embedding(conn: sqlite3.Connection, keyframe_id: int, emb: bytes, dim: int) -> None:
    conn.execute(
        "INSERT INTO vpr_embeddings(keyframe_id, dim, embedding) VALUES(?,?,?) "
        "ON CONFLICT(keyframe_id) DO UPDATE SET dim=excluded.dim, embedding=excluded.embedding;",
        (int(keyframe_id), int(dim), emb),
    )
    conn.commit()


def upsert_local_features(
    conn: sqlite3.Connection,
    keyframe_id: int,
    kpts: bytes,
    desc: bytes,
    scores: bytes,
    image_w: int,
    image_h: int,
) -> None:
    conn.execute(
        "INSERT INTO local_features(keyframe_id, kpts, desc, scores, image_w, image_h) VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(keyframe_id) DO UPDATE SET kpts=excluded.kpts, desc=excluded.desc, scores=excluded.scores, image_w=excluded.image_w, image_h=excluded.image_h;",
        (int(keyframe_id), kpts, desc, scores, int(image_w), int(image_h)),
    )
    conn.commit()


def upsert_px_to_enu(
    conn: sqlite3.Connection,
    keyframe_id: int,
    a11: float,
    a12: float,
    a21: float,
    a22: float,
    valid: bool,
) -> None:
    conn.execute(
        "INSERT INTO px_to_enu(keyframe_id, a11,a12,a21,a22, valid) VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(keyframe_id) DO UPDATE SET a11=excluded.a11,a12=excluded.a12,a21=excluded.a21,a22=excluded.a22,valid=excluded.valid;",
        (int(keyframe_id), float(a11), float(a12), float(a21), float(a22), int(bool(valid))),
    )
    conn.commit()


def fetch_embedding_matrix(conn: sqlite3.Connection) -> Tuple[List[int], int, List[bytes]]:
    rows = conn.execute("SELECT keyframe_id, dim, embedding FROM vpr_embeddings ORDER BY keyframe_id ASC;").fetchall()
    if not rows:
        raise RuntimeError("В БД нет embedding'ов. Запустите индексацию.")
    ids = [int(r["keyframe_id"]) for r in rows]
    dim = int(rows[0]["dim"])
    blobs = [bytes(r["embedding"]) for r in rows]
    return ids, dim, blobs


def fetch_keyframe_meta(conn: sqlite3.Connection, keyframe_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, video_id, frame_idx, ts_ms, lat, lng, image_path FROM keyframes WHERE id=?;",
        (int(keyframe_id),),
    ).fetchone()
    if row is None:
        raise KeyError(f"Keyframe {keyframe_id} не найден.")
    return row


def fetch_all_keyframe_meta(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return list(conn.execute("SELECT id, video_id, frame_idx, ts_ms, lat, lng, image_path FROM keyframes ORDER BY id ASC;").fetchall())


def fetch_local_features(conn: sqlite3.Connection, keyframe_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT kpts, desc, scores, image_w, image_h FROM local_features WHERE keyframe_id=?;",
        (int(keyframe_id),),
    ).fetchone()
    if row is None:
        raise KeyError(f"Local features для keyframe {keyframe_id} не найдены.")
    return row


def fetch_px_to_enu(conn: sqlite3.Connection, keyframe_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT a11,a12,a21,a22,valid FROM px_to_enu WHERE keyframe_id=?;",
        (int(keyframe_id),),
    ).fetchone()



def list_videos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT id, path, fps, note FROM videos ORDER BY id ASC;"
    ).fetchall()
    return list(rows)


def fetch_route_points(
    conn: sqlite3.Connection,
    video_id: int,
    step: int = 10,
) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT id, video_id, ts_ms, lat, lng, image_path "
        "FROM keyframes WHERE video_id=? ORDER BY ts_ms ASC;",
        (int(video_id),),
    ).fetchall()
    if step <= 1:
        return list(rows)
    return [row for i, row in enumerate(rows) if i % int(step) == 0 or i == len(rows) - 1]


def fetch_keyframes_by_video_ts_window(
    conn: sqlite3.Connection,
    video_id: int,
    ts_center_ms: float,
    back_ms: float,
    forward_ms: float,
    limit: int = 80,
) -> list[sqlite3.Row]:
    ts_min = float(ts_center_ms) - float(back_ms)
    ts_max = float(ts_center_ms) + float(forward_ms)
    rows = conn.execute(
        "SELECT id, video_id, frame_idx, ts_ms, lat, lng, image_path "
        "FROM keyframes "
        "WHERE video_id=? AND ts_ms BETWEEN ? AND ? "
        "ORDER BY ABS(ts_ms - ?) ASC, ts_ms ASC "
        "LIMIT ?;",
        (int(video_id), float(ts_min), float(ts_max), float(ts_center_ms), int(limit)),
    ).fetchall()
    return list(rows)


def count_keyframes(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM keyframes;").fetchone()
    return int(row["c"]) if row is not None else 0
