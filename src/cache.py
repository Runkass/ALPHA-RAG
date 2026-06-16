"""SQLite cache for LLM answers (thread-safe, WAL, batched writes)."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path


class AnswerCache:
    def __init__(self, db_path: Path, *, batch_size: int = 50) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._batch_size = batch_size
        self._local = threading.local()
        self._pending = 0
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=60000;")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS answers (
                cache_key TEXT PRIMARY KEY,
                q_id INTEGER,
                answer TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_answers_q_id ON answers(q_id)")
        conn.commit()

    @staticmethod
    def make_key(q_id: int, query: str, context: str) -> str:
        payload = f"{q_id}|{query}|{context}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> str | None:
        row = self._connect().execute(
            "SELECT answer FROM answers WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return row[0] if row else None

    def set(self, cache_key: str, q_id: int, answer: str, *, flush: bool = False) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO answers (cache_key, q_id, answer) VALUES (?, ?, ?)",
            (cache_key, q_id, answer),
        )
        with self._lock:
            self._pending += 1
            need_flush = flush or self._pending >= self._batch_size
            if need_flush:
                self._pending = 0
        if need_flush:
            conn.commit()

    def flush(self) -> None:
        with self._lock:
            self._pending = 0
        self._connect().commit()

    def has_q_id(self, q_id: int) -> bool:
        row = self._connect().execute(
            "SELECT 1 FROM answers WHERE q_id = ? LIMIT 1",
            (q_id,),
        ).fetchone()
        return row is not None

    def get_by_q_id(self, q_id: int) -> str | None:
        row = self._connect().execute(
            "SELECT answer FROM answers WHERE q_id = ? ORDER BY rowid DESC LIMIT 1",
            (q_id,),
        ).fetchone()
        return row[0] if row else None

    def count_distinct_q_ids(self) -> int:
        row = self._connect().execute(
            "SELECT COUNT(DISTINCT q_id) FROM answers"
        ).fetchone()
        return int(row[0]) if row else 0

    def delete_by_q_id(self, q_id: int) -> None:
        self._connect().execute("DELETE FROM answers WHERE q_id = ?", (q_id,))
        self._connect().commit()

    def delete_by_q_ids(self, q_ids: list[int]) -> int:
        if not q_ids:
            return 0
        conn = self._connect()
        conn.executemany("DELETE FROM answers WHERE q_id = ?", [(q,) for q in q_ids])
        conn.commit()
        return len(q_ids)

    def close(self) -> None:
        self.flush()
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
