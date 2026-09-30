"""MentorChatStore: the Trade Mentor app's own sqlite store (single owner: the app).

Chat turns, sessions, challenges (Phase 2+), profile notes, pack cache and
embeddings. WAL + busy_timeout so the night can read it ``mode=ro`` while the app
writes. Timestamps are UTC, tz-aware ISO. A failed write is logged and returns
None: it loses the turn, never raises into the UI, and never touches the journal.
Call it from a worker thread; each call opens its own connection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from array import array
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_utc TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    ts_utc TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    pack_ids_json TEXT NOT NULL DEFAULT '[]',
    model TEXT NOT NULL DEFAULT '',
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    tool_calls_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns(session_id, id);
CREATE TABLE IF NOT EXISTS challenges (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    symbol TEXT NOT NULL DEFAULT '',
    claim TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    issued_utc TEXT NOT NULL,
    graded_utc TEXT NOT NULL DEFAULT '',
    outcome_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS profile_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS pack_cache (
    name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    built_utc TEXT NOT NULL,
    pack_json TEXT NOT NULL,
    PRIMARY KEY (name, args_json)
);
CREATE TABLE IF NOT EXISTS embeddings (
    kind TEXT NOT NULL,
    ref_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    vector_blob BLOB NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (kind, ref_id, model)
);
"""

BUSY_TIMEOUT_MS = 5000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(list(value or ()) if not isinstance(value, dict) else value, default=str, sort_keys=True)


def _args_key(args: dict[str, Any] | None) -> str:
    return json.dumps(dict(args or {}), sort_keys=True, default=str)


class MentorChatStore:
    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            from project_paths import MENTOR_CHAT_DB_FILE

            path = MENTOR_CHAT_DB_FILE
        self.path = Path(path)
        self._ready = False

    # ----------------------------------------------------------------- plumbing
    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        if not self._ready:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            self._ready = True
        return conn

    def _write(self, what: str, sql: str, params: Sequence[Any]) -> int | None:
        try:
            with closing(self._connect()) as conn, conn:
                cursor = conn.execute(sql, tuple(params))
                return int(cursor.lastrowid or 0)
        except Exception:  # noqa: BLE001 - a lost chat row is logged, never raised into the UI
            logging.exception("Trade Mentor store: %s write failed (%s)", what, self.path)
            return None

    def _read(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        try:
            with closing(self._connect()) as conn:
                return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]
        except Exception:  # noqa: BLE001
            logging.exception("Trade Mentor store: read failed (%s)", self.path)
            return []

    # ----------------------------------------------------------------- sessions and turns
    def start_session(self, model: str = "") -> int | None:
        return self._write("session", "INSERT INTO sessions (started_utc, model) VALUES (?, ?)", (utc_now(), model))

    def add_turn(
        self,
        session_id: int | None,
        role: str,
        text: str,
        *,
        pack_ids: Iterable[str] = (),
        model: str = "",
        latency_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        tool_calls: Iterable[Any] = (),
    ) -> int | None:
        return self._write(
            "turn",
            "INSERT INTO turns (session_id, ts_utc, role, text, pack_ids_json, model, latency_ms, "
            "prompt_tokens, completion_tokens, tool_calls_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                utc_now(),
                str(role),
                str(text),
                _json(pack_ids),
                str(model or ""),
                latency_ms,
                prompt_tokens,
                completion_tokens,
                _json(tool_calls),
            ),
        )

    def turns(self, session_id: int | None = None, *, limit: int = 200) -> list[dict[str, Any]]:
        if session_id is None:
            return self._read("SELECT * FROM turns ORDER BY id DESC LIMIT ?", (int(limit),))[::-1]
        return self._read(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, int(limit))
        )[::-1]

    # ----------------------------------------------------------------- memory
    def add_profile_note(self, text: str, source: str = "remember") -> int | None:
        return self._write(
            "profile note", "INSERT INTO profile_notes (ts_utc, text, source) VALUES (?, ?, ?)", (utc_now(), text, source)
        )

    def profile_notes(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._read("SELECT * FROM profile_notes ORDER BY id DESC LIMIT ?", (int(limit),))[::-1]

    # ----------------------------------------------------------------- caches
    def put_pack(self, name: str, args: dict[str, Any] | None, pack_json: str, built_utc: str = "") -> int | None:
        return self._write(
            "pack cache",
            "INSERT OR REPLACE INTO pack_cache (name, args_json, built_utc, pack_json) VALUES (?, ?, ?, ?)",
            (name, _args_key(args), built_utc or utc_now(), pack_json),
        )

    def get_pack(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any] | None:
        rows = self._read(
            "SELECT * FROM pack_cache WHERE name = ? AND args_json = ?", (name, _args_key(args))
        )
        return rows[0] if rows else None

    def put_embedding(self, kind: str, ref_id: int, model: str, vector: Sequence[float], text: str = "") -> int | None:
        blob = array("f", [float(x) for x in vector]).tobytes()
        return self._write(
            "embedding",
            "INSERT OR REPLACE INTO embeddings (kind, ref_id, model, vector_blob, text) VALUES (?, ?, ?, ?, ?)",
            (kind, int(ref_id), model, blob, text),
        )

    def embeddings(self, model: str) -> list[dict[str, Any]]:
        rows = self._read("SELECT kind, ref_id, text, vector_blob FROM embeddings WHERE model = ?", (model,))
        out = []
        for row in rows:
            vector = array("f")
            vector.frombytes(row.pop("vector_blob"))
            out.append({**row, "vector": list(vector)})
        return out

    def unembedded_turns(self, model: str, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._read(
            "SELECT t.id, t.text FROM turns t LEFT JOIN embeddings e "
            "ON e.kind = 'turn' AND e.ref_id = t.id AND e.model = ? "
            "WHERE e.ref_id IS NULL AND t.role IN ('user', 'assistant') AND length(t.text) > 0 "
            "ORDER BY t.id LIMIT ?",
            (model, int(limit)),
        )
