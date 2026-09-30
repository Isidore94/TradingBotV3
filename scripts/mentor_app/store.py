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
    graded_utc TEXT,
    outcome_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profile_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    retired_utc TEXT,
    checked_utc TEXT,
    asked_utc TEXT
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
#: Columns added to ``profile_notes`` after Phase 3 (retire, "still true?" check and ask).
NOTE_COLUMNS = ("retired_utc", "checked_utc", "asked_utc")
#: app_state key for one PT day's service counters (uncited numbers, brain-offline minutes).
DAY_STATS_KEY = "stats:{day}"
_CHALLENGES_TABLE = SCHEMA[SCHEMA.index("CREATE TABLE IF NOT EXISTS challenges"):].split(";", 1)[0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(list(value or ()) if not isinstance(value, dict) else value, default=str, sort_keys=True)


def _args_key(args: dict[str, Any] | None) -> str:
    return json.dumps(dict(args or {}), sort_keys=True, default=str)


def _copy_old_challenges(conn: sqlite3.Connection) -> None:
    """Copy every row of ``challenges_old`` into the nullable table ('' becomes NULL), then drop it."""
    conn.execute(
        "INSERT OR IGNORE INTO challenges (id, kind, symbol, claim, evidence_ids_json, issued_utc, graded_utc, "
        "outcome_json) SELECT id, kind, symbol, claim, evidence_ids_json, issued_utc, NULLIF(graded_utc, ''), "
        "outcome_json FROM challenges_old"
    )
    conn.execute("DROP TABLE challenges_old")


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None


def _open_graded_column(conn: sqlite3.Connection) -> None:
    """Phase 0-2 created ``graded_utc`` NOT NULL: rebuild the table nullable, keeping every row.

    One explicit transaction (sqlite DDL is transactional), rolled back whole on any failure, so a
    failed open leaves the old table as it was and the next open tries again. A ``challenges_old``
    left by an interrupted run is finished first.
    """
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if _has_table(conn, "challenges_old"):
            conn.execute(_CHALLENGES_TABLE)
            _copy_old_challenges(conn)
        else:
            columns = {row[1]: row for row in conn.execute("PRAGMA table_info(challenges)").fetchall()}
            graded = columns.get("graded_utc")
            if graded is not None and graded[3]:
                conn.execute("ALTER TABLE challenges RENAME TO challenges_old")
                conn.execute(_CHALLENGES_TABLE)
                _copy_old_challenges(conn)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _add_note_columns(conn: sqlite3.Connection) -> None:
    """Add the Phase 4 note columns to an older ``profile_notes`` (nullable, rows kept)."""
    have = {row[1] for row in conn.execute("PRAGMA table_info(profile_notes)").fetchall()}
    for name in NOTE_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE profile_notes ADD COLUMN {name} TEXT")
    conn.commit()


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
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(SCHEMA)
                _open_graded_column(conn)
                _add_note_columns(conn)
            except BaseException:
                conn.close()
                raise
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

    def profile_notes(self, *, limit: int = 50, include_retired: bool = False) -> list[dict[str, Any]]:
        """The newest ``limit`` notes, oldest first; a retired note only when asked for."""
        where = "" if include_retired else " WHERE retired_utc IS NULL"
        return self._read(f"SELECT * FROM profile_notes{where} ORDER BY id DESC LIMIT ?", (int(limit),))[::-1]

    def _note_stamp(self, what: str, column: str, note_id: int, when: str = "") -> bool:
        if column not in NOTE_COLUMNS:
            raise ValueError(column)
        if not self._read("SELECT id FROM profile_notes WHERE id = ?", (int(note_id),)):
            return False
        written = self._write(what, f"UPDATE profile_notes SET {column} = ? WHERE id = ?", (when or utc_now(), int(note_id)))
        return written is not None

    def retire_note(self, note_id: int) -> bool:
        """``/forget``: the note stays in the table, marked retired; it is never deleted."""
        return self._note_stamp("note retire", "retired_utc", note_id)

    def check_note(self, note_id: int) -> bool:
        """``/keep``: the trader says the note is still true; its age restarts."""
        return self._note_stamp("note keep", "checked_utc", note_id)

    def mark_note_asked(self, note_id: int, when: str = "") -> bool:
        return self._note_stamp("note asked", "asked_utc", note_id, when)

    def search_text(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Plain substring search over turns and live notes, newest first (recall with the brain down)."""
        like = f"%{str(query or '').strip()}%"
        return self._read(
            "SELECT 'note' AS kind, id AS ref_id, text, ts_utc FROM profile_notes "
            "WHERE retired_utc IS NULL AND text LIKE ? "
            "UNION ALL SELECT 'turn' AS kind, id AS ref_id, text, ts_utc FROM turns "
            "WHERE role IN ('user', 'assistant') AND text LIKE ? ORDER BY ts_utc DESC LIMIT ?",
            (like, like, int(limit)),
        )

    # ----------------------------------------------------------------- day stats
    def bump_day_stats(self, day: str, **counts: float) -> bool:
        """Add to one PT day's service counters (call from the one IO thread: read-modify-write)."""
        key = DAY_STATS_KEY.format(day=day)
        try:
            current = json.loads(self.get_state(key) or "{}")
        except ValueError:
            current = {}
        for name, value in counts.items():
            current[name] = round(float(current.get(name) or 0) + float(value), 3)
        return self.set_state(key, json.dumps(current, sort_keys=True))

    def day_stats(self, day: str) -> dict[str, Any]:
        try:
            return dict(json.loads(self.get_state(DAY_STATS_KEY.format(day=day)) or "{}"))
        except ValueError:
            return {}

    # ----------------------------------------------------------------- challenges
    def add_challenge(
        self,
        challenge_id: str,
        *,
        kind: str,
        symbol: str,
        claim: str,
        evidence_ids: Iterable[str] = (),
        issued_utc: str = "",
        outcome: dict[str, Any] | None = None,
    ) -> bool:
        """Insert one open challenge (``graded_utc`` NULL); an id already issued is left alone."""
        rowid = self._write(
            "challenge",
            "INSERT OR IGNORE INTO challenges (id, kind, symbol, claim, evidence_ids_json, issued_utc, graded_utc, "
            "outcome_json) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
            (str(challenge_id), str(kind), str(symbol or ""), str(claim), _json(evidence_ids),
             issued_utc or utc_now(), json.dumps(dict(outcome or {}), sort_keys=True, default=str)),
        )
        return bool(rowid)

    def challenges(self, *, kind: str | None = None, open_only: bool = False) -> list[dict[str, Any]]:
        """Challenges oldest first; ``open_only`` = not yet graded (NULL, or '' in an old table)."""
        sql = "SELECT * FROM challenges WHERE 1 = 1"
        params: list[Any] = []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if open_only:
            sql += " AND (graded_utc IS NULL OR graded_utc = '')"
        return self._read(sql + " ORDER BY issued_utc, id", params)

    def update_challenge(self, challenge_id: str, *, outcome: dict[str, Any], graded_utc: str | None = None) -> bool:
        written = self._write(
            "challenge grade",
            "UPDATE challenges SET outcome_json = ?, graded_utc = ? WHERE id = ?",
            (json.dumps(dict(outcome), sort_keys=True, default=str), graded_utc, str(challenge_id)),
        )
        return written is not None

    # ----------------------------------------------------------------- app state
    def set_state(self, key: str, value: str) -> bool:
        written = self._write(
            "app state", "INSERT OR REPLACE INTO app_state (key, value, updated_utc) VALUES (?, ?, ?)",
            (str(key), str(value), utc_now()),
        )
        return written is not None

    def get_state(self, key: str) -> str | None:
        rows = self._read("SELECT value FROM app_state WHERE key = ?", (str(key),))
        return str(rows[0]["value"]) if rows else None

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

    def embedded_refs(self, kind: str, model: str) -> set[int]:
        rows = self._read("SELECT ref_id FROM embeddings WHERE kind = ? AND model = ?", (kind, model))
        return {int(row["ref_id"]) for row in rows}

    def unembedded_notes(self, model: str, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._read(
            "SELECT n.id, n.text FROM profile_notes n LEFT JOIN embeddings e "
            "ON e.kind = 'note' AND e.ref_id = n.id AND e.model = ? "
            "WHERE e.ref_id IS NULL AND n.retired_utc IS NULL ORDER BY n.id LIMIT ?",
            (model, int(limit)),
        )

    def unembedded_turns(self, model: str, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._read(
            "SELECT t.id, t.text FROM turns t LEFT JOIN embeddings e "
            "ON e.kind = 'turn' AND e.ref_id = t.id AND e.model = ? "
            "WHERE e.ref_id IS NULL AND t.role IN ('user', 'assistant') AND length(t.text) > 0 "
            "ORDER BY t.id LIMIT ?",
            (model, int(limit)),
        )
