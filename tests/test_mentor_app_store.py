"""Trade Mentor chat store: round trip, WAL, tz-aware stamps, a failed write never raises."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import project_paths  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402


def test_the_db_lives_under_the_persistent_runtime_dir():
    assert project_paths.MENTOR_CHAT_DB_FILE.name == "mentor_chat.sqlite3"
    assert project_paths.MENTOR_CHAT_DB_FILE.parent == project_paths.PERSISTENT_RUNTIME_DATA_DIR
    assert MentorChatStore().path == project_paths.MENTOR_CHAT_DB_FILE


def test_turn_round_trip_with_tz_aware_utc_stamps(tmp_path):
    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    session = store.start_session("gemma3:12b")
    assert session
    store.add_turn(session, "user", "Is NVDA still worth it?")
    store.add_turn(
        session,
        "assistant",
        "Auto is DESK [ctx:auto_mode].",
        pack_ids=["ctx:auto_mode"],
        model="gemma3:12b",
        latency_ms=812,
        prompt_tokens=900,
        completion_tokens=40,
        tool_calls=[{"name": "context_pack", "arguments": {}}],
    )
    rows = store.turns(session)
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[1]["latency_ms"] == 812 and '"ctx:auto_mode"' in rows[1]["pack_ids_json"]
    assert "context_pack" in rows[1]["tool_calls_json"]
    for row in rows:
        assert datetime.fromisoformat(row["ts_utc"]).utcoffset().total_seconds() == 0


def test_the_store_runs_in_wal_mode(tmp_path):
    db = tmp_path / "mentor_chat.sqlite3"
    MentorChatStore(db).start_session()
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "turns", "challenges", "profile_notes", "pack_cache", "embeddings"} <= tables


def test_a_failed_turn_write_is_logged_and_returns_none(tmp_path, caplog):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    store = MentorChatStore(blocker / "mentor_chat.sqlite3")
    assert store.add_turn(1, "user", "hello") is None
    assert store.turns(1) == []
    assert "write failed" in caplog.text


def test_profile_notes_pack_cache_and_embeddings(tmp_path):
    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    store.add_profile_note("I stop after two losses")
    assert [row["text"] for row in store.profile_notes()] == ["I stop after two losses"]
    store.put_pack("context_pack", {}, '{"rows": []}')
    store.put_pack("context_pack", {}, '{"rows": [1]}')
    assert store.get_pack("context_pack")["pack_json"] == '{"rows": [1]}'
    session = store.start_session()
    turn = store.add_turn(session, "user", "NVDA")
    assert [row["id"] for row in store.unembedded_turns("nomic-embed-text")] == [turn]
    store.put_embedding("turn", turn, "nomic-embed-text", [0.5, 0.25], text="NVDA")
    got = store.embeddings("nomic-embed-text")
    assert got[0]["vector"] == [0.5, 0.25] and got[0]["text"] == "NVDA"
    assert store.unembedded_turns("nomic-embed-text") == []


def test_threads_that_open_a_fresh_store_together_all_get_in(tmp_path):
    """The window's IO thread and the frontier thread may both be first; one sets up, the others wait."""
    import threading

    for round_no in range(20):
        store = MentorChatStore(tmp_path / f"chat{round_no}.sqlite3")
        gate = threading.Barrier(6)
        spent: list = []

        def read(store=store, gate=gate, spent=spent):
            gate.wait()
            spent.append(store.frontier_spent("2026-09-30"))

        def write(store=store, gate=gate):
            gate.wait()
            store.start_session("m")

        threads = [threading.Thread(target=read) for _ in range(4)] + [threading.Thread(target=write)
                                                                         for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert spent == [0.0] * 4, f"round {round_no}: a first open lost the race ({spent})"
