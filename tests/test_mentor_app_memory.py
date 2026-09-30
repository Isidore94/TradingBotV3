"""Mentor app P4: morning memory (byte-stable block, budget), /memory /forget /keep /recall, and the
once-a-week "still true?" Inbox item."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import memory  # noqa: E402
from mentor_app.chat_model import SYSTEM_PROMPT, ChatModel  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 29, 10, 0, tzinfo=PT)  # a Tuesday session
OLD = "2026-09-15T17:00:00.000+00:00"  # 14 days before NOW


def _digest(root: Path, day: str, items: list[str], questions: list[str] = ()) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"session_date": day, "digest": [{"text": text, "evidence_refs": ["fact:turns"]} for text in items],
               "open_questions": [{"text": text, "evidence_refs": ["fact:turns"]} for text in questions]}
    (root / f"mentor_day_digest_{day}.json").write_text(json.dumps(payload), encoding="utf-8")


def _note(store: MentorChatStore, text: str, stamp: str = OLD) -> int:
    note_id = store.add_profile_note(text)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE profile_notes SET ts_utc = ? WHERE id = ?", (stamp, note_id))
    return int(note_id)


# ---------------------------------------------------------------- the block
def test_the_memory_block_is_byte_stable_and_sits_between_rules_and_context(tmp_path):
    root = tmp_path / "ai"
    _digest(root, "2026-09-28", ["Asked about NVDA twice."], ["Is two losses still the stop?"])
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    _note(store, "rule: I stop after two losses")
    one = memory.load(store, ai_root=root)
    two = memory.load(store, ai_root=root)
    assert one.text == two.text and one.text.encode() == two.text.encode()
    assert "[mem:digest:2026092800] (2026-09-28) Asked about NVDA twice." in one.text
    assert "[mem:digest:2026092850] (2026-09-28) open question: Is two losses still the stop?" in one.text
    assert "[mem:note:1] (2026-09-15) rule: I stop after two losses" in one.text
    body = ChatModel.system_message("[ctx:auto_mode] DESK", one.text)["content"]
    assert body.startswith(SYSTEM_PROMPT) and body.index("# Memory") < body.index("# Desk context")
    first = ChatModel().messages(context_text="[ctx:a] x", memory_block=one.text)[0]
    assert first == ChatModel().messages(context_text="[ctx:a] x", memory_block=two.text)[0]


def test_the_budget_drops_the_oldest_first_and_is_deterministic():
    items = [memory.MemoryItem(f"mem:note:{i}", "note", i, f"2026-09-{i:02d}", "x" * 200) for i in range(1, 21)]
    one = memory.render(items, budget_tokens=300)
    two = memory.render(list(reversed(items)), budget_tokens=300)
    assert one.text == two.text and one.dropped == two.dropped > 0
    kept = [item.ref_id for item in one.items]
    assert kept == sorted(kept) and kept[-1] == 20 and 1 not in kept, "newest kept, oldest dropped"
    assert memory.estimate_tokens(one.text) <= 300


def test_only_the_last_five_digests_are_loaded(tmp_path):
    root = tmp_path / "ai"
    for day in range(20, 28):
        _digest(root, f"2026-09-{day}", [f"day {day}"])
    loaded = memory.load(MentorChatStore(tmp_path / "chat.sqlite3"), ai_root=root)
    assert [item.day for item in loaded.items] == [f"2026-09-{day}" for day in range(23, 28)]


def test_a_night_digest_becomes_morning_memory(tmp_path):
    from ai_jobs import mentor_review

    store = MentorChatStore(tmp_path / "chat.sqlite3")
    store.add_turn(store.start_session(), "user", "NVDA?")
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE turns SET ts_utc = '2026-09-29T17:00:00.000+00:00'")
    reply = {"digest": [{"text": "He asked about NVDA.", "evidence_refs": ["turn:1"]}], "open_questions": []}
    mentor_review.run_mentor_review(session_date="2026-09-29", now=datetime(2026, 9, 29, 23, 0, tzinfo=PT),
                                    chat_db=store.path, ai_root=tmp_path / "ai",
                                    request=lambda **_: {"summary": reply, "model": "m"})
    loaded = memory.load(store, ai_root=tmp_path / "ai")
    assert "[mem:digest:2026092900] (2026-09-29) He asked about NVDA." in loaded.text


def test_still_true_candidate_rules():
    rows = [
        {"id": 1, "text": "I like NVDA", "ts_utc": OLD},
        {"id": 2, "text": "rule: two losses and I stop", "ts_utc": OLD, "retired_utc": "x"},
        {"id": 3, "text": "Rule: no trades before 06:45", "ts_utc": OLD},
        {"id": 4, "text": "rule: fresh one", "ts_utc": "2026-09-28T17:00:00+00:00"},
    ]
    assert memory.still_true_candidate(rows, NOW)["id"] == 3
    rows[2]["asked_utc"] = "2026-09-27T17:00:00+00:00"
    assert memory.still_true_candidate(rows, NOW) is None, "asked within 7 days"
    rows[2]["asked_utc"] = None
    rows[2]["checked_utc"] = "2026-09-26T17:00:00+00:00"
    assert memory.still_true_candidate(rows, NOW) is None, "/keep restarted its age"


# ---------------------------------------------------------------- the window
@pytest.fixture(scope="module")
def app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(app, tmp_path, monkeypatch):
    from mentor_app import settings
    from mentor_app.inbox import Inbox
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    clock = {"now": NOW}
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        inbox=Inbox(per_day_cap=6, now=lambda: clock["now"]),
        stream_post=lambda url, payload, cancelled: [],
        post=lambda url, payload, timeout: {},
        now=lambda: clock["now"],
        mentor_enabled=False,
        liked_source=lambda: [],
        memory_root=tmp_path / "ai",
    )
    win.clock = clock
    yield win
    win.shutdown()
    win.deleteLater()


def _drain(win, app):
    for _ in range(50):
        win._io.submit(lambda: None).result(5)
        app.processEvents()
        if not win.queue.run_one():
            app.processEvents()
            if not win.queue.pending():
                break
    win._io.submit(lambda: None).result(5)
    app.processEvents()


def _text(win):
    return win.transcript.toPlainText()


def test_forget_retires_a_note_and_never_deletes_it(window, app):
    note_id = _note(window.store, "rule: I stop after two losses")
    window.send(f"/forget mem:note:{note_id}")
    _drain(window, app)
    rows = window.store.profile_notes(include_retired=True)
    assert len(rows) == 1 and rows[0]["retired_utc"], "kept, marked retired"
    assert window.store.profile_notes() == []
    assert "Retired [mem:note:1]" in _text(window)


def test_memory_command_lists_what_was_loaded_with_ids(window, app):
    _note(window.store, "rule: I stop after two losses")
    _digest(window._memory_root, "2026-09-28", ["Asked about NVDA twice."])
    window._submit_io(window._open_session)
    _drain(window, app)
    window.send("/memory")
    text = _text(window)
    assert "[mem:note:1]" in text and "[mem:digest:2026092800]" in text
    assert "# Memory" in window._memory_block


def test_recall_answers_by_substring_with_the_brain_off(window, app):
    from mentor_packs import recall

    window.install_recall_fallback()  # what start_background does; the brain stays off
    _note(window.store, "I stop trading after two losses")
    recall.set_searcher(None)
    window.send("/recall losses")
    _drain(window, app)
    assert "[mem:note:1]" in _text(window)


def test_still_true_posts_once_per_note_per_seven_days(window, app):
    _note(window.store, "rule: I stop after two losses")
    window.send("/help")
    _drain(window, app)
    items = [item for item in window.inbox.items() if item.kind == "memory"]
    assert len(items) == 1 and items[0].text.startswith("You said: rule: I stop after two losses. Still true?")
    assert "(/keep 1 | /forget 1)" in items[0].text
    assert window.store.profile_notes()[0]["asked_utc"]
    window.send("/help")  # same day: not again
    window.clock["now"] = NOW + timedelta(days=1)  # Wednesday: asked yesterday
    window.send("/help")
    _drain(window, app)
    assert len([item for item in window.inbox.items() if item.kind == "memory"]) == 1
    window.clock["now"] = NOW + timedelta(days=8)  # the next Wednesday: a week later, ask again
    window.send("/help")
    _drain(window, app)
    assert len([item for item in window.inbox.items() if item.kind == "memory"]) == 2


def test_still_true_never_posts_when_the_cap_is_used(window, app):
    _note(window.store, "rule: I stop after two losses")
    window.inbox.per_day_cap = 0
    window.send("/help")
    _drain(window, app)
    assert window.inbox.items() == []
    assert window.store.profile_notes()[0]["asked_utc"] is None, "not asked, so it may ask another day"


def test_keep_refreshes_the_note(window, app):
    _note(window.store, "rule: I stop after two losses")
    window.send("/keep 1")
    _drain(window, app)
    assert window.store.profile_notes()[0]["checked_utc"]
    assert "Still true: [mem:note:1]" in _text(window)


@pytest.mark.parametrize("wall", ["2026-01-05T12:00:00.000+00:00", "2027-06-01T12:00:00.000+00:00"])
def test_still_true_uses_the_windows_clock_on_any_date(window, app, monkeypatch, wall):
    from mentor_app import store as store_module

    monkeypatch.setattr(store_module, "utc_now", lambda: wall)
    _note(window.store, "rule: I stop after two losses")
    window.send("/help")
    _drain(window, app)
    assert window.store.profile_notes()[0]["asked_utc"].startswith("2026-09-29")
    window.clock["now"] = NOW + timedelta(days=8)
    window.send("/help")
    _drain(window, app)
    assert len([item for item in window.inbox.items() if item.kind == "memory"]) == 2
