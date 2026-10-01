"""P18 D: asks -> routines. The night counts what he asks per half hour; the app gets it ready and says so quietly."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import mentor_review  # noqa: E402
from mentor_app import commands, memory, routines  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import reads_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
SESSION = "2026-09-29"


def _days(n, last="2026-09-29"):
    end = datetime.fromisoformat(last).date()
    return [(end - timedelta(days=i)).isoformat() for i in range(n)][::-1]


def _chat(tmp_path, plan):
    """``plan``: (day, HH:MM PT, [pack names]) asks; each user turn is followed by its reply."""
    path = tmp_path / "mentor_chat.sqlite3"
    store = MentorChatStore(path)
    session = store.start_session("m")
    stamps = []
    for day, hhmm, packs in plan:
        stamps.append(datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=PT).astimezone(timezone.utc))
        store.add_turn(session, "user", "what's the tape?")
        stamps.append(stamps[-1] + timedelta(seconds=20))
        store.add_turn(session, "assistant", "answer",
                       tool_calls=[{"name": name, "arguments": {}, "source": "auto", "dropped": False}
                                   for name in packs] + [{"name": "veto_pack", "dropped": True}])
    with sqlite3.connect(path) as conn:
        for n, stamp in enumerate(stamps, start=1):
            conn.execute("UPDATE turns SET ts_utc = ? WHERE id = ?", (stamp.isoformat(), n))
    return path, store


def test_the_asks_view_has_bucket_weekday_packs_and_kind(tmp_path):
    _path, store = _chat(tmp_path, [("2026-09-29", "06:41", ["regime_pack", "rs_pack"]),
                                    ("2026-09-29", "07:05", ["gate_pack"]), ("2026-09-29", "07:20", [])])
    asks = store.asks()
    assert [(a["bucket_pt"], a["weekday"], a["packs"], a["kind"]) for a in asks] == [
        ("06:30", "Tue", ["regime_pack", "rs_pack"], "tape"), ("07:00", "Tue", ["gate_pack"], "pre_trade"),
        ("07:00", "Tue", [], "chat")]


def test_a_routine_is_a_pack_asked_on_five_of_the_last_ten_session_days():
    asks = [{"day_pt": day, "bucket_pt": "06:30", "packs": ["regime_pack"], "turn_id": n}
            for n, day in enumerate(_days(5))]
    asks += [{"day_pt": day, "bucket_pt": "06:30", "packs": ["rs_pack"], "turn_id": 99} for day in _days(4)]
    asks += [{"day_pt": day, "bucket_pt": "09:00", "packs": ["journal_pack"], "turn_id": 50} for day in _days(12)]
    table = routines.find_routines(asks, SESSION)
    assert table["session_days"] == _days(10)
    assert table["routines"] == [{"bucket": "06:30", "packs": [{"name": "regime_pack", "days": 5}]},
                                 {"bucket": "09:00", "packs": [{"name": "journal_pack", "days": 10}]}]
    old = routines.find_routines(asks[:5], "2026-09-29")
    assert routines.routine_line(table, old) == "You usually ask regime_pack at 06:30 PT; journal_pack at 09:00 PT"
    assert routines.routine_line(table, table) == ""


def test_the_night_writes_the_routine_file_and_the_brief_line_only_when_it_changes(tmp_path):
    plan = [(day, "06:35", ["regime_pack"]) for day in _days(5)]
    path, _store = _chat(tmp_path, plan)

    def review(**kw):
        return mentor_review.run_mentor_review(
            session_date=SESSION, now=datetime(2026, 9, 29, 23, 0, tzinfo=PT), chat_db=path, ai_root=tmp_path / "ai",
            veto_outcomes=tmp_path / "veto.csv", journal_db=tmp_path / "none.sqlite3", ask=False,
            reads_sources=reads_pack.Sources(entries=lambda: [], grades=lambda: []), **kw)

    assert review()["status"] == "ok"
    table = json.loads((tmp_path / "ai" / "mentor_routines.json").read_text(encoding="utf-8"))
    assert table["routines"] == [{"bucket": "06:30", "packs": [{"name": "regime_pack", "days": 5}]}]
    brief_file = tmp_path / "ai" / f"mentor_coach_brief_{SESSION}.json"
    brief = json.loads(brief_file.read_text(encoding="utf-8"))
    assert brief["routine"] == "You usually ask regime_pack at 06:30 PT"
    brief_file.unlink()
    review()  # a rerun of the same night keeps its line
    assert json.loads(brief_file.read_text(encoding="utf-8"))["routine"] == "You usually ask regime_pack at 06:30 PT"
    texts = [item.text for item in memory.coach_items({"session_date": SESSION, "routine": brief["routine"]})]
    assert texts == ["routine: You usually ask regime_pack at 06:30 PT"]
    # The next night, nothing changed: no line.
    nxt = mentor_review.run_mentor_review(
        session_date="2026-09-30", now=datetime(2026, 9, 30, 23, 0, tzinfo=PT), chat_db=path, ai_root=tmp_path / "ai",
        veto_outcomes=tmp_path / "veto.csv", journal_db=tmp_path / "none.sqlite3", ask=False,
        reads_sources=reads_pack.Sources(entries=lambda: [], grades=lambda: []))
    assert nxt["status"] in ("ok", "skipped")
    later = tmp_path / "ai" / "mentor_coach_brief_2026-09-30.json"
    if later.exists():
        assert json.loads(later.read_text(encoding="utf-8"))["routine"] == ""


def test_routine_commands_parse():
    assert commands.handle("/routine") == commands.CommandResult("routine")
    assert commands.handle("/forget routine 6:30") == commands.CommandResult("forget_routine", "", "06:30")
    assert commands.handle("/forget routine 06:45").action == "error"
    assert commands.handle("/forget 12").action == "forget"  # a note id still works
    assert "/routine" in commands.HELP_TEXT


# ------------------------------------------------------------------ the window
@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    table = {"schema": "mentor_routines_v1", "session_date": SESSION, "session_days": _days(10), "min_days": 5,
             "routines": [{"bucket": "06:30", "packs": [{"name": "habits_pack", "days": 6}]}]}
    path = tmp_path / "mentor_routines.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    clock = {"now": datetime(2026, 9, 30, 6, 31, tzinfo=PT)}
    queue = PrefetchQueue()
    win = MentorWindow(store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=queue,
                       stream_post=lambda url, payload, cancelled: [], post=lambda url, payload, timeout: {},
                       routines_path=path, now=lambda: clock["now"])
    win.clock = clock
    win._io.submit(win._load_routines).result(5)
    yield win
    win.shutdown()
    win.deleteLater()


def _wait(win, test, seconds=5.0):
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if test():
            return True
        time.sleep(0.02)
    return False


def test_the_window_prefetches_the_buckets_packs_once_and_shows_a_quiet_chip(window):
    inbox_before = len(window.inbox.items())
    window.maybe_prefetch_routine()
    assert window.queue.run_one()  # one job on the prefetch queue (refresh priority)
    assert _wait(window, lambda: window.routine_chip.isVisibleTo(window))
    assert window.routine_chip.text() == "usual 06:30 read ready"
    assert len(window.inbox.items()) == inbox_before  # never the Inbox
    window.maybe_prefetch_routine()  # once per bucket
    assert window.queue.run_one() is False
    window._show_routine()
    assert "Habits" in window.transcript.toPlainText()
    assert not window.routine_chip.isVisibleTo(window)
    window.clock["now"] = datetime(2026, 9, 30, 7, 1, tzinfo=PT)  # no routine at 07:00
    window.maybe_prefetch_routine()
    assert window._routine_bucket_done == "06:30"


def test_forget_routine_persists_and_routine_prints_the_table(window):
    window.send("/routine")
    assert "06:30 PT: habits_pack (6 days)" in window.transcript.toPlainText()
    window.send("/forget routine 06:30")
    window._io.submit(lambda: None).result(5)
    assert routines.forgotten_list(window.store.get_state(routines.FORGOTTEN_KEY)) == ["06:30"]
    window.maybe_prefetch_routine()
    assert window._routine_bucket_done == ""  # forgotten: nothing prefetched
    window._io.submit(window._load_routines).result(5)
    assert window._routine_forgotten == ["06:30"]
