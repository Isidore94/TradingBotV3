"""Mentor app P8: /book. Questrade is read on demand only (/book, /check, 06:20 PT) on the news
thread, reused for 15 min, backed off 1 h after a failure, never with the desk closed; the card
says its source; no model, no Inbox, never an order."""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import questrade_positions as qp  # noqa: E402
from mentor_app import book_jobs, commands, settings  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import book_pack, gate_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
TUE_0700 = datetime(2026, 9, 29, 7, 0, tzinfo=PT)


class _Fetch:
    """A fake ``fetch_book``: counts calls; answers a snapshot stamped at the call time, or a failure."""

    def __init__(self, fail: str = "") -> None:
        self.calls: list[datetime] = []
        self.fail = fail

    def __call__(self, now):
        self.calls.append(now)
        if self.fail:
            return None, self.fail
        return qp.fixture_snapshot(now.astimezone(timezone.utc).isoformat(timespec="seconds")), ""


@pytest.fixture
def store(tmp_path):
    return MentorChatStore(tmp_path / "mentor_chat.sqlite3")


# ---------------------------------------------------------------- the command and the fetch policy
def test_book_command_parses_and_is_in_help():
    assert commands.handle("/book").action == "book"
    assert commands.handle("/book NVDA").action == "error"
    assert "/book" in commands.HELP_TEXT


def test_a_fresh_snapshot_is_reused_for_15_minutes(store):
    fetch = _Fetch()
    assert book_jobs.ensure_book(store, TUE_0700, fetch=fetch)["fetched"]
    assert book_jobs.ensure_book(store, TUE_0700 + timedelta(minutes=15), fetch=fetch)["reason"] == "fresh"
    assert book_jobs.ensure_book(store, TUE_0700 + timedelta(minutes=16), fetch=fetch)["fetched"]
    assert len(fetch.calls) == 2


def test_a_failure_backs_off_one_hour_then_tries_again(store):
    fetch = _Fetch(fail="RuntimeError: 400 Client Error")
    out = book_jobs.ensure_book(store, TUE_0700, fetch=fetch)
    assert out["failed"] and book_jobs.stored_status(store)["reason"] == "RuntimeError: 400 Client Error"
    later = book_jobs.ensure_book(store, TUE_0700 + timedelta(minutes=59), fetch=fetch)
    assert not later["fetched"] and "backing off" in later["reason"] and len(fetch.calls) == 1
    fetch.fail = ""
    assert book_jobs.ensure_book(store, TUE_0700 + timedelta(hours=1), fetch=fetch)["fetched"]
    assert book_jobs.stored_status(store) is None, "a good read clears the failure"


def test_no_token_is_not_a_backoff(store):
    fetch = _Fetch(fail="no token")
    book_jobs.ensure_book(store, TUE_0700, fetch=fetch)
    book_jobs.ensure_book(store, TUE_0700 + timedelta(minutes=1), fetch=fetch)
    assert len(fetch.calls) == 2, "no token sends no request, so a pasted token works at once"


def test_the_desk_closed_means_no_fetch(store):
    fetch = _Fetch()
    out = book_jobs.ensure_book(store, TUE_0700, fetch=fetch, desk_closed=lambda: True)
    assert out == {"fetched": False, "reason": "the desk is closed"} and fetch.calls == []
    assert book_jobs.ensure_book(store, TUE_0700, fetch=fetch, desk_closed=lambda: None)["fetched"], \
        "an unknown desk is not a closed one"


def test_the_morning_fetch_is_once_a_weekday_from_0620():
    schedule = book_jobs.BookSchedule()
    assert not schedule.due(datetime(2026, 9, 29, 6, 19, tzinfo=PT))
    assert schedule.due(datetime(2026, 9, 29, 6, 20, tzinfo=PT))
    schedule.mark(datetime(2026, 9, 29, 6, 20, tzinfo=PT))
    assert not schedule.due(datetime(2026, 9, 29, 9, 0, tzinfo=PT)), "never a periodic poll"
    assert schedule.due(datetime(2026, 9, 30, 6, 20, tzinfo=PT))
    assert not book_jobs.BookSchedule().due(datetime(2026, 10, 3, 7, 0, tzinfo=PT)), "Saturday"


def test_the_card_names_its_source_and_never_orders(store):
    book_jobs.ensure_book(store, TUE_0700, fetch=_Fetch())
    src = book_jobs.store_sources(store, book_pack.fixture_sources())
    text = book_jobs.card_markdown(book_pack.build(now=TUE_0700, sources=src))
    assert text.startswith("**Book**: Source: Questrade positions at Tue 09-29 07:00 PT `[book:source]`")
    assert "`[book:pos:222:AMD]`" in text and "`[book:hint:no_shorts_registered]`" in text
    assert text.rstrip().endswith(book_jobs.FOOTER) and "never orders" in book_jobs.FOOTER


def test_book_jobs_are_qt_free_and_never_touch_an_order_or_the_inbox():
    for name in ("mentor_app/book_jobs.py", "mentor_packs/book_pack.py", "questrade_positions.py"):
        source = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert "PySide6" not in source and "inbox" not in source.lower(), name
        assert "/orders" not in source and "place_order" not in source, name


# ---------------------------------------------------------------- the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    clock = {"now": TUE_0700}
    desk = {"free": False}
    fetch = _Fetch()
    model_calls: list = []
    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    world = gate_pack.fixture_sources(tmp_path / "world")

    def gate_builder(req):
        book = book_jobs.store_sources(store, book_pack.fixture_sources())
        src = dataclasses.replace(world, book_snapshot=book.snapshot, book_status=book.status,
                                  accounts=lambda: book_pack.FIXTURE_JOURNAL_ACCOUNTS)
        return gate_pack.build(req.side, req.symbol, req.size, req.stop, req.entry, now=clock["now"], sources=src)

    window = MentorWindow(
        store=store, stream_post=lambda *a, **k: model_calls.append(a) or [],
        post=lambda *a, **k: model_calls.append(a) or {}, now=lambda: clock["now"], mentor_enabled=False,
        desk_probe=lambda: desk["free"], book_fetch=fetch, book_sources=book_pack.fixture_sources,
        gate_builder=gate_builder, assess_request=lambda **k: model_calls.append(k) or {},
        gate_request=lambda **k: model_calls.append(k) or {},
    )
    window.clock, window.desk, window.fetch, window.model_calls = clock, desk, fetch, model_calls
    yield window
    window.shutdown()
    window.deleteLater()


def _drain(window):
    while window.news_queue.run_one() or window.queue.run_one():
        pass


def test_book_reads_questrade_on_the_news_thread_with_no_model_and_no_inbox(win):
    before = win.inbox.badge()
    win.send("/book")
    assert win.queue.pending() == [] and win.news_queue.pending() == ["book"], "network work: the news thread"
    _drain(win)
    text = win.transcript.toPlainText()
    assert "Source: Questrade positions at Tue 09-29 07:00 PT" in text and "SHORT AMD 100" in text
    assert "never orders" in text and len(win.fetch.calls) == 1
    assert win.model_calls == [] and win.inbox.badge() == before
    win.clock["now"] = TUE_0700 + timedelta(minutes=5)
    win.send("/book")
    _drain(win)
    assert len(win.fetch.calls) == 1, "a snapshot under 15 min old is reused"


def test_book_while_ai_is_paused_still_reads(win):
    import ai_pause

    ai_pause.pause_for("2h", win.clock["now"])
    try:
        win.check_ai_pause()
        win.send("/book")
        _drain(win)
    finally:
        ai_pause.resume()
    assert len(win.fetch.calls) == 1 and "Questrade positions" in win.transcript.toPlainText()


def test_book_with_the_desk_closed_is_the_labelled_journal(win):
    win.desk["free"] = True
    win.send("/book")
    _drain(win)
    text = win.transcript.toPlainText()
    assert win.fetch.calls == [] and "journal open trades (Questrade unavailable: not fetched yet)" in text
    assert "not read from Questrade now: the desk is closed" in text


def test_check_reads_the_book_on_the_news_thread_and_the_gate_uses_it(win):
    win._brain_reason = "off"
    win.send("/check short NVDA 400 stop 3.20 entry 3.05")
    assert "book-fetch" in win.news_queue.pending_keys() and not any(
        k == "book-fetch" for k in win.queue.pending_keys())
    _drain(win)
    text = win.transcript.toPlainText()
    assert len(win.fetch.calls) == 1 and "gate:NVDA:book:source" in text and "Questrade positions" in text
    assert "gate:NVDA:book:hint:short_account" in text


def test_the_0620_fetch_is_queued_once_at_refresh_priority(win):
    win.clock["now"] = datetime(2026, 9, 29, 6, 20, tzinfo=PT)
    win.maybe_fetch_book()
    win.maybe_fetch_book()
    assert win.news_queue.pending_keys() == ["book-fetch"] and win.queue.pending() == []
    _drain(win)
    win.clock["now"] = datetime(2026, 9, 29, 7, 20, tzinfo=PT)
    win.maybe_fetch_book()
    assert win.news_queue.pending() == [] and len(win.fetch.calls) == 1


# ---------------------------------------------------------------- P14: a chat question about the book
def _chat_turn(window, text):
    import time

    from PySide6.QtWidgets import QApplication

    window._brain_ok, window._endpoint, window._model, window._native_tools = True, "http://x", "gemma4:12b", True
    window.send(text)
    worker = window._worker
    assert worker is not None and worker.wait(5000)
    deadline = time.monotonic() + 5
    while window._worker is not None and time.monotonic() < deadline:
        QApplication.processEvents()
    window._io.submit(lambda: None).result(5)
    payload = window.model_calls[-1][1]
    return "\n".join(str(m.get("content") or "") for m in payload["messages"] if m.get("role") == "tool")


def test_a_chat_book_question_with_no_fresh_snapshot_queues_the_read_and_says_so(win):
    """P14: "what am I holding" used to answer from the journal because only /book read the broker."""
    first = _chat_turn(win, "what am I holding")
    assert "Source: journal open trades" in first and book_jobs.PENDING_NOTE in first
    assert "book-fetch" in win.news_queue.pending_keys() and win.fetch.calls == []
    _drain(win)
    assert len(win.fetch.calls) == 1
    win.clock["now"] = TUE_0700 + timedelta(minutes=2)
    second = _chat_turn(win, "and my exposure now?")
    assert "Source: Questrade positions at Tue 09-29 07:00 PT" in second and book_jobs.PENDING_NOTE not in second
    assert "book-fetch" not in win.news_queue.pending_keys() and len(win.fetch.calls) == 1, "fresh: no second read"


def _source(pack):
    return next(row for row in pack.rows if row["id"] == "book:source")["text"]


def test_the_note_says_fetch_running_only_when_a_read_was_queued(store):
    """Review of bba9077c: backing off, desk closed or no token is a labelled journal view, never "fetch running"."""
    from mentor_packs.registry import make_pack

    queued: list = []
    journal = book_pack.build(now=TUE_0700, sources=book_jobs.store_sources(store, book_pack.fixture_sources()))
    for why in ("broker read backing off", "desk closed"):
        noted = book_jobs.note_pending_fetch(journal, lambda why=why: queued.append(why), state=lambda why=why: why)
        assert _source(noted).endswith(f"; journal view ({why})") and book_jobs.PENDING_NOTE not in _source(noted)
    assert queued == [], "no read queued while backing off or with the desk closed"
    no_token = book_jobs.note_pending_fetch(journal, lambda: queued.append("tok") or True, state=lambda: "no broker token")
    assert _source(no_token).endswith("; journal view (no broker token)") and queued == ["tok"], "a new token is read"
    assert book_jobs.note_pending_fetch(journal, lambda: None) is not journal
    assert _source(book_jobs.note_pending_fetch(journal, lambda: None)).endswith("journal view (broker read not queued)")
    other = make_pack("pick_pack", [{"id": "pick:X:asof", "kind": "source", "source": "journal", "text": "x"}])
    build = book_jobs.chat_pack_builder(lambda name, args: other, request_fetch=lambda: queued.append(1) or True)
    assert build("pick_pack", {}) is other and queued == ["tok"]
    noted = book_jobs.note_pending_fetch(journal, lambda: queued.append(1) or True)
    assert _source(noted).endswith(book_jobs.PENDING_NOTE) and queued == ["tok", 1]
    assert [row["id"] for row in noted.rows] == [row["id"] for row in journal.rows]


def test_chat_fetch_state_reads_the_policy(store):
    assert book_jobs.chat_fetch_state(store, TUE_0700, lambda: False) == ""
    assert book_jobs.chat_fetch_state(store, TUE_0700, lambda: True) == "desk closed"
    book_jobs.ensure_book(store, TUE_0700, fetch=_Fetch(fail="no token"))
    assert book_jobs.chat_fetch_state(store, TUE_0700, lambda: False) == "no broker token"
    book_jobs.ensure_book(store, TUE_0700, fetch=_Fetch(fail="RuntimeError: 400"))
    assert book_jobs.chat_fetch_state(store, TUE_0700, lambda: False) == "broker read backing off"
    book_jobs.ensure_book(store, TUE_0700 + timedelta(hours=1), fetch=_Fetch())
    assert book_jobs.chat_fetch_state(store, TUE_0700 + timedelta(hours=1), lambda: False) == "fresh"


def test_a_chat_book_question_with_the_desk_closed_says_so_and_queues_nothing(win):
    win.desk["free"] = True
    first = _chat_turn(win, "what am I holding")
    assert "journal view (desk closed)" in first and book_jobs.PENDING_NOTE not in first
    assert "book-fetch" not in win.news_queue.pending_keys() and win.fetch.calls == []
