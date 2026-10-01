"""P1: the Trade Mentor card inside the app is the desk's card, driven by the desk's code.

Golden equivalence: the same scenario run through the desk (`MainWindow`, flag off)
and through the app (`AppMentorHost`) writes identical `opportunity_events` rows per
type. Plus: no forked host logic, the card sits under the chat and never pops, the
app builds no Mentor with the flag off, and the AI fill uses the 5080 when it is up.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_desk_isolation import fresh_mentor_pull_tally  # noqa: E402,F401
from tj14b_support import (  # noqa: E402
    REVIEWED,
    SESSION,
    add_round_trip,
    mark_covered,
    new_store,
    pacific,
    slot_at,
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


class _NoContext(QObject):
    """A context reader that never reaches Yahoo."""

    contextReady = Signal(str, object)
    contextUnavailable = Signal(str, object)

    def request_context(self, request_id, *, now=None):
        return False

    def shutdown(self, timeout_ms=250):
        pass


class _Journal:
    def write_entry(self, **_kwargs):
        return {"ok": True, "entry": {}}


class _FakeImport:
    def pull_recent_questrade(self, days):
        return True

    def shutdown(self):
        pass


def _app_host(tmp_path, monkeypatch):
    from mentor_app.card_host import AppMentorHost
    from ui.services.trade_mentor_service import TradeMentorService

    QApplication.instance() or QApplication([])
    service = TradeMentorService(
        state_path=tmp_path / "app_slots.json",
        clock=lambda: pacific(SESSION, 9, 5),
        idle_seconds=lambda: 0.0,
        session_locked=lambda: False,
    )
    host = AppMentorHost(service=service, context_service=_NoContext())
    monkeypatch.setattr(host, "_journal_import_service", lambda: _FakeImport())
    return host, host.dock.mentor_card


def _desk_host(monkeypatch):
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    window = MainWindow(UiState(workspace_mode="workspace"))
    monkeypatch.setattr(window, "_journal_import_service", lambda: _FakeImport())
    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.set_context_service(_NoContext())
    return window, card


def _rows_by_type(store) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for row in store.list_opportunity_events(limit=100_000):
        clean = {k: v for k, v in dict(row).items() if k not in ("event_id", "created_at")}
        grouped[str(row["event_type"])].append(json.dumps(clean, sort_keys=True, default=str))
    return {kind: sorted(rows) for kind, rows in grouped.items()}


def _run(path_name, scenario, tmp_path, monkeypatch):
    import journal_store

    folder = tmp_path / path_name
    folder.mkdir()
    store = new_store(folder)
    mark_covered(store, REVIEWED)
    trades = scenario["setup"](store)
    with monkeypatch.context() as patch:
        patch.setattr(journal_store, "JournalStore", lambda *a, **k: store)
        if path_name == "desk":
            host, card = _desk_host(patch)
        else:
            host, card = _app_host(folder, patch)
        card._clock = lambda: pacific(SESSION, 9, 5)
        card._journal = _Journal()
        patch.setattr(card, "_missing_prediction_reason", lambda _h: "")
        patch.setattr(card, "_start_ai_fill", lambda *args: None)
        try:
            host._show_trade_mentor_prompt(slot_at(SESSION, scenario.get("hour", 9)))
            scenario["act"](host, card, trades)
        finally:
            if path_name == "desk":
                host.close()
            else:
                host.shutdown()
    return _rows_by_type(store)


def _one_trade(store):
    return [add_round_trip(store, "DRAM")]


def _two_trades(store):
    return [add_round_trip(store, "AAA", entry_hour=7), add_round_trip(store, "BBB", entry_hour=8)]


def _answer_first_and_leave(host, card, trades):
    import trade_mentor_trade_check as check

    first, second = trades
    for combo, _text in card._answer_inputs[first].values():
        combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))
    card._trade_save_buttons[first].click()
    card._answer_inputs[second]["thesis"][1].setText("Bought the bounce off VWAP.")
    host._show_trade_mentor_prompt(slot_at(SESSION, 10))


def _answer_the_questions(host, card, trades):
    assert card._question_inputs, "an unplanned trade is asked where it came from (TJ-12)"
    for _subject, combo in card._question_inputs.values():
        combo.setCurrentIndex(next(i for i in range(combo.count()) if combo.itemData(i)))
    assert card.save_questions()["ok"] is True


SCENARIOS = {
    # test_tj12_mentor_wake: the woken question kinds are asked and their answers stored.
    "questions_answered": {"setup": _one_trade, "act": _answer_the_questions, "expect": {"NOTE"}},
    # test_mentor_asks_once: answering the read files the untouched trade and retires it.
    "submit_files_and_retires": {
        "setup": _one_trade, "act": lambda host, card, trades: card.submit(), "expect": {"MENTOR_ASKED"},
    },
    # test_mentor_asks_once: skipping the card also retires the trade.
    "skip_retires": {"setup": _one_trade, "act": lambda host, card, trades: card.skip(), "expect": {"MENTOR_ASKED"}},
    # test_mentor_stores_an_answered_trade: one trade saved, the other filed on leave.
    "answer_one_file_the_other": {
        "setup": _two_trades, "act": _answer_first_and_leave, "expect": {"MENTOR_ASKED", "RECALLED"},
    },
    # test_mentor_stop_first / TJ-9 ride: an ordinary 11:00 slot still carries the trade.
    "ordinary_slot_rides": {
        "setup": _one_trade, "hour": 11, "act": lambda host, card, trades: card.skip(), "expect": {"MENTOR_ASKED"},
    },
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_app_writes_the_same_journal_events_as_the_desk(name, tmp_path, monkeypatch):
    scenario = SCENARIOS[name]
    desk = _run("desk", scenario, tmp_path, monkeypatch)
    app = _run("app", scenario, tmp_path, monkeypatch)
    assert desk, "the scenario must write something"
    assert scenario["expect"] <= set(desk), sorted(desk)
    assert {kind: len(rows) for kind, rows in app.items()} == {kind: len(rows) for kind, rows in desk.items()}
    assert app == desk


def test_the_app_host_runs_the_desks_code_not_a_fork():
    from mentor_app.card_host import AppMentorHost
    from ui.app import MainWindow

    for name in (
        "_show_trade_mentor_prompt",
        "_trade_check_is_owed",
        "_show_mentor_questions",
        "_mentor_question_state",
        "_mentor_answered",
        "_mentor_journal_pull",
        "_on_econ_view",
    ):
        assert getattr(AppMentorHost, name) is getattr(MainWindow, name), name
    assert AppMentorHost._mentor_origin_lanes.__func__ is MainWindow._mentor_origin_lanes.__func__


def test_recap_walk_still_reads_retired_subjects_from_the_slots_file():
    source = (SCRIPTS_DIR / "ui" / "widgets" / "recap_walk_cards.py").read_text(encoding="utf-8")
    assert 'state.get("retired_subjects")' in source and "TRADE_MENTOR_SLOTS_FILE" in source


# --------------------------------------------------------------------------- the window
@pytest.fixture()
def app_window(tmp_path, monkeypatch):
    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    host, _card = _app_host(tmp_path, monkeypatch)
    posted: list = []
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        post=lambda url, payload, timeout: posted.append((url, payload)) or {},
        card_host=host,
    )
    # A fixed 10:00 PT clock: the Inbox refuses items in its quiet hours (06:30-07:00 PT).
    from datetime import datetime, timezone

    win.inbox.now = lambda: datetime(2026, 9, 30, 17, 0, tzinfo=timezone.utc)
    yield win
    win.shutdown()
    win.deleteLater()


def test_a_due_slot_shows_the_card_under_the_chat_and_an_inbox_line(app_window, tmp_path, monkeypatch):
    import journal_store

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    add_round_trip(store, "DRAM")
    monkeypatch.setattr(journal_store, "JournalStore", lambda *a, **k: store)
    before = app_window.transcript.toPlainText()
    host = app_window.card_host
    host.trade_mentor_service.promptDue.emit(slot_at(SESSION, 9))
    assert host.dock.isVisibleTo(app_window) and host.dock.mentor_card.isVisibleTo(app_window)
    assert app_window.inbox_header.text() == "Inbox (1)"
    assert "Trade Mentor 09:00" in app_window.inbox.items()[0].text
    assert app_window.transcript.toPlainText() == before, "the transcript only moves when the trader types"
    assert app_window.read_button.isVisible() or app_window.read_button.isVisibleTo(app_window)


def test_give_a_read_and_pause_are_app_commands(app_window):
    app_window.send("/read")
    assert app_window.card_host.dock.mentor_card.isVisibleTo(app_window)
    app_window.send("/pause")
    assert app_window.card_host.trade_mentor_service.is_paused()
    assert not app_window.card_host.dock.mentor_card.isVisibleTo(app_window)


def test_with_the_flag_off_the_app_owns_no_mentor(tmp_path, monkeypatch):
    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    win = MentorWindow(store=MentorChatStore(tmp_path / "c.sqlite3"), queue=PrefetchQueue(), mentor_enabled=False)
    try:
        assert win.card_host is None, "the desk owns the Mentor; a second TradeMentorService would be two owners"
        win.send("/read")
        assert "on the desk" in win.transcript.toPlainText()
    finally:
        win.shutdown()
        win.deleteLater()


def test_the_ai_fill_goes_to_the_5080_when_the_brain_is_up(app_window, monkeypatch):
    import ai_summary

    assert app_window._brain_fill_request() is None, "brain off: the card keeps its local path"
    app_window._brain_ok, app_window._endpoint, app_window._model = True, "http://127.0.0.1:11436", "gpt-oss:20b"
    calls: list = []
    monkeypatch.setattr(ai_summary, "request_ai_summary", lambda **kwargs: calls.append(kwargs) or {"summary": {}})
    request = app_window._brain_fill_request()
    request(provider="local", model="gemma3:12b", api_key="", evidence={})
    assert calls[0]["endpoint"] == "http://127.0.0.1:11436/v1" and calls[0]["model"] == "gpt-oss:20b"
    card = app_window.card_host.dock.mentor_card
    assert card._ai_fill_request() is not None


def test_request_ai_summary_honours_an_explicit_local_endpoint(monkeypatch):
    import ai_summary

    monkeypatch.setattr(ai_summary, "local_endpoint_url", lambda: "")
    urls: list = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}

        text = ""

        def raise_for_status(self):
            pass

    def post(url, **kwargs):
        urls.append(url)
        return _Resp()

    with contextlib.suppress(Exception):  # the empty draft may fail validation; the URL is what matters
        ai_summary.request_ai_summary(
            provider="local", model="m", api_key="", evidence={"x": 1}, post=post,
            schema={"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            endpoint="http://127.0.0.1:11436/v1",
        )
    assert urls and urls[0] == "http://127.0.0.1:11436/v1/chat/completions"


def test_the_morning_econ_block_shows_in_the_dock(tmp_path, monkeypatch):
    host, _card = _app_host(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(host, "_econ_morning_has_started", lambda: True)
        monkeypatch.setattr(host, "_auto_mode_now", lambda: "DESK")
        monkeypatch.setattr(host.trade_mentor_service, "enabled", lambda: True)
        view = {"session": SESSION.isoformat(), "today": [], "week": []}
        host._on_econ_view(view)
        assert host.dock.econ_block.isVisibleTo(host.dock)
        assert host.trade_mentor_service.econ_brief_shown(SESSION.isoformat())
    finally:
        host.shutdown()
