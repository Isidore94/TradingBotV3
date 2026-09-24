"""Review my day: the walk on screen - saves, resume, keys, workers, banner, rebuild.

Trader, 2026-09-23: a guided walk of about five minutes in Day Review.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

SESSION = "2026-09-22"


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _wait(app, predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


class FakePositions:
    def __init__(self, start=None):
        self.rows = {SESSION: dict(start)} if start else {}

    def get(self, session):
        return dict(self.rows.get(session) or {})

    def put(self, session, row):
        self.rows[session] = dict(row)


class FakeWriter:
    """Records every call and the thread it ran on."""

    def __init__(self, fail=None):
        self.calls: list[tuple[str, dict]] = []
        self.threads: list[int] = []
        self.fail = fail or {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(**kwargs):
            self.threads.append(threading.get_ident())
            self.calls.append((name, kwargs))
            if name in self.fail:
                import recap_store

                raise recap_store.RecapWriteError(self.fail[name])
            return {"id": f"rc-{len(self.calls)}", "ok": True}

        return call

    def names(self):
        return [name for name, _ in self.calls]


def _subject(kind, subject_id, **detail):
    import mentor_questions

    return mentor_questions.Subject(
        kind=kind, subject_id=subject_id, options=("planned_off_the_desk", "impulse"),
        prompt=f"{kind}?", detail=detail,
    )


def _cards():
    from ui.widgets import recap_walk_cards as cards

    trade = {
        "trade_id": "t1", "symbol": "AAA", "direction": "LONG", "status": "CLOSED",
        "opened_at": f"{SESSION}T10:00:00-04:00", "closed_at": f"{SESSION}T11:00:00-04:00",
        "net_pnl": -20.0, "average_entry_price": 10.0, "average_exit_price": 9.8,
    }
    picks = [{
        "symbol": "BBB", "side": "LONG", "category": "real_miss", "result": "ran +12.0% after (run)",
        "reason": "late", "recipe": "D1 pullback", "recipe_lately": {}, "where_to_look": [],
        "surfaced": {"first_at": f"{SESSION}T09:45:00-04:00", "surfaces": [{"surface": "Focus"}]},
    }]
    mentor = {"t1": [
        _subject("trade_origin", "t1", trade_id="t1", symbol="AAA"),
        _subject("exit_draft_review", "t1@2026-09-22", trade_id="t1", key="t1@2026-09-22",
                 exit_session=SESSION, fields={"watching": [{"quote": "vwap"}]}),
    ]}
    payload = {"session_date": SESSION, "trades": [trade], "reads": [], "name_charts": {}}
    return cards.build_cards(
        SESSION, payload, findability={"picks": picks},
        environment={"main_intraday_label": "bullish_weak", "timeline": [{"words": "opening auto regime"}]},
        mentor_by_trade=mentor, rule={"id": "rc-rule", "text": "wait for the 5m close"}, streak=2,
    )


def _walk(app, *, positions=None, writer=None, loader=None, rebuilder=None, clue_flow=False):
    from ui.widgets.recap_walk import RecapWalk

    made = _cards()
    walk = RecapWalk(
        SESSION, {"session_date": SESSION},
        loader=loader or (lambda s, p, z: {"cards": made}),
        writer=writer if writer is not None else FakeWriter(),
        rebuilder=rebuilder or (lambda s: None),
        positions=positions if positions is not None else FakePositions(),
        clue_flow=clue_flow,
    )
    walk.resize(1200, 800)
    walk.show()
    walk.start()
    assert _wait(app, lambda: walk._loaded and (walk.current_card() is not None or walk.is_finished_screen()))
    return walk


def _go_to_kind(walk, kind):
    index = next(i for i, c in enumerate(walk.cards) if c["kind"] == kind)
    walk.go_to(index)
    return index


# -- saves route to the right writer --------------------------------------------
def test_each_card_saves_through_its_own_writer(app):
    writer = FakeWriter()
    walk = _walk(app, writer=writer)

    _go_to_kind(walk, "miss")
    walk.option_buttons(walk.index)[1].click()  # Real miss
    assert _wait(app, lambda: "card_answer" in writer.names())
    name, kwargs = writer.calls[-1]
    assert kwargs["option"] == "real_miss" and kwargs["card"]["kind"] == "miss"

    _go_to_kind(walk, "trade")
    walk.option_buttons(walk.index)[0].click()  # the Mentor's trade_origin
    assert _wait(app, lambda: "mentor_answer" in writer.names())
    _name, kwargs = writer.calls[-1]
    assert kwargs["subject"].kind == "trade_origin" and kwargs["option"] == "planned_off_the_desk"

    _go_to_kind(walk, "environment")
    walk.option_buttons(walk.index)[0].click()
    assert _wait(app, lambda: "environment_verdict" in writer.names())
    _name, kwargs = writer.calls[-1]
    assert kwargs["verdict"] == "agree" and kwargs["auto_label"] == "bullish_weak"

    _go_to_kind(walk, "lesson")
    walk.option_buttons(walk.index)[0].click()  # kept yesterday's rule: yes
    assert _wait(app, lambda: "rule_check" in writer.names())
    assert writer.calls[-1][1]["answer"] == "yes" and writer.calls[-1][1]["rule_id"] == "rc-rule"
    widgets = walk.lesson_widgets()
    widgets["boxes"]["keep"].setText("waiting for the close")
    widgets["boxes"]["keep"].returnPressed.emit()
    assert _wait(app, lambda: "lesson" in writer.names())
    assert writer.calls[-1][1]["keep"] == "waiting for the close"
    widgets["rule_text"].setText("no trades before 9:45")
    widgets["rule_text"].returnPressed.emit()
    assert _wait(app, lambda: "rule" in writer.names())
    assert writer.calls[-1][1]["text"] == "no trades before 9:45"
    assert _wait(app, lambda: walk.tomorrow_rule == "no trades before 9:45")


def test_the_exit_draft_yes_confirms_through_the_card_writer(app):
    writer = FakeWriter()
    walk = _walk(app, writer=writer)
    index = _go_to_kind(walk, "trade")
    texts = [b.text() for b in walk.option_buttons(index)]
    assert texts[0].startswith("1") and "planned" in texts[0]
    # The draft's Yes / Fix row follows the Mentor question on the same card.
    yes = next(b for b in walk.cards_stack.currentWidget().findChildren(type(walk.next_button))
               if b.property("walk_option") == "yes")
    yes.click()
    assert _wait(app, lambda: "exit_confirm" in writer.names())
    assert writer.calls[-1][1]["draft"]["key"] == "t1@2026-09-22"


def test_a_failed_save_says_not_saved_and_why(app):
    writer = FakeWriter(fail={"card_answer": "disk full"})
    walk = _walk(app, writer=writer)
    index = _go_to_kind(walk, "miss")
    walk.option_buttons(index)[0].click()
    label = walk._status_labels[index]
    assert _wait(app, lambda: label.text().startswith("not saved"))
    assert "disk full" in label.text()
    assert walk.saved_count == 0


def test_the_real_writer_routes_a_mentor_answer_to_the_journal(tmp_path, monkeypatch):
    """`record_card_answer(mentor_subject=...)` lands in the Mentor's store, not the recap file."""
    import project_paths
    from ui.widgets.recap_walk import WalkWriter

    recap_file = tmp_path / "day_recap_events.jsonl"
    monkeypatch.setattr(project_paths, "DAY_RECAP_EVENTS_FILE", recap_file)

    class Store:
        def __init__(self):
            self.rows = []

        def record_opportunity_event(self, **kwargs):
            self.rows.append(kwargs)
            return kwargs

    store = Store()
    writer = WalkWriter(journal_store=store)
    card = {"id": "trade:t1", "kind": "trade", "subject": {"trade_id": "t1", "symbol": "AAA"}}
    result = writer.mentor_answer(
        session=SESSION, card=card, subject=_subject("trade_origin", "t1", trade_id="t1", symbol="AAA"),
        option="impulse",
    )
    assert result["routed_to"] == "mentor"
    assert store.rows and store.rows[0]["payload"]["trade_origin"] == "impulse"
    assert not recap_file.exists()

    row = writer.card_answer(session=SESSION, card={"id": "miss:BBB:LONG", "kind": "miss",
                                                     "subject": {"symbol": "BBB"}}, option="good_pass")
    assert row["kind"] == "card_answer" and recap_file.exists()


# -- resume ----------------------------------------------------------------------
def test_the_walk_resumes_where_it_was_left(app):
    positions = FakePositions({"index": 2, "finished": False})
    walk = _walk(app, positions=positions)
    assert walk.index == 2
    walk.next()
    assert positions.get(SESSION) == {"index": 3, "finished": False}


def test_a_finished_walk_opens_on_its_summary_and_remembers_it(app):
    positions = FakePositions()
    walk = _walk(app, positions=positions)
    walk.go_to(len(walk.cards) - 1)
    walk.next()
    assert walk.is_finished_screen()
    assert positions.get(SESSION)["finished"] is True
    again = _walk(app, positions=positions)
    assert again.is_finished_screen()


# -- keyboard --------------------------------------------------------------------
def test_keys_move_choose_and_exit(app):
    writer = FakeWriter()
    walk = _walk(app, writer=writer)
    assert walk.index == 0
    QTest.keyClick(walk, Qt.Key.Key_Right)
    assert walk.index == 1
    QTest.keyClick(walk, Qt.Key.Key_Space)
    assert walk.index == 2
    QTest.keyClick(walk, Qt.Key.Key_Left)
    assert walk.index == 1
    index = _go_to_kind(walk, "miss")
    QTest.keyClick(walk, Qt.Key.Key_1)
    assert _wait(app, lambda: writer.names() == ["card_answer"])
    assert writer.calls[0][1]["option"] == walk.cards[index]["options"][0][0]
    left = []
    walk.exited.connect(lambda: left.append(True))
    QTest.keyClick(walk, Qt.Key.Key_Escape)
    assert left == [True]


def test_enter_in_a_text_box_saves_and_arrows_stay_in_the_box(app):
    writer = FakeWriter()
    walk = _walk(app, writer=writer)
    _go_to_kind(walk, "lesson")
    box = walk.lesson_widgets()["boxes"]["stop"]
    box.setFocus()
    QTest.keyClicks(box, "chasing")
    QTest.keyClick(box, Qt.Key.Key_Left)
    assert walk.cards[walk.index]["kind"] == "lesson"  # the cursor moved, not the card
    QTest.keyClick(box, Qt.Key.Key_Return)
    assert _wait(app, lambda: "lesson" in writer.names())
    assert writer.calls[-1][1]["stop"] == "chasing"


# -- nothing on the Qt thread ----------------------------------------------------
def test_the_cards_and_the_saves_are_built_off_the_qt_thread(app):
    from ui.widgets.recap_walk import RecapWalk

    made = _cards()
    seen: list[int] = []

    def slow_loader(session, payload, zone):
        seen.append(threading.get_ident())
        time.sleep(0.4)
        return {"cards": made}

    writer = FakeWriter()
    walk = RecapWalk(SESSION, {"session_date": SESSION}, loader=slow_loader, writer=writer,
                     rebuilder=lambda s: None, positions=FakePositions(), clue_flow=False)
    started = time.perf_counter()
    walk.start()
    assert time.perf_counter() - started < 0.2
    assert walk.current_card() is None  # still building
    assert _wait(app, lambda: walk.current_card() is not None)
    main = threading.get_ident()
    assert seen and seen[0] != main
    walk.option_buttons(_go_to_kind(walk, "miss"))[0].click()
    assert _wait(app, lambda: writer.threads)
    assert all(ident != main for ident in writer.threads)


# -- the rebuild -----------------------------------------------------------------
def test_the_day_record_is_rebuilt_after_the_lesson_when_something_was_saved(app):
    rebuilt: list[tuple[str, int]] = []
    walk = _walk(app, rebuilder=lambda s: rebuilt.append((s, threading.get_ident())))
    walk.option_buttons(_go_to_kind(walk, "miss"))[0].click()
    assert _wait(app, lambda: walk.saved_count == 1)
    walk.go_to(len(walk.cards) - 1)
    walk.next()
    assert _wait(app, lambda: rebuilt)
    assert rebuilt[0][0] == SESSION and rebuilt[0][1] != threading.get_ident()
    walk.exit_walk()  # nothing new since: no second rebuild
    app.processEvents()
    assert walk.rebuilds_started == 1


def test_exit_rebuilds_only_when_something_was_saved(app):
    rebuilt: list[str] = []
    walk = _walk(app, rebuilder=rebuilt.append)
    walk.exit_walk()
    assert walk.rebuilds_started == 0
    walk.option_buttons(_go_to_kind(walk, "miss"))[0].click()
    assert _wait(app, lambda: walk.saved_count == 1)
    walk.exit_walk()
    assert _wait(app, lambda: rebuilt == [SESSION])


# -- the clue hook ---------------------------------------------------------------
def test_a_chart_card_offers_mark_a_clue_through_the_hook(app):
    from ui.widgets import recap_walk_cards as cards

    calls = []

    class Flow:
        class marker:  # noqa: N801 - the ClueFlow shape
            @staticmethod
            def is_active():
                return False

            class activeChanged:  # noqa: N801
                @staticmethod
                def connect(_slot):
                    return None

        def load(self):
            calls.append("load")

        def set_active(self, on):
            calls.append(("active", on))

    def hook(chart, session, symbol, timeframe, **links):
        calls.append((session, symbol, timeframe, links.get("card_id")))
        return Flow()

    bars = [{"dt": datetime(2026, 9, 22, h, m), "open": 1, "high": 2, "low": 0.5, "close": 1.5}
            for h in (10, 11) for m in (0, 5, 10)]
    payload = {"session_date": SESSION, "trades": [], "reads": [], "spy_m5_bars": bars, "name_charts": {}}
    made = cards.build_cards(SESSION, payload, environment={"main_intraday_label": "neutral_chop"})
    walk = _walk(app, loader=lambda s, p, z: {"cards": made}, clue_flow=hook)
    index = _go_to_kind(walk, "environment")
    button = walk.clue_button(index)
    assert button is not None and "SPY" in button.text()
    assert (SESSION, "SPY", "M5", "environment") in calls and "load" in calls
    button.click()
    assert ("active", True) in calls


# -- the Day Review page: entry points -------------------------------------------
class _Service:
    def read_day(self, session_date, **_kwargs):
        from ui.services.day_review_service import empty_payload

        return empty_payload(session_date)


@pytest.fixture
def panel(app, monkeypatch):
    from ui.panels import day_review_panel

    widget = day_review_panel.DayReviewPanel(service=_Service(), clock=lambda: datetime(2026, 9, 22, 17, 30))
    widget.show_session(SESSION)
    for _ in range(100):
        if widget._worker is None:
            break
        widget._worker.wait(50)
        app.processEvents()
    yield widget
    widget.shutdown()
    widget.deleteLater()
    app.processEvents()


def _render(panel, **extra):
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    payload.update(extra)
    panel.render(payload)


def test_the_banner_shows_only_after_the_close(panel, monkeypatch):
    import ui.widgets.recap_walk as walk_mod

    monkeypatch.setattr(walk_mod, "walk_finished", lambda session, positions=None: False)
    ready: list[bool] = []
    panel.walkReadyChanged.connect(ready.append)
    _render(panel, provisional=True)
    assert not panel.walk_banner.isVisibleTo(panel)
    assert ready in ([], [False])
    _render(panel, provisional=False)
    assert panel.walk_banner.isVisibleTo(panel)
    assert ready[-1] is True
    assert panel.walk_button.isEnabled()


def test_no_banner_before_the_close_on_the_clock(app, monkeypatch):
    from ui.panels import day_review_panel
    from ui.services.day_review_service import empty_payload

    widget = day_review_panel.DayReviewPanel(service=_Service(), clock=lambda: datetime(2026, 9, 22, 12, 0))
    try:
        payload = empty_payload(SESSION)
        payload["provisional"] = False
        widget.render(payload)
        assert not widget.walk_banner.isVisibleTo(widget)
    finally:
        widget.shutdown()
        widget.deleteLater()


def test_saved_clues_draw_on_the_pages_trade_chart(panel, app, monkeypatch, tmp_path):
    from zoneinfo import ZoneInfo

    import project_paths
    import recap_store
    from ui.widgets.clue_marker import drawn_clue_count

    monkeypatch.setattr(project_paths, "DAY_RECAP_EVENTS_FILE", tmp_path / "recap.jsonl")
    et = ZoneInfo("America/New_York")
    bars = [{"dt": datetime(2026, 9, 22, 10, m, tzinfo=et), "open": 10, "high": 11, "low": 9.5, "close": 10.5,
             "volume": 100} for m in range(0, 30, 5)]
    recap_store.record_clue(session_date=SESSION, symbol="AAA", timeframe="M5", bar_time=bars[2]["dt"],
                            price=10.5, clue_tag="vwap_reclaim")
    trade = {"trade_id": "t1", "symbol": "AAA", "direction": "LONG", "status": "CLOSED",
             "opened_at": f"{SESSION}T10:00:00-04:00", "net_pnl": 5.0}
    _render(panel, trades=[trade], name_charts={"AAA": {"bars": bars, "markers": ()}})
    assert panel.trade_chart_symbol() == "AAA"
    assert _wait(app, lambda: drawn_clue_count(panel._trade_chart) == 1)


def test_the_walk_replaces_the_page_body_and_exit_brings_it_back(panel, app, monkeypatch):
    import ui.widgets.recap_walk as walk_mod

    made = _cards()
    monkeypatch.setattr(walk_mod, "_default_loader", lambda s, p, z: {"cards": made})
    monkeypatch.setattr(walk_mod, "_default_rebuilder", lambda s: None)
    _render(panel, provisional=False)
    panel.walk_button.click()
    assert panel.walk_is_open()
    walk = panel.walk()
    assert _wait(app, lambda: walk.current_card() is not None)
    walk.exit_walk()
    assert not panel.walk_is_open()
    assert panel.body_stack.currentWidget() is panel.scroll
    panel.open_walk()
    assert panel.walk() is walk  # same session: resumed, not rebuilt
