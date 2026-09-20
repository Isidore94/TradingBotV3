"""TJ-10 item 7 - the ONE payload gains `reads` and `congruence`, and the page
draws them.

Packet `.claude/packets/TJ-10.md` item 7; `plan.md` §12.4 "TJ-10" item 3 and
"TJ-1" item 3 (one read, on one worker, and the page computes nothing). RED
before the build.

The contract these tests pin
----------------------------

``scripts/ui/services/day_review_service.py``
    * ``PAYLOAD_KEYS`` gains ``reads`` and ``congruence``; ``empty_payload``
      carries both PRESENT and empty. (Note for the builder: `PAYLOAD_KEYS`
      does not in fact declare every key `read_day` writes today -
      ``walkaway_backfill_sessions`` and ``error`` are both absent from it -
      so add the two new keys to BOTH.)
    * ``read_day`` fills them ON THE WORKER, from the entries and the decisions
      that one read already opened. It starts no second store read for them.
    * ``DayReviewService.build_reads_for(session_date)`` is the named seam the
      post-close tick calls, and ``_IndexBuildWorker`` calls it on the worker
      thread. A failure there costs the grades and nothing else.

``scripts/ui/panels/day_review_panel.py``
    * ``render`` shows a verdict CHIP beside each graded entry in "What you
      said" and prints the congruence lines under the story. It formats and
      computes nothing: it calls no function in ``market_read_grades``.
    * An entry with no read, and a ``No view`` answer, get no verdict chip -
      the page never shows a verdict nobody measured.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Day Review page uses PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

import tj10_support as fx  # noqa: E402

SESSION = fx.SESSION
NOW = datetime(2026, 9, 19, 8, 0)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# -- the payload -------------------------------------------------------------


def test_the_payload_declares_the_two_new_keys_present_and_empty():
    from ui.services.day_review_service import PAYLOAD_KEYS, empty_payload

    assert "reads" in PAYLOAD_KEYS
    assert "congruence" in PAYLOAD_KEYS

    blank = empty_payload(SESSION)
    assert blank["reads"] in ((), [])
    assert blank["congruence"] in ((), [])


class _Journal:
    def __init__(self, entries=()):
        self._entries = list(entries)

    def entries_about(self, _session):
        return [dict(row) for row in self._entries]

    def daily_story(self, _session):
        return None

    def theses_for(self, _session):
        return []


def _wire(monkeypatch, *, entries=(), annotations=(), session_bars=None):
    """`read_day` over plain dicts. No live store is opened."""
    import chart_snapshot
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import journal_store
    from ui.services.day_review_service import DayReviewService

    store = {"annotations": list(annotations), "pick_feedback": [],
             "swing_favorites": [], "review_events": []}

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl",
                        lambda name, *a, **k: _Store(store.get(name, [])))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        journal_store, "JournalStore",
        lambda *a, **k: type("_J", (), {"list_trades": lambda self: []})(),
    )
    monkeypatch.setattr(day_review_bars, "read_session_bars",
                        lambda *a, **k: dict(session_bars or {}))
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: True)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *a, **k: False)
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])

    service = DayReviewService(journal_service=_Journal(entries))
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])
    return service


def test_read_day_carries_one_read_row_per_clicked_prediction(monkeypatch):
    clicked = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    service = _wire(
        monkeypatch, entries=[clicked], session_bars={"SPY": fx.session_tape()}
    )

    payload = service.read_day(SESSION, now=NOW)

    reads = list(payload["reads"])
    assert len(reads) == 1, payload.get("error")
    assert reads[0]["entry_id"] == clicked["entry_id"]
    assert reads[0]["source"] == "click"
    assert reads[0]["verdict"]


def test_read_day_carries_the_three_congruence_lines(monkeypatch):
    import market_read_grades as grades

    d1 = fx.mentor_entry(
        direction="down", horizon="next_5_sessions", timeframe="D1"
    )
    service = _wire(
        monkeypatch, entries=[d1], session_bars={"SPY": fx.session_tape()}
    )

    payload = service.read_day(SESSION, now=NOW)

    assert tuple(line["kind"] for line in payload["congruence"]) == (
        grades.CONGRUENCE_KINDS
    )


# -- the post-close seam -----------------------------------------------------


class _SeamService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def read_day(self, session_date, **_kwargs):
        return {"session_date": session_date}

    def build_index_for(self, session_date, **_kwargs):
        self.calls.append(("index", threading.get_ident()))
        return {}

    def build_session_bars_for(self, session_date, **_kwargs):
        self.calls.append(("bars", threading.get_ident()))
        return None

    def build_reads_for(self, session_date, **_kwargs):
        self.calls.append(("reads", threading.get_ident()))
        return []


def _drain(qapp, panel, *, timeout: float = 5.0) -> None:
    worker = panel._index_worker
    if worker is not None:
        worker.wait(int(timeout * 1000))
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and panel._index_worker is not None:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_the_post_close_tick_grades_the_session_on_the_worker(qapp, monkeypatch):
    import daily_recap_schedule
    from ui.panels.day_review_panel import DayReviewPanel

    monkeypatch.setattr(daily_recap_schedule, "due_session", lambda *a, **k: None)
    monkeypatch.setattr(
        daily_recap_schedule, "post_close_due_session", lambda *a, **k: SESSION
    )

    service = _SeamService()
    panel = DayReviewPanel(service=service, clock=lambda: NOW)
    monkeypatch.setattr(panel, "reload", lambda: None)
    monkeypatch.setattr(panel, "show_session", lambda _session: None)
    slot_thread = threading.get_ident()
    try:
        assert panel.poll_auto_read() == SESSION
        _drain(qapp, panel)
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()

    kinds = [name for name, _thread in service.calls]
    assert "reads" in kinds, "the post-close tick never graded the session"
    reads_thread = next(
        thread for name, thread in service.calls if name == "reads"
    )
    assert reads_thread != slot_thread, "the grading ran on the Qt timer's thread"


# -- the page ----------------------------------------------------------------


class _StubService:
    """Reads nothing. The page is what is under test."""

    def __init__(self, payload=None) -> None:
        self.payload = payload or {}
        self.reads = 0

    def read_day(self, session_date, **_kwargs):
        self.reads += 1
        payload = dict(self.payload)
        payload.setdefault("session_date", session_date)
        return payload


def _panel_payload():
    from ui.services.day_review_service import empty_payload

    clicked = fx.mentor_entry(
        direction="up", horizon="rest_of_day", timeframe="M5",
        text="market is holding the 50sma",
    )
    declined = fx.mentor_entry(
        direction="no_view", horizon="rest_of_day", timeframe="M5",
        created_at=fx.STAMP.replace(hour=9),
    )
    silent = fx.old_entry(
        "empty", text="just watching", created_at=fx.STAMP.replace(hour=10)
    )

    payload = empty_payload(SESSION)
    payload["entries"] = [clicked, declined, silent]
    payload["reads"] = [
        {
            "read_id": "rd-1", "entry_id": clicked["entry_id"], "session": SESSION,
            "source": "click", "direction": "up", "horizon": "rest_of_day",
            "verdict": "right", "observation": clicked["mentor"]["observation"],
        },
        {
            "read_id": "rd-2", "entry_id": declined["entry_id"], "session": SESSION,
            "source": "click", "direction": "no_view", "horizon": "rest_of_day",
            "verdict": "", "observation": "",
        },
    ]
    payload["congruence"] = [
        {"kind": "desk_d1_label", "text": "your D1 view is Down; the desk reads "
         "SPY trending_down", "verdict": "agrees", "counts": {}, "source_ids": [],
         "timeframe": "D1", "missing": ""},
        {"kind": "picks_side_mix", "text": "3 of 3 D1 likes were LONG",
         "verdict": "disagrees", "counts": {"long": 3, "short": 0},
         "source_ids": ["a", "b", "c"], "timeframe": "D1", "missing": ""},
        {"kind": "fills_bias", "text": "no fills today", "verdict": "unmeasured",
         "counts": {}, "source_ids": [], "timeframe": "", "missing": "your fills"},
    ]
    return payload, clicked, declined, silent


@pytest.fixture()
def panel(qapp, monkeypatch):
    from ui.panels.day_review_panel import DayReviewPanel

    widget = DayReviewPanel(service=_StubService(), clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    yield widget
    try:
        widget.shutdown()
    except Exception:  # noqa: BLE001
        pass
    widget.deleteLater()
    qapp.processEvents()


def test_a_graded_entry_shows_its_verdict_chip_and_an_ungraded_one_shows_none(panel):
    payload, clicked, declined, silent = _panel_payload()

    panel.render(payload)

    chips = panel.verdict_chips()
    assert chips[clicked["entry_id"]] == "right"
    assert not chips.get(declined["entry_id"]), (
        "a No view answer was given a verdict nobody measured"
    )
    assert not chips.get(silent["entry_id"])


def test_the_congruence_lines_are_shown_under_the_story(panel):
    payload, _clicked, _declined, _silent = _panel_payload()

    panel.render(payload)

    text = panel.congruence_text()
    assert "3 of 3 D1 likes were LONG" in text
    assert "no fills today" in text


def test_rendering_computes_no_read_and_no_line_on_the_paint_path(panel, monkeypatch):
    """The worker builds them; the page formats them."""
    import market_read_grades as grades

    def _never(*_args, **_kwargs):
        raise AssertionError("the paint path built a read row or a congruence line")

    monkeypatch.setattr(grades, "read_rows", _never)
    monkeypatch.setattr(grades, "grade_read", _never)
    monkeypatch.setattr(grades, "congruence_lines", _never)

    payload, _clicked, _declined, _silent = _panel_payload()
    panel.render(payload)


def test_an_empty_payload_paints_without_a_chip_or_a_line(panel):
    """A first paint and a failed read are the same shape (TJ-1)."""
    from ui.services.day_review_service import empty_payload

    panel.render(empty_payload(SESSION))

    assert panel.verdict_chips() == {}
