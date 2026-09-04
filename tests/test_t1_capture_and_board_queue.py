"""Packet T1.3 - a board chart holds no place and queues nothing.

Trader, 2026-09-04:

    "additionally when I click on ANYTHING from the RS/RW board it should not
    make a queue of picks if I click on more nor should it add to the 'waiting'
    list. once i look and click off, its done."

On `main` @ 6e05878 every board in the alert column charts through
`_chart_board_symbol` / `_chart_strength_board_symbol` -> `chart_symbol` -> a
`MANUAL_CHART_TAG` alert -> `_select_review_alert`, which sets
`_current_review_holds_place = not _is_m5_review_alert(alert)`. `MANUAL_CHART_TAG`
is in that method's exempt list, so it returns False and a manual chart HOLDS A
PLACE: the next board click inserts the previous look at the head of
`_review_queue`, and five clicks build a four-deep waiting list the trader then
has to click through.

The new contract:

* a `MANUAL_CHART_TAG` alert holds NO place;
* when the chart being replaced is a manual chart it is neither re-inserted nor
  given a `skip` event - it was a look, never a shown alert, so it belongs in
  no P(take | shown) denominator. Write NOTHING;
* the M5-alert-bar branch (`skip` with reason `clicked_away_from_m5_alert`) is
  untouched - a different population, and `review_learning` keys on that string;
* a D1 / Focus / armed chart that HELD a place still goes back to the head.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _d1(symbol: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="08:25:00",
        symbol=symbol,
        side=side,
        trigger=f"({side.lower()}) zone1 reject at AVWAPE",
        timeframe="D1",
        tag=f"d1_flag_{side.lower()}",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({side.lower()}) zone1 reject",
        is_d1=True,
    )


def _m5(symbol: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="07:09:19",
        symbol=symbol,
        side=side,
        trigger="[S-TIER] VWAP reclaim",
        timeframe="5m",
        tag="green",
        raw_text=f"[S-TIER] VWAP reclaim {symbol} ({side.lower()})",
    )


@pytest.fixture
def panel(tmp_path, monkeypatch):
    import pick_feedback
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    pick_feedback.clear_reviewed_today_cache()
    made = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(made, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(made, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(made.chart_review, "_reviewed_symbols", lambda: set())
    yield made
    made.close()
    made.deleteLater()


@pytest.fixture
def events(panel, monkeypatch):
    """Every review event the panel writes, as (action, kwargs)."""
    written: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        panel, "_record_review_event", lambda action, **kw: written.append((action, kw))
    )
    return written


def _five_board_clicks(panel) -> None:
    """One click on each board in the alert column, then a second on the first.

    These are the real slots the four boards' `symbolActivated` signals are
    connected to; three of the boards are children of this very panel, so the
    widget signal itself is used where it exists.
    """
    panel.rrs_snapshot.symbolActivated.emit("META", "SHORT")
    panel.entry_board.symbolActivated.emit("NVDA", "LONG")
    panel.focus_strength.symbolActivated.emit("AMD", "LONG")
    panel._chart_strength_board_symbol("SOXL", "long")
    panel.rrs_snapshot.symbolActivated.emit("TSLA", "LONG")


# ---------------------------------------------------------------------------
# the trader's sentence, measured
# ---------------------------------------------------------------------------
def test_five_board_clicks_leave_the_waiting_list_empty(panel):
    """"it should not make a queue of picks if I click on more"."""
    _five_board_clicks(panel)

    assert [alert.symbol for alert in panel._review_queue] == [], (
        "clicking five board names built a waiting list: "
        f"{[a.symbol for a in panel._review_queue]}"
    )
    assert panel._current_review_alert is not None
    assert panel._current_review_alert.symbol == "TSLA", "the last look is the chart"


def test_the_pane_still_reads_queue_clear_after_five_board_clicks(panel):
    """"nor should it add to the 'waiting' list" - the label the trader reads."""
    assert panel.chart_review.queue_label.text() in ("", "queue clear")

    _five_board_clicks(panel)

    assert panel.chart_review.queue_label.text() == "queue clear", (
        f"the pane says {panel.chart_review.queue_label.text()!r}"
    )


def test_a_board_chart_holds_no_place_in_the_waiting_list(panel):
    """The flag itself, at the seam that decides it."""
    panel.rrs_snapshot.symbolActivated.emit("META", "SHORT")

    assert panel._current_review_holds_place is False


def test_a_board_click_away_writes_nothing_at_all(panel, events):
    """A look is not a shown alert, so it belongs in no P(take | shown)
    denominator - neither as a take nor as a skip.

    The queue assertion rides along because "no skip written" is ALSO true of
    today's build, which answers a click-away by re-queueing instead. Silence
    only means the right thing when nothing was queued either.
    """
    _five_board_clicks(panel)

    assert [a.symbol for a in panel._review_queue] == []
    assert [action for action, _kw in events if action == "skip"] == [], (
        "a manual chart must never be skip-counted"
    )


def test_the_trader_can_click_away_from_a_board_look_and_be_done(panel, events):
    """"once i look and click off, its done." Another board click simply
    replaces the chart; nothing is re-inserted anywhere."""
    panel.rrs_snapshot.symbolActivated.emit("META", "SHORT")
    looked = panel._current_review_alert

    panel.entry_board.symbolActivated.emit("NVDA", "LONG")

    assert looked not in panel._review_queue
    assert [a.symbol for a in panel._review_queue] == []
    assert panel._current_review_alert.symbol == "NVDA"


def test_skipping_a_board_look_never_puts_it_back(panel):
    """`_skip_review_alert` and `_advance_review_queue` are checked too: no path
    anywhere re-inserts a MANUAL_CHART alert."""
    panel.rrs_snapshot.symbolActivated.emit("META", "SHORT")
    looked = panel._current_review_alert

    panel._skip_review_alert(looked)

    assert [a.symbol for a in panel._review_queue] == []
    assert panel._current_review_alert is None


# ---------------------------------------------------------------------------
# and the two populations that are NOT this one
# ---------------------------------------------------------------------------
def test_a_dequeued_d1_chart_still_returns_to_the_head_when_a_board_replaces_it(
    panel,
):
    """The existing rule, which must survive: a chart that HELD a place goes
    back so a look-elsewhere never loses it."""
    panel.add_alert(_d1("MUFG", "SHORT"))
    panel.add_alert(_d1("XOM"))
    assert panel._current_review_alert.symbol == "MUFG"
    assert [a.symbol for a in panel._review_queue] == ["XOM"]

    panel.rrs_snapshot.symbolActivated.emit("META", "SHORT")

    assert [a.symbol for a in panel._review_queue] == ["MUFG", "XOM"]
    assert panel._current_review_alert.symbol == "META"


def test_a_board_look_in_front_of_a_queue_still_does_not_join_it(panel):
    """The other half: the D1 rows behind stay exactly as they were."""
    panel.add_alert(_d1("MUFG", "SHORT"))
    panel.add_alert(_d1("XOM"))
    panel.rrs_snapshot.symbolActivated.emit("META", "SHORT")
    assert [a.symbol for a in panel._review_queue] == ["MUFG", "XOM"]

    panel.entry_board.symbolActivated.emit("NVDA", "LONG")

    assert [a.symbol for a in panel._review_queue] == ["MUFG", "XOM"], (
        "the META look must not have joined the queue"
    )
    assert panel._current_review_alert.symbol == "NVDA"


def test_the_m5_bar_click_away_still_writes_its_skip(panel, events):
    """Byte-for-byte untouched: `review_learning` keys on the reason string."""
    first, second = _m5("NVDA"), _m5("AMD")
    panel.add_alert(first)
    panel.add_alert(second)
    panel.chart_alert(first)
    panel.chart_alert(second)

    skips = [kw for action, kw in events if action == "skip"]
    assert len(skips) == 1
    assert skips[0]["alert"] is first
    assert skips[0]["detail"] == {"reason": "clicked_away_from_m5_alert"}
    assert [a.symbol for a in panel._review_queue] == []


# ---------------------------------------------------------------------------
# fix round 1, ADVISORY 3: looking at a QUEUED name takes it out of the queue
# ---------------------------------------------------------------------------
def test_looking_at_a_queued_name_from_a_board_takes_it_out_of_the_queue(panel, events):
    """This is the intended meaning of "once i look and click off, its done",
    and it is worth pinning because it is the one case where a board look
    changes the waiting list at all - by REMOVING the name it charted.

    `_select_review_alert` drops both the outgoing and the incoming symbol from
    the queue before deciding what to do with the outgoing chart. So looking at
    AAPL - which was waiting behind MUFG - and then clicking away leaves AAPL
    out: the trader has now seen it. No `skip` is written for it, because the
    look is not a shown alert; the alert it REPLACED (MUFG, a D1 row that held
    a place) still goes back to the head.
    """
    panel.add_alert(_d1("MUFG", "SHORT"))
    panel.add_alert(_d1("AAPL"))
    panel.add_alert(_d1("XOM"))
    assert panel._current_review_alert.symbol == "MUFG"
    assert [a.symbol for a in panel._review_queue] == ["AAPL", "XOM"]

    panel.rrs_snapshot.symbolActivated.emit("AAPL", "LONG")

    assert [a.symbol for a in panel._review_queue] == ["MUFG", "XOM"], (
        "AAPL left the waiting list because the trader is looking at it"
    )
    assert panel._current_review_alert.symbol == "AAPL"

    panel.entry_board.symbolActivated.emit("NVDA", "LONG")

    assert [a.symbol for a in panel._review_queue] == ["MUFG", "XOM"], (
        "and it did not come back when the look was clicked away from"
    )
    assert [action for action, _kw in events if action == "skip"] == []
