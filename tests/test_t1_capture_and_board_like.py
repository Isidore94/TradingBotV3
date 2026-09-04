"""Packets T1.2 and T2 - the two likes, and only one of them moves the chart.

Trader, 2026-09-04, first pass (T1.2):

    "Additionally the 'like' button in the visual chart review should NOT
    advance the char to the next page because i still need time to enter alerts
    etc."

Trader, 2026-09-04, second pass (T2), about the capture tab's claim list:

    "pretty close. for the 'like and claim' part of the capture tab, a double
    click of any of the setups there should be sufficient. I shouldnt have to
    type anything below that box. and then double clicking that box should
    advance the chart."

So the two like modes now part company, and this file pins the split:

* a **CLAIMED** like (Alt+K, a digit, a double-click on a setup, the rail's
  "Like + claim setup" button) needs NO why, commits on the gesture, and
  **ADVANCES** the chart - the pane's `likeAdvanceRequested`, the panel's
  `_advance_after_like`;
* a **QUICK** like (Alt+L, the rail's "♥ Quick like", the chart's "♥ Like"
  button) is unchanged by T2: still no claim, still `likeRecorded` ->
  `_after_like`, and the chart still **STAYS** so the trader can arm alerts;
* both record the review event **`like_advance`** - historical, because
  `review_learning.TAKE_ACTIONS` keys on that exact string - through ONE shared
  helper, so the two handlers cannot drift;
* neither places anything, neither parks the symbol, neither drops a Focus
  pick, and neither sweeps the symbol's other queued alerts.

The "♥ Like" button's OPTIONAL note box (P9) stays - it is on the chart, not in
the capture window, and the trader did not name it.
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

from PySide6.QtWidgets import QApplication, QInputDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _neuter_cohort_merges(rail) -> None:
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True, "added": 0}


def _d1_alert(symbol: str, side: str = "LONG"):
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


@pytest.fixture
def panel(tmp_path, monkeypatch):
    """AAPL on the chart, NVDA waiting behind it, every store in tmp."""
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
    monkeypatch.setattr(
        made.chart_review.capture_rail,
        "_annotations_path",
        tmp_path / "trader_annotations.jsonl",
    )
    _neuter_cohort_merges(made.chart_review.capture_rail)
    made.add_alert(_d1_alert("AAPL"))
    made.add_alert(_d1_alert("NVDA", "SHORT"))
    assert made._current_review_alert.symbol == "AAPL"
    assert [a.symbol for a in made._review_queue] == ["NVDA"]
    yield made
    made.close()
    made.deleteLater()


def _claim(rail, why: str = "reclaimed the band") -> None:
    rail.setup_list.setCurrentRow(0)
    rail.like_note_input.setText(why)


def _assert_chart_advanced(panel, gone: str = "AAPL", now: str = "NVDA") -> None:
    """T2: a CLAIMED like moves on to the next waiting chart."""
    assert panel._current_review_alert is not None, (
        "the claimed like must advance to the next chart, not clear the pane"
    )
    assert panel._current_review_alert.symbol == now, (
        f"the chart is still {panel._current_review_alert.symbol}; a claimed "
        "like advances (trader, 2026-09-04 second pass)"
    )
    assert [a.symbol for a in panel._review_queue] == [], (
        f"{gone} must not be re-queued behind the chart it just left"
    )
    assert gone not in panel._ignored_symbols, "a like never parks the symbol"


def _assert_chart_stayed(
    panel, symbol: str = "AAPL", queue: tuple[str, ...] = ("NVDA",)
) -> None:
    assert panel._current_review_alert is not None, "the chart was taken away"
    assert panel._current_review_alert.symbol == symbol, (
        f"the chart advanced to {panel._current_review_alert.symbol}; the trader "
        "still needs it to arm an alert"
    )
    assert [a.symbol for a in panel._review_queue] == list(queue), (
        "the waiting list must be untouched by a quick like"
    )
    assert symbol not in panel._ignored_symbols, "a like never parks the symbol"


# ---------------------------------------------------------------------------
# T2: a CLAIMED like advances; every QUICK like path still leaves the chart
# ---------------------------------------------------------------------------
def test_a_claimed_like_advances_the_chart(panel):
    """Alt+K or a digit, with or without a why - the claimed like ADVANCES.

    Rewritten for packet T2 (trader, 2026-09-04 second pass: *"double clicking
    that box should advance the chart"*). It asserted the chart stayed up under
    T1.2, which is now true of the QUICK like only.
    """
    rail = panel.chart_review.capture_rail
    _claim(rail)

    assert rail.commit_like() is not None

    _assert_chart_advanced(panel)


def test_a_double_clicked_claim_advances_the_chart(panel):
    """The trader's actual gesture: `setup_list.itemActivated` -> `commit_like`.

    Rewritten for packet T2: the double-click commits and advances.
    """
    rail = panel.chart_review.capture_rail
    _claim(rail, "second test at the 20")

    rail.setup_list.itemActivated.emit(rail.setup_list.item(0))

    _assert_chart_advanced(panel)


def test_a_double_clicked_claim_with_nothing_typed_writes_one_row_and_advances(
    panel, tmp_path
):
    """T2's whole point: *"I shouldnt have to type anything below that box."*

    A double-click on a setup with an EMPTY why writes ONE `like_claim` row
    carrying the claim, with no why on it, and moves to the next chart.
    """
    import json

    rail = panel.chart_review.capture_rail
    rail.like_note_input.setText("")

    rail.setup_list.itemActivated.emit(rail.setup_list.item(0))

    path = tmp_path / "trader_annotations.jsonl"
    assert path.exists(), "a claim with no why must still write its row"
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["event_type"] for row in lines] == ["like_claim"], "exactly one row"
    assert lines[0]["claimed_setup_id"], "the claim is what the row is for"
    assert lines[0].get("note", "") == "", "nothing was typed, so there is no why"
    assert lines[0]["like_mode"] == "claimed"
    _assert_chart_advanced(panel)


def test_the_rails_quick_like_button_leaves_the_chart_up(panel, monkeypatch):
    """The rail's own "♥ Quick like" is the other verb and it does not move."""
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *a, **k: ("", True))
    rail = panel.chart_review.capture_rail

    assert rail.prompt_quick_like() is not None

    _assert_chart_stayed(panel)


def test_a_quick_like_by_key_leaves_the_chart_up(panel):
    """Alt+L: one key, no claim, no box - and now no advance either."""
    rail = panel.chart_review.capture_rail

    assert rail.commit_quick_like() is not None

    _assert_chart_stayed(panel)


def test_the_chart_like_button_leaves_the_chart_up(panel, monkeypatch):
    """The "♥ Like" button still offers its optional note; OK with an empty
    box is a plain quick like, and the chart stays."""
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *a, **k: ("", True))

    panel.chart_review.quick_like_button.click()

    _assert_chart_stayed(panel)


def test_the_like_button_still_offers_its_optional_note_box(panel, monkeypatch):
    """P9 is untouched: the BUTTON prompts, and Cancel records nothing."""
    asked: list = []
    monkeypatch.setattr(
        QInputDialog,
        "getMultiLineText",
        lambda *a, **k: (asked.append(True), ("", False))[1],
    )

    panel.chart_review.quick_like_button.click()

    assert asked == [True], "the chart button's optional note box stays"
    _assert_chart_stayed(panel)


# ---------------------------------------------------------------------------
# what a like still records
# ---------------------------------------------------------------------------
def test_a_like_still_records_like_advance_under_its_historical_name(
    panel, monkeypatch
):
    """`review_learning.TAKE_ACTIONS` keys on the string `like_advance`.

    Renaming it would drop every past like out of the take side of the
    scoreboard, so the name stays for BOTH modes. Rewritten for packet T2: the
    claimed like advances again, the quick like still does not, and the two
    handlers record through one shared helper so they cannot drift.
    """
    import review_learning

    recorded: list[str] = []
    monkeypatch.setattr(
        panel, "_record_review_event", lambda action, **kw: recorded.append(action)
    )
    rail = panel.chart_review.capture_rail
    _claim(rail)
    assert rail.commit_like() is not None

    assert "like_advance" in recorded
    assert "like_advance" in review_learning.TAKE_ACTIONS
    assert "remove_today" not in recorded
    assert "skip" not in recorded
    _assert_chart_advanced(panel)

    # And the quick like on the chart that is now up records the same action.
    recorded.clear()
    assert rail.commit_quick_like() is not None
    assert recorded == ["like_advance"]
    _assert_chart_stayed(panel, "NVDA", queue=())


def test_a_like_writes_exactly_one_annotation_row(panel, tmp_path):
    """Unchanged: the like is one row and no placement."""
    import json

    rail = panel.chart_review.capture_rail
    _claim(rail, "clean base")
    assert rail.commit_like() is not None

    lines = [
        json.loads(line)
        for line in (tmp_path / "trader_annotations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["event_type"] for row in lines] == ["like_claim"]
    assert lines[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# the names: two signals, one per mode, each doing what it is called
# ---------------------------------------------------------------------------
def test_the_pane_offers_both_like_signals_one_per_mode(panel):
    """Rewritten for packet T2: BOTH signals exist and the mode picks one.

    T1.2 asserted `likeAdvanceRequested` was absent. It is back, because there
    is now a like that really does advance - and `likeRecorded` stays for the
    quick like that really does not.
    """
    pane = panel.chart_review
    assert hasattr(pane, "likeRecorded"), "the quick like reports and stays"
    assert hasattr(pane, "likeAdvanceRequested"), "the claimed like advances"

    fired: list[str] = []
    pane.likeRecorded.connect(lambda _a: fired.append("recorded"))
    pane.likeAdvanceRequested.connect(lambda _a: fired.append("advance"))
    rail = pane.capture_rail

    _claim(rail)
    assert rail.commit_like() is not None
    assert fired == ["advance"], "a CLAIMED like fires the advance signal"

    fired.clear()
    assert rail.commit_quick_like() is not None
    assert fired == ["recorded"], "a QUICK like fires the report signal"


def test_the_panel_has_a_handler_for_each_mode(panel):
    """Rewritten for packet T2: `_after_like` stays, `_advance_after_like` returns."""
    assert callable(getattr(panel, "_after_like", None))
    assert callable(getattr(panel, "_advance_after_like", None))


def test_no_tooltip_on_the_like_button_still_claims_the_chart_moves_on(panel):
    """A tooltip is the only place this rule is written on screen."""
    tip = panel.chart_review.quick_like_button.toolTip().lower()
    assert "moves on" not in tip
    assert "retire" not in tip
