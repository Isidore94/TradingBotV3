"""Packet T1.2 - a LIKE records the judgement and leaves the chart up.

Trader, 2026-09-04:

    "Additionally the 'like' button in the visual chart review should NOT
    advance the char to the next page because i still need time to enter alerts
    etc."

On `main` @ 6e05878 every like path - the claimed like (Alt+K / digit /
double-click / the rail button), the quick like by key (Alt+L) and the chart's
"♥ Like" button - reaches `_record_like`, emits `captured(EVENT_LIKE_CLAIM)`,
and `AlertChartReview._on_captured` fires `likeAdvanceRequested`, which the
panel answers with `_advance_after_like` -> `_advance_review_queue`. The chart
the trader wanted to arm an alert on is gone before they can.

The new contract:

* the pane's signal is `likeRecorded` (a signal called "advance" that does not
  advance is a lie) and the panel's handler is `_after_like`;
* the review event keeps the name **`like_advance`** - historical, because
  `review_learning.TAKE_ACTIONS` keys on that exact string - and now means
  "liked; the symbol keeps alerting and the chart stays";
* nothing else changes: no ignore-set entry, no Focus drop, no placement, and
  the symbol is still marked reviewed today.

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


def _assert_chart_stayed(panel, symbol: str = "AAPL") -> None:
    assert panel._current_review_alert is not None, "the chart was taken away"
    assert panel._current_review_alert.symbol == symbol, (
        f"the chart advanced to {panel._current_review_alert.symbol}; the trader "
        "still needs it to arm an alert"
    )
    assert [a.symbol for a in panel._review_queue] == ["NVDA"], (
        "the waiting list must be untouched by a like"
    )
    assert symbol not in panel._ignored_symbols, "a like never parks the symbol"


# ---------------------------------------------------------------------------
# every like path leaves the chart alone
# ---------------------------------------------------------------------------
def test_a_claimed_like_leaves_the_chart_up(panel):
    """Alt+K, a digit, the why, Enter - the claimed like."""
    rail = panel.chart_review.capture_rail
    _claim(rail)

    assert rail.commit_like() is not None

    _assert_chart_stayed(panel)


def test_a_double_clicked_claim_leaves_the_chart_up(panel):
    """The trader's actual gesture: `setup_list.itemActivated` -> `commit_like`."""
    rail = panel.chart_review.capture_rail
    _claim(rail, "second test at the 20")

    rail.setup_list.itemActivated.emit(rail.setup_list.item(0))

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
    scoreboard, so the name stays and its MEANING changes: liked, the symbol
    keeps alerting, and since 2026-09-04 the chart stays.
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
    _assert_chart_stayed(panel)


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
# the names, because a signal called "advance" that does not advance is a lie
# ---------------------------------------------------------------------------
def test_the_pane_offers_like_recorded_and_no_advance_signal(panel):
    pane = panel.chart_review
    assert hasattr(pane, "likeRecorded"), "the like's signal is `likeRecorded` now"
    assert not hasattr(pane, "likeAdvanceRequested"), (
        "a signal named 'advance' that does not advance is a lie"
    )


def test_the_panel_handler_is_named_after_like(panel):
    assert callable(getattr(panel, "_after_like", None))
    assert not hasattr(panel, "_advance_after_like")


def test_no_tooltip_on_the_like_button_still_claims_the_chart_moves_on(panel):
    """A tooltip is the only place this rule is written on screen."""
    tip = panel.chart_review.quick_like_button.toolTip().lower()
    assert "moves on" not in tip
    assert "retire" not in tip
