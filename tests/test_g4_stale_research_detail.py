"""Packet G4 - a detail pane never outlives the context that opened it.

The GUI review of 2026-09-06 found the Day-trade Tracker showing the
`lrsi_cross50` explanation while the Combos tab was open with no combo
selected. `ResearchExplanationView.show_row` sets HTML and `setVisible(True)`
and nothing ever takes it back down: `self.tabs` and `self.decisions_tabs` have
no `currentChanged` handler, and `_on_refresh_finished` / `_on_held_run_loaded`
replace every model row without touching the pane. So the last row the trader
clicked stays on screen through every tab switch and every re-aggregation,
beside a table that no longer contains it. That is a correctness defect, not
polish: the numbers on the right are read as the numbers on the left.

Every test here drives the REAL path - the table's own `clicked` signal into the
lambda the panel connected, `QTabWidget.setCurrentIndex` into the real
`currentChanged`, and `_refreshFinished.emit` into the real refresh slot, which
is exactly what the worker thread does. The disk reads are monkeypatched at the
panel module's own seams (`_load_performance_rows`, `load_bounce_learning_state`,
`load_held_run_report`) so no live store is touched and the fixture numbers are
the ones under test.

Visibility is asserted with `isHidden()` rather than `isVisible()`: the panel is
never `show()`n in a test, so a child of a hidden parent reports
`isVisible() == False` whatever the pane itself was told. `isHidden()` is the
widget's own explicit hide flag and is the thing `setVisible` moves.

Items: G4.1 (the widget gains `clear()` and `shown_identity`) and G4.2 (the
Day-trade Tracker clears on a context change and re-reads on a data revision).
G4.3 / tests 5-6 are DEFERRED to packet G4b and are deliberately absent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6", reason="the Day-trade Tracker is a Qt widget")

from ui.panels import daytrade_tracker_panel as panel_module  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures: a panel whose every disk read is a fixture
# ---------------------------------------------------------------------------


def _performance_row(dimension: str, segment: str, avg_close_r: str, **extra) -> dict:
    """One row shaped like a line of `intraday_bounce_performance.csv`.

    Every value is a STRING because `_load_performance_rows` builds these with
    `csv.DictReader`, and a real row has each column PRESENT - blank rather than
    absent - which is what the aggregate writer emits.
    """
    row = {
        "dimension": dimension,
        "direction": "long",
        "segment": segment,
        "sample_count": "40",
        "avg_close_r": avg_close_r,
        "median_close_r": "0.20",
        "avg_mfe_r": "1.10",
        "avg_mae_r": "-0.50",
        "positive_eod_rate": "0.55",
        "target_1r_rate": "0.50",
        "target_2r_rate": "0.20",
        "stop_rate": "0.30",
        "recommendation": "keep",
        "example_symbols": "AAPL",
    }
    row.update(extra)
    return row


#: The state the panel opens on: one Bounce Types row and one Combos row, so a
#: switch from the first tab to the second lands on a table that has rows of its
#: own - the defect is not "the next table is empty", it is "the pane still
#: describes the previous one".
BASE_ROWS = (
    _performance_row("bounce_type", "vwap", "0.40"),
    _performance_row("bounce_type", "ema9", "0.15"),
    _performance_row("bounce_combo", "vwap+ema9", "0.25"),
)


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class _Panel:
    """A built panel plus the mutable fixture its refresh reads back."""

    def __init__(self, panel, app, rows_box: list) -> None:
        self.panel = panel
        self.app = app
        self._rows_box = rows_box

    # -- driving the real seams ------------------------------------------
    def table(self, dimension: str):
        return self.panel._dimension_tables[dimension][0]

    def index_of(self, dimension: str, segment: str):
        """The proxy index of the row whose `segment` is `segment`.

        Found by scanning the model rather than assuming a position: the table
        sorts by the headline and an unmeasured fixture leaves the order to the
        sort's tie-break.
        """
        from ui.models.tracker_table_model import ROW_ROLE

        table = self.table(dimension)
        proxy = table.model()
        for position in range(proxy.rowCount()):
            index = proxy.index(position, 0)
            row = index.data(ROW_ROLE)
            if isinstance(row, dict) and str(row.get("segment")) == segment:
                return index
        raise AssertionError(f"no {dimension} row for segment {segment!r}")

    def click(self, dimension: str, segment: str) -> None:
        """The table's own `clicked` signal - the connection the panel made."""
        self.table(dimension).clicked.emit(self.index_of(dimension, segment))
        self.app.processEvents()

    def switch_dimension_tab(self, dimension: str) -> None:
        keys = [key for key, _label in panel_module.DIMENSION_TABS]
        self.panel.tabs.setCurrentIndex(keys.index(dimension))
        self.app.processEvents()

    def switch_decision_tab(self, dimension: str) -> None:
        keys = [key for key, _label in panel_module.DECISION_TABS]
        self.panel.decisions_tabs.setCurrentIndex(keys.index(dimension))
        self.app.processEvents()

    def refresh_with(self, rows) -> None:
        """A data revision through the real slot the worker thread fires."""
        self._rows_box[:] = [dict(row) for row in rows]
        self.panel._refreshFinished.emit("bounce learning refreshed")
        self.app.processEvents()

    # -- reading the pane -------------------------------------------------
    @property
    def view(self):
        return self.panel.explanation_view

    @property
    def text(self) -> str:
        return self.view.toPlainText()

    def settle(self, predicate, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return bool(predicate())


@pytest.fixture
def tracker(qapp, monkeypatch):
    """The panel with every read replaced by a fixture, construction settled."""
    rows_box: list = [dict(row) for row in BASE_ROWS]

    monkeypatch.setattr(
        panel_module, "_load_performance_rows", lambda: [dict(row) for row in rows_box]
    )
    monkeypatch.setattr(panel_module, "load_bounce_learning_state", lambda: {})
    monkeypatch.setattr(
        panel_module,
        "load_held_run_report",
        lambda: {"summaries": {}, "window": {}, "outcome_coverage": {}},
    )
    import review_learning

    monkeypatch.setattr(
        review_learning, "load_review_learning_state", lambda *a, **k: None
    )

    made = panel_module.DaytradeTrackerPanel()
    # G7.1: the constructor no longer reads on the Qt thread - the first show
    # does - so this fixture asks for the read the page under test needs. The
    # trigger only; every assertion below is unchanged.
    made.reload_from_disk()
    made.start_decisions_refresh(rebuild=False)
    harness = _Panel(made, qapp, rows_box)
    # `start_decisions_refresh` is single-flight; the button coming back is
    # exactly "the construction read has landed and nothing is in flight".
    assert harness.settle(lambda: made.decisions_button.isEnabled()), (
        "the construction read never finished"
    )
    try:
        yield harness
    finally:
        made.shutdown()
        made.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# 1. a tab switch is a context change
# ---------------------------------------------------------------------------


def test_switching_to_a_different_dimension_tab_clears_the_explanation(tracker):
    """The exact thing the GUI review saw: a bounce-type explanation still
    standing over the Combos table with no combo selected."""
    tracker.click("bounce_type", "vwap")
    assert not tracker.view.isHidden(), "the click should have opened the pane"
    assert "vwap" in tracker.text

    tracker.switch_dimension_tab("bounce_combo")

    assert tracker.view.isHidden(), (
        "the Combos tab is showing and the pane still describes a bounce type"
    )
    assert tracker.text.strip() == "", "a cleared pane keeps no text to come back"


# ---------------------------------------------------------------------------
# 2. a data revision repaints the same row from the NEW numbers
# ---------------------------------------------------------------------------


def test_a_refresh_repaints_the_open_row_with_the_new_number(tracker):
    """Re-aggregation moved this segment's average R from +0.40 to +1.25.

    The pane stays open - the row is still in the table the trader is looking
    at - and it must show +1.25R. Showing +0.40R beside a table that now reads
    +1.25 is the same defect wearing a number instead of a name, and re-showing
    the CACHED row dict would reproduce it exactly.
    """
    tracker.click("bounce_type", "vwap")
    assert "+0.40R" in tracker.text

    revised = [
        _performance_row("bounce_type", "vwap", "1.25"),
        _performance_row("bounce_type", "ema9", "0.15"),
        _performance_row("bounce_combo", "vwap+ema9", "0.25"),
    ]
    tracker.refresh_with(revised)

    assert not tracker.view.isHidden(), (
        "the row is still in the current table, so the pane stays open"
    )
    assert "+1.25R" in tracker.text, "the pane is showing the pre-refresh number"
    assert "+0.40R" not in tracker.text


# ---------------------------------------------------------------------------
# 3. a data revision that drops the row closes the pane
# ---------------------------------------------------------------------------


def test_a_refresh_that_drops_the_open_row_hides_the_explanation(tracker):
    """The segment is gone from the table; the explanation of it must go too.

    The table is NOT empty afterwards - `ema9` survives - so a pane that stays
    up is describing a row the trader can no longer see, not merely lagging an
    empty table.
    """
    tracker.click("bounce_type", "vwap")
    assert not tracker.view.isHidden()

    tracker.refresh_with(
        [
            _performance_row("bounce_type", "ema9", "0.15"),
            _performance_row("bounce_combo", "vwap+ema9", "0.25"),
        ]
    )

    assert tracker.index_of("bounce_type", "ema9") is not None, (
        "the fixture must leave the table populated, or this proves nothing"
    )
    with pytest.raises(AssertionError):
        tracker.index_of("bounce_type", "vwap")

    assert tracker.view.isHidden(), (
        "the pane is explaining a segment the refreshed table no longer holds"
    )
    assert tracker.text.strip() == ""


# ---------------------------------------------------------------------------
# 4. the My Decisions sub-tabs are a context change too
# ---------------------------------------------------------------------------


def test_switching_my_decisions_sub_tabs_clears_the_explanation(tracker):
    """`decisions_tabs` is the second tab strip and has the same rule.

    One `currentChanged` handler on the outer strip is not the whole fix: the
    trader moves between Veto Reasons, Tier and the rest inside My Decisions,
    and the pane must not survive that move either.
    """
    tracker.click("bounce_type", "vwap")
    assert not tracker.view.isHidden()

    tracker.switch_decision_tab("tier")

    assert tracker.view.isHidden(), (
        "a My Decisions sub-tab changed under an explanation that outlived it"
    )
    assert tracker.text.strip() == ""


# ---------------------------------------------------------------------------
# 7. the widget itself (G4.1)
# ---------------------------------------------------------------------------


def test_the_explanation_view_can_be_cleared_and_remembers_what_it_shows(qapp):
    """`clear()` empties AND hides; `show_row(..., identity=...)` records it.

    `QTextEdit` already gives this class a `clear()` that empties the document
    and leaves the widget standing, so "it has a clear()" is not the fix - the
    override has to take the pane down as well, which is what a caller asking
    for a context change means by it. `shown_identity` is how a caller asks
    "is what I am about to draw the thing you are already showing"; without it
    every re-show is a guess from the display text.
    """
    from ui.widgets.research_explanation_view import ResearchExplanationView

    view = ResearchExplanationView()
    try:
        row = {
            "dimension": "bounce_type",
            "direction": "long",
            "segment": "vwap",
            "sample_count": "40",
            "avg_close_r": "0.40",
        }
        identity = ("daytrade_performance", "bounce_type", "vwap")

        view.show_row("daytrade_performance", row)
        assert not view.isHidden()
        assert view.toPlainText().strip() != ""

        view.clear()
        assert view.toPlainText().strip() == "", "clear() left text behind"
        assert view.isHidden(), (
            "clear() emptied the pane and left the empty pane on screen"
        )

        view.show_row("daytrade_performance", row, identity=identity)
        assert not view.isHidden()
        assert view.shown_identity == identity

        view.clear()
        assert view.shown_identity is None, (
            "a cleared pane still claims to be showing something"
        )
    finally:
        view.deleteLater()
        qapp.processEvents()
