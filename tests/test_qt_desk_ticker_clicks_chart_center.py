"""On the Trading Desk, every ticker click lands on the centre chart (trader, 2026-09-03).

    "when i click on a ticker anywhere while on the trading desk tab, i want the
    chart to come up on the visual chart review chart we have in the center of
    that tab. right now i click things in the auto RS/RW board or the master
    avwap setups board and it does a pop up. thats fine on other tabs, but the
    main tab should always be centralized with the main chart"

What these pin:

* every board INSIDE the Alert Center (RS/RW, entry, Focus strength) charts in
  the review pane and opens no popup - the rule the M5 Strength Board got on
  2026-08-31, now for all of them;
* the setups column's four panels (setups table, RS Window, Industry Board,
  Watchlists) do the same while they are a column of the desk (workspace mode),
  and go back to the popup as tabs of their own, where the pane is not visible;
* the click uses `chart_symbol` - the lookup box's door - so the side travels,
  the row is a MANUAL_CHART and never a scanner alert, and nothing goes through
  `_enqueue_review_alert`;
* the popup is still the door for a board on ANOTHER page (the AWAY Recap).
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

_app = QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def qt_desk():
    from ui.app import MainWindow
    from ui.state import UiState

    window = MainWindow(UiState(workspace_mode="workspace"))
    yield window
    try:
        window.close()
    except Exception:
        pass


@pytest.fixture
def popups(monkeypatch):
    """Every popup the desk would have opened, as (symbol, side)."""
    import ui.widgets.symbol_snapshot_dialog as snapshot_dialog

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        snapshot_dialog,
        "show_symbol_snapshot",
        lambda owner, symbol, **kwargs: seen.append((symbol, kwargs.get("side") or "")),
    )
    return seen


def _charted(center):
    current = center._current_review_alert
    assert current is not None, "nothing reached the review pane"
    return current


def _setup_rows(desk):
    from ui.models.setup import SetupRow

    desk.master_panel.set_rows(
        [
            SetupRow(symbol="NVDA", side="LONG", score=90.0),
            SetupRow(symbol="TSLA", side="SHORT", score=80.0),
            SetupRow(symbol="AMD", side="LONG", score=70.0),
        ]
    )
    panel = desk.master_panel
    symbol_column = next(
        column for column, (key, _label) in enumerate(panel.model.COLUMNS) if key == "symbol"
    )
    return panel, symbol_column


# ---------------------------------------------------------------------------
# 1. The boards inside the Alert Center
# ---------------------------------------------------------------------------
def test_the_rs_rw_board_charts_in_the_review_pane(qt_desk, popups):
    center = qt_desk.trading_panel.alert_center

    center.rrs_snapshot.symbolActivated.emit("meta", "SHORT")

    assert popups == []
    current = _charted(center)
    assert (current.symbol, current.side) == ("META", "SHORT")
    assert center.chart_review.title.text().startswith("META")


def test_the_entry_board_charts_in_the_review_pane(qt_desk, popups):
    center = qt_desk.trading_panel.alert_center

    center.entry_board.symbolActivated.emit("NVDA", "LONG")

    assert popups == []
    assert _charted(center).symbol == "NVDA"


def test_the_focus_strength_board_charts_in_the_review_pane(qt_desk, popups):
    center = qt_desk.trading_panel.alert_center

    center.focus_strength.symbolActivated.emit("AMD", "LONG")

    assert popups == []
    assert _charted(center).symbol == "AMD"


def test_a_board_click_is_a_manual_chart_and_never_a_scanner_alert(qt_desk, popups, monkeypatch):
    from ui.models.bounce import MANUAL_CHART_TAG

    center = qt_desk.trading_panel.alert_center
    enqueued: list[object] = []
    monkeypatch.setattr(center, "_enqueue_review_alert", enqueued.append)
    feed_before = len(center._alerts)

    center.rrs_snapshot.symbolActivated.emit("SOXL", "LONG")

    assert enqueued == []
    assert len(center._alerts) == feed_before
    assert _charted(center).tag == MANUAL_CHART_TAG
    assert _charted(center).trigger == "Charted from the RS/RW board"


def test_a_feed_ticker_name_click_charts_the_alert_itself(qt_desk, popups):
    """The real alert with its trigger, not a manual chart of the same name."""
    from ui.models.bounce import BounceAlert

    center = qt_desk.trading_panel.alert_center
    alert = BounceAlert.from_callback("NVDA LONG [A-TIER] test bounce", "bounce")

    center._show_symbol_snapshot(alert)

    assert popups == []
    assert _charted(center) is alert


# ---------------------------------------------------------------------------
# 2. The setups column, as a column of the desk
# ---------------------------------------------------------------------------
def test_a_setups_table_symbol_click_charts_in_the_review_pane(qt_desk, popups):
    desk = qt_desk.trading_panel
    panel, symbol_column = _setup_rows(desk)

    panel.table.clicked.emit(panel.proxy.index(1, symbol_column))

    assert popups == []
    current = _charted(desk.alert_center)
    assert (current.symbol, current.side) == ("TSLA", "SHORT")
    assert current.trigger == "Charted from the Master AVWAP setups"


def test_the_space_walk_lands_in_the_review_pane_too(qt_desk, popups):
    desk = qt_desk.trading_panel
    panel, symbol_column = _setup_rows(desk)
    panel.table.setCurrentIndex(panel.proxy.index(0, symbol_column))

    panel._open_next_symbol_snapshot()

    assert popups == []
    assert _charted(desk.alert_center).symbol == "TSLA"


def test_an_rs_window_click_charts_in_the_review_pane(qt_desk, popups):
    desk = qt_desk.trading_panel
    panel = desk.rs_window_panel
    panel.model.set_rows([{"symbol": "AAA", "side": "LONG", "excess": 1.2}])

    panel.table.clicked.emit(panel.table.model().index(0, 0))

    assert popups == []
    current = _charted(desk.alert_center)
    assert (current.symbol, current.side) == ("AAA", "LONG")


def test_an_industry_board_etf_click_charts_in_the_review_pane(qt_desk, popups):
    from ui.panels.industry_panel import SECTOR_COLUMNS, _fill_table

    desk = qt_desk.trading_panel
    panel = desk.industry_panel
    _fill_table(
        panel.sector_table,
        SECTOR_COLUMNS,
        [{"etf": "XLK", "sector": "Technology", "rs_score": "2.5"}],
    )
    etf_column = next(
        index for index, (key, _label) in enumerate(SECTOR_COLUMNS) if key == "etf"
    )

    panel.sector_table.cellClicked.emit(0, etf_column)

    assert popups == []
    current = _charted(desk.alert_center)
    assert (current.symbol, current.side) == ("XLK", "LONG")


def test_a_watchlist_line_charts_in_the_review_pane(qt_desk, popups):
    desk = qt_desk.trading_panel

    desk.watchlists_panel._open_symbol_snapshot("AAPL")

    assert popups == []
    assert _charted(desk.alert_center).symbol == "AAPL"


# ---------------------------------------------------------------------------
# 3. Other tabs keep the popup
# ---------------------------------------------------------------------------
def test_as_tabs_of_their_own_the_setups_panels_keep_the_popup(qt_desk, popups):
    """The pane is on a different tab there, so a chart in it would be unseen."""
    desk = qt_desk.trading_panel
    panel, symbol_column = _setup_rows(desk)
    desk.alert_center.chart_symbol("SPY")
    try:
        desk.set_mode("tabs")

        panel.table.clicked.emit(panel.proxy.index(0, symbol_column))

        assert popups == [("NVDA", "LONG")]
        assert _charted(desk.alert_center).symbol == "SPY", "the pane was not touched"
    finally:
        desk.set_mode("workspace")

    panel.table.clicked.emit(panel.proxy.index(2, symbol_column))
    assert popups == [("NVDA", "LONG")], "back on the desk, back to the pane"
    assert _charted(desk.alert_center).symbol == "AMD"


def test_a_board_on_another_page_still_opens_the_popup(qt_desk, popups):
    """`show_board_symbol` is the AWAY Recap's door and it is not on the desk."""
    center = qt_desk.trading_panel.alert_center
    center.chart_symbol("SPY")

    center.show_board_symbol("QQQ", "LONG")

    assert popups == [("QQQ", "LONG")]
    assert _charted(center).symbol == "SPY"


def test_a_standalone_panel_without_a_sink_keeps_the_popup(popups, tmp_path, monkeypatch):
    """No desk, no sink: exactly the behaviour every existing panel test pins."""
    import chart_snapshot
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: [])
    panel = MasterAvwapPanel(None, review_events_path=tmp_path / "events.jsonl")
    panel.set_rows([SetupRow(symbol="NVDA", side="LONG", score=90.0)])
    symbol_column = next(
        column for column, (key, _label) in enumerate(panel.model.COLUMNS) if key == "symbol"
    )

    panel.table.clicked.emit(panel.proxy.index(0, symbol_column))

    assert popups == [("NVDA", "LONG")]
    panel.close()
