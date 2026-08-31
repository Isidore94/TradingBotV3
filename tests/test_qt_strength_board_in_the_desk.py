"""The M5 Strength Board lives in the Desk's Strength window (trader 2026-08-31).

    "The Strength Board tab is good but it really should be modified to fit in
    the 'strength' window in the trading desk - either integrated directly or
    be positioned below it."

Positioned below it, and the left-nav page is gone. What these tests pin is the
part that is easy to get wrong when a surface moves house:

* the page is removed from **every** structure that tracks pages, not two of
  three - the bug `test_qt_page_specs` was written for;
* `StrengthBoardService` is still ONE object with ONE timer, so the move added
  no second fetcher;
* the add path still re-runs the M5 Focus adoption gate at click time;
* nothing in the strength path constructs an IB client - the board is batched
  yfinance and **zero IB traffic**, which is what keeps the locked pacing
  budget out of this.
"""

from __future__ import annotations

import ast
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


# ---------------------------------------------------------------------------
# 1. The page is gone from every site that tracks pages
# ---------------------------------------------------------------------------
def test_the_left_nav_no_longer_carries_a_strength_board_page():
    from ui.app import PAGE_SPECS

    titles = [spec.title for spec in PAGE_SPECS]
    assert "Strength Board" not in titles
    assert "strength_board_panel" not in [spec.attribute for spec in PAGE_SPECS]


def test_every_page_tracking_site_agrees_after_the_removal(qt_desk):
    """PAGE_SPECS, the nav buttons and the page stack, all three."""
    from ui.app import PAGE_SPECS

    assert [b.text() for b in qt_desk.nav_buttons] == [s.title for s in PAGE_SPECS]
    assert qt_desk.pages.count() == len(PAGE_SPECS)
    for index, spec in enumerate(PAGE_SPECS):
        qt_desk._select_page(index)
        assert qt_desk.title_label.text() == spec.title


def test_the_window_no_longer_builds_a_strength_board_page_attribute(qt_desk):
    """The panel is a child of the Alert Center now, not a window attribute."""
    assert not hasattr(qt_desk, "strength_board_panel")


# ---------------------------------------------------------------------------
# 2. One service, one timer
# ---------------------------------------------------------------------------
def test_the_desk_builds_exactly_one_strength_board_service(qt_desk):
    from ui.services.strength_board_service import StrengthBoardService

    found = qt_desk.findChildren(StrengthBoardService)
    assert len(found) == 1
    assert found[0] is qt_desk.strength_board_service


def test_the_hosted_panel_reads_the_one_service(qt_desk):
    panel = qt_desk.trading_panel.alert_center.strength_board
    assert panel is not None
    assert panel.service is qt_desk.strength_board_service


def test_one_interval_fires_the_service_start_once(qt_desk, monkeypatch):
    """Single ownership, measured rather than asserted: drive the ONE timer and
    count the fetch attempts. A second consumer with its own timer, or a panel
    that refreshed on show, would raise this count."""
    service = qt_desk.strength_board_service
    starts: list[bool] = []
    monkeypatch.setattr(service, "_due", lambda _now: True)
    monkeypatch.setattr(service, "_start", lambda manual=False: starts.append(manual))

    service._timer.timeout.emit()

    assert starts == [False], "one interval, one fetch attempt"


def test_the_panel_owns_no_clock_of_its_own():
    """No surface grew its own timer when the board changed hosts."""
    source = (SCRIPTS_DIR / "ui" / "panels" / "strength_board_panel.py").read_text(
        encoding="utf-8"
    )
    assert "QTimer" not in source, "the panel shows; the service owns the clock"
    assert "refresh_now" in source, "and asks the one owner when the trader clicks"


# ---------------------------------------------------------------------------
# 3. The gate still runs at click time
# ---------------------------------------------------------------------------
def _row(symbol, *, last, prev_high, prev_low, vwap):
    return {
        "symbol": symbol, "strength": 10.0, "last": last,
        "prev_high": prev_high, "prev_low": prev_low, "session_vwap": vwap,
        "day_pct": 1.5, "vwap_distance_pct": 0.8,
    }


class _Signal:
    def connect(self, _slot):
        pass


class _Service:
    def __init__(self, board):
        self._board = board
        self.boardChanged = _Signal()
        self.statusChanged = _Signal()

    def board(self):
        return dict(self._board)

    def status_text(self):
        return "Strength board: test"

    def refresh_now(self):
        return True


def _hosted_panel(board, tmp_path):
    from focus_picks import FocusPickStore
    from ui.panels.strength_board_panel import StrengthBoardPanel
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )
    panel = StrengthBoardPanel(service=_Service(board), focus_service=FocusService(store))
    return panel, store


def test_an_add_from_the_embedded_board_still_calls_the_gate(tmp_path, monkeypatch):
    board = {
        "long": [_row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=101.0)],
        "short": [],
    }
    panel, store = _hosted_panel(board, tmp_path)
    calls: list[tuple] = []
    import focus_adoption_gate

    real = focus_adoption_gate.passes_focus_adoption_gate

    def _spy(side, last, prev_high, prev_low, vwap):
        calls.append((side, last, prev_high, prev_low, vwap))
        return real(side, last, prev_high, prev_low, vwap)

    monkeypatch.setattr(focus_adoption_gate, "passes_focus_adoption_gate", _spy)

    panel._add_one("NVDA", "long")

    assert calls == [("long", 105.0, 100.0, 98.0, 101.0)]
    assert store.focus_symbols("long", "m5") == ["NVDA"]


def test_a_stale_row_is_still_refused_from_the_embedded_board(tmp_path):
    """The board is up to 15 minutes old wherever it is drawn."""
    board = {
        "long": [_row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=106.0)],
        "short": [],
    }
    panel, store = _hosted_panel(board, tmp_path)
    messages: list[str] = []
    panel.statusChanged.connect(messages.append)

    panel._add_one("NVDA", "long")

    assert store.focus_symbols("long", "m5") == []
    assert any("not above session VWAP" in text for text in messages)


# ---------------------------------------------------------------------------
# 4. Zero IB traffic
# ---------------------------------------------------------------------------
_STRENGTH_PATH_FILES = (
    SCRIPTS_DIR / "ui" / "services" / "strength_board_service.py",
    SCRIPTS_DIR / "ui" / "panels" / "strength_board_panel.py",
    SCRIPTS_DIR / "strength_scan.py",
)

_IB_NAMES = ("EClient", "EWrapper", "IBApi", "reqHistoricalData", "reqMktData")


def test_nothing_in_the_strength_path_constructs_an_ib_client():
    """Batched yfinance only. An IB client here would spend the locked pacing
    budget in `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` sec 5.2-5.3."""
    for path in _STRENGTH_PATH_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("ibapi"), f"{path.name}: {alias.name}"
            if isinstance(node, ast.ImportFrom):
                assert not str(node.module or "").startswith("ibapi"), path.name
            if isinstance(node, ast.Name):
                assert node.id not in _IB_NAMES, f"{path.name} names {node.id}"
            if isinstance(node, ast.Attribute):
                assert node.attr not in _IB_NAMES, f"{path.name} names {node.attr}"


def test_the_board_fetch_goes_through_the_batched_yfinance_helper():
    source = (SCRIPTS_DIR / "ui" / "services" / "strength_board_service.py").read_text(
        encoding="utf-8"
    )
    assert "fetch_intraday_profiles" in source
    assert "zero ib traffic" in source.lower()


# ---------------------------------------------------------------------------
# 5. Where it sits
# ---------------------------------------------------------------------------
def test_the_board_sits_under_the_strength_window_not_beside_the_charts(qt_desk):
    from ui.widgets.collapsible_section import CollapsibleSection
    from ui.widgets.focus_strength_board import FocusStrengthBoard

    center = qt_desk.trading_panel.alert_center
    column = center.strength_column
    order = [column.layout().itemAt(i).widget() for i in range(column.layout().count())]
    assert isinstance(order[0], FocusStrengthBoard)
    assert isinstance(order[1], CollapsibleSection)
    # The section's body is a scroll area so the board's own minimum width
    # stops there instead of reaching the desk splitter and widening the alert
    # column at the charts' expense.
    assert order[1].content().widget() is center.strength_board


def test_the_section_starts_collapsed_so_it_steals_no_space(qt_desk):
    """Default-off costs the charts nothing; the trader opens it when wanted."""
    center = qt_desk.trading_panel.alert_center
    section = center.strength_board_section
    assert section.is_expanded() is False
    assert section.content().isVisible() is False
    # And closed, it costs the alert column nothing: the header takes the width
    # it is given rather than demanding its title, and the board's own 270 px
    # minimum is held behind the scroll area.
    tabs_floor = center.tabs.minimumWidth()
    assert tabs_floor + center.strength_column.minimumSizeHint().width() <= 360


def test_expanding_the_section_never_moves_the_arm_bar(qt_desk):
    """The charts own the review pane. This section is in the ALERT column."""
    center = qt_desk.trading_panel.alert_center
    before = center.chart_review.arm_bar.parentWidget()
    center.strength_board_section.set_expanded(True)
    try:
        assert center.chart_review.arm_bar.parentWidget() is before
    finally:
        center.strength_board_section.set_expanded(False)


# ---------------------------------------------------------------------------
# 6. A row click charts into the Visual Alert Review pane, not a popup
#    (trader, 2026-08-31: "when I click on a stock in this M5 strength board
#    it should come up on the Visual chart review in the trading desk")
# ---------------------------------------------------------------------------
def test_a_row_click_charts_in_the_review_pane_and_opens_no_popup(qt_desk, monkeypatch):
    center = qt_desk.trading_panel.alert_center
    popups: list[str] = []
    monkeypatch.setattr(
        center, "_show_board_symbol_snapshot", lambda *a, **k: popups.append("popup")
    )

    center.strength_board.symbolActivated.emit("nvda", "long")

    assert popups == [], "the popup was the old page's answer, not this one"
    current = center._current_review_alert
    assert current is not None and current.symbol == "NVDA"
    assert center.chart_review.title.text().startswith("NVDA")


def test_the_side_travels_with_the_click(qt_desk):
    """A short charted as a plain WATCH reads as the wrong thesis."""
    center = qt_desk.trading_panel.alert_center

    center.strength_board.symbolActivated.emit("soxs", "short")

    assert center._current_review_alert.side == "SHORT"


def test_the_charted_row_is_a_manual_chart_not_an_alert(qt_desk):
    """Nothing fired - the trader was looking. So the pane stays muted, the
    alert feed is untouched, and no review alert is invented."""
    from ui.models.bounce import MANUAL_CHART_TAG

    center = qt_desk.trading_panel.alert_center
    feed_before = len(center._alerts)

    center.strength_board.symbolActivated.emit("amd", "long")

    assert center._current_review_alert.tag == MANUAL_CHART_TAG
    assert len(center._alerts) == feed_before, "a look is not a scanner alert"


def test_a_click_never_goes_through_the_scanner_door(qt_desk, monkeypatch):
    """`_enqueue_review_alert` drops in AWAY, drops parked symbols, diverts M5
    to the alert bar and can hide a row behind movers-only. A name the trader
    clicked must appear, so the click uses the lookup box's door instead."""
    center = qt_desk.trading_panel.alert_center
    enqueued: list[object] = []
    monkeypatch.setattr(center, "_enqueue_review_alert", enqueued.append)

    center.strength_board.symbolActivated.emit("tsla", "long")

    assert enqueued == []
    assert center._current_review_alert.symbol == "TSLA"


def test_an_ignored_symbol_is_un_ignored_by_clicking_it(qt_desk):
    """Same rule the lookup box has: "Remove for today" must not make a name
    silently un-chartable, which reads as the board being broken."""
    center = qt_desk.trading_panel.alert_center
    center._ignored_symbols.add("MSFT")

    center.strength_board.symbolActivated.emit("msft", "long")

    assert "MSFT" not in center._ignored_symbols
    assert center._current_review_alert.symbol == "MSFT"


def test_the_lookup_box_still_behaves_exactly_as_it_did(qt_desk):
    """`chart_symbol` grew two optional arguments; its defaults are the box."""
    center = qt_desk.trading_panel.alert_center

    assert center.chart_symbol("intc") is True

    assert center._current_review_alert.side == "WATCH"
    assert center._current_review_alert.trigger == "Charted on demand"
    assert center.chart_symbol("(BULLISH_STRONG)") is False
