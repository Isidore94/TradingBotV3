"""S1.4 - a Master AVWAP ticker click charts into the Visual Chart Review.

Trader, 2026-09-03: *"when I hit a ticker on the master avwap setups tab, I dont
want the chart to be a pop up, I want it to come up on the visual chart review
instead."*

Today `_on_table_clicked` sends the symbol column to `_open_symbol_snapshot`,
which calls `show_symbol_snapshot` - a dialog over the top of the desk. The model
to copy is already in the tree: the M5 Strength Board's row click goes through
`AlertCenterPanel.chart_symbol` (the lookup box's door) and deliberately not
through `_enqueue_review_alert`.

Driven through the real handlers: the table click handler with a real proxy
index, and the desk that owns both panels.
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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

from PySide6.QtWidgets import QApplication, QInputDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _row():
    from ui.models.setup import SetupRow

    return SetupRow(
        symbol="LNG",
        side="SHORT",
        score=245.0,
        bucket="favorite_setup",
        setup_tags=["AVWAP_BREAKOUT"],
        expected_r=0.85,
        raw={"setup_family": "avwap_breakout"},
    )


def _cell(panel, key: str):
    column = next(
        index for index, (name, _label) in enumerate(panel.model.COLUMNS) if name == key
    )
    return panel.proxy.index(0, column)


@pytest.fixture
def popups(monkeypatch):
    """Every snapshot popup this packet takes off the click path."""
    import ui.widgets.symbol_snapshot_dialog as dialog_module

    opened: list[str] = []
    monkeypatch.setattr(
        dialog_module,
        "show_symbol_snapshot",
        lambda *args, **kwargs: opened.append(str(args[1] if len(args) > 1 else "")),
    )
    # The ✕ still asks today; answer CANCEL so a headless run cannot block.
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: ("", False)))
    monkeypatch.setattr(
        QInputDialog, "getMultiLineText", staticmethod(lambda *a, **k: ("", False))
    )
    return opened


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from focus_picks import FocusPickStore
    from ui.annotations import verdicts
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    from ui.services.focus_service import FocusService

    # ★ and ✕ write a verdict annotation, and `verdicts` defaults to the LIVE
    # C:\TradingBotData\trader_annotations.jsonl. Redirect before any click.
    annotations = tmp_path / "trader_annotations.jsonl"
    for name in ("record_like", "record_dislike"):
        real = getattr(verdicts, name)
        monkeypatch.setattr(
            verdicts,
            name,
            (lambda real: lambda **kwargs: real(**{**kwargs, "path": annotations}))(real),
        )

    service = FocusService(
        FocusPickStore(
            focus_longs_path=tmp_path / "focus_longs.txt",
            focus_shorts_path=tmp_path / "focus_shorts.txt",
            longs_path=tmp_path / "longs.txt",
            shorts_path=tmp_path / "shorts.txt",
            membership_path=tmp_path / "focus_pick_membership.json",
        )
    )
    widget = MasterAvwapPanel(service, review_events_path=tmp_path / "events.jsonl")
    widget.set_rows([_row()])
    yield widget
    widget.deleteLater()


def _charted(panel) -> list:
    assert hasattr(panel, "symbolActivated"), (
        "MasterAvwapPanel must offer symbolActivated(symbol, side) - the same "
        "signal the group tape and the strength board already use"
    )
    seen: list = []
    panel.symbolActivated.connect(lambda symbol, side: seen.append((symbol, side)))
    return seen


# ---------------------------------------------------------------------------
def test_a_single_click_on_the_symbol_charts_and_opens_no_popup(panel, popups):
    seen = _charted(panel)

    panel._on_table_clicked(_cell(panel, "symbol"))

    assert seen == [("LNG", "SHORT")], seen
    assert popups == [], "the snapshot window leaves the click path"


def test_a_double_click_on_the_symbol_charts_and_opens_no_popup(panel, popups):
    seen = _charted(panel)

    panel._open_symbol_snapshot_from_double_click(_cell(panel, "symbol"))

    assert seen == [("LNG", "SHORT")], seen
    assert popups == []


def test_the_star_and_the_x_cells_never_chart(panel, popups):
    """They are click targets of their own and keep their verbs."""
    seen = _charted(panel)

    panel._on_table_clicked(_cell(panel, "favorite"))
    panel._on_table_clicked(_cell(panel, "dislike"))

    assert seen == [], seen
    assert popups == []


def test_the_snapshot_window_is_still_reachable_from_the_row_menu(panel):
    """Delete nothing: the dialog stays one right-click away."""
    labels = [label for label, _callback in panel.table._row_actions]
    assert any("napshot" in label for label in labels), labels


# ---------------------------------------------------------------------------
# the wiring, on the desk that owns both panels
# ---------------------------------------------------------------------------
def test_the_desk_routes_a_setups_ticker_click_into_the_review_pane(_qapp, monkeypatch):
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.panels.bounce_panel import BouncePanel

    calls: list = []
    monkeypatch.setattr(
        AlertCenterPanel,
        "chart_symbol",
        lambda self, symbol, *, side="", origin="": calls.append((symbol, side, origin))
        or True,
    )
    # `BouncePanel.__init__` ends in `QTimer.singleShot(0, self.start)`, which
    # connects to the live TWS on the first processEvents. Nothing here is about
    # BounceBot and a test must not reach a broker.
    monkeypatch.setattr(BouncePanel, "start", lambda self: None)

    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        desk.master_panel.set_rows([_row()])
        QApplication.processEvents()
        desk.master_panel._on_table_clicked(_cell(desk.master_panel, "symbol"))
        QApplication.processEvents()
    finally:
        desk.shutdown()
        desk.close()

    assert len(calls) == 1, calls
    symbol, side, origin = calls[0]
    assert (symbol, side) == ("LNG", "SHORT")
    assert "aster" in origin and "AVWAP" in origin, (
        f"the chart must say where it came from, got {origin!r}"
    )
