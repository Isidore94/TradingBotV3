import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _service(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "focus_pick_membership.json",
    )
    return FocusService(store)


def test_focus_panel_add_renders_chips(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    editor = panel.long_editor
    editor.add_input.setText("nvda, aapl")
    editor.add_from_input()

    assert panel.service.focus_symbols("long") == ["NVDA", "AAPL"]
    assert editor.chip_flow.count() == 2  # focusChanged rebuilt the chips
    assert editor.add_input.text() == ""  # input cleared


def test_focus_panel_chip_remove_updates_store(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.long_editor.add_input.setText("NVDA AAPL")
    panel.long_editor.add_from_input()

    panel.long_editor._remove("NVDA")  # simulates a chip's × button

    assert panel.service.focus_symbols("long") == ["AAPL"]
    assert panel.long_editor.chip_flow.count() == 1


def test_focus_panel_sides_are_independent(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.long_editor.add_input.setText("NVDA")
    panel.long_editor.add_from_input()
    panel.short_editor.add_input.setText("TSLA")
    panel.short_editor.add_from_input()

    assert panel.service.focus_symbols("long") == ["NVDA"]
    assert panel.service.focus_symbols("short") == ["TSLA"]
    assert panel.long_editor.chip_flow.count() == 1
    assert panel.short_editor.chip_flow.count() == 1


def test_focus_panel_marks_live_bounce_alert(tmp_path):
    from PySide6.QtWidgets import QLabel

    from ui.models.bounce import BounceAlert
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.long_editor.add_input.setText("NVDA")
    panel.long_editor.add_from_input()

    panel.record_bounce_alert(
        BounceAlert(time_text="09:30:00", symbol="NVDA", side="LONG", trigger="VWAP reclaim", timeframe="5m")
    )

    chip = panel.long_editor.chip_flow.itemAt(0).widget()
    labels = [label.text() for label in chip.findChildren(QLabel)]
    assert "BOUNCE" in labels
    assert any("09:30:00 bounce - LONG 5m VWAP reclaim" == text for text in labels)


def test_master_workspace_no_longer_tabs_focus_picks(tmp_path):
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    from ui.panels.theta_panel import ThetaPanel
    from ui.panels.trading_desk import MasterAvwapWorkspace
    from ui.panels.watchlists_panel import WatchlistsPanel

    service = _service(tmp_path)
    workspace = MasterAvwapWorkspace(
        MasterAvwapPanel(service),
        ThetaPanel(),
        WatchlistsPanel(),
    )

    assert [workspace.tabs.tabText(index) for index in range(workspace.tabs.count())] == [
        "Setups",
        "Theta Plays",
        "Watchlists",
    ]


def test_focus_picks_is_top_level_app_page():
    from ui.app import MainWindow
    from ui.state import UiState

    window = MainWindow(UiState(workspace_mode="workspace"))
    labels = [button.text() for button in window.nav_buttons]

    assert labels == ["Trading Desk", "Focus Picks", "Journal", "Research", "Settings"]
    assert window.pages.widget(1) is window.trading_panel.focus_picks_panel


def _row(symbol, side, **kwargs):
    from ui.models.setup import SetupRow

    return SetupRow(symbol=symbol, side=side, **kwargs)


def test_master_panel_add_to_focus_routes_by_side(tmp_path):
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    service = _service(tmp_path)
    panel = MasterAvwapPanel(service)
    panel.set_rows(
        [
            _row("NVDA", "LONG", score=90.0, bucket="favorite_setup"),
            _row("TSLA", "SHORT", score=80.0, bucket="favorite_setup"),
        ]
    )

    def proxy_index(symbol):
        for proxy_row in range(panel.proxy.rowCount()):
            index = panel.proxy.index(proxy_row, 0)
            if index.data() == symbol:
                return index
        raise AssertionError(f"{symbol} not in table")

    panel._add_row_to_focus(proxy_index("NVDA"))
    panel._add_row_to_focus(proxy_index("TSLA"))

    assert service.focus_symbols("long") == ["NVDA"]  # LONG row -> Focus Longs
    assert service.focus_symbols("short") == ["TSLA"]  # SHORT row -> Focus Shorts
    # delegate marker lookup reflects focus membership
    assert panel.delegate._is_focus(_row("NVDA", "LONG")) is True
    assert panel.delegate._is_focus(_row("XYZ", "LONG")) is False


def test_alert_feed_item_focus_highlight():
    from ui.models.bounce import BounceAlert
    from ui.widgets.alert_feed_item import AlertFeedItem

    alert = BounceAlert(time_text="09:30:00", symbol="NVDA", side="LONG", trigger="VWAP reclaim")
    focus_item = AlertFeedItem(alert, is_focus=True)
    plain_item = AlertFeedItem(alert, is_focus=False)
    assert "border-left" in focus_item.styleSheet()  # gold stripe for focus names
    assert plain_item.styleSheet() == ""


def test_rrs_board_marks_focus_aligned_only():
    from ui.widgets import rrs_snapshot

    payload = {"threshold": 1.0, "results": [("RS", "NVDA", 3.0, 1.0), ("RW", "TSLA", -3.0, 1.0)]}
    star = "&#9733;"

    # focus long shown as RS + focus short shown as RW -> both flagged
    aligned = rrs_snapshot._scope_html(payload, "SPY", {"long": {"NVDA"}, "short": {"TSLA"}})
    assert aligned.count(star) == 2
    # no focus -> no markers
    assert star not in rrs_snapshot._scope_html(payload, "SPY", {"long": set(), "short": set()})
    # misaligned (focus long that shows as RW) -> NOT flagged
    assert star not in rrs_snapshot._scope_html(payload, "SPY", {"long": {"TSLA"}, "short": set()})
