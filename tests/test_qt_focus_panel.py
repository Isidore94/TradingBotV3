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
    editor = panel.swing_long_editor
    editor.add_input.setText("nvda, aapl")
    editor.add_from_input()

    assert panel.service.focus_symbols("long") == ["NVDA", "AAPL"]
    assert panel.service.focus_symbols("long", "swing") == ["NVDA", "AAPL"]
    assert panel.service.focus_symbols("long", "m5") == []
    assert editor.chip_flow.count() == 2  # focusChanged rebuilt the chips
    assert editor.add_input.text() == ""  # input cleared


def test_focus_panel_chip_remove_updates_store(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.swing_long_editor.add_input.setText("NVDA AAPL")
    panel.swing_long_editor.add_from_input()

    panel.swing_long_editor._remove("NVDA")  # simulates a chip's × button

    assert panel.service.focus_symbols("long") == ["AAPL"]
    assert panel.swing_long_editor.chip_flow.count() == 1


def test_focus_panel_sides_are_independent(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.swing_long_editor.add_input.setText("NVDA")
    panel.swing_long_editor.add_from_input()
    panel.swing_short_editor.add_input.setText("TSLA")
    panel.swing_short_editor.add_from_input()

    assert panel.service.focus_symbols("long") == ["NVDA"]
    assert panel.service.focus_symbols("short") == ["TSLA"]
    assert panel.swing_long_editor.chip_flow.count() == 1
    assert panel.swing_short_editor.chip_flow.count() == 1


def test_focus_panel_categories_are_independent(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel
    from watchlist_utils import read_watchlist_symbols

    panel = FocusPicksPanel(_service(tmp_path))
    panel.swing_long_editor.add_input.setText("NVDA")
    panel.swing_long_editor.add_from_input()
    panel.m5_long_editor.add_input.setText("AAPL")
    panel.m5_long_editor.add_from_input()

    assert panel.service.focus_symbols("long", "swing") == ["NVDA"]
    assert panel.service.focus_symbols("long", "m5") == ["AAPL"]
    assert panel.service.focus_category("NVDA") == "swing"
    assert panel.service.focus_category("AAPL") == "m5"
    # Swing picks sync into the swing watchlist; m5 picks into longs.txt.
    assert read_watchlist_symbols(tmp_path / "swinglongs.txt") == ["NVDA"]
    assert read_watchlist_symbols(tmp_path / "longs.txt") == ["AAPL"]


def test_focus_panel_marks_live_bounce_alert(tmp_path):
    from PySide6.QtWidgets import QLabel

    from ui.models.bounce import BounceAlert
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.swing_long_editor.add_input.setText("NVDA")
    panel.swing_long_editor.add_from_input()

    panel.record_bounce_alert(
        BounceAlert(time_text="09:30:00", symbol="NVDA", side="LONG", trigger="VWAP reclaim", timeframe="5m")
    )

    chip = panel.swing_long_editor.chip_flow.itemAt(0).widget()
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

    assert labels == [
        "Trading Desk",
        "Focus Picks",
        "Journal",
        "Universe",
        "Research",
        "Auto Pilot",
        "A.I. Summary",
        "System Health",
        "Settings",
    ]
    assert window.pages.widget(1) is window.trading_panel.focus_picks_panel
    assert window.market_regime_status.text().startswith("Auto regime:")
    assert window.technical_integrity_status.text().startswith("Technicals:")
    window._set_auto_regime({"env_key": "bearish_strong", "label": "Bearish Strong"})
    assert window.market_regime_status.text() == "Auto: Bearish Strong"


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

    def proxy_index(symbol, column=2):  # columns 0/1 are the ★/✕ verdict columns
        for proxy_row in range(panel.proxy.rowCount()):
            index = panel.proxy.index(proxy_row, column)
            if panel.proxy.index(proxy_row, 2).data() == symbol:
                return index
        raise AssertionError(f"{symbol} not in table")

    panel._add_row_to_focus(proxy_index("NVDA"))
    panel._add_row_to_focus(proxy_index("TSLA"), "m5")

    assert service.focus_symbols("long") == ["NVDA"]  # LONG row -> Focus Longs
    assert service.focus_symbols("short") == ["TSLA"]  # SHORT row -> Focus Shorts
    assert service.focus_category("NVDA") == "swing"  # ★ default
    assert service.focus_category("TSLA") == "m5"  # explicit menu choice
    # delegate marker lookup reflects focus membership
    assert panel.delegate._is_focus(_row("NVDA", "LONG")) is True
    assert panel.delegate._is_focus(_row("XYZ", "LONG")) is False

    # The ★ column click toggles: lit star -> unfavorite; hollow star -> swing add.
    panel._on_table_clicked(proxy_index("NVDA", column=0))
    assert service.is_focus("NVDA") is False
    panel._on_table_clicked(proxy_index("NVDA", column=0))
    assert service.focus_category("NVDA") == "swing"

    # The x column logs a dislike (with the setups origin) and unfavorites.
    feedback = []
    service.record_feedback = lambda symbol, side, verdict, **kw: feedback.append((symbol, verdict, kw))
    panel._record_dislike(panel.model.rows()[0], "chased, no level")
    assert service.is_focus("NVDA") is False
    symbol, verdict, kwargs = next(entry for entry in feedback if entry[1] == "dislike")
    assert (symbol, verdict) == ("NVDA", "dislike")
    assert kwargs["origin"] == "setups"
    assert kwargs["reason"] == "chased, no level"
    assert "bucket=" in kwargs["context"]


def test_master_panel_report_poll_refreshes_only_when_signature_changes():
    from types import SimpleNamespace

    from ui.panels.master_avwap_panel import MasterAvwapPanel

    calls = []
    panel = SimpleNamespace(
        _report_signatures={"report": (1, 100)},
        _current_report_signatures=lambda: {"report": (1, 100)},
        refresh_from_reports=lambda emit_empty=False: calls.append(emit_empty),
    )
    MasterAvwapPanel._poll_report_changes(panel)
    assert calls == []

    panel._current_report_signatures = lambda: {"report": (2, 120)}
    MasterAvwapPanel._poll_report_changes(panel)
    assert calls == [False]


def test_autopilot_ownership_disables_the_setups_page_scheduler():
    from types import SimpleNamespace

    from ui.panels.master_avwap_panel import MasterAvwapPanel

    notes = []
    panel = SimpleNamespace(
        external_scheduler_owner="",
        scheduler_enabled=True,
        _refresh_scheduler_status=lambda note="": notes.append(note),
    )

    MasterAvwapPanel.set_external_scheduler_owner(panel, "Auto Pilot")

    assert panel.external_scheduler_owner == "Auto Pilot"
    assert panel.scheduler_enabled is False
    assert "owns hourly scans" in notes[-1]

    MasterAvwapPanel.set_external_scheduler_owner(panel, "")
    assert panel.external_scheduler_owner == ""
    assert "available" in notes[-1]


def test_alert_feed_item_focus_highlight():
    from ui.models.bounce import BounceAlert
    from ui.widgets.alert_feed_item import AlertFeedItem

    alert = BounceAlert(time_text="09:30:00", symbol="NVDA", side="LONG", trigger="VWAP reclaim")
    focus_item = AlertFeedItem(alert, focus_category="swing")
    plain_item = AlertFeedItem(alert)
    assert "border-left" in focus_item.styleSheet()  # gold frame for focus names
    assert plain_item.styleSheet() == ""


def test_alert_feed_item_star_reflects_favorite_state():
    from PySide6.QtWidgets import QToolButton

    from ui.models.bounce import BounceAlert
    from ui.widgets.alert_feed_item import AlertFeedItem

    alert = BounceAlert(time_text="09:30:00", symbol="NVDA", side="LONG", trigger="VWAP reclaim")

    def buttons_by_text(item):
        return {button.text(): button for button in item.findChildren(QToolButton)}

    hollow_item = AlertFeedItem(alert, show_favorite_button=True, favorite_hint="Swing Focus")
    hollow = buttons_by_text(hollow_item)
    assert set(hollow) == {"☆", "✕"}  # star to favorite, ✕ to dislike-with-reason
    assert "Swing Focus" in hollow["☆"].toolTip()
    assert "pick_feedback" in hollow["✕"].toolTip()
    lit_item = AlertFeedItem(alert, focus_category="swing", show_favorite_button=True)
    lit = buttons_by_text(lit_item)
    assert set(lit) == {"★", "✕"}
    assert "Unfavorite" in lit["★"].toolTip()
    # No symbol -> nothing to click.
    no_symbol = AlertFeedItem(
        BounceAlert(time_text="09:30:00", symbol="", side="WATCH", trigger="regime note"),
        show_favorite_button=True,
    )
    assert no_symbol.findChildren(QToolButton) == []


def test_alert_center_routes_detail_out_when_embedded_pane_disabled(tmp_path):
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(_service(tmp_path))
    routed = []
    panel.setupRequested.connect(routed.append)
    alert = BounceAlert(time_text="09:30:00", symbol="NVDA", side="LONG", trigger="VWAP reclaim")

    panel._show_alert_detail(alert)  # default: embedded pane renders the plan
    assert routed == []
    assert panel.detail_view.isVisibleTo(panel)

    panel.set_embedded_detail_enabled(False)  # workspace mode: one detail pane
    assert not panel.detail_view.isVisibleTo(panel)
    panel._show_alert_detail(alert)
    assert routed and routed[0]["symbol"] == "NVDA" and routed[0]["side"] == "LONG"
    assert not panel.detail_view.isVisibleTo(panel)


def test_entry_assist_button_specs_cover_all_situations():
    from ui.panels.bounce_panel import entry_assist_button_specs

    # Disconnected: every button exists but is disabled.
    specs = {spec["command"]: spec for spec in entry_assist_button_specs({})}
    assert set(specs) == {"pullback_window", "bounce_window", "strongest_30m", "weakest_30m", "movers_30m"}
    assert all(not spec["enabled"] for spec in specs.values())
    assert specs["pullback_window"]["advanced"] and specs["bounce_window"]["advanced"]
    assert not specs["strongest_30m"]["advanced"]

    # Connected, bullish strong: all enabled, pullback window is the recommended action.
    specs = {
        spec["command"]: spec
        for spec in entry_assist_button_specs({"env_key": "bullish_strong", "window_active": False})
    }
    assert all(spec["enabled"] for spec in specs.values())
    assert specs["pullback_window"]["label"] == "⏱ Pullback started"
    assert specs["pullback_window"]["recommended"]
    assert "hold up" in specs["pullback_window"]["tooltip"]
    assert specs["bounce_window"]["label"] == "⏱ Bounce started"
    assert not specs["bounce_window"]["recommended"]
    assert specs["strongest_30m"]["label"] == "Strongest 30m"
    assert specs["weakest_30m"]["label"] == "Weakest 30m"
    assert "BOTH" in specs["movers_30m"]["tooltip"]

    # Active long window flips the pullback button to its "over" state.
    specs = {
        spec["command"]: spec
        for spec in entry_assist_button_specs(
            {
                "env_key": "bullish_strong",
                "window_active": True,
                "window_started": "10:05",
                "window_sides": ["long"],
            }
        )
    }
    assert specs["pullback_window"]["label"] == "Pullback over → strongest (since 10:05)"
    assert specs["bounce_window"]["label"] == "⏱ Bounce started"

    # Active short window flips the bounce button, even under a long-side regime.
    specs = {
        spec["command"]: spec
        for spec in entry_assist_button_specs(
            {
                "env_key": "bullish_weak",
                "window_active": True,
                "window_started": "11:00",
                "window_sides": ["short"],
            }
        )
    }
    assert specs["bounce_window"]["label"] == "Bounce over → weakest (since 11:00)"
    assert specs["bounce_window"]["recommended"]  # open window = the action to finish
    assert specs["strongest_30m"]["recommended"]  # bullish_weak's regime pick

    # Weak/chop regimes recommend their movers list.
    specs = {spec["command"]: spec for spec in entry_assist_button_specs({"env_key": "bearish_weak"})}
    assert specs["weakest_30m"]["recommended"]
    specs = {spec["command"]: spec for spec in entry_assist_button_specs({"env_key": "neutral_chop"})}
    assert specs["movers_30m"]["recommended"]


def test_bounce_panel_defaults_user_to_na_and_hides_manual_windows(monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        monkeypatch.setattr(QTimer, "singleShot", lambda *_args: None)
        from ui.panels.bounce_panel import BouncePanel
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    panel = BouncePanel()
    assert panel.environment_input.currentData() == ""
    assert panel.technical_integrity_label.text().startswith("Technicals:")
    assert "N/A" in panel.environment_input.currentText()
    assert panel.entry_assist_buttons["pullback_window"].isHidden()
    assert panel.entry_assist_buttons["bounce_window"].isHidden()
    assert not panel.entry_assist_buttons["strongest_30m"].isHidden()

    panel.entry_assist_advanced_button.setChecked(True)
    assert not panel.entry_assist_buttons["pullback_window"].isHidden()
    panel.stop()


def test_auto_regime_readout_formatting():
    from ui.panels.bounce_panel import format_auto_regime_reading

    chip, tip = format_auto_regime_reading({})
    assert chip == "Auto regime: n/a"
    assert "SPY" in tip

    reading = {
        "env_key": "bullish_weak",
        "label": "Bullish Weak",
        "source": "vwap",
        "day_pct": 0.42,
        "last_close": 626.14,
        "prev_close": 623.51,
        "bar_time": "10:35",
        "override_active": False,
        "active_env_key": "bullish_weak",
        "active_label": "Bullish Weak",
        "strong_abs_pct": 0.5,
        "band_fraction_needed": 0.6,
        "vwap": 625.30,
        "stdev": 0.87,
        "above_band_frac": 0.41,
        "below_band_frac": 0.03,
    }
    chip, tip = format_auto_regime_reading(reading)
    assert chip == "Auto: Bullish Weak"
    assert "Bullish Strong: 41%" in tip and "needs >= 60%" in tip
    assert "Bearish Strong: 3%" in tip
    assert "Bullish Weak: SPY above VWAP and green on the day - YES" in tip

    # Manual override: the chip shows both the forced and the auto regime.
    reading["override_active"] = True
    reading["active_label"] = "Bearish Strong"
    chip, tip = format_auto_regime_reading(reading)
    assert chip == "Manual: Bearish Strong (auto sees Bullish Weak)"
    assert "auto keeps measuring" in tip

    # Young session: day%-rule fallback text instead of VWAP possibilities.
    young = {k: v for k, v in reading.items() if k not in {"vwap", "stdev", "above_band_frac", "below_band_frac"}}
    young.update({"override_active": False, "source": "day_pct"})
    chip, tip = format_auto_regime_reading(young)
    assert "day% rule applies" in tip


def test_favorite_category_routing_for_alerts():
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import favorite_category_for_alert

    m5 = BounceAlert(time_text="09:30:00", symbol="AAOI", side="LONG", timeframe="5m")
    h1 = BounceAlert(time_text="09:30:00", symbol="NVDA", side="LONG", timeframe="H1")
    d1 = BounceAlert(time_text="09:30:00", symbol="AAPL", side="LONG", is_d1=True)
    untimed = BounceAlert(time_text="09:30:00", symbol="TSLA", side="SHORT")
    assert favorite_category_for_alert(m5) == "m5"
    assert favorite_category_for_alert(h1) == "swing"
    assert favorite_category_for_alert(d1) == "swing"
    assert favorite_category_for_alert(untimed) == "m5"


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
