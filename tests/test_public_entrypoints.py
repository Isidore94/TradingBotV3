import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_legacy_script_imports_still_resolve_to_live_modules():
    import bounce_bot
    import bounce_bot_lib.legacy as bounce_legacy
    import gui
    import gui_app.app as gui_app
    import market_prep_gui.tabs as market_prep_tabs
    import market_prep_tab
    import master_avwap
    import master_avwap_lib.legacy as master_legacy

    assert master_avwap is master_legacy
    assert gui is gui_app
    assert market_prep_tab is market_prep_tabs
    assert bounce_bot is bounce_legacy


def test_master_avwap_grouped_package_imports_expose_existing_behavior():
    import master_avwap
    from master_avwap_lib.data.daily_bars import fetch_daily_bars
    from master_avwap_lib.gui import MasterAvwapGUI
    from master_avwap_lib.indicators import compute_indicator_frame
    from master_avwap_lib.outputs.market_prep import build_market_prep_payload
    from master_avwap_lib.outputs.reports import write_priority_setup_report
    from master_avwap_lib.runner import run_master
    from master_avwap_lib.setups.priority import build_priority_setup_summary
    from master_avwap_lib.theta.reports import write_theta_put_report
    from master_avwap_lib.tracker import build_tracker_stats_rows

    assert run_master is master_avwap.run_master
    assert MasterAvwapGUI is master_avwap.MasterAvwapGUI
    assert fetch_daily_bars is master_avwap.fetch_daily_bars
    assert compute_indicator_frame is master_avwap.compute_indicator_frame
    assert build_priority_setup_summary is master_avwap.build_priority_setup_summary
    assert build_tracker_stats_rows is master_avwap.build_tracker_stats_rows
    assert write_theta_put_report is master_avwap.write_theta_put_report
    assert write_priority_setup_report is master_avwap.write_priority_setup_report
    assert build_market_prep_payload is master_avwap.build_market_prep_payload


def test_gui_package_imports_expose_existing_panels():
    import gui
    from gui_app.bounce_panel import BounceBotController
    from gui_app.master_panel import SimpleMasterAvwapPanel
    from gui_app.storage_controls import TrackerStorageControls
    from gui_app.theme import configure_theme
    from gui_app.watchlist_editor import WatchlistEditorArea
    from market_prep_gui.market_prep_panel import MarketPrepTab
    from market_prep_gui.ticker_lookup_panel import TickerLookupTab
    import market_prep_tab

    assert BounceBotController is gui.BounceBotController
    assert SimpleMasterAvwapPanel is gui.SimpleMasterAvwapPanel
    assert TrackerStorageControls is gui.TrackerStorageControls
    assert WatchlistEditorArea is gui.WatchlistEditorArea
    assert configure_theme is gui.configure_theme
    assert MarketPrepTab is market_prep_tab.MarketPrepTab
    assert TickerLookupTab is market_prep_tab.TickerLookupTab


def test_bounce_bot_grouped_package_imports_expose_existing_behavior():
    import bounce_bot
    from bounce_bot_lib.alerts import append_alert_message, configure_alert_tags
    from bounce_bot_lib.feedback import record_bounce_feedback
    from bounce_bot_lib.gui import start_gui
    from bounce_bot_lib.ib_client import BounceBot
    from bounce_bot_lib.runner import run_bot_with_gui

    assert BounceBot is bounce_bot.BounceBot
    assert append_alert_message is bounce_bot.append_alert_message
    assert configure_alert_tags is bounce_bot.configure_alert_tags
    assert record_bounce_feedback is bounce_bot.record_bounce_feedback
    assert run_bot_with_gui is bounce_bot.run_bot_with_gui
    assert start_gui is bounce_bot.start_gui


def test_mini_pc_shared_scan_delegates_to_master_avwap_outputs(monkeypatch):
    import master_avwap
    import master_avwap_mini_pc
    import project_paths

    calls = {}

    def fake_run_master(**kwargs):
        calls["kwargs"] = kwargs
        return {
            "watchlist_label": "home folder watchlists + swing watchlists",
            "theta_put_rows": [],
            "theta_pcs_rows": [],
        }

    monkeypatch.setattr(master_avwap, "run_master", fake_run_master)
    monkeypatch.setattr(master_avwap_mini_pc, "run_master", fake_run_master)
    monkeypatch.setattr(
        master_avwap_mini_pc,
        "filter_watchlists_by_previous_day_levels",
        lambda: {"status": "ok", "message": "filter ok"},
    )

    filter_summary, scan_result = master_avwap_mini_pc.run_master_with_watchlist_filter(
        update_setup_tracker=False
    )

    assert filter_summary["message"] == "filter ok"
    assert scan_result["watchlist_label"] == "home folder watchlists + swing watchlists"
    # include_theta was removed from run_master (theta is unconditional with
    # deferred enrichment); passing it crashed every scheduled scan.
    assert calls["kwargs"] == {
        "use_shared_watchlists": True,
        "update_setup_tracker": False,
        "require_ib_for_setup_tracker": True,
    }
    assert master_avwap_mini_pc.MASTER_AVWAP_REPORT_FILE == project_paths.MASTER_AVWAP_REPORT_FILE
    assert master_avwap_mini_pc.MASTER_AVWAP_PRIORITY_SETUPS_FILE == project_paths.MASTER_AVWAP_PRIORITY_SETUPS_FILE
    assert master_avwap_mini_pc.THETA_PUTS_FILE == master_avwap.THETA_PUTS_FILE
