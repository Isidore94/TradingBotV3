import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_legacy_script_imports_still_resolve_to_live_modules():
    import bounce_bot
    import bounce_bot_lib.legacy as bounce_legacy
    import master_avwap
    import master_avwap_lib.legacy as master_legacy

    assert master_avwap is master_legacy
    assert bounce_bot is bounce_legacy


def test_the_tk_stack_is_gone_and_nothing_imports_tkinter_at_module_scope():
    """Assessment packet F2 (2026-09-03): the Tk GUI, its shims, the Tk journal
    and market-prep tabs and TickerMover were removed. The desk is scripts/ui."""
    import importlib.util

    for name in ("gui", "gui_app", "market_prep_gui", "market_prep_tab", "journal_tab", "TickerMover"):
        assert importlib.util.find_spec(name) is None, f"{name} should be gone"
    assert not (SCRIPTS_DIR / "master_avwap_lib" / "gui.py").exists()
    assert not (SCRIPTS_DIR / "bounce_bot_lib" / "gui.py").exists()
    for path in SCRIPTS_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "\nimport tkinter" not in text and "\nfrom tkinter" not in text, f"{path} imports tkinter"


def test_master_avwap_grouped_package_imports_expose_existing_behavior():
    import master_avwap
    from master_avwap_lib.data.daily_bars import fetch_daily_bars
    from master_avwap_lib.indicators import compute_indicator_frame
    from master_avwap_lib.outputs.market_prep import build_market_prep_payload
    from master_avwap_lib.outputs.reports import write_priority_setup_report
    from master_avwap_lib.runner import run_master
    from master_avwap_lib.setups.priority import build_priority_setup_summary
    from master_avwap_lib.theta.reports import write_theta_put_report
    from master_avwap_lib.tracker import build_tracker_stats_rows

    assert run_master is master_avwap.run_master
    assert fetch_daily_bars is master_avwap.fetch_daily_bars
    assert compute_indicator_frame is master_avwap.compute_indicator_frame
    assert build_priority_setup_summary is master_avwap.build_priority_setup_summary
    assert build_tracker_stats_rows is master_avwap.build_tracker_stats_rows
    assert write_theta_put_report is master_avwap.write_theta_put_report
    assert write_priority_setup_report is master_avwap.write_priority_setup_report
    assert build_market_prep_payload is master_avwap.build_market_prep_payload


def test_bounce_bot_grouped_package_imports_expose_existing_behavior():
    import bounce_bot
    from bounce_bot_lib.feedback import record_bounce_feedback
    from bounce_bot_lib.ib_client import BounceBot
    from bounce_bot_lib.runner import run_bot_with_gui

    assert BounceBot is bounce_bot.BounceBot
    assert record_bounce_feedback is bounce_bot.record_bounce_feedback
    assert run_bot_with_gui is bounce_bot.run_bot_with_gui
