import sys
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_ignored_alert_symbols_round_trip_for_one_market_date(tmp_path):
    from alert_review_state import (
        load_ignored_alert_symbols,
        save_ignored_alert_symbols,
    )

    path = tmp_path / "ignored.txt"
    today = "2026-07-23"
    assert load_ignored_alert_symbols(path, market_date=today) == set()

    written = save_ignored_alert_symbols(
        ["tsla", "NVDA", "tsla"],
        path,
        market_date=today,
    )

    assert written == {"NVDA", "TSLA"}
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "market_date": today,
        "symbols": ["NVDA", "TSLA"],
    }
    assert load_ignored_alert_symbols(path, market_date=today) == {"NVDA", "TSLA"}
    assert load_ignored_alert_symbols(path, market_date="2026-07-24") == set()


def test_legacy_permanent_ignore_list_is_not_carried_forward(tmp_path):
    from alert_review_state import load_ignored_alert_symbols

    path = tmp_path / "ignored.txt"
    path.write_text("NVDA\nTSLA\n", encoding="utf-8")

    assert load_ignored_alert_symbols(path, market_date="2026-07-23") == set()
