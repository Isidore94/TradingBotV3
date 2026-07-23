import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_ignored_alert_symbols_round_trip_sorted_and_normalized(tmp_path):
    from alert_review_state import (
        load_ignored_alert_symbols,
        save_ignored_alert_symbols,
    )

    path = tmp_path / "ignored.txt"
    assert load_ignored_alert_symbols(path) == set()

    written = save_ignored_alert_symbols(["tsla", "NVDA", "tsla"], path)

    assert written == {"NVDA", "TSLA"}
    assert path.read_text(encoding="utf-8") == "NVDA\nTSLA\n"
    assert load_ignored_alert_symbols(path) == {"NVDA", "TSLA"}
