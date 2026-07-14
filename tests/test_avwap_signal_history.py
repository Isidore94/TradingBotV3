from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib.runner import _sanitize_existing_avwap_signal_rows  # noqa: E402


def test_signal_history_sanitizer_drops_shifted_row_and_canonicalizes_aliases():
    frame = pd.DataFrame(
        [
            {"run_date": ".626", "symbol": "near_favorite_zone", "trade_date": "0.0", "side": "1.0"},
            {"run_date": "2026-07-13", "symbol": "AAPL", "trade_date": "2026-07-13", "side": "buy"},
            {"run_date": "2026-07-13", "symbol": "NVDA", "trade_date": "2026-07-13", "side": "SHORT"},
        ]
    )

    cleaned, dropped = _sanitize_existing_avwap_signal_rows(frame)

    assert dropped == 1
    assert cleaned[["symbol", "side"]].to_dict("records") == [
        {"symbol": "AAPL", "side": "LONG"},
        {"symbol": "NVDA", "side": "SHORT"},
    ]
