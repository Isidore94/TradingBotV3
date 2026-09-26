"""S17 golden fixture: one recorded week (2026-09-21 .. 09-25) through the regime table.

Bars: the research lake's SPY and XLK M5 (from 2026-09-01) and the durable D1
store for SPY, QQQ, IWM and XLK, copied to scratch and frozen in
`fixtures/market_regime_week_v1.json`. QQQ and IWM have no M5 in the lake, so
their intraday cells are honestly `unknown`. Any change to a read, a window or
the classifier changes this file. Regenerate only on a deliberate rule change:
`REGEN_MARKET_REGIME_GOLDEN=1`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

FIXTURE = ROOT / "tests" / "fixtures" / "market_regime_week_v1.json"
GOLDEN = ROOT / "tests" / "fixtures" / "market_regime_week_golden_v1.jsonl"
WEEK = [date(2026, 9, 21) + timedelta(days=offset) for offset in range(5)]
SYMBOLS = ("SPY", "QQQ", "IWM", "XLK")


def load_fixture() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    d1 = {
        symbol: [
            {"symbol": symbol, "session_date": date.fromisoformat(day), "open": o, "high": h, "low": low, "close": c, "volume": v}
            for day, o, h, low, c, v in rows
        ]
        for symbol, rows in raw["d1"].items()
    }
    m5 = {}
    for symbol, rows in raw["m5"].items():
        m5[symbol] = []
        for start, o, h, low, c, v in rows:
            begin = datetime.fromisoformat(start)
            m5[symbol].append(
                {
                    "symbol": symbol, "interval_start": begin, "interval_end": begin + timedelta(minutes=5),
                    "open": o, "high": h, "low": low, "close": c, "volume": v, "is_complete": True,
                }
            )
    return d1, m5


def render() -> str:
    import market_regimes as mr

    d1, m5 = load_fixture()
    stamp = datetime(2026, 9, 26, 1, 0, tzinfo=mr.MARKET_TZ)
    lines = []
    for day in WEEK:
        for symbol in SYMBOLS:
            row = mr.session_row(symbol, day, d1_rows=d1.get(symbol, []), m5_rows=m5.get(symbol, []), computed_at=stamp)
            lines.append(json.dumps(row, sort_keys=True))
    return "\n".join(lines) + "\n"


def test_the_recorded_week_matches_the_golden_table():
    text = render()
    if os.environ.get("REGEN_MARKET_REGIME_GOLDEN") == "1":
        GOLDEN.write_text(text, encoding="utf-8", newline="\n")
    assert GOLDEN.read_bytes().decode("utf-8").replace("\r\n", "\n") == text


def test_the_golden_week_reads_every_timeframe_where_bars_exist():
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(WEEK) * len(SYMBOLS)
    for row in rows:
        timeframes = row["timeframes"]
        assert set(timeframes) == {"M5", "M30", "H1", "H4", "D1", "W"}
        assert timeframes["D1"] != "unknown" and timeframes["W"] != "unknown", row["symbol"]
        if row["symbol"] in ("SPY", "XLK"):
            assert "unknown" not in timeframes.values(), (row["session_date"], row["symbol"], timeframes)
        else:
            assert {timeframes[tf] for tf in ("M5", "M30", "H1", "H4")} == {"unknown"}
        assert row["structure"]["sma20"]["status"] == "ok"
        assert row["structure"]["daily_channel"]["label"] != "unknown"
