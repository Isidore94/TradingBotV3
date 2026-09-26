"""`journal_regime_fill` output is byte-identical after S17 moved its reads into `market_regimes`.

The golden text was written by the pre-refactor code (branch base 0b2b25d1) over
the deterministic tapes below; the refactor must reproduce it exactly.
Regenerate only on a deliberate rule change: `REGEN_JOURNAL_REGIME_GOLDEN=1`.
"""

from __future__ import annotations

import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_warehouse import exchange_calendar as xcal  # noqa: E402

GOLDEN = ROOT / "tests" / "fixtures" / "journal_regime_fill_identity_v1.txt"
SYMBOLS = ("SPY", "QQQ", "IWM")
FIRST_TRADE_DAY = date(2026, 8, 17)


def _sessions(start: date, end: date) -> list[date]:
    return [session.session_date for session in xcal.sessions_between(start, end)]


def _d1_tapes() -> dict[str, list[dict]]:
    days = _sessions(date(2026, 3, 2), date(2026, 8, 21))
    tapes: dict[str, list[dict]] = {}
    for offset, symbol in enumerate(SYMBOLS):
        rng = random.Random(1000 + offset)
        price = 400.0 + 50 * offset
        rows = []
        for day in days:
            drift = rng.uniform(-6.0, 6.0) + (1.5 if day.month in (4, 5) else -1.0 if day.month == 8 else 0.0)
            open_ = price
            close = max(10.0, price + drift)
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": day,
                    "open": round(open_, 2),
                    "high": round(max(open_, close) + rng.uniform(0.1, 3.0), 2),
                    "low": round(min(open_, close) - rng.uniform(0.1, 3.0), 2),
                    "close": round(close, 2),
                    "volume": 1_000_000,
                }
            )
            price = close
        tapes[symbol] = rows
    return tapes


def _spy_m5() -> list[dict]:
    rng = random.Random(77)
    rows = []
    price = 520.0
    for day in _sessions(date(2026, 8, 10), date(2026, 8, 21)):
        session = xcal.trading_session(day)
        step = 0.6 if day.day % 3 == 0 else -0.5 if day.day % 3 == 1 else 0.0
        for index in range(78):
            start = session.rth_open_at + timedelta(minutes=5 * index)
            close = price + step * 0.2 + rng.uniform(-0.4, 0.4)
            rows.append(
                {
                    "symbol": "SPY",
                    "interval_start": start,
                    "interval_end": start + timedelta(minutes=5),
                    "open": round(price, 2),
                    "high": round(max(price, close) + 0.1, 2),
                    "low": round(min(price, close) - 0.1, 2),
                    "close": round(close, 2),
                    "volume": 1000 + index,
                    "is_complete": True,
                }
            )
            price = close
    return rows


def _trades() -> list[dict]:
    rows = []
    for index, day in enumerate(_sessions(FIRST_TRADE_DAY, date(2026, 8, 21))):
        for hour, minute in ((9, 50), (10, 35), (13, 5), (15, 55)):
            rows.append({"trade_id": f"T{index}{hour}", "opened_at": f"{day.isoformat()}T{hour:02d}:{minute:02d}:00-04:00"})
    rows.append({"trade_id": "DATE_ONLY", "opened_at": "2026-08-14", "trade_date": "2026-08-14"})
    rows.append({"trade_id": "BEFORE_OPEN", "opened_at": "2026-08-13T08:15:00-04:00"})
    rows.append({"trade_id": "WEEKEND", "opened_at": "2026-08-16T11:00:00-04:00"})
    return rows


class _FakeStore:
    def __init__(self, regime_rows):
        self._regime_rows = regime_rows

    def list_regime_rows(self):
        return list(self._regime_rows)

    def list_trades(self):
        return _trades()

    def upsert_auto_regime(self, trade_date, **fields):
        return trade_date != "2026-08-18"


def render() -> str:
    from journal_regime_fill import fill_regimes, format_summary, read_session_regime

    tapes = _d1_tapes()
    m5 = _spy_m5()
    loader = lambda: (tapes, m5, "fixture")  # noqa: E731
    store = _FakeStore(
        [
            {"trade_date": "2026-08-19", "source": "trader", "mid_term_regime": "down"},
        ]
    )
    lines = []
    summary = fill_regimes(store, apply=True, loader=loader)
    lines.append(format_summary(summary, applied=True))
    for reading in summary["readings"]:
        lines.append(repr(reading))
    dry = fill_regimes(store, apply=False, loader=loader, since=date(2026, 8, 18), until=date(2026, 8, 20))
    lines.append(format_summary(dry, applied=False))
    at = datetime(2026, 8, 20, 11, 40, tzinfo=xcal.EXCHANGE_TZ)
    lines.append(repr(read_session_regime("2026-08-20", at, d1_by_symbol=tapes, spy_m5=m5)))
    lines.append(repr(read_session_regime("2026-08-20", None, d1_by_symbol={}, spy_m5=())))
    return "\n".join(lines) + "\n"


def test_journal_regime_fill_output_is_byte_identical():
    text = render()
    if os.environ.get("REGEN_JOURNAL_REGIME_GOLDEN") == "1":
        GOLDEN.write_text(text, encoding="utf-8", newline="\n")
    # Git may check the golden out with CRLF; the comparison is of the text itself.
    assert GOLDEN.read_bytes().decode("utf-8").replace("\r\n", "\n") == text


class _Entry:
    def __init__(self, partition):
        self.partition = partition


class _FakeLake:
    def __init__(self, _root):
        tapes = _d1_tapes()
        self._d1 = [row for symbol in SYMBOLS for row in tapes[symbol][-40:]]
        m5 = _spy_m5()
        self._m5 = {"month=2026-07": m5[:10], "month=2026-08": m5[10:]}
        self.manifest = self

    def resolve(self, dataset):
        assert dataset == "bar_m5"
        return type("R", (), {"entries": [_Entry(key) for key in ("month=2026-08", "month=2026-07")]})()

    def read_rows(self, dataset, partition=None, symbols=None):
        if dataset == "bar_d1":
            return [row for row in self._d1 if row["symbol"] in symbols]
        return [row for row in self._m5[partition] if row["symbol"] in symbols]


def test_the_benchmark_loader_is_unchanged(tmp_path, monkeypatch):
    import pandas as pd

    import journal_regime_fill
    from research_warehouse import config, ingest_existing, store

    monkeypatch.setattr(config, "get_research_store_dir", lambda: tmp_path)
    monkeypatch.setattr(store, "ResearchStore", _FakeLake)
    tapes = _d1_tapes()

    def durable(symbol):
        rows = tapes[symbol][-60:]
        return pd.DataFrame(
            {
                "datetime": [pd.Timestamp(row["session_date"]) for row in rows],
                **{key: [row[key] for row in rows] for key in ("open", "high", "low", "close", "volume")},
            }
        )

    monkeypatch.setattr(ingest_existing, "read_durable_daily_bars", durable)
    d1, spy_m5, source = journal_regime_fill.load_benchmark_bars()
    assert source == "lake+durable_d1"
    assert [len(d1[symbol]) for symbol in SYMBOLS] == [60, 60, 60]
    assert d1["SPY"][0] == tapes["SPY"][-40]
    extra = d1["QQQ"][-1]
    assert extra["session_date"] == tapes["QQQ"][-41]["session_date"] and extra["symbol"] == "QQQ"
    assert float(extra["close"]) == tapes["QQQ"][-41]["close"]
    assert spy_m5 == _spy_m5()[:10] + _spy_m5()[10:]
