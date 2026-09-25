"""P0-2 2e: Yahoo daily-bar misses and refreshes are fetched in batches; the bars are unchanged.

`yfinance.download` runs for real on both paths; only `Ticker.history` (the one
network call per ticker) is replaced by a fixture. The per-symbol path and the
batch path must return equal frames and write equal cache and durable files.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from master_avwap_lib import legacy  # noqa: E402

TODAY = pd.Timestamp(datetime.now().date())


def _history_frame(symbol: str, days: int) -> pd.DataFrame:
    """What `Ticker.history(period=f"{days}d")` returns: exchange-local dates, int volume."""
    dates = pd.bdate_range(end=TODAY - pd.Timedelta(days=1), start=TODAY - pd.Timedelta(days=days))
    if symbol == "GAP":
        dates = dates.delete(len(dates) // 2)  # a missing session forces NaN alignment in a batch
    base = 10.0 + (sum(map(ord, symbol)) % 50)
    rows = []
    for index, _day in enumerate(dates):
        close = round(base + index * 0.25, 2)
        rows.append([close - 0.1, close + 0.5, close - 0.5, close, close, 1_000_000 + index])
    frame = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(dates, name="Date").tz_localize("America/New_York"),
        columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    )
    frame["Volume"] = frame["Volume"].astype("int64")
    return frame


class _FakeTicker:
    calls: list[tuple[str, str]] = []

    def __init__(self, ticker):
        self.ticker = str(ticker).upper()

    def history(self, period=None, **_kwargs):
        _FakeTicker.calls.append((self.ticker, period))
        if self.ticker == "ERR":
            raise RuntimeError("fixture: Yahoo refused")
        if self.ticker == "NIL":
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
        return _history_frame(self.ticker, int(str(period).rstrip("d")))


#: symbol -> (days asked by the scan, cache state): "stale" = history but old, "cold" = nothing.
REQUESTS = {
    "AAA": (40, "stale"),
    "GAP": (40, "stale"),
    "ERR": (40, "stale"),
    "NIL": (40, "stale"),
    "CCC": (200, "cold"),
    "DDD": (200, "cold"),
    "LONE": (260, "cold"),
}


def _stale_cache_frame() -> pd.DataFrame:
    dates = pd.bdate_range(end=TODAY - pd.Timedelta(days=12), periods=160)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [50.0] * len(dates),
            "high": [51.0] * len(dates),
            "low": [49.0] * len(dates),
            "close": [50.5] * len(dates),
            "volume": [900_000] * len(dates),
        }
    )


@pytest.fixture
def world(tmp_path, monkeypatch):
    import yfinance
    import yfinance.multi

    # The real `download` on both paths (the suite's offline guard replaces it);
    # only the per-ticker network call is the fixture.
    monkeypatch.setattr(yfinance, "download", yfinance.multi.download)
    monkeypatch.setattr(yfinance.multi, "Ticker", _FakeTicker)
    monkeypatch.setattr(legacy, "daily_bars_source_pin", lambda: "auto")

    def build(name: str) -> Path:
        root = tmp_path / name
        monkeypatch.setattr(legacy, "DAILY_BARS_CACHE_DIR", root / "cache")
        monkeypatch.setattr(legacy, "MASTER_AVWAP_DAILY_BARS_DIR", root / "durable")
        for state in (legacy._DAILY_BAR_FRAME_CACHE, legacy._DAILY_BAR_CACHE_TOUCHED_AT, legacy._DAILY_BAR_LIVE_FAILURE_AT):
            state.clear()
        legacy.reset_daily_bar_fetch_counts()
        for symbol, (_days, state) in REQUESTS.items():
            if state == "stale":
                legacy._write_cached_daily_bar_frame(symbol, _stale_cache_frame())
                legacy._DAILY_BAR_CACHE_TOUCHED_AT[symbol] = datetime.now() - timedelta(hours=2)
        _FakeTicker.calls = []
        return root

    yield build
    for state in (legacy._DAILY_BAR_FRAME_CACHE, legacy._DAILY_BAR_CACHE_TOUCHED_AT, legacy._DAILY_BAR_LIVE_FAILURE_AT):
        state.clear()
    legacy.reset_daily_bar_fetch_counts()


def _fetch_all() -> dict[str, pd.DataFrame]:
    return {symbol: legacy.fetch_daily_bars(None, symbol, days) for symbol, (days, _state) in REQUESTS.items()}


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_batched_bars_equal_the_per_symbol_bars(world):
    single_root = world("single")
    single = _fetch_all()
    single_calls = sorted(_FakeTicker.calls)
    single_files = _files(single_root)
    assert legacy._DAILY_BAR_FETCH_COUNTS["batch_calls"] == 0

    batch_root = world("batch")
    stored = legacy.prefetch_daily_bars_from_yahoo(None, {symbol: days for symbol, (days, _s) in REQUESTS.items()})
    batched = _fetch_all()

    assert stored == 4, "AAA, GAP (refresh period) and CCC, DDD (cold period); LONE is alone"
    assert legacy._DAILY_BAR_FETCH_COUNTS["batch_calls"] == 2
    assert legacy._DAILY_BAR_FETCH_COUNTS["served_from_batch"] == 4
    assert not legacy._DAILY_BAR_YAHOO_PREFETCH, "every stored frame was used"
    for symbol in REQUESTS:
        pd.testing.assert_frame_equal(batched[symbol], single[symbol], check_exact=True)
        assert legacy._get_daily_bar_source(batched[symbol]) == legacy._get_daily_bar_source(single[symbol])
    assert _files(batch_root) == single_files
    # ERR and NIL were asked twice (batch, then their own call); every other request once.
    assert sorted(set(_FakeTicker.calls)) == sorted(set(single_calls))
    assert sum(1 for symbol, _p in _FakeTicker.calls if symbol == "ERR") == 2


def test_the_scan_line_counts_hits_misses_and_refreshes(world, monkeypatch):
    world("count")
    legacy.prefetch_daily_bars_from_yahoo(None, {symbol: days for symbol, (days, _s) in REQUESTS.items()})
    _fetch_all()
    # AAA was just refreshed; with its last bar counted as the latest session it is now a hit.
    monkeypatch.setattr(legacy.daily_bar_cache, "cache_holds_latest_completed_session", lambda _last: True)
    legacy.fetch_daily_bars(None, "AAA", 40)
    line = legacy.daily_bar_fetch_summary_line()
    assert line.startswith("[daily-bar cache] lookups=8 hits=1 (12%) misses=3 refreshes=4 ")
    assert "served_from_batch=4 batch_calls=2 unused_batch_frames=0" in line


def test_no_batch_when_daily_bars_go_to_ib_first(world):
    world("ib")

    class _Ib:
        pass

    assert legacy.prefetch_daily_bars_from_yahoo(_Ib(), {"AAA": 40, "GAP": 40}) == 0
    assert _FakeTicker.calls == []
