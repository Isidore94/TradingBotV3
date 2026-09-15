"""PCT-1 review round - the intraday history cache: batching and the bell.

Two blockers the reviewer reproduced against copies of the live stores:

* **B4** - one tick that armed 95 watches issued 285 single-ticker
  `yf.download` calls on 286 threads. `request` now only ENQUEUES; the
  cache's ONE worker drains the queue in chunks of at most
  `DOWNLOAD_CHUNK_SYMBOLS` and issues one multi-ticker download per chunk.
* **B5** - the builder's after-the-bell refusal dropped the CLOSING bar of
  every session: a 13:05 poll was refused, so the 12:45-13:00 M15 bar was
  never fetched while the health cell still read `from yfinance`. The clause
  is gone and the rule is "once per completed session bucket, including the
  closing one, never outside the session's own buckets".

Every download here is a fake list; nothing in this file touches yfinance.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SESSION_DAY = datetime(2026, 8, 26)  # a Wednesday: session 06:30 -> 13:00 local
NEXT_SESSION_DAY = datetime(2026, 8, 27)


def _at(day: datetime, hour: int, minute: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


class _Frame:
    columns = None

    def __init__(self, bars):
        self._bars = list(bars or ())
        self.empty = not self._bars

    def iterrows(self):
        for bar in self._bars:
            yield bar["dt"], {
                "Open": bar["open"],
                "High": bar["high"],
                "Low": bar["low"],
                "Close": bar["close"],
            }


def _session_bars(day: datetime, interval_minutes: int) -> list[dict]:
    """Every bucket of one regular session, oldest first."""
    start = _at(day, 6, 30)
    count = 390 // interval_minutes
    return [
        {
            "dt": start + timedelta(minutes=interval_minutes * index),
            "open": 100.0 + index,
            "high": 100.5 + index,
            "low": 99.5 + index,
            "close": 100.0 + index,
            "volume": 1_000,
        }
        for index in range(count)
    ]


def _settle(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [
            thread
            for thread in threading.enumerate()
            if thread.is_alive() and "history" in thread.name
        ]
        if not alive:
            return
        time.sleep(0.01)
    raise AssertionError("a history worker never finished")


# ---------------------------------------------------------------------------
# B5 - the closing bucket is a completed bucket
# ---------------------------------------------------------------------------
def test_the_hourly_cadence_is_what_shipped_before_pct_1():
    """The reviewer's own poll sequence, pinned.

    07:45 / 12:45 / 13:05 / 15:00 / 19:00 -> THREE downloads: the 06:30
    bucket, the 11:30 one, and the short 12:30 one the bell closes. The
    builder's after-bell clause made the third a refusal, which is how the
    session's last hour stopped being fetched at all.
    """
    from intraday_history import IntradayHistoryCache

    seen: list[list[str]] = []

    def _download(symbols, **_kwargs):
        seen.append(list(symbols))
        return {str(symbols[0]): _Frame([])}

    cache = IntradayHistoryCache(60, downloader=_download)
    answers = []
    for moment in (
        _at(SESSION_DAY, 7, 45),
        _at(SESSION_DAY, 12, 45),
        _at(SESSION_DAY, 13, 5),
        _at(SESSION_DAY, 15, 0),
        _at(SESSION_DAY, 19, 0),
    ):
        answers.append(cache.request("AAPL", now=moment))
        _settle()

    assert answers == [True, True, True, False, False]
    assert len(seen) == 3


def test_the_closing_quarter_hour_bar_is_fetched_after_the_bell():
    """The bar the builder's clause lost: 12:45-13:00 on a 15-minute grid."""
    from intraday_history import IntradayHistoryCache

    bars = _session_bars(SESSION_DAY, 15)
    assert bars[-1]["dt"] == _at(SESSION_DAY, 12, 45)

    cache = IntradayHistoryCache(15, downloader=lambda s, **k: {s[0]: _Frame(bars)})
    # Midday first, exactly as the armed poll would have asked.
    assert cache.request("AAPL", now=_at(SESSION_DAY, 12, 1)) is True
    _settle()
    assert cache.bars_for("AAPL")[-1]["dt"] == _at(SESSION_DAY, 11, 45)

    assert cache.request("AAPL", now=_at(SESSION_DAY, 13, 5)) is True
    _settle()

    assert cache.bars_for("AAPL")[-1]["dt"] == _at(SESSION_DAY, 12, 45)
    # ...and then nothing more until the next session's first bucket closes.
    assert cache.request("AAPL", now=_at(SESSION_DAY, 19, 0)) is False
    assert cache.request("AAPL", now=_at(NEXT_SESSION_DAY, 6, 40)) is False
    assert cache.request("AAPL", now=_at(NEXT_SESSION_DAY, 6, 46)) is True
    _settle()


# ---------------------------------------------------------------------------
# B4 - one worker, batched downloads
# ---------------------------------------------------------------------------
def _many(count: int) -> list[str]:
    return [f"SYM{index:03d}" for index in range(count)]


def test_ninety_five_symbols_are_two_downloads_on_one_worker():
    from intraday_history import DOWNLOAD_CHUNK_SYMBOLS, IntradayHistoryCache

    assert DOWNLOAD_CHUNK_SYMBOLS == 50
    calls: list[list[str]] = []
    workers: set[str] = set()
    release = threading.Event()

    def _download(symbols, **_kwargs):
        calls.append(list(symbols))
        workers.add(threading.current_thread().name)
        release.wait(0.05)  # hold the worker so a second one would be visible
        return {name: _Frame([]) for name in symbols}

    cache = IntradayHistoryCache(15, downloader=_download)
    symbols = _many(95)
    queued = [cache.request(name, now=_at(SESSION_DAY, 12, 1)) for name in symbols]
    _settle(timeout=10.0)

    assert queued == [True] * 95
    assert len(calls) == 2, [len(chunk) for chunk in calls]
    assert sorted(len(chunk) for chunk in calls) == [45, 50]
    # Every symbol asked for exactly once, and ONE worker did all of it.
    assert sorted(name for chunk in calls for name in chunk) == sorted(symbols)
    assert len(workers) == 1, workers
    assert "history" in next(iter(workers))


def test_a_second_ask_inside_one_bucket_is_never_queued_twice():
    from intraday_history import IntradayHistoryCache

    calls: list[list[str]] = []

    def _download(symbols, **_kwargs):
        calls.append(list(symbols))
        return {name: _Frame([]) for name in symbols}

    cache = IntradayHistoryCache(15, downloader=_download)
    first = cache.request("AAPL", now=_at(SESSION_DAY, 12, 1))
    second = cache.request("AAPL", now=_at(SESSION_DAY, 12, 5))
    _settle()

    assert (first, second) == (True, False)
    assert calls == [["AAPL"]]


def test_one_bad_symbol_in_a_good_batch_marks_only_itself():
    from intraday_history import IntradayHistoryCache

    bars = _session_bars(SESSION_DAY, 15)

    def _download(symbols, **_kwargs):
        return {
            name: (None if name == "SYM001" else _Frame(bars)) for name in symbols
        }

    cache = IntradayHistoryCache(15, downloader=_download)
    for name in _many(3):
        cache.request(name, now=_at(SESSION_DAY, 13, 5))
    _settle()

    assert cache.last_refresh_failed("SYM001") is True
    assert cache.unavailable("SYM001") is True
    for name in ("SYM000", "SYM002"):
        assert cache.last_refresh_failed(name) is False
        assert len(cache.bars_for(name)) == 26


def test_the_cache_says_when_its_last_bucket_closed():
    """The reader the armed poll asks before it spends a pass over the bars."""
    from intraday_history import IntradayHistoryCache

    m15 = IntradayHistoryCache(15, downloader=lambda s, **k: {})
    h1 = IntradayHistoryCache(60, downloader=lambda s, **k: {})

    assert m15.last_completed_bucket_end(_at(SESSION_DAY, 12, 1)) == _at(
        SESSION_DAY, 12, 0
    )
    assert m15.last_completed_bucket_end(_at(SESSION_DAY, 12, 5)) == _at(
        SESSION_DAY, 12, 0
    )
    # The bell closes the last bucket of both grids.
    assert m15.last_completed_bucket_end(_at(SESSION_DAY, 13, 5)) == _at(
        SESSION_DAY, 13, 0
    )
    assert h1.last_completed_bucket_end(_at(SESSION_DAY, 13, 5)) == _at(
        SESSION_DAY, 13, 0
    )
