"""RV-H1-HISTORY, items 1/2/5: the backup H1 history is kept FRESH.

Red-before-fix tests for review blocker B1. Three properties, each driven
through the real seam and each proved by a WRONG PRODUCT RESULT on the sweep
tip c4df3ac8:

* **the need is measured on the PRIMARY series** (`_h1_bars_for_watch`). Today
  the panel measures it on whichever series `h1_bars_for_watch` CHOSE, so once
  the fallback holds 45 bars the refresh is never asked for again and the watch
  is judged on ageing bars until the rule's `STALE_AFTER` retires it for good.
  The review's own reproduction: `fallback_bars 45 source yfinance
  refresh_requests 0`;
* **the refresh cadence follows completed SESSION-ALIGNED H1 buckets**
  (`H1HistoryCache.request`), not the wall-clock hour. The primary series
  (`indicators.h1_ema_bounce.closed_h1_bars`) buckets from the session open -
  06:30, 07:30 ... 12:30, the last one 30 minutes long and closed at the bell -
  so a new answer can only exist when one of those has completed. Today the
  refusal is keyed on `moment.replace(minute=0, ...)`, which both refetches
  inside one bucket and refetches all evening when no bucket can complete;
* **nothing on the Qt thread fetches** (item 5) - the guard stays.

No network: every `H1HistoryCache` here is built with a fake downloader, the
way `tests/test_ws_10c_h1_retester_builder.py:76` `_h1_cache` does. `yfinance`
is never imported by this file.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_ws_10c_h1_retester import (  # noqa: E402
    WATCH_KIND,
    golden_long_h1_bars,
    golden_m5_series,
)

#: The desk's regular session on the local (Pacific) clock, measured from
#: ``market_session.get_market_session_open_naive`` in this environment:
#: 06:30 -> 13:00, so the completed H1 buckets are 06:30, 07:30 ... 12:30 and
#: the last one closes at the bell.
SESSION_DAY = datetime(2026, 8, 26)  # a Wednesday, a full exchange session
NEXT_SESSION_DAY = datetime(2026, 8, 27)  # a Thursday, the next one


def _at(day: datetime, hour: int, minute: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Harness: a real H1HistoryCache with a fake download, and the real panel.
# ---------------------------------------------------------------------------
class _Frame:
    """The shape `frame_to_h1_bars` reads, built from plain bar dicts."""

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


def _cache(bars, *, on_download=None):
    """A real `H1HistoryCache` whose "download" is this list. No yfinance."""
    from h1_history import H1HistoryCache

    rows = list(bars or ())

    def _download(symbol, **kwargs):
        if on_download is not None:
            on_download(symbol)
        return _Frame(rows)

    return H1HistoryCache(downloader=_download)


def _settle(timeout: float = 5.0) -> None:
    """Wait for every in-flight H1 fetch thread to finish.

    `request` is deliberately non-blocking, so without this the NEXT request
    would be refused because one is still in flight - a refusal for the wrong
    reason, which would make the cadence assertions below meaningless.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("h1-history-") and thread.is_alive()
        ]
        if not alive:
            return
        time.sleep(0.01)
    raise AssertionError("an H1 history fetch thread never finished")


def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - headless without Qt
        pytest.skip("PySide6 not installed")
    return QApplication.instance() or QApplication([])


def _panel(monkeypatch, tmp_path, *, m5_bars):
    _qt_app()
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    panel._chart_watches_path = tmp_path / "chart_watches.json"
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: list(m5_bars))
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])
    return panel


class _AskRecorder:
    """A cache stand-in that answers from memory and records every ASK.

    This is the panel's real collaborator surface (`bars_for` / `request` /
    `unavailable`); what is under test is the panel's DECISION to ask, so the
    asks are counted rather than performed.
    """

    def __init__(self, fallback):
        self.fallback = [dict(bar) for bar in (fallback or ())]
        self.requests: list[datetime | None] = []

    def bars_for(self, symbol):
        return [dict(bar) for bar in self.fallback]

    def request(self, symbol, *, now=None):
        self.requests.append(now)
        return True

    def unavailable(self, symbol):
        return False

    def last_refresh_failed(self, symbol):
        return False


def _armed(panel, cache):
    """Arm the H1 watch and forget the asks ARMING itself made.

    `arm_chart_watch_for` builds the watch's reason and health line through the
    same `_h1_bars_for_watch` seam, on the real wall clock. Those asks are not
    what any test below is measuring, so the recorder starts from zero after
    the watch exists.
    """
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)
    cache.requests.clear()
    return watch


# ---------------------------------------------------------------------------
# Item 1 - the need is measured on the PRIMARY series
# ---------------------------------------------------------------------------
def test_a_full_fallback_is_still_refreshed_while_the_primary_window_is_short(
    monkeypatch, tmp_path
):
    """The review's reproduction, at the panel's own seam.

    The desk's cached M5 window aggregates to 35 completed H1 bars - short of
    the 45-bar warm-up, permanently, because SN2 keeps five sessions. The
    fallback holds exactly 45. A day later the watch is still being judged on
    yesterday's fetch and NOTHING has asked for a new one.
    """
    bars, _ = golden_long_h1_bars()
    primary = golden_m5_series(bars[:35])
    fallback = [dict(bar) for bar in bars[-45:]]

    panel = _panel(monkeypatch, tmp_path, m5_bars=primary)
    cache = _AskRecorder(fallback)
    panel._h1_history = cache
    watch = _armed(panel, cache)

    # The primary really is short and the fallback really is exactly full:
    # this is a freshness question, not a warm-up one.
    from indicators.h1_ema_bounce import WARMUP_BARS, closed_h1_bars

    assert len(closed_h1_bars(primary)) == 35
    assert len(fallback) == WARMUP_BARS == 45

    a_day_later = _at(NEXT_SESSION_DAY, 7, 30)
    series, source = panel._h1_bars_for_watch(watch, now=a_day_later)

    assert (len(series), source) == (45, "yfinance")
    assert len(cache.requests) == 1, (
        "fallback_bars %d source %s refresh_requests %d"
        % (len(series), source, len(cache.requests))
    )
    assert cache.requests[0] == a_day_later


def test_an_empty_fallback_and_a_short_primary_asks_for_the_first_fetch(
    monkeypatch, tmp_path
):
    """The warm-up case, kept as the guard: 30 primary bars, nothing fetched."""
    bars, _ = golden_long_h1_bars()
    primary = golden_m5_series(bars[:30])

    panel = _panel(monkeypatch, tmp_path, m5_bars=primary)
    cache = _AskRecorder([])
    panel._h1_history = cache
    watch = _armed(panel, cache)

    series, source = panel._h1_bars_for_watch(watch, now=_at(SESSION_DAY, 11, 45))

    assert (len(series), source) == (30, "cache")
    assert len(cache.requests) == 1


def test_a_full_primary_history_never_asks_even_with_a_full_fallback_present(
    monkeypatch, tmp_path
):
    """The property the fallback exists to protect, at the cache-need seam.

    55 primary bars answer on their own, so the network is never asked - even
    though a 45-bar fallback is sitting right there. Measuring need on the
    primary must not turn into measuring it on nothing.
    """
    bars, _ = golden_long_h1_bars()

    panel = _panel(monkeypatch, tmp_path, m5_bars=golden_m5_series(bars))
    cache = _AskRecorder(bars[-45:])
    panel._h1_history = cache
    watch = _armed(panel, cache)

    series, source = panel._h1_bars_for_watch(watch, now=_at(SESSION_DAY, 11, 45))

    assert (len(series), source) == (55, "cache")
    assert cache.requests == []


# ---------------------------------------------------------------------------
# Item 2 - the cadence is a completed SESSION-ALIGNED bucket, not a clock hour
# ---------------------------------------------------------------------------
def test_two_asks_inside_one_session_bucket_are_one_request_not_two():
    """11:45 and 12:15 are the same in-progress 11:30 bucket.

    Nothing new has completed between them - the 11:30 bucket does not close
    until 12:30 - so the second ask cannot produce a different answer and must
    be refused. The clock-hour key sees 11:00 then 12:00 and refetches.
    """
    cache = _cache([])
    asks = [
        (_at(SESSION_DAY, 11, 45), True),  # the 10:30 bucket closed at 11:30
        (_at(SESSION_DAY, 12, 15), False),  # still the same completed bucket
        (_at(SESSION_DAY, 13, 0), True),  # the short 12:30 bucket closed at the bell
    ]

    got: list[bool] = []
    for moment, _expected in asks:
        got.append(cache.request("AAPL", now=moment))
        _settle()

    assert got == [expected for _moment, expected in asks], (
        "requests at %s -> %s"
        % ([m.strftime("%H:%M") for m, _ in asks], got)
    )


def test_no_bucket_completes_after_the_bell_so_the_evening_asks_for_nothing():
    """Outside the session there is no new question, so there is no new fetch.

    After the 12:30 bucket closes at 13:00 the next completed bucket is the
    NEXT session's 06:30, closing at 07:30. Every poll in between - the 60 s
    armed poll runs all evening - must be refused. The clock-hour key grants a
    fresh fetch every hour until midnight.
    """
    cache = _cache([])
    asks = [
        (_at(SESSION_DAY, 13, 0), True),  # the 12:30 bucket, at the bell
        (_at(SESSION_DAY, 14, 0), False),
        (_at(SESSION_DAY, 16, 0), False),
        (_at(NEXT_SESSION_DAY, 6, 45), False),  # 06:30 has not closed yet
        (_at(NEXT_SESSION_DAY, 7, 30), True),  # now it has
    ]

    got: list[bool] = []
    for moment, _expected in asks:
        got.append(cache.request("AAPL", now=moment))
        _settle()

    assert got == [expected for _moment, expected in asks], (
        "requests at %s -> %s"
        % ([m.strftime("%m-%d %H:%M") for m, _ in asks], got)
    )


def test_only_one_fetch_is_ever_in_flight_for_a_symbol():
    """Unchanged, kept as the guard: a second ask during a fetch is refused."""
    from h1_history import H1HistoryCache

    started = threading.Event()
    release = threading.Event()

    def _slow(symbol, **kwargs):
        started.set()
        release.wait(5.0)
        return _Frame([])

    cache = H1HistoryCache(downloader=_slow)
    try:
        assert cache.request("AAPL", now=_at(SESSION_DAY, 11, 45)) is True
        assert started.wait(5.0)
        # A DIFFERENT completed bucket, so only the in-flight guard can refuse.
        assert cache.request("AAPL", now=_at(NEXT_SESSION_DAY, 7, 30)) is False
    finally:
        release.set()
        _settle()


# ---------------------------------------------------------------------------
# Item 5 - nothing on the Qt thread fetches
# ---------------------------------------------------------------------------
def test_the_armed_poll_never_downloads_on_the_calling_thread(monkeypatch, tmp_path):
    """The 60 s armed poll runs on the Qt thread; a yfinance round trip there
    is the freeze this whole module was written to avoid."""
    bars, _ = golden_long_h1_bars()
    primary = golden_m5_series(bars[:35])
    threads: list[str] = []

    # An empty download, so the fallback never fills and every poll below has
    # a real reason to ask - the thread question is asked of a poll that DOES
    # reach the downloader.
    cache = _cache(
        [], on_download=lambda symbol: threads.append(threading.current_thread().name)
    )
    panel = _panel(monkeypatch, tmp_path, m5_bars=primary)
    panel._h1_history = cache
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)
    _settle()
    threads.clear()

    caller = threading.current_thread().name
    panel._h1_bars_for_watch(watch, now=_at(NEXT_SESSION_DAY, 7, 30))
    _settle()

    assert threads, "the poll never asked for the refresh at all"
    assert caller not in threads, (
        "the download ran on the calling thread %r (threads=%s)" % (caller, threads)
    )
