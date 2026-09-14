"""RV-H1-HISTORY, items 3/4: a fetched bar completes on the SESSION rule, and
a failed refresh keeps the last good data and says so.

Red-before-fix tests for review blocker B1.

* **Item 3.** `frame_to_h1_bars` completes a bar through
  `completed_bars.is_completed_bar(bar, 60, now)` - `start + 60 min`. The
  primary series it has to agree with (`indicators.h1_ema_bounce.closed_h1_bars`)
  is SESSION-aligned: the 12:30 bucket is a 30-minute hour that closes at the
  bell, 13:00 market-local. So today the fetched series is missing the last bar
  of every session for a whole hour after the bell - the two sources disagree
  about what a completed bar is, which is exactly the thing the fallback exists
  to avoid.
* **Item 4.** A refresh that fails AFTER a success must keep the previous bars,
  be retried at the next boundary, and be reported honestly in the armed
  inventory's health cell. Today the health cell still reads a flat
  `H1 from yfinance`, so the trader cannot tell fresh bars from bars that
  stopped updating hours ago. `unavailable`'s meaning (nothing was EVER
  fetched) is unchanged and is pinned here too.

No network: the `H1HistoryCache` here is built with a fake downloader, the way
`tests/test_ws_10c_h1_retester_builder.py:76` `_h1_cache` does.
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


def _settle(timeout: float = 5.0) -> None:
    """Wait for every in-flight H1 fetch thread to finish.

    `request` is non-blocking by design, so a fetch the ARMING click started
    can still be running while the assertions below run; without this the
    success/failure sequence under test would race with it.
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


# ---------------------------------------------------------------------------
# Item 3 - the short closing bucket completes at the bell
# ---------------------------------------------------------------------------
def test_the_short_closing_bucket_is_admitted_at_the_bell_not_an_hour_later():
    """06:30 ... 12:30 with the last one 30 minutes long, on BOTH sources.

    At 12:59 only the 11:30 bar (which ended at 12:30) has completed. At 13:00
    the bell has rung and the 12:30 bar has completed too. Today the fetched
    series waits for 13:30 and so is one bar short of the primary for a full
    hour every session.
    """
    from h1_history import frame_to_h1_bars

    rows = [
        {
            "dt": _at(SESSION_DAY, 11, 30),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
        },
        {
            "dt": _at(SESSION_DAY, 12, 30),
            "open": 100.5,
            "high": 102.0,
            "low": 100.0,
            "close": 101.75,
        },
    ]

    before_the_bell = frame_to_h1_bars(_Frame(rows), now=_at(SESSION_DAY, 12, 59))
    assert [bar["dt"] for bar in before_the_bell] == [_at(SESSION_DAY, 11, 30)]

    at_the_bell = frame_to_h1_bars(_Frame(rows), now=_at(SESSION_DAY, 13, 0))
    assert [bar["dt"] for bar in at_the_bell] == [
        _at(SESSION_DAY, 11, 30),
        _at(SESSION_DAY, 12, 30),
    ], "admitted %d of 2 bars at the bell" % len(at_the_bell)
    assert at_the_bell[-1]["close"] == 101.75


def test_the_two_sources_agree_on_the_session_that_just_closed():
    """The property behind item 3, stated as the two series side by side.

    Same session, same bars: the desk's own aggregation and the fetched frame
    must return the same bucket starts at 13:00, or the fallback is not a
    substitute for the primary at all.
    """
    from h1_history import frame_to_h1_bars
    from indicators.h1_ema_bounce import closed_h1_bars

    bars, _ = golden_long_h1_bars()
    one_session = [bar for bar in bars if bar["dt"].date() == SESSION_DAY.date()]
    # The golden's 08-26 session stops at 11:30, so hang the short closing
    # bucket on the end by hand - that is the bar the two rules disagree about.
    closing = dict(one_session[-1])
    closing["dt"] = _at(SESSION_DAY, 12, 30)
    one_session = one_session + [closing]

    primary = closed_h1_bars(golden_m5_series(one_session))
    fetched = frame_to_h1_bars(_Frame(one_session), now=_at(SESSION_DAY, 13, 0))

    assert [bar["dt"] for bar in fetched] == [bar["dt"] for bar in primary]


# ---------------------------------------------------------------------------
# Item 4 - a failed refresh keeps the data and is reported honestly
# ---------------------------------------------------------------------------
def test_a_failed_refresh_after_a_success_keeps_the_bars_and_says_they_are_stale(
    monkeypatch, tmp_path
):
    """The health cell must distinguish fresh bars from bars that stopped.

    45 bars landed, then the next refresh failed. The bars are still the best
    answer available and the watch keeps being judged on them - but the trader
    reading the armed inventory has to be told they stopped updating, or an
    ageing verdict looks exactly like a live one.
    """
    from h1_history import H1HistoryCache

    bars, _ = golden_long_h1_bars()
    fallback = [dict(bar) for bar in bars[-45:]]
    primary = golden_m5_series(bars[:35])
    state = {"fail": False, "failures": 0}

    def _download(symbol, **kwargs):
        if state["fail"]:
            state["failures"] += 1
            raise RuntimeError("yfinance is down")
        return _Frame(fallback)

    cache = H1HistoryCache(downloader=_download)
    panel = _panel(monkeypatch, tmp_path, m5_bars=primary)
    panel._h1_history = cache
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)
    _settle()

    # First refresh lands: 45 bars, and the cell says where they came from.
    assert len(cache.fetch_now("AAPL", now=_at(SESSION_DAY, 13, 0))) == 45
    assert panel._armed_watch_note(watch) == "H1 from yfinance"
    _settle()

    # The next one fails. Nothing is thrown away - `fetch_now` hands back the
    # bars that are still the best answer, and `bars_for` still reads them.
    state["fail"] = True
    assert len(cache.fetch_now("AAPL", now=_at(SESSION_DAY, 14, 0))) == 45
    assert state["failures"] == 1  # the failure really happened
    assert len(cache.bars_for("AAPL")) == 45
    assert cache.bars_for("AAPL")[-1]["dt"] == fallback[-1]["dt"]
    series, source = panel._h1_bars_for_watch(watch, now=_at(SESSION_DAY, 14, 0))
    assert (len(series), source) == (45, "yfinance")

    assert panel._armed_watch_note(watch) == "H1 from yfinance (stale - last refresh failed)"

    # `unavailable` keeps its meaning: nothing was EVER fetched.
    assert cache.unavailable("AAPL") is False
    assert cache.last_refresh_failed("AAPL") is True

    # And it is retried at the next completed bucket, not abandoned.
    assert cache.request("AAPL", now=_at(NEXT_SESSION_DAY, 7, 30)) is True


def test_a_symbol_that_never_fetched_still_reads_unavailable(monkeypatch, tmp_path):
    """Unchanged, kept as the guard - do not let item 4 blur these two states."""
    from h1_history import H1HistoryCache

    bars, _ = golden_long_h1_bars()
    primary = golden_m5_series(bars[:35])

    def _boom(symbol, **kwargs):
        raise RuntimeError("no network")

    cache = H1HistoryCache(downloader=_boom)
    panel = _panel(monkeypatch, tmp_path, m5_bars=primary)
    panel._h1_history = cache
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)

    assert cache.fetch_now("AAPL", now=_at(SESSION_DAY, 13, 0)) == []
    assert cache.unavailable("AAPL") is True
    assert (
        panel._armed_watch_note(watch)
        == "not measured (35 of 45 H1 bars, yfinance unavailable)"
    )
