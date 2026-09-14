"""Red tests for the Trade Mentor popup context (WS-TM gate 110 follow-up).

The UI already files the trader's words.  These tests pin the narrow extra fact
sheet: a light read-only market snapshot that follows each individual note.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from time import monotonic, sleep
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.qt
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtTest = pytest.importorskip("PySide6.QtTest")

PACIFIC = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 14, 10, 0, tzinfo=PACIFIC)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _sessions_ending(last: date, count: int) -> list[date]:
    import market_calendar

    found: list[date] = []
    day = last
    while len(found) < count:
        if market_calendar.is_session(day):
            found.append(day)
        day = date.fromordinal(day.toordinal() - 1)
    return list(reversed(found))


def _m5_bars(*, end_minute: int = 55) -> list[dict]:
    """A whole regular session, whose last seven closes are 100 through 106.

    Session VWAP needs the session's volume. A fixture with only the last seven
    bars is not a smaller valid session; it is a different claim. `end_minute`
    is measured from 09:00 so the stale fixture can end at 09:00 without ever
    constructing an invalid negative minute.
    """
    session_open = datetime(2026, 9, 14, 6, 30, tzinfo=PACIFIC)
    last_start = datetime(2026, 9, 14, 9, 0, tzinfo=PACIFIC) + timedelta(
        minutes=end_minute
    )
    bars: list[dict] = []
    starts: list[datetime] = []
    stamp = session_open
    while stamp <= last_start:
        starts.append(stamp)
        stamp += timedelta(minutes=5)
    for index, stamp in enumerate(starts):
        # The final seven bars give the exact 30-minute fact. Earlier session
        # volume makes session VWAP a real measured value, not an approximation.
        close = (
            100.0 + (index - (len(starts) - 7))
            if index >= len(starts) - 7
            else 90.0
        )
        bars.append(
            {
                "dt": stamp,
                "open": close - 0.25,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000.0 + index,
            }
        )
    return bars


def _d1_bars(last: date = date(2026, 9, 11)) -> list[dict]:
    """21 exchange sessions, giving an exact SMA20 and five-session return."""
    return [
        {
            "dt": day.isoformat(),
            "open": 99.5 + index,
            "high": 100.5 + index,
            "low": 99.0 + index,
            "close": 100.0 + index,
            "volume": 1_000_000.0,
        }
        for index, day in enumerate(_sessions_ending(last, 21))
    ]


def _every_symbol(value):
    from trade_mentor_context import SYMBOLS

    return {symbol: value() for symbol in SYMBOLS}


def _reading(context: dict, symbol: str) -> dict:
    return next(row for row in context["readings"] if row["symbol"] == symbol)


def _signal_arrived(spy, *, timeout_seconds: float = 2.0) -> bool:
    """Wait without starving a Python worker on Windows.

    PySide's ``QSignalSpy.wait`` can return ``False`` here despite incrementing
    its count after a Python ``QThread`` delivery. Pumping Qt and yielding a
    few milliseconds proves the actual signal rather than that harness quirk.
    """
    deadline = monotonic() + timeout_seconds
    app = QtWidgets.QApplication.instance()
    while monotonic() < deadline:
        app.processEvents()
        if spy.count() >= 1:
            return True
        sleep(0.01)
    app.processEvents()
    return spy.count() >= 1


def test_context_is_a_small_flat_all_symbol_snapshot_with_completed_bar_arithmetic():
    """A 30-minute M5 fact and a five-session D1 fact are shown, never guessed."""
    from trade_mentor_context import SYMBOLS, build_context

    context = build_context(
        now=NOW,
        m5_bars=_every_symbol(_m5_bars),
        d1_bars=_every_symbol(_d1_bars),
        sources={"m5": "cached", "d1": "yahoo"},
    )

    assert SYMBOLS == (
        "VXX", "RSP", "USO", "TLT", "IWM", "QQQ", "SPY", "XLB", "XLC",
        "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    )
    assert context["schema"] == "trade_mentor_context_v1"
    assert context["captured_at"] == NOW.isoformat()
    assert context["sources"] == {"m5": "cached", "d1": "yahoo"}
    assert [row["symbol"] for row in context["readings"]] == list(SYMBOLS)
    assert len(json.dumps(context, sort_keys=True).encode("utf-8")) <= 6 * 1024

    spy = _reading(context, "SPY")
    assert spy["m5_status"] == "measured"
    assert spy["m5_change_30m_pct"] == pytest.approx(6.0)
    assert spy["m5_direction"] == "up"
    assert spy["m5_vs_session_vwap"] == "above"
    assert spy["d1_status"] == "measured"
    assert spy["d1_change_5d_pct"] == pytest.approx((120.0 - 115.0) / 115.0 * 100.0)
    assert spy["d1_vs_sma20"] == "above"
    assert spy["m5_as_of"] == "2026-09-14T09:55:00-07:00"
    assert spy["d1_as_of"] == "2026-09-11"
    # Flat scalar keys are the depth-safe persisted contract for ai_summary._bounded.
    assert not isinstance(spy["m5_change_30m_pct"], dict)
    assert not isinstance(spy["d1_change_5d_pct"], dict)


def test_context_excludes_forming_future_and_malformed_bars_and_names_short_or_stale_data():
    from trade_mentor_context import build_context

    clean = build_context(
        now=NOW, m5_bars={"VXX": _m5_bars()}, d1_bars={"VXX": _d1_bars()}
    )
    forming = _m5_bars() + [
        {
            # This bar has started, but cannot finish until after `NOW`.
            "dt": datetime(2026, 9, 14, 10, 0, tzinfo=PACIFIC),
            "open": 106.0, "high": 201.0, "low": 105.0, "close": 200.0, "volume": 1.0,
        }
    ]
    without_future = build_context(
        now=NOW, m5_bars={"VXX": forming}, d1_bars={"VXX": _d1_bars()}
    )
    assert _reading(without_future, "VXX") == _reading(clean, "VXX")

    duplicate = _m5_bars() + [dict(_m5_bars()[-1])]
    malformed = build_context(
        now=NOW,
        m5_bars={"VXX": duplicate, "RSP": [{"dt": "2026-09-14T09:30:00"}]},
        d1_bars={"VXX": _d1_bars(), "RSP": _d1_bars()},
    )
    vxx = _reading(malformed, "VXX")
    rsp = _reading(malformed, "RSP")
    assert vxx["m5_status"] == "unavailable"
    assert vxx["m5_change_30m_pct"] is None
    assert "duplicate" in vxx["m5_reason"].lower()
    assert rsp["m5_status"] == "unavailable"
    assert rsp["m5_change_30m_pct"] is None
    assert "aware" in rsp["m5_reason"].lower()

    # Six surviving bars spanning 35 minutes are not a 30-minute measurement.
    gap = [bar for bar in _m5_bars() if bar["dt"].strftime("%H:%M") != "09:40"]
    gappy = build_context(now=NOW, m5_bars={"VXX": gap}, d1_bars={"VXX": _d1_bars()})
    gappy_vxx = _reading(gappy, "VXX")
    assert gappy_vxx["m5_status"] == "unavailable"
    assert gappy_vxx["m5_change_30m_pct"] is None
    assert "gap" in gappy_vxx["m5_reason"].lower()

    stale_now = datetime(2026, 9, 14, 11, 0, tzinfo=PACIFIC)
    stale = build_context(
        now=stale_now,
        m5_bars={"VXX": _m5_bars(end_minute=0)},
        d1_bars={"VXX": _d1_bars(date(2026, 9, 10))},
    )
    stale_vxx = _reading(stale, "VXX")
    assert stale_vxx["m5_status"] == "stale"
    assert stale_vxx["m5_change_30m_pct"] is None
    assert "stale" in stale_vxx["m5_reason"].lower()
    assert stale_vxx["d1_status"] == "stale"
    assert stale_vxx["d1_change_5d_pct"] is None
    assert "stale" in stale_vxx["d1_reason"].lower()


class _GateLoader:
    def __init__(self, response):
        self.response = response
        self.calls: list[tuple[str, tuple[str, ...], int]] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, timeframe, symbols, *, now, timeout_seconds):
        self.calls.append((timeframe, tuple(symbols), threading.get_ident()))
        self.started.set()
        self.release.wait(1.0)
        return self.response[timeframe]


def test_context_service_batches_off_the_gui_thread_caches_and_throttles_failures():
    """No timer fetches. The only work starts from a prompt/manual request."""
    from trade_mentor_context import SYMBOLS
    from ui.services.trade_mentor_context_service import TradeMentorContextService

    response = {"m5": _every_symbol(_m5_bars), "d1": _every_symbol(_d1_bars)}
    loader = _GateLoader(response)
    service = TradeMentorContextService(loader=loader, clock=lambda: NOW, timeout_seconds=1)
    ready = QtTest.QSignalSpy(service.contextReady)
    main_thread = threading.get_ident()
    started = monotonic()
    assert service.request_context("slot-0900", now=NOW) is True
    assert monotonic() - started < 0.25, "submitting a read cannot wait for Yahoo"
    assert loader.started.wait(1.0)
    assert loader.calls[0][2] != main_thread, "the loader never runs on Qt's thread"
    assert service.request_context("slot-0901", now=NOW) is False, "one worker at a time"
    loader.release.set()
    assert _signal_arrived(ready)
    assert [(kind, names) for kind, names, _thread in loader.calls] == [
        ("m5", SYMBOLS), ("d1", SYMBOLS)
    ]

    # Same hour: the card gets a snapshot again without another batch.
    cached = QtTest.QSignalSpy(service.contextReady)
    assert service.request_context("manual-0910", now=NOW + timedelta(minutes=10)) is True
    assert _signal_arrived(cached, timeout_seconds=0.5)
    assert len(loader.calls) == 2
    service.shutdown(timeout_ms=250)

    class _BrokenLoader:
        def __init__(self):
            self.calls = 0

        def __call__(self, *_args, **_kwargs):
            self.calls += 1
            raise OSError("offline")

    broken = _BrokenLoader()
    failed = TradeMentorContextService(loader=broken, clock=lambda: NOW, timeout_seconds=1)
    unavailable = QtTest.QSignalSpy(failed.contextUnavailable)
    assert failed.request_context("slot-1000", now=NOW) is True
    assert _signal_arrived(unavailable)
    assert broken.calls == 1
    assert failed.request_context("slot-1001", now=NOW + timedelta(minutes=5)) is False
    assert broken.calls == 1, "a failed hour is throttled rather than retried"
    failed.shutdown(timeout_ms=250)
