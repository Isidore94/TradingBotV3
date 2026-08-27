"""Every IB historical request must free its bar buffer when it is done.

Measured on the desk 2026-08-27: the desk settled at ~2.5 GB between warehouse
builds instead of ~1 GB, and crept all session. `BounceBot` stores every IB
historical reply in `self.data[reqId]`, and only the RRS path
(`request_historical_bars`) and the contract-bars path popped it. Five other
request paths deleted `self.data_ready_events[reqId]` and left the bar list
behind: **206 KB per 390-bar request, ~400 requests per scan cycle = ~80 MB a
cycle, 1.5-2 GB over a session**, held until the process exits.

The trader authorised this one `legacy.py` edit (2026-08-27 build prompt),
limited to freeing the buffer on the paths that already free the event. It is
not a detector change - every one of those paths copies the bars into
`latest_bars` / a DataFrame / an ATR before the buffer is dropped, and a
repo-wide sweep found each `reqId` is read exactly once, by the function that
created it, with `self.data` never iterated, persisted or touched outside the
class. It is still verified LIKE one: the golden fixtures and the whole
BounceBot suite must pass unchanged.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def bot():
    """A BounceBot with the IB socket replaced, nothing else stubbed."""
    from bounce_bot_lib.legacy import BounceBot

    made = BounceBot.__new__(BounceBot)
    made.data_lock = threading.Lock()
    made.data = {}
    made.data_ready_events = {}
    made.reqid_to_symbol = {}
    made.atr_cache = {}
    made.latest_bars = {}
    made._req_id = 100
    return made


def _bar(index):
    return SimpleNamespace(
        date=f"20260827  09:{30 + index:02d}:00",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000,
    )


# ------------------------------------------------------- the late-bar channel


def test_a_bar_for_an_unknown_reqid_is_dropped_not_stashed_forever(bot):
    """The second leak channel. `historicalData` auto-created
    `self.data[reqId] = []` for any reqId, so a straggler arriving after its
    requester gave up re-created the buffer nobody would ever free."""
    bot.historicalData(999, _bar(0))
    assert 999 not in bot.data, (
        "a bar nobody is waiting for must be discarded; auto-creating its "
        "buffer is how a timed-out request leaks after the fact"
    )


def test_a_bar_for_a_live_request_is_still_collected(bot):
    """The drop must not break the normal path: a buffer that EXISTS still
    fills, which is the only reason any of these requests work."""
    bot.data[42] = []
    bot.historicalData(42, _bar(0))
    bot.historicalData(42, _bar(1))
    assert len(bot.data[42]) == 2
    assert bot.data[42][0]["close"] == 100.5


def test_a_late_bar_cannot_mutate_the_list_its_caller_is_still_reading(bot):
    """Aliasing hazard: callers bind `bars = self.data.get(reqId)` - the same
    list object - and then work with it. A straggler appending to that list
    mid-read would change the data under them."""
    bot.data[42] = []
    bot.historicalData(42, _bar(0))
    holding = bot.data.pop(42)
    bot.historicalData(42, _bar(1))
    assert len(holding) == 1, "the caller's list must not grow behind its back"
    assert 42 not in bot.data


# --------------------------------------------------------- the request paths


def _drive(bot, method, *args, bars=3, **kwargs):
    """Run one request path with a fake IB that answers on the worker's event.

    `reqHistoricalData` is replaced by something that fills the buffer and
    sets the ready event, so the path runs to completion without a socket.
    """
    issued: list[int] = []

    def fake_request(*, reqId, **_rest):
        issued.append(reqId)
        for index in range(bars):
            bot.historicalData(reqId, _bar(index))
        bot.historicalDataEnd(reqId, "", "")

    bot.reqHistoricalData = fake_request
    bot.create_stock_contract = lambda symbol: SimpleNamespace(symbol=symbol)
    bot.getReqId = lambda: bot.__dict__.setdefault("_n", 0) or bot.__dict__.update(
        _n=bot.__dict__["_n"] + 1
    ) or bot.__dict__["_n"]
    method(*args, **kwargs)
    return issued


def _assert_drained(bot, issued, label):
    assert issued, f"{label}: no request was issued; the test proves nothing"
    leaked = [req for req in issued if req in bot.data]
    assert not leaked, (
        f"{label}: self.data still holds {leaked} after the request finished - "
        "that buffer is never freed again and is the session-long leak"
    )
    assert not [req for req in issued if req in bot.data_ready_events], (
        f"{label}: the ready event leaked too"
    )


def _one_symbol_scan_set(bot, symbol="AAA"):
    """`build_atr_cache` fetches whatever `get_scan_symbol_set` returns."""
    bot.get_scan_symbol_set = lambda: {symbol}


def test_build_atr_cache_frees_its_buffers(bot):
    """~400 of these per scan cycle at 206 KB each."""
    _one_symbol_scan_set(bot)
    issued = _drive(bot, bot.build_atr_cache, bars=30)
    _assert_drained(bot, issued, "build_atr_cache")
    assert bot.atr_cache, "the ATR must still be computed from the bars"


def test_build_atr_cache_frees_its_buffer_on_a_timeout_too(bot, monkeypatch):
    """The timeout branch is the one that matters most: a request that never
    answered still allocated its buffer."""
    _one_symbol_scan_set(bot)

    def silent_request(*, reqId, **_rest):
        pass  # never answers, never sets the event

    bot.reqHistoricalData = silent_request
    bot.create_stock_contract = lambda symbol: SimpleNamespace(symbol=symbol)
    bot.getReqId = lambda: 7

    class _Immediate(threading.Event):
        def wait(self, timeout=None):
            return False  # time out at once rather than burning 15 s

    monkeypatch.setattr(threading, "Event", _Immediate)
    bot.build_atr_cache()
    assert 7 not in bot.data, "a timed-out request must free its buffer as well"
    assert 7 not in bot.data_ready_events


@pytest.mark.parametrize(
    "method_name",
    ["check_dynamic_vwap_touches", "check_dynamic_vwap2_touches", "check_eod_vwap_touches"],
)
def test_the_vwap_touch_checks_free_their_buffers(bot, method_name):
    """Three more request paths, one per symbol per sweep, same leak."""
    bot.get_scan_symbol_set = lambda: {"AAA"}
    bot.longs = ["AAA"]
    bot.shorts = []
    issued = _drive(bot, getattr(bot, method_name), bars=30)
    _assert_drained(bot, issued, method_name)


@pytest.mark.parametrize(
    "method_name",
    ["check_dynamic_vwap_touches", "check_dynamic_vwap2_touches", "check_eod_vwap_touches"],
)
def test_the_vwap_touch_checks_free_their_buffers_on_a_timeout(bot, monkeypatch, method_name):
    bot.get_scan_symbol_set = lambda: {"AAA"}
    bot.longs = ["AAA"]
    bot.shorts = []
    bot.reqHistoricalData = lambda **_kwargs: None  # never answers
    bot.create_stock_contract = lambda symbol: SimpleNamespace(symbol=symbol)
    bot.getReqId = lambda: 11

    class _Immediate(threading.Event):
        def wait(self, timeout=None):
            return False

    monkeypatch.setattr(threading, "Event", _Immediate)
    getattr(bot, method_name)()
    assert 11 not in bot.data, "a timed-out VWAP check must free its buffer too"
    assert 11 not in bot.data_ready_events


def test_a_full_sweep_leaves_no_buffers_behind_at_all(bot):
    """The end state that matters: after a cycle's worth of requests,
    `self.data` is EMPTY. ~400 requests a cycle at 206 KB is the 1.5-2 GB a
    session that made the desk settle at 2.5 GB instead of 1 GB."""
    bot.get_scan_symbol_set = lambda: {"AAA", "BBB", "CCC"}
    bot.longs = ["AAA", "BBB", "CCC"]
    bot.shorts = []
    for method in (
        bot.build_atr_cache,
        bot.check_dynamic_vwap_touches,
        bot.check_dynamic_vwap2_touches,
        bot.check_eod_vwap_touches,
    ):
        bot.atr_cache.clear()
        _drive(bot, method, bars=30)
    assert bot.data == {}, f"buffers left behind after a sweep: {sorted(bot.data)}"
    assert bot.data_ready_events == {}
