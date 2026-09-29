"""The GUI thread never waits on the BounceBot child pipe for status reads.

2026-09-29 stall log: `refresh_health` -> `connection_status` and
`refresh_auto_regime` -> `get_auto_regime_reading` queued behind the proxy's
RPC lock for 1-2 s at every bar close (40 s of GUI stalls in one session).
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
FIXTURE = Path(__file__).with_name("fixtures") / "sn1_fake_bot.py"

#: A GUI-thread status read must answer well inside this; the fake child takes 2 s.
FAST_SECONDS = 0.25
SLOW_CHILD_SECONDS = 2.0


class _AliveProcess:
    exitcode = None

    @staticmethod
    def is_alive():
        return True


class _SlowConnection:
    """A child pipe whose every answer takes `delay` seconds."""

    def __init__(self, values: dict, delay: float) -> None:
        self.values = values
        self.delay = delay
        self._pending = None
        self.requests: list[str] = []

    def send(self, request) -> None:
        self._pending = request
        self.requests.append(str(request.get("name")))

    def poll(self, timeout=None) -> bool:
        time.sleep(self.delay)
        return True

    def recv(self):
        request = self._pending
        return {"id": request["id"], "ok": True, "result": self.values[request["name"]]}

    def close(self) -> None:
        pass


def _proxy(connection):
    from ui.services.bounce_process import BounceProcessProxy

    proxy = object.__new__(BounceProcessProxy)
    proxy._closed = threading.Event()
    proxy._rpc_lock = threading.Lock()
    proxy._ids = itertools.count(1)
    proxy._connection = connection
    proxy._process = _AliveProcess()
    if hasattr(proxy, "_init_status_cache"):
        proxy._init_status_cache()
        proxy._gui_thread = threading.current_thread()  # this test stands in for the Qt thread
    return proxy


def _values(**overrides):
    values = {
        "connection_status": True,
        "get_auto_regime_reading": {"label": "trend"},
        "entry_assist_state": {"window_active": False},
        "get_market_environment": "bullish",
        "pacing_delay_remaining": 0.0,
    }
    values.update(overrides)
    return values


def _timed(read):
    started = time.monotonic()
    try:
        value = read()
    except Exception as exc:  # noqa: BLE001 - the test inspects it
        value = exc
    return value, time.monotonic() - started


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_gui_status_reads_before_any_answer_are_unknown_and_instant():
    connection = _SlowConnection(_values(), SLOW_CHILD_SECONDS)
    proxy = _proxy(connection)

    connected, took = _timed(lambda: proxy.connection_status)
    assert took < FAST_SECONDS
    assert connected is False  # unknown reads as not connected, never "confirmed"

    for name in ("get_auto_regime_reading", "entry_assist_state", "get_market_environment", "pacing_delay_remaining"):
        value, took = _timed(getattr(proxy, name))
        assert took < FAST_SECONDS, name
        assert isinstance(value, Exception), name  # callers map this to unknown
    assert connection.requests == []  # the GUI thread sent nothing down the pipe


def test_gui_status_reads_serve_the_cache_while_the_child_is_busy():
    connection = _SlowConnection(_values(), 0.0)
    proxy = _proxy(connection)
    proxy._status_interval = 0.05
    proxy._start_status_poller()
    try:
        assert _wait_for(lambda: _timed(proxy.get_auto_regime_reading)[0] == {"label": "trend"})
        assert proxy.connection_status is True

        # Another thread's RPC holds the lock and the child is slow at bar close.
        connection.delay = SLOW_CHILD_SECONDS
        holder = threading.Thread(target=lambda: proxy._rpc("call", "get_market_environment"), daemon=True)
        holder.start()
        assert _wait_for(lambda: proxy._rpc_lock.locked(), timeout=1.0)

        connected, took = _timed(lambda: proxy.connection_status)
        assert took < FAST_SECONDS and connected is True
        reading, took = _timed(proxy.get_auto_regime_reading)
        assert took < FAST_SECONDS and reading == {"label": "trend"}
        environment, took = _timed(proxy.get_market_environment)
        assert took < FAST_SECONDS and environment == "bullish"
        holder.join(5.0)

        # A new answer from the poller replaces the cached one.
        connection.delay = 0.0
        connection.values["get_auto_regime_reading"] = {"label": "chop"}
        connection.values["connection_status"] = False
        assert _wait_for(lambda: _timed(proxy.get_auto_regime_reading)[0] == {"label": "chop"})
        assert _wait_for(lambda: proxy.connection_status is False)
    finally:
        proxy._closed.set()
        proxy._stop_status_poller()
    assert not proxy._status_thread.is_alive()


def test_a_stale_cached_status_is_unknown(monkeypatch):
    from ui.services import bounce_process

    proxy = _proxy(_SlowConnection(_values(), SLOW_CHILD_SECONDS))
    proxy._store_status("get", "connection_status", True)
    assert proxy.connection_status is True
    monkeypatch.setattr(bounce_process, "STATUS_STALE_SECONDS", -1.0)
    assert proxy.connection_status is False


def test_non_gui_threads_still_read_the_child_live():
    connection = _SlowConnection(_values(get_market_environment="bearish"), 0.0)
    proxy = _proxy(connection)
    result = {}
    worker = threading.Thread(target=lambda: result.update(env=proxy.get_market_environment()))
    worker.start()
    worker.join(5.0)
    assert result == {"env": "bearish"}
    assert connection.requests == ["get_market_environment"]
    # ...and that answer now serves the GUI thread too.
    assert proxy.get_market_environment() == "bearish"


def test_status_call_with_arguments_goes_to_the_child():
    connection = _SlowConnection(_values(), 0.0)
    proxy = _proxy(connection)
    assert proxy.get_market_environment("x") == "bullish"
    assert connection.requests == ["get_market_environment"]


@pytest.mark.timeout(60)
def test_real_child_status_poller_fills_the_cache_and_stops_with_the_proxy():
    from ui.services.bounce_process import BounceProcessProxy

    proxy = BounceProcessProxy(lambda message, tag: None, launcher_spec=f"{FIXTURE}:run_bot_with_gui")
    try:
        proxy._gui_thread = threading.current_thread()
        assert proxy.connection_status is True  # seeded from the child's hello
        assert _wait_for(lambda: _timed(proxy.get_auto_regime_reading)[0] == {"label": "mixed"}, timeout=10.0)
        assert proxy._status_thread.is_alive()
    finally:
        proxy.stop(timeout=2.0)
    assert not proxy._status_thread.is_alive()
    assert not proxy.process.is_alive()
