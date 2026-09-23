"""GUI-side cache of the scanner child's M5 chart bars.

On the process proxy every `bot.m5_chart_bars` call is a blocking RPC to the
BounceBot child. Timer polls, the armed list, the snapshot chart and the arm
path asked for bars on the Qt thread, and a busy child stalled the desk for up
to 38 s at a time (2026-09-23 stall log).

This cache is the only thing the Qt thread reads for a proxy bot:

* ``peek`` is memory only. ``None`` means "never fetched" - UNKNOWN, which a
  caller must treat as unknown, never as "no bars" or "confirmed".
* One worker thread (this object's) refreshes every key asked for in the last
  ``WANT_TTL_SECONDS``, at most once per ``REFRESH_SECONDS``, one RPC per
  (symbol, sessions). Never-fetched keys go first.
* The value is always `m5_chart_bars`' own output, so a poll given the same
  bars decides exactly what it decided before.

An in-process bot is not served here: its `m5_chart_bars` is a memory read and
callers keep reading it directly.
"""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, Signal

#: A wanted key is refetched at most this often.
REFRESH_SECONDS = 15.0
#: A key nobody has asked for in this long stops being refreshed.
WANT_TTL_SECONDS = 300.0
#: How long the idle worker sleeps between checks.
IDLE_WAIT_SECONDS = 1.0


def is_process_proxy(bot) -> bool:
    return bool(getattr(bot, "is_process_proxy", False))


def _stamp(bars) -> tuple:
    if not bars:
        return (0, None)
    last = bars[-1]
    stamp = last.get("dt") if isinstance(last, dict) else getattr(last, "dt", None)
    return (len(bars), stamp)


class M5BarCache(QObject):
    """One owner of the proxy M5 bar cache and its refresh thread."""

    #: (symbol) - bars for this symbol were fetched for the first time or changed.
    barsUpdated = Signal(str)

    def __init__(self, parent=None, *, refresh_seconds: float = REFRESH_SECONDS) -> None:
        super().__init__(parent)
        self.refresh_seconds = float(refresh_seconds)
        self._lock = threading.Lock()
        self._bot = None
        # (symbol, sessions) -> (bars, fetched_at monotonic)
        self._entries: dict[tuple[str, int], tuple[list, float]] = {}
        # (symbol, sessions) -> last asked monotonic
        self._wanted: dict[tuple[str, int], float] = {}
        # Small whole-bot values (e.g. `d1_zone_arms`): name -> (value, fetched_at)
        self._values: dict[tuple[str, bool], tuple[object, float]] = {}
        self._wanted_values: dict[tuple[str, bool], float] = {}
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self.fetches = 0

    @staticmethod
    def _key(symbol: str, sessions: int) -> tuple[str, int]:
        return (str(symbol or "").strip().upper(), max(1, int(sessions)))

    def _adopt_bot(self, bot) -> None:
        """Caller holds the lock. A new child process starts an empty cache."""
        if bot is not self._bot:
            self._bot = bot
            self._entries.clear()
            self._wanted.clear()
            self._values.clear()
            self._wanted_values.clear()

    # -- Qt-thread half: memory only -----------------------------------------
    def peek(self, bot, symbol: str, sessions: int = 1) -> list | None:
        """Cached bars, or None when never fetched. Marks the key wanted."""
        if bot is None:
            return None
        key = self._key(symbol, sessions)
        if not key[0]:
            return None
        now = time.monotonic()
        with self._lock:
            self._adopt_bot(bot)
            self._wanted[key] = now
            entry = self._entries.get(key)
            due = entry is None or now - entry[1] >= self.refresh_seconds
        if due:
            self._ensure_worker()
            self._wake.set()
        return None if entry is None else entry[0]

    def peek_value(self, bot, name: str, *, call: bool = False):
        """A cached bot attribute (or zero-arg method result), or None if unknown."""
        if bot is None:
            return None
        key = (str(name), bool(call))
        now = time.monotonic()
        with self._lock:
            self._adopt_bot(bot)
            self._wanted_values[key] = now
            entry = self._values.get(key)
            due = entry is None or now - entry[1] >= self.refresh_seconds
        if due:
            self._ensure_worker()
            self._wake.set()
        return None if entry is None else entry[0]

    def is_known(self, bot, symbol: str, sessions: int = 1) -> bool:
        return self.peek(bot, symbol, sessions) is not None

    # -- any non-Qt thread ---------------------------------------------------
    def fetch_now(self, bot, symbol: str, sessions: int = 1) -> list:
        """RPC now and store the answer. Never call this on the Qt thread."""
        key = self._key(symbol, sessions)
        bars = list(bot.m5_chart_bars(key[0], max_sessions=key[1]) or [])
        self._store(bot, key, bars)
        return bars

    def _store(self, bot, key, bars: list) -> None:
        with self._lock:
            if bot is not self._bot:
                if self._bot is not None:
                    return  # an answer from a retired child
                self._bot = bot
            previous = self._entries.get(key)
            self._entries[key] = (bars, time.monotonic())
            self._wanted.setdefault(key, time.monotonic())
        self.fetches += 1
        if previous is None or _stamp(previous[0]) != _stamp(bars):
            try:
                self.barsUpdated.emit(key[0])
            except RuntimeError:
                pass  # the QObject is gone at shutdown

    # -- the worker ----------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._closing.clear()
            self._thread = threading.Thread(
                target=self._run, name="m5-bar-cache", daemon=True
            )
            self._thread.start()

    def _due_keys(self) -> tuple[object, list[tuple[str, int]]]:
        now = time.monotonic()
        with self._lock:
            bot = self._bot
            for key in [k for k, asked in self._wanted.items() if now - asked > WANT_TTL_SECONDS]:
                self._wanted.pop(key, None)
            fresh, stale = [], []
            for key in self._wanted:
                entry = self._entries.get(key)
                if entry is None:
                    fresh.append(key)
                elif now - entry[1] >= self.refresh_seconds:
                    stale.append((entry[1], key))
        stale.sort()
        return bot, fresh + [key for _at, key in stale]

    def _refresh_values(self, bot) -> None:
        now = time.monotonic()
        with self._lock:
            for key in [k for k, asked in self._wanted_values.items() if now - asked > WANT_TTL_SECONDS]:
                self._wanted_values.pop(key, None)
            due = [
                key
                for key in self._wanted_values
                if key not in self._values or now - self._values[key][1] >= self.refresh_seconds
            ]
        for name, call in due:
            try:
                value = getattr(bot, name)
                if call:
                    value = value()
            except Exception:
                logging.debug("Bot value %s could not be read.", name, exc_info=True)
                continue
            with self._lock:
                if bot is self._bot:
                    self._values[(name, call)] = (value, time.monotonic())

    def _run(self) -> None:
        while not self._closing.is_set():
            bot, keys = self._due_keys()
            if bot is not None:
                self._refresh_values(bot)
            if bot is None or not keys:
                self._wake.wait(IDLE_WAIT_SECONDS)
                self._wake.clear()
                continue
            for key in keys:
                if self._closing.is_set():
                    return
                try:
                    bars = list(bot.m5_chart_bars(key[0], max_sessions=key[1]) or [])
                except Exception:
                    # Unknown stays unknown; a known entry keeps its last answer
                    # and is retried after the refresh period.
                    logging.debug("M5 cache fetch failed for %s.", key[0], exc_info=True)
                    with self._lock:
                        entry = self._entries.get(key)
                        if entry is not None:
                            self._entries[key] = (entry[0], time.monotonic())
                    continue
                self._store(bot, key, bars)
            # Nothing is due again for at least the refresh period.
            self._wake.wait(IDLE_WAIT_SECONDS)
            self._wake.clear()

    def wait_known(self, bot, symbol: str, sessions: int = 1, timeout: float = 5.0) -> bool:
        """Test helper: block until the key has a fetched answer."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.peek(bot, symbol, sessions) is not None:
                return True
            time.sleep(0.01)
        return False

    def shutdown(self, timeout: float = 1.0) -> None:
        self._closing.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
        with self._lock:
            self._entries.clear()
            self._wanted.clear()
            self._values.clear()
            self._wanted_values.clear()
            self._bot = None


_SHARED: M5BarCache | None = None


def shared_m5_cache() -> M5BarCache:
    """The one cache, shared by the Alert Center, its charts and the snapshot dialog."""
    global _SHARED
    if _SHARED is None:
        _SHARED = M5BarCache()
    return _SHARED


def reset_shared_m5_cache() -> None:
    """Test seam: drop the singleton."""
    global _SHARED
    if _SHARED is not None:
        _SHARED.shutdown()
    _SHARED = None
