"""The one door to ``yfinance.download`` inside a process.

yfinance keeps every ``download`` result in the module-global
``yfinance.shared._DFS`` and resets it at the start of each call, so two
overlapping calls in one process can strand each other forever. Every
in-process caller goes through :func:`download`, which holds one lock.

`install_process_guard` (the desk calls it at start) also routes code that
calls ``yf.download`` directly through the same lock, and every guarded call
has a deadline: yfinance's wait for its worker threads raises
`YahooDownloadTimeout` past CALL_TIMEOUT_SECONDS instead of spinning forever.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
import time
from typing import Any

_LOCK = threading.RLock()
#: A guarded call that is still waiting on yfinance's threads after this long gives up.
CALL_TIMEOUT_SECONDS = 240.0
_DEADLINE = threading.local()


class YahooDownloadTimeout(RuntimeError):
    """A yfinance download outlived CALL_TIMEOUT_SECONDS; its result is abandoned."""


class _BoundedTime:
    """Stands in for ``time`` inside ``yfinance.multi``: its wait-loop sleep raises
    once the calling thread's deadline has passed. Everything else is ``time``."""

    def sleep(self, seconds: float) -> None:
        deadline = getattr(_DEADLINE, "at", None)
        if deadline is not None and time.monotonic() > deadline:
            raise YahooDownloadTimeout(
                f"yfinance download still waiting after {CALL_TIMEOUT_SECONDS:.0f}s"
            )
        time.sleep(seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _bound_multi_wait(yf_module: Any) -> None:
    """Swap ``yfinance.multi``'s ``time`` for the bounded one (once; a stub has none)."""
    multi = getattr(yf_module, "multi", None)
    if multi is not None and getattr(multi, "_time", None) is time:
        multi._time = _BoundedTime()


@contextlib.contextmanager
def _guard():
    """The process lock plus a deadline for the outermost guarded call on this thread."""
    with _LOCK:
        outer = getattr(_DEADLINE, "at", None) is None
        if outer:
            _DEADLINE.at = time.monotonic() + CALL_TIMEOUT_SECONDS
        try:
            yield
        finally:
            if outer:
                _DEADLINE.at = None


def install_process_guard(yf_module: Any = None) -> bool:
    """Route every ``yf_module.download`` call in this process through the guard,
    including callers that use ``yf.download`` directly. Idempotent; True when the
    module is guarded afterwards."""
    if yf_module is None:
        import yfinance as yf_module
    _bound_multi_wait(yf_module)
    current = getattr(yf_module, "download", None)
    if current is None:
        return False
    if getattr(current, "_tbv3_guarded", False):
        return True

    @functools.wraps(current)
    def guarded(*args: Any, **kwargs: Any):
        with _guard():
            return current(*args, **kwargs)

    guarded._tbv3_guarded = True  # type: ignore[attr-defined]
    yf_module.download = guarded
    return True


def download(*args: Any, **kwargs: Any):
    """Call ``yfinance.download`` with the same arguments, one call at a time."""
    import yfinance as yf  # looked up per call so test stubs of the module apply

    _bound_multi_wait(yf)
    with _guard():
        return yf.download(*args, **kwargs)


def download_frames(tickers, *, yf_module=None, **kwargs) -> tuple[dict[str, Any], dict[str, str]]:
    """One batched ``download``; returns each ticker's own frame and error, read under the lock."""
    if yf_module is None:
        import yfinance as yf_module

    multi = getattr(yf_module, "multi", None)
    realign = getattr(multi, "_realign_dfs", None)
    realigned: list[bool] = []

    def _flag_realign(*args, **kw):
        realigned.append(True)
        return realign(*args, **kw)

    _bound_multi_wait(yf_module)
    with _guard():
        if realign is not None:
            multi._realign_dfs = _flag_realign  # yfinance rewrites every frame when its concat fails
        try:
            yf_module.download(list(tickers), **kwargs)
        finally:
            if realign is not None:
                multi._realign_dfs = realign
        shared = yf_module.shared
        frames = dict(shared._DFS)
        errors = dict(shared._ERRORS)
    if realigned:
        # Realigned frames are reindexed and de-duplicated, so they may differ from a
        # per-symbol download: use none of them.
        logging.warning(
            "yfinance realigned a %d-ticker batch; its frames are discarded and fetched one by one.",
            len(frames),
        )
        return {}, errors
    return frames, errors
