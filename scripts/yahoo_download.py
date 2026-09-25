"""The one door to ``yfinance.download`` inside a process.

yfinance keeps every ``download`` result in the module-global
``yfinance.shared._DFS`` and resets it at the start of each call, so two
overlapping calls in one process can strand each other forever. Every
in-process caller goes through :func:`download`, which holds one lock.
"""

from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()


def download(*args: Any, **kwargs: Any):
    """Call ``yfinance.download`` with the same arguments, one call at a time."""
    import yfinance as yf  # looked up per call so test stubs of the module apply

    with _LOCK:
        return yf.download(*args, **kwargs)


def download_frames(tickers, *, yf_module=None, **kwargs) -> tuple[dict[str, Any], dict[str, str]]:
    """One batched ``download``; returns each ticker's own frame and error, read under the lock."""
    if yf_module is None:
        import yfinance as yf_module

    with _LOCK:
        yf_module.download(list(tickers), **kwargs)
        shared = yf_module.shared
        frames = dict(shared._DFS)
        errors = dict(shared._ERRORS)
    return frames, errors
