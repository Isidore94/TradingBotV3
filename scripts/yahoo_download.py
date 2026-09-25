"""The one door to ``yfinance.download`` inside a process.

yfinance keeps every ``download`` result in the module-global
``yfinance.shared._DFS`` and resets it at the start of each call, so two
overlapping calls in one process can strand each other forever. Every
in-process caller goes through :func:`download`, which holds one lock.
"""

from __future__ import annotations

import logging
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

    multi = getattr(yf_module, "multi", None)
    realign = getattr(multi, "_realign_dfs", None)
    realigned: list[bool] = []

    def _flag_realign(*args, **kw):
        realigned.append(True)
        return realign(*args, **kw)

    with _LOCK:
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
