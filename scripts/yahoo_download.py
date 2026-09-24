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
