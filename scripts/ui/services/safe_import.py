from __future__ import annotations

"""One-at-a-time first import of the heavy engine modules.

``master_avwap_lib/__init__.py`` imports its own ``legacy`` submodule, so the
package is only usable once that import has run to completion. While the
chart path was synchronous this never mattered: everything imported it from
the GUI thread, in order.

Now the snapshot build runs on a pool worker while the D1 backfill and
forming-candle fetches run on their own threads, and two of them racing to be
the FIRST importer makes one observe a partially initialized package:

    ImportError: cannot import name 'legacy' from partially initialized
    module 'master_avwap_lib' (most likely due to a circular import)

Python's per-module import lock does not save a circular import from this.
Every off-GUI-thread importer on the chart path goes through here instead, so
the first import is serialized and the rest are cache hits.
"""

import threading

_LOCK = threading.RLock()


def master_avwap_legacy():
    """``master_avwap_lib.legacy``, imported under the shared lock."""
    with _LOCK:
        from master_avwap_lib import legacy

        return legacy


def warm() -> None:
    """Force the heavy imports now, so later concurrent users cannot race.

    Called at the top of every worker task on the chart path. After the first
    call this is a dict lookup.
    """
    with _LOCK:
        try:
            import chart_snapshot  # noqa: F401
            from master_avwap_lib import legacy  # noqa: F401
        except Exception:
            # A failed warm is not fatal here - the real call site will raise
            # (and log) with the context that actually matters.
            pass
