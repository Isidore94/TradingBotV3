"""Provider request / cache-hit / throttle / failure counters (plan.md sec 6.3).

This is the capture point that did not exist anywhere in the repo: the Health
page's "Provider request, cache-hit, throttling, and failure counts" dimension
emitted a permanent UNKNOWN because nothing recorded these facts.

Design constraints, in order of importance:

* **Counting only.**  Nothing here may change fetch ordering, retry behaviour,
  caching policy, or any provider response.  The recorder is fire-and-forget
  and swallows its own errors so an accounting bug can never break a scan.
* **Per-run scope.**  Counters are reset at the start of a master scan and
  flushed into that scan's run manifest, so each manifest carries the counts
  for exactly its own run - the shape trend reporting and the Milestone 1
  benchmark dimensions need.  (The in-process totals are also readable at any
  time for the Health page's live view.)
* **Honest keys.**  ``throttle`` is only ever recorded from a real pacing-class
  signal (IBKR error 162/366 or a pacing/rate-limit message), never inferred.
  A zero therefore means "no throttling was observed", not "not measured" -
  "not measured" is the absence of the counters block entirely.
"""

from __future__ import annotations

import threading
from collections import Counter

OUTCOME_REQUEST = "request"
OUTCOME_CACHE_HIT = "cache_hit"
OUTCOME_FAILURE = "failure"
OUTCOME_THROTTLE = "throttle"
OUTCOMES = (OUTCOME_REQUEST, OUTCOME_CACHE_HIT, OUTCOME_FAILURE, OUTCOME_THROTTLE)

#: Prefix for run-manifest counter keys, e.g. ``provider.daily_bars.request``.
MANIFEST_PREFIX = "provider"

_lock = threading.Lock()
_counts: Counter[str] = Counter()


def record(endpoint: str, outcome: str, n: int = 1) -> None:
    """Count one provider-boundary event.  Never raises."""
    try:
        key = f"{str(endpoint).strip()}.{str(outcome).strip()}"
        with _lock:
            _counts[key] += int(n)
    except Exception:
        pass


def snapshot() -> dict[str, int]:
    """Current per-run counts as a plain dict (endpoint.outcome -> count)."""
    with _lock:
        return {key: int(value) for key, value in sorted(_counts.items())}


def totals() -> dict[str, int]:
    """Counts rolled up per outcome across every endpoint."""
    rollup: Counter[str] = Counter()
    for key, value in snapshot().items():
        outcome = key.rsplit(".", 1)[-1]
        rollup[outcome] += value
    return {outcome: int(rollup.get(outcome, 0)) for outcome in OUTCOMES}


def reset() -> None:
    """Start a fresh per-run scope (called at the top of a master scan)."""
    with _lock:
        _counts.clear()


def flush_to_manifest(recorder=None) -> dict[str, int]:
    """Write the current counts into the active run manifest.

    Returns the snapshot that was flushed.  Missing recorder or a recorder
    error is non-fatal: the counts stay readable in-process either way.
    """
    counts = snapshot()
    try:
        if recorder is None:
            from diagnostics import get_active_recorder

            recorder = get_active_recorder()
        if recorder is not None:
            for key, value in counts.items():
                recorder.set_counter(f"{MANIFEST_PREFIX}.{key}", value)
            # Always stamp presence, even for an all-zero run: the audit needs
            # "measured and zero" to be distinguishable from "not measured".
            recorder.set_counter(f"{MANIFEST_PREFIX}.captured", 1)
    except Exception:
        pass
    return counts
