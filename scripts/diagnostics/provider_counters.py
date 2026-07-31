"""Provider telemetry v2: honest boundary accounting (plan.md sec 6.3).

v1 stamped ``provider.captured=1`` unconditionally, counted one wrapper-level
"request" that could hide an IBKR attempt *plus* a Yahoo fallback attempt, and
let the audit grade an empty block HEALTHY.  v2 fixes each of those:

* **Distinct concepts.**  A *logical lookup* (wrapper call) is not an
  *outbound attempt* (a real IBKR/Yahoo/Nasdaq call), and failure ratios are
  only meaningful per (family, provider) against matching attempt
  denominators.  Key grammar, all under one family:

  - ``<family>.lookup``            logical wrapper call
  - ``<family>.cache_hit``         served without any outbound attempt
  - ``<family>.attempt.<provider>``  one real outbound call
  - ``<family>.success.<provider>``  provider returned a usable response
  - ``<family>.failure.<provider>``  provider failed / returned nothing usable
  - ``<family>.throttle.<provider>`` pacing-class signal only, never inferred
  - ``<family>.fallback_used``     a later provider served after an earlier one failed
  - ``<family>.refresh_unusable``  wrapper-level: live refresh yielded nothing
    usable (excluded from ratios; a provider "success" can still be stale)

* **Completeness is declared, not assumed.**  :data:`FAMILIES_EXPECTED` is the
  inventory of provider boundaries reachable from ``run_master``;
  :data:`FAMILIES_INSTRUMENTED` is what this build actually counts.  Both are
  flushed with the manifest so the audit can prove "all required boundaries
  were active" before calling an all-zero run healthy - and must report
  PARTIAL, never HEALTHY, when they differ.

* **Capture failures are observable.**  The recording helper stays non-fatal
  to scans, but every swallowed error increments ``capture_errors``, which is
  flushed with the run.  Stamping the schema over a silent capture failure is
  therefore impossible.

* **Run isolation.**  Counts live in a per-run bucket opened by
  :func:`begin_run`.  After a flush closes the bucket, a late worker's records
  land in a visible *orphan* bucket that is reported on the next flush -
  never merged into the next run's counts.
"""

from __future__ import annotations

import threading
from collections import Counter

SCHEMA_VERSION = 2

OUTCOME_LOOKUP = "lookup"
OUTCOME_CACHE_HIT = "cache_hit"
OUTCOME_ATTEMPT = "attempt"
OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_THROTTLE = "throttle"
OUTCOME_FALLBACK = "fallback_used"
OUTCOME_UNUSABLE = "refresh_unusable"

#: Provider-qualified outcomes (``family.outcome.provider``).
PROVIDER_QUALIFIED = (OUTCOME_ATTEMPT, OUTCOME_SUCCESS, OUTCOME_FAILURE, OUTCOME_THROTTLE)

#: Every provider/cache boundary reachable from run_master (the inventory).
#: deep_daily_levels rides the daily_bars boundary and is counted there.
FAMILIES_EXPECTED = (
    "daily_bars",
    "intraday_bars",
    "symbol_metadata",
    "earnings_dates",
    "earnings_calendar",
    "theta_options",
)
#: What this build actually instruments.  The audit treats any gap between
#: this and FAMILIES_EXPECTED as PARTIAL coverage - never healthy.
FAMILIES_INSTRUMENTED = (
    "daily_bars",
    "intraday_bars",
    "symbol_metadata",
    "earnings_dates",
    "earnings_calendar",
    "theta_options",
)

MANIFEST_PREFIX = "provider"

_lock = threading.Lock()
_run_counts: Counter[str] = Counter()
_orphan_counts: Counter[str] = Counter()
_run_open = False
_run_generation = 0
_capture_errors = 0
_orphan_events_last_run = 0


def begin_run() -> int:
    """Open a fresh per-run bucket; returns the run generation token."""
    global _run_open, _run_generation, _capture_errors
    with _lock:
        _run_counts.clear()
        _run_open = True
        _run_generation += 1
        _capture_errors = 0
        return _run_generation


def record(family: str, outcome: str, provider: str | None = None, n: int = 1) -> None:
    """Count one boundary event.  Non-fatal, but never silently lossy:
    any internal failure increments ``capture_errors``."""
    global _capture_errors
    try:
        parts = [str(family).strip(), str(outcome).strip()]
        if provider is not None:
            parts.append(str(provider).strip())
        key = ".".join(parts)
        if not all(parts):
            raise ValueError(f"malformed provider counter key: {key!r}")
        increment = int(n)
        with _lock:
            if _run_open:
                _run_counts[key] += increment
            else:
                # A worker that outlived its run: visible, never contaminating.
                _orphan_counts[key] += increment
    except Exception:
        try:
            with _lock:
                _capture_errors += 1
        except Exception:
            pass


def note_capture_error() -> None:
    """Record that an instrumentation site itself failed."""
    global _capture_errors
    with _lock:
        _capture_errors += 1


def snapshot() -> dict[str, int]:
    with _lock:
        return {key: int(value) for key, value in sorted(_run_counts.items())}


def orphan_snapshot() -> dict[str, int]:
    with _lock:
        return {key: int(value) for key, value in sorted(_orphan_counts.items())}


def reset() -> None:
    """Test hook: drop all state, including orphans."""
    global _run_open, _capture_errors, _orphan_events_last_run
    with _lock:
        _run_counts.clear()
        _orphan_counts.clear()
        _run_open = False
        _capture_errors = 0
        _orphan_events_last_run = 0


def flush_to_manifest(recorder=None) -> dict[str, int]:
    """Flush the active run's counts into the manifest and CLOSE the bucket.

    Writes, besides the per-key counts: the schema version, the declared
    expected/instrumented family lists (via manifest outputs), the number of
    capture errors, and the number of orphan events observed since the last
    flush.  Late records arriving after this call go to the orphan bucket and
    surface on the *next* flush - they can never contaminate the next run.
    """
    global _run_open, _capture_errors
    with _lock:
        counts = {key: int(value) for key, value in sorted(_run_counts.items())}
        orphans = sum(_orphan_counts.values())
        _orphan_counts.clear()
        capture_errors = _capture_errors
        _run_open = False
        _run_counts.clear()
    try:
        if recorder is None:
            from diagnostics import get_active_recorder

            recorder = get_active_recorder()
        if recorder is not None:
            for key, value in counts.items():
                recorder.set_counter(f"{MANIFEST_PREFIX}.{key}", value)
            recorder.set_counter(f"{MANIFEST_PREFIX}.schema_version", SCHEMA_VERSION)
            recorder.set_counter(f"{MANIFEST_PREFIX}.capture_errors", capture_errors)
            recorder.set_counter(f"{MANIFEST_PREFIX}.orphan_events", orphans)
            outputs = getattr(recorder, "outputs", None)
            if isinstance(outputs, dict):
                outputs["provider_families_expected"] = ",".join(FAMILIES_EXPECTED)
                outputs["provider_families_instrumented"] = ",".join(FAMILIES_INSTRUMENTED)
    except Exception:
        # The flush itself failing must not kill the scan, but it must not be
        # silent either: the in-process state still shows what happened.
        note_capture_error()
    return counts
