"""Bounded retry for the durability packet's recovery fetches.

Recovery paths (docs/DURABILITY_CATCHUP_PLAN.md Tier B) re-request history the
live process failed to capture. When that request fails they must write an
explicit, permanent ``data_gap`` marker -- missing data is uncertainty, never
confirmation, and it is never quietly filled in.

The defect this module fixes is *when* that marker becomes permanent. Both the
technical-integrity follow-up sweep and the breadth-ledger backfill wrote the
gap after a **single** failed request, and then set a per-session marker that
stops the sweep from ever running again for that session. So one transient
broker hiccup -- a pacing violation, a momentary disconnect, an empty response
while the provider is still catching up -- turned into permanently missing
research evidence that no later run would retry (checkpoint review 2026-08-08
second review).

A bounded retry closes that gap without weakening the honesty rule: a few extra
requests, a short backoff, both logged, and only after exhaustion is the gap
written. What is recorded afterwards is unchanged.

Two properties keep this safe to call from a live thread:

* the retry count is small and fixed, so a broken provider costs a bounded
  number of extra requests, never an open-ended loop;
* callers sweeping many symbols share one wall-clock ``deadline``. Once it
  passes, remaining symbols still get their one attempt but no further sleeps,
  so a provider that is down for everything cannot stall the caller's thread
  (and, for the TI sweeper, its lock) for symbol-count x backoff seconds.

This module is pure infrastructure: it holds no engine state, makes no trading
decision, and changes no detector, scoring or alert behaviour.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence


#: Two retries after the first attempt: enough to ride out a hiccup, few enough
#: that an outage costs three requests per symbol rather than an open loop.
DEFAULT_RETRIES = 2

#: Deliberately short. These run on the recorder's own thread, sometimes while
#: a lock is held, so the backoff is sized to survive a momentary disconnect --
#: not to wait out a real outage, which is what the data_gap row is for.
DEFAULT_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 1.5)

#: Whole-sweep sleep budget shared across symbols (see module docstring).
DEFAULT_RETRY_BUDGET_SECONDS = 30.0


@dataclass(frozen=True)
class RetryOutcome:
    """What a bounded-retry fetch produced, and how hard it had to try."""

    value: Any
    #: Total attempts made, including the first. 1 means it worked first time.
    attempts: int
    #: True when every attempt failed, i.e. the caller may now write the gap.
    exhausted: bool
    #: repr of the last exception, or "" when the failures were empty results.
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.exhausted

    def describe(self) -> str:
        if not self.exhausted:
            return (
                "succeeded first attempt"
                if self.attempts <= 1
                else f"succeeded on attempt {self.attempts}"
            )
        suffix = f": {self.error}" if self.error else " (no data returned)"
        return f"exhausted {self.attempts} attempt(s){suffix}"


def retry_deadline(
    budget_seconds: float = DEFAULT_RETRY_BUDGET_SECONDS,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> float:
    """Wall-clock point after which shared-budget callers stop sleeping."""
    return monotonic() + max(0.0, float(budget_seconds))


def fetch_with_bounded_retry(
    fetch: Callable[[], Any],
    *,
    label: str,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: Sequence[float] = DEFAULT_BACKOFF_SECONDS,
    empty_is_failure: bool = True,
    deadline: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> RetryOutcome:
    """Call ``fetch`` until it yields data, up to ``retries`` extra attempts.

    An exception and (when ``empty_is_failure``) a falsy result are both
    treated as transient: a provider that answers "nothing" while it is still
    catching up is the same class of failure as one that raises, and both used
    to finalise a permanent gap on the spot.

    Never raises. The exhausted outcome carries the last error so the caller
    can put it in the gap row's reason, exactly as it did before.
    """
    attempts = 0
    last_error = ""
    total = max(0, int(retries)) + 1
    for index in range(total):
        attempts = index + 1
        try:
            value = fetch()
        except Exception as exc:  # a broker hiccup is a gap, not a crash
            last_error = repr(exc)
            value = None
        else:
            if value or not empty_is_failure:
                if index:
                    logging.info("%s recovered on attempt %d.", label, attempts)
                return RetryOutcome(value=value, attempts=attempts, exhausted=False)
            last_error = ""

        if index >= total - 1:
            break
        if deadline is not None and monotonic() >= deadline:
            logging.warning(
                "%s failed (%s); shared retry budget is spent, so this attempt stands.",
                label,
                last_error or "no data returned",
            )
            break
        pause = float(
            backoff_seconds[index] if index < len(backoff_seconds) else backoff_seconds[-1]
        ) if backoff_seconds else 0.0
        logging.warning(
            "%s failed on attempt %d/%d (%s); retrying in %.1fs.",
            label,
            attempts,
            total,
            last_error or "no data returned",
            pause,
        )
        if pause > 0:
            sleep(pause)

    logging.warning(
        "%s exhausted after %d attempt(s) (%s); recording an explicit data gap.",
        label,
        attempts,
        last_error or "no data returned",
    )
    return RetryOutcome(value=None, attempts=attempts, exhausted=True, error=last_error)
