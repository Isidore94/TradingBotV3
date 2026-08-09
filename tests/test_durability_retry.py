"""Bounded retry before a recovery fetch's gap becomes permanent.

The durability packet's two recovery paths -- the technical-integrity follow-up
sweep and the breadth-ledger backfill -- wrote an explicit ``data_gap`` row and
set a per-session marker after a *single* failed historical request. The marker
is what makes it permanent: it stops the session from ever being swept again.
So one pacing violation, one momentary disconnect, one empty response while the
provider was still catching up, cost that session's research evidence for good
(checkpoint review 2026-08-08 second review).

What gets written after the retries are exhausted is deliberately unchanged.
Missing data is still uncertainty, still explicit, still never invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from durability_retry import (  # noqa: E402
    fetch_with_bounded_retry,
    retry_deadline,
)


class _Sleeps:
    """Records backoff instead of spending it."""

    def __init__(self):
        self.pauses: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.pauses.append(seconds)


def test_first_attempt_success_costs_nothing_extra():
    sleeps = _Sleeps()
    outcome = fetch_with_bounded_retry(lambda: ["bar"], label="t", sleep=sleeps)

    assert outcome.value == ["bar"]
    assert outcome.attempts == 1
    assert outcome.ok and not outcome.exhausted
    assert sleeps.pauses == []


def test_transient_exception_then_success_is_recovered():
    sleeps = _Sleeps()
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("pacing violation")
        return ["bar"]

    outcome = fetch_with_bounded_retry(_fetch, label="t", sleep=sleeps)

    assert outcome.value == ["bar"]
    assert outcome.attempts == 2
    assert not outcome.exhausted
    assert outcome.error == ""
    assert len(sleeps.pauses) == 1


def test_transient_empty_response_then_success_is_recovered():
    # A provider that answers "nothing" while it is still catching up is the
    # same class of failure as one that raises -- and used to be indistinguishable
    # from "this data does not exist".
    sleeps = _Sleeps()
    responses = [[], [], ["bar"]]
    outcome = fetch_with_bounded_retry(
        lambda: responses.pop(0), label="t", sleep=sleeps
    )

    assert outcome.value == ["bar"]
    assert outcome.attempts == 3
    assert not outcome.exhausted


def test_exhaustion_reports_the_last_error_and_stops_at_the_bound():
    sleeps = _Sleeps()
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        raise RuntimeError("IB disconnected")

    outcome = fetch_with_bounded_retry(_fetch, label="t", retries=2, sleep=sleeps)

    assert outcome.exhausted and not outcome.ok
    assert outcome.value is None
    assert calls["n"] == 3, "retries must be bounded, never an open loop"
    assert outcome.attempts == 3
    assert "IB disconnected" in outcome.error
    assert len(sleeps.pauses) == 2, "no sleep after the final attempt"


def test_persistent_emptiness_still_exhausts_without_an_error_string():
    outcome = fetch_with_bounded_retry(lambda: [], label="t", sleep=_Sleeps())

    assert outcome.exhausted
    assert outcome.error == ""
    assert "no data returned" in outcome.describe()


def test_empty_is_failure_off_accepts_an_empty_first_answer():
    sleeps = _Sleeps()
    outcome = fetch_with_bounded_retry(
        lambda: [], label="t", empty_is_failure=False, sleep=sleeps
    )

    assert not outcome.exhausted
    assert outcome.attempts == 1
    assert sleeps.pauses == []


def test_a_spent_shared_budget_stops_the_sleeping_not_the_attempt():
    # A provider that is down for every symbol must not stall the caller's
    # thread (and, for the sweeper, its lock) for symbol-count x backoff.
    sleeps = _Sleeps()
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        raise RuntimeError("down")

    outcome = fetch_with_bounded_retry(
        _fetch,
        label="t",
        deadline=retry_deadline(0.0, monotonic=lambda: 100.0),
        monotonic=lambda: 100.0,
        sleep=sleeps,
    )

    assert calls["n"] == 1, "budget spent: one attempt, no retries"
    assert outcome.exhausted
    assert sleeps.pauses == []


def test_the_helper_never_raises_out_of_a_broken_fetch():
    # A failed recovery must leave the ledger as it found it, so the fetch's
    # exception becomes an exhausted outcome rather than a crash upstream.
    outcome = fetch_with_bounded_retry(
        lambda: (_ for _ in ()).throw(ValueError("bad rows")),
        label="t",
        sleep=_Sleeps(),
    )
    assert outcome.exhausted and "bad rows" in outcome.error


# --- completeness (Sol 5.6 verification review, item 6) -------------------


def test_an_incomplete_response_is_retried_like_a_failure():
    sleeps = _Sleeps()
    responses = [["a", "b"], ["a", "b", "c", "d"]]

    outcome = fetch_with_bounded_retry(
        lambda: responses.pop(0),
        label="t",
        is_complete=lambda rows: len(rows) >= 4,
        sleep=sleeps,
    )

    assert outcome.attempts == 2
    assert not outcome.exhausted
    assert outcome.value == ["a", "b", "c", "d"]


def test_a_complete_response_is_accepted_immediately():
    sleeps = _Sleeps()
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        return ["a", "b", "c", "d"]

    outcome = fetch_with_bounded_retry(
        _fetch, label="t", is_complete=lambda rows: len(rows) >= 4, sleep=sleeps
    )

    assert calls["n"] == 1
    assert sleeps.pauses == []
    assert not outcome.exhausted


def test_a_persistently_incomplete_response_exhausts_but_keeps_what_it_got():
    # A caller that would rather record a short window than nothing must still
    # be able to see the partial data, and must know it is partial.
    outcome = fetch_with_bounded_retry(
        lambda: ["a", "b"],
        label="t",
        is_complete=lambda rows: len(rows) >= 4,
        sleep=_Sleeps(),
    )

    assert outcome.exhausted
    assert outcome.value == ["a", "b"]
    assert outcome.error == "incomplete response"
