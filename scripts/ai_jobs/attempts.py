"""Verified retries inside one night slot.

Ask the model, run the slot's own deterministic verifier, and on a rejection ask
again with the rejection folded into the evidence. The first reply the verifier
passes wins; there is no model judge. The retry loop is pure; the one clock seam
is `night_window_gate`, which a slot passes as `may_retry`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

#: Model calls one slot may spend on one answer before it keeps the last good file.
VERIFIED_ATTEMPTS = 3

T = TypeVar("T")


class AttemptsRejected(ValueError):
    """Every attempt was rejected (or the retries were stopped); one reason per attempt."""

    def __init__(self, reasons: Sequence[str], *, attempts: int, stopped: str = "") -> None:
        self.reasons = tuple(reasons)
        self.attempts = int(attempts)
        self.stopped = str(stopped or "")
        super().__init__(describe_rejections(self.reasons, self.attempts, stopped=self.stopped))


@dataclass(frozen=True)
class Verified(Generic[T]):
    """The reply that passed, what the verifier made of it, and what came before."""

    result: Mapping[str, Any]
    value: T
    attempt: int
    attempts: int
    rejections: tuple[str, ...]
    evidence: Mapping[str, Any]

    @property
    def label(self) -> str:
        return f"attempt {self.attempt}/{self.attempts}"


def describe_rejections(reasons: Sequence[str], attempts: int, *, stopped: str = "") -> str:
    """`attempt 1/3: why; attempt 2/3: why` (+ why the next attempt was not made)."""
    parts = [f"attempt {index}/{attempts}: {reason}" for index, reason in enumerate(reasons, 1)]
    if stopped:
        parts.append(f"no attempt {len(reasons) + 1}/{attempts} ({stopped})")
    return "; ".join(parts)


#: A fed-back reason is cut to this many characters so a retry's prompt stays bounded.
FEEDBACK_REASON_CHARS = 300


def feedback_line(reason: str) -> str:
    """One instruction sentence quoting why an earlier reply was rejected."""
    text = str(reason)[:FEEDBACK_REASON_CHARS]
    return f' An earlier reply was rejected for this reason. Do not repeat it: "{text}"'


def instructions_feedback(evidence: Mapping[str, Any], reasons: Sequence[str]) -> dict[str, Any]:
    """The evidence with every rejection reason appended to its `instructions`."""
    out = dict(evidence)
    out["instructions"] = str(out.get("instructions") or "") + "".join(
        feedback_line(reason) for reason in reasons
    )
    return out


def night_window_gate(
    now: datetime | None, *, reserve_minutes: float
) -> Callable[[], tuple[bool, str]]:
    """`may_retry` that asks the night window, on a clock starting at `now` and moving.

    A retry is another model call, and the slot's reserve bought only the first.
    """
    from ai_jobs import window

    start = now or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    began = time.monotonic()

    def _gate() -> tuple[bool, str]:
        moment = start + timedelta(seconds=time.monotonic() - began)
        try:
            return window.launch_allowed(moment, reserve_minutes=reserve_minutes)
        except Exception as exc:  # noqa: BLE001 - an unanswerable window stops the retries
            return False, f"the night window could not be read: {exc}"

    return _gate


def verified_attempts(
    request: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    validate: Callable[[Mapping[str, Any]], T],
    *,
    evidence: Mapping[str, Any],
    with_feedback: Callable[[Mapping[str, Any], Sequence[str]], Mapping[str, Any]] = instructions_feedback,
    attempts: int = VERIFIED_ATTEMPTS,
    may_retry: Callable[[], tuple[bool, str]] | None = None,
) -> Verified[T]:
    """The first reply `validate` accepts, asking at most `attempts` times.

    `request(evidence)` is one model call; its errors propagate (a model that did
    not answer is not a rejection). `validate(result)` returns the kept value or
    raises; each raise is one rejection, fed back through `with_feedback` for the
    next call. `may_retry()` is asked before every call after the first. Raises
    `AttemptsRejected` naming every reason when no reply passed.
    """
    total = max(1, int(attempts))
    reasons: list[str] = []
    current: Mapping[str, Any] = evidence
    for attempt in range(1, total + 1):
        if attempt > 1:
            if may_retry is not None:
                allowed, why = may_retry()
                if not allowed:
                    raise AttemptsRejected(reasons, attempts=total, stopped=why or "retry not allowed")
            current = with_feedback(evidence, list(reasons))
        result = request(current)
        try:
            value = validate(result)
        except Exception as exc:  # noqa: BLE001 - any verifier raise is a rejection
            reasons.append(str(exc) or type(exc).__name__)
            continue
        return Verified(
            result=result,
            value=value,
            attempt=attempt,
            attempts=total,
            rejections=tuple(reasons),
            evidence=current,
        )
    raise AttemptsRejected(reasons, attempts=total)


__all__ = [
    "VERIFIED_ATTEMPTS",
    "FEEDBACK_REASON_CHARS",
    "AttemptsRejected",
    "Verified",
    "describe_rejections",
    "feedback_line",
    "instructions_feedback",
    "night_window_gate",
    "verified_attempts",
]
