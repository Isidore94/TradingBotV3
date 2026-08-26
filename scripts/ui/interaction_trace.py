"""Which click a stall belongs to. Diagnostics only - it decides nothing.

The stall watchdog answers "what frame held the GUI thread". It cannot answer
"what was the trader doing", and for a whole class of samples that is the only
useful question: a stall whose modal frame is inside Qt's own event dispatch
names no application code at all, so the log says the desk froze and nothing
about why. The 2026-08-25 capture is full of them.

This module carries the missing half. A surface calls `begin` when a trader
action starts, `mark` as it passes each stage, and `end` when the last one
lands; the watchdog stamps whatever is open onto every record it writes. An
event-loop-only sample then reads as "page select -> model_apply", which is a
lead.

Three properties it must keep:

**It never defers, skips or schedules anything.** It has no timer, no thread,
no sleep and no wait - a test asserts that at the source level, the same rule
`ScanCycleClock` carries, and for the same reason: a measuring helper that
could change when work runs would be a scheduling change wearing a diagnostic
label. Every function here returns immediately.

**The watchdog reads it from another thread, and must never block.** The live
state is one module-level tuple, replaced whole on every transition. A reader
gets the previous snapshot or the next one, never a half-written one, and
never waits - so there is no lock anywhere in this file. That is deliberate:
a lock here would let a diagnostic stall the thing it exists to measure.

**It is bounded and it fails quiet.** The completed-span ring has a fixed
size. Nothing here raises into a caller: an instrument that can break a click
is worse than no instrument.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

#: The stages a desk interaction passes through, in the order they happen.
#: Not enforced - a surface that skips one is normal (not every click charts),
#: and a surface that invents one is recorded as it named it. This tuple is
#: documentation plus the vocabulary the reader groups on.
STAGES: tuple[str, ...] = (
    "page_select",
    "tab_select",
    "model_apply",
    "layout",
    "first_paint",
    "chart_request",
    "chart_ready",
)

#: Completed spans kept in memory for a reader. Small on purpose: this is a
#: live aid, not an evidence store, and nothing downstream may depend on it.
MAX_SPANS = 200

#: (interaction_id, kind, detail, stage, started_perf) or None.
#: Replaced whole, never mutated in place - see the module docstring.
_current: tuple[str, str, str, str, float] | None = None
_counter = 0
_spans: deque[dict[str, Any]] = deque(maxlen=MAX_SPANS)


def begin(kind: str, detail: str = "") -> str:
    """Open an interaction and return its id. Replaces any open one.

    Replacing rather than nesting is the honest model for a desk: a trader who
    clicks a second page while the first is still laying out has abandoned the
    first, and the stalls from here on belong to the new click. The abandoned
    span is still recorded, marked `superseded`, because "the click nobody
    waited for" is exactly the kind of thing worth seeing in a slow session.
    """
    global _current, _counter
    try:
        _counter += 1
        _close(outcome="superseded")
        interaction_id = f"{str(kind or 'interaction').strip() or 'interaction'}-{_counter}"
        _current = (
            interaction_id,
            str(kind or ""),
            str(detail or ""),
            "begin",
            time.perf_counter(),
        )
        return interaction_id
    except Exception:  # pragma: no cover - an instrument may not break a click
        return ""


def mark(stage: str) -> None:
    """Record that the open interaction reached `stage`. No-op if none is."""
    global _current
    try:
        snapshot = _current
        if snapshot is None:
            return
        interaction_id, kind, detail, _previous, started = snapshot
        _current = (interaction_id, kind, detail, str(stage or ""), started)
    except Exception:  # pragma: no cover
        return


def end(outcome: str = "done") -> None:
    """Close the open interaction, if there is one."""
    try:
        _close(outcome=outcome)
    except Exception:  # pragma: no cover
        return


def _close(*, outcome: str) -> None:
    global _current
    snapshot = _current
    _current = None
    if snapshot is None:
        return
    interaction_id, kind, detail, stage, started = snapshot
    _spans.append(
        {
            "interaction_id": interaction_id,
            "kind": kind,
            "detail": detail,
            "last_stage": stage,
            "outcome": outcome,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
    )


def current() -> dict[str, Any] | None:
    """What is open right now, for a reader on ANY thread. Never blocks.

    Returns a fresh dict so a caller cannot reach into the live state, and
    `None` when nothing is open - which is the normal case for a desk sitting
    idle, and is why a stall with no interaction is not a missing measurement.
    """
    snapshot = _current
    if snapshot is None:
        return None
    interaction_id, kind, detail, stage, started = snapshot
    return {
        "interaction_id": interaction_id,
        "kind": kind,
        "detail": detail,
        "stage": stage,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


def recent_spans() -> list[dict[str, Any]]:
    """The completed spans still in the ring, oldest first."""
    return list(_spans)


def reset() -> None:
    """Drop all state. For tests and for a fresh session."""
    global _current, _counter
    _current = None
    _counter = 0
    _spans.clear()
