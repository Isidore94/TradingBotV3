"""The market read on three axes, and its evening grade (WISHLIST P2-8 8a/8b).

The morning read says what kind of day it is on three axes, each from what was
known at the prior session's close:

* **SPY state** - the desk's D1 environment label (`d1_environment_store`).
* **Breadth** - % of `universe_all` above SMA20/SMA50 and the A/D count
  (`market_breadth_store`).
* **Internals** - VXX, HYG, TLT, RSP and MAGS (`market_internals`), read as
  short phrases: "vol bid", "narrow tape", ...

The evening grades each axis that leans a way against what SPY did next: the
move from the basis session's close to the graded session's close, in the
basis session's daily ATR(14), with the read grader's own flat band
(`market_read_grades.FLAT_BAND_ATR`, rule `atr_0.25_v1`). An axis with no lean
is `no_call`, never right or wrong. Missing data is `unmeasured`.

Display and grading only. The SPY-pause champion, every detector, score and
alert are untouched; nothing here is read by any of them. The rules above the
"loaders" line are PURE and versioned; the loaders only read local files.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

_log = logging.getLogger(__name__)

#: The whole rule set's name. A changed threshold or phrase is a new version.
AXES_RULE_VERSION = "market_axes_v1"

#: Breadth lean: >= STRONG % of names above SMA20 leans up, <= WEAK leans down.
BREADTH_STRONG_PCT = 60.0
BREADTH_WEAK_PCT = 40.0

#: A live internals snapshot older than this is missing, not a reading.
INTERNALS_MAX_AGE_MINUTES = 30

#: How many internals phrases the regime chip carries (the tooltip has all).
CHIP_PHRASES = 2

#: (axis, state) -> phrase. `market_internals` states are risk_on / risk_off
#: from the risk point of view: VXX rising is risk_off, RSP beating SPY is
#: risk_on, MAGS beating SPY is risk_on.
_PHRASES = {
    ("vol", "risk_off"): "vol bid",
    ("vol", "risk_on"): "vol offered",
    ("breadth", "risk_on"): "broad tape",
    ("breadth", "risk_off"): "narrow tape",
    ("credit", "risk_on"): "credit firm",
    ("credit", "risk_off"): "credit weak",
    ("concentration", "risk_on"): "mega-caps lead",
    ("concentration", "risk_off"): "mega-caps lag",
    ("duration", "risk_off"): "bonds bid",
    ("duration", "risk_on"): "bonds offered",
}
_PHRASE_ORDER = ("vol", "breadth", "credit", "concentration", "duration")

#: The internals tape word -> lean. Tilts and mixed carry no lean.
_TAPE_LEAN = {"risk_on": "up", "risk_off": "down"}

VERDICT_NO_CALL = "no_call"
VERDICT_PENDING = "pending"
UNMEASURED_PREFIX = "unmeasured"


# ---------------------------------------------------------------------------
# the pure rules
# ---------------------------------------------------------------------------
def spy_axis(label: Any) -> dict[str, Any]:
    """The SPY state axis from the desk's D1 label."""
    from market_read_grades import LABEL_DIRECTION

    state = str(label or "").strip() or "unknown"
    return {
        "axis": "spy",
        "state": state,
        "lean": LABEL_DIRECTION.get(state, ""),
        "text": f"SPY {state.replace('_', ' ')}",
    }


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def breadth_axis(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """The breadth axis from one stored breadth row. No row is `unknown`."""
    pct20 = _number((row or {}).get("pct_above_sma20"))
    if pct20 is None:
        return {"axis": "breadth", "state": "unknown", "lean": "", "text": "breadth unknown"}
    if pct20 >= BREADTH_STRONG_PCT:
        state, lean = "strong", "up"
    elif pct20 <= BREADTH_WEAK_PCT:
        state, lean = "weak", "down"
    else:
        state, lean = "mixed", ""
    parts = [f"{pct20:.0f}% > SMA20"]
    pct50 = _number((row or {}).get("pct_above_sma50"))
    if pct50 is not None:
        parts.append(f"{pct50:.0f}% > SMA50")
    up, down = (row or {}).get("advancers"), (row or {}).get("decliners")
    if isinstance(up, int) and isinstance(down, int) and (up or down):
        parts.append(f"A/D {up}/{down}")
    return {
        "axis": "breadth",
        "state": state,
        "lean": lean,
        "text": f"breadth {state} ({', '.join(parts)})",
    }


def _parse_stamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value or "").strip())
    except ValueError:
        return None


def internals_fresh(snapshot: Mapping[str, Any] | None, now: datetime | None) -> bool:
    """False when `now` is given and the snapshot is older than the max age."""
    if now is None:
        return True
    stamp = _parse_stamp((snapshot or {}).get("as_of"))
    if stamp is None:
        return False
    if (stamp.tzinfo is None) != (now.tzinfo is None):
        return False
    age = now - stamp
    return -timedelta(minutes=1) <= age <= timedelta(minutes=INTERNALS_MAX_AGE_MINUTES)


def internals_phrases(snapshot: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The measured internals as phrases, in a fixed order. Unknown/flat are left out."""
    readings = (snapshot or {}).get("readings")
    if not isinstance(readings, Mapping):
        return ()
    out: list[str] = []
    for axis in _PHRASE_ORDER:
        reading = readings.get(axis)
        if not isinstance(reading, Mapping):
            continue
        phrase = _PHRASES.get((axis, str(reading.get("state") or "")))
        if phrase:
            out.append(phrase)
    return tuple(out)


def internals_axis(
    snapshot: Mapping[str, Any] | None, *, now: datetime | None = None
) -> dict[str, Any] | None:
    """The internals axis, or None (omitted) when nothing was measured or it is stale."""
    if not isinstance(snapshot, Mapping) or not internals_fresh(snapshot, now):
        return None
    phrases = internals_phrases(snapshot)
    if not phrases:
        return None
    tape = str(snapshot.get("tape") or "")
    return {
        "axis": "internals",
        "state": tape or "unknown",
        "lean": _TAPE_LEAN.get(tape, ""),
        "phrases": phrases,
        "text": "internals " + " + ".join(phrases),
    }


def regime_line(label: Any, snapshot: Mapping[str, Any] | None, *, now: datetime | None = None) -> str:
    """`bearish_weak + vol bid + narrow tape`; just the label when internals are missing."""
    text = str(label or "")
    axis = internals_axis(snapshot, now=now)
    if axis is None or not text:
        return text
    return " + ".join([text, *axis["phrases"][:CHIP_PHRASES]])


def morning_read(
    basis: str,
    *,
    d1_label: Any,
    breadth_row: Mapping[str, Any] | None,
    internals_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The three-axis read from the `basis` session's close. Internals omitted when missing."""
    axes = [spy_axis(d1_label), breadth_axis(breadth_row)]
    internals = internals_axis(internals_snapshot)
    if internals is not None:
        axes.append(internals)
    return {"rule_version": AXES_RULE_VERSION, "basis": str(basis or "")[:10], "axes": axes}


def read_line(read: Mapping[str, Any] | None) -> str:
    """One line: `Market read (from 2026-09-24 close): SPY mixed · breadth weak (...) · internals ...`."""
    axes = list((read or {}).get("axes") or ())
    if not axes:
        return ""
    basis = str((read or {}).get("basis") or "")
    head = f"Market read (from {basis} close): " if basis else "Market read: "
    return head + " · ".join(str(axis.get("text") or "") for axis in axes)


def _closes(bars: Sequence[Any]) -> dict[str, float]:
    from indicators.d1_environment import session_of

    out: dict[str, float] = {}
    for bar in bars or ():
        stamp = session_of(bar)
        raw = bar.get("close") if isinstance(bar, Mapping) else getattr(bar, "close", None)
        close = _number(raw)
        if stamp and close is not None and close > 0:
            out[stamp] = close
    return out


def grade_axes(
    read: Mapping[str, Any] | None,
    *,
    spy_daily_bars: Sequence[Any],
    target_session: str,
    now: datetime,
) -> list[dict[str, Any]]:
    """Grade each axis against SPY from the basis close to `target_session`'s close. PURE."""
    import market_early_close
    import market_read_grades as grader

    basis = str((read or {}).get("basis") or "")[:10]
    target = str(target_session or "")[:10]
    closes = _closes(spy_daily_bars)
    move_atr: float | None = None
    shared = ""
    try:
        target_day = date.fromisoformat(target)
    except ValueError:
        target_day = None
    if target_day is None or not basis:
        shared = f"{UNMEASURED_PREFIX}:no_session"
    elif now < market_early_close.session_close(target_day):
        shared = f"{VERDICT_PENDING} {target}"
    elif basis not in closes or target not in closes:
        shared = f"{UNMEASURED_PREFIX}:no_daily_close"
    else:
        atr = grader.daily_atr(list(spy_daily_bars or ()), through=basis)
        if atr is None or atr <= 0:
            shared = f"{UNMEASURED_PREFIX}:no_atr"
        else:
            move_atr = (closes[target] - closes[basis]) / atr
    out: list[dict[str, Any]] = []
    for axis in (read or {}).get("axes") or ():
        lean = str(axis.get("lean") or "")
        if not lean:
            verdict = VERDICT_NO_CALL
        elif shared:
            verdict = shared
        else:
            verdict = grader._verdict_for(lean, move_atr)
        out.append({
            "axis": str(axis.get("axis") or ""),
            "state": str(axis.get("state") or ""),
            "lean": lean,
            "verdict": verdict,
            "move_atr": None if move_atr is None else round(move_atr, 3),
            "flat_band_rule": grader.FLAT_BAND_RULE,
        })
    return out


def grade_summary(grades: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Integer counts of right / wrong / flat / pending / no_call / unmeasured."""
    counts = {"right": 0, "wrong": 0, "flat": 0, "pending": 0, "no_call": 0, "unmeasured": 0}
    for grade in grades or ():
        verdict = str(grade.get("verdict") or "")
        if verdict in ("right", "wrong", "flat", VERDICT_NO_CALL):
            counts[verdict] += 1
        elif verdict.startswith(VERDICT_PENDING):
            counts["pending"] += 1
        else:
            counts["unmeasured"] += 1
    return counts


# ---------------------------------------------------------------------------
# loaders (local files only; call these off the Qt thread)
# ---------------------------------------------------------------------------
def internals_snapshot_for_session(session: str) -> dict[str, Any] | None:
    """The session's close-vs-prior-close internals from the durable M5 tape, or None."""
    try:
        import market_calendar
        from day_review_bars import read_session_bars
        from market_internals import INTERNALS_SYMBOLS, build_internals_snapshot

        day = date.fromisoformat(str(session)[:10])
        prior = market_calendar.previous_session(day)
        today = read_session_bars(day.isoformat()) or {}
        before = read_session_bars(prior.isoformat()) or {}
    except Exception:  # noqa: BLE001 - a missing tape is a missing axis
        _log.debug("Internals tape unreadable for %s.", session, exc_info=True)
        return None
    if not today:
        return None
    bars: dict[str, list[Any]] = {}
    for symbol in INTERNALS_SYMBOLS:
        series = list(before.get(symbol) or ()) + list(today.get(symbol) or ())
        if series:
            bars[symbol] = sorted(series, key=lambda bar: bar["dt"])
    snapshot = build_internals_snapshot(bars, as_of=day.isoformat())
    return snapshot if snapshot.get("tape") != "unknown" else None


def morning_read_for(basis: str) -> dict[str, Any]:
    """The read built from the `basis` session's close, from local stores."""
    label = ""
    try:
        import d1_environment_store

        label = d1_environment_store.label_for_session(basis, "SPY")
    except Exception:  # noqa: BLE001
        _log.debug("D1 label unreadable for %s.", basis, exc_info=True)
    row = None
    try:
        import market_breadth_store

        row = market_breadth_store.row_for_session(basis)
    except Exception:  # noqa: BLE001
        _log.debug("Breadth row unreadable for %s.", basis, exc_info=True)
    return morning_read(
        basis,
        d1_label=label,
        breadth_row=row,
        internals_snapshot=internals_snapshot_for_session(basis),
    )


def latest_morning_read(now: datetime | None = None) -> dict[str, Any]:
    """The read from the last completed session's close (Market Prep)."""
    import market_calendar

    moment = now or datetime.now().astimezone()
    basis = market_calendar.last_completed_session(moment).isoformat()
    read = morning_read_for(basis)
    read["line"] = read_line(read)
    return read


def day_axes(session: str, now: datetime) -> dict[str, Any]:
    """Day Review: the morning read going INTO `session`, and its grade."""
    import market_calendar
    import market_read_grades as grader

    day = date.fromisoformat(str(session)[:10])
    basis = market_calendar.previous_session(day).isoformat()
    read = morning_read_for(basis)
    grades = grade_axes(
        read,
        spy_daily_bars=grader.daily_bars_for_symbol("SPY"),
        target_session=day.isoformat(),
        now=now,
    )
    return {
        "read": read,
        "line": read_line(read),
        "grades": grades,
        "summary": grade_summary(grades),
    }


__all__ = [
    "AXES_RULE_VERSION",
    "BREADTH_STRONG_PCT",
    "BREADTH_WEAK_PCT",
    "CHIP_PHRASES",
    "INTERNALS_MAX_AGE_MINUTES",
    "VERDICT_NO_CALL",
    "breadth_axis",
    "day_axes",
    "grade_axes",
    "grade_summary",
    "internals_axis",
    "internals_phrases",
    "internals_snapshot_for_session",
    "latest_morning_read",
    "morning_read",
    "morning_read_for",
    "read_line",
    "regime_line",
    "spy_axis",
]
