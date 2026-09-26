"""The trader's structural regime journal (S16 item 1): pure readers over its rows.

The trader types the regime; nothing here infers one. Rows come from
``JournalStore.list_structural_regime`` (append-only table ``structural_regime``).
A row may supersede an earlier one (``supersedes``) and a later row with the same
``start_date`` replaces an earlier one; nothing is ever edited in place.
Before the first segment the regime is unknown (``None``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping

SOURCE_TRADER = "trader"

#: The closed vocabulary, in the order the trader sees it.
VOCABULARY: tuple[str, ...] = (
    "bull_run",
    "weekly_hh_then_compression",
    "bear_channel_lower_highs",
    "range",
    "capitulation",
    "recovery",
)

LABELS: dict[str, str] = {
    "bull_run": "bull run",
    "weekly_hh_then_compression": "weekly higher highs then compression",
    "bear_channel_lower_highs": "bear channel, lower highs",
    "range": "range",
    "capitulation": "capitulation",
    "recovery": "recovery",
}

#: The Mentor question kind that asks for the regime.
QUESTION_KIND = "structural_regime"
#: The weekly answer that keeps the current regime ("still <current>").
STILL_PREFIX = "still_"


@dataclass(frozen=True)
class Prefill:
    """One past regime the trader described, offered for a one-click confirm."""

    key: str
    regime: str
    start_date: str
    structure_note: str
    label: str


#: The three past regimes the trader gave on 2026-09-26. Offered, never written
#: until the trader clicks Confirm; the start date is editable.
PREFILLS: tuple[Prefill, ...] = (
    Prefill("2026-03_bull_run", "bull_run", "2026-03-01", "March-May 2026 bull run", "March-May 2026"),
    Prefill(
        "2026-06_weekly_hh_then_compression",
        "weekly_hh_then_compression",
        "2026-06-01",
        "June-July weekly higher highs then compression",
        "June-July 2026",
    ),
    Prefill(
        "2026-08_bear_channel_lower_highs",
        "bear_channel_lower_highs",
        "2026-08-01",
        "August-now bear channel with lower highs",
        "August 2026 - now",
    ),
)

#: A prefill counts as confirmed once a row of its regime starts this close to it.
PREFILL_MATCH_DAYS = 45


def label(regime: Any) -> str:
    text = str(regime or "").strip()
    return LABELS.get(text, text.replace("_", " "))


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _segment_id(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("segment_id") or 0)
    except (TypeError, ValueError):
        return 0


def effective_segments(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The timeline the trader's rows say, oldest first.

    Drops a row a later row supersedes (by id, or by the same start date), then
    joins back-to-back segments of the same regime into one that keeps the first
    start date and the latest note.
    """
    ordered = sorted((dict(row) for row in rows or () if isinstance(row, Mapping)), key=_segment_id)
    superseded: set[int] = set()
    latest_for_start: dict[str, int] = {}
    for row in ordered:
        target = row.get("supersedes")
        if target not in (None, ""):
            try:
                superseded.add(int(target))
            except (TypeError, ValueError):
                pass
        start = _day(row.get("start_date"))
        if start is not None:
            latest_for_start[start.isoformat()] = _segment_id(row)
    live = []
    for row in ordered:
        start = _day(row.get("start_date"))
        regime = str(row.get("regime") or "").strip()
        if start is None or not regime or _segment_id(row) in superseded:
            continue
        if latest_for_start.get(start.isoformat()) != _segment_id(row):
            continue
        live.append(row)
    live.sort(key=lambda row: (str(row.get("start_date"))[:10], _segment_id(row)))
    merged: list[dict[str, Any]] = []
    for row in live:
        if merged and merged[-1]["regime"] == str(row.get("regime") or "").strip():
            first = merged[-1]
            merged[-1] = {**row, "start_date": first["start_date"], "first_segment_id": first["first_segment_id"]}
            continue
        merged.append(
            {
                **row,
                "regime": str(row.get("regime") or "").strip(),
                "start_date": str(row.get("start_date"))[:10],
                "first_segment_id": _segment_id(row),
            }
        )
    return merged


def _session_count(start: date, on: date) -> int | None:
    """Exchange sessions in [start, on], or None when the calendar cannot say."""
    try:
        from market_calendar import trading_days_between

        return int(trading_days_between(start - timedelta(days=1), on))
    except Exception:  # noqa: BLE001 - outside the calendar is unknown
        return None


def regime_at(rows: Iterable[Mapping[str, Any]], on: Any) -> dict[str, Any] | None:
    """The segment in force on ``on`` with its day count, or None (unknown).

    ``day_count`` is calendar days with the start date as day 1; ``session_count``
    is exchange sessions in [start, on] (None outside the calendar).
    """
    day = _day(on)
    if day is None:
        return None
    timeline = effective_segments(rows)
    found: dict[str, Any] | None = None
    for index, segment in enumerate(timeline):
        start = _day(segment["start_date"])
        if start is None or start > day:
            break
        found = dict(segment)
        following = timeline[index + 1]["start_date"] if index + 1 < len(timeline) else ""
        found["end_date"] = (_day(following) - timedelta(days=1)).isoformat() if following else ""
    if found is None:
        return None
    start = _day(found["start_date"])
    found["day_count"] = (day - start).days + 1
    found["session_count"] = _session_count(start, day)
    found["label"] = label(found["regime"])
    return found


def current_regime(rows: Iterable[Mapping[str, Any]], today: Any) -> dict[str, Any] | None:
    """The segment in force today, or None when the trader has typed none yet."""
    return regime_at(rows, today)


def pending_prefills(rows: Iterable[Mapping[str, Any]]) -> list[Prefill]:
    """The trader's three past regimes not yet confirmed into the table."""
    written = [
        (str(row.get("regime") or "").strip(), _day(row.get("start_date")))
        for row in rows or ()
        if isinstance(row, Mapping)
    ]
    out = []
    for prefill in PREFILLS:
        default = _day(prefill.start_date)
        if any(
            regime == prefill.regime and start is not None and abs((start - default).days) <= PREFILL_MATCH_DAYS
            for regime, start in written
        ):
            continue
        out.append(prefill)
    return out


def week_key(day: Any) -> str:
    """The ISO week a weekly regime question is about, e.g. ``2026-W39``."""
    value = _day(day)
    if value is None:
        return ""
    year, week, _ = value.isocalendar()
    return f"{year}-W{week:02d}"


def lane(rows: Iterable[Mapping[str, Any]], today: Any) -> dict[str, Any]:
    """Everything the Mentor needs, built from rows already read (pure)."""
    all_rows = [dict(row) for row in rows or () if isinstance(row, Mapping)]
    answered: dict[str, dict[str, str]] = {}
    for row in all_rows:
        entered = _day(row.get("entered_at"))
        if entered is not None:
            answered[f"{QUESTION_KIND}:{week_key(entered)}"] = {"answered_at": entered.isoformat()}
    return {
        "loaded": True,
        "today": (_day(today) or date.min).isoformat(),
        "rows": all_rows,
        "current": current_regime(all_rows, today),
        "prefills": [prefill.__dict__ for prefill in pending_prefills(all_rows)],
        "answered": answered,
    }


def load_lane(store: Any, today: Any) -> dict[str, Any]:
    """Read the table once and build the lane. IO: run it on a worker."""
    return lane(store.list_structural_regime(), today)


def segment_from_answer(
    answer_state: str,
    *,
    current: Mapping[str, Any] | None,
    session: Any,
) -> dict[str, Any]:
    """What one weekly answer appends. Raises ValueError for a non-regime answer.

    ``still_<regime>`` re-affirms the current segment (same start date and
    regime, superseding it, so the timeline and day count do not move); a
    vocabulary regime starts a new segment on the card's session date.
    """
    state = str(answer_state or "").strip()
    if state.startswith(STILL_PREFIX):
        if not current:
            raise ValueError("there is no current regime to keep")
        regime = state[len(STILL_PREFIX):]
        if regime != str(current.get("regime") or ""):
            raise ValueError(f"{state!r} does not match the current regime")
        return {
            "start_date": str(current.get("start_date"))[:10],
            "regime": regime,
            "structure_note": str(current.get("structure_note") or ""),
            "supersedes": current.get("segment_id"),
        }
    if state not in VOCABULARY:
        raise ValueError(f"{state!r} is not a regime")
    start = _day(session)
    if start is None:
        raise ValueError("a new regime needs the card's session date")
    return {"start_date": start.isoformat(), "regime": state, "structure_note": ""}
