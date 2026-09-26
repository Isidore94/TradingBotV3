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


# -- S16 item 5: the machine's labels checked against the trader's regime ------

#: The Mentor question kind that asks "still a <regime>?" after a disagreement.
CHECK_KIND = "regime_check"
#: Consecutive disagreeing sessions before the Mentor asks.
DISAGREE_SESSIONS = 3
#: The table symbol whose labels are compared with the trader's regime.
CHECK_SYMBOL = "SPY"

#: Machine label -> the trader regimes it is consistent with. The one mapping.
MACHINE_TO_TRADER: dict[str, tuple[str, ...]] = {
    "bear_channel": ("bear_channel_lower_highs", "capitulation"),
    "uptrend": ("bull_run", "recovery", "weekly_hh_then_compression"),
    "compression": ("weekly_hh_then_compression", "range"),
}
#: How the Mentor says each machine label.
MACHINE_WORDS: dict[str, str] = {
    "bear_channel": "a daily lower-high channel with a bearish D1",
    "uptrend": "a daily higher-low channel with a bullish D1",
    "compression": "compressed daily ranges with no trend",
}


def machine_regime(row: Mapping[str, Any] | None) -> str | None:
    """The machine's structural label for one regime-table row, or None (unknown).

    From the row's own facts only: `bear_channel` = daily channel `lh_ll` and a
    bearish D1 env_key; `uptrend` = `hh_hl` and a bullish D1; `compression` =
    ATR percentile compressed with a `mixed` channel or a `neutral_chop` D1.
    Anything else (unknown bars, a channel and a D1 that disagree) is None.
    """
    if not isinstance(row, Mapping):
        return None
    structure = row.get("structure") if isinstance(row.get("structure"), Mapping) else {}
    channel_facts = structure.get("daily_channel") if isinstance(structure.get("daily_channel"), Mapping) else {}
    channel = str(channel_facts.get("label") or "")
    atr = structure.get("atr") if isinstance(structure.get("atr"), Mapping) else {}
    timeframes = row.get("timeframes") if isinstance(row.get("timeframes"), Mapping) else {}
    d1 = str(timeframes.get("D1") or "")
    if channel == "lh_ll" and d1.startswith("bearish"):
        return "bear_channel"
    if channel == "hh_hl" and d1.startswith("bullish"):
        return "uptrend"
    if atr.get("compressed") is True and (channel == "mixed" or d1 == "neutral_chop"):
        return "compression"
    return None


def machine_agrees(row: Mapping[str, Any] | None, trader_regime: Any) -> bool | None:
    """True/False when the machine has a label and the trader a regime; None is unknown.

    Unknown on either side never counts as disagreement.
    """
    machine = machine_regime(row)
    regime = str(trader_regime or "").strip()
    if machine is None or regime not in VOCABULARY:
        return None
    return regime in MACHINE_TO_TRADER[machine]


def _latest_entry(rows: Iterable[Mapping[str, Any]]) -> date | None:
    days = [_day(row.get("entered_at")) for row in rows or () if isinstance(row, Mapping)]
    known = [day for day in days if day is not None]
    return max(known) if known else None


def disagreement_runs(
    rows: Iterable[Mapping[str, Any]],
    table_rows: Iterable[Mapping[str, Any]],
    today: Any,
    *,
    symbol: str = CHECK_SYMBOL,
    reset_on_answer: bool = True,
) -> list[dict[str, Any]]:
    """Runs of consecutive table sessions where the machine disagrees with the trader.

    Oldest first. A session that agrees or is unknown on either side ends a run;
    with ``reset_on_answer`` so does a session on or before the trader's latest
    regime entry (an answer resets the count). The last run is `open` when it
    reaches the newest table session up to ``today``.
    """
    trader_rows = [dict(row) for row in rows or () if isinstance(row, Mapping)]
    last_day = _day(today)
    answered = _latest_entry(trader_rows) if reset_on_answer else None
    dated = [
        (_day(row.get("session_date")), row)
        for row in table_rows or ()
        if isinstance(row, Mapping) and str(row.get("symbol") or "").upper() == symbol
    ]
    sessions = sorted(
        ((day, row) for day, row in dated if day is not None and (last_day is None or day <= last_day)),
        key=lambda item: item[0],
    )
    runs: list[dict[str, Any]] = []
    current: list[tuple[date, Mapping[str, Any], str]] = []

    def _close(open_run: bool) -> None:
        if current:
            runs.append({
                "start": current[0][0].isoformat(),
                "end": current[-1][0].isoformat(),
                "sessions": [day.isoformat() for day, _row, _regime in current],
                "streak": len(current),
                "trader_regime": current[-1][2],
                "machine": machine_regime(current[-1][1]) or "",
                "symbol": symbol,
                "open": open_run,
            })
        current.clear()

    for day, row in sessions:
        regime = str((regime_at(trader_rows, day) or {}).get("regime") or "")
        if (answered is not None and day <= answered) or machine_agrees(row, regime) is not False:
            _close(False)
            continue
        if current and current[-1][2] != regime:
            _close(False)
        current.append((day, row, regime))
    _close(True)
    return runs


def open_disagreement(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The run the Mentor asks about: open and at least `DISAGREE_SESSIONS` long."""
    for run in runs or ():
        if run.get("open") and int(run.get("streak") or 0) >= DISAGREE_SESSIONS:
            return dict(run)
    return None


def lane(
    rows: Iterable[Mapping[str, Any]],
    today: Any,
    table_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Everything the Mentor needs, built from rows already read (pure)."""
    all_rows = [dict(row) for row in rows or () if isinstance(row, Mapping)]
    answered: dict[str, dict[str, str]] = {}
    for row in all_rows:
        entered = _day(row.get("entered_at"))
        if entered is not None:
            answered[f"{QUESTION_KIND}:{week_key(entered)}"] = {"answered_at": entered.isoformat()}
    table = [row for row in table_rows or () if isinstance(row, Mapping)]
    runs = disagreement_runs(all_rows, table, today)
    ask = open_disagreement(runs)
    for run in runs + disagreement_runs(all_rows, table, today, reset_on_answer=False):
        # A run the trader answered or the labels closed is never asked again.
        if int(run.get("streak") or 0) >= DISAGREE_SESSIONS and (ask is None or run["start"] != ask["start"]):
            answered[f"{CHECK_KIND}:{run['start']}"] = {"answered_at": str(run.get("end") or "")}
    return {
        "loaded": True,
        "today": (_day(today) or date.min).isoformat(),
        "rows": all_rows,
        "current": current_regime(all_rows, today),
        "prefills": [prefill.__dict__ for prefill in pending_prefills(all_rows)],
        "answered": answered,
        "disagreement": ask,
    }


def _table_rows(path: Any = None) -> list[dict[str, Any]]:
    """The regime table's rows, read-only; an unreadable table is no rows."""
    try:
        import market_regimes

        if path is None:
            from project_paths import MARKET_REGIME_TABLE_FILE

            path = MARKET_REGIME_TABLE_FILE
        return market_regimes.read_table(path)
    except Exception:  # noqa: BLE001 - no table is no disagreement, never a question
        return []


def load_lane(store: Any, today: Any, *, table_path: Any = None) -> dict[str, Any]:
    """Read the journal and the regime table once and build the lane. IO: run it on a worker."""
    return lane(store.list_structural_regime(), today, _table_rows(table_path))


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
