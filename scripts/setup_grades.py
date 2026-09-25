"""One grade ladder - PROVEN / A / B / C / D / New - from the trackers' own results.

Trader, 2026-09-22: *"Proven should be an amazing setup. A and B should be great
setups with real data. C and D can be less good."* The cut-offs were written
BEFORE anyone looked at who landed where, previewed on live data, and approved
as they stand ("keep the cut offs").

Two populations, ONE ladder, never pooled:

* **Swing** - a Setup Tracker family at its row grain ``(side, bucket, family)``,
  read off ``master_avwap_setup_type_recent_stats.csv`` (the tracker's lately
  window, live namespace only - study groups are research, not picks).
* **Day trade** - an M5 alert type ``(bounce_type, side)`` over the lately
  window of the outcome log, judged as a bracket trade: did it reach **+1R before
  -1R**? A row that shows both for the first time is a LOSS (the adverse extreme
  first, as `real_miss` does). An alert that touched neither is undecided and
  counted apart, never a zero.

Presentation only. Nothing here is read by a detector, a score, an alert
decision, a watchlist, Focus, the review queue or ``review_policy.json``: the
desk uses a grade to ORDER lists (best first, nothing withheld) and to print a
badge. Pure: rows in, plain dicts out, no file access, no clock.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

SCHEMA = "setup_grades_v1"

PROVEN, A, B, C, D, NEW = "PROVEN", "A", "B", "C", "D", "New"
GRADES = (PROVEN, A, B, C, D, NEW)

#: Best first. New sorts ABOVE D: unknown is not the same as measured-bad, but
#: it never outranks a measured C.
_SORT_RANK = {PROVEN: 0, A: 1, B: 2, C: 3, NEW: 4, D: 5}
UNGRADED_RANK = _SORT_RANK[NEW]

MIN_N = 30          # evidence_stats.MIN_REPORTABLE_N, stated here as a rule value
PROVEN_MIN_N = 100
PROVEN_MIN_SESSIONS = 15
PROVEN_MIN_LOW_BOUND = 0.60
A_MIN_SESSIONS = 10
A_MIN_LOW_BOUND = 0.55
B_MIN_LOW_BOUND = 0.50
C_MIN_WIN_RATE = 0.50

RULES_TEXT = (
    "PROVEN: 100+ closed over 15+ sessions, win-rate low bound >= 60%, avg R > 0. "
    "A: 30+ over 10+ sessions, low bound >= 55%, avg R > 0. "
    "B: 30+, low bound >= 50%. C: 30+, win rate >= 50%. D: 30+, below 50%. "
    "New: under 30."
)


def wilson_lower_bound(wins: int, n: int) -> float | None:
    """The desk's ONE Wilson (95%, `swing_headline`)."""
    from swing_headline import wilson_lower_bound as _wilson

    return _wilson(int(wins), int(n)) if n else None


def grade_for(
    *, n: int, sessions: int, wins: int, avg_r: float | None
) -> dict[str, Any]:
    """The ladder, once. Returns the grade with the numbers it was read from."""
    n = int(n or 0)
    wins = int(wins or 0)
    sessions = int(sessions or 0)
    win_rate = (wins / n) if n else None
    low = wilson_lower_bound(wins, n)
    positive = avg_r is not None and float(avg_r) > 0
    if n < MIN_N:
        grade = NEW
    elif (
        n >= PROVEN_MIN_N
        and sessions >= PROVEN_MIN_SESSIONS
        and (low or 0) >= PROVEN_MIN_LOW_BOUND
        and positive
    ):
        grade = PROVEN
    elif sessions >= A_MIN_SESSIONS and (low or 0) >= A_MIN_LOW_BOUND and positive:
        grade = A
    elif (low or 0) >= B_MIN_LOW_BOUND:
        grade = B
    elif (win_rate or 0) >= C_MIN_WIN_RATE:
        grade = C
    else:
        grade = D
    return {
        "grade": grade,
        "n": n,
        "wins": wins,
        "sessions": sessions,
        "win_rate": win_rate,
        "low_bound": low,
        "avg_r": None if avg_r is None else float(avg_r),
    }


def sort_rank(grade: str | None) -> int:
    return _SORT_RANK.get(str(grade or NEW), UNGRADED_RANK)


# ---------------------------------------------------------------------------
# swing
# ---------------------------------------------------------------------------


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None  # NaN is unmeasured


def swing_key(side: Any, bucket: Any, family: Any) -> str:
    return "|".join(
        (
            str(side or "").strip().upper(),
            str(bucket or "").strip().lower(),
            str(family or "general").strip().lower() or "general",
        )
    )


def swing_cells(recent_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One graded cell per live tracker family row.

    Wins, losses and flats are COUNTS (never a recency-weighted rate), and a
    flat is not a win. Avg R is the representative (primary-stop) closed R,
    falling back to the cross-variant mean - the same preference as the tracker.
    """
    cells = []
    for row in recent_rows or ():
        if str(row.get("namespace") or "live").strip().lower() != "live":
            continue
        wins = _int(row.get("n_wins"))
        n = wins + _int(row.get("n_losses")) + _int(row.get("n_flats"))
        avg_r = _float(row.get("representative_closed_r"))
        if avg_r is None:
            avg_r = _float(row.get("avg_closed_r"))
        cell = grade_for(n=n, sessions=_int(row.get("n_entry_sessions")), wins=wins, avg_r=avg_r)
        cell.update(
            side=str(row.get("side") or "").strip().upper(),
            bucket=str(row.get("priority_bucket") or "").strip(),
            family=str(row.get("setup_family") or "general").strip() or "general",
        )
        cell["key"] = swing_key(cell["side"], cell["bucket"], cell["family"])
        cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# day trade: +1R before -1R
# ---------------------------------------------------------------------------

WIN, LOSS, UNDECIDED, OPEN = "win", "loss", "undecided", "open"


def _flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def bracket_results(outcome_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One result per alert: did it reach +1R before -1R?

    The flags on the outcome log are cumulative, so the FIRST row (by bars
    elapsed) on which either is set decides. Both set first on the same row is a
    LOSS - the adverse extreme is assumed first when the bar cannot say.
    """
    from setup_scoreboard import bounce_type_from_event_id

    by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcome_rows or ():
        event_id = str(row.get("event_id") or "").strip()
        if event_id:
            by_event[event_id].append(row)
    results = []
    for event_id, rows in by_event.items():
        ordered = sorted(
            enumerate(rows), key=lambda item: (_int(item[1].get("bars_elapsed")), item[0])
        )
        outcome = None
        for _index, row in ordered:
            target, stop = _flag(row.get("target_1r_hit")), _flag(row.get("stop_hit"))
            if stop:
                outcome = LOSS
                break
            if target:
                outcome = WIN
                break
        if outcome is None:
            finished = any(str(row.get("event_type") or "").strip().lower() == "final" for row in rows)
            outcome = UNDECIDED if finished else OPEN
        first = rows[0]
        results.append(
            {
                "event_id": event_id,
                "trade_date": str(first.get("trade_date") or "").strip(),
                "side": str(first.get("direction") or "").strip().upper(),
                "bounce_type": bounce_type_from_event_id(event_id),
                "result": outcome,
            }
        )
    return results


def daytrade_cells(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One graded cell per (bounce type, side).

    An alert carrying several bounce types counts under EACH of them, as the
    Daytrade Tracker counts it. The bracket trade's R is +1 or -1, so avg R is
    ``2 * win_rate - 1``: positive exactly when it wins more than it loses.
    """
    from held_run_score import bounce_components

    tallies: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results or ():
        for component in bounce_components(result.get("bounce_type")) or ():
            key = (str(component), str(result.get("side") or "").upper())
            tally = tallies.setdefault(
                key, {"wins": 0, "losses": 0, "undecided": 0, "open": 0, "sessions": set()}
            )
            outcome = result.get("result")
            if outcome == WIN:
                tally["wins"] += 1
            elif outcome == LOSS:
                tally["losses"] += 1
            elif outcome == UNDECIDED:
                tally["undecided"] += 1
            else:
                tally["open"] += 1
            if outcome in (WIN, LOSS) and result.get("trade_date"):
                tally["sessions"].add(result["trade_date"])
    cells = []
    for (bounce_type, side), tally in tallies.items():
        n = tally["wins"] + tally["losses"]
        avg_r = (2 * tally["wins"] / n - 1) if n else None
        cell = grade_for(n=n, sessions=len(tally["sessions"]), wins=tally["wins"], avg_r=avg_r)
        cell.update(
            bounce_type=bounce_type,
            side=side,
            undecided=tally["undecided"],
            open=tally["open"],
            key=daytrade_key(bounce_type, side),
        )
        cells.append(cell)
    return cells


def daytrade_key(bounce_type: Any, side: Any) -> str:
    return f"{str(bounce_type or '').strip().lower()}|{str(side or '').strip().upper()}"


# ---------------------------------------------------------------------------
# the payload the desk reads
# ---------------------------------------------------------------------------


def build_payload(
    *,
    recent_rows: Iterable[Mapping[str, Any]] | None,
    outcome_rows: Iterable[Mapping[str, Any]] | None,
    as_of: str = "",
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "as_of": str(as_of or ""),
        "rules": RULES_TEXT,
        "swing": swing_cells(recent_rows or ()),
        "daytrade": daytrade_cells(bracket_results(outcome_rows or ())),
    }


def _best_first(cells: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        cells,
        key=lambda cell: (
            sort_rank(cell.get("grade")),
            -(cell.get("low_bound") or 0.0),
            -(cell.get("n") or 0),
        ),
    )


def daytrade_order(payload: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """``[(bounce_type, SIDE)]`` best first, graded cells above New only.

    The shape `working_lately.priority_rank` already reads, so the M5 bar and
    the waiting list sort by grade with no change to their own code. An
    unlisted alert (New, D, or a type the log has not seen) sorts after every
    listed one, in arrival order.
    """
    cells = [cell for cell in (payload or {}).get("daytrade") or () if sort_rank(cell.get("grade")) < UNGRADED_RANK]
    return [(str(cell["bounce_type"]), str(cell["side"])) for cell in _best_first(cells)]


def swing_lookup(payload: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    return {str(cell.get("key")): cell for cell in (payload or {}).get("swing") or ()}


def daytrade_lookup(payload: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    return {str(cell.get("key")): cell for cell in (payload or {}).get("daytrade") or ()}


def swing_sort_key(cell: Mapping[str, Any] | None) -> tuple:
    """Best first; a row with no graded family is New."""
    if not cell:
        return (UNGRADED_RANK, 0.0)
    return (sort_rank(cell.get("grade")), -(cell.get("low_bound") or 0.0))


def daytrade_grade_for_alert(
    payload_lookup: Mapping[str, Mapping[str, Any]], bounce_type: Any, side: Any
) -> str:
    """The alert's grade: the BEST grade among the bounce types it carries.

    An alert whose types the log has never graded is New. A measured D stays D
    even though New sorts above it.
    """
    from held_run_score import bounce_components

    best: str | None = None
    for part in str(bounce_type or "").replace(";", "-").split("-"):
        for component in bounce_components(part) or ():
            cell = payload_lookup.get(daytrade_key(component, side))
            if not cell:
                continue
            grade = str(cell.get("grade") or NEW)
            if best is None or sort_rank(grade) < sort_rank(best):
                best = grade
    return best or NEW


def badge(grade: str | None) -> str:
    grade = str(grade or NEW)
    return "NEW" if grade == NEW else grade


# ---------------------------------------------------------------------------
# hold-out view (P2-9 9b) - read-only, beside the cells above
# ---------------------------------------------------------------------------

#: What a window with no cell for a key says.
NOT_IN_WINDOW = "none in this window"


def holdout_text(cell: Mapping[str, Any] | None) -> str:
    """One graded cell as the hold-out column prints it. Under the floor says so."""
    if not cell:
        return NOT_IN_WINDOW
    n = int(cell.get("n") or 0)
    if n < MIN_N:
        return f"n<{MIN_N} (n={n})"
    rate = cell.get("win_rate")
    low = cell.get("low_bound")
    avg = cell.get("avg_r")
    parts = [badge(cell.get("grade"))]
    if rate is not None:
        parts.append(f"win {float(rate):.0%}" + (f" (low {float(low):.0%})" if low is not None else ""))
    if avg is not None:
        parts.append(f"avg {float(avg):+.2f}R")
    parts.append(f"n={n}")
    return " · ".join(parts)


def holdout_view(
    recent_cells: Iterable[Mapping[str, Any]] | None,
    prior_cells: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Each key's recent cell beside the prior window's, recent best first.

    Both sides are the SAME ladder (`swing_cells` / `daytrade_cells`) over two
    date windows; nothing here recomputes a grade.
    """
    recent = {str(cell.get("key")): cell for cell in recent_cells or ()}
    prior = {str(cell.get("key")): cell for cell in prior_cells or ()}
    keys = [str(cell.get("key")) for cell in _best_first(recent.values())]
    keys += [key for key in (str(cell.get("key")) for cell in _best_first(prior.values())) if key not in recent]
    return [
        {
            "key": key,
            "recent_grade": (recent.get(key) or {}).get("grade"),
            "prior_grade": (prior.get(key) or {}).get("grade"),
            "recent_n": int((recent.get(key) or {}).get("n") or 0),
            "prior_n": int((prior.get(key) or {}).get("n") or 0),
            "recent_text": holdout_text(recent.get(key)),
            "prior_text": holdout_text(prior.get(key)),
        }
        for key in keys
    ]
