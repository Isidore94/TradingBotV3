"""One grade ladder - PROVEN / A / B / C / D / New - from the trackers' own results.

Trader, 2026-09-22: *"Proven should be an amazing setup. A and B should be great
setups with real data. C and D can be less good."* The cut-offs were written
BEFORE anyone looked at who landed where, previewed on live data, and approved
as they stand ("keep the cut offs").

Two populations, ONE ladder, never pooled:

* **Swing** - a Setup Tracker family at its row grain ``(side, bucket, family)``,
  read off ``master_avwap_setup_type_recent_stats.csv`` (the tracker's lately
  window, live namespace only - study groups are research, not picks). The
  ladder reads its WIN AGAINST THE TAPE when 30+ picks have one: the pick's
  5-session side return beat SPY's same-side return over the same sessions
  (`swing_tape_stats`). Under that it reads the plain win and says
  ``tape: unknown``.
* **Day trade** - an M5 alert type ``(bounce_type, side)`` over the lately
  window of the outcome log, judged as a bracket trade: did it reach **+1R before
  -1R**? A row that shows both for the first time is a LOSS (the adverse extreme
  first, as `real_miss` does). An alert that touched neither is undecided and
  counted apart, never a zero. SPY-relative does not apply to a bracket.
  Beside it (S10c): a 2R grade on the same ladder (+2R before -1R), the EOD
  close R mean and median, and the share that reached +2R. The 1:1 grade is
  the one badges, the Show filter and sorting read.

PROVEN and A also need the family's cumulative R over the window to be >= 0
(`cum_r_lately`, the sum of each pick's own R); unknown is not >= 0.

Presentation only. Nothing here is read by a detector, a score, an alert
decision, a watchlist, Focus, the review queue or ``review_policy.json``: the
desk uses a grade to ORDER lists (best first, nothing withheld) and to print a
badge. Pure: rows in, plain dicts out, no file access, no clock.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

SCHEMA = "setup_grades_v2"

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
    "PROVEN: 100+ closed over 15+ sessions, win-rate low bound >= 60%, avg R > 0, "
    "cum R >= 0. "
    "A: 30+ over 10+ sessions, low bound >= 55%, avg R > 0, cum R >= 0. "
    "B: 30+, low bound >= 50%. C: 30+, win rate >= 50%. D: 30+, below 50%. "
    "New: under 30. "
    "Swing wins are wins vs SPY (the pick's 5-session side return beat SPY's) "
    "when 30+ picks have one, else the plain win with 'tape: unknown'. "
    "Day trade wins are +1R before -1R (the 1:1 bracket). "
    "2R grade: +2R before -1R on the same ladder, avg R = 3p - 1."
)

#: The swing tape horizon: the tracker's 5-session favorable-direction question.
TAPE_HORIZON_SESSIONS = 5
TAPE_OUTCOME_KIND = "favorable_direction_session_v2"
TAPE_UNKNOWN = "tape: unknown"
#: Said after n when some picks have no tape result yet (their 5 sessions are not done).
TAPE_N_NOTE = " (tape n excludes the newest ~5 sessions)"


def wilson_lower_bound(wins: int, n: int) -> float | None:
    """The desk's ONE Wilson (95%, `swing_headline`)."""
    from swing_headline import wilson_lower_bound as _wilson

    return _wilson(int(wins), int(n)) if n else None


def grade_for(
    *,
    n: int,
    sessions: int,
    wins: int,
    avg_r: float | None,
    cum_r_lately: float | None = None,
    tape: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ladder, once. Returns the grade with the numbers it was read from.

    ``tape`` (swing only) is ``{"wins", "n", "sessions"}`` of wins vs SPY; when
    its n meets the floor the ladder reads it instead of the plain win. PROVEN
    and A need ``cum_r_lately >= 0``; None (unknown) blocks them.
    """
    n = int(n or 0)
    wins = int(wins or 0)
    sessions = int(sessions or 0)
    win_rate = (wins / n) if n else None
    low = wilson_lower_bound(wins, n)
    positive = avg_r is not None and float(avg_r) > 0
    cum_r = _float(cum_r_lately)
    cum_ok = cum_r is not None and cum_r >= 0
    extra: dict[str, Any] = {"cum_r_lately": cum_r}
    ladder_sessions, ladder_rate, ladder_low, ladder_n = sessions, win_rate, low, n
    if tape is not None:
        tape_n = int(tape.get("n") or 0)
        tape_wins = int(tape.get("wins") or 0)
        tape_sessions = int(tape.get("sessions") or 0)
        tape_rate = (tape_wins / tape_n) if tape_n else None
        tape_low = wilson_lower_bound(tape_wins, tape_n)
        on_tape = tape_n >= MIN_N
        extra.update(
            tape_n=tape_n,
            tape_wins=tape_wins,
            tape_sessions=tape_sessions,
            tape_win_rate=tape_rate,
            tape_low_bound=tape_low,
            tape_unknown=int(tape.get("unknown") or 0),
            grade_basis="tape" if on_tape else "plain",
            tape_note="" if on_tape else TAPE_UNKNOWN,
        )
        if on_tape:
            ladder_sessions, ladder_rate, ladder_low, ladder_n = (
                tape_sessions, tape_rate, tape_low, tape_n
            )
    if n < MIN_N:
        grade = NEW
    elif (
        ladder_n >= PROVEN_MIN_N
        and ladder_sessions >= PROVEN_MIN_SESSIONS
        and (ladder_low or 0) >= PROVEN_MIN_LOW_BOUND
        and positive
        and cum_ok
    ):
        grade = PROVEN
    elif (
        ladder_sessions >= A_MIN_SESSIONS
        and (ladder_low or 0) >= A_MIN_LOW_BOUND
        and positive
        and cum_ok
    ):
        grade = A
    elif (ladder_low or 0) >= B_MIN_LOW_BOUND:
        grade = B
    elif (ladder_rate or 0) >= C_MIN_WIN_RATE:
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
        **extra,
    }


def ladder_low_bound(cell: Mapping[str, Any] | None) -> float:
    """The low bound the grade was read from: the tape's when it was used."""
    cell = cell or {}
    key = "tape_low_bound" if cell.get("grade_basis") == "tape" else "low_bound"
    return float(cell.get(key) or 0.0)


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


WIN, LOSS, UNDECIDED, OPEN, UNKNOWN = "win", "loss", "undecided", "open", "unknown"


def tape_result(
    pick: Mapping[str, Any],
    horizon_row: Mapping[str, Any] | None,
    spy_closes: Mapping[str, float] | None,
    *,
    as_of: str = "",
) -> str:
    """WIN / LOSS vs SPY over the pick's 5 sessions, or UNKNOWN.

    Long wins when its return beats SPY's; short wins when its short return
    (-ret) beats SPY's short return (-spy_ret). ``side_return_pct`` is already
    side-adjusted. A missing row, close or measurement, or a target after
    ``as_of``, is UNKNOWN - never a win or a loss. SPY is read only on the
    entry session (the scan date) and the target session.
    """
    row = horizon_row or {}
    if str(row.get("measured") or "").strip().lower() != "true":
        return UNKNOWN
    if str(row.get("maturity") or "").strip().lower() != "mature":
        return UNKNOWN
    side_return = _float(row.get("side_return_pct"))
    entry_day = str(row.get("scan_date") or "").strip()[:10]
    target_day = str(row.get("target_session") or "").strip()[:10]
    if side_return is None or not entry_day or not target_day:
        return UNKNOWN
    if as_of and target_day > str(as_of)[:10]:
        return UNKNOWN
    closes = spy_closes or {}
    spy_entry, spy_target = _float(closes.get(entry_day)), _float(closes.get(target_day))
    if spy_entry is None or spy_entry <= 0 or spy_target is None:
        return UNKNOWN
    spy_return = (spy_target / spy_entry - 1.0) * 100.0
    side = str(pick.get("side") or row.get("side") or "").strip().upper()
    if side not in {"LONG", "SHORT"}:
        return UNKNOWN
    spy_side_return = spy_return if side == "LONG" else -spy_return
    return WIN if side_return > spy_side_return else LOSS


def horizon_index(
    horizon_rows: Iterable[Mapping[str, Any]] | None,
    *,
    horizon: int = TAPE_HORIZON_SESSIONS,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """``{(SYMBOL, SIDE, scan_date): row}`` for the 5-session v2 rows."""
    index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in horizon_rows or ():
        if str(row.get("outcome_kind") or "").strip() != TAPE_OUTCOME_KIND:
            continue
        if _int(row.get("horizon_sessions")) != int(horizon):
            continue
        key = (
            str(row.get("symbol") or "").strip().upper(),
            str(row.get("side") or "").strip().upper(),
            str(row.get("scan_date") or "").strip()[:10],
        )
        index[key] = row
    return index


def swing_tape_stats(
    picks: Iterable[Mapping[str, Any]] | None,
    horizon_rows: Any,
    spy_closes: Mapping[str, float] | None,
    *,
    as_of: str = "",
) -> dict[str, dict[str, Any]]:
    """Per swing key: wins vs SPY, their n and sessions, and the cum R.

    ``picks`` are the tracker's selected episodes in the window
    (`looking_back.swing_pick_results`: symbol, session = scan date, side,
    bucket, family, r). ``horizon_rows`` is the horizon file's rows or a
    `horizon_index`. ``cum_r_lately`` sums each closed pick's own R; None when
    the key has no closed pick.
    """
    index = horizon_rows if isinstance(horizon_rows, Mapping) else horizon_index(horizon_rows)
    stats: dict[str, dict[str, Any]] = {}
    for pick in picks or ():
        side = str(pick.get("side") or "").strip().upper()
        key = swing_key(side, pick.get("bucket"), pick.get("family"))
        cell = stats.setdefault(
            key, {"wins": 0, "n": 0, "unknown": 0, "_sessions": set(), "_r": []}
        )
        r = _float(pick.get("r"))
        if r is not None:
            cell["_r"].append(r)
        scan_day = str(pick.get("session") or "").strip()[:10]
        row = index.get((str(pick.get("symbol") or "").strip().upper(), side, scan_day))
        outcome = tape_result(pick, row, spy_closes, as_of=as_of)
        if outcome == UNKNOWN:
            cell["unknown"] += 1
            continue
        cell["n"] += 1
        cell["wins"] += 1 if outcome == WIN else 0
        cell["_sessions"].add(scan_day)
    return {
        key: {
            "wins": cell["wins"],
            "n": cell["n"],
            "unknown": cell["unknown"],
            "sessions": len(cell["_sessions"]),
            "cum_r_lately": round(sum(cell["_r"]), 6) if cell["_r"] else None,
        }
        for key, cell in stats.items()
    }


def swing_cells(
    recent_rows: Iterable[Mapping[str, Any]],
    tape_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One graded cell per live tracker family row.

    Wins, losses and flats are COUNTS (never a recency-weighted rate), and a
    flat is not a win. Avg R is the representative (primary-stop) closed R,
    falling back to the cross-variant mean - the same preference as the tracker.
    ``tape_stats`` is `swing_tape_stats` over the same window; a key it lacks
    (or None) has its tape and cum R unknown.
    """
    tape_stats = tape_stats or {}
    cells = []
    for row in recent_rows or ():
        if str(row.get("namespace") or "live").strip().lower() != "live":
            continue
        wins = _int(row.get("n_wins"))
        n = wins + _int(row.get("n_losses")) + _int(row.get("n_flats"))
        avg_r = _float(row.get("representative_closed_r"))
        if avg_r is None:
            avg_r = _float(row.get("avg_closed_r"))
        tape = tape_stats.get(
            swing_key(row.get("side"), row.get("priority_bucket"), row.get("setup_family"))
        ) or {}
        cell = grade_for(
            n=n,
            sessions=_int(row.get("n_entry_sessions")),
            wins=wins,
            avg_r=avg_r,
            cum_r_lately=tape.get("cum_r_lately"),
            tape=tape or {"wins": 0, "n": 0, "sessions": 0},
        )
        cell.update(
            side=str(row.get("side") or "").strip().upper(),
            bucket=str(row.get("priority_bucket") or "").strip(),
            family=str(row.get("setup_family") or "general").strip() or "general",
        )
        cell["key"] = swing_key(cell["side"], cell["bucket"], cell["family"])
        cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# side by tape (S5): one display line, never a gate
# ---------------------------------------------------------------------------

#: How many scan sessions the side-by-tape line looks back over.
SIDE_BY_TAPE_SESSIONS = 20
SIDE_BY_TAPE_UNKNOWN = (
    "Side by tape: unknown (no measured 5-session results vs SPY yet)."
)


def _spy_side_return(
    row: Mapping[str, Any], side: str, spy_closes: Mapping[str, float] | None
) -> float | None:
    """SPY's same-side return over the row's entry -> target sessions, or None."""
    closes = spy_closes or {}
    entry = _float(closes.get(str(row.get("scan_date") or "").strip()[:10]))
    target = _float(closes.get(str(row.get("target_session") or "").strip()[:10]))
    if entry is None or entry <= 0 or target is None:
        return None
    spy_return = (target / entry - 1.0) * 100.0
    return spy_return if side == "LONG" else -spy_return


def side_by_tape(
    horizon_rows: Any,
    spy_closes: Mapping[str, float] | None,
    *,
    as_of: str = "",
    sessions: int = SIDE_BY_TAPE_SESSIONS,
) -> dict[str, Any]:
    """Longs and shorts vs SPY over the last ``sessions`` scan sessions.

    Every 5-session horizon row whose `tape_result` is WIN or LOSS counts; an
    UNKNOWN row is left out. The window is the newest ``sessions`` scan dates
    that have at least one decided row. Per side: n, wins, the beat share and
    the mean excess (side return minus SPY's same-side return, in points).
    """
    index = horizon_rows if isinstance(horizon_rows, Mapping) else horizon_index(horizon_rows)
    decided: list[tuple[str, str, bool, float]] = []
    for (_symbol, side, scan_day), row in index.items():
        outcome = tape_result({"side": side}, row, spy_closes, as_of=as_of)
        if outcome == UNKNOWN:
            continue
        spy_side = _spy_side_return(row, side, spy_closes)
        side_return = _float(row.get("side_return_pct"))
        if spy_side is None or side_return is None:
            continue
        decided.append((scan_day, side, outcome == WIN, side_return - spy_side))
    days = sorted({day for day, *_ in decided})[-int(sessions):] if sessions > 0 else []
    kept = set(days)
    out: dict[str, Any] = {
        "sessions": len(days),
        "first": days[0] if days else "",
        "last": days[-1] if days else "",
        "as_of": str(as_of or "")[:10],
    }
    for side in ("LONG", "SHORT"):
        rows = [(won, excess) for day, s, won, excess in decided if s == side and day in kept]
        n = len(rows)
        wins = sum(1 for won, _ in rows if won)
        out[side.lower()] = {
            "n": n,
            "wins": wins,
            "beat_pct": (wins / n * 100.0) if n else None,
            "excess_pct": (sum(excess for _, excess in rows) / n) if n else None,
        }
    return out


def side_by_tape_line(summary: Mapping[str, Any] | None) -> str:
    """The one line, from a `side_by_tape` summary. Formatting only."""
    summary = summary or {}
    count = _int(summary.get("sessions"))
    if not count:
        return SIDE_BY_TAPE_UNKNOWN

    def part(side: str) -> str:
        cell = summary.get(side) or {}
        beat, excess = _float(cell.get("beat_pct")), _float(cell.get("excess_pct"))
        if excess is not None:
            excess = round(excess, 2) + 0.0  # never print "-0.00"
        if side == "long":
            if not _int(cell.get("n")) or beat is None or excess is None:
                return "longs vs SPY unknown"
            return f"longs beat SPY {beat:.0f}% (excess {excess:+.2f}%)"
        if not _int(cell.get("n")) or beat is None or excess is None:
            return "shorts unknown"
        return f"shorts {beat:.0f}% ({excess:+.2f}%)"

    long_n = _int((summary.get("long") or {}).get("n"))
    short_n = _int((summary.get("short") or {}).get("n"))
    return (
        f"Last {count} sessions, tape-relative: {part('long')}, {part('short')}. "
        f"n {long_n} long / {short_n} short, scan dates "
        f"{summary.get('first')} to {summary.get('last')}."
    )


# ---------------------------------------------------------------------------
# day trade: +1R before -1R
# ---------------------------------------------------------------------------


def _flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


#: `setup_scoreboard.RISK_FLOOR_PCT_OF_ENTRY`: a stop closer than this % of entry has no usable R.
EOD_RISK_FLOOR_PCT = 0.1


def _first_decisive(ordered: list, target_field: str) -> str | None:
    """WIN / LOSS on the first row where the target or the stop is set; stop wins a tie."""
    for _index, row in ordered:
        if _flag(row.get("stop_hit")):
            return LOSS
        if _flag(row.get(target_field)):
            return WIN
    return None


def _eod_r(final_row: Mapping[str, Any] | None) -> float | None:
    """The final row's ``close_r``, or None when it is blank, the old 0/entry
    sentinel (`setup_scoreboard.unsettled_close_mask`) or under the risk floor."""
    if not final_row:
        return None
    close_r = _float(final_row.get("close_r"))
    if close_r is None or close_r in (float("inf"), float("-inf")):
        return None
    entry = _float(final_row.get("entry_price"))
    if close_r == 0 and entry is not None and _float(final_row.get("eod_close")) == entry:
        return None
    risk = _float(final_row.get("risk_per_share"))
    if not entry or risk is None or abs(risk) / abs(entry) * 100.0 < EOD_RISK_FLOOR_PCT:
        return None
    return close_r


def bracket_results(outcome_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One result per alert: did it reach +1R before -1R? And +2R before -1R?

    The flags on the outcome log are cumulative, so the FIRST row (by bars
    elapsed) on which either is set decides. Both set first on the same row is a
    LOSS - the adverse extreme is assumed first when the bar cannot say.
    A finished alert also carries ``eod_r`` (its final row's close R, None
    when unsettled) and ``reached_2r`` (any row flags +2R); an open one has None.
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
        finals = [row for _i, row in ordered if str(row.get("event_type") or "").strip().lower() == "final"]
        finished = bool(finals)
        undecided = UNDECIDED if finished else OPEN
        outcome = _first_decisive(ordered, "target_1r_hit") or undecided
        outcome_2r = _first_decisive(ordered, "target_2r_hit") or undecided
        first = rows[0]
        results.append(
            {
                "event_id": event_id,
                "trade_date": str(first.get("trade_date") or "").strip(),
                "side": str(first.get("direction") or "").strip().upper(),
                "bounce_type": bounce_type_from_event_id(event_id),
                "result": outcome,
                "result_2r": outcome_2r,
                "eod_r": _eod_r(finals[-1]) if finished else None,
                "reached_2r": any(_flag(row.get("target_2r_hit")) for row in rows) if finished else None,
            }
        )
    return results


def daytrade_cells(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One graded cell per (bounce type, side).

    An alert carrying several bounce types counts under EACH of them, as the
    Daytrade Tracker counts it. The bracket trade's R is +1 or -1, so avg R is
    ``2 * win_rate - 1``: positive exactly when it wins more than it loses.

    S10c: beside it, ``*_2r`` fields grade +2R before -1R on the same ladder
    (R is +2 or -1, so avg R is ``3p - 1``); ``eod_r_mean`` / ``eod_r_median``
    over ``eod_n`` settled finals; ``reach_2r_rate`` over finished alerts. A
    result without those fields (an older caller) leaves them unmeasured.
    """
    from held_run_score import bounce_components

    tallies: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results or ():
        for component in bounce_components(result.get("bounce_type")) or ():
            key = (str(component), str(result.get("side") or "").upper())
            tally = tallies.setdefault(
                key,
                {
                    "wins": 0, "losses": 0, "undecided": 0, "open": 0, "sessions": set(),
                    "wins_2r": 0, "losses_2r": 0, "undecided_2r": 0, "sessions_2r": set(),
                    "eod": [], "reach_hits": 0, "reach_n": 0,
                },
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
            outcome_2r = result.get("result_2r")
            if outcome_2r == WIN:
                tally["wins_2r"] += 1
            elif outcome_2r == LOSS:
                tally["losses_2r"] += 1
            elif outcome_2r == UNDECIDED:
                tally["undecided_2r"] += 1
            if outcome_2r in (WIN, LOSS) and result.get("trade_date"):
                tally["sessions_2r"].add(result["trade_date"])
            eod_r = _float(result.get("eod_r"))
            if eod_r is not None:
                tally["eod"].append(eod_r)
            if result.get("reached_2r") is not None:
                tally["reach_n"] += 1
                tally["reach_hits"] += 1 if result.get("reached_2r") else 0
    cells = []
    for (bounce_type, side), tally in tallies.items():
        n = tally["wins"] + tally["losses"]
        avg_r = (2 * tally["wins"] / n - 1) if n else None
        cell = grade_for(
            n=n,
            sessions=len(tally["sessions"]),
            wins=tally["wins"],
            avg_r=avg_r,
            # Each bracket alert is +1R or -1R.
            cum_r_lately=float(tally["wins"] - tally["losses"]) if n else None,
        )
        cell.update(
            bounce_type=bounce_type,
            side=side,
            undecided=tally["undecided"],
            open=tally["open"],
            key=daytrade_key(bounce_type, side),
            **_second_grade(tally),
        )
        cells.append(cell)
    return cells


def _second_grade(tally: Mapping[str, Any]) -> dict[str, Any]:
    """The 2R grade, EOD close R and reach-2R fields of one tally (S10c)."""
    from statistics import median

    wins, losses = tally["wins_2r"], tally["losses_2r"]
    n = wins + losses
    graded = grade_for(
        n=n,
        sessions=len(tally["sessions_2r"]),
        wins=wins,
        avg_r=(3 * wins / n - 1) if n else None,
        # Each 2R bracket alert is +2R or -1R.
        cum_r_lately=float(2 * wins - losses) if n else None,
    )
    eod = tally["eod"]
    reach_n = tally["reach_n"]
    return {
        "grade_2r": graded["grade"],
        "n_2r": n,
        "wins_2r": wins,
        "sessions_2r": graded["sessions"],
        "win_rate_2r": graded["win_rate"],
        "low_bound_2r": graded["low_bound"],
        "avg_r_2r": graded["avg_r"],
        "cum_r_2r": graded["cum_r_lately"],
        "undecided_2r": tally["undecided_2r"],
        "eod_n": len(eod),
        "eod_r_mean": round(sum(eod) / len(eod), 6) if eod else None,
        "eod_r_median": round(float(median(eod)), 6) if eod else None,
        "reach_2r_hits": tally["reach_hits"],
        "reach_2r_n": reach_n,
        "reach_2r_rate": (tally["reach_hits"] / reach_n) if reach_n else None,
    }


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
    swing_tape: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "as_of": str(as_of or ""),
        "rules": RULES_TEXT,
        "swing": swing_cells(recent_rows or (), swing_tape),
        "daytrade": daytrade_cells(bracket_results(outcome_rows or ())),
    }


def _best_first(cells: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        cells,
        key=lambda cell: (
            sort_rank(cell.get("grade")),
            -ladder_low_bound(cell),
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
    return (sort_rank(cell.get("grade")), -ladder_low_bound(cell))


def daytrade_cell_for_alert(
    payload_lookup: Mapping[str, Mapping[str, Any]], bounce_type: Any, side: Any
) -> Mapping[str, Any] | None:
    """The graded cell of the BEST-graded bounce type the alert carries, or None."""
    from held_run_score import bounce_components

    best: Mapping[str, Any] | None = None
    for part in str(bounce_type or "").replace(";", "-").split("-"):
        for component in bounce_components(part) or ():
            cell = payload_lookup.get(daytrade_key(component, side))
            if not cell:
                continue
            grade = str(cell.get("grade") or NEW)
            if best is None or sort_rank(grade) < sort_rank(str(best.get("grade") or NEW)):
                best = cell
    return best


def daytrade_grade_for_alert(
    payload_lookup: Mapping[str, Mapping[str, Any]], bounce_type: Any, side: Any
) -> str:
    """The alert's grade: the BEST grade among the bounce types it carries.

    An alert whose types the log has never graded is New. A measured D stays D
    even though New sorts above it.
    """
    cell = daytrade_cell_for_alert(payload_lookup, bounce_type, side)
    return str(cell.get("grade") or NEW) if cell else NEW


def badge(grade: str | None) -> str:
    grade = str(grade or NEW)
    return "NEW" if grade == NEW else grade


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.0f}%"


def cell_line(cell: Mapping[str, Any] | None) -> str:
    """One line: ``PROVEN · win vs SPY 62% (low 55%) · cum R +4.1 · n 114``.

    A swing cell graded on the tape shows its win vs SPY and that n; one graded
    on the plain win adds ``tape: unknown``. A day-trade cell with the S10c
    fields reads ``1:1 C · 2R D · EOD +0.04R · n 427`` (n = 1:1 decided); one
    without them shows its bracket win. Unknown cum R prints as ``cum R unknown``.
    """
    if not cell:
        return badge(NEW)
    if "grade_2r" in cell:
        eod = _float(cell.get("eod_r_mean"))
        return " · ".join(
            (
                f"1:1 {badge(cell.get('grade'))}",
                f"2R {badge(cell.get('grade_2r'))}",
                f"EOD {eod:+.2f}R" if eod is not None else "EOD unknown",
                f"n {int(cell.get('n') or 0)}",
            )
        )
    parts = [badge(cell.get("grade"))]
    n = int(cell.get("n") or 0)
    if cell.get("grade_basis") == "tape":
        rate, low, label = cell.get("tape_win_rate"), cell.get("tape_low_bound"), "win vs SPY"
        n = int(cell.get("tape_n") or 0)
    else:
        rate, low, label = cell.get("win_rate"), cell.get("low_bound"), "win"
    if rate is not None:
        parts.append(f"{label} {_pct(rate)}" + (f" (low {_pct(low)})" if low is not None else ""))
    if cell.get("tape_note"):
        parts.append(str(cell["tape_note"]))
    cum_r = _float(cell.get("cum_r_lately"))
    parts.append(f"cum R {cum_r:+.1f}" if cum_r is not None else "cum R unknown")
    parts.append(f"n {n}" + (TAPE_N_NOTE if int(cell.get("tape_unknown") or 0) > 0 else ""))
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# hold-out view (P2-9 9b) - read-only, beside the cells above
# ---------------------------------------------------------------------------

#: What a window with no cell for a key says.
NOT_IN_WINDOW = "none in this window"
#: What the prior column says when the same key has no prior cell.
NO_PRIOR = "no prior"
#: The tape and cum-R numbers each hold-out row carries per window.
HOLDOUT_FIELDS = ("tape_win_rate", "tape_low_bound", "tape_n", "cum_r_lately")


def holdout_text(cell: Mapping[str, Any] | None, *, missing: str = NOT_IN_WINDOW) -> str:
    """One graded cell as the hold-out column prints it. Under the floor says so."""
    if not cell:
        return missing
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
    if cell.get("grade_basis") == "tape" and cell.get("tape_win_rate") is not None:
        tape_low = cell.get("tape_low_bound")
        parts.append(
            f"vs SPY {float(cell['tape_win_rate']):.0%}"
            + (f" (low {float(tape_low):.0%})" if tape_low is not None else "")
            + f" n={int(cell.get('tape_n') or 0)}"
        )
    elif cell.get("tape_note"):
        parts.append(str(cell["tape_note"]))
    cum_r = _float(cell.get("cum_r_lately"))
    if cum_r is not None:
        parts.append(f"cum {cum_r:+.1f}R")
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
            "prior_text": holdout_text(prior.get(key), missing=NO_PRIOR),
            **{
                f"{window}_{field}": (cells.get(key) or {}).get(field)
                for window, cells in (("recent", recent), ("prior", prior))
                for field in HOLDOUT_FIELDS
            },
        }
        for key in keys
    ]
