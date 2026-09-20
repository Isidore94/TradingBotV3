"""The prediction ledger's reader - skill against three naive rules (TJ-16).

`plan.md` §12.4 TJ-16 item 2: *"Skill against naive baselines, never a bare hit
rate. Every accuracy cell is shown beside what `always Up`, `same as the last
hour` and `with the D1 environment` would have scored on the SAME stamps ...
**Calibration:** accuracy by `How sure` - High must beat Low or the page says it
does not."*

A bare hit rate is not a skill number. A trader who is right 60% of the time on
a tape where `always Up` scored 60% has shown nothing, and a page that printed
the 60% alone would be flattering them with the market's own drift. So every
cell here is printed beside what the three naive rules scored on the SAME forty
stamps, answering the same questions at the same moments.

Six rules hold it to evidence rather than to encouragement.

1. **The CURRENT row per read, never both.** A matured grade is a NEW row naming
   the old one (TJ-10), so a reader that counted the file would report two calls
   where the trader made one. `market_read_grades.current_grades` is the ONE
   place that decides; a superseded row stays on disk and is hidden.
2. **A stated call and an inferred stance are never pooled** (decision 0021
   answer 29). :func:`read_ledger` can be asked for either or for both, and
   :func:`build_readout` RAISES `market_read_grades.PoolingError` on a mix
   rather than printing a rate that describes neither.
3. **The two horizons are never pooled.** `Rest of day` and `Next 5 sessions`
   answer different questions; one number over both answers neither.
4. **The band is CALLED, never copied.** A stored verdict is the desk's own
   measurement and is never re-measured here. A BASELINE has no stored verdict -
   nobody graded the naive rule on the night - so its verdict is measured NOW,
   through `market_read_grades._verdict_for`, the same rule and the same band
   the trader's own rows were graded with. A reader carrying its own
   `abs(move) <= 0.25` drifts the day the band moves; TJ-15 set the precedent
   with `real_miss.verdict`.
5. **A baseline that cannot answer is `unmeasured`, never wrong.** `compressed`
   is not a direction, so the D1-environment rule has no answer on those stamps
   and they LEAVE its fraction. Counting them as losses for the naive rule would
   flatter the trader with data nobody measured (plan.md sec 5).
6. **Nothing measured is not a rate of zero.** An empty ledger reads `None` with
   a sentence saying there are no clicked calls yet - which is the state the
   desk is in on 2026-09-20, the day the click card shipped.

PURE except for the store functions, which are FILE READS meant for a worker:
they take a root, they never raise on a folder that does not exist, and today on
the live desk that folder does not exist. Nothing here calls a model, opens a
network, or reaches a detector, score, alert, watchlist, Focus, the review queue
or `review_policy.json`. Every number is REPORTED.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import market_read_grades as grades

_log = logging.getLogger(__name__)

#: The readout's own name. A later shape is a new version, never a re-reading.
SCHEMA = "prediction_readout_v1"

#: The two horizons, in the order they are printed, and they are never pooled.
HORIZONS = ("rest_of_day", "next_5_sessions")

#: How a horizon is named in a sentence.
HORIZON_LABELS = {
    "rest_of_day": "Rest of day",
    "next_5_sessions": "Next 5 sessions",
}

#: What the naive rules are CALLED on a trader-facing line.
BASELINE_LABELS = {
    "always_up": "always Up",
    "same_as_the_last_hour": "same as the last hour",
    "with_the_d1_environment": "with the D1 environment",
}

#: The `How sure` buckets, in the order a calibration table prints them.
CONFIDENCE_ORDER = ("low", "medium", "high")

#: What an empty ledger says. The words are asserted by a test because they are
#: the first thing the trader will ever see on these surfaces.
EMPTY_STATEMENT = (
    "no clicked calls yet - this fills in once you answer a Mentor card with a "
    "direction, and nothing here is a rate of zero until then"
)

#: A baseline the naive rule could not answer. It is NOT a loss for the rule.
_BASELINE_UNMEASURED = f"{grades.UNMEASURED_PREFIX}:baseline_has_no_direction"
_BASELINE_NO_MOVE = f"{grades.UNMEASURED_PREFIX}:no_measured_move"


# ---------------------------------------------------------------------------
# the store read
# ---------------------------------------------------------------------------
def read_ledger(
    sessions: Iterable[Any], *, root: Any = None, source: str = ""
) -> list[dict[str, Any]]:
    """The CURRENT grade row per read, oldest session first. Never raises.

    ``source`` filters (`market_read_grades.SOURCE_CLICK` /
    ``SOURCE_EXTRACTED``); ``""`` is everything, which is exactly why
    :func:`build_readout` has to refuse a mix.

    A session whose file does not exist contributes nothing - on 2026-09-20 the
    live desk has no `reads/` folder at all, and a reader that raised on that
    would take the Day Review page down on the day the click card shipped.
    """
    wanted = str(source or "").strip()
    out: list[dict[str, Any]] = []
    for value in sessions or ():
        session = str(value or "")[:10]
        if not session:
            continue
        try:
            stored = grades.read_grades(session, root=root)
        except OSError:  # pragma: no cover - read_grades already swallows this
            _log.debug("The read ledger was unreadable for %s.", session)
            continue
        for row in grades.current_grades(stored):
            if wanted and str(row.get("source") or "") != wanted:
                continue
            out.append(dict(row))
    return out


def available_sessions(*, root: Any = None) -> list[str]:
    """Every session the read ledger holds a file for, oldest first.

    One directory listing, not a walk of the rows: the whole-ledger population
    ("all", as against "lately") is every session on disk, and a caller must be
    able to ask which those are without opening one of them.
    """
    folder = grades.reads_path("", root=root).parent
    try:
        files = sorted(folder.glob("*.jsonl"))
    except OSError:
        return []
    return [path.stem for path in files if path.stem]


# ---------------------------------------------------------------------------
# the baselines - the same stamps, answered by a naive rule
# ---------------------------------------------------------------------------
def _read_with_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """One grade row as the READ it graded, with its own context attached.

    `market_read_grades.baseline_reads` reads `context`, and the context lives
    on the GRADE rather than on the read inside it. Merging here is what makes
    the naive rule point-in-time by construction - it sees the snapshot taken at
    the stamp and no bar at all.
    """
    read = row.get("read")
    made = dict(read) if isinstance(read, Mapping) else {}
    made.setdefault("read_id", str(row.get("read_id") or ""))
    made.setdefault("entry_id", str(row.get("entry_id") or ""))
    made.setdefault("session", str(row.get("session") or ""))
    made.setdefault("horizon", str(row.get("horizon") or ""))
    made.setdefault("source", str(row.get("source") or ""))
    made.setdefault("direction", str(row.get("direction") or ""))
    context = row.get("context")
    made["context"] = dict(context) if isinstance(context, Mapping) else {}
    return made


def _baseline_verdict(row: Mapping[str, Any], mirrored: Mapping[str, Any]) -> str:
    """What the naive rule scored on ONE stamp.

    The trader's own verdict decides the STATE: a stamp whose horizon is still
    open is open for the baseline too, and one the desk could not measure is
    unmeasured for both. Only a CLOSED stamp is judged, and it is judged with
    the grader's own band through its module attribute - never a copy.
    """
    verdict = str(row.get("verdict") or "")
    if verdict.startswith(grades.PENDING_PREFIX):
        return verdict
    if verdict not in (grades.VERDICT_RIGHT, grades.VERDICT_WRONG, grades.VERDICT_FLAT):
        return verdict or _BASELINE_UNMEASURED
    direction = str(mirrored.get("direction") or "")
    if direction not in grades.GRADABLE_DIRECTIONS:
        # `compressed` is not a direction: the rule has no answer here and these
        # stamps leave its fraction rather than counting against it.
        return _BASELINE_UNMEASURED
    try:
        move_atr = float(row.get("move_atr"))
    except (TypeError, ValueError):
        return _BASELINE_NO_MOVE
    if move_atr != move_atr:  # NaN
        return _BASELINE_NO_MOVE
    # THE ONE BAND, CALLED through the module attribute (TJ-15's precedent).
    return grades._verdict_for(direction, move_atr)


def baseline_cell(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    """One naive rule's accuracy cell over the SAME stamps, in the same order."""
    reads = [_read_with_context(row) for row in rows]
    mirrored = grades.baseline_reads(reads, name)
    scored = [
        {"verdict": _baseline_verdict(row, mirror)}
        for row, mirror in zip(rows, mirrored)
    ]
    cell = grades.accuracy(scored)
    cell["baseline"] = str(name)
    cell["label"] = BASELINE_LABELS.get(str(name), str(name))
    return cell


def _baselines(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: baseline_cell(rows, name) for name in grades.BASELINES}


# ---------------------------------------------------------------------------
# calibration - accuracy by `How sure`
# ---------------------------------------------------------------------------
def _confidence_of(row: Mapping[str, Any]) -> str:
    read = row.get("read")
    if isinstance(read, Mapping):
        stated = str(read.get("confidence") or "").strip()
        if stated:
            return stated
    context = row.get("context")
    if isinstance(context, Mapping):
        stated = str(context.get("confidence") or "").strip()
        if stated and stated != grades.UNMEASURED:
            return stated
    return ""


def _calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accuracy per `How sure`, and an honest sentence about High against Low.

    plan.md TJ-16 item 2: *"High must beat Low or the page says it does not"*.
    The comparison is only made when BOTH buckets cleared
    `evidence_stats.MIN_REPORTABLE_N` and both have a measured rate; otherwise
    `high_beats_low` is ``None`` and the sentence says the question is not
    answerable yet, which is not the same as "no".
    """
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        level = _confidence_of(row)
        if not level:
            continue
        buckets.setdefault(level, []).append(row)

    def _order(level: str) -> tuple[int, str]:
        try:
            return (CONFIDENCE_ORDER.index(level), level)
        except ValueError:
            return (len(CONFIDENCE_ORDER), level)

    cells: list[dict[str, Any]] = []
    for level in sorted(buckets, key=_order):
        cell = grades.accuracy(buckets[level])
        cell["confidence"] = level
        cells.append(cell)

    by_level = {str(cell["confidence"]): cell for cell in cells}
    low = by_level.get("low")
    high = by_level.get("high")
    comparable = (
        low is not None
        and high is not None
        and low.get("rate") is not None
        and high.get("rate") is not None
        and bool(low.get("meets_floor"))
        and bool(high.get("meets_floor"))
    )
    if not comparable:
        beats: bool | None = None
        statement = (
            "not answerable yet: `How sure` is only compared once BOTH the High "
            "and the Low bucket have measured calls over the reporting floor"
        )
    else:
        beats = bool(high["rate"] > low["rate"])
        if beats:
            statement = (
                f"High beat Low: {_percent(high['rate'])} right when you were sure "
                f"(n {high['n']}) against {_percent(low['rate'])} when you were not "
                f"(n {low['n']}) - observational, and neither is a promise"
            )
        else:
            statement = (
                f"High did not beat Low: {_percent(high['rate'])} right when you "
                f"were sure (n {high['n']}) against {_percent(low['rate'])} when "
                f"you were not (n {low['n']}) - the confidence you report is not "
                "tracking the calls you get right"
            )
    return {"cells": cells, "high_beats_low": beats, "statement": statement}


def _percent(rate: Any) -> str:
    try:
        return f"{float(rate) * 100:.0f}%"
    except (TypeError, ValueError):
        return "unmeasured"


# ---------------------------------------------------------------------------
# the readout
# ---------------------------------------------------------------------------
def horizon_of(row: Mapping[str, Any]) -> str:
    """Which horizon a grade row answers, or ``""``. The ONE mapping.

    Read off the grade and then off the read inside it, because the two are
    written by the same function and a reader that guessed from the timeframe
    would answer differently for a D1 row with a rest-of-day call on it.
    """
    horizon = str(row.get("horizon") or "")
    if horizon in HORIZONS:
        return horizon
    read = row.get("read")
    if isinstance(read, Mapping):
        horizon = str(read.get("horizon") or "")
        if horizon in HORIZONS:
            return horizon
    return ""


def _horizon_statement(name: str, cell: Mapping[str, Any], baselines: Mapping[str, Any]) -> str:
    label = HORIZON_LABELS.get(name, name)
    if not cell["n"]:
        pending = int(cell.get("pending") or 0)
        waiting = f", {pending} still open" if pending else ""
        return f"{label}: nothing measured yet{waiting}"
    parts = [f"{label}: right {_percent(cell['rate'])} (n {cell['n']})"]
    for rule in grades.BASELINES:
        naive = baselines.get(rule) or {}
        parts.append(
            f"{BASELINE_LABELS.get(rule, rule)} {_percent(naive.get('rate'))} "
            f"(n {naive.get('n', 0)})"
        )
    tail = ""
    if cell.get("pending"):
        tail = f"; {cell['pending']} still open and in neither half"
    if not cell.get("meets_floor"):
        tail += "; too few to call"
    return "; ".join(parts) + tail


def build_readout(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The whole readout for one population of grade rows. PURE.

    Raises `market_read_grades.PoolingError` when the rows do not share one
    ``source`` (decision 0021 answer 29).
    """
    listed = [dict(row) for row in rows or ()]
    sources = {str(row.get("source") or "") for row in listed if row.get("source")}
    if len(sources) > 1:
        raise grades.PoolingError(
            "a clicked read and an extracted one are different evidence and are "
            f"never pooled in one readout: {sorted(sources)}"
        )
    by_horizon: dict[str, list[dict[str, Any]]] = {name: [] for name in HORIZONS}
    for row in listed:
        name = horizon_of(row)
        if name:
            by_horizon[name].append(row)

    horizons: dict[str, Any] = {}
    for name in HORIZONS:
        mine = by_horizon[name]
        cell = grades.accuracy(mine)
        baselines = _baselines(mine)
        horizons[name] = {
            "horizon": name,
            "label": HORIZON_LABELS.get(name, name),
            "accuracy": cell,
            "baselines": baselines,
            "calibration": _calibration(mine),
            "statement": _horizon_statement(name, cell, baselines),
        }

    empty = not listed
    sessions = sorted({str(row.get("session") or "")[:10] for row in listed if row.get("session")})
    if empty:
        statement = EMPTY_STATEMENT
    else:
        statement = (
            f"{len(listed)} read(s) over {len(sessions)} session(s); every accuracy "
            "cell is printed beside what the three naive rules scored on the SAME "
            "stamps, and the two horizons are never pooled. Observational: a "
            "baseline is a comparison, not a competitor."
        )
    return {
        "schema": SCHEMA,
        "source": next(iter(sources), ""),
        "empty": empty,
        "statement": statement,
        "sessions": sessions,
        "horizons": horizons,
    }


# ---------------------------------------------------------------------------
# Day Review's "Your reads" line (TJ-12 ships the page; this is its reader)
# ---------------------------------------------------------------------------
def your_reads(
    session: str, *, root: Any = None, source: str = grades.SOURCE_CLICK
) -> dict[str, Any]:
    """One session's tally beside its baselines, as NUMBERS.

    A FILE READ meant for a worker: it takes a root, it never raises on a folder
    that does not exist, and the counts travel as integers so the page never
    re-derives them on the Qt thread.
    """
    day = str(session or "")[:10]
    rows = read_ledger([day], root=root, source=source) if day else []
    cell = grades.accuracy(rows)
    baselines = _baselines(rows)
    if not rows:
        text = EMPTY_STATEMENT
    else:
        best = max(
            grades.BASELINES,
            key=lambda name: (baselines[name].get("right") or 0, name),
        )
        naive = baselines[best]
        pending = f", {cell['pending']} still open" if cell.get("pending") else ""
        text = (
            f"Your reads: {cell['right']} right of {cell['n']}{pending} - "
            f"{BASELINE_LABELS.get(best, best)} scored "
            f"{naive.get('right', 0)} of {naive.get('n', 0)} on the same stamps"
        )
    return {
        "text": text,
        "session": day,
        "n": int(cell["n"]),
        "right": int(cell["right"]),
        "wrong": int(cell["wrong"]),
        "flat": int(cell["flat"]),
        "pending": int(cell["pending"]),
        "empty": not rows,
        "baselines": baselines,
    }


__all__ = [
    "BASELINE_LABELS",
    "CONFIDENCE_ORDER",
    "EMPTY_STATEMENT",
    "HORIZONS",
    "HORIZON_LABELS",
    "SCHEMA",
    "available_sessions",
    "baseline_cell",
    "build_readout",
    "horizon_of",
    "read_ledger",
    "your_reads",
]
