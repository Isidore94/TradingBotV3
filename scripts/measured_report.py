"""One shared MEASURED report - WISHLIST 10K steps 2 and 5, packet WS-RP.

WISHLIST 10K asks one numerical contract with **five separate answers**, frozen
before the results are read:

1. **Total profit** - actual closed-trade broker net P&L, counted ONCE per
   ``trade_id`` through :func:`preference_trade_outcomes.trade_level_summary`,
   split the way the LEGS say (`journal_exposure`: a long put is bearish money
   however its ``direction`` column is spelled).
2. **Biggest opportunity** - the best available movement AFTER the observation,
   side-adjusted, with the adverse side beside it. ``mfe_r`` only where a stop
   was known; where it was not, the R is UNKNOWN and the percentage is still
   measured. Those are different facts and neither is a zero.
3. **Quickest result** - time to the recipe's first target in TRADING minutes,
   with the hit / unhit / pending / unknown counts beside the median so the
   denominator is visible. ``time_to_mfe_min`` is a separate hindsight fact and
   never the speed answer.
4. **End of day** - the side-adjusted mark at the stated session close, naming
   the clock and the exit convention. An entry AT the close has no same-session
   forward measure at all, so that cell is ``unavailable``, never ``0.00``.
5. **Last day or two** - two INDEPENDENT controls: which observation sessions
   were selected, and how long they were followed. Both are counted in exchange
   sessions through `market_calendar`, never in calendar days.

**Pure.** Nothing here writes, fetches, calls a model or opens a socket.
`ai_jobs.measured_report_publish` publishes it; `ui/panels/daily_recap_panel.py`
renders it; both read the SAME cells, so the page and the export can never
disagree.

**One report id.** ``report_id`` is a sha1 over the sorted ``cell_id = value =
state`` lines plus ``as_of`` alone, and ``as_of`` is the SESSION the evidence
belongs to rather than a wall clock - so a timer tick cannot move the id and a
cell that matured must (ST6's rule for `working_lately.snapshot_id`, same
reason).

**Every number is somebody else's.** The statistics come from
`evidence_stats`; the money comes from `trade_level_summary`; the bias split
comes from `journal_exposure`; the sessions come from `market_calendar`. This
module computes no new statistic, and an extremum it reports ("the biggest
move") is a SELECTED observation, labelled ``retrospective, result-selected``
wherever it is shown beside its full denominator.

**Nothing here reaches a detector, a score, an alert, a watchlist, Focus, the
review queue or `review_policy.json`.** It is a reader.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import evidence_stats
import market_calendar

_log = logging.getLogger(__name__)

#: The contract's own version. It travels on every cell, because a number
#: without the version of the rule that produced it cannot be compared with the
#: same number read next month.
REPORT_VERSION = "measured_report_v1"

#: The six sections WISHLIST 10K names, in its order. `missing_evidence` is a
#: VIEW over the others - every cell that is not `measured` - so a reader never
#: has to hunt for what was not answered.
SECTION_NAMES: tuple[str, ...] = (
    "market_thoughts",
    "measured_context",
    "opportunity_results",
    "preference_decisions",
    "actual_trades",
    "missing_evidence",
)

STATE_MEASURED = "measured"
STATE_PENDING = "pending"
STATE_UNKNOWN = "unknown"

#: The sentence beside an entry taken AT the close. There is no bar after the
#: close in that session, so "how did it end the day" has no answer for it -
#: which is not the same answer as "it ended flat".
UNAVAILABLE_ENTRY_AT_CLOSE = (
    "entered at the session close: there is no same-session bar after it, so "
    "the end-of-day mark is not measurable for this observation"
)

#: 32 KiB of UTF-8, WISHLIST 10K's proposed headline budget for the brief a
#: frontier model is handed. The brief states what it dropped.
HANDOFF_MARKDOWN_CAP_BYTES = 32 * 1024

#: The follow-through horizons the D1 scan publishes: `SCAN_FACTOR_HORIZONS` is
#: ``(1, 3, 5, 10)`` (`master_avwap_lib/legacy.py:11079`, read 2026-09-13).
#: QUOTED, never imported - `legacy.py` is an ask-first file and this module
#: only measures. A two-session follow-through therefore has no row on this
#: desk at all, and the cell for it says so instead of falling back to 1 or 3.
PUBLISHED_FOLLOW_THROUGH_HORIZONS: tuple[int, ...] = (1, 3, 5, 10)

#: The default two controls: the last two sessions, followed one session on.
#: "Last day or two", the way the trader asked it.
DEFAULT_SELECTION_SESSIONS = 2
DEFAULT_FOLLOW_THROUGH_SESSIONS = 1

#: Regular open. `market_calendar` publishes `session_close` (which handles an
#: early close) but no open, so the one constant lives here and is used only to
#: bound a trading-minute count.
_SESSION_OPEN = time(9, 30)

#: What a best/worst table IS. Printed on every one of them: picking the three
#: biggest moves out of a measured set is a retrospective view of that set, not
#: an estimate of anything.
SELECTED_TABLE_LABEL = "retrospective, result-selected"

_CLOCK_EXCHANGE = "America/New_York exchange sessions"
_CLOCK_SESSION_CLOSE = "America/New_York session close (16:00, or the early close)"
_CLOCK_BROKER = (
    "no clock: a broker file is authoritative for money and blind to time, so "
    "the money is attributed by trade_id and never by a timestamp"
)


# ---------------------------------------------------------------------------
# the cell and the report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One numeric answer, with everything a reader needs to trust it.

    `value` is ``None`` whenever `state` is not ``measured``; a cell that could
    not be measured carries the SENTENCE in `unavailable` rather than a zero.
    """

    cell_id: str
    metric: str
    unit: str
    value: float | int | None
    n: int = 0
    distinct_sessions: int = 0
    distinct_symbols: int = 0
    population: str = ""
    window: tuple[str, str] = ("", "")
    reference_clock: str = ""
    exit_policy: str = ""
    version: str = REPORT_VERSION
    state: str = STATE_MEASURED
    sources: tuple[str, ...] = ()
    unavailable: str = ""
    section: str = "measured_context"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "metric": self.metric,
            "unit": self.unit,
            "value": self.value,
            "n": self.n,
            "distinct_sessions": self.distinct_sessions,
            "distinct_symbols": self.distinct_symbols,
            "population": self.population,
            "window": list(self.window),
            "reference_clock": self.reference_clock,
            "exit_policy": self.exit_policy,
            "version": self.version,
            "state": self.state,
            "sources": list(self.sources),
            "unavailable": self.unavailable,
            "section": self.section,
        }


@dataclass(frozen=True)
class MeasuredReport:
    """The whole contract for one session, computed once and read twice."""

    session_date: str
    as_of: str
    report_id: str
    sections: dict[str, tuple[Cell, ...]]
    generated_at: str
    tracker_snapshot_id: str = ""
    open_theses: tuple[dict[str, Any], ...] = ()
    source_paths: tuple[str, ...] = ()
    policy: tuple[str, ...] = ()
    example_tables: tuple[dict[str, Any], ...] = ()
    _cells: tuple[Cell, ...] = field(default_factory=tuple, repr=False)

    def cells(self) -> tuple[Cell, ...]:
        """Every cell ONCE, in report order. `missing_evidence` is a view."""
        return self._cells

    def cell(self, cell_id: str) -> Cell:
        for cell in self._cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(cell_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_VERSION,
            "session_date": self.session_date,
            "as_of": self.as_of,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "tracker_snapshot_id": self.tracker_snapshot_id,
            "open_theses": [dict(row) for row in self.open_theses],
            "source_paths": list(self.source_paths),
            "policy": list(self.policy),
            "example_tables": [dict(table) for table in self.example_tables],
            "cells": [cell.as_dict() for cell in self._cells],
            "sections": {
                name: [cell.cell_id for cell in rows]
                for name, rows in self.sections.items()
            },
            "note": (
                "Measured by code. No model was called, nothing here ranks, "
                "scores, gates or alerts, and every cell names the file its "
                "number came from."
            ),
        }


def compute_report_id(cells: Sequence[Cell], as_of: str) -> str:
    """sha1 over the sorted cells and the as-of ALONE.

    Not the clock, not the source paths, not the generation time: two runs over
    the same evidence must produce the same id, and a cell that matured must
    produce a different one.
    """
    digest = hashlib.sha1()
    for line in sorted(f"{cell.cell_id}={cell.value!r}={cell.state}" for cell in cells):
        digest.update(line.encode("utf-8"))
        digest.update(b"\x1e")
    digest.update(str(as_of).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportSources:
    """Every store the report reads, by path. Nothing is discovered."""

    intraday_outcomes: Path | None = None
    session_horizon_outcomes: Path | None = None
    preference_report: Path | None = None
    journal_trades: Path | None = None
    working_lately: Path | None = None
    market_theses: Path | None = None

    @classmethod
    def from_project_paths(cls) -> "ReportSources":
        """The desk's own stores, addressed by their named constants."""
        import project_paths

        # The Working-lately snapshot, at the per-machine path
        # `working_lately_service.default_store_dir()` publishes it to. Spelled
        # here rather than imported, because that module is a QObject service
        # and this one is read by a headless overnight slot.
        working_lately = (
            Path(project_paths.LOCAL_SETTINGS_DIR) / "working_lately" / "snapshot_latest.json"
        )
        return cls(
            intraday_outcomes=project_paths.INTRADAY_BOUNCE_OUTCOMES_FILE,
            session_horizon_outcomes=(
                project_paths.MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE
            ),
            preference_report=_preference_report_path(),
            journal_trades=None,  # the journal is a database; read through its store
            working_lately=working_lately,
            market_theses=getattr(project_paths, "MARKET_THESES_FILE", None),
        )

    def existing(self) -> tuple[str, ...]:
        """Only the paths that are really there. A citation is a promise."""
        out: list[str] = []
        for path in (
            self.intraday_outcomes,
            self.session_horizon_outcomes,
            self.preference_report,
            self.journal_trades,
            self.working_lately,
            self.market_theses,
        ):
            if path and Path(path).exists():
                out.append(str(path))
        return tuple(out)


def _preference_report_path() -> Path | None:
    try:
        import preference_trade_outcomes

        return preference_trade_outcomes.REPORT_FILE
    except Exception:  # noqa: BLE001 - a missing reader is a missing source
        return None


def _cite(*paths: Any) -> tuple[str, ...]:
    """The subset of `paths` that exists, as strings. Never a promise."""
    out: list[str] = []
    for path in paths:
        if not path:
            continue
        try:
            if Path(path).exists():
                out.append(str(path))
        except OSError:  # pragma: no cover - an unreadable name is not a source
            continue
    return tuple(out)


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """A float, or None. An empty column is silence, never a zero."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _rows_from_csv(path: Any) -> list[dict[str, str]]:
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rows_from_jsonl(path: Any) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, Mapping):
            out.append(dict(row))
    return out


def _read_journal_trades(path: Any) -> list[dict[str, Any]]:
    """The journal's trades: a JSONL export when one is named, else the store."""
    if path and Path(path).exists():
        return _rows_from_jsonl(path)
    try:
        import journal_store

        return list(journal_store.JournalStore().list_trades())
    except Exception as exc:  # noqa: BLE001 - a closed journal is uncertainty
        _log.info("Measured report: the journal could not be read (%s).", exc)
        return []


def _summary(
    values: Sequence[float],
    *,
    symbols: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """`evidence_stats` is the only place a statistic is computed."""
    return evidence_stats.summarize(
        values, symbols=list(symbols or []), sessions=list(sessions or []), clip=None
    )


# ---------------------------------------------------------------------------
# trading minutes
# ---------------------------------------------------------------------------


def trading_minutes_between(start: datetime, end: datetime) -> float | None:
    """Minutes the market was OPEN between two aware moments.

    15:50 to 09:40 the next session is 20 trading minutes. A wall clock says
    1,070, and a speed answer computed on a wall clock is a different question
    answered by accident.
    """
    if start is None or end is None:
        return None
    try:
        first = start.astimezone(market_calendar.MARKET_TZ)
        last = end.astimezone(market_calendar.MARKET_TZ)
    except (TypeError, ValueError):  # pragma: no cover - a naive stamp is a defect
        return None
    if last <= first:
        return 0.0
    total = 0.0
    cursor = first.date()
    final = last.date()
    for _ in range(400):  # a target hit a year later is not a speed answer
        if cursor > final:
            break
        try:
            session = market_calendar.is_session(cursor)
        except Exception:  # noqa: BLE001 - outside the calendar's range
            return None
        if session:
            opened = datetime.combine(cursor, _SESSION_OPEN, tzinfo=market_calendar.MARKET_TZ)
            closed = market_calendar.session_close(cursor)
            low = max(opened, first) if cursor == first.date() else opened
            high = min(closed, last) if cursor == final else closed
            low = max(low, opened)
            high = min(high, closed)
            if high > low:
                total += (high - low).total_seconds() / 60.0
        cursor += timedelta(days=1)
    return round(total, 4)


def _aware(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=market_calendar.MARKET_TZ)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=market_calendar.MARKET_TZ)


def _selection_window(session_date: str, sessions: int) -> tuple[str, ...]:
    """The last `sessions` EXCHANGE sessions ending at `session_date`.

    Counted through `market_calendar`, so Labor Day and the weekend are not in
    it - a window in calendar days lands on a day the desk never scanned.
    """
    try:
        cursor = date.fromisoformat(str(session_date)[:10])
    except ValueError:
        return ()
    days: list[str] = [cursor.isoformat()]
    for _ in range(max(0, int(sessions) - 1)):
        try:
            cursor = market_calendar.previous_session(cursor)
        except Exception:  # noqa: BLE001 - a calendar refusal narrows the window
            break
        days.append(cursor.isoformat())
    return tuple(sorted(days))


# ---------------------------------------------------------------------------
# the five answers
# ---------------------------------------------------------------------------


def _money_cells(
    preference_rows: Sequence[Mapping[str, Any]],
    journal_trades: Sequence[Mapping[str, Any]],
    *,
    session_date: str,
    sources: tuple[str, ...],
) -> list[Cell]:
    """Answer 1. Money ONCE per trade, split the way the legs say."""
    import journal_exposure
    import preference_trade_outcomes

    window = (session_date, session_date)
    version = f"{REPORT_VERSION}+{preference_trade_outcomes.SCHEMA}"
    matched = [row for row in preference_rows if str(row.get("trade_id") or "").strip()]

    def _money(rows: Sequence[Mapping[str, Any]]) -> tuple[float, int]:
        summary = preference_trade_outcomes.trade_level_summary(rows)
        return float(summary["net_pnl"]), int(summary["n_trades_matched"])

    total, n_trades = _money(matched)
    cells = [
        Cell(
            cell_id="total_profit.all",
            metric="closed-trade net P&L, counted once per trade_id",
            unit="CAD",
            value=round(total, 4),
            n=n_trades,
            distinct_symbols=len({str(row.get("symbol") or "") for row in matched}),
            distinct_sessions=len({str(row.get("session_date") or "") for row in matched}),
            population=(
                "every trade a statement in the preference report matched, "
                "summed at TRADE grain (a trade discussed twice is one P&L)"
            ),
            window=window,
            reference_clock=_CLOCK_BROKER,
            exit_policy="broker close - the money is realized, not modelled",
            version=version,
            state=STATE_MEASURED if n_trades else STATE_UNKNOWN,
            unavailable="" if n_trades else "no statement in the window matched a trade",
            sources=sources,
            section="actual_trades",
        )
    ]

    exposures = journal_exposure.classify_all(journal_trades)
    bias_by_trade = {
        trade_id: exposure.market_bias for trade_id, exposure in exposures.items()
    }
    instrument_by_trade = {
        trade_id: exposure.instrument for trade_id, exposure in exposures.items()
    }

    def _bucket(rows: Sequence[Mapping[str, Any]], label: str, cell_id: str,
                metric: str, population: str) -> Cell:
        value, count = _money(rows)
        measured = bool(rows) and count > 0
        return Cell(
            cell_id=cell_id,
            metric=metric,
            unit="CAD",
            value=round(value, 4) if measured else None,
            n=count,
            distinct_symbols=len({str(row.get("symbol") or "") for row in rows}),
            distinct_sessions=len({str(row.get("session_date") or "") for row in rows}),
            population=population,
            window=window,
            reference_clock=_CLOCK_BROKER,
            exit_policy="broker close - the money is realized, not modelled",
            version=version,
            state=STATE_MEASURED if measured else STATE_UNKNOWN,
            unavailable=(
                "" if measured
                else f"no matched trade in this window carried {label} exposure"
            ),
            sources=sources,
            section="actual_trades",
        )

    # Shorter id FIRST wherever one id is a prefix of another
    # (`...bias.bullish` / `...bias.bullish_or_neutral`): the brief truncates by
    # dropping a TAIL, and a dropped cell whose id survives inside a longer one
    # would be counted as present.
    bias_order = (
        (journal_exposure.BIAS_BEARISH, "bearish"),
        (journal_exposure.BIAS_BEARISH_OR_NEUTRAL, "bearish-or-neutral"),
        (journal_exposure.BIAS_BULLISH, "bullish"),
        (journal_exposure.BIAS_BULLISH_OR_NEUTRAL, "bullish-or-neutral"),
        (journal_exposure.BIAS_UNKNOWN, "undetermined"),
    )
    for bias, label in bias_order:
        rows = [
            row for row in matched
            if bias_by_trade.get(str(row.get("trade_id") or "").strip(),
                                 journal_exposure.BIAS_UNKNOWN) == bias
        ]
        cells.append(_bucket(
            rows,
            label,
            f"total_profit.bias.{bias}",
            f"closed-trade net P&L on {label} exposure",
            (
                f"matched trades whose LEGS make them {label} "
                "(`journal_exposure`: a long put is bearish money however its "
                "`direction` column reads)"
            ),
        ))

    for instrument, label in (("STK", "stock"), ("OPT", "option")):
        rows = [
            row for row in matched
            if instrument_by_trade.get(str(row.get("trade_id") or "").strip(), "")
            == instrument
        ]
        cells.append(_bucket(
            rows,
            label,
            f"total_profit.instrument.{label}",
            f"closed-trade net P&L on {label} positions",
            f"matched trades whose security type is {instrument}",
        ))
    return cells


def _day_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    session_date: str,
    sources: tuple[str, ...],
) -> list[Cell]:
    """Answers 2 and 4 for the intraday population, side-adjusted at the source.

    `intraday_bounce_outcomes.csv` stores `mfe_pct` / `mae_pct` / `eod_move_pct`
    ALREADY side-adjusted (`bounce_bot_lib/legacy.py`): a short whose close fell
    5% reads +5.00. Recomputing the move from prices here would invert every
    short, so nothing here recomputes anything.
    """
    window = (session_date, session_date)
    version = f"{REPORT_VERSION}+intraday_bounce_outcomes"
    population = (
        f"every intraday bounce observation recorded for {session_date}, "
        "side-adjusted by the outcome writer"
    )

    def _series(column: str) -> tuple[list[float], list[str], list[str]]:
        values: list[float] = []
        symbols: list[str] = []
        sessions: list[str] = []
        for row in rows:
            number = _number(row.get(column))
            if number is None:
                continue
            values.append(number)
            symbols.append(str(row.get("symbol") or ""))
            sessions.append(str(row.get("trade_date") or ""))
        return values, symbols, sessions

    def _cell_for(cell_id: str, column: str, metric: str, unit: str,
                  pick, exit_policy: str, clock: str) -> Cell:
        values, symbols, sessions = _series(column)
        summary = _summary(values, symbols=symbols, sessions=sessions)
        measured = bool(values)
        return Cell(
            cell_id=cell_id,
            metric=metric,
            unit=unit,
            value=round(float(pick(values, summary)), 4) if measured else None,
            n=int(summary["n"]),
            distinct_symbols=int(summary["counts"]["symbols"]),
            distinct_sessions=int(summary["counts"]["sessions"]),
            population=population,
            window=window,
            reference_clock=clock,
            exit_policy=exit_policy,
            version=version,
            state=STATE_MEASURED if measured else STATE_UNKNOWN,
            unavailable=(
                "" if measured
                else f"no row for {session_date} carried a measured `{column}`"
            ),
            sources=sources,
            section="opportunity_results",
        )

    day_rows = len(rows)
    cells = [
        Cell(
            cell_id="measured_context.day_observations",
            metric="intraday bounce observations recorded for the session",
            unit="observations",
            value=day_rows,
            n=day_rows,
            distinct_symbols=len({str(row.get("symbol") or "") for row in rows}),
            distinct_sessions=1 if rows else 0,
            population=population,
            window=window,
            reference_clock=_CLOCK_EXCHANGE,
            exit_policy="not applicable - this is a count of observations",
            version=version,
            state=STATE_MEASURED if rows else STATE_UNKNOWN,
            unavailable="" if rows else f"no intraday row exists for {session_date}",
            sources=sources,
            section="measured_context",
        ),
        _cell_for(
            "biggest_opportunity.day.mfe_pct",
            "mfe_pct",
            "the best available movement after the observation (the maximum)",
            "percent",
            lambda values, _summary_: max(values),
            "none - MFE is a path fact, measured before any exit rule applies",
            _CLOCK_EXCHANGE,
        ),
        _cell_for(
            "biggest_opportunity.day.mae_pct",
            "mae_pct",
            "the worst adverse movement after the observation (the minimum)",
            "percent",
            lambda values, _summary_: min(values),
            "none - MAE is a path fact, measured before any exit rule applies",
            _CLOCK_EXCHANGE,
        ),
    ]

    # R needs risk. Where the trader never sized the bounce, `risk_per_share`
    # and `mfe_r` are present and EMPTY, so R is not knowable - and the
    # percentage beside it is still measured. Two facts, neither a zero.
    r_values = [value for value in (_number(row.get("mfe_r")) for row in rows)
                if value is not None]
    r_summary = _summary(r_values)
    cells.append(Cell(
        cell_id="biggest_opportunity.day.mfe_r",
        metric="the best available movement in R (needs a recorded risk)",
        unit="R",
        value=round(max(r_values), 4) if r_values else None,
        n=int(r_summary["n"]),
        population=population,
        window=window,
        reference_clock=_CLOCK_EXCHANGE,
        exit_policy="none - MFE is a path fact",
        version=version,
        state=STATE_MEASURED if r_values else STATE_UNKNOWN,
        unavailable=(
            "" if r_values else
            "no intraday row in this session recorded a risk per share, so R is "
            "not knowable here; the percentage beside it is measured"
        ),
        sources=sources if r_values else (),
        section="opportunity_results",
    ))

    cells.append(_cell_for(
        "end_of_day.day.eod_move_pct",
        "eod_move_pct",
        "the side-adjusted move held into the session close (mean)",
        "percent",
        lambda _values, summary: summary["raw"]["mean"],
        "eod_hold - the mark at the session close, as the outcome writer stored it",
        _CLOCK_SESSION_CLOSE,
    ))
    return cells


def _swing_cells(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    session_date: str,
    sources: tuple[str, ...],
    unavailable: str,
) -> list[Cell]:
    """Answers 2, 3 and 4 for the warehouse population.

    `rows` is None when the warehouse could not be read at all. Every cell then
    carries the REAL reason - an unreachable store is uncertainty, and the rest
    of the report is published regardless.
    """
    window = (session_date, session_date)
    version = f"{REPORT_VERSION}+outcome_path/house_default_v1"
    population = (
        f"warehouse occurrences entered on {session_date}, the latest outcome "
        "row per (occurrence, recipe, definition)"
    )
    rows = list(rows or ()) if rows is not None else None
    blocked = rows is None

    def _blocked_cell(cell_id: str, metric: str, unit: str, *, section: str,
                      clock: str = _CLOCK_EXCHANGE, exit_policy: str = "",
                      n: int = 0) -> Cell:
        return Cell(
            cell_id=cell_id,
            metric=metric,
            unit=unit,
            value=None,
            n=n,
            population=population,
            window=window,
            reference_clock=clock,
            exit_policy=exit_policy or "not reached - the population could not be read",
            version=version,
            state=STATE_UNKNOWN,
            unavailable=unavailable,
            sources=(),
            section=section,
        )

    definitions = (
        ("measured_context.swing_occurrences",
         "warehouse occurrences entered in the session", "occurrences",
         "measured_context", _CLOCK_EXCHANGE, "not applicable - a count"),
        ("biggest_opportunity.swing.mfe_r",
         "the best available movement in R (only where a stop was known)", "R",
         "opportunity_results", _CLOCK_EXCHANGE, "none - MFE is a path fact"),
        ("biggest_opportunity.swing.mae_r",
         "the worst adverse movement in R (only where a stop was known)", "R",
         "opportunity_results", _CLOCK_EXCHANGE, "none - MAE is a path fact"),
        ("biggest_opportunity.swing.time_to_mfe_min",
         "how long the best movement took - a HINDSIGHT fact, not the speed answer",
         "minutes", "opportunity_results", _CLOCK_EXCHANGE, "none - a path fact"),
        ("quickest_result.swing.hit_rate",
         "share of MEASURED occurrences that reached the first target", "rate",
         "opportunity_results", _CLOCK_EXCHANGE, "first target of the recipe"),
        ("quickest_result.swing.hits",
         "occurrences that reached the first target", "occurrences",
         "opportunity_results", _CLOCK_EXCHANGE, "first target of the recipe"),
        ("quickest_result.swing.unhit",
         "occurrences that closed without reaching the first target", "occurrences",
         "opportunity_results", _CLOCK_EXCHANGE, "first target of the recipe"),
        ("quickest_result.swing.pending",
         "occurrences still running", "occurrences",
         "opportunity_results", _CLOCK_EXCHANGE, "first target of the recipe"),
        ("quickest_result.swing.unknown",
         "occurrences whose bars ran out - unknown, never 'did not hit'",
         "occurrences", "opportunity_results", _CLOCK_EXCHANGE,
         "first target of the recipe"),
        ("quickest_result.swing.median_trading_minutes",
         "median TRADING minutes from entry to the first target, among the hits",
         "trading_minutes", "opportunity_results", _CLOCK_EXCHANGE,
         "first target of the recipe"),
        ("end_of_day.swing.r_at_eod",
         "the R held into the entry session's close (mean)", "R",
         "opportunity_results", _CLOCK_SESSION_CLOSE,
         "eod_hold - marked at the entry session's close"),
        ("end_of_day.swing.entry_at_close.r_at_eod",
         "the same mark for occurrences entered AT the close", "R",
         "opportunity_results", _CLOCK_SESSION_CLOSE,
         "eod_hold - marked at the entry session's close"),
    )
    if blocked:
        return [
            _blocked_cell(cell_id, metric, unit, section=section, clock=clock,
                          exit_policy=policy)
            for cell_id, metric, unit, section, clock, policy in definitions
        ]

    hits: list[Mapping[str, Any]] = []
    unhit: list[Mapping[str, Any]] = []
    pending: list[Mapping[str, Any]] = []
    unknown: list[Mapping[str, Any]] = []
    at_close: list[Mapping[str, Any]] = []
    for row in rows:
        entry_at = _aware(row.get("entry_at"))
        if entry_at is not None:
            try:
                closed_at = market_calendar.session_close(
                    entry_at.astimezone(market_calendar.MARKET_TZ).date()
                )
            except Exception:  # noqa: BLE001
                closed_at = None
            if closed_at is not None and entry_at >= closed_at:
                at_close.append(row)
        if _aware(row.get("first_hit_at")) is not None:
            hits.append(row)
            continue
        state = str(row.get("result_state") or "").strip().lower()
        if state == "closed":
            unhit.append(row)
        elif state == "open":
            pending.append(row)
        else:
            # `truncated`, blank, anything a later writer invents: the bars ran
            # out under it. Unknown is never folded into "did not hit".
            unknown.append(row)

    def _series(column: str) -> tuple[list[float], list[str]]:
        values: list[float] = []
        symbols: list[str] = []
        for row in rows:
            number = _number(row.get(column))
            if number is None:
                continue
            values.append(number)
            symbols.append(str(row.get("symbol") or row.get("occurrence_id") or ""))
        return values, symbols

    def _extremum(cell_id: str, column: str, metric: str, pick, exit_policy: str) -> Cell:
        values, symbols = _series(column)
        summary = _summary(values, symbols=symbols, sessions=[session_date] * len(values))
        return Cell(
            cell_id=cell_id,
            metric=metric,
            unit="R",
            value=round(float(pick(values)), 4) if values else None,
            n=int(summary["n"]),
            distinct_symbols=int(summary["counts"]["symbols"]),
            distinct_sessions=int(summary["counts"]["sessions"]),
            population=population,
            window=window,
            reference_clock=_CLOCK_EXCHANGE,
            exit_policy=exit_policy,
            version=version,
            state=STATE_MEASURED if values else STATE_UNKNOWN,
            unavailable=(
                "" if values else
                "no occurrence in this session recorded a stop distance, so R is "
                "not knowable for any of them"
            ),
            sources=sources if values else (),
            section="opportunity_results",
        )

    def _count_cell(cell_id: str, metric: str, count: int, exit_policy: str) -> Cell:
        return Cell(
            cell_id=cell_id,
            metric=metric,
            unit="occurrences",
            value=count,
            n=count,
            distinct_sessions=1 if rows else 0,
            population=population,
            window=window,
            reference_clock=_CLOCK_EXCHANGE,
            exit_policy=exit_policy,
            version=version,
            state=STATE_MEASURED,
            sources=sources,
            section="opportunity_results",
        )

    cells: list[Cell] = [Cell(
        cell_id="measured_context.swing_occurrences",
        metric="warehouse occurrences entered in the session",
        unit="occurrences",
        value=len(rows),
        n=len(rows),
        distinct_sessions=1 if rows else 0,
        population=population,
        window=window,
        reference_clock=_CLOCK_EXCHANGE,
        exit_policy="not applicable - a count",
        version=version,
        state=STATE_MEASURED if rows else STATE_UNKNOWN,
        unavailable="" if rows else f"no occurrence was entered on {session_date}",
        sources=sources if rows else (),
        section="measured_context",
    )]

    cells.append(_extremum(
        "biggest_opportunity.swing.mfe_r", "mfe_r",
        "the best available movement in R (only where a stop was known)",
        max, "none - MFE is a path fact",
    ))
    cells.append(_extremum(
        "biggest_opportunity.swing.mae_r", "mae_r",
        "the worst adverse movement in R (only where a stop was known)",
        min, "none - MAE is a path fact",
    ))

    # The SAME occurrences the speed answer measures - the ones that hit - read
    # the other way. That is the point of carrying both: "it took three hours to
    # reach its best price" and "it reached its target in 25 trading minutes"
    # are different facts about one set of rows, and only the second is a speed.
    hindsight: list[float] = []
    hindsight_symbols: list[str] = []
    for row in hits:
        number = _number(row.get("time_to_mfe_min"))
        if number is None:
            continue
        hindsight.append(number)
        hindsight_symbols.append(str(row.get("symbol") or row.get("occurrence_id") or ""))
    hindsight_summary = _summary(hindsight, symbols=hindsight_symbols)
    cells.append(Cell(
        cell_id="biggest_opportunity.swing.time_to_mfe_min",
        metric="how long the best movement took - a HINDSIGHT fact, not the speed answer",
        unit="minutes",
        value=hindsight_summary["raw"]["median"] if hindsight else None,
        n=int(hindsight_summary["n"]),
        distinct_symbols=int(hindsight_summary["counts"]["symbols"]),
        population=(
            population + "; read over the occurrences that HIT, so it is the "
            "same rows the speed answer measures and a different question"
        ),
        window=window,
        reference_clock=_CLOCK_EXCHANGE,
        exit_policy="none - a path fact read after the fact",
        version=version,
        state=STATE_MEASURED if hindsight else STATE_UNKNOWN,
        unavailable="" if hindsight else "no occurrence recorded a time to its best move",
        sources=sources if hindsight else (),
        section="opportunity_results",
    ))

    measured_n = len(hits) + len(unhit)
    cells.append(Cell(
        cell_id="quickest_result.swing.hit_rate",
        metric="share of MEASURED occurrences that reached the first target",
        unit="rate",
        value=round(len(hits) / measured_n, 4) if measured_n else None,
        n=measured_n,
        distinct_sessions=1 if rows else 0,
        population=(
            population + "; the denominator is what was MEASURED (hit + unhit), "
            "with pending and unknown counted beside it rather than assumed"
        ),
        window=window,
        reference_clock=_CLOCK_EXCHANGE,
        exit_policy="first target of the recipe",
        version=version,
        state=STATE_MEASURED if measured_n else STATE_UNKNOWN,
        unavailable="" if measured_n else "nothing in this session has matured yet",
        sources=sources if measured_n else (),
        section="opportunity_results",
    ))
    cells.append(_count_cell(
        "quickest_result.swing.hits", "occurrences that reached the first target",
        len(hits), "first target of the recipe"))
    cells.append(_count_cell(
        "quickest_result.swing.unhit",
        "occurrences that closed without reaching the first target",
        len(unhit), "first target of the recipe"))
    cells.append(_count_cell(
        "quickest_result.swing.pending", "occurrences still running",
        len(pending), "first target of the recipe"))
    cells.append(_count_cell(
        "quickest_result.swing.unknown",
        "occurrences whose bars ran out - unknown, never 'did not hit'",
        len(unknown), "first target of the recipe"))

    elapsed: list[float] = []
    elapsed_symbols: list[str] = []
    for row in hits:
        minutes = trading_minutes_between(
            _aware(row.get("entry_at")), _aware(row.get("first_hit_at"))
        )
        if minutes is None:
            continue
        elapsed.append(minutes)
        elapsed_symbols.append(str(row.get("symbol") or row.get("occurrence_id") or ""))
    elapsed_summary = _summary(elapsed, symbols=elapsed_symbols)
    cells.append(Cell(
        cell_id="quickest_result.swing.median_trading_minutes",
        metric="median TRADING minutes from entry to the first target, among the hits",
        unit="trading_minutes",
        value=elapsed_summary["raw"]["median"] if elapsed else None,
        n=int(elapsed_summary["n"]),
        distinct_symbols=int(elapsed_summary["counts"]["symbols"]),
        population=(
            population + "; only the occurrences that HIT, counted on the "
            "exchange clock so a target reached after an overnight gap is not "
            "credited with the hours the market was shut"
        ),
        window=window,
        reference_clock=_CLOCK_EXCHANGE,
        exit_policy="first target of the recipe",
        version=version,
        state=STATE_MEASURED if elapsed else STATE_UNKNOWN,
        unavailable="" if elapsed else "no occurrence reached its first target yet",
        sources=sources if elapsed else (),
        section="opportunity_results",
    ))

    at_close_ids = {id(row) for row in at_close}
    eod_values: list[float] = []
    eod_symbols: list[str] = []
    for row in rows:
        if id(row) in at_close_ids:
            continue
        number = _number(row.get("r_at_eod"))
        if number is None:
            continue
        eod_values.append(number)
        eod_symbols.append(str(row.get("symbol") or row.get("occurrence_id") or ""))
    eod_summary = _summary(eod_values, symbols=eod_symbols, sessions=[session_date] * len(eod_values))
    cells.append(Cell(
        cell_id="end_of_day.swing.r_at_eod",
        metric="the R held into the entry session's close (mean)",
        unit="R",
        value=eod_summary["raw"]["mean"] if eod_values else None,
        n=int(eod_summary["n"]),
        distinct_symbols=int(eod_summary["counts"]["symbols"]),
        distinct_sessions=int(eod_summary["counts"]["sessions"]),
        population=(
            population + "; occurrences entered AT the close are EXCLUDED rather "
            "than zero-filled and counted in their own cell"
        ),
        window=window,
        reference_clock=_CLOCK_SESSION_CLOSE,
        exit_policy="eod_hold - marked at the entry session's close",
        version=version,
        state=STATE_MEASURED if eod_values else STATE_UNKNOWN,
        unavailable="" if eod_values else "no occurrence carried a measured `r_at_eod`",
        sources=sources if eod_values else (),
        section="opportunity_results",
    ))
    cells.append(Cell(
        cell_id="end_of_day.swing.entry_at_close.r_at_eod",
        metric="the same mark for occurrences entered AT the close",
        unit="R",
        value=None,
        n=len(at_close),
        distinct_sessions=1 if at_close else 0,
        population="occurrences whose entry moment is the session close itself",
        window=window,
        reference_clock=_CLOCK_SESSION_CLOSE,
        exit_policy="eod_hold - marked at the entry session's close",
        version=version,
        state=STATE_UNKNOWN,
        unavailable=UNAVAILABLE_ENTRY_AT_CLOSE,
        sources=(),
        section="opportunity_results",
    ))
    return cells


def _last_sessions_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    session_date: str,
    selection_sessions: int,
    follow_through_sessions: int,
    sources: tuple[str, ...],
) -> list[Cell]:
    """Answer 5. TWO controls, and they are independent.

    Which observation sessions were selected is one question; how long they
    were followed is another. Holding one and moving the other moves exactly
    one answer - which is the whole point of "the last day or two" being two
    numbers rather than "48 hours".
    """
    window_days = _selection_window(session_date, selection_sessions)
    window = (window_days[0], window_days[-1]) if window_days else (session_date, session_date)
    version = f"{REPORT_VERSION}+session_horizon_outcomes"
    selected = [row for row in rows if str(row.get("scan_date") or "") in set(window_days)]
    observations = {str(row.get("scan_row_id") or row.get("observation_id") or "")
                    for row in selected}
    observations.discard("")
    horizons_in_file = sorted({
        int(value) for value in (
            _number(row.get("horizon_sessions")) for row in rows
        ) if value is not None
    })

    cells = [Cell(
        cell_id="last_sessions.selection.observations",
        metric="distinct scan observations inside the selection window",
        unit="observations",
        value=len(observations),
        n=len(observations),
        distinct_symbols=len({str(row.get("symbol") or "") for row in selected}),
        distinct_sessions=len({str(row.get("scan_date") or "") for row in selected}),
        population=(
            f"scan observations whose scan_date falls in the last "
            f"{selection_sessions} EXCHANGE session(s) ending {session_date}"
        ),
        window=window,
        reference_clock=_CLOCK_EXCHANGE,
        exit_policy="not applicable - a count of observations",
        version=version,
        state=STATE_MEASURED if observations else STATE_UNKNOWN,
        unavailable=(
            "" if observations
            else "no scan observation falls inside the selected sessions"
        ),
        sources=sources if observations else (),
        section="opportunity_results",
    )]

    at_horizon = [
        row for row in selected
        if (_number(row.get("horizon_sessions")) or 0) == float(follow_through_sessions)
    ]
    measured_rows = [
        row for row in at_horizon
        if str(row.get("measured") or "").strip().lower() in {"true", "1", "yes"}
        and _number(row.get("side_return_pct")) is not None
    ]
    values = [float(_number(row.get("side_return_pct"))) for row in measured_rows]
    summary = _summary(
        values,
        symbols=[str(row.get("symbol") or "") for row in measured_rows],
        sessions=[str(row.get("scan_date") or "") for row in measured_rows],
    )
    published = follow_through_sessions in (horizons_in_file or PUBLISHED_FOLLOW_THROUGH_HORIZONS)
    if values:
        unavailable = ""
    elif not published:
        unavailable = (
            f"the desk never publishes a {follow_through_sessions}-session "
            "follow-through: the D1 scan writes horizons "
            f"{', '.join(str(value) for value in (horizons_in_file or PUBLISHED_FOLLOW_THROUGH_HORIZONS))} "
            "(`SCAN_FACTOR_HORIZONS`), so there is no row to read - unknown, not zero"
        )
    else:
        unavailable = (
            f"no observation in the selected sessions has matured at "
            f"{follow_through_sessions} session(s) yet"
        )
    cells.append(Cell(
        cell_id="last_sessions.follow_through.side_return_pct",
        metric=(
            f"mean side-adjusted return {follow_through_sessions} session(s) "
            "after the selected observations"
        ),
        unit="percent",
        value=summary["raw"]["mean"] if values else None,
        n=int(summary["n"]),
        distinct_symbols=int(summary["counts"]["symbols"]),
        distinct_sessions=int(summary["counts"]["sessions"]),
        population=(
            f"the selected observations followed {follow_through_sessions} "
            "exchange session(s) on; the two controls are independent"
        ),
        window=window,
        reference_clock=_CLOCK_EXCHANGE,
        exit_policy=(
            "entry session close to target session close - the horizon file's "
            "own convention, not a stop rule"
        ),
        version=version,
        state=STATE_MEASURED if values else STATE_UNKNOWN,
        unavailable=unavailable,
        sources=sources if values else (),
        section="opportunity_results",
    ))

    pending = len(at_horizon) - len(measured_rows)
    cells.append(Cell(
        cell_id="last_sessions.follow_through.pending",
        metric="selected observations at that horizon still waiting to mature",
        unit="observations",
        value=pending,
        n=pending,
        population="the selected observations at the chosen follow-through horizon",
        window=window,
        reference_clock=_CLOCK_EXCHANGE,
        exit_policy="not applicable - a count",
        version=version,
        state=STATE_MEASURED,
        sources=sources,
        section="opportunity_results",
    ))
    return cells


def _preference_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    session_date: str,
    sources: tuple[str, ...],
) -> list[Cell]:
    """The decisions themselves - statements, matches and the duplicate count."""
    import preference_trade_outcomes

    summary = preference_trade_outcomes.trade_level_summary(rows)
    version = f"{REPORT_VERSION}+{preference_trade_outcomes.SCHEMA}"
    window = (session_date, session_date)
    population = (
        "every explicit verdict in the preference report - a statement is what "
        "the trader SAID, and three statements about one name are three "
        "statements and one trade"
    )

    def _count(cell_id: str, metric: str, value: int) -> Cell:
        return Cell(
            cell_id=cell_id,
            metric=metric,
            unit="rows",
            value=int(value),
            n=int(value),
            population=population,
            window=window,
            reference_clock=_CLOCK_BROKER,
            exit_policy="not applicable - a count of decisions",
            version=version,
            state=STATE_MEASURED,
            sources=sources,
            section="preference_decisions",
        )

    return [
        _count("preference_decisions.statements", "explicit verdicts recorded",
               int(summary["n_statements"])),
        _count("preference_decisions.matched_trades",
               "distinct trades a verdict was matched to",
               int(summary["n_trades_matched"])),
        _count("preference_decisions.duplicate_statement_rows",
               "matched statement rows beyond one per trade - ON PURPOSE",
               int(summary["duplicate_statement_rows"])),
    ]


def _thesis_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    session_date: str,
    sources: tuple[str, ...],
) -> tuple[list[Cell], tuple[dict[str, Any], ...]]:
    """What the trader THOUGHT - the open theses, counted, never narrated."""
    open_rows = tuple(
        dict(row) for row in rows
        if str(row.get("status") or "").strip().lower() == "open"
    )
    closed = sum(
        1 for row in rows if str(row.get("status") or "").strip().lower() == "closed"
    )
    version = f"{REPORT_VERSION}+market_theses"
    window = (session_date, session_date)
    cells = [
        Cell(
            cell_id="market_thoughts.open_theses",
            metric="market theses still open",
            unit="theses",
            value=len(open_rows),
            n=len(open_rows),
            population="the Market Journal's thesis sidecar, `status = open`",
            window=window,
            reference_clock=_CLOCK_EXCHANGE,
            exit_policy="not applicable - a count",
            version=version,
            state=STATE_MEASURED if rows else STATE_UNKNOWN,
            unavailable="" if rows else "no thesis has been written",
            sources=sources if rows else (),
            section="market_thoughts",
        ),
        Cell(
            cell_id="market_thoughts.closed_theses",
            metric="market theses already resolved",
            unit="theses",
            value=closed,
            n=closed,
            population="the Market Journal's thesis sidecar, `status = closed`",
            window=window,
            reference_clock=_CLOCK_EXCHANGE,
            exit_policy="not applicable - a count",
            version=version,
            state=STATE_MEASURED if rows else STATE_UNKNOWN,
            unavailable="" if rows else "no thesis has been written",
            sources=sources if rows else (),
            section="market_thoughts",
        ),
    ]
    return cells, open_rows


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


def _example_tables(
    day_rows: Sequence[Mapping[str, Any]],
    swing_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    """Best and worst, each with its FULL denominator and what it is.

    Picking the three biggest moves out of a measured set says nothing about
    the set. The label is on every table for that reason.
    """
    tables: list[dict[str, Any]] = []

    def _table(table_id: str, title: str, rows: Sequence[tuple[str, float]],
               denominator: int, unit: str, reverse: bool) -> None:
        if not rows or denominator <= 0:
            return
        ordered = sorted(rows, key=lambda item: item[1], reverse=reverse)[:3]
        tables.append({
            "table_id": table_id,
            "title": title,
            "label": SELECTED_TABLE_LABEL,
            "unit": unit,
            "denominator": int(denominator),
            "rows": [{"name": name, "value": round(value, 4)} for name, value in ordered],
            "note": (
                f"{len(ordered)} of {denominator} measured observation(s), "
                "selected on the result being shown"
            ),
        })

    day_best = [
        (str(row.get("symbol") or ""), float(_number(row.get("mfe_pct"))))
        for row in day_rows if _number(row.get("mfe_pct")) is not None
    ]
    day_worst = [
        (str(row.get("symbol") or ""), float(_number(row.get("mae_pct"))))
        for row in day_rows if _number(row.get("mae_pct")) is not None
    ]
    _table("day.best_moves", "Biggest intraday moves after the observation",
           day_best, len(day_best), "percent", True)
    _table("day.worst_moves", "Worst intraday adverse moves after the observation",
           day_worst, len(day_worst), "percent", False)

    if swing_rows:
        swing_best = [
            (str(row.get("occurrence_id") or row.get("symbol") or ""),
             float(_number(row.get("mfe_r"))))
            for row in swing_rows if _number(row.get("mfe_r")) is not None
        ]
        _table("swing.best_moves", "Biggest swing movement in R",
               swing_best, len(swing_best), "R", True)
    return tuple(tables)


def build_report(
    session_date: str,
    *,
    now: datetime | None = None,
    sources: ReportSources | None = None,
    warehouse: Any | None = None,
    selection_sessions: int = DEFAULT_SELECTION_SESSIONS,
    follow_through_sessions: int = DEFAULT_FOLLOW_THROUGH_SESSIONS,
) -> MeasuredReport:
    """The whole contract for one session. Pure: it reads and computes.

    `warehouse` is a seam with two members - `read_outcomes(session_date, now=)`
    and `source_paths`. `None` means no warehouse was configured, which is the
    default on a desk where `research_store_dir` was never set: the swing cells
    are `unknown` with that reason and every other answer is still published.
    """
    session = str(session_date)[:10]
    resolved = sources or ReportSources.from_project_paths()
    moment = now or datetime.now()

    intraday_rows = [
        row for row in _rows_from_csv(resolved.intraday_outcomes)
        if str(row.get("trade_date") or "")[:10] == session
    ]
    horizon_rows = _rows_from_csv(resolved.session_horizon_outcomes)
    preference_rows = _rows_from_csv(resolved.preference_report)
    journal_trades = _read_journal_trades(resolved.journal_trades)
    thesis_rows = _rows_from_jsonl(resolved.market_theses)

    swing_rows: list[dict[str, Any]] | None = None
    warehouse_sources: tuple[str, ...] = ()
    warehouse_reason = (
        "no research warehouse is configured on this desk "
        "(`research_store_dir` unset), so the swing population was not read"
    )
    if warehouse is not None:
        try:
            swing_rows = [dict(row) for row in warehouse.read_outcomes(session, now=moment)]
            warehouse_sources = _cite(*tuple(getattr(warehouse, "source_paths", ()) or ()))
            warehouse_reason = ""
        except Exception as exc:  # noqa: BLE001 - an unreachable store is uncertainty
            swing_rows = None
            warehouse_reason = f"the research warehouse could not be read: {exc}"

    intraday_sources = _cite(resolved.intraday_outcomes)
    horizon_sources = _cite(resolved.session_horizon_outcomes)
    money_sources = _cite(resolved.preference_report, resolved.journal_trades)
    thesis_sources = _cite(resolved.market_theses)

    thesis_cells, open_theses = _thesis_cells(
        thesis_rows, session_date=session, sources=thesis_sources
    )
    day_cells = _day_cells(intraday_rows, session_date=session, sources=intraday_sources)
    swing_cells = _swing_cells(
        swing_rows, session_date=session, sources=warehouse_sources,
        unavailable=warehouse_reason,
    )
    last_cells = _last_sessions_cells(
        horizon_rows, session_date=session, selection_sessions=selection_sessions,
        follow_through_sessions=follow_through_sessions, sources=horizon_sources,
    )
    preference_cells = _preference_cells(
        preference_rows, session_date=session, sources=_cite(resolved.preference_report)
    )
    money_cells = _money_cells(
        preference_rows, journal_trades, session_date=session, sources=money_sources
    )

    ordered: list[Cell] = []
    ordered.extend(thesis_cells)
    ordered.extend(cell for cell in day_cells if cell.section == "measured_context")
    ordered.extend(cell for cell in swing_cells if cell.section == "measured_context")
    ordered.extend(cell for cell in day_cells if cell.section == "opportunity_results")
    ordered.extend(cell for cell in swing_cells if cell.section == "opportunity_results")
    ordered.extend(last_cells)
    ordered.extend(preference_cells)
    ordered.extend(money_cells)

    sections: dict[str, tuple[Cell, ...]] = {}
    for name in SECTION_NAMES:
        if name == "missing_evidence":
            sections[name] = tuple(
                cell for cell in ordered if cell.state != STATE_MEASURED
            )
        else:
            sections[name] = tuple(cell for cell in ordered if cell.section == name)

    as_of = session  # the SESSION the evidence belongs to, never the wall clock
    tracker_snapshot_id = _tracker_snapshot_id(resolved.working_lately)
    return MeasuredReport(
        session_date=session,
        as_of=as_of,
        report_id=compute_report_id(ordered, as_of),
        sections=sections,
        generated_at=moment.isoformat(timespec="seconds"),
        tracker_snapshot_id=tracker_snapshot_id,
        open_theses=open_theses,
        source_paths=resolved.existing() + warehouse_sources,
        policy=(
            f"selection window: {selection_sessions} exchange session(s) ending {session}",
            f"follow-through horizon: {follow_through_sessions} exchange session(s)",
            "money: once per trade_id, `preference_trade_outcomes.trade_level_summary`",
            "statistics: `evidence_stats` only; no statistic is computed here",
            "best/worst tables: " + SELECTED_TABLE_LABEL,
        ),
        example_tables=_example_tables(intraday_rows, swing_rows),
        _cells=tuple(ordered),
    )


def _tracker_snapshot_id(path: Any) -> str:
    """The Working-lately snapshot id, verbatim. Never recomputed here."""
    if not path or not Path(path).exists():
        return ""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if isinstance(payload, Mapping):
        return str(payload.get("snapshot_id") or "")
    return ""


def report_from_payload(payload: Mapping[str, Any]) -> MeasuredReport:
    """Rebuild a report from its published JSON, id and all.

    The Daily Recap's Review tab reads the PUBLISHED file rather than rebuilding
    the numbers, so the page cannot disagree with the export by construction.
    """
    cells = tuple(
        Cell(
            cell_id=str(row.get("cell_id") or ""),
            metric=str(row.get("metric") or ""),
            unit=str(row.get("unit") or ""),
            value=row.get("value"),
            n=int(row.get("n") or 0),
            distinct_sessions=int(row.get("distinct_sessions") or 0),
            distinct_symbols=int(row.get("distinct_symbols") or 0),
            population=str(row.get("population") or ""),
            window=tuple(str(part) for part in (row.get("window") or ("", ""))),
            reference_clock=str(row.get("reference_clock") or ""),
            exit_policy=str(row.get("exit_policy") or ""),
            version=str(row.get("version") or REPORT_VERSION),
            state=str(row.get("state") or STATE_UNKNOWN),
            sources=tuple(str(part) for part in (row.get("sources") or ())),
            unavailable=str(row.get("unavailable") or ""),
            section=str(row.get("section") or "measured_context"),
        )
        for row in (payload.get("cells") or ())
    )
    sections: dict[str, tuple[Cell, ...]] = {}
    for name in SECTION_NAMES:
        if name == "missing_evidence":
            sections[name] = tuple(cell for cell in cells if cell.state != STATE_MEASURED)
        else:
            sections[name] = tuple(cell for cell in cells if cell.section == name)
    return MeasuredReport(
        session_date=str(payload.get("session_date") or ""),
        as_of=str(payload.get("as_of") or ""),
        report_id=str(payload.get("report_id") or ""),
        sections=sections,
        generated_at=str(payload.get("generated_at") or ""),
        tracker_snapshot_id=str(payload.get("tracker_snapshot_id") or ""),
        open_theses=tuple(dict(row) for row in (payload.get("open_theses") or ())),
        source_paths=tuple(str(part) for part in (payload.get("source_paths") or ())),
        policy=tuple(str(part) for part in (payload.get("policy") or ())),
        example_tables=tuple(dict(table) for table in (payload.get("example_tables") or ())),
        _cells=cells,
    )


# ---------------------------------------------------------------------------
# the frontier handoff - user-initiated, offline, capped
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierHandoff:
    """A readable brief, the versioned payload, and the manifest beside them."""

    report_id: str
    session_date: str
    markdown: str
    payload: dict[str, Any]
    manifest: dict[str, Any]

    def write(self, directory: Any) -> dict[str, str]:
        """Three files under `directory`. The trader's click, never a job."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        markdown_path = target / f"frontier_handoff_{self.session_date}.md"
        payload_path = target / f"frontier_handoff_{self.session_date}.json"
        manifest_path = target / "manifest.json"
        markdown_path.write_text(self.markdown, encoding="utf-8")
        payload_path.write_text(
            json.dumps(self.payload, indent=1, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(self.manifest, indent=1, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return {
            "markdown": str(markdown_path),
            "payload": str(payload_path),
            "manifest": str(manifest_path),
        }


def _chars_per_token() -> float:
    """The desk's ONE token ratio. There is no tokenizer here.

    `ai_summary._ESTIMATED_CHARS_PER_TOKEN` (2.5, measured 2026-08-28) is the
    whole of the AI layer's token arithmetic, so the manifest quotes THAT and
    says the count is an estimate. A real token count only ever comes back from
    a server as `usage.prompt_tokens`.
    """
    try:
        import ai_summary

        return float(ai_summary._ESTIMATED_CHARS_PER_TOKEN)
    except Exception:  # noqa: BLE001 - the constant is pinned; this is the floor
        return 2.5


def _cell_line(cell: Cell) -> str:
    if cell.state == STATE_MEASURED:
        body = f"{cell.value} {cell.unit}"
    else:
        body = f"{cell.state.upper()} - {cell.unavailable}"
    return (
        f"- `{cell.cell_id}` = {body} "
        f"(n={cell.n}, {cell.distinct_symbols} symbol(s), "
        f"{cell.distinct_sessions} session(s), window {cell.window[0]}..{cell.window[1]})"
    )


def build_handoff(
    report: MeasuredReport,
    *,
    sources: ReportSources | None = None,
    cap_bytes: int = HANDOFF_MARKDOWN_CAP_BYTES,
) -> FrontierHandoff:
    """The compact handoff: brief, payload, manifest. Offline and user-initiated.

    Nothing here calls a model, uploads anything or opens a socket. The brief is
    capped at `cap_bytes` of UTF-8 and STATES what it dropped, because a brief
    that silently ends is a brief a reader trusts for a question it cannot
    answer.
    """
    cells = report.cells()
    total = len(cells)
    tracker_snapshot_id = report.tracker_snapshot_id
    source_paths = list(report.source_paths)
    if sources is not None:
        tracker_snapshot_id = _tracker_snapshot_id(sources.working_lately) or tracker_snapshot_id
        for path in sources.existing():
            if path not in source_paths:
                source_paths.append(path)
    source_paths = [path for path in source_paths if Path(path).exists()]

    header = "\n".join([
        f"# Measured report - {report.session_date}",
        "",
        f"report_id: {report.report_id}",
        f"as_of: {report.as_of} (the session the evidence belongs to, not a clock)",
        f"tracker snapshot: {tracker_snapshot_id or 'not published'}",
        "",
        "Measured by code from the desk's own stores. Every line names its "
        "population and its n; a line that says UNKNOWN was not measured, which "
        "is not the same as zero. Best/worst tables are "
        f"{SELECTED_TABLE_LABEL} and carry their full denominator.",
        "",
    ])

    included: list[Cell] = []
    body_lines: list[str] = []
    current_section = ""
    budget = int(cap_bytes) - len(header.encode("utf-8"))
    # Reserve the omitted line: it is the one line that must always survive.
    reserve = len(_omitted_line(total, total).encode("utf-8")) + 2
    used = 0
    for cell in cells:
        addition: list[str] = []
        if cell.section != current_section:
            addition.append(f"\n## {cell.section}\n")
        addition.append(_cell_line(cell))
        cost = len(("\n".join(addition) + "\n").encode("utf-8"))
        if used + cost > budget - reserve:
            break
        body_lines.extend(addition)
        used += cost
        current_section = cell.section
        included.append(cell)

    omitted = total - len(included)
    markdown = header + "\n".join(body_lines) + "\n\n" + _omitted_line(omitted, total) + "\n"

    tables_block = _tables_markdown(report.example_tables)
    if tables_block and len(
        (markdown + tables_block).encode("utf-8")
    ) <= int(cap_bytes):
        markdown = markdown + tables_block

    payload = {
        "schema": f"{REPORT_VERSION}_handoff",
        "report_id": report.report_id,
        "as_of": report.as_of,
        "session_date": report.session_date,
        "cells": [cell.as_dict() for cell in cells],
        "sections": {name: [cell.cell_id for cell in rows]
                     for name, rows in report.sections.items()},
        "example_tables": [dict(table) for table in report.example_tables],
        "policy": list(report.policy),
        "open_theses": [dict(row) for row in report.open_theses],
        "tracker_snapshot_id": tracker_snapshot_id,
        "source_paths": source_paths,
    }
    chars_per_token = _chars_per_token()
    manifest = {
        "schema": f"{REPORT_VERSION}_manifest",
        "report_id": report.report_id,
        "as_of": report.as_of,
        "session_date": report.session_date,
        "generated_at": report.generated_at,
        "tracker_snapshot_id": tracker_snapshot_id,
        "source_paths": source_paths,
        "cells_total": total,
        "cells_in_brief": len(included),
        "omitted_cells": omitted,
        "markdown_bytes": len(markdown.encode("utf-8")),
        "markdown_cap_bytes": int(cap_bytes),
        "markdown_tokens_estimated": round(len(markdown) / chars_per_token, 2),
        "chars_per_token": chars_per_token,
        "token_count_note": (
            "an ESTIMATE. There is no tokenizer on this desk: "
            "`ai_summary._ESTIMATED_CHARS_PER_TOKEN` is the AI layer's whole "
            "token arithmetic, and a real count only ever comes back from a "
            "server as `usage.prompt_tokens`"
        ),
        "open_theses": [dict(row) for row in report.open_theses],
        "missing_evidence": [
            {"cell_id": cell.cell_id, "reason": cell.unavailable}
            for cell in report.sections.get("missing_evidence", ())
        ],
        "drill_down": {
            "cells": "frontier_handoff_%s.json" % report.session_date,
            "note": "every cell names the file its number was read from",
        },
        "policy": list(report.policy),
        "note": (
            "Exported by the trader's click. No model was called, nothing was "
            "uploaded, and nothing here reaches a detector, score, alert, "
            "watchlist, Focus, the review queue or `review_policy.json`."
        ),
    }
    return FrontierHandoff(
        report_id=report.report_id,
        session_date=report.session_date,
        markdown=markdown,
        payload=payload,
        manifest=manifest,
    )


def _omitted_line(omitted: int, total: int) -> str:
    return (
        f"_This brief omitted {int(omitted)} of {int(total)} cells to stay inside "
        f"its size budget; the JSON payload beside it carries all of them._"
    )


def _tables_markdown(tables: Iterable[Mapping[str, Any]]) -> str:
    rows = list(tables or ())
    if not rows:
        return ""
    out = ["", "## examples", ""]
    for table in rows:
        out.append(f"**{table.get('title')}** - {table.get('label')}, "
                   f"{len(table.get('rows') or ())} of {table.get('denominator')} measured")
        for row in table.get("rows") or ():
            out.append(f"  - {row.get('name')}: {row.get('value')} {table.get('unit')}")
        out.append("")
    return "\n".join(out)


def render_markdown(report: MeasuredReport) -> str:
    """The published `.md` sibling: the whole report, uncapped."""
    return build_handoff(report, cap_bytes=1 << 30).markdown


__all__ = [
    "Cell",
    "FrontierHandoff",
    "HANDOFF_MARKDOWN_CAP_BYTES",
    "MeasuredReport",
    "PUBLISHED_FOLLOW_THROUGH_HORIZONS",
    "REPORT_VERSION",
    "ReportSources",
    "SECTION_NAMES",
    "SELECTED_TABLE_LABEL",
    "STATE_MEASURED",
    "STATE_PENDING",
    "STATE_UNKNOWN",
    "UNAVAILABLE_ENTRY_AT_CLOSE",
    "build_handoff",
    "build_report",
    "compute_report_id",
    "render_markdown",
    "report_from_payload",
    "trading_minutes_between",
]
