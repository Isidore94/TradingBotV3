"""Research > Results: a VIEW over evidence that already exists (packet G5.1).

Trader, 2026-09-06 (decision 0016 answer 7, AMENDED that day): *"Research gains
a **Results** landing page the trader may read - Bot setups / My trades and
Swing / Day trading kept as four separate populations, never pooled. It is the
FULL readout; the Desk's 'what is working lately' line remains the primary
surface, and both read ONE evidence snapshot."*

**The one rule this module lives under.** The page may compute a VIEW -
ordering, banding, labels - from statistics that already exist, and may never
compute a new statistic, threshold or eligibility rule. Every number a bot row
shows is the SAME OBJECT the `working_lately.EvidenceCell` carries; nothing here
re-derives one, and `rate * n` appears nowhere. A weighted rate never becomes an
integer count. The floors are the cell's own (`n_floor` / `meets_floor`, carried
by the snapshot because the swing cells are floored by `MIN_REPORTABLE_N` and
the day-trade cells by the aggregator's own `min_n`), the concentration refusal
is `EvidenceCell.concentrated`, the ordering basis is `working_lately.rank_basis`
and the window is `evidence_stats.lately_window`. Four borrowed rules, no new
ones.

**Four populations, never pooled.** `population` x `horizon` names one of them:
bot/swing (the two swing KINDS as two labelled sections, themselves never
pooled), bot/day, mine/swing, mine/day. `band_cells` refuses a mixed-kind list
the way `working_lately.pool_cells` refuses across its axes - the refusal is the
mechanism, and there is deliberately no formula behind it.

Pure: fixtures in, frozen dataclasses out. No file is opened here, no clock is
read, and nothing scores, ranks a queue, alerts, or writes anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import evidence_stats
import journal_analytics
import journal_trade_shape
import working_lately
from working_lately import EvidenceCell


#: How many cells a band shows. A page, not a leaderboard.
BAND_SIZE = 3

#: The sentence a page with no reading prints, on every card.
NO_SNAPSHOT = "no snapshot yet - the desk builds one after the window shows"

#: The sentence a bucket with nothing the TRADER named prints instead of a
#: table. A leaderboard built out of machine guesses would answer a question
#: nobody asked.
NO_CONFIRMED_TAGS = "no confirmed tags yet - nothing here names a setup"

#: What a bot row copies out of its cell, verbatim. Every one of these is
#: handed on as the cell's OWN object, so a reader can prove by identity that
#: nothing was recomputed on the way to the screen.
CELL_VALUE_FIELDS = (
    "statistic",
    "uncertainty_low",
    "n_eligible",
    "n_graded",
    "n_pending",
    "n_excluded",
    "n_symbols",
    "n_sessions",
    "n_floor",
    "top_symbol_share",
    "top_session_share",
)

#: One title and one population sentence per snapshot kind. The two swing kinds
#: measure DIFFERENT things - a closed R on the representative exit is not a
#: favorable close-to-close direction - so they are two sections and the page
#: says which is which rather than leaving the trader to infer it from a number.
KIND_TITLES = {
    "swing_trade_r": "Swing - closed R on the representative exit",
    "swing_favorable": "Swing - favorable direction at the declared horizon",
    "daytrade_held_run": "Day trading - held, then ran",
}

KIND_SENTENCES = {
    "swing_trade_r": (
        "Closed R on the representative exit, from the setup tracker's own "
        "recent family rows. Win rate leads; the lower bound is what the page "
        "sorts on."
    ),
    "swing_favorable": (
        "Favorable close-to-close direction at the declared horizon, in "
        "PERCENT units. A favorable direction is not a stop-rule win and this "
        "section never pools with the closed-R one above it."
    ),
    "daytrade_held_run": (
        "Held, then ran: P(the level held in the first 30 minutes) x the "
        "trimmed-mean MFE_R of the ones that held. Not an exit, and not a "
        "probability of profit."
    ),
}

#: The bucket a "My trades" section shows, and what it means.
MINE_TITLES = {
    "day": "My trades - opened and closed the same session",
    "swing": "My trades - held across sessions",
    "unknown_timing": "My trades - unknown timing",
}

MINE_SENTENCES = {
    "day": "Closed trades that opened and closed inside one session, with real clock times on both ends.",
    "swing": "Closed trades held across more than one session.",
    "unknown_timing": (
        "Closed trades whose fill times a broker file did not carry (midnight "
        "market-local means the time is not known). Shown under BOTH horizons "
        "and counted in neither: a broker file is authoritative for money and "
        "blind to time."
    ),
}


# ---------------------------------------------------------------------------
# the frozen view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultsRow:
    """One line of the page: a snapshot cell, or one "My trades" grouping.

    `cell` is the snapshot's own `EvidenceCell` for a bot row and `None` for a
    My-trades row. `values` carries the RAW numbers - the same objects their
    source holds - and `display` the strings a table shows.
    """

    cell: EvidenceCell | None
    line: str
    reason: str
    eligible: bool
    values: Mapping[str, Any]
    display: Mapping[str, str]


@dataclass(frozen=True)
class ResultsBands:
    """Stronger lately, Weaker lately, and everything that cannot say."""

    stronger: tuple[ResultsRow, ...] = ()
    weaker: tuple[ResultsRow, ...] = ()
    not_enough: tuple[ResultsRow, ...] = ()


@dataclass(frozen=True)
class ResultsSection:
    """One population's whole readout - and never more than one population."""

    key: str
    title: str
    kind: str
    sentence: str
    verdict_line: str
    bands: ResultsBands
    rows: tuple[ResultsRow, ...]
    studies: tuple[ResultsRow, ...]
    stats: Mapping[str, Any]
    analytics: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ResultsView:
    """What the page renders: the selection, its sections, and its provenance."""

    population: str
    horizon: str
    window: str
    window_label: str
    window_start: str
    window_end: str
    sections: tuple[ResultsSection, ...]
    freshness_line: str


# ---------------------------------------------------------------------------
# the banding rule
# ---------------------------------------------------------------------------


def _is_study(cell: EvidenceCell) -> bool:
    return str(cell.namespace or "").strip().lower() != "live"


def _measured(cell: EvidenceCell) -> bool:
    """Did anybody take this reading? An absent statistic is not a low one."""
    return cell.statistic is not None and cell.uncertainty_low is not None


def _eligible(cell: EvidenceCell) -> bool:
    """The cell's OWN eligibility, borrowed whole - no rule is invented here.

    `meets_floor` was decided by whichever aggregator built the cell and
    `concentrated` by `working_lately.CONCENTRATION_LIMIT`. The third clause is
    not a threshold: a cell nobody measured has no statistic to band.
    """
    return bool(cell.meets_floor) and not cell.concentrated and _measured(cell)


def ineligibility_reason(cell: EvidenceCell) -> str:
    """Why this cell cannot be banded, in the order the packet asks for it."""
    if not cell.meets_floor:
        return (
            f"below the evidence floor - {cell.n_graded} graded of "
            f"{cell.n_eligible} against a floor of {cell.n_floor}"
        )
    if cell.concentrated:
        parts = []
        if cell.top_symbol_share is not None:
            parts.append(f"one name is {float(cell.top_symbol_share):.2f} of the sample")
        if cell.top_session_share is not None:
            parts.append(f"one session is {float(cell.top_session_share):.2f}")
        return "too concentrated to lead - " + ", ".join(parts or ["share unmeasured"])
    if not _measured(cell):
        if cell.n_pending:
            return (
                f"not measured yet - {cell.n_pending} pending, {cell.n_graded} graded"
            )
        return f"not measured - no statistic was taken ({cell.n_graded} graded)"
    return ""


def _rank_value(cell: EvidenceCell) -> float:
    """The kind's OWN ranking field, read through `working_lately.rank_basis`."""
    field, _margin = working_lately.rank_basis(cell.kind)
    value = getattr(cell, field, None)
    return 0.0 if value is None else float(value)


def _by_rank(cells: Sequence[EvidenceCell]) -> list[EvidenceCell]:
    """Best first on the kind's own basis, then the larger sample, then name."""
    return sorted(
        cells,
        key=lambda cell: (-_rank_value(cell), -int(cell.n_eligible or 0), cell.name),
    )


def _lowest_statistic(cells: Sequence[EvidenceCell]) -> list[EvidenceCell]:
    return sorted(
        cells,
        key=lambda cell: (
            float(cell.statistic) if cell.statistic is not None else 0.0,
            cell.name,
        ),
    )


def band_cells(cells: Iterable[EvidenceCell]) -> ResultsBands:
    """Band ONE kind's cells. Raises `ValueError` on a mixed-kind list.

    Stronger lately is the eligible cells in the snapshot's own order (its
    kind's ranking field, descending), top three. Weaker lately is the three
    LOWEST by statistic among the eligible cells that Stronger did not take, so
    a cell can never stand in both bands however few cells there are. Not
    enough evidence is every ineligible cell with the reason it is one. A study
    is in no band at all: an unpromoted idea may not lead.
    """
    cells = list(cells)
    kinds = {str(cell.kind) for cell in cells}
    if len(kinds) > 1:
        raise ValueError(
            "band_cells refuses a mixed-kind cell list: "
            + ", ".join(sorted(kinds))
            + " - these measure different things and are never pooled"
        )
    live = [cell for cell in cells if not _is_study(cell)]
    eligible = [cell for cell in live if _eligible(cell)]
    stronger = _by_rank(eligible)[:BAND_SIZE]
    taken = {cell.identity_tuple() for cell in stronger}
    remaining = [cell for cell in eligible if cell.identity_tuple() not in taken]
    weaker = _lowest_statistic(remaining)[:BAND_SIZE]
    not_enough = _by_rank([cell for cell in live if not _eligible(cell)])
    return ResultsBands(
        stronger=tuple(_cell_row(cell) for cell in stronger),
        weaker=tuple(_cell_row(cell) for cell in weaker),
        not_enough=tuple(_cell_row(cell) for cell in not_enough),
    )


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------


def _number(value: Any) -> str:
    return "unmeasured" if value is None else f"{float(value):.4g}"


def _sample_text(cell: EvidenceCell) -> str:
    """How much evidence, spelled the way this KIND counts it.

    The swing cells grade a population (`n_graded` of `n_eligible`); the
    day-trade score is not a rate and grades nothing, so its cell states the
    episode count alone. Printing "0 graded" beside a day-trade score would
    read as a measurement that failed.
    """
    if int(cell.n_graded or 0):
        return f"{cell.n_graded} graded of {cell.n_eligible}"
    return f"{cell.n_eligible} episode(s)"


def _coverage_text(cell: EvidenceCell) -> str:
    body = f"{cell.n_symbols} symbol(s) / {cell.n_sessions} session(s)"
    if cell.top_symbol_share is None and cell.top_session_share is None:
        return f"{body}; concentration unmeasured"
    top_symbol = "unmeasured" if cell.top_symbol_share is None else f"{float(cell.top_symbol_share):.2f}"
    top_session = "unmeasured" if cell.top_session_share is None else f"{float(cell.top_session_share):.2f}"
    return f"{body}; top name {top_symbol}, top session {top_session}"


def _cell_row(cell: EvidenceCell) -> ResultsRow:
    """One bot row. Every number is the cell's own object, handed straight on."""
    reason = ineligibility_reason(cell)
    values = {field: getattr(cell, field) for field in CELL_VALUE_FIELDS}
    values.update(
        {
            "kind": cell.kind,
            "side": cell.side,
            "family": cell.family,
            "name": cell.name,
            "namespace": cell.namespace,
            "statistic_name": cell.statistic_name,
            "uncertainty_kind": cell.uncertainty_kind,
            "latest_measured_session": cell.latest_measured_session,
        }
    )
    display = {
        # ONE line for a band card, spelled the way the Trading Desk's own
        # Working-lately line spells it. The full provenance stays available as
        # `ResultsRow.line` (`EvidenceCell.line()`), which is a dozen clauses
        # long and belongs in a tooltip and the detail pane, not stacked three
        # deep in a third-width card - measured at 1920x1080, three of them left
        # the shortlist under the cards 26 pixels tall.
        "headline": (
            f"{cell.name} - {_number(cell.statistic)} "
            f"(>= {_number(cell.uncertainty_low)}, n={cell.n_eligible}, "
            f"{cell.n_sessions} sessions)"
        ),
        "side": str(cell.side or "").upper(),
        "family": str(cell.family or ""),
        "sample": _sample_text(cell),
        "statistic": _number(cell.statistic),
        "lower_bound": _number(cell.uncertainty_low),
        "n_symbols": str(cell.n_symbols),
        "n_sessions": str(cell.n_sessions),
        "coverage": _coverage_text(cell),
        "eligibility": reason or "eligible",
        "namespace": str(cell.namespace or ""),
    }
    return ResultsRow(
        cell=cell,
        line=cell.line(),
        reason=reason,
        eligible=not reason,
        values=values,
        display=display,
    )


# ---------------------------------------------------------------------------
# bot sections
# ---------------------------------------------------------------------------


def _verdict_line(snapshot: Mapping[str, Any] | None, kind: str) -> str:
    """The kind's verdict, printed VERBATIM - its own state and its own reason.

    Never softened into one word for every state that is not a leader: "no
    clear leader" and "no evidence" are different facts, and the reason is the
    machine's own sentence rather than a paraphrase of it.
    """
    entry = ((snapshot or {}).get("verdicts") or {}).get(kind)
    if not isinstance(entry, Mapping):
        return f"{kind}: no verdict in this snapshot"
    state = str(entry.get("state") or "")
    reason = str(entry.get("reason") or "")
    name = str(entry.get("leader_name") or "")
    bits = [f"{kind}: {state}"]
    if name and state in {"leader", "last_reliable_reading"}:
        bits.append(name)
    if reason:
        bits.append(reason)
    policy = str(entry.get("policy_line") or "")
    if policy:
        bits.append(policy)
    return " - ".join(bits)


def _bot_section(snapshot: Mapping[str, Any] | None, kind: str) -> ResultsSection:
    cells = [
        cell
        for cell in working_lately.cells_from_payload(snapshot)
        if str(cell.kind) == kind
    ]
    live = [cell for cell in cells if not _is_study(cell)]
    studies = [cell for cell in cells if _is_study(cell)]
    bands = band_cells(cells)
    rows = _by_rank([cell for cell in live if _eligible(cell)])
    rows += _by_rank([cell for cell in live if not _eligible(cell)])
    sentence = KIND_SENTENCES.get(kind, "")
    if not cells:
        sentence = f"{NO_SNAPSHOT}. {sentence}".strip()
    else:
        sentence = f"{sentence} {working_lately.observational_caveat(snapshot, kind)}".strip()
    return ResultsSection(
        key=kind,
        title=KIND_TITLES.get(kind, kind),
        kind=kind,
        sentence=sentence,
        verdict_line=_verdict_line(snapshot, kind) if cells else NO_SNAPSHOT,
        bands=bands,
        rows=tuple(_cell_row(cell) for cell in rows),
        studies=tuple(_cell_row(cell) for cell in _by_rank(studies)),
        stats={
            "kind": kind,
            "cells": len(cells),
            "live_cells": len(live),
            "eligible_cells": len([cell for cell in live if _eligible(cell)]),
            "study_cells": len(studies),
        },
    )


def _bot_sections(snapshot: Mapping[str, Any] | None, horizon: str) -> tuple[ResultsSection, ...]:
    kinds = ("daytrade_held_run",) if horizon == "day" else ("swing_trade_r", "swing_favorable")
    return tuple(_bot_section(snapshot, kind) for kind in kinds)


# ---------------------------------------------------------------------------
# My trades
# ---------------------------------------------------------------------------


def _moment(value: Any):
    """A market-local moment, through the journal's OWN coercion.

    Imported rather than re-implemented: `journal_trade_shape` owns what a
    journal timestamp means, including that a naive one is market-local, and a
    second parser here would be a second opinion about the trader's own clock.
    """
    return journal_trade_shape._coerce_datetime(value)  # noqa: SLF001


def holding_bucket(trade: Any) -> str:
    """`day`, `swing` or `unknown_timing` for one closed trade.

    `unknown_timing` first: a broker file stamps both ends midnight
    market-local, which `journal_trade_shape.is_date_only` recognises as "the
    time is not known". Calling one of those a day trade because its two dates
    match would be inventing a fill time the file never carried.
    """
    opened = _moment(getattr(trade, "opened_at", ""))
    closed = _moment(getattr(trade, "closed_at", ""))
    if opened is None or closed is None:
        return "unknown_timing"
    if journal_trade_shape.is_date_only(opened) or journal_trade_shape.is_date_only(closed):
        return "unknown_timing"
    return "day" if opened.date() == closed.date() else "swing"


def _raw(trade: Any) -> dict[str, Any]:
    row = getattr(trade, "raw", None)
    return dict(row) if isinstance(row, Mapping) else {}


def _has_planned_risk(row: Mapping[str, Any]) -> bool:
    value = row.get("planned_risk")
    if value is None:
        return False
    return str(value).strip() != ""


def _pnl_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row.get(key) or 0.0))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _mine_stats(trades: Sequence[Any], analytics: Mapping[str, Any]) -> dict[str, Any]:
    """The bucket's counts. Integers at this table's own grain, never a rate scaled up."""
    rows = [_raw(trade) for trade in trades]
    overall = dict(analytics.get("overall") or {})
    key = str(analytics.get("pnl_key") or "net_pnl")
    values = _pnl_values(rows, key)
    fees = 0.0
    for trade in trades:
        try:
            fees += float(getattr(trade, "fees", None) or 0.0)
        except (TypeError, ValueError):
            continue
    confirmed = len([row for row in rows if journal_analytics._confirmed_setup_tags(row)])  # noqa: SLF001
    provisional = len([row for row in rows if journal_analytics._provisional_setup_tags(row)])  # noqa: SLF001
    return {
        "trade_ids": tuple(str(getattr(trade, "trade_id", "")) for trade in trades),
        "trades": overall.get("trades", len(trades)),
        "closed": overall.get("closed", len(trades)),
        "wins": overall.get("wins", 0),
        "losses": overall.get("losses", 0),
        "flats": len([value for value in values if value == 0.0]),
        "net_pnl": overall.get("net_pnl"),
        "fees": fees,
        "pnl_key": key,
        "pnl_note": analytics.get("pnl_note") or "",
        "currencies": tuple(analytics.get("currencies") or ()),
        "n_with_r": len([row for row in rows if _has_planned_risk(row)]),
        "confirmed_tagged": confirmed,
        "provisional_tagged": provisional,
        "total": len(trades),
    }


def _mine_row(entry: Mapping[str, Any]) -> ResultsRow:
    values = dict(entry)
    win_rate = entry.get("win_rate")
    net = entry.get("net_pnl")
    display = {
        "label": str(entry.get("label") or ""),
        "trades": str(entry.get("trades") or 0),
        "closed": str(entry.get("closed") or 0),
        "wins": str(entry.get("wins") or 0),
        "losses": str(entry.get("losses") or 0),
        "win_rate": "unmeasured" if win_rate is None else f"{float(win_rate):.4g}",
        "net_pnl": "unmeasured" if net is None else f"{float(net):,.2f}",
    }
    line = (
        f"{display['label']}: {display['closed']} closed, {display['wins']}W / "
        f"{display['losses']}L, net {display['net_pnl']}"
    )
    return ResultsRow(
        cell=None,
        line=line,
        reason="",
        eligible=True,
        values=values,
        display=display,
    )


def _exposure_text(trades: Sequence[Any]) -> str:
    """Options exposure BY INSTRUMENT, spelled ST5's way - never as direction.

    A long option is not a bullish setup: `journal_exposure` classifies the
    structure and the bias separately, and this page reports only what the
    instrument was. Counts, no money and no bias.
    """
    kinds: dict[str, int] = {}
    for trade in trades:
        kind = str(_raw(trade).get("instrument_kind") or "").strip().upper() or "UNKNOWN"
        kinds[kind] = kinds.get(kind, 0) + 1
    if not kinds:
        return ""
    return "by instrument: " + ", ".join(
        f"{name} {count}" for name, count in sorted(kinds.items())
    )


def _mine_section(key: str, trades: Sequence[Any], currency_mode: Any) -> ResultsSection:
    rows = [_raw(trade) for trade in trades]
    analytics = journal_analytics.build_analytics_summary(rows, currency_mode)
    stats = _mine_stats(trades, analytics)
    groups = (analytics.get("groups") or {}).get("my setups") or []
    if stats["confirmed_tagged"]:
        table = tuple(_mine_row(entry) for entry in groups)
        sentence = (
            f"{stats['total']} closed trade(s); {stats['confirmed_tagged']} carry a "
            f"tag the trader confirmed and {stats['provisional_tagged']} a provisional "
            f"one. {_exposure_text(trades)}".strip()
        )
    else:
        table = ()
        sentence = NO_CONFIRMED_TAGS
    bucket_line = (
        f"{stats['total']} trade(s), {stats['wins']}W / {stats['losses']}L / "
        f"{stats['flats']} flat, net "
        f"{'unmeasured' if stats['net_pnl'] is None else format(float(stats['net_pnl']), ',.2f')} "
        f"({format(stats['fees'], ',.2f')} in fees and commission), "
        f"{stats['n_with_r']} with planned risk recorded"
    )
    summary_row = ResultsRow(
        cell=None,
        line=bucket_line,
        reason="",
        eligible=True,
        values=dict(stats),
        display={
            "label": MINE_TITLES.get(key, key),
            "line": bucket_line,
            "headline": bucket_line,
        },
    )
    return ResultsSection(
        key=key,
        title=MINE_TITLES.get(key, key),
        kind="journal",
        sentence=sentence,
        verdict_line=bucket_line,
        bands=ResultsBands(stronger=(summary_row,)),
        rows=table,
        studies=(),
        stats=stats,
        analytics=analytics,
    )


def _mine_sections(
    trades: Sequence[Any], horizon: str, currency_mode: Any
) -> tuple[ResultsSection, ...]:
    closed = [trade for trade in trades if _is_closed(trade)]
    buckets: dict[str, list[Any]] = {"day": [], "swing": [], "unknown_timing": []}
    for trade in closed:
        buckets[holding_bucket(trade)].append(trade)
    key = "day" if horizon == "day" else "swing"
    return (
        _mine_section(key, buckets[key], currency_mode),
        _mine_section("unknown_timing", buckets["unknown_timing"], currency_mode),
    )


def _is_closed(trade: Any) -> bool:
    closed = getattr(trade, "is_closed", None)
    if isinstance(closed, bool):
        return closed
    return str(getattr(trade, "status", "") or "").upper() == "CLOSED"


# ---------------------------------------------------------------------------
# window and freshness
# ---------------------------------------------------------------------------


def _window_of(window: Any, as_of: Any) -> tuple[str, str, str, str]:
    """`(name, start, end, label)`. "Lately" is SESSIONS, walked on the calendar."""
    if isinstance(window, (tuple, list)) and len(window) == 2:
        start, end = str(window[0]), str(window[1])
        return "custom", start, end, f"Custom window {start} to {end}"
    if str(window) == "all":
        return "all", "", "", "All history - every session the evidence covers"
    start, end = evidence_stats.lately_window(as_of)
    return (
        "recent",
        start,
        end,
        f"Recent {evidence_stats.LATELY_SESSIONS} sessions ({start} to {end})",
    )


def _bot_freshness(snapshot: Mapping[str, Any] | None) -> str:
    snapshot = snapshot or {}
    identity = str(snapshot.get("snapshot_id") or "")
    if not identity:
        return NO_SNAPSHOT
    bits = [
        f"snapshot {identity[:8]}",
        f"as of {snapshot.get('as_of') or 'an unstated session'}",
        f"built {snapshot.get('built_at') or 'at an unstated time'}",
    ]
    for name, entry in sorted((snapshot.get("sources") or {}).items()):
        if not isinstance(entry, Mapping):
            continue
        bits.append(f"{name} <- {entry.get('path') or 'an unnamed file'} @ {entry.get('mtime')}")
    return "; ".join(bits)


def _mine_freshness(trades: Sequence[Any]) -> str:
    closed = [trade for trade in trades if _is_closed(trade)]
    stamps = [str(getattr(trade, "closed_at", "") or "") for trade in closed]
    stamps = [stamp for stamp in stamps if stamp]
    newest = max(stamps) if stamps else ""
    return (
        f"My trades; {len(closed)} closed trade(s) read; newest closed "
        f"{newest or 'never - no closed trade carries a time'}"
    )


# ---------------------------------------------------------------------------
# the view
# ---------------------------------------------------------------------------


def build_results_view(
    *,
    population: str,
    horizon: str,
    window: Any = "recent",
    snapshot: Mapping[str, Any] | None = None,
    journal_trades: Sequence[Any] | None = None,
    as_of: Any = None,
    currency_mode: Any = None,
) -> ResultsView:
    """One selection's whole readout. Pure: nothing here opens a file.

    `population` in {"bot", "mine"}, `horizon` in {"swing", "day"}, `window` in
    {"recent", "all", (start, end)}. A bot view reads ONLY the snapshot (through
    `working_lately.cells_from_payload`, because `snapshot["cells"]` is
    compacted); a My-trades view reads ONLY the journal. Neither ever sees the
    other's population.
    """
    population = "mine" if str(population) == "mine" else "bot"
    horizon = "day" if str(horizon) == "day" else "swing"
    trades = list(journal_trades or ())
    name, start, end, label = _window_of(window, as_of)
    if population == "mine":
        sections = _mine_sections(trades, horizon, currency_mode)
        freshness = _mine_freshness(trades)
    else:
        sections = _bot_sections(snapshot, horizon)
        freshness = _bot_freshness(snapshot)
    return ResultsView(
        population=population,
        horizon=horizon,
        window=name,
        window_label=label,
        window_start=start,
        window_end=end,
        sections=sections,
        freshness_line=f"{label}. {freshness}",
    )
