"""One drillable answer in three populations, cut by the measured environment.

Packet WS-10I item 2 (WISHLIST 10I). The trader asked one question - "how does
setup S do when the market is in state E?" - and it has three honest answers
that must never be added up:

1. **opportunity evidence** (`opportunity_cells`) - every recorded eligible
   example of the setup in that state, whether or not anybody traded it;
2. **personal execution** (`personal_cells`) - the trades actually taken, with
   the losses, the fees and whether a risk was ever planned;
3. **thesis review** (`thesis_review`) - what was EXPECTED against what
   happened, in three verdicts that are never merged into one grade.

**Shadow only.** Nothing here detects, scores, alerts, gates, ranks a
watchlist or promotes anything (plan.md sec 5): it reads rows other parts of
the desk already wrote and arranges them.

The rules this file is bound by
-------------------------------

* **The statistics are `evidence_stats`' and `swing_headline`'s own.** ONE
  Wilson (`swing_headline.wilson_lower_bound`), ONE floor
  (`evidence_stats.MIN_REPORTABLE_N`), ONE concentration limit
  (`working_lately.CONCENTRATION_LIMIT`). Nothing here invents a second.
* **Nothing pools theta, day trades and swings.** A sold put is premium, a
  same-session stock trade is a day trade, and a swing is a swing; each is
  counted by name and none is folded into another.
* **`unknown` is its own cell.** A session nobody labelled is not a quiet
  session, and pooling it into a measured cell would make up a reading.
* **Win rate / held-run FIRST, with n, distinct sessions, distinct symbols, the
  bound and the family's own baseline beside it** (decision 0016). The swing
  population leads with the favorable-direction rate (ST1: the tier file's
  `win` is the sign of a close-to-close percent move, never a stop-rule
  verdict) and the day-trade population with `held_run_score`.
* **A cell may be reported and still refuse to LEAD.** Below the floor, or more
  than `CONCENTRATION_LIMIT` of it from one name or one session, and the cell
  prints with its refusal - thirty rows of one ticker is the best-looking rate
  on any page and the least useful.
* **Money is counted once per trade.** Two confirmed tags are two statements
  about one trade: it shows in both cells and contributes its P&L once, the
  grain rule `preference_trade_outcomes.trade_level_summary` already owns.
* **Commission keeps the sign the importer gave it.** Nothing here `abs()`es a
  fee.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import context_join
import evidence_stats
import held_run_score
import journal_analytics
import swing_headline
import working_lately
from indicators.d1_environment import LABEL_UNKNOWN

#: The column a joined row carries its D1 label in - `d1_environment_join`'s own
#: name, imported as a string rather than restated, so one rename moves both.
DEFAULT_ENVIRONMENT_KEY = "d1_environment"

#: The two populations this module knows how to cut. A third is a third reader,
#: never a third branch inside one of these.
POPULATION_SWING = "swing"
POPULATION_DAY_TRADE = "day_trade"

#: What an unlabelled session reads as. `indicators.d1_environment`'s spelling.
UNKNOWN = LABEL_UNKNOWN

#: Why a cell that is on the page may not lead. Printed, never silent.
REFUSAL_FLOOR = (
    "below the evidence floor (n={n} against {floor}) - reported as discovery, "
    "not as an answer"
)
REFUSAL_CONCENTRATION = (
    "too concentrated to lead - {share:.0%} of this cell is one {label}, over "
    "the declared {limit:.0%} limit"
)
REFUSAL_UNMEASURED = "unmeasured - there is no statistic to lead with"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _field(trade: Any, name: str, default: Any = "") -> Any:
    """One value off a journal trade, whether it is a row or an object.

    The journal hands out objects with a `raw` mapping; a test and a CLI hand
    in plain rows. Both are the same trade and neither is the odd one out.
    """
    if isinstance(trade, Mapping):
        return trade.get(name, default)
    value = getattr(trade, name, None)
    if value is not None:
        return value
    raw = getattr(trade, "raw", None)
    if isinstance(raw, Mapping):
        return raw.get(name, default)
    return default


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN is not a number the desk may add up
        return default
    return number


def _environment_of(
    row: Mapping[str, Any],
    *,
    environment_key: str,
    labels_by_session: Mapping[str, str] | None,
    date_field: str,
) -> str:
    """The row's own joined label, else the label table, else `unknown`.

    The row's column wins because `d1_environment_join.attach_environment` is
    what put it there, under the benchmark and rule version the caller chose.
    """
    label = _text(row.get(environment_key))
    if label:
        return label
    if labels_by_session:
        session = _text(row.get(date_field))[:10]
        if session:
            return _text(labels_by_session.get(session)) or UNKNOWN
    return UNKNOWN


def _concentration(labels: Sequence[str]) -> dict[str, Any]:
    """`evidence_stats`' own concentration rule, never a second copy here."""
    return evidence_stats._concentration(list(labels))  # noqa: SLF001


def _lead_verdict(
    *,
    statistic: float | None,
    n: int,
    meets_floor: bool,
    symbol_share: float | None,
    session_share: float | None,
) -> tuple[bool, bool, str]:
    """`(can_lead, concentrated, refusal)` - the ONE eligibility rule.

    Order is deliberate: the floor first, because a three-row cell is not
    concentrated, it is empty. `CONCENTRATION_LIMIT` is EXCEEDED and never
    reached, so a cell split evenly over two sessions sits at exactly 0.5 and
    stays eligible - that is the smallest honest spread the desk sees.
    """
    limit = working_lately.CONCENTRATION_LIMIT
    concentrated = bool(
        (symbol_share is not None and symbol_share > limit)
        or (session_share is not None and session_share > limit)
    )
    if statistic is None:
        return False, concentrated, REFUSAL_UNMEASURED
    if not meets_floor:
        return False, concentrated, REFUSAL_FLOOR.format(
            n=n, floor=evidence_stats.MIN_REPORTABLE_N
        )
    if concentrated:
        worst, label = (
            (symbol_share, "name")
            if (symbol_share or 0.0) >= (session_share or 0.0)
            else (session_share, "session")
        )
        return False, True, REFUSAL_CONCENTRATION.format(
            share=worst or 0.0, label=label, limit=limit
        )
    return True, False, ""


# ---------------------------------------------------------------------------
# population 1 - opportunity evidence
# ---------------------------------------------------------------------------


def _swing_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    environment_key: str,
    labels_by_session: Mapping[str, str] | None,
    date_field: str,
    family_field: str,
    window_sessions: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    families: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        family = _text(row.get(family_field)) or UNKNOWN
        side = _text(row.get("side")).upper() or UNKNOWN
        environment = _environment_of(
            row,
            environment_key=environment_key,
            labels_by_session=labels_by_session,
            date_field=date_field,
        )
        groups.setdefault((family, side, environment), []).append(row)
        families.setdefault((family, side), []).append(row)

    baselines = {
        key: swing_headline.headline_from_tracker_rows(
            f"{key[1]} {key[0]}".strip(), group, sessions=window_sessions
        )
        for key, group in families.items()
    }

    cells: list[dict[str, Any]] = []
    for (family, side, environment), group in groups.items():
        headline = swing_headline.headline_from_tracker_rows(
            f"{side} {family} / {environment}".strip(), group, sessions=window_sessions
        )
        graded = [row for row in group if _is_graded(row)]
        symbols = _concentration([_text(row.get("symbol")).upper() for row in graded])
        sessions = _concentration([_text(row.get(date_field))[:10] for row in graded])
        can_lead, concentrated, refusal = _lead_verdict(
            statistic=headline.win_rate,
            n=headline.n,
            meets_floor=headline.meets_floor,
            symbol_share=symbols.get("top_share"),
            session_share=sessions.get("top_share"),
        )
        baseline = baselines[(family, side)]
        cells.append(
            {
                "population_kind": POPULATION_SWING,
                "family": family,
                "side": side,
                "environment": environment,
                # The statistic FIRST, and what it IS beside it (ST1).
                "statistic": headline.win_rate,
                "statistic_name": "favorable direction (close to close)",
                "lower_bound": headline.win_rate_lb,
                "uncertainty_kind": "wilson lower bound",
                "outcome_kind": headline.outcome_kind,
                "avg_r": headline.avg_r,
                "avg_unit": headline.avg_unit,
                "n": headline.n,
                "n_rows": len(group),
                "n_ungraded": len(group) - headline.n,
                "n_sessions": int(sessions.get("distinct") or 0),
                "n_symbols": int(symbols.get("distinct") or 0),
                "top_symbol": symbols.get("top"),
                "top_symbol_share": symbols.get("top_share"),
                "top_session": sessions.get("top"),
                "top_session_share": sessions.get("top_share"),
                "concentrated": concentrated,
                "can_lead": can_lead,
                "lead_refusal": refusal,
                "meets_floor": headline.meets_floor,
                "n_floor": evidence_stats.MIN_REPORTABLE_N,
                "window_sessions": int(window_sessions),
                "coverage": (headline.n / len(group)) if group else None,
                # The setup's OWN record, across every environment, so a cell is
                # read against the thing it is a cut of and never against the page.
                "baseline_statistic": baseline.win_rate,
                "baseline_lower_bound": baseline.win_rate_lb,
                "baseline_n": baseline.n,
                "line": headline.sentence(),
            }
        )
    cells.sort(key=lambda cell: (cell["family"], cell["side"], cell["environment"]))
    return cells


def _is_graded(row: Mapping[str, Any]) -> bool:
    """Whether `win` answered. A PRESENT AND EMPTY `win` is unmeasured.

    The same vocabulary `swing_headline.headline_from_tracker_rows` counts on,
    so the denominator here and the denominator there are the same rows.
    """
    verdict = _text(row.get("win")).lower()
    return verdict in {"1", "true", "yes", "win", "0", "false", "no", "loss"}


def _day_trade_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels_by_session: Mapping[str, str] | None,
    date_field: str,
    environment_key: str,
    as_of: Any,
    window_sessions: int,
) -> list[dict[str, Any]]:
    """The day-trade cut: `held_run_score` per (family, side, D1 environment).

    **The D1 label of the session, never the alert's own `market_environment`.**
    The outcome rows carry a `context_json.market_environment` stamped at alert
    registration - a different vocabulary, on a different clock, answering a
    different question. Reading it as the D1 label would print two unrelated
    words under one heading.
    """
    episodes = held_run_score.build_episodes(rows, as_of=as_of)
    sessions_by_event: dict[str, str] = {}
    for row in rows:
        event_id = _text(row.get("event_id"))
        if event_id and event_id not in sessions_by_event:
            sessions_by_event[event_id] = _text(row.get(date_field))[:10]

    groups: dict[tuple[str, str, str], held_run_score.Segment] = {}
    families: dict[tuple[str, str], held_run_score.Segment] = {}
    for episode in episodes:
        session = sessions_by_event.get(episode.event_id, "")
        environment = _environment_of(
            {environment_key: "", date_field: session},
            environment_key=environment_key,
            labels_by_session=labels_by_session,
            date_field=date_field,
        )
        family = episode.bounce_type or UNKNOWN
        side = (episode.direction or UNKNOWN).upper()
        key = (family, side, environment)
        segment = groups.get(key)
        if segment is None:
            segment = held_run_score.Segment(key=(family, "all", environment, "unknown"))
            groups[key] = segment
        segment.add(episode)
        family_segment = families.get((family, side))
        if family_segment is None:
            family_segment = held_run_score.Segment(key=(family, "all", "all", "unknown"))
            families[(family, side)] = family_segment
        family_segment.add(episode)

    cells: list[dict[str, Any]] = []
    for (family, side, environment), segment in groups.items():
        summary = segment.summary()
        concentration = summary.get("concentration") or {}
        symbol_share = ((concentration.get("by_symbol") or {}).get("top_share"))
        session_share = ((concentration.get("by_session") or {}).get("top_share"))
        can_lead, concentrated, refusal = _lead_verdict(
            statistic=summary.get("held_run_score"),
            n=int(summary.get("n_held") or 0),
            meets_floor=bool(summary.get("meets_floor")),
            symbol_share=symbol_share,
            session_share=session_share,
        )
        baseline = families[(family, side)].summary()
        bootstrap = summary.get("score_bootstrap") or {}
        cells.append(
            {
                "population_kind": POPULATION_DAY_TRADE,
                "family": family,
                "side": side,
                "environment": environment,
                "statistic": summary.get("held_run_score"),
                "statistic_name": held_run_score.HELD_RUN_STATISTIC_NAME,
                "lower_bound": bootstrap.get("low"),
                "uncertainty_kind": "session-block bootstrap low",
                "outcome_kind": held_run_score.HELD_RUN_OUTCOME_KIND,
                "n": int(summary.get("n") or 0),
                "n_rows": int(summary.get("n") or 0),
                "n_held": int(summary.get("n_held") or 0),
                "n_broken": int(summary.get("n_broken") or 0),
                "n_measured": int(summary.get("n_measured") or 0),
                "n_unmeasured": int(summary.get("n_unmeasured") or 0),
                "n_pending": int(summary.get("n_pending") or 0),
                # held / MEASURED, and the unmeasured are counted, never assumed (Q1).
                "hold_rate": summary.get("hold_rate"),
                "mean_mfe_r_of_held": summary.get("mean_mfe_r_of_held"),
                "n_sessions": int(summary.get("n_sessions") or 0),
                "n_symbols": int(summary.get("n_symbols") or 0),
                "top_symbol_share": symbol_share,
                "top_session_share": session_share,
                "concentrated": concentrated,
                "can_lead": can_lead,
                "lead_refusal": refusal,
                "meets_floor": bool(summary.get("meets_floor")),
                "n_floor": int(summary.get("n_floor") or evidence_stats.MIN_REPORTABLE_N),
                "window_sessions": int(window_sessions),
                "coverage": summary.get("coverage"),
                "baseline_statistic": baseline.get("held_run_score"),
                "baseline_n": int(baseline.get("n") or 0),
                "baseline_lower_bound": (baseline.get("score_bootstrap") or {}).get("low"),
                "alert_environment_is_not_this": True,
            }
        )
    cells.sort(key=lambda cell: (cell["family"], cell["side"], cell["environment"]))
    return cells


def opportunity_cells(
    rows: Iterable[Mapping[str, Any]] | None,
    *,
    environment_key: str = DEFAULT_ENVIRONMENT_KEY,
    population_kind: str = POPULATION_SWING,
    labels_by_session: Mapping[str, str] | None = None,
    date_field: str = "scan_date",
    family_field: str = "setup_family",
    window_sessions: int = evidence_stats.LATELY_SESSIONS,
    as_of: Any = None,
) -> list[dict[str, Any]]:
    """Every recorded eligible example of a setup, cut by the state it happened in.

    `rows` arrive ALREADY windowed and already made eligible by their own
    reader - `swing_evidence.read_eligible_rows(..., window=lately_window())`
    for the swing population, `held_run_score.read_outcome_rows` for the day
    one. This function windows nothing: `window_sessions` is the length the
    caller declares so each cell can SAY what it covers, and a cell that said
    "20 sessions" while its reader had handed over six months would be the
    heading lying about the number under it.

    Nothing is dropped. A cell under the floor and a cell made of one ticker
    both appear, both carrying the reason they may not lead.
    """
    table = [row for row in (rows or ()) if isinstance(row, Mapping)]
    if str(population_kind) == POPULATION_DAY_TRADE:
        return _day_trade_cells(
            table,
            labels_by_session=labels_by_session,
            date_field=date_field,
            environment_key=environment_key,
            as_of=as_of,
            window_sessions=window_sessions,
        )
    return _swing_cells(
        table,
        environment_key=environment_key,
        labels_by_session=labels_by_session,
        date_field=date_field,
        family_field=family_field,
        window_sessions=window_sessions,
    )


# ---------------------------------------------------------------------------
# population 2 - personal execution
# ---------------------------------------------------------------------------

#: Why a trade is not in the swing cells. Each reason is its OWN sentence: a
#: single "excluded" bucket would hide that three different things happened.
REASON_THETA = "premium sold on an option - theta, never pooled with a swing"
REASON_DAY_TRADE = "opened and closed in one session - a day trade, not a swing"
REASON_SWING_NOT_DAY = "held past its session - a swing, not a day trade"
REASON_UNCONFIRMED = "no tag the trader confirmed - a machine guess is not my setup"
REASON_OPEN = "still open - an open mark is exposure, not a result"
REASON_UNKNOWN_TIMING = "the fill times are not known, so the holding period is not either"

#: The ST5 status partition. `uncertain` is a CROSS-CUTTING label that pools no
#: money, so it is not one of these.
STATUS_COMPLETE = "complete"
STATUS_PARTLY_CLOSED = "partly_closed"
STATUS_OPEN = "open_exposure"


def _is_option(trade: Any) -> bool:
    kind = _text(_field(trade, "security_type")).upper()
    return kind in {"OPT", "OPTION", "FOP"}


def _holding(trade: Any) -> str:
    """`day`, `swing` or `unknown_timing`, by the journal's own clock rules.

    `unknown_timing` FIRST: a broker file stamps both ends midnight
    market-local, and calling one of those a day trade because its two dates
    match would invent a fill time the file never carried.
    """
    import journal_trade_shape  # noqa: PLC0415

    opened = journal_trade_shape._coerce_datetime(_field(trade, "opened_at"))  # noqa: SLF001
    closed = journal_trade_shape._coerce_datetime(_field(trade, "closed_at"))  # noqa: SLF001
    if opened is None or closed is None:
        return "unknown_timing"
    if journal_trade_shape.is_date_only(opened) or journal_trade_shape.is_date_only(closed):
        return "unknown_timing"
    return "day" if opened.date() == closed.date() else "swing"


def _status_of(trade: Any) -> str:
    from journal_store import is_partly_closed

    status = _text(_field(trade, "status")).upper()
    # The journal writes CLOSED_PARTIAL; older spellings are still accepted.
    if is_partly_closed(status):
        return STATUS_PARTLY_CLOSED
    if status == "CLOSED":
        return STATUS_COMPLETE
    return STATUS_OPEN


def _confirmed_tags(trade: Any) -> list[str]:
    """The tags the TRADER stands behind - `journal_analytics`' own rule."""
    status = _text(_field(trade, "tag_status")) or journal_analytics.TAG_STATUS_CONFIRMED
    if status != journal_analytics.TAG_STATUS_CONFIRMED:
        return []
    return journal_analytics.split_tags(_field(trade, "setup_tags"))


def _money(trade: Any) -> dict[str, Any]:
    return {
        "net_pnl": _float(_field(trade, "net_pnl"), 0.0) or 0.0,
        "fees": _float(_field(trade, "fees"), 0.0) or 0.0,
        # The SIGN the importer gave it. Nothing here abs()es a commission.
        "commission": _float(_field(trade, "commission"), 0.0) or 0.0,
        "has_planned_risk": _text(_field(trade, "planned_risk")) != "",
    }


def personal_cells(
    trades: Iterable[Any] | None,
    *,
    labels_by_session: Mapping[str, str] | None = None,
    horizon: str = "swing",
    benchmark: str = context_join.DEFAULT_BENCHMARK,
) -> dict[str, Any]:
    """The trader's OWN execution in each state, by the ENTRY context.

    A trade is placed by the environment known when the fill happened, not the
    one known when the opportunity was seen: those are two different questions
    and `context_join` answers both separately.

    One trade with two confirmed tags appears in two cells - that is what the
    cells are for - and its money lands in `totals` ONCE.
    `totals["duplicate_tag_rows"]` is the difference between the two grains, so
    a reader can see the cell sum is larger ON PURPOSE.

    Every trade handed in is accounted for: either it is inside a cell or it is
    in `excluded` with the reason, by name. Nothing is silently dropped.
    """
    wanted = "day" if str(horizon) == "day" else "swing"
    rows = list(trades or ())
    excluded: dict[str, str] = {}
    populations: dict[str, dict[str, Any]] = {
        STATUS_COMPLETE: {"n_trades": 0, "net_pnl": 0.0},
        STATUS_PARTLY_CLOSED: {"n_trades": 0, "net_pnl": 0.0},
        # An open position has NO result and its mark is never money (ST5).
        STATUS_OPEN: {"n_trades": 0, "net_pnl": None},
        "theta": {"n_trades": 0, "net_pnl": 0.0},
        "day_trade": {"n_trades": 0, "net_pnl": 0.0},
        "swing": {"n_trades": 0, "net_pnl": 0.0},
        "unconfirmed": {"n_trades": 0, "net_pnl": 0.0},
        "unknown_timing": {"n_trades": 0, "net_pnl": 0.0},
    }

    kept: list[tuple[Any, list[str]]] = []
    for trade in rows:
        trade_id = _text(_field(trade, "trade_id")) or _text(_field(trade, "id"))
        money = _money(trade)
        status = _status_of(trade)
        populations[status]["n_trades"] += 1
        if populations[status]["net_pnl"] is not None:
            populations[status]["net_pnl"] += money["net_pnl"]
        if status == STATUS_OPEN:
            excluded[trade_id] = REASON_OPEN
            continue
        if _is_option(trade):
            populations["theta"]["n_trades"] += 1
            populations["theta"]["net_pnl"] += money["net_pnl"]
            excluded[trade_id] = REASON_THETA
            continue
        holding = _holding(trade)
        populations.setdefault(holding, {"n_trades": 0, "net_pnl": 0.0})
        populations[holding]["n_trades"] += 1
        populations[holding]["net_pnl"] += money["net_pnl"]
        if holding == "unknown_timing":
            excluded[trade_id] = REASON_UNKNOWN_TIMING
            continue
        if holding != wanted:
            excluded[trade_id] = REASON_DAY_TRADE if holding == "day" else REASON_SWING_NOT_DAY
            continue
        tags = _confirmed_tags(trade)
        if not tags:
            populations["unconfirmed"]["n_trades"] += 1
            populations["unconfirmed"]["net_pnl"] += money["net_pnl"]
            excluded[trade_id] = REASON_UNCONFIRMED
            continue
        kept.append((trade, tags))

    # The ENTRY context, one ref per kept trade, side by side with nothing else.
    refs = [dict(row=trade, tags=tags) for trade, tags in kept]
    clocks = [
        {"opened_at": _field(trade, "opened_at"), "trade_id": _text(_field(trade, "trade_id"))}
        for trade, _tags in kept
    ]
    context_join.attach_context(
        clocks,
        when=context_join.WHEN_ENTRY,
        labels_by_session=labels_by_session,
        clock_field="opened_at",
        benchmark=benchmark,
    )

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    totals = {
        "n_trades": 0,
        "net_pnl": 0.0,
        "fees": 0.0,
        "commission": 0.0,
        "n_with_planned_risk": 0,
        "wins": 0,
        "losses": 0,
        "flats": 0,
        "duplicate_tag_rows": 0,
    }
    counted: set[str] = set()
    for index, (trade, tags) in enumerate(kept):
        ref = clocks[index][context_join.ENTRY_CONTEXT_FIELD]
        trade_id = _text(_field(trade, "trade_id"))
        money = _money(trade)
        for tag in tags:
            key = (tag, ref.label)
            cell = cells.get(key)
            if cell is None:
                cell = {
                    "setup": tag,
                    "environment": ref.label,
                    "environment_certainty": ref.certainty,
                    "environment_flagged": ref.flagged,
                    "basis": context_join.WHEN_ENTRY,
                    "trade_ids": [],
                    "n_trades": 0,
                    "net_pnl": 0.0,
                    "fees": 0.0,
                    "commission": 0.0,
                    "wins": 0,
                    "losses": 0,
                    "flats": 0,
                    "n_with_planned_risk": 0,
                }
                cells[key] = cell
            if trade_id not in cell["trade_ids"]:
                cell["trade_ids"].append(trade_id)
                cell["n_trades"] += 1
                cell["net_pnl"] += money["net_pnl"]
                cell["fees"] += money["fees"]
                cell["commission"] += money["commission"]
                cell["n_with_planned_risk"] += 1 if money["has_planned_risk"] else 0
                if money["net_pnl"] > 0:
                    cell["wins"] += 1
                elif money["net_pnl"] < 0:
                    cell["losses"] += 1
                else:
                    cell["flats"] += 1
        if trade_id in counted:
            continue
        counted.add(trade_id)
        totals["n_trades"] += 1
        totals["net_pnl"] += money["net_pnl"]
        totals["fees"] += money["fees"]
        totals["commission"] += money["commission"]
        totals["n_with_planned_risk"] += 1 if money["has_planned_risk"] else 0
        if money["net_pnl"] > 0:
            totals["wins"] += 1
        elif money["net_pnl"] < 0:
            totals["losses"] += 1
        else:
            totals["flats"] += 1
    totals["duplicate_tag_rows"] = len(
        [trade_id for trade_id, _tags in _tag_counts(kept).items() if _tags > 1]
    )

    ordered = sorted(cells.values(), key=lambda cell: (cell["setup"], cell["environment"]))
    for cell in ordered:
        cell["statistic"] = (cell["wins"] / cell["n_trades"]) if cell["n_trades"] else None
        cell["statistic_name"] = "profitable trades"
        cell["lower_bound"] = swing_headline.wilson_lower_bound(cell["wins"], cell["n_trades"])
        cell["meets_floor"] = cell["n_trades"] >= evidence_stats.MIN_REPORTABLE_N
        cell["n_floor"] = evidence_stats.MIN_REPORTABLE_N

    meets_floor = totals["n_trades"] >= evidence_stats.MIN_REPORTABLE_N
    best = None
    if meets_floor:
        eligible = [cell for cell in ordered if cell["meets_floor"] and cell["lower_bound"] is not None]
        if eligible:
            best = max(eligible, key=lambda cell: cell["lower_bound"])["setup"]
    return {
        "population_kind": "personal",
        "horizon": wanted,
        "basis": context_join.WHEN_ENTRY,
        "benchmark": benchmark,
        "cells": ordered,
        "totals": totals,
        "excluded": excluded,
        "populations": populations,
        # No "best" word below the floor: three trades is not an answer, and a
        # page that named one anyway would be the floor printed and ignored.
        "best_setup": best,
        "meets_floor": meets_floor,
        "n_floor": evidence_stats.MIN_REPORTABLE_N,
    }


def _tag_counts(kept: Sequence[tuple[Any, Sequence[str]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade, tags in kept:
        trade_id = _text(_field(trade, "trade_id"))
        counts[trade_id] = counts.get(trade_id, 0) + len(tags)
    return counts


# ---------------------------------------------------------------------------
# population 3 - thesis review, in THREE verdicts
# ---------------------------------------------------------------------------

CALL_RIGHT = "right"
CALL_WRONG = "wrong"
CALL_OPEN = "open"

#: Which stances point which way. Anything else is not a directional claim and
#: is left OPEN rather than graded against a tape it never named.
_UP_STANCES = {"bullish", "long", "up", "higher"}
_DOWN_STANCES = {"bearish", "short", "down", "lower"}


def _session_after(start: Any, sessions: int) -> str:
    """The ISO session `sessions` trading days after `start`, or `""`."""
    import market_calendar  # noqa: PLC0415
    from datetime import date as _date, timedelta as _timedelta  # noqa: PLC0415

    try:
        cursor = _date.fromisoformat(str(start)[:10])
    except ValueError:
        return ""
    for _ in range(max(int(sessions or 0), 0)):
        moved = False
        for _step in range(30):
            cursor = cursor + _timedelta(days=1)
            try:
                if market_calendar.is_session(cursor):
                    moved = True
                    break
            except Exception:  # noqa: BLE001 - outside the calendar is uncertainty
                return ""
        if not moved:
            return ""
    return cursor.isoformat()


def _market_call(
    thesis: Mapping[str, Any],
    paths: Mapping[str, Mapping[str, Any]] | None,
    as_of: Any,
) -> dict[str, Any]:
    """Right, wrong or OPEN, by the benchmark's own later path.

    A call is graded only when the path HAS a close at the horizon's last
    session. Not reached, not recorded and not directional are all `open`: the
    desk says it cannot answer rather than calling a thesis wrong because the
    data stops.
    """
    stance = _text(thesis.get("stance")).lower()
    benchmarks = [str(one).strip().upper() for one in (thesis.get("benchmarks") or ())]
    table: Mapping[str, Any] = {}
    benchmark = ""
    for name in benchmarks:
        candidate = (paths or {}).get(name)
        if isinstance(candidate, Mapping) and candidate:
            table, benchmark = candidate, name
            break
    start_session = _text(thesis.get("session_date")) or _text(thesis.get("created_at"))[:10]
    try:
        sessions = int(thesis.get("horizon_sessions") or context_join.DEFAULT_THESIS_SESSIONS)
    except (TypeError, ValueError):
        sessions = context_join.DEFAULT_THESIS_SESSIONS
    end_session = _session_after(start_session, sessions)
    invalidated = _text(thesis.get("invalidated_at"))[:10]
    if invalidated and end_session and invalidated < end_session:
        end_session = invalidated
    payload = {
        "market_call": CALL_OPEN,
        "market_call_reason": "",
        "market_call_basis": {
            "benchmark": benchmark,
            "stance": stance,
            "start_session": start_session,
            "end_session": end_session,
            "start_close": None,
            "end_close": None,
            "as_of": str(as_of or "")[:10],
        },
    }
    if not table:
        payload["market_call_reason"] = "no recorded path for this thesis's benchmark"
        return payload
    start_close = _float(table.get(start_session))
    end_close = _float(table.get(end_session)) if end_session else None
    payload["market_call_basis"]["start_close"] = start_close
    payload["market_call_basis"]["end_close"] = end_close
    if start_close is None:
        payload["market_call_reason"] = "no close recorded on the session it was written in"
        return payload
    if end_close is None:
        payload["market_call_reason"] = (
            "the horizon has no recorded close yet - open, never wrong"
        )
        return payload
    if stance not in _UP_STANCES and stance not in _DOWN_STANCES:
        payload["market_call_reason"] = "the stance names no direction, so nothing can grade it"
        return payload
    move = end_close - start_close
    if move == 0:
        payload["market_call_reason"] = "the benchmark finished flat"
        return payload
    up = move > 0
    right = up if stance in _UP_STANCES else not up
    payload["market_call"] = CALL_RIGHT if right else CALL_WRONG
    payload["market_call_reason"] = (
        f"{benchmark} {start_close:.2f} on {start_session} to {end_close:.2f} on "
        f"{end_session} against a {stance} stance"
    )
    return payload


def thesis_review(
    theses: Sequence[Mapping[str, Any]] | None,
    *,
    opportunity_rows: Sequence[Mapping[str, Any]] | None = None,
    trades: Sequence[Any] | None = None,
    benchmark_paths: Mapping[str, Mapping[str, Any]] | None = None,
    as_of: Any = None,
    labels_by_session: Mapping[str, str] | None = None,
    benchmark: str = context_join.DEFAULT_BENCHMARK,
) -> list[dict[str, Any]]:
    """Per thesis: what was expected, what happened, and whether the trade paid.

    **Three verdicts, three keys, never one grade.** `market_call` is about the
    tape, `setup_held` about the opportunities the thesis covered, and
    `trade_profitable` about money. The case that makes the rule is the common
    one: the call was RIGHT and the trade LOST. Blending those into a single
    score destroys exactly the information the trader keeps a thesis for.

    The inputs are copied before they are linked, so a caller's rows come back
    the way they were handed in.
    """
    rows = [dict(row) for row in (opportunity_rows or ()) if isinstance(row, Mapping)]
    context_join.link_theses(rows, theses, benchmark=benchmark)

    trade_rows = []
    for trade in trades or ():
        trade_rows.append(
            {
                "trade_id": _text(_field(trade, "trade_id")),
                "opened_at": _field(trade, "opened_at"),
                "money": _money(trade),
            }
        )
    context_join.attach_context(
        trade_rows,
        when=context_join.WHEN_ENTRY,
        labels_by_session=labels_by_session,
        clock_field="opened_at",
        benchmark=benchmark,
    )
    context_join.link_theses(
        trade_rows, theses, when=context_join.WHEN_ENTRY, benchmark=benchmark
    )

    reviews: list[dict[str, Any]] = []
    for thesis in theses or ():
        if not isinstance(thesis, Mapping):
            continue
        thesis_id = _text(thesis.get("thesis_id"))
        linked_rows = [
            row
            for row in rows
            if any(
                _text(entry.get("thesis_id")) == thesis_id
                for entry in (row.get(context_join.LINKED_THESES_FIELD) or ())
            )
        ]
        linked_trades = [
            row
            for row in trade_rows
            if any(
                _text(entry.get("thesis_id")) == thesis_id
                for entry in (row.get(context_join.LINKED_THESES_FIELD) or ())
            )
        ]
        headline = swing_headline.headline_from_tracker_rows(thesis_id, linked_rows)
        symbols = _concentration(
            [_text(row.get("symbol")).upper() for row in linked_rows if _is_graded(row)]
        )
        sessions = _concentration(
            [_text(row.get("scan_date"))[:10] for row in linked_rows if _is_graded(row)]
        )
        seen: set[str] = set()
        net = 0.0
        fees = 0.0
        for row in linked_trades:
            trade_id = row["trade_id"]
            if trade_id in seen:
                continue
            seen.add(trade_id)
            net += row["money"]["net_pnl"]
            fees += row["money"]["fees"]
        review = {
            "thesis_id": thesis_id,
            "stance": _text(thesis.get("stance")),
            "claim": _text(thesis.get("claim")),
            "benchmarks": [str(one) for one in (thesis.get("benchmarks") or ())],
            # 1. the tape
            **_market_call(thesis, benchmark_paths, as_of),
            # 2. the opportunities
            "setup_held": {
                "n": headline.n,
                "n_rows": len(linked_rows),
                "statistic": headline.win_rate,
                "statistic_name": "favorable direction (close to close)",
                "lower_bound": headline.win_rate_lb,
                "outcome_kind": headline.outcome_kind,
                "meets_floor": headline.meets_floor,
                "n_floor": evidence_stats.MIN_REPORTABLE_N,
                "n_symbols": int(symbols.get("distinct") or 0),
                "n_sessions": int(sessions.get("distinct") or 0),
            },
            # 3. the money, once per trade
            "trade_profitable": {
                "profitable": (net > 0) if seen else None,
                "net_pnl": net if seen else None,
                "fees": fees if seen else None,
                "n_trades": len(seen),
            },
            "linked_observation_ids": [
                _text(row.get("observation_id")) for row in linked_rows
            ],
            "linked_trade_ids": sorted(seen),
            "as_of": str(as_of or "")[:10],
        }
        reviews.append(review)
    return reviews
