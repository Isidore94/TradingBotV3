"""Grades per structural regime (S16 item 3). Pure: rows in, plain dicts out.

Trader, 2026-09-26: "what's important is KNOWING the market regime and then
having setups you KNOW work in it ... it just is what it is." So every grade here
is per regime (the trader's structural regime on the row's date, `regime_join`),
never pooled across regimes; a cell with no rows in the current regime says
"untested in this regime" instead of a guess. The existing pooled grade
(`setup_grades`) rides beside it labelled "all regimes"; badges, sorting and the
Show filter keep reading that pooled grade and nothing here changes it.

* Swing: the tracker's episodes (`looking_back.swing_pick_results`, full
  history), per ``side|bucket|family`` and regime, on the SAME ladder
  (`setup_grades.grade_for`). LONG is judged on the raw result (closed R > 0);
  SHORT on the win vs SPY once 30+ picks have one, with the raw win beside it.
* Day trade: `setup_grades.daytrade_cells` over the alerts of that regime.
* Journal trades, Focus and the liked / vetoed / passed / rejected cohorts:
  counts and raw win rate per regime and side (shorts also vs SPY).

Presentation only: nothing here feeds a detector, a score, an alert, Focus,
the queue or ``review_policy.json``.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Mapping

import regime_join
import setup_grades

SCHEMA = "setup_grades_by_regime_v1"
UNTESTED = regime_join.UNTESTED
ALL_REGIMES = regime_join.ALL_REGIMES
#: Focus / cohort horizon read per regime (sessions).
COHORT_HORIZON = 5
#: The cohort families of `human_focus_tracking.COHORT_BASE_BY_SOURCE_PREFIX`, most specific first.
_COHORT_PREFIXES = (
    ("human_focus_swing", "focus_swing"),
    ("human_focus_m5", "focus_m5"),
    ("human_focus_veto", "veto"),
    ("human_focus_like", "like"),
    ("human_focus_pass", "pass"),
    ("human_focus_rejection", "focus_"),
    ("human_focus_pick", "focus_pick"),
)
KIND_ORDER = ("swing", "day trade", "cohort", "journal")

_SWING_FIELDS = (
    "grade", "n", "wins", "sessions", "win_rate", "low_bound", "avg_r", "cum_r_lately",
    "grade_basis", "tape_n", "tape_wins", "tape_win_rate", "tape_low_bound",
)
_DAY_FIELDS = (
    "grade", "n", "wins", "sessions", "win_rate", "low_bound", "avg_r", "cum_r_lately",
    "grade_2r", "n_2r", "eod_r_mean", "undecided", "open",
)


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _day(value: Any) -> str:
    return str(value or "").strip()[:10]


def _trim(cell: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {name: cell.get(name) for name in fields if name in cell}


# ---------------------------------------------------------------------------
# swing
# ---------------------------------------------------------------------------


def swing_cells_by_regime(
    picks: Iterable[Mapping[str, Any]] | None,
    horizon_index: Mapping[tuple[str, str, str], Mapping[str, Any]] | None,
    spy_closes: Mapping[str, float] | None,
    joiner: regime_join.Joiner,
    *,
    as_of: str = "",
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{swing key: {regime: graded cell}}``. Each regime is its own population.

    A pick counts once its representative R is closed (flat is not a win). LONG
    grades on that raw win; SHORT passes its wins vs SPY (`setup_grades.tape_result`)
    to the ladder, which reads them once 30+ are decided.
    """
    index = horizon_index or {}
    tallies: dict[tuple[str, str], dict[str, Any]] = {}
    meta: dict[str, tuple[str, str, str]] = {}
    for pick in picks or ():
        side = str(pick.get("side") or "").strip().upper()
        if side not in {"LONG", "SHORT"}:
            continue
        key = setup_grades.swing_key(side, pick.get("bucket"), pick.get("family"))
        meta.setdefault(key, (side, str(pick.get("bucket") or "").strip(), str(pick.get("family") or "general")))
        session = _day(pick.get("session"))
        regime = joiner.label(session)
        tally = tallies.setdefault(
            (key, regime),
            {"rs": [], "wins": 0, "days": set(), "tape_wins": 0, "tape_n": 0, "tape_days": set(), "tape_unknown": 0},
        )
        r = _float(pick.get("r"))
        if r is not None:
            tally["rs"].append(r)
            tally["wins"] += 1 if r > 0 else 0
            tally["days"].add(session)
        if side == "SHORT":
            row = index.get((str(pick.get("symbol") or "").strip().upper(), side, session))
            outcome = setup_grades.tape_result(pick, row, spy_closes, as_of=as_of)
            if outcome == setup_grades.UNKNOWN:
                tally["tape_unknown"] += 1
            else:
                tally["tape_n"] += 1
                tally["tape_wins"] += 1 if outcome == setup_grades.WIN else 0
                tally["tape_days"].add(session)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (key, regime), tally in tallies.items():
        side, bucket, family = meta[key]
        rs = tally["rs"]
        n = len(rs)
        tape = None
        if side == "SHORT":
            tape = {
                "wins": tally["tape_wins"], "n": tally["tape_n"],
                "sessions": len(tally["tape_days"]), "unknown": tally["tape_unknown"],
            }
        cell = setup_grades.grade_for(
            n=n,
            sessions=len(tally["days"]),
            wins=tally["wins"],
            avg_r=(sum(rs) / n) if n else None,
            cum_r_lately=round(sum(rs), 6) if n else None,
            tape=tape,
        )
        trimmed = _trim(cell, _SWING_FIELDS)
        trimmed["basis"] = "raw" if side == "LONG" else (
            "vs SPY" if cell.get("grade_basis") == "tape" else "raw (vs SPY under 30)"
        )
        out.setdefault(key, {"_meta": {"side": side, "bucket": bucket, "family": family}})[regime] = trimmed
    return out


# ---------------------------------------------------------------------------
# day trade
# ---------------------------------------------------------------------------


def daytrade_cells_by_regime(
    bracket: Iterable[Mapping[str, Any]] | None,
    joiner: regime_join.Joiner,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{daytrade key: {regime: graded cell}}`` from `setup_grades.bracket_results`."""
    by_regime: dict[str, list[Mapping[str, Any]]] = {}
    for result in bracket or ():
        by_regime.setdefault(joiner.label(result.get("trade_date")), []).append(result)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for regime, results in by_regime.items():
        for cell in setup_grades.daytrade_cells(results):
            key = str(cell.get("key"))
            out.setdefault(key, {"_meta": {"side": cell.get("side"), "bounce_type": cell.get("bounce_type")}})
            out[key][regime] = _trim(cell, _DAY_FIELDS)
    return out


# ---------------------------------------------------------------------------
# journal trades and the Focus / cohort outcomes
# ---------------------------------------------------------------------------


def _rate_cell(values: list[tuple[bool, str]], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    n = len(values)
    wins = sum(1 for won, _day_text in values if won)
    return {
        "n": n,
        "wins": wins,
        "sessions": len({day for _won, day in values}),
        "win_rate": (wins / n) if n else None,
        "low_bound": setup_grades.wilson_lower_bound(wins, n),
        **dict(extra or {}),
    }


def journal_by_regime(
    trades: Iterable[Mapping[str, Any]] | None,
    joiner: regime_join.Joiner,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{SIDE: {regime: {n, wins, win_rate, low_bound, pnl}}}`` over CLOSED trades.

    The regime is the one on the entry date (``opened_at``, else ``trade_date``);
    a win is net P&L > 0 (raw, both sides).
    """
    acc: dict[tuple[str, str], dict[str, Any]] = {}
    for trade in trades or ():
        if str(trade.get("status") or "").strip().lower() != "closed":
            continue
        direction = str(trade.get("direction") or "").strip().upper()
        side = "SHORT" if direction.startswith("SHORT") else ("LONG" if direction.startswith("LONG") else "")
        pnl = _float(trade.get("net_pnl"))
        if not side or pnl is None:
            continue
        day = _day(trade.get("opened_at")) or _day(trade.get("trade_date"))
        entry = acc.setdefault((side, joiner.label(day)), {"values": [], "pnl": 0.0})
        entry["values"].append((pnl > 0, day))
        entry["pnl"] += pnl
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (side, regime), entry in acc.items():
        out.setdefault(side, {})[regime] = _rate_cell(entry["values"], {"pnl": round(entry["pnl"], 2)})
    return out


def cohort_of(source: Any) -> str:
    """The cohort family a Focus / cohort outcome row belongs to (`human_focus_tracking`'s)."""
    text = str(source or "focus_pick").strip() or "focus_pick"
    for base, prefix in _COHORT_PREFIXES:
        if text == prefix or text.startswith(prefix + "_"):
            return base
    return "human_focus_pick"


def cohorts_by_regime(
    outcome_rows: Iterable[Mapping[str, Any]] | None,
    joiner: regime_join.Joiner,
    spy_closes: Mapping[str, float] | None = None,
    *,
    horizon: int = COHORT_HORIZON,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{cohort|SIDE: {regime: cell}}`` at ``horizon`` sessions, dated by the pick day.

    A win is the side-adjusted return > 0 (raw). SHORT cells also carry the win
    vs SPY's same-side return over the same sessions (unknown without SPY).
    """
    closes = spy_closes or {}
    acc: dict[tuple[str, str], dict[str, Any]] = {}
    for row in outcome_rows or ():
        value = _float(row.get(f"h{horizon}_return"))
        if value is None:
            continue
        side = "SHORT" if str(row.get("side") or "").strip().upper().startswith("SHORT") else "LONG"
        day = _day(row.get("trade_date"))
        key = f"{cohort_of(row.get('source'))}|{side}"
        entry = acc.setdefault((key, joiner.label(day)), {"values": [], "tape": []})
        entry["values"].append((value > 0, day))
        if side == "SHORT":
            start, end = _float(closes.get(_day(row.get("entry_date")))), _float(closes.get(_day(row.get(f"h{horizon}_date"))))
            if start and start > 0 and end is not None:
                spy_side = -(end / start - 1.0)
                entry["tape"].append((value > spy_side, day))
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (key, regime), entry in acc.items():
        cell = _rate_cell(entry["values"])
        if key.endswith("|SHORT"):
            tape = entry["tape"]
            cell["tape_n"] = len(tape)
            cell["tape_win_rate"] = (sum(1 for won, _ in tape if won) / len(tape)) if tape else None
        out.setdefault(key, {})[regime] = cell
    return out


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def current_regime(segments: Iterable[Mapping[str, Any]] | None, today: Any) -> dict[str, Any] | None:
    """The segment in force on ``today`` with its calendar day count, or None."""
    day = _day(today)
    found = None
    for segment in sorted(segments or (), key=lambda s: _day(s.get("start_date"))):
        start = _day(segment.get("start_date"))
        if start and day and start <= day:
            found = segment
    if found is None:
        return None
    start = _day(found.get("start_date"))
    count = (date.fromisoformat(day) - date.fromisoformat(start)).days + 1
    regime = str(found.get("regime") or "")
    return {"regime": regime, "label": regime_join.regime_label(regime), "start_date": start, "day_count": count}


def build_payload(
    *,
    segments: Iterable[Mapping[str, Any]] | None,
    today: Any,
    swing_picks: Iterable[Mapping[str, Any]] | None = (),
    horizon_index: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None,
    spy_closes: Mapping[str, float] | None = None,
    bracket: Iterable[Mapping[str, Any]] | None = (),
    trades: Iterable[Mapping[str, Any]] | None = (),
    cohort_rows: Iterable[Mapping[str, Any]] | None = (),
    grades: Mapping[str, Any] | None = None,
    as_of: str = "",
    m5_window: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Everything the two trackers show per regime. ``grades`` is the pooled payload."""
    timeline = list(segments or ())
    joiner = regime_join.Joiner(timeline)
    current = current_regime(timeline, today)
    swing = swing_cells_by_regime(swing_picks, horizon_index, spy_closes, joiner, as_of=as_of)
    day = daytrade_cells_by_regime(bracket, joiner)
    journal = journal_by_regime(trades, joiner)
    cohorts = cohorts_by_regime(cohort_rows, joiner, spy_closes)
    pooled_swing = setup_grades.swing_lookup(grades)
    pooled_day = setup_grades.daytrade_lookup(grades)
    present: set[str] = set()

    def entries(cells: Mapping[str, Mapping[str, Any]], pooled: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, by_regime in cells.items():
            regimes = {name: cell for name, cell in by_regime.items() if name != "_meta"}
            present.update(regimes)
            whole = pooled.get(key)
            out[key] = {
                **dict(by_regime.get("_meta") or {}),
                "all": {"grade": whole.get("grade"), "n": whole.get("n")} if whole else None,
                "by_regime": regimes,
            }
        return out

    swing_out = entries(swing, pooled_swing)
    day_out = entries(day, pooled_day)
    for by_regime in list(journal.values()) + list(cohorts.values()):
        present.update(by_regime)
    order = regime_join.ordered_regimes(present, (current or {}).get("regime"), timeline)
    return {
        "schema": SCHEMA,
        "as_of": str(as_of or _day(today)),
        "current": current,
        "regimes": order,
        "labels": {name: regime_join.regime_label(name) for name in order},
        "segments": [{"start_date": _day(s.get("start_date")), "regime": s.get("regime")} for s in timeline],
        "windows": {"swing": "all tracker history", "daytrade": list(m5_window or [])},
        "swing": swing_out,
        "daytrade": day_out,
        "journal": journal,
        "cohorts": cohorts,
    }


# ---------------------------------------------------------------------------
# text (formatting only; the surfaces call these)
# ---------------------------------------------------------------------------


def _pct(value: Any) -> str:
    number = _float(value)
    return "?" if number is None else f"{number * 100:.0f}%"


def cell_text(cell: Mapping[str, Any] | None) -> str:
    """One regime cell: ``B · win 58% · n 42`` (graded) or ``win 55% · n 20`` (counts)."""
    if not cell or not int(cell.get("n") or 0):
        return UNTESTED
    n = int(cell.get("n") or 0)
    parts = []
    if "grade" in cell:
        parts.append(setup_grades.badge(cell.get("grade")))
    if cell.get("grade_basis") == "tape" and cell.get("tape_win_rate") is not None:
        parts.append(f"vs SPY {_pct(cell.get('tape_win_rate'))}")
    parts.append(f"win {_pct(cell.get('win_rate'))}")
    if "pnl" in cell:
        parts.append(f"P&L {float(cell['pnl']):+,.0f}")
    elif "tape_n" in cell and "grade" not in cell and cell.get("tape_win_rate") is not None:
        parts.append(f"vs SPY {_pct(cell.get('tape_win_rate'))}")
    parts.append(f"n {n}")
    return " · ".join(parts)


def this_regime_text(by_regime: Mapping[str, Any] | None, current: Mapping[str, Any] | None) -> str:
    """The current regime's cell, or `UNTESTED`; says so when no regime is typed."""
    if not current:
        return "no regime typed yet"
    return cell_text((by_regime or {}).get(str(current.get("regime") or "")))


def by_regime_text(
    by_regime: Mapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
    *,
    pooled: Mapping[str, Any] | None = None,
    include_current: bool = True,
) -> str:
    """Every regime's cell, current first, then ``all regimes`` (the pooled grade)."""
    payload = payload or {}
    by_regime = by_regime or {}
    current = str((payload.get("current") or {}).get("regime") or "")
    labels = payload.get("labels") or {}
    parts = []
    for regime in payload.get("regimes") or ():
        if regime == current and not include_current:
            continue
        cell = by_regime.get(regime)
        if regime != current and not cell:
            continue
        name = str(labels.get(regime) or regime_join.regime_label(regime))
        parts.append(f"{name}{' (now)' if regime == current else ''}: {cell_text(cell)}")
    if pooled is not None:
        if pooled:
            parts.append(f"{ALL_REGIMES}: {setup_grades.badge(pooled.get('grade'))} n {int(pooled.get('n') or 0)}")
        else:
            parts.append(f"{ALL_REGIMES}: none")
    return " · ".join(parts) or UNTESTED


def status_sentence(payload: Mapping[str, Any] | None) -> str:
    payload = payload or {}
    current = payload.get("current")
    windows = payload.get("windows") or {}
    day = list(windows.get("daytrade") or [])
    span = f"day trade over {day[0]} to {day[-1]}" if len(day) == 2 else "day trade over the desk's recent alerts"
    if not current:
        if not payload:
            return "Grades by regime: not built yet."
        return (
            "No regime typed yet - answer the Mentor's regime question. Until then every row is "
            "'regime unknown' and nothing is graded per regime."
        )
    return (
        f"Regime now: {current.get('label')} since {current.get('start_date')} (day {current.get('day_count')}). "
        f"Each grade is inside one regime, current first; '{UNTESTED}' means no rows there yet. "
        f"'{ALL_REGIMES}' is the pooled grade the badges and sorting read. Swing over all tracker history, {span}. "
        "Longs are judged on the raw result; shorts vs SPY and raw."
    )


REGIME_TABLE_COLUMNS = (
    ("kind", "Kind"),
    ("side", "Side"),
    ("setup", "Setup"),
    ("this_regime", "This regime"),
    ("other_regimes", "Other regimes"),
    ("all_regimes", "All regimes"),
)


def _rank(cell: Mapping[str, Any] | None) -> tuple:
    if not cell or not int(cell.get("n") or 0):
        return (1, setup_grades.UNGRADED_RANK + 1, 0)
    return (0, setup_grades.sort_rank(cell.get("grade")) if "grade" in cell else setup_grades.UNGRADED_RANK,
            -int(cell.get("n") or 0))


def table_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The Setup Tracker's By regime rows: current regime's best first per kind."""
    payload = payload or {}
    current = payload.get("current")
    regime = str((current or {}).get("regime") or "")
    rows: list[tuple[tuple, dict[str, Any]]] = []

    def add(kind: str, side: str, setup: str, by_regime: Mapping[str, Any], pooled: Any) -> None:
        cell = by_regime.get(regime) if regime else None
        rows.append((
            (KIND_ORDER.index(kind), *_rank(cell), setup, side),
            {
                "kind": kind,
                "side": side,
                "setup": setup,
                "this_regime": this_regime_text(by_regime, current),
                "other_regimes": by_regime_text(by_regime, payload, include_current=False) if any(
                    name != regime for name in by_regime) else "none",
                "all_regimes": (
                    f"{setup_grades.badge(pooled.get('grade'))} n {int(pooled.get('n') or 0)}" if pooled
                    else ("none" if kind in {"swing", "day trade"} else "")
                ),
            },
        ))

    for entry in (payload.get("swing") or {}).values():
        setup = f"{entry.get('family')} ({entry.get('bucket')})"
        add("swing", str(entry.get("side") or ""), setup, entry.get("by_regime") or {}, entry.get("all") or {})
    for entry in (payload.get("daytrade") or {}).values():
        add("day trade", str(entry.get("side") or ""), str(entry.get("bounce_type") or ""),
            entry.get("by_regime") or {}, entry.get("all") or {})
    for key, by_regime in (payload.get("cohorts") or {}).items():
        cohort, _, side = str(key).partition("|")
        add("cohort", side, cohort.replace("human_focus_", ""), by_regime or {}, None)
    for side, by_regime in (payload.get("journal") or {}).items():
        add("journal", str(side), "my closed trades", by_regime or {}, None)
    rows.sort(key=lambda item: item[0])
    return [row for _key, row in rows]


# ---------------------------------------------------------------------------
# Weekend Prep: what has worked / is untested in the current regime (S16 item 4)
# ---------------------------------------------------------------------------

#: "Worked" = this grade or better inside the current regime (the ladder already
#: needs n >= `setup_grades.MIN_N` for any letter grade).
WORKED_MIN_GRADE = setup_grades.B
NO_REGIME_TEXT = "Regime unknown: type the regime in the Mentor first."


def _setups(payload: Mapping[str, Any]) -> list[tuple[str, str, str, Mapping[str, Any]]]:
    out = []
    for entry in (payload.get("swing") or {}).values():
        out.append(("swing", str(entry.get("side") or ""), f"{entry.get('family')} ({entry.get('bucket')})",
                    entry.get("by_regime") or {}))
    for entry in (payload.get("daytrade") or {}).values():
        out.append(("day trade", str(entry.get("side") or ""), str(entry.get("bounce_type") or ""),
                    entry.get("by_regime") or {}))
    return out


def regime_setups(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Swing and day-trade setups that worked (B or better, n at the floor) and those
    with no rows in the current regime. Grades are this module's own (shorts vs SPY
    once 30+ have one, raw beside it). ``current`` is None when no regime is typed."""
    payload = payload or {}
    current = payload.get("current") or None
    regime = str((current or {}).get("regime") or "")
    worked: list[tuple[tuple, dict[str, Any]]] = []
    untested: list[dict[str, Any]] = []
    tested_not_worked = 0
    if regime:
        limit = setup_grades.sort_rank(WORKED_MIN_GRADE)
        for kind, side, setup, by_regime in _setups(payload):
            cell = by_regime.get(regime)
            n = int((cell or {}).get("n") or 0)
            row = {"kind": kind, "side": side, "setup": setup}
            rank = setup_grades.sort_rank((cell or {}).get("grade"))
            if not n:
                untested.append(row)
            elif n >= setup_grades.MIN_N and rank <= limit:
                worked.append(((rank, -n, kind, setup, side), {**row, "text": cell_text(cell)}))
            else:
                tested_not_worked += 1
    worked.sort(key=lambda item: item[0])
    return {
        "current": current,
        "worked": [row for _key, row in worked],
        "untested": untested,
        "tested_not_worked": tested_not_worked,
    }


def regime_setups_text(payload: Mapping[str, Any] | None) -> str:
    """The Weekend Prep card's text. Facts only; formatting only."""
    view = regime_setups(payload)
    current = view["current"]
    if not current:
        return NO_REGIME_TEXT
    lines = [
        f"Regime now: {current.get('label')} since {current.get('start_date')} (day {current.get('day_count')}).",
        f"Worked in this regime (B or better, n {setup_grades.MIN_N}+):" + ("" if view["worked"] else " none yet"),
    ]
    lines += [f"  {row['kind']} {row['side']} {row['setup']}: {row['text']}" for row in view["worked"]]
    lines.append("Untested in this regime (no rows yet):" + ("" if view["untested"] else " none"))
    lines += [f"  {row['kind']} {row['side']} {row['setup']}" for row in view["untested"]]
    if view["tested_not_worked"]:
        lines.append(f"{view['tested_not_worked']} more tested here, not B or better (Setup Tracker, By regime).")
    lines.append("Longs judged raw; shorts vs SPY once 30+ have it, raw beside it.")
    return "\n".join(lines)
