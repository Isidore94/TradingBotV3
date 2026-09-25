"""P8-P5: the journal's plain truths, in plain words.

Stocks vs options, longs vs shorts, confirmed setups, the bot's grade of each
traded setup as of the entry, and how exits went against MFE/MAE. Every line
counts CLOSED trades whose entry is real (`counts_in_pnl`) and sums the one P&L
column the caller's surface already uses. Missing is unknown, never zero.

Pure except the two loaders (`grade_reader`, `measure_exits`), which read files
and run on workers only. Display only: nothing here feeds a detector, a score,
an alert, Focus, the queue or `review_policy.json`.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import journal_analytics as ja

STOCK, OPTION = "STK", "OPT"
LONG, SHORT = "LONG", "SHORT"
NO_GRADE = "no grade"
#: The week card's floors (`week_coach.MIN_N` / `THIN_MIN_N`): under 5 is too
#: few to tell, 5-9 is thin, 10+ is plain.
MIN_N = 10
THIN_MIN_N = 5
TOO_FEW = "too few to tell"
#: Grades at or below this are "D-or-worse" (D is the lowest on the ladder).
WORST_GRADES = frozenset({"D"})


# ---------------------------------------------------------------------------
# per-trade fields
# ---------------------------------------------------------------------------
def trade_instrument(row: Mapping[str, Any]) -> str:
    return ja.trade_instrument(dict(row))


def trade_direction(row: Mapping[str, Any]) -> str:
    return ja.trade_direction(dict(row))


def _setup_tag(row: Mapping[str, Any]) -> str:
    """The first confirmed setup tag that is a setup (no rejection, no link)."""
    for tag in ja._confirmed_setup_tags(dict(row)):
        if ja.is_rejection_tag(tag) or ja.is_link_candidate(tag):
            continue
        if tag.split("|", 1)[0].strip():
            return tag
    return ""


def confirmed_family(row: Mapping[str, Any]) -> str:
    """The family of the trader's confirmed setup tag (`family | bucket | zone`), or ""."""
    return _setup_tag(row).split("|", 1)[0].strip().lower()


def confirmed_bucket(row: Mapping[str, Any]) -> str:
    """The bucket after the family in the stored `family | bucket | zone` text, or ""."""
    family = confirmed_family(row)
    if not family:
        return ""
    # `split_tags` splits a pipe-only string into separate tags, so read the raw text.
    for chunk in re.split(r"[;,]", str(row.get("setup_tags") or "")):
        parts = [part.strip().lower() for part in chunk.split("|")]
        if parts[0] == family and len(parts) > 1:
            return parts[1]
    return ""


def counted(trades: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The closed trades with a real entry: the only ones money lines add up."""
    return [dict(row) for row in trades if ja.counts_in_pnl(dict(row))]


def close_day(row: Mapping[str, Any]) -> date | None:
    return ja._parse_date(row.get("closed_at") or row.get("trade_date") or row.get("opened_at"))


def in_window(trades: Iterable[Mapping[str, Any]], first: date, last: date) -> list[dict[str, Any]]:
    """Trades whose close date is in `[first, last]`."""
    out = []
    for row in trades:
        day = close_day(row)
        if day is not None and first <= day <= last:
            out.append(dict(row))
    return out


# ---------------------------------------------------------------------------
# money lines
# ---------------------------------------------------------------------------
def money_text(value: float | None, label: str = "") -> str:
    if value is None:
        return "money unknown"
    sign = "-" if value < 0 else "+"
    return f"{sign}${abs(value):,.0f}" + (f" {label}" if label else "")


def _sum(rows: Sequence[Mapping[str, Any]], pnl_key: str) -> float | None:
    if not pnl_key:
        return None
    values = [ja._coerce_float(row.get(pnl_key)) for row in rows]
    if any(value is None for value in values):
        return None
    return float(sum(values))  # type: ignore[arg-type]


def _noun(n: int) -> str:
    return "trade" if n == 1 else "trades"


def split_lines(trades: Iterable[Mapping[str, Any]], pnl_key: str, label: str = "") -> list[str]:
    """`Stocks: N trades, $X. Options: N trades, $Y.` and `Longs: N, $X. Shorts: N, $Y.`"""
    rows = counted(trades)
    stocks = [row for row in rows if trade_instrument(row) == STOCK]
    options = [row for row in rows if trade_instrument(row) == OPTION]
    longs = [row for row in rows if trade_direction(row) == LONG]
    shorts = [row for row in rows if trade_direction(row) == SHORT]
    def part(name: str, group: list[dict[str, Any]], noun: bool = True) -> str:
        count = f"{len(group)} {_noun(len(group))}" if noun else str(len(group))
        if not group:
            return f"{name}: {count}."
        return f"{name}: {count}, {money_text(_sum(group, pnl_key), label)}."

    return [
        f"{part('Stocks', stocks)} {part('Options', options)}",
        f"{part('Longs', longs, False)} {part('Shorts', shorts, False)}",
    ]


def setup_lines(trades: Iterable[Mapping[str, Any]], pnl_key: str, label: str = "") -> list[str]:
    """`By setup (confirmed only): family, n, win %, expectancy`, one line per family."""
    rows = counted(trades)
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = confirmed_family(row)
        if family:
            families.setdefault(family, []).append(row)
    tagged = sum(len(group) for group in families.values())
    coverage = f"{tagged} of {len(rows)} closed {_noun(len(rows))} have a confirmed setup."
    if not families:
        return [f"By setup (confirmed only): none yet. {coverage}"]
    lines = []
    for family, group in sorted(families.items(), key=lambda item: (-len(item[1]), item[0])):
        n = len(group)
        head = f"By setup (confirmed only): {family.replace('_', ' ')}, n {n}"
        if n < THIN_MIN_N:
            lines.append(f"{head}, {TOO_FEW}.")
            continue
        wins = sum(1 for row in group if (ja._coerce_float(row.get("net_pnl")) or 0.0) > 0)
        total = _sum(group, pnl_key)
        expectancy = money_text(total / n if total is not None else None, label)
        thin = f", thin ({n})" if n < MIN_N else ""
        lines.append(f"{head}, win {wins / n:.0%}, expectancy {expectancy}{thin}.")
    lines.append(coverage)
    return lines


def truth_lines(trades: Iterable[Mapping[str, Any]], pnl_key: str, label: str = "") -> list[str]:
    rows = list(trades)
    return split_lines(rows, pnl_key, label) + setup_lines(rows, pnl_key, label)


def cad_truth_lines(trades: Iterable[Mapping[str, Any]]) -> list[str]:
    """`truth_lines` in CAD, or counts with a stated reason when CAD is refused."""
    rows = counted(trades)
    pnl_key, note = ja.resolve_pnl_key(rows, "CAD")
    lines = truth_lines(rows, pnl_key, "CAD")
    if not pnl_key and note:
        lines.append(f"Money not shown: {note}.")
    return lines


# ---------------------------------------------------------------------------
# trader vs bot: the bot's grade of the traded setup, as of the entry
# ---------------------------------------------------------------------------
def entry_moment(row: Mapping[str, Any]) -> datetime | None:
    from journal_setup_evidence import MARKET_TZ, aware_moment

    return aware_moment(row.get("opened_at"), MARKET_TZ)


def grade_reader(history_dir: Path | str | None = None) -> Callable[[datetime], Mapping[str, Any] | None]:
    """`setup_grades_history.grades_as_of`, remembered per moment. Worker only."""
    import setup_grades_history

    memo: dict[datetime, Mapping[str, Any] | None] = {}

    def read(when: datetime) -> Mapping[str, Any] | None:
        if when not in memo:
            memo[when] = setup_grades_history.grades_as_of(when, history_dir=history_dir)
        return memo[when]

    return read


def _no_grade(why: str, family: str = "") -> dict[str, Any]:
    return {"grade": NO_GRADE, "why": why, "family": family, "written_at": ""}


def bot_grade(
    trade: Mapping[str, Any],
    grades_at: Callable[[datetime], Mapping[str, Any] | None],
) -> dict[str, Any]:
    """The bot's grade of this trade's confirmed setup, never one written after the entry.

    A swing reads the swing population's (side, bucket, family) cell, or the
    family's largest cell on that side when the tag names no bucket. A day
    trade reads the bounce-type cell its setup names; otherwise "no grade".
    """
    import setup_grades
    from journal_setup_evidence import underlying_view

    family = confirmed_family(trade)
    if not family:
        return _no_grade("no confirmed setup")
    moment = entry_moment(trade)
    if moment is None:
        return _no_grade("no entry time", family)
    payload = grades_at(moment)
    if not payload:
        return _no_grade("no grade written before the entry", family)
    try:
        written = datetime.fromisoformat(str(payload.get("written_at") or ""))
    except ValueError:
        return _no_grade("the grade has no write time", family)
    if written.tzinfo is None or written > moment:
        return _no_grade("no grade written before the entry", family)
    _underlying, side = underlying_view(trade)
    if side not in (LONG, SHORT):
        return _no_grade("no side", family)
    cell: Mapping[str, Any] | None = None
    if ja.trade_horizon(dict(trade)) == "day":
        from held_run_score import bounce_components

        lookup = setup_grades.daytrade_lookup(payload)
        found = [lookup[key] for key in (
            setup_grades.daytrade_key(part, side) for part in bounce_components(family)
        ) if key in lookup]
        if found:
            cell = min(found, key=lambda item: setup_grades.sort_rank(item.get("grade")))
    else:
        lookup = setup_grades.swing_lookup(payload)
        bucket = confirmed_bucket(trade)
        if bucket:
            # A named bucket reads only its own cell, never another bucket's grade.
            cell = lookup.get(setup_grades.swing_key(side, bucket, family))
            if cell is None:
                return _no_grade("bucket not graded", family)
        else:
            same = [
                item for item in lookup.values()
                if str(item.get("family") or "").lower() == family and str(item.get("side") or "").upper() == side
            ]
            if same:
                cell = max(same, key=lambda item: (int(item.get("n") or 0), str(item.get("key") or "")))
    if cell is None:
        return _no_grade("the bot has no grade for this setup", family)
    return {
        "grade": str(cell.get("grade") or setup_grades.NEW),
        "why": "",
        "family": family,
        "written_at": str(payload.get("written_at") or ""),
    }


def grade_text(result: Mapping[str, Any] | None) -> str:
    grade = str((result or {}).get("grade") or NO_GRADE)
    return f"bot grade: {grade}"


def bot_grades(
    trades: Iterable[Mapping[str, Any]],
    grades_at: Callable[[datetime], Mapping[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    """`trade_id -> bot_grade` for each closed trade. Worker only."""
    out = {}
    for row in trades:
        if str(row.get("status") or "").upper() != "CLOSED":
            continue
        out[str(row.get("trade_id") or "")] = bot_grade(row, grades_at)
    return out


def worst_setups_line(
    trades: Iterable[Mapping[str, Any]], grades: Mapping[str, Mapping[str, Any]], *, span: str = "this week"
) -> str:
    """`You traded N D-or-worse setups this week (names).`"""
    rows = [row for row in trades if str(row.get("status") or "").upper() == "CLOSED"]
    worst = [
        row for row in rows
        if str((grades.get(str(row.get("trade_id") or "")) or {}).get("grade") or "") in WORST_GRADES
    ]
    ungraded = sum(
        1 for row in rows
        if str((grades.get(str(row.get("trade_id") or "")) or {}).get("grade") or NO_GRADE) == NO_GRADE
    )
    names = sorted({str(row.get("symbol") or "").upper() for row in worst if row.get("symbol")})
    line = f"You traded {len(worst)} D-or-worse {'setup' if len(worst) == 1 else 'setups'} {span}"
    line += f" ({', '.join(names)})." if names else "."
    if ungraded:
        line += f" {ungraded} of {len(rows)} closed {_noun(len(rows))} have no bot grade."
    return line


# ---------------------------------------------------------------------------
# exit scoreboard (needs stops for the loser half)
# ---------------------------------------------------------------------------
def measure_exits(
    trades: Iterable[Mapping[str, Any]], measure: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
) -> dict[str, Mapping[str, Any]]:
    """`trade_id -> journal_excursion` result for each counted trade. Worker only."""
    if measure is None:
        import journal_excursion

        measure = journal_excursion.measure_trade
    out = {}
    for row in counted(trades):
        try:
            out[str(row.get("trade_id") or "")] = dict(measure(row))
        except Exception:  # noqa: BLE001 - an unreadable trade is unknown
            out[str(row.get("trade_id") or "")] = {"state": "unknown"}
    return out


def _captured_share(row: Mapping[str, Any], result: Mapping[str, Any]) -> float | None:
    mfe = ja._coerce_float(result.get("mfe"))
    entry = ja._coerce_float(row.get("average_entry_price"))
    exit_ = ja._coerce_float(row.get("average_exit_price"))
    side = trade_direction(row)
    if result.get("state") != "measured" or not mfe or entry is None or exit_ is None or side not in (LONG, SHORT):
        return None
    move = (exit_ - entry) if side == LONG else (entry - exit_)
    return move / mfe


def exit_scoreboard(
    trades: Iterable[Mapping[str, Any]], excursions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Winners' captured share of MFE and losers held past the stop, with the lines."""
    rows = counted(trades)
    winners = [row for row in rows if (ja._coerce_float(row.get("net_pnl")) or 0.0) > 0]
    losers = [row for row in rows if (ja._coerce_float(row.get("net_pnl")) or 0.0) < 0]
    shares = []
    for row in winners:
        share = _captured_share(row, excursions.get(str(row.get("trade_id") or "")) or {})
        if share is not None:
            shares.append(share)
    no_stop = sum(1 for row in rows if ja._coerce_float(row.get("planned_stop")) is None)
    past, judged = 0, 0
    for row in losers:
        mae_r = ja._coerce_float((excursions.get(str(row.get("trade_id") or "")) or {}).get("mae_r"))
        if mae_r is None:
            continue
        judged += 1
        past += 1 if mae_r > 1.0 else 0
    avg_share = sum(shares) / len(shares) if shares else None
    if not winners:
        kept = "Winners kept: unknown (no winners)."
    elif avg_share is None:
        kept = f"Winners kept: unknown (no bars for {len(winners)} of {len(winners)} winners)."
    else:
        kept = f"Winners kept {avg_share:.0%} of their best move (n {len(shares)})."
    if not judged:
        held = f"Losers held past the stop: unknown (no stop on {no_stop} of {len(rows)} trades)."
    else:
        held = f"Losers held past the stop: {past} of {judged}."
        if no_stop:
            held += f" No stop on {no_stop} of {len(rows)} trades."
    return {
        "captured_avg": avg_share, "captured_n": len(shares),
        "past_stop": past, "past_stop_n": judged,
        "no_stop": no_stop, "trades": len(rows),
        "lines": [kept, held],
    }
