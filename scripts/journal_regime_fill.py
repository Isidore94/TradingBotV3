"""Fill the journal's market environment (`regimes`) from benchmark bars.

One row per journal date (the date `list_trades` keys regimes by: the first
10 characters of `opened_at`). Three trend labels, each `up` / `down` /
`range` / `unknown`, read by the champion Auto Market Bias formula that
`research_warehouse.market_bias_context` (auto_market_bias_multiframe_v3)
already uses, over SPY:

* `mid_term_regime`   - D1, the 20 sessions completed before that date;
* `short_term_regime` - D1, the 5 sessions completed before that date;
* `intraday_regime`   - M5, that session's bars completed by the day's FIRST
  entry (unknown when the fill is date-only or the session has no M5 bars).

QQQ and IWM D1 reads go in `notes`. Point in time: every read uses only bars
completed before the moment it describes. Missing bars give `unknown`, never a
guess. Rows are written with `source='auto'`; a row the trader wrote (or edited)
is never touched.

Nightly: the `journal_auto_tag` slot calls `fill_regimes(store, apply=True)`.
By hand (dry run by default):

    python scripts/journal_regime_fill.py [--apply] [--db PATH] [--since D] [--until D]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import market_regimes

MARKET_TZ = ZoneInfo("America/New_York")
BENCHMARKS = ("SPY", "QQQ", "IWM")
PRIMARY = "SPY"
UNKNOWN = "unknown"
MID_WINDOW = 20
SHORT_WINDOW = 5
#: SPY M5 history handed to the intraday read (M30 needs ~2 sessions of bars).
M5_LOOKBACK = timedelta(days=10)
#: Written into every auto row's notes so a reader knows which rule made it.
REGIME_RULE = "journal_regime_auto_v1"

#: Champion env_key -> trend label.
_TREND = {
    "bullish_strong": "up",
    "bullish_weak": "up",
    "bearish_strong": "down",
    "bearish_weak": "down",
    "neutral_chop": "range",
}


def trend_label(env_key: Any) -> str:
    """`up` / `down` / `range` for a champion env_key; anything else is `unknown`."""
    return _TREND.get(str(env_key or "").strip().lower(), UNKNOWN)


@dataclass(frozen=True)
class RegimeReading:
    """One journal date's auto market environment."""

    trade_date: str
    mid_term_regime: str
    short_term_regime: str
    intraday_regime: str
    notes: str

    def fields(self) -> dict[str, str]:
        return {
            "mid_term_regime": self.mid_term_regime,
            "short_term_regime": self.short_term_regime,
            "intraday_regime": self.intraday_regime,
            "notes": self.notes,
        }


def _aware(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.replace(tzinfo=MARKET_TZ)


def _is_date_only(moment: datetime) -> bool:
    local = moment.astimezone(MARKET_TZ)
    return (local.hour, local.minute, local.second, local.microsecond) == (0, 0, 0, 0)


# One definition, shared with the S17 regime table (`market_regimes`).
_session_day = market_regimes.session_day
_d1_read = market_regimes.d1_env_key
_completed_before = market_regimes.completed_before


def _intraday_read(
    entry_at: datetime | None, spy_m5: Sequence[Mapping[str, Any]], spy_d1: Sequence[Mapping[str, Any]]
) -> tuple[str, str, str]:
    """`(M5 env_key, M30 env_key, as-of text)` at the day's first entry."""
    if entry_at is None or _is_date_only(entry_at) or not spy_m5:
        return UNKNOWN, UNKNOWN, "no entry time"
    from research_warehouse import exchange_calendar as xcal

    session = xcal.session_for(entry_at)
    if session is None:
        return UNKNOWN, UNKNOWN, "not a session"
    moment = min(entry_at, session.rth_close_at)
    readings, note = market_regimes.intraday_env_keys(moment, spy_m5, spy_d1, lookback=M5_LOOKBACK)
    if note:
        return UNKNOWN, UNKNOWN, note
    stamp = moment.astimezone(MARKET_TZ).strftime("%H:%M ET")
    return readings["M5"], readings["M30"], f"first entry {stamp}"


def read_session_regime(
    trade_date: str,
    first_entry_at: datetime | None,
    *,
    d1_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    spy_m5: Sequence[Mapping[str, Any]] = (),
) -> RegimeReading:
    """The auto environment for one journal date, from bars known at the time."""
    day = date.fromisoformat(str(trade_date)[:10])
    reads: dict[str, tuple[str, str]] = {}
    for symbol in BENCHMARKS:
        completed = _completed_before(d1_by_symbol.get(symbol) or (), day)
        reads[symbol] = (_d1_read(completed, MID_WINDOW), _d1_read(completed, SHORT_WINDOW))
    spy_completed = _completed_before(d1_by_symbol.get(PRIMARY) or (), day)
    m5_key, m30_key, as_of = _intraday_read(first_entry_at, spy_m5, spy_completed)
    mid_key, short_key = reads[PRIMARY]
    others = "; ".join(
        f"{symbol} D1 {reads[symbol][0]}, 5d {reads[symbol][1]}" for symbol in BENCHMARKS if symbol != PRIMARY
    )
    notes = (
        f"auto ({REGIME_RULE}, champion Auto Market Bias; bars completed before the session, "
        f"intraday at {as_of}): SPY D1 {mid_key}, 5d {short_key}, M5 {m5_key}, M30 {m30_key}; {others}"
    )
    return RegimeReading(
        trade_date=day.isoformat(),
        mid_term_regime=trend_label(mid_key),
        short_term_regime=trend_label(short_key),
        intraday_regime=trend_label(m5_key),
        notes=notes,
    )


def journal_dates(trades: Iterable[Mapping[str, Any]]) -> dict[str, datetime | None]:
    """Each journal date with the day's first TIMED entry (None when none has a time)."""
    firsts: dict[str, datetime | None] = {}
    for trade in trades or ():
        opened = trade.get("opened_at") or trade.get("trade_date")
        key = str(opened or "").strip()[:10]
        if len(key) != 10:
            continue
        moment = _aware(trade.get("opened_at"))
        current = firsts.get(key)
        if moment is None or _is_date_only(moment):
            firsts.setdefault(key, None)
            continue
        if current is None or moment < current:
            firsts[key] = moment
    return firsts


@dataclass
class RegimeFillPlan:
    writes: list[RegimeReading]
    unchanged: int = 0
    trader_owned: int = 0
    dates: int = 0
    bars_source: str = ""


def plan_fill(
    store: Any,
    *,
    d1_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    spy_m5: Sequence[Mapping[str, Any]] = (),
    since: date | None = None,
    until: date | None = None,
) -> RegimeFillPlan:
    """What an apply would write: every journal date not owned by the trader whose auto read changed."""
    existing = {str(row.get("trade_date") or ""): row for row in store.list_regime_rows()}
    firsts = journal_dates(store.list_trades())
    plan = RegimeFillPlan(writes=[])
    for key in sorted(firsts):
        try:
            day = date.fromisoformat(key)
        except ValueError:
            continue
        if (since and day < since) or (until and day > until):
            continue
        plan.dates += 1
        row = existing.get(key)
        if row is not None and str(row.get("source") or "") != "auto":
            plan.trader_owned += 1
            continue
        reading = read_session_regime(key, firsts[key], d1_by_symbol=d1_by_symbol, spy_m5=spy_m5)
        if row is not None and all(
            str(row.get(name) or "") == value for name, value in reading.fields().items()
        ):
            plan.unchanged += 1
            continue
        plan.writes.append(reading)
    return plan


def apply_fill(store: Any, plan: RegimeFillPlan) -> dict[str, int]:
    """Write the plan through `upsert_auto_regime`. A journal write that fails raises."""
    summary = {"written": 0, "refused": 0}
    for reading in plan.writes:
        if store.upsert_auto_regime(reading.trade_date, **reading.fields()):
            summary["written"] += 1
        else:
            summary["refused"] += 1
    return summary


def load_benchmark_bars() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], str]:
    """Read-only: benchmark D1 (lake, then the durable D1 store) and SPY M5 (lake).

    Returns `(d1_by_symbol, spy_m5, source text)`. Never writes; an unreadable
    source contributes nothing. The read itself is `market_regimes.load_bars`.
    """
    d1_by_symbol, m5_by_symbol, source = market_regimes.load_bars(BENCHMARKS, (PRIMARY,))
    return d1_by_symbol, m5_by_symbol[PRIMARY], source


def fill_regimes(
    store: Any,
    *,
    apply: bool = False,
    since: date | None = None,
    until: date | None = None,
    loader: Callable[[], tuple[dict, list, str]] | None = None,
) -> dict[str, Any]:
    """Plan (and with `apply`, write) the auto regimes. Returns a summary dict."""
    d1_by_symbol, spy_m5, source = (loader or load_benchmark_bars)()
    if not any(d1_by_symbol.get(symbol) for symbol in BENCHMARKS):
        return {"status": "no_bars", "dates": 0, "written": 0, "planned": 0, "bars_source": source}
    plan = plan_fill(store, d1_by_symbol=d1_by_symbol, spy_m5=spy_m5, since=since, until=until)
    plan.bars_source = source
    applied = apply_fill(store, plan) if apply else {"written": 0, "refused": 0}
    unknown = sum(
        1
        for reading in plan.writes
        if UNKNOWN in (reading.mid_term_regime, reading.short_term_regime, reading.intraday_regime)
    )
    return {
        "status": "ok",
        "dates": plan.dates,
        "planned": len(plan.writes),
        "unchanged": plan.unchanged,
        "trader_owned": plan.trader_owned,
        "with_unknown": unknown,
        "bars_source": source,
        "readings": plan.writes,
        **applied,
    }


def format_summary(summary: Mapping[str, Any], *, applied: bool) -> str:
    lines = [
        "Journal market environment fill",
        f"bars: {summary.get('bars_source')}",
        f"journal dates considered: {summary.get('dates', 0)}",
        f"trader-owned rows left alone: {summary.get('trader_owned', 0)}",
        f"auto rows unchanged: {summary.get('unchanged', 0)}",
        f"rows to write: {summary.get('planned', 0)} ({summary.get('with_unknown', 0)} with an unknown field)",
    ]
    for reading in list(summary.get("readings") or ())[:400]:
        lines.append(
            f"  {reading.trade_date}  mid {reading.mid_term_regime:<7} short {reading.short_term_regime:<7} "
            f"intraday {reading.intraday_regime}"
        )
    if applied:
        lines.append(f"WRITTEN: {summary.get('written', 0)}  REFUSED (trader row): {summary.get('refused', 0)}")
    else:
        lines.append("Dry run: nothing written. Re-run with --apply to write.")
    if summary.get("status") == "no_bars":
        lines.append("No benchmark bars were readable; nothing planned.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    parser.add_argument("--db", default="", help="journal database (default: the live one)")
    parser.add_argument("--since", default="", help="first journal date, YYYY-MM-DD")
    parser.add_argument("--until", default="", help="last journal date, YYYY-MM-DD")
    args = parser.parse_args(argv)

    from journal_store import JournalStore

    store = JournalStore(Path(args.db)) if args.db else JournalStore()
    store.initialize_schema()
    summary = fill_regimes(
        store,
        apply=args.apply,
        since=date.fromisoformat(args.since) if args.since else None,
        until=date.fromisoformat(args.until) if args.until else None,
    )
    print(format_summary(summary, applied=args.apply))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "REGIME_RULE",
    "RegimeReading",
    "apply_fill",
    "fill_regimes",
    "journal_dates",
    "load_benchmark_bars",
    "plan_fill",
    "read_session_regime",
    "trend_label",
]
