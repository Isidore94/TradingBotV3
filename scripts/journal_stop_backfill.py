"""Backfill an empty journal plan (stop, entry, risk) from what the desk logged before the entry.

For each trade whose `trade_annotations` plan is empty, the first source that
gives one unambiguous stop on the right side of the fill wins:

* `backfill_m5_alert` - the M5 alert the desk registered for that underlying and
  side on the entry's own session, closest before the first fill
  (`intraday_bounce_outcomes.csv`, `registered` rows, stamped by `logged_at`;
  joined with `journal_setup_evidence.same_session_before_entry`);
* `backfill_d1_plan` - the D1 detail plan (`entry_plan.plan_for_row`) built from
  the latest `master_avwap_setup_daily.csv` scan on or before the entry date,
  same symbol and side, read on its last session completed BEFORE the entry
  date (the entry day's own row carries that day's later close) and no more
  than `journal_setup_evidence.D1_LOOKBACK` old.

Written through `JournalStore.backfill_risk_fields`: `planned_stop` and
`planned_entry` are the source's own; `planned_risk` = |average entry - stop| x
quantity_opened in the trade's currency (the same convention as the Trades
tab's alert prefill); `risk_source` names the source. A trade with any plan
field or risk_source already set is never touched. Options are skipped: the
source's stop is on the underlying, the fill is a premium.

Dry run by default:

    python scripts/journal_stop_backfill.py [--apply] [--since YYYY-MM-DD] [--trade-id ID ...] [--db PATH]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SOURCE_M5 = "backfill_m5_alert"
SOURCE_D1 = "backfill_d1_plan"
SOURCES = (SOURCE_M5, SOURCE_D1)

#: `setup_daily` band columns -> the band labels `setup_docs.build_trade_plan` reads.
_BAND_COLUMNS = {
    "current_upper_1": "UPPER_1",
    "current_lower_1": "LOWER_1",
    "current_upper_2": "UPPER_2",
    "current_lower_2": "LOWER_2",
    "current_upper_3": "UPPER_3",
    "current_lower_3": "LOWER_3",
}


#: Marks a stamp that carried no zone of its own.
_NAIVE = timezone(timedelta(0), "naive")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number != 0 else None


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


@dataclass(frozen=True)
class Proposal:
    """One stop found for one trade."""

    source: str
    stop: float
    entry: float
    ref: str


@dataclass
class TradeResult:
    trade_id: str
    line: str
    outcome: str  # filled:<source> | skipped | no_source
    proposal: Proposal | None = None
    risk: float | None = None
    written: bool = False


@dataclass
class Summary:
    results: list[TradeResult] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        counts = {source: 0 for source in SOURCES}
        counts.update({"skipped": 0, "no_source": 0})
        for result in self.results:
            key = result.outcome.split(":", 1)[1] if result.outcome.startswith("filled:") else result.outcome
            counts[key] = counts.get(key, 0) + 1
        return counts


# ---------------------------------------------------------------- source A
def load_m5_alerts(path: Path, symbols: set[str]) -> tuple[list[Any], dict[str, Mapping[str, Any]]] | None:
    """(evidence rows, event_id -> CSV row) for `registered` alerts on `symbols`; None when unreadable."""
    from journal_setup_evidence import EvidenceRow, aware_moment

    rows: list[Any] = []
    by_ref: dict[str, Mapping[str, Any]] = {}
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("event_type") or "").strip().lower() != "registered":
                    continue
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol not in symbols:
                    continue
                side = str(row.get("direction") or "").strip().upper()
                # A naive stamp cannot be placed against the fill: aware only.
                at = aware_moment(row.get("logged_at"), naive_tz=_NAIVE)
                ref = str(row.get("event_id") or "").strip()
                if side not in {"LONG", "SHORT"} or at is None or at.tzinfo is _NAIVE or not ref:
                    continue
                rows.append(EvidenceRow(symbol=symbol, side=side, at=at, family="alert_fired",
                                        kind="m5_alert", horizon="m5", ref=ref))
                by_ref[ref] = row
    except (OSError, csv.Error, UnicodeDecodeError):
        logging.warning("M5 alert outcomes unreadable: %s", path, exc_info=True)
        return None
    return rows, by_ref


def m5_proposal(trade: Mapping[str, Any], rows: Iterable[Any], by_ref: Mapping[str, Mapping[str, Any]]) -> Proposal | None:
    """The closest same-session alert before the fill that carries an entry and a stop."""
    from journal_setup_evidence import same_session_before_entry

    for row in same_session_before_entry(trade, rows):
        csv_row = by_ref.get(row.ref) or {}
        stop, entry = _number(csv_row.get("stop_price")), _number(csv_row.get("entry_price"))
        if stop is not None and entry is not None:
            return Proposal(SOURCE_M5, stop, entry, row.ref)
    return None


# ---------------------------------------------------------------- source B
def load_setup_daily(path: Path, symbols: set[str]) -> dict[tuple[str, str], list[Mapping[str, Any]]] | None:
    """(symbol, side) -> setup_daily rows for `symbols`; None when unreadable."""
    found: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol not in symbols:
                    continue
                side = str(row.get("side") or "").strip().upper()
                found.setdefault((symbol, side), []).append(row)
    except (OSError, csv.Error, UnicodeDecodeError):
        logging.warning("Setup daily file unreadable: %s", path, exc_info=True)
        return None
    return found


def _row_date(row: Mapping[str, Any], key: str) -> date | None:
    try:
        return date.fromisoformat(str(row.get(key) or "")[:10])
    except ValueError:
        return None


def d1_proposal(trade: Mapping[str, Any], daily: Mapping[tuple[str, str], list[Mapping[str, Any]]]) -> Proposal | None:
    """The D1 plan stop of the latest scan on or before the entry date; None when absent or ambiguous.

    A `setup_daily` row is one scanned setup (`scan_date`) read on one session
    (`trade_date`). Only sessions completed before the entry date count, within
    `D1_LOOKBACK`; the latest scan with such a row wins, read on its latest one.
    """
    from entry_plan import plan_for_row
    from journal_setup_evidence import D1_LOOKBACK, MARKET_TZ, aware_moment, underlying_view

    entry_at = aware_moment(trade.get("opened_at"))
    symbol, side = underlying_view(trade)
    if entry_at is None or not symbol or not side:
        return None
    entry_day = entry_at.astimezone(MARKET_TZ).date()
    earliest = entry_day - timedelta(days=D1_LOOKBACK.days)
    dated = [
        (scan, session, row)
        for row in daily.get((symbol, side), ())
        if (scan := _row_date(row, "scan_date")) is not None and scan <= entry_day
        and (session := _row_date(row, "trade_date")) is not None and earliest <= session < entry_day
    ]
    if not dated:
        return None
    latest_scan = max(scan for scan, _session, _row in dated)
    latest_session = max(session for scan, session, _row in dated if scan == latest_scan)
    plans: dict[float, Proposal] = {}
    for scan, session, row in dated:
        if scan != latest_scan or session != latest_session:
            continue
        bands = {label: value for column, label in _BAND_COLUMNS.items()
                 if (value := _number(row.get(column))) is not None}
        family = str(row.get("setup_id") or "").rsplit(":", 1)[-1]
        vwap = _number(row.get("current_avwape"))
        plan = plan_for_row(
            symbol=symbol, side=side, setup_family=family, last_close=row.get("close"),
            levels_by_symbol={symbol: {"bands": bands, "vwap": vwap, "atr20": _number(row.get("atr20")),
                                       "last_close": _number(row.get("close"))}},
        )
        if not plan or plan.get("stop") is None or plan.get("entry") is None:
            continue
        stop = float(plan["stop"])
        ref = f"{row.get('setup_id') or ''} @ {session.isoformat()}"
        plans.setdefault(round(stop, 6), Proposal(SOURCE_D1, stop, float(plan["entry"]), ref))
    return next(iter(plans.values())) if len(plans) == 1 else None


# ---------------------------------------------------------------- the pass
def _stop_on_right_side(side: str, entry: float, stop: float) -> bool:
    return stop < entry if side == "LONG" else stop > entry


def _is_option(trade: Mapping[str, Any]) -> bool:
    from journal_setup_evidence import option_underlying

    if str(trade.get("security_type") or "").strip().upper() in {"OPT", "OPTION", "FOP"}:
        return True
    return bool(option_underlying(trade.get("symbol"))[1])


def plan_trade(trade: Mapping[str, Any], alerts, daily) -> TradeResult:
    trade_id = str(trade.get("trade_id") or "")
    label = f"{trade_id} {str(trade.get('opened_at') or '')[:10]} {trade.get('symbol')} {trade.get('direction')}"
    if not all(_blank(trade.get(key)) for key in ("planned_stop", "planned_entry", "planned_risk", "risk_source")):
        return TradeResult(trade_id, f"{label}: skipped (plan already set, source '{trade.get('risk_source') or ''}')", "skipped")
    if _is_option(trade):
        return TradeResult(trade_id, f"{label}: skipped (option: a stop on the underlying is not a premium stop)", "skipped")
    security = str(trade.get("security_type") or "").strip().upper()
    if security not in {"STK", "ETF", "UNKNOWN", ""}:
        return TradeResult(trade_id, f"{label}: skipped (security type {security})", "skipped")
    side = str(trade.get("direction") or "").strip().upper()
    actual = _number(trade.get("average_entry_price"))
    if actual is None or side not in {"LONG", "SHORT"}:
        return TradeResult(trade_id, f"{label}: skipped (no entry price or side)", "skipped")
    rejected: list[str] = []
    for proposal in (
        m5_proposal(trade, *alerts) if alerts is not None else None,
        d1_proposal(trade, daily) if daily is not None else None,
    ):
        if proposal is None:
            continue
        if not (_stop_on_right_side(side, actual, proposal.stop)
                and _stop_on_right_side(side, proposal.entry, proposal.stop)):
            rejected.append(f"{proposal.source} stop {proposal.stop:g} on the wrong side")
            continue
        quantity = _number(trade.get("quantity_opened"))
        risk = abs(actual - proposal.stop) * abs(quantity) if quantity is not None else None
        currency = str(trade.get("currency") or "").strip()
        line = (f"{label}: {proposal.source} stop {proposal.stop:g} entry {proposal.entry:g} "
                f"risk {'unknown' if risk is None else f'{risk:.2f} {currency}'.strip()} ({proposal.ref})")
        return TradeResult(trade_id, line, f"filled:{proposal.source}", proposal, risk)
    reason = "; ".join(rejected) if rejected else "no alert or D1 plan before the entry"
    return TradeResult(trade_id, f"{label}: no source ({reason})", "no_source")


def run_backfill(
    store,
    *,
    apply: bool = False,
    since: date | None = None,
    trade_ids: Sequence[str] = (),
    outcomes_path: Path | None = None,
    setup_daily_path: Path | None = None,
) -> Summary:
    """Plan (and with `apply`, write) every selected trade. A journal write that fails raises."""
    from journal_setup_evidence import underlying_view

    if outcomes_path is None or setup_daily_path is None:
        from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE, MASTER_AVWAP_SETUP_DAILY_FILE

        outcomes_path = outcomes_path or Path(INTRADAY_BOUNCE_OUTCOMES_FILE)
        setup_daily_path = setup_daily_path or Path(MASTER_AVWAP_SETUP_DAILY_FILE)
    wanted = {str(value) for value in trade_ids if str(value).strip()}
    trades = [
        trade for trade in store.list_trades(date_from=since.isoformat() if since else None)
        if not wanted or str(trade.get("trade_id") or "") in wanted
    ]
    summary = Summary()
    symbols = {underlying_view(trade)[0] for trade in trades} - {""}
    alerts = load_m5_alerts(Path(outcomes_path), symbols) if symbols else ([], {})
    daily = load_setup_daily(Path(setup_daily_path), symbols) if symbols else {}
    if alerts is None:
        summary.unreadable.append(str(outcomes_path))
    if daily is None:
        summary.unreadable.append(str(setup_daily_path))
    for trade in sorted(trades, key=lambda row: str(row.get("opened_at") or "")):
        result = plan_trade(trade, alerts, daily)
        if apply and result.proposal is not None:
            result.written = store.backfill_risk_fields(
                result.trade_id,
                planned_entry=result.proposal.entry,
                planned_stop=result.proposal.stop,
                planned_risk=result.risk,
                risk_source=result.proposal.source,
            )
            if not result.written:
                result.outcome = "skipped"
                result.line += " -> not written (a plan appeared meanwhile)"
        summary.results.append(result)
    return summary


def format_summary(summary: Summary, *, applied: bool) -> str:
    lines = [result.line for result in summary.results]
    counts = summary.counts()
    verb = "Filled" if applied else "Would fill"
    lines.append(
        f"{verb}: {counts[SOURCE_M5]} by {SOURCE_M5}, {counts[SOURCE_D1]} by {SOURCE_D1}; "
        f"skipped {counts['skipped']}; no source found {counts['no_source']} "
        f"(of {len(summary.results)} trades)."
    )
    for path in summary.unreadable:
        lines.append(f"Unreadable source (its trades are unknown, not 'no source'): {path}")
    if not applied:
        lines.append("Dry run: nothing written. Add --apply to write.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    parser.add_argument("--db", default="", help="journal database (default: the live one)")
    parser.add_argument("--since", default="", help="first trade date, YYYY-MM-DD")
    parser.add_argument("--trade-id", action="append", default=[], help="only this trade (repeatable)")
    parser.add_argument("--outcomes", default="", help="intraday_bounce_outcomes.csv (default: the live one)")
    parser.add_argument("--setup-daily", default="", help="master_avwap_setup_daily.csv (default: the live one)")
    args = parser.parse_args(argv)

    from journal_store import JournalStore

    store = JournalStore(Path(args.db)) if args.db else JournalStore()
    summary = run_backfill(
        store,
        apply=args.apply,
        since=date.fromisoformat(args.since) if args.since else None,
        trade_ids=args.trade_id,
        outcomes_path=Path(args.outcomes) if args.outcomes else None,
        setup_daily_path=Path(args.setup_daily) if args.setup_daily else None,
    )
    print(format_summary(summary, applied=args.apply))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["SOURCES", "SOURCE_D1", "SOURCE_M5", "d1_proposal", "format_summary", "load_m5_alerts",
           "load_setup_daily", "m5_proposal", "plan_trade", "run_backfill"]
