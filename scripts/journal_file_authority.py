"""When a broker file disagrees with the live sync, the file wins — on money.

Trader decision, 2026-08-28: *"these should be sources of truth moreso than the
auto input IMO"*, resolved to **money only** once the cost of the blunt version
was measured. Neither broker's downloadable file carries a time of day, so
letting a file take over every day it covers would discard the only intraday
timestamps the journal has — every session bucket, every "what time do I trade
best" question, and the shape tags built on them.

So the rule is split by what each source is actually good for:

* the **live sync** (Questrade API, IBKR Flex/socket) keeps the day when the two
  agree, because it alone knows *when* each fill happened and how it was split;
* the **file** takes the day when they disagree, because it is the broker's own
  statement of what the money was, and that is the number a tax return uses.

**Agreement is measured in cash, per (account, day), not per trade.** A trade
can span days, so a day's P&L is not even defined; what a day does have is a
cash impact, and both sources must agree on it:

    signed cash = (+1 sell / -1 buy) x quantity x price x multiplier
                  - commission - fees

Comparing computed cash rather than the file's own Gross/Net column is
deliberate — Questrade reports in the trade's currency and IBKR in the account's
base currency, so their columns are not comparable to each other, while this
formula is.

**Taking a day over is append-only.** Invariant I3 forbids deleting or editing a
broker row, so the sync's executions are retired with ``VOID_EXECUTION``
adjustments carrying a stated reason. They stay in ``raw_executions`` and in the
audit list, stop applying at the next rebuild, and a superseding record undoes
the whole thing. Nothing is destroyed, which matters because the trader can
change their mind about a day.

**The tolerance is per fill, not flat.** Questrade books each row's Gross Amount
rounded to the cent while the journal recomputes price x quantity, so a busy day
accumulates fractions of a cent per fill and a flat threshold would either fire
on rounding or miss a real difference on a quiet day.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from journal_identity import normalize_security_type

#: A day's cash may differ by this much before the file is treated as
#: disagreeing, plus :data:`TOLERANCE_PER_FILL` for each fill the file lists.
TOLERANCE_BASE = 0.02

#: Per-fill allowance. Measured on the trader's own Questrade export: the worst
#: single execution differed by half a cent, so a cent per fill is a bound
#: rather than a guess.
TOLERANCE_PER_FILL = 0.01

#: Sides that take cash out of the account.
_BUY_SIDES = frozenset({"BUY", "BOT", "BTO", "BTC", "COVER"})


@dataclass
class DayCash:
    """One (account, day) as both sources see it."""

    account: str
    day: date
    file_cash: float
    journal_cash: float
    file_fills: int
    journal_fills: int
    journal_uids: list[str] = field(default_factory=list)

    @property
    def difference(self) -> float:
        return self.file_cash - self.journal_cash

    @property
    def tolerance(self) -> float:
        return TOLERANCE_BASE + TOLERANCE_PER_FILL * max(self.file_fills, self.journal_fills)

    @property
    def disagrees(self) -> bool:
        return abs(self.difference) > self.tolerance


def _multiplier_for(row: Mapping[str, Any]) -> float:
    for key in ("multiplier",):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    raw = row.get("raw_json")
    if isinstance(raw, str) and raw:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        try:
            value = float(payload.get("multiplier"))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return 100.0 if normalize_security_type(row.get("security_type")) in {"OPT", "FOP"} else 1.0


def signed_cash(row: Mapping[str, Any]) -> float:
    """What this fill did to the account's cash, in the fill's own currency.

    Computed rather than read off a Gross/Net column, because the two brokers do
    not report those in the same currency and the whole point here is to compare
    them to each other.
    """
    try:
        quantity = abs(float(row.get("quantity") or 0.0))
        price = abs(float(row.get("price") or 0.0))
    except (TypeError, ValueError):
        return 0.0
    side = str(row.get("side") or "").strip().upper()
    sign = -1.0 if side in _BUY_SIDES else 1.0
    try:
        commission = float(row.get("commission") or 0.0)
    except (TypeError, ValueError):
        commission = 0.0
    try:
        fees = float(row.get("fees") or 0.0)
    except (TypeError, ValueError):
        fees = 0.0
    return sign * quantity * price * _multiplier_for(row) - commission - fees


def _as_row(execution: Any) -> Mapping[str, Any]:
    if isinstance(execution, Mapping):
        return execution
    as_row = getattr(execution, "as_row", None)
    return as_row() if callable(as_row) else vars(execution)


def cash_by_day(executions: Iterable[Any]) -> dict[tuple[str, date], tuple[float, int]]:
    """``(account, day) -> (cash, fill count)`` for a set of executions."""
    totals: dict[tuple[str, date], list[float]] = {}
    for execution in executions:
        row = _as_row(execution)
        account = str(row.get("account_number") or "")
        day = row.get("trade_date")
        if isinstance(day, str):
            try:
                day = date.fromisoformat(day[:10])
            except ValueError:
                continue
        if not isinstance(day, date):
            continue
        entry = totals.setdefault((account, day), [0.0, 0.0])
        entry[0] += signed_cash(row)
        entry[1] += 1
    return {key: (value[0], int(value[1])) for key, value in totals.items()}


def compare_days(
    store: Any,
    *,
    broker: str,
    file_executions: Sequence[Any],
    sources: Iterable[str],
) -> list[DayCash]:
    """Every (account, day) both the file and the live sync have an opinion on.

    Only days the sync actually holds are returned - a day the file alone covers
    is not a disagreement, it is a gap, and the importer fills it by the ordinary
    route.
    """
    file_cash = cash_by_day(file_executions)
    accounts = sorted({account for account, _ in file_cash})
    if not accounts:
        return []
    source_list = sorted(set(sources))
    account_slots = ",".join("?" for _ in accounts)
    source_slots = ",".join("?" for _ in source_list)
    with store.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM raw_executions
            WHERE broker = ?
              AND account_number IN ({account_slots})
              AND source IN ({source_slots})
            """,
            [str(broker).upper(), *accounts, *source_list],
        ).fetchall()

    journal: dict[tuple[str, date], list[Any]] = {}
    for raw in rows:
        row = {key: raw[key] for key in raw.keys()}
        day = row.get("trade_date")
        try:
            parsed = date.fromisoformat(str(day)[:10])
        except ValueError:
            continue
        journal.setdefault((str(row.get("account_number") or ""), parsed), []).append(row)

    comparisons: list[DayCash] = []
    for key, entries in sorted(journal.items()):
        if key not in file_cash:
            # The sync has a day the file does not mention. That is not the
            # file disagreeing - it is the file not covering it - and taking
            # the day over would delete real fills.
            continue
        cash, fills = file_cash[key]
        comparisons.append(
            DayCash(
                account=key[0],
                day=key[1],
                file_cash=cash,
                journal_cash=sum(signed_cash(row) for row in entries),
                file_fills=fills,
                journal_fills=len(entries),
                journal_uids=[str(row.get("execution_uid") or "") for row in entries],
            )
        )
    return comparisons


def apply_file_authority(
    store: Any,
    *,
    broker: str,
    file_executions: Sequence[Any],
    sources: Iterable[str],
    label: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Hand every disagreeing day to the file, keeping the days that agree.

    Returns what it looked at and what it changed. ``dry_run`` measures without
    writing, which is what the "Check a statement..." button uses so the trader
    can see which days would move before any of them do.
    """
    comparisons = compare_days(
        store, broker=broker, file_executions=file_executions, sources=sources
    )
    disagreeing = [item for item in comparisons if item.disagrees]

    taken: list[dict[str, Any]] = []
    if not dry_run:
        by_day: dict[tuple[str, date], list[Any]] = {}
        for execution in file_executions:
            row = _as_row(execution)
            try:
                day = date.fromisoformat(str(row.get("trade_date"))[:10])
            except ValueError:
                continue
            by_day.setdefault((str(row.get("account_number") or ""), day), []).append(execution)

        for item in disagreeing:
            reason = (
                f"{label}: the file and the live sync disagree on this day by "
                f"{item.difference:+.2f} (tolerance {item.tolerance:.2f}); the "
                "broker's own file is authoritative for money."
            )
            for uid in item.journal_uids:
                if not uid:
                    continue
                store.record_adjustment(
                    action="VOID_EXECUTION",
                    target_uid=uid,
                    reason=reason,
                    source="file_authority",
                    payload={
                        "broker": str(broker).upper(),
                        "account": item.account,
                        "day": item.day.isoformat(),
                        "file_cash": round(item.file_cash, 4),
                        "journal_cash": round(item.journal_cash, 4),
                    },
                )
            replacements = by_day.get((item.account, item.day), [])
            if replacements:
                store.upsert_executions(replacements)
            taken.append(
                {
                    "account": item.account,
                    "day": item.day.isoformat(),
                    "difference": round(item.difference, 4),
                    "voided": len(item.journal_uids),
                    "written": len(replacements),
                }
            )

    return {
        "days_compared": len(comparisons),
        "days_in_agreement": len(comparisons) - len(disagreeing),
        "days_taken_over": len(disagreeing),
        "days": [
            {
                "account": item.account,
                "day": item.day.isoformat(),
                "file_cash": round(item.file_cash, 4),
                "journal_cash": round(item.journal_cash, 4),
                "difference": round(item.difference, 4),
                "tolerance": round(item.tolerance, 4),
                "disagrees": item.disagrees,
            }
            for item in disagreeing
        ],
        "taken": taken,
        "dry_run": bool(dry_run),
    }


def describe_authority(report: Mapping[str, Any]) -> str:
    """The override as a line a person can read."""
    compared = int(report.get("days_compared") or 0)
    if not compared:
        return "No day is covered by both the file and the live sync."
    taken = int(report.get("days_taken_over") or 0)
    agreed = int(report.get("days_in_agreement") or 0)
    if not taken:
        return (
            f"{compared} day(s) covered by both; all {agreed} agree, so the live "
            "sync keeps them and its trade times survive."
        )
    verb = "would take over" if report.get("dry_run") else "took over"
    worst = sorted(
        report.get("days") or [], key=lambda row: -abs(float(row.get("difference") or 0.0))
    )[:3]
    detail = ", ".join(f"{row['account']} {row['day']} {row['difference']:+.2f}" for row in worst)
    return (
        f"{compared} day(s) covered by both; {agreed} agree. The file {verb} "
        f"{taken} day(s) where the money differs: {detail}"
    )


__all__ = [
    "TOLERANCE_BASE",
    "TOLERANCE_PER_FILL",
    "DayCash",
    "apply_file_authority",
    "cash_by_day",
    "compare_days",
    "describe_authority",
    "signed_cash",
]
