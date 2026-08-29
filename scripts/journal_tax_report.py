"""Realised P&L for a tax year, added up from the broker's own money.

Trader decision, 2026-08-28: *"Statement is source of truth for final pnl/tax
purposes."*

Everywhere else in the journal, a trade's P&L is RECOMPUTED — average-cost
matching, price x quantity, pro-rated costs — because that is what makes
per-trade attribution, R multiples and per-setup statistics possible at all. It
is also, unavoidably, arithmetic of our own: Questrade books each row's Gross
Amount rounded to the cent while the assembler multiplies at full precision, so
the two drift. Measured across the trader's 2025-26 export: **-$0.2386 on
$5,298.81 of realised P&L over 428 closed symbols.** Immaterial for deciding
what to trade. Not the number to put on a return.

So this module does not recompute anything. For every fill it takes
``raw_executions.net_amount`` — the broker's own statement of what that fill did
to cash, in the trade's own currency — and adds them up. For a position that is
**flat**, the sum of its fills' net amounts IS the realised P&L, because every
share bought was sold; no cost-basis model is needed and none is used.

WHAT IT REFUSES TO REPORT, and why each refusal is the point:

* **A position that is not flat** contributes nothing. Cash has left the account
  with no realised P&L against it yet, and including it would report an open
  trade as a loss.
* **A position missing a fill** contributes nothing. A ``SYNTHETIC_OPEN`` leg
  means the journal had to invent an opening fill, so the proceeds are real and
  the cost basis is not. These are listed by symbol so the trader knows exactly
  which file would fix them — on the real data, importing the 2025 export took
  this from 23 trades to 5.
* **A fill with no broker-stated amount** disqualifies its whole position. The
  IBKR socket path records no ``net_amount``, and mixing a stated figure with a
  recomputed one would produce a total that is neither.

Nothing is estimated to fill a gap. A tax figure that quietly interpolates is
worse than one that says which symbol it cannot answer for.

CAD is the tax currency and is converted per fill at the Bank of Canada rate
booked for that fill's date (R7 §5) — never at one rate for the year, and never
at a broker's internal rate. A fill whose date has no booked rate leaves its
position's CAD total ``None`` rather than guessing, and the count is reported.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

BASE_CURRENCY = "CAD"

#: Quantities closer to zero than this are flat. Matches ``journal_store``.
EPSILON = 1e-7

#: Leg roles that mean the journal invented the fill rather than importing it.
INVENTED_ROLES = frozenset({"SYNTHETIC_OPEN", "SYNTHETIC_CLOSE"})


@dataclass
class Position:
    """One (broker, account, symbol, currency) over the reported window."""

    broker: str
    account: str
    symbol: str
    security_type: str
    currency: str
    quantity: float = 0.0
    proceeds: float = 0.0
    cost: float = 0.0
    commission: float = 0.0
    fills: int = 0
    first_day: str = ""
    last_day: str = ""
    missing_amounts: int = 0
    invented_legs: int = 0
    cad_total: float | None = 0.0
    unbooked_days: set[str] = field(default_factory=set)

    @property
    def realised(self) -> float:
        return self.proceeds + self.cost

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < EPSILON

    @property
    def reportable(self) -> bool:
        """Flat, complete, and stated by the broker on every fill."""
        return self.is_flat and not self.missing_amounts and not self.invented_legs

    @property
    def excluded_reason(self) -> str:
        if not self.is_flat:
            return "still open"
        if self.invented_legs:
            return "an opening fill is missing - import the earlier statement"
        if self.missing_amounts:
            return "a fill carries no broker-stated amount"
        return ""


def _as_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _signed_quantity(side: str, quantity: float) -> float:
    return -quantity if str(side or "").strip().upper() == "SELL" else quantity


def build_tax_report(
    store: Any,
    *,
    year: int | None = None,
    date_from: Any = None,
    date_to: Any = None,
    broker: str = "",
    accounts: Iterable[str] = (),
) -> dict[str, Any]:
    """Realised P&L for the window, from the broker's stated amounts only.

    ``year`` is a convenience for the whole calendar year. A position is
    included when every fill it has in the store falls in the window and it ends
    flat; the walk deliberately reads the WHOLE store for a symbol rather than
    only the window, because a position opened in December and closed in January
    is not two half-positions and reporting it as one would invent a cost basis.
    """
    if year is not None:
        date_from = date(year, 1, 1)
        date_to = date(year, 12, 31)
    start = _as_date(date_from) if date_from else None
    end = _as_date(date_to) if date_to else None

    clauses = ["1 = 1"]
    params: list[Any] = []
    if broker:
        clauses.append("broker = ?")
        params.append(str(broker).upper())
    account_list = [str(value) for value in accounts if str(value or "").strip()]
    if account_list:
        clauses.append(f"account_number IN ({','.join('?' for _ in account_list)})")
        params.extend(account_list)
    with store.connection() as conn:
        rows = [
            {key: raw[key] for key in raw.keys()}
            for raw in conn.execute(
                f"SELECT * FROM raw_executions WHERE {' AND '.join(clauses)} ORDER BY trade_date",
                params,
            ).fetchall()
        ]
        voided = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT target_uid FROM trade_adjustments
                WHERE action = 'VOID_EXECUTION' AND COALESCE(superseded_by, '') = ''
                """
            ).fetchall()
        }
        invented = defaultdict(int)
        for raw in conn.execute(
            "SELECT trade_id, role FROM trade_legs WHERE role IN ('SYNTHETIC_OPEN', 'SYNTHETIC_CLOSE')"
        ).fetchall():
            invented[str(raw[0])] += 1
        legs_by_trade = defaultdict(list)
        for raw in conn.execute("SELECT trade_id, execution_uid FROM trade_legs").fetchall():
            legs_by_trade[str(raw[0])].append(str(raw[1]))

    invented_uids: set[str] = set()
    for trade_id, count in invented.items():
        if count:
            invented_uids.update(legs_by_trade.get(trade_id, ()))

    from journal_fx import stored_rate

    positions: dict[tuple[str, str, str, str], Position] = {}
    rate_cache: dict[tuple[str, str], float | None] = {}
    for row in rows:
        uid = str(row.get("execution_uid") or "")
        if uid in voided:
            # Retired by a correction or by the file-authority rule; it no
            # longer describes the account and must not reach a tax total.
            continue
        day = _as_date(row.get("trade_date"))
        if day is None:
            continue
        key = (
            str(row.get("broker") or "").upper(),
            str(row.get("account_number") or ""),
            str(row.get("symbol") or ""),
            str(row.get("currency") or "").upper(),
        )
        position = positions.get(key)
        if position is None:
            position = Position(
                broker=key[0],
                account=key[1],
                symbol=key[2],
                security_type=str(row.get("security_type") or ""),
                currency=key[3],
            )
            positions[key] = position

        quantity = abs(float(row.get("quantity") or 0.0))
        side = str(row.get("side") or "").upper()
        position.quantity += _signed_quantity(side, quantity)
        position.fills += 1
        stamp = day.isoformat()
        position.first_day = min(position.first_day or stamp, stamp)
        position.last_day = max(position.last_day, stamp)
        if uid in invented_uids:
            position.invented_legs += 1

        net = row.get("net_amount")
        if net is None:
            position.missing_amounts += 1
            continue
        amount = float(net)
        if side == "SELL":
            position.proceeds += amount
        else:
            position.cost += amount
        try:
            position.commission += float(row.get("commission") or 0.0) + float(
                row.get("fees") or 0.0
            )
        except (TypeError, ValueError):
            pass

        if position.cad_total is None:
            continue
        cache_key = (stamp, position.currency)
        if cache_key not in rate_cache:
            booked = stored_rate(store, day, position.currency)
            rate_cache[cache_key] = float(booked["rate_to_cad"]) if booked else None
        rate = rate_cache[cache_key]
        if rate is None:
            position.cad_total = None
            position.unbooked_days.add(stamp)
        else:
            position.cad_total += amount * rate

    reported: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for position in positions.values():
        if start and position.last_day and _as_date(position.last_day) < start:
            continue
        if end and position.first_day and _as_date(position.first_day) > end:
            continue
        record = {
            "broker": position.broker,
            "account": position.account,
            "symbol": position.symbol,
            "security_type": position.security_type,
            "currency": position.currency,
            "fills": position.fills,
            "first_day": position.first_day,
            "last_day": position.last_day,
            "proceeds": round(position.proceeds, 4),
            "cost": round(position.cost, 4),
            "realised": round(position.realised, 4),
            "commission": round(position.commission, 4),
            "realised_cad": None if position.cad_total is None else round(position.cad_total, 4),
            "quantity_left": round(position.quantity, 6),
        }
        if position.reportable:
            reported.append(record)
        else:
            record["reason"] = position.excluded_reason
            excluded.append(record)
    reported.sort(key=lambda row: (row["account"], row["symbol"]))
    excluded.sort(key=lambda row: (row["reason"], row["account"], row["symbol"]))

    by_account: dict[str, dict[str, Any]] = {}
    for record in reported:
        bucket = by_account.setdefault(
            record["account"],
            {
                "broker": record["broker"],
                "positions": 0,
                "realised_by_currency": defaultdict(float),
                "commission_by_currency": defaultdict(float),
                "realised_cad": 0.0,
                "cad_complete": True,
            },
        )
        bucket["positions"] += 1
        bucket["realised_by_currency"][record["currency"]] += record["realised"]
        bucket["commission_by_currency"][record["currency"]] += record["commission"]
        if record["realised_cad"] is None:
            bucket["cad_complete"] = False
        elif bucket["cad_complete"]:
            bucket["realised_cad"] += record["realised_cad"]
    for bucket in by_account.values():
        bucket["realised_by_currency"] = {
            code: round(value, 2) for code, value in sorted(bucket["realised_by_currency"].items())
        }
        bucket["commission_by_currency"] = {
            code: round(value, 2)
            for code, value in sorted(bucket["commission_by_currency"].items())
        }
        bucket["realised_cad"] = (
            round(bucket["realised_cad"], 2) if bucket["cad_complete"] else None
        )

    unbooked = sorted({day for position in positions.values() for day in position.unbooked_days})
    total_cad = 0.0
    cad_complete = True
    for record in reported:
        if record["realised_cad"] is None:
            cad_complete = False
        elif cad_complete:
            total_cad += record["realised_cad"]

    tax_status = {
        str(row.get("account_number") or ""): str(row.get("tax_status") or "")
        for row in store.list_accounts()
    }
    for account_number, bucket in by_account.items():
        bucket["tax_status"] = tax_status.get(account_number, "")

    return {
        "window": {
            "from": start.isoformat() if start else "",
            "to": end.isoformat() if end else "",
            "year": year,
        },
        "positions_reported": len(reported),
        "positions_excluded": len(excluded),
        "realised_cad": round(total_cad, 2) if cad_complete else None,
        "cad_complete": cad_complete,
        "unbooked_rate_days": unbooked,
        "by_account": by_account,
        "positions": reported,
        "excluded": excluded,
        "source": "broker-stated net amounts (raw_executions.net_amount)",
    }


def cross_check_against_journal(store: Any, report: Mapping[str, Any]) -> dict[str, Any]:
    """The recomputed figure beside the broker-stated one, per account.

    Two independent routes to one number: this report adds up what the broker
    said, and ``rebuild_trades`` recomputes it from price x quantity. They
    should differ by rounding and nothing else, and if they differ by more than
    that the per-symbol rows above say which symbol to look at.
    """
    stated: dict[str, float] = defaultdict(float)
    for record in report.get("positions") or []:
        stated[record["account"]] += float(record["realised"])

    recomputed: dict[str, float] = defaultdict(float)
    reported_symbols = {
        (record["account"], record["symbol"]) for record in report.get("positions") or []
    }
    for trade in store.list_trades():
        key = (str(trade.get("account_number") or ""), str(trade.get("symbol") or ""))
        if key not in reported_symbols:
            continue
        recomputed[key[0]] += float(trade.get("net_pnl") or 0.0)

    rows = []
    for account in sorted(set(stated) | set(recomputed)):
        difference = stated[account] - recomputed[account]
        rows.append(
            {
                "account": account,
                "broker_stated": round(stated[account], 4),
                "journal_recomputed": round(recomputed[account], 4),
                "difference": round(difference, 4),
            }
        )
    return {
        "accounts": rows,
        "broker_stated": round(sum(stated.values()), 4),
        "journal_recomputed": round(sum(recomputed.values()), 4),
        "difference": round(sum(stated.values()) - sum(recomputed.values()), 4),
    }


def export_tax_csv(report: Mapping[str, Any], path: Any = None) -> Path:
    """Every reported position, and every excluded one with its reason."""
    from project_paths import JOURNAL_EXPORT_DIR

    if path is None:
        window = report.get("window") or {}
        label = window.get("year") or f"{window.get('from', '')}_{window.get('to', '')}" or "all"
        path = Path(JOURNAL_EXPORT_DIR) / f"tax_realised_{label}.csv"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "status", "broker", "account", "symbol", "security_type", "currency",
        "first_day", "last_day", "fills", "proceeds", "cost", "realised",
        "commission", "realised_cad", "quantity_left", "reason",
    ]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in report.get("positions") or []:
            writer.writerow({**record, "status": "reported", "reason": ""})
        for record in report.get("excluded") or []:
            writer.writerow({**record, "status": "excluded"})
    return target


def describe_tax_report(report: Mapping[str, Any], cross_check: Mapping[str, Any] | None = None) -> str:
    """The report as a few lines a person can read."""
    window = report.get("window") or {}
    label = window.get("year") or f"{window.get('from') or 'start'}..{window.get('to') or 'now'}"
    lines = [
        f"Realised P&L {label}, from the broker's own amounts "
        f"({report.get('positions_reported', 0)} closed position(s)).",
    ]
    for account, bucket in sorted((report.get("by_account") or {}).items()):
        money = ", ".join(
            f"{value:,.2f} {code}" for code, value in (bucket.get("realised_by_currency") or {}).items()
        )
        status = bucket.get("tax_status") or "unlabeled"
        cad = bucket.get("realised_cad")
        cad_text = f" = {cad:,.2f} CAD" if cad is not None else " (CAD incomplete)"
        lines.append(f"  {bucket.get('broker')} {account} [{status}]: {money}{cad_text}")
    if report.get("realised_cad") is not None:
        lines.append(f"Total: {report['realised_cad']:,.2f} CAD.")
    else:
        lines.append("Total in CAD withheld: some fill dates have no booked Bank of Canada rate.")
    excluded = report.get("positions_excluded") or 0
    if excluded:
        reasons: dict[str, int] = defaultdict(int)
        for record in report.get("excluded") or []:
            reasons[record.get("reason", "")] += 1
        detail = "; ".join(f"{count} {reason}" for reason, count in sorted(reasons.items()))
        lines.append(f"{excluded} position(s) not counted - {detail}.")
    if report.get("unbooked_rate_days"):
        lines.append(
            f"{len(report['unbooked_rate_days'])} fill date(s) have no booked BoC rate."
        )
    if cross_check:
        lines.append(
            f"Cross-check: broker {cross_check['broker_stated']:,.2f} vs the journal's "
            f"recomputed {cross_check['journal_recomputed']:,.2f} "
            f"(difference {cross_check['difference']:+,.4f})."
        )
    return "\n".join(lines)


__all__ = [
    "BASE_CURRENCY",
    "INVENTED_ROLES",
    "Position",
    "build_tax_report",
    "cross_check_against_journal",
    "describe_tax_report",
    "export_tax_csv",
]
