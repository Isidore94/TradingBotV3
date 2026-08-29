"""Import an IBKR Transaction History file into the journal.

The Questrade half of this lives in ``journal_statement_import``. IBKR's export
is a different animal in three ways that each silently produce a wrong number if
carried over unchanged, so it gets its own reader rather than a widened one.

**1. It is a SECTIONED csv, not a table.** Every line begins with a section name
(``Statement``, ``Transaction History``, ``Summary``) and then ``Header`` or
``Data``. A plain ``csv.DictReader`` reads the first section's header and then
misaligns every later row against it.

**2. Money is in the BASE currency and prices are not.** On the trader's own
file, ``Price`` is USD while ``Gross Amount`` and ``Net Amount`` are CAD — a
3-share sell at 366.19 USD books a gross of 1516.905456. Passing both through
untouched would compute a USD gross P&L and subtract a CAD commission from it,
which is a wrong number that looks entirely plausible. Executions are therefore
stored in the trade's OWN currency, with the cost converted using the rate the
row itself implies:

    rate = |Gross Amount| / |quantity x price x multiplier|

That rate is IB's own, for that trade, on that day. It is recorded in
``raw_json`` as evidence and is deliberately NOT booked into ``fx_rates``, which
is a Bank-of-Canada table by design (R7 §5) — a broker's internal rate is not
the rate a tax return uses. Across 608 trade rows the implied rate ran
1.35530–1.45270, which is the USD/CAD band for the period and is the check that
this reading is right rather than a coincidence.

**3. Account numbers arrive MASKED** — ``U***2524``, ``U***7396``. A masked
number cannot be an identity: the same account reached through Flex or the
socket carries its full number, and treating the two as different accounts
splits one position in half. :func:`resolve_account_number` unmasks against the
accounts the journal already knows, and only when exactly one of them fits.
Nothing is guessed: an unresolved mask keeps its masked form, is reported, and
the trader can map it.

Everything the Questrade importer learned applies here too: the file has **no
time of day**, so executions are written at midnight market-local and
``journal_trade_shape.is_date_only`` refuses to name a session for them; and a
file **never writes into a (broker, account, day) a richer source already
covers**, because the two give one fill different uids and the upsert cannot see
the duplicate.

Options need no parsing — IB already writes OCC (``DRAM  261218P00055000``) —
but the 100 multiplier still has to reach the gross, and it is part of the
implied-rate denominator above. An ``Assignment`` is a real fill: the row that
says ``Buy 100 ROUNDHILL MEMORY ETF (Assignment)`` is how an option that was
assigned becomes a stock position, and dropping it leaves the position open
forever with nothing that can close it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from journal_file_authority import apply_file_authority
from journal_identity import canonical_option_symbol, normalize_security_type, stable_execution_uid
from journal_importers import NormalizedExecution, _cash_txn_uid, classify_activity_type
from journal_statement_import import (
    RICHER_SOURCES,
    _coerce_date,
    statement_timestamp,
)

#: ``raw_executions.source`` for a row from an IBKR transaction file.
IB_FILE_SOURCE = "IBKR_FILE"

#: The section every trade and cash row lives in.
TRANSACTION_SECTION = "Transaction History"

#: Transaction types that move shares or contracts. ``Assignment`` is here
#: because an assigned option becomes a stock position through a real fill; the
#: side comes from the description, which reads "Buy 100 ..." or "Sell 100 ...".
TRADE_TYPES = frozenset({"BUY", "SELL", "ASSIGNMENT", "EXERCISE", "EXPIRATION"})

#: Everything else is cash. IB's vocabulary is wider than Questrade's.
CASH_TYPE_TO_ACTIVITY = {
    "SALES TAX": "FEE",
    "OTHER FEE": "FEE",
    "COMMISSION ADJUSTMENT": "FEE",
    "DEBIT INTEREST": "INTEREST",
    "CREDIT INTEREST": "INTEREST",
    "BROKER INTEREST PAID": "INTEREST",
    "BROKER INTEREST RECEIVED": "INTEREST",
    "DIVIDEND": "DIVIDEND",
    "PAYMENT IN LIEU OF DIVIDEND": "DIVIDEND",
    "WITHHOLDING TAX": "FEE",
    "DEPOSIT": "OTHER",
    "WITHDRAWAL": "OTHER",
    "FOREX TRADE COMPONENT": "FX",
    "ADJUSTMENT": "OTHER",
}

#: Leading verb of an assignment/exercise description.
_ASSIGNMENT_SIDE = re.compile(r"^\s*(?P<side>BUY|SELL)\b", re.IGNORECASE)

#: How far an implied FX rate may sit from 1.0 before it is treated as a real
#: conversion rather than a same-currency row. A base-currency trade implies
#: exactly 1.0; floating point puts it a hair off.
_SAME_CURRENCY_EPSILON = 1e-6


@dataclass
class IBRow:
    """One parsed Transaction History line."""

    sequence: int
    trade_date: date
    account: str
    description: str
    transaction_type: str
    symbol: str
    quantity: float | None
    price: float | None
    price_currency: str
    gross_amount: float | None
    commission: float | None
    net_amount: float | None


@dataclass
class IBParse:
    """Everything one IBKR transaction file yielded."""

    executions: list[NormalizedExecution] = field(default_factory=list)
    cash: list[dict[str, Any]] = field(default_factory=list)
    accounts: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    unresolved_accounts: list[str] = field(default_factory=list)
    trade_days: set[tuple[str, date]] = field(default_factory=set)
    cash_days: set[tuple[str, date]] = field(default_factory=set)
    base_currency: str = ""

    @property
    def date_range(self) -> tuple[date | None, date | None]:
        days = [day for _, day in (self.trade_days | self.cash_days)]
        return (min(days), max(days)) if days else (None, None)


# -- reading the sectioned file ----------------------------------------------


def read_ib_sections(path: Path) -> dict[str, list[dict[str, str]]]:
    """Every section of an IBKR csv as heading->value dicts.

    IBKR writes one file containing several tables. Each line names its section
    and says whether it is that section's ``Header`` or a ``Data`` row, so the
    header has to be tracked PER SECTION - reading the first one and applying it
    to everything, which is what a plain ``DictReader`` does, misaligns every
    row after the first table.
    """
    sections: dict[str, list[dict[str, str]]] = {}
    headers: dict[str, list[str]] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            section, kind = row[0].strip(), row[1].strip()
            if kind == "Header":
                headers[section] = [cell.strip() for cell in row[2:]]
                sections.setdefault(section, [])
                continue
            if kind != "Data":
                continue
            names = headers.get(section)
            if not names:
                continue
            sections.setdefault(section, []).append(
                {name: value for name, value in zip(names, row[2:])}
            )
    return sections


def looks_like_ib_transactions(path: Path) -> bool:
    """Is this an IBKR transaction file rather than a Questrade statement?

    Read from the file's own first bytes rather than from its name, because the
    name is whatever the trader saved it as and both brokers ship ``.csv``.
    """
    try:
        with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return TRANSACTION_SECTION in head or head.lstrip().startswith("Statement,Header")


# -- masked account numbers --------------------------------------------------


def mask_matches(masked: str, candidate: str) -> bool:
    """Could ``candidate`` be the account this mask hides?

    ``U***7396`` hides the middle of ``U4867396``: same length, same visible
    characters. Length is part of the test because ``U***7396`` and a longer
    ``U12345697396`` share a suffix and are not the same account.
    """
    masked = str(masked or "").strip()
    candidate = str(candidate or "").strip()
    if not masked or not candidate or len(masked) != len(candidate):
        return False
    return all(left == "*" or left == right for left, right in zip(masked, candidate))


def resolve_account_number(
    masked: str, known: Iterable[str], *, filename_hint: str = ""
) -> str:
    """Unmask an account against the ones the journal already holds.

    Returns the masked value unchanged when it cannot be resolved with
    certainty. That is the point: a masked number that silently becomes a
    guessed real one would merge or split positions on a hunch, and the same
    account reached through Flex carries its full number - so a wrong answer
    here shows up as half a position, which is exactly the class of defect R7
    was built to end.

    Resolution requires EXACTLY ONE candidate to fit. The filename is consulted
    only as another candidate, never as an override: an IBKR export is named for
    one account but can contain rows for several.
    """
    text = str(masked or "").strip()
    if "*" not in text:
        return text
    candidates = {str(value).strip() for value in known if str(value or "").strip()}
    hint = str(filename_hint or "").strip()
    if hint:
        candidates.add(hint)
    fits = sorted(value for value in candidates if mask_matches(text, value))
    return fits[0] if len(fits) == 1 else text


def account_hint_from_filename(path: Path) -> str:
    """The account an IBKR export is named for, e.g. ``U4867396.TRANSACTIONS...``.

    Searched anywhere in the name, not just at the start: a file that has been
    downloaded, copied or re-saved often picks up a prefix, and losing the hint
    to one would leave an account masked that the file itself could name.
    """
    match = re.search(r"\b([A-Z]{1,2}\d{6,12})\b", Path(path).name)
    return match.group(1) if match else ""


# -- parsing rows ------------------------------------------------------------


def _optional_float(value: Any) -> float | None:
    """IBKR writes ``-`` for "this column does not apply to this row"."""
    text = str(value or "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse_rows(rows: Iterable[Mapping[str, Any]]) -> list[IBRow]:
    parsed: list[IBRow] = []
    for index, raw in enumerate(rows):
        lookup = {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in raw.items()}
        trade_date = _coerce_date(lookup.get("date"))
        if trade_date is None:
            continue
        parsed.append(
            IBRow(
                sequence=index,
                trade_date=trade_date,
                account=str(lookup.get("account") or "").strip(),
                description=str(lookup.get("description") or "").strip(),
                transaction_type=str(lookup.get("transactiontype") or "").strip(),
                symbol=str(lookup.get("symbol") or "").strip(),
                quantity=_optional_float(lookup.get("quantity")),
                price=_optional_float(lookup.get("price")),
                price_currency=str(lookup.get("pricecurrency") or "").strip().upper(),
                gross_amount=_optional_float(lookup.get("grossamount")),
                commission=_optional_float(lookup.get("commission")),
                net_amount=_optional_float(lookup.get("netamount")),
            )
        )
    return parsed


def side_for(row: IBRow) -> str:
    """BUY or SELL, or ``""`` when the row is not a fill.

    A plain ``Buy``/``Sell`` says so in its type. An ``Assignment`` says it in
    the description ("Buy 100 ROUNDHILL MEMORY ETF (Assignment)"), and the
    quantity's sign is the fallback - IB signs a sale negative.
    """
    kind = row.transaction_type.strip().upper()
    if kind in {"BUY", "SELL"}:
        return kind
    if kind not in TRADE_TYPES:
        return ""
    match = _ASSIGNMENT_SIDE.match(row.description)
    if match:
        return match.group("side").upper()
    if row.quantity is None or row.quantity == 0:
        return ""
    return "BUY" if row.quantity > 0 else "SELL"


def implied_fx_rate(row: IBRow, multiplier: float) -> float | None:
    """Base-currency amount over native amount, from the row's own numbers.

    ``None`` when the row cannot support the division. This is the rate IB used
    for this trade; it converts the cost columns into the trade's own currency
    so a USD gross is never reduced by a CAD commission.
    """
    if row.gross_amount is None or row.quantity is None or row.price is None:
        return None
    native = abs(row.quantity) * abs(row.price) * (multiplier or 1.0)
    if native <= 0:
        return None
    rate = abs(row.gross_amount) / native
    return rate if rate > 0 else None


def _execution_from_row(row: IBRow, account_number: str, *, ordinal: int = 0) -> NormalizedExecution | None:
    side = side_for(row)
    if not side or not row.quantity or row.price is None or not row.symbol or row.symbol == "-":
        return None

    compact = re.sub(r"\s+", " ", row.symbol).strip()
    is_option = bool(re.fullmatch(r"[A-Z0-9.]{1,6}\s*\d{6}[CP]\d{8}", compact.replace(" ", "")))
    security_type = normalize_security_type("OPT" if is_option else "STK")
    symbol = canonical_option_symbol(compact, security_type)
    multiplier = 100.0 if security_type == "OPT" else 1.0

    rate = implied_fx_rate(row, multiplier)
    currency = row.price_currency or "USD"
    # Costs arrive in the base currency; the trade is stored in its own. A rate
    # we could not derive means we must not convert, so the cost is carried at
    # face value and the row says so rather than scaling by a guess.
    #
    # The SIGN is flipped rather than dropped. IB writes a charge as a negative
    # cash amount and the journal stores a cost as positive, but 18 of 609 rows
    # on the trader's own file carry a POSITIVE commission - a rebate - and
    # taking abs() there turned each credit into a charge. That single mistake
    # was the entire $2.17 by which the file and the journal disagreed.
    commission_base = -(row.commission or 0.0)
    if rate and abs(rate - 1.0) > _SAME_CURRENCY_EPSILON:
        commission_native = commission_base / rate
        converted = True
    else:
        commission_native = commission_base
        converted = rate is not None

    signature = (
        account_number,
        row.trade_date.isoformat(),
        symbol,
        side,
        f"{abs(row.quantity):.6f}",
        f"{row.price:.8f}",
        f"{abs(commission_base):.6f}",
        currency,
    )
    digest = hashlib.sha256("|".join(signature).encode("utf-8")).hexdigest()[:16]
    payload = {
        "source": IB_FILE_SOURCE,
        "file_sequence": row.sequence,
        "transaction_type": row.transaction_type,
        "description": row.description,
        "ib_symbol": row.symbol,
        "gross_amount_base": row.gross_amount,
        "net_amount_base": row.net_amount,
        "commission_base": row.commission,
        "implied_fx_rate": rate,
        "cost_converted_to_native": converted,
        "multiplier": multiplier,
        "time_of_day_known": False,
    }
    return NormalizedExecution(
        execution_uid=stable_execution_uid(
            "IBKR", account_number, f"ibfile-{ordinal:03d}-{digest}"
        ),
        source=IB_FILE_SOURCE,
        broker="IBKR",
        account_number=account_number,
        account_label=account_number,
        account_type="",
        symbol=symbol,
        security_type=security_type,
        currency=currency,
        side=side,
        quantity=abs(row.quantity),
        price=abs(row.price),
        timestamp=statement_timestamp(row.trade_date),
        trade_date=row.trade_date.isoformat(),
        commission=round(commission_native, 6),
        fees=0.0,
        # Stored in the trade's OWN currency, not IB's base. `net_amount` is
        # the broker's own statement of what the fill did to cash, and the tax
        # report adds those up rather than recomputing price x quantity - so it
        # has to mean the same thing for every broker in the store. Questrade
        # and Flex already report natively; only this file does not, and the
        # base figures stay in raw_json as evidence.
        gross_amount=(
            row.gross_amount / rate
            if rate and row.gross_amount is not None
            else row.gross_amount
        ),
        net_amount=(
            row.net_amount / rate if rate and row.net_amount is not None else row.net_amount
        ),
        order_id="",
        exchange_exec_id="",
        raw_json=json.dumps(payload, sort_keys=True, default=str),
    )


def _cash_from_row(row: IBRow, account_number: str, base_currency: str, *, ordinal: int = 0) -> dict[str, Any] | None:
    if not account_number:
        return None
    amount = row.net_amount if row.net_amount is not None else row.gross_amount
    if amount is None:
        return None
    kind = row.transaction_type.strip().upper()
    activity = CASH_TYPE_TO_ACTIVITY.get(kind) or classify_activity_type(row.transaction_type)
    symbol = "" if row.symbol in {"", "-"} else row.symbol
    return {
        "txn_uid": _cash_txn_uid(
            "IBKR",
            account_number,
            row.trade_date.isoformat(),
            activity,
            symbol,
            amount,
            row.description,
            ordinal,
        ),
        "broker": "IBKR",
        "account_number": account_number,
        "txn_date": row.trade_date.isoformat(),
        "activity_type": activity,
        "description": row.description,
        "symbol": symbol,
        # Cash rows are reported in the base currency and are not converted:
        # unlike a trade they carry no price to imply a rate from.
        "amount": amount,
        "currency": base_currency or "CAD",
        "raw_json": json.dumps(
            {
                "source": IB_FILE_SOURCE,
                "file_sequence": row.sequence,
                "transaction_type": row.transaction_type,
                "base_currency": base_currency,
            },
            sort_keys=True,
            default=str,
        ),
    }


def parse_ib_transactions(
    sections: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    known_accounts: Iterable[str] = (),
    filename_hint: str = "",
) -> IBParse:
    """Everything an IBKR transaction file yields, with what it could not read."""
    result = IBParse()
    summary = {
        str(item.get("Field Name") or "").strip(): str(item.get("Field Value") or "").strip()
        for item in sections.get("Summary", [])
    }
    result.base_currency = (summary.get("Base Currency") or "CAD").upper()

    known = list(known_accounts)
    rows = parse_rows(sections.get(TRANSACTION_SECTION, []))
    resolved: dict[str, str] = {}
    for row in rows:
        if row.account and row.account not in resolved:
            resolved[row.account] = resolve_account_number(
                row.account, known, filename_hint=filename_hint
            )
    result.unresolved_accounts = sorted(
        masked for masked, value in resolved.items() if "*" in value
    )
    result.accounts = [
        {"number": value, "type": "", "label": value}
        for value in sorted(set(resolved.values()))
        if value
    ]

    seen_fills: Counter[tuple[Any, ...]] = Counter()
    seen_cash: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        account_number = resolved.get(row.account, row.account)
        kind = row.transaction_type.strip().upper()
        if kind in TRADE_TYPES:
            probe = _execution_from_row(row, account_number)
            if probe is None:
                result.skipped.append(
                    {
                        "sequence": row.sequence,
                        "reason": "trade row is not a readable fill",
                        "type": row.transaction_type,
                        "symbol": row.symbol,
                        "date": row.trade_date.isoformat(),
                    }
                )
                continue
            key = (account_number, row.trade_date, probe.symbol, probe.side, probe.quantity, probe.price)
            seen_fills[key] += 1
            execution = _execution_from_row(row, account_number, ordinal=seen_fills[key] - 1)
            if execution is not None:
                result.executions.append(execution)
                result.trade_days.add((account_number, row.trade_date))
            continue

        cash_key = (account_number, row.trade_date, kind, row.symbol, row.net_amount, row.description)
        seen_cash[cash_key] += 1
        cash = _cash_from_row(
            row, account_number, result.base_currency, ordinal=seen_cash[cash_key] - 1
        )
        if cash is None:
            result.skipped.append(
                {
                    "sequence": row.sequence,
                    "reason": "cash row has no account or amount",
                    "type": row.transaction_type,
                    "date": row.trade_date.isoformat(),
                }
            )
            continue
        result.cash.append(cash)
        result.cash_days.add((account_number, row.trade_date))
    return result


def read_ib_file(path: Path, *, known_accounts: Iterable[str] = ()) -> IBParse:
    """Read and parse an IBKR transaction file in one step."""
    return parse_ib_transactions(
        read_ib_sections(Path(path)),
        known_accounts=known_accounts,
        filename_hint=account_hint_from_filename(Path(path)),
    )


# -- applying it to the store ------------------------------------------------


def import_ib_transaction_file(
    store: Any, path: Path, *, rebuild: bool = True, file_authority: bool = True
) -> dict[str, Any]:
    """Read an IBKR transaction file and write what it may safely write.

    Same day-level rule as the Questrade importer: a file never writes into a
    ``(broker, account, day)`` that a richer source already covers, because a
    Flex or socket row for that fill has a real execution id and a real time and
    the two identities cannot be compared by the upsert.
    """
    known = [
        str(row.get("account_number") or "")
        for row in store.list_accounts()
        if str(row.get("broker") or "").upper() == "IBKR"
    ]
    parse = read_ib_file(Path(path), known_accounts=known)
    account_numbers = [str(account["number"]) for account in parse.accounts if account.get("number")]

    start, end = parse.date_range
    run_id = store.start_import_run(
        "IBKR_FILE",
        account_number=",".join(sorted(account_numbers)),
        trigger="trader_file_import",
        coverage_start=start.isoformat() if start else "",
        coverage_end=end.isoformat() if end else "",
    )
    try:
        if parse.accounts:
            store.upsert_accounts("IBKR", parse.accounts)

        blocked = _blocked_days(store, account_numbers)
        executions = [
            execution
            for execution in parse.executions
            if (execution.account_number, _coerce_date(execution.trade_date)) not in blocked
        ]
        cash = [
            row
            for row in parse.cash
            if (row["account_number"], _coerce_date(row["txn_date"])) not in blocked
        ]
        skipped_days = sorted(
            {day for day in parse.trade_days | parse.cash_days if day in blocked},
            key=lambda item: (item[0], item[1]),
        )

        written_executions = store.upsert_executions(executions) if executions else 0
        written_cash = store.upsert_cash_transactions(cash) if cash else 0

        # Same rule as the Questrade importer: the file outranks Flex and the
        # socket on money, and they keep the day when the two agree so their
        # trade times survive.
        authority = apply_file_authority(
            store,
            broker="IBKR",
            file_executions=parse.executions,
            sources=RICHER_SOURCES,
            label=Path(path).name,
            dry_run=not file_authority,
        )
        if rebuild and (written_executions or authority.get("days_taken_over")):
            store.rebuild_trades()

        written_days = sorted(
            {
                (execution.account_number, _coerce_date(execution.trade_date))
                for execution in executions
            },
            key=lambda item: (item[0], item[1]),
        )
        if written_days:
            from journal_coverage import COVERED
            from journal_coverage import mark_coverage as mark

            for account_number, day in written_days:
                if day is None:
                    continue
                mark(
                    store,
                    broker="IBKR",
                    account_number=account_number,
                    day=day,
                    status=COVERED,
                    source=IB_FILE_SOURCE,
                    import_run_id=run_id,
                    message=f"IBKR transaction file: {Path(path).name}",
                )

        summary = {
            "file": Path(path).name,
            "broker": "IBKR",
            "base_currency": parse.base_currency,
            "executions_written": int(written_executions),
            "cash_written": int(written_cash),
            "days_written": len(written_days),
            "days_skipped_richer_source": len(skipped_days),
            "authority": authority,
            "skipped_days": [(account, day.isoformat()) for account, day in skipped_days],
            "unreadable_rows": len(parse.skipped),
            "unresolved_accounts": parse.unresolved_accounts,
            "accounts": sorted(account_numbers),
            "coverage_start": start.isoformat() if start else "",
            "coverage_end": end.isoformat() if end else "",
        }
        store.finish_import_run(
            run_id,
            status="OK",
            imported_executions=int(written_executions),
            message=describe_ib_summary(summary),
        )
        return summary
    except Exception as exc:  # noqa: BLE001 - the run records its own failure
        store.finish_import_run(
            run_id, status="FAILED", imported_executions=0, message=f"{type(exc).__name__}: {exc}"
        )
        raise


def _blocked_days(store: Any, account_numbers: Sequence[str]) -> set[tuple[str, date]]:
    if not account_numbers:
        return set()
    accounts = ",".join("?" for _ in account_numbers)
    sources = ",".join("?" for _ in sorted(RICHER_SOURCES))
    with store.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT account_number, trade_date
            FROM raw_executions
            WHERE broker = 'IBKR'
              AND account_number IN ({accounts})
              AND source IN ({sources})
            """,
            [*account_numbers, *sorted(RICHER_SOURCES)],
        ).fetchall()
    blocked: set[tuple[str, date]] = set()
    for account_number, trade_date in rows:
        day = _coerce_date(trade_date)
        if day is not None:
            blocked.add((str(account_number), day))
    return blocked


def describe_ib_summary(summary: Mapping[str, Any]) -> str:
    parts = [
        f"{summary.get('executions_written', 0)} executions",
        f"{summary.get('cash_written', 0)} cash rows",
        f"{summary.get('days_written', 0)} days",
    ]
    if summary.get("days_skipped_richer_source"):
        parts.append(f"{summary['days_skipped_richer_source']} day(s) already covered by Flex")
    authority = summary.get("authority") or {}
    if authority.get("days_taken_over"):
        parts.append(f"{authority['days_taken_over']} day(s) taken over on a money difference")
    elif authority.get("days_compared"):
        parts.append(f"{authority['days_compared']} shared day(s) agree")
    if summary.get("unreadable_rows"):
        parts.append(f"{summary['unreadable_rows']} row(s) unreadable")
    unresolved = summary.get("unresolved_accounts") or []
    if unresolved:
        parts.append(f"account(s) still masked: {', '.join(unresolved)}")
    span_start = summary.get("coverage_start") or ""
    span_end = summary.get("coverage_end") or ""
    span = f" {span_start}..{span_end}" if span_start else ""
    return f"{summary.get('file', 'IBKR file')}{span}: " + ", ".join(parts)


__all__ = [
    "CASH_TYPE_TO_ACTIVITY",
    "IB_FILE_SOURCE",
    "IBParse",
    "IBRow",
    "TRADE_TYPES",
    "account_hint_from_filename",
    "describe_ib_summary",
    "implied_fx_rate",
    "import_ib_transaction_file",
    "looks_like_ib_transactions",
    "mask_matches",
    "parse_ib_transactions",
    "parse_rows",
    "read_ib_file",
    "read_ib_sections",
    "resolve_account_number",
    "side_for",
]
