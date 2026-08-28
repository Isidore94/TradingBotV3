"""Import a Questrade activity statement (.xlsx or .csv) into the journal.

WHY THIS EXISTS

Questrade's ``/v1/accounts/{id}/executions`` endpoint has a retention horizon.
On this desk it reaches back to 2026-06-10 and no further, which is why 44 of
the 45 ``activities report trades the executions endpoint did not return`` days
can never be repaired by retrying: the fills are simply gone from the API. The
trader can, however, download an activity statement for any period from the
Questrade portal, and asked that we be able to process one.

WHAT A STATEMENT IS AND IS NOT

It is authoritative for money. Across 884 trade rows in the first file we read,
``Net Amount == Gross Amount + Commission`` held to the cent with **zero**
exceptions, so the single Commission column is the complete cost of the trade -
there are no separate fee columns hiding a number, which is exactly the thing a
tax total must not miss.

It is NOT authoritative for time, and this is the constraint the whole module is
shaped around:

* **No time of day.** Every row is stamped "12:00:00 AM". A statement says a
  trade happened on a DATE. Executions are therefore written at midnight
  market-local, which ``journal_trade_shape.is_date_only`` recognises as "time
  unknown" and refuses to tag with a session bucket. Writing them at 09:30
  instead would have tagged an entire imported year ``opening_drive``.
* **Fills are aggregated.** Some rows say "AVG PRICE" in as many words, so one
  statement row can be several fills. Quantities and money still reconcile;
  individual fill prices do not survive.
* **No execution id, and no intraday sequence.** Two rows for one symbol on one
  day carry nothing that says which came first. We preserve the statement's own
  row order (see ``sequence``) because it is the broker's own listing and it is
  reproducible, but a same-day round trip's LONG/SHORT label is that ordering's
  claim rather than a measured fact. What this cannot get wrong is the day's
  money: a symbol that starts and ends a day flat realises the same total P&L
  whichever way the legs are paired, because every leg is closed either way.
  Per-trade attribution inside such a day is best-effort; the day total, which
  is what a tax return adds up, is exact.

THE RULE THAT KEEPS IT SAFE

**A statement never writes into a (broker, account, day) that a richer source
already covers.** The API rows carry real execution ids, real timestamps and
split fills; statement rows carry none of those, and the two have different
identities, so nothing in the ``execution_uid`` upsert would notice they are the
same fill. Importing both would silently double a position. The skip is at DAY
granularity because that is the granularity the statement can be trusted at, and
the count of skipped days is reported rather than swallowed.

Reading .xlsx is done with ``zipfile`` and ``xml.etree`` rather than by adding
``openpyxl``: an xlsx is a zip of XML, the sheet Questrade emits is one flat
table, and a new third-party dependency is a packaging trigger that would owe a
frozen rebuild for a file format we can already read in fifty lines.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from journal_identity import (
    canonical_option_symbol,
    normalize_security_type,
    stable_execution_uid,
)
from journal_importers import (
    NormalizedExecution,
    _cash_txn_uid,
    classify_activity_type,
    normalize_side,
)

MARKET_TZ = ZoneInfo("America/New_York")

#: ``raw_executions.source`` for a row that came from a statement file. Ranked
#: below ``QT_API`` in ``journal_migrate.SOURCE_RANK`` so that if the two ever
#: do land on one uid, the API row - which has a real time and a real execution
#: id - wins. The day-level skip below is the primary defence; this is the belt.
STATEMENT_SOURCE = "QT_STATEMENT"

#: Sources whose presence on a day makes that day off-limits to a statement.
#: Everything except a previous statement import of the same day, which is
#: idempotent because the surrogate uid is deterministic.
RICHER_SOURCES = frozenset({"QT_API", "IBKR_FLEX", "IBKR_SOCKET", "MANUAL", "CSV"})

#: The Questrade activity types that describe a trade. Everything else is cash.
TRADE_ACTIVITY_TYPES = frozenset({"TRADES"})

#: Column headings we need, lower-cased and stripped of punctuation.
_COLUMN_ALIASES = {
    "transactiondate": "transaction_date",
    "settlementdate": "settlement_date",
    "action": "action",
    "symbol": "symbol",
    "description": "description",
    "quantity": "quantity",
    "price": "price",
    "grossamount": "gross_amount",
    "commission": "commission",
    "netamount": "net_amount",
    "currency": "currency",
    "account": "account_number",
    "accountnumber": "account_number",
    "activitytype": "activity_type",
    "accounttype": "account_type",
}

#: Questrade account-type wording -> the journal's tax vocabulary. Seeded, never
#: forced: ``set_account_tax_status`` is trader-owned (I6), so this only fills a
#: blank and an unrecognised wording stays unlabeled rather than guessed.
TAX_STATUS_BY_ACCOUNT_TYPE = {
    "INDIVIDUAL TFSA": "TAX_FREE",
    "TFSA": "TAX_FREE",
    "INDIVIDUAL RRSP": "TAX_DEFERRED",
    "RRSP": "TAX_DEFERRED",
    "SPOUSAL RRSP": "TAX_DEFERRED",
    "LIRA": "TAX_DEFERRED",
    "RESP": "TAX_DEFERRED",
    "FAMILY RESP": "TAX_DEFERRED",
    "INDIVIDUAL MARGIN": "TAXABLE",
    "MARGIN": "TAXABLE",
    "INDIVIDUAL CASH": "TAXABLE",
    "CASH": "TAXABLE",
}

#: An option line in the Description column, e.g.
#: "PUT SPY 06/23/26 737 STATE STREET SPDR S&P 500 ETF WE ACTED AS AGENT".
#: The Symbol column for these rows is a Questrade internal id (``8SVDLK9``),
#: which is meaningless outside their system and would make every contract its
#: own position, so the real contract is read from the description instead.
_OPTION_DESCRIPTION = re.compile(
    r"^\s*(?P<right>CALL|PUT)\s+"
    r"(?P<underlying>[A-Z][A-Z0-9.\-]{0,5})\s+"
    r"(?P<month>\d{2})/(?P<day>\d{2})/(?P<year>\d{2})\s+"
    r"(?P<strike>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


@dataclass
class StatementRow:
    """One parsed statement line, before it becomes an execution or cash row."""

    sequence: int
    transaction_date: date
    settlement_date: date | None
    action: str
    symbol: str
    description: str
    quantity: float
    price: float
    gross_amount: float | None
    commission: float
    net_amount: float | None
    currency: str
    account_number: str
    activity_type: str
    account_type: str


@dataclass
class StatementParse:
    """Everything one statement file yielded, including what it could not."""

    executions: list[NormalizedExecution] = field(default_factory=list)
    cash: list[dict[str, Any]] = field(default_factory=list)
    accounts: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    trade_days: set[tuple[str, date]] = field(default_factory=set)
    cash_days: set[tuple[str, date]] = field(default_factory=set)

    @property
    def date_range(self) -> tuple[date | None, date | None]:
        days = [day for _, day in (self.trade_days | self.cash_days)]
        return (min(days), max(days)) if days else (None, None)


# -- reading the file --------------------------------------------------------


def _normalize_heading(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def read_xlsx_table(path: Path) -> list[dict[str, str]]:
    """The first worksheet of an xlsx as a list of heading->value dicts.

    Deliberately narrow: one sheet, a heading row, string cells. That is what a
    broker activity export is, and a general-purpose reader would be a much
    larger surface to get subtly wrong. Both inline strings and the shared
    string table are handled, because which one a writer emits is its own
    choice - Questrade's export uses inline ``t="str"`` values and has no
    ``sharedStrings.xml`` at all, but Excel re-saving the same file adds one.
    """
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(Path(path)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.iter(f"{namespace}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{namespace}t")))
        sheet_names = [
            name
            for name in archive.namelist()
            if name.startswith("xl/worksheets/") and name.endswith(".xml")
        ]
        if not sheet_names:
            raise ValueError("no worksheet found in workbook")
        root = ElementTree.fromstring(archive.read(sorted(sheet_names)[0]))

    rows: list[dict[str, str]] = []
    headings: dict[str, str] = {}
    for row_node in root.iter(f"{namespace}row"):
        cells: dict[str, str] = {}
        for cell in row_node.iter(f"{namespace}c"):
            reference = cell.get("r") or ""
            column = re.match(r"[A-Z]+", reference)
            if column is None:
                continue
            value_node = cell.find(f"{namespace}v")
            text = value_node.text if value_node is not None else None
            if text is None:
                inline = cell.find(f"{namespace}is")
                text = (
                    "".join(node.text or "" for node in inline.iter(f"{namespace}t"))
                    if inline is not None
                    else ""
                )
            elif cell.get("t") == "s":
                try:
                    text = shared[int(text)]
                except (ValueError, IndexError):
                    text = ""
            cells[column.group(0)] = text or ""
        if not headings:
            headings = {column: value for column, value in cells.items() if value.strip()}
            continue
        if not any(value.strip() for value in cells.values()):
            continue
        rows.append({headings[column]: value for column, value in cells.items() if column in headings})
    return rows


def read_csv_table(path: Path) -> list[dict[str, str]]:
    """The same shape as :func:`read_xlsx_table`, for a CSV export."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return [
            {str(key or ""): str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_statement_table(path: Path) -> list[dict[str, str]]:
    """Read an activity statement by extension. ``.xlsx``/``.xlsm`` or ``.csv``."""
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return read_xlsx_table(path)
    if suffix in {".csv", ".txt", ".tsv"}:
        return read_csv_table(path)
    raise ValueError(f"unsupported statement file type: {suffix or path}")


# -- parsing one row ---------------------------------------------------------


def _coerce_float(value: Any, default: float = 0.0) -> float:
    text = str(value or "").strip().replace(",", "").replace("$", "")
    if not text:
        return default
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    try:
        number = float(text)
    except ValueError:
        return default
    return -number if negative else number


def _coerce_date(value: Any) -> date | None:
    """The date half of a statement stamp. The clock half is always midnight."""
    text = str(value or "").strip()
    if not text:
        return None
    head = text.split(" ")[0]
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(head, pattern).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def statement_timestamp(day: date) -> str:
    """Midnight MARKET-local for a statement date.

    Not the desk's zone and not UTC: ``journal_trade_shape.is_date_only`` asks
    whether the market-local clock reads 00:00:00, and it is that check which
    stops an imported year being tagged ``premarket``. Attaching the desk's
    Pacific zone here would land at 03:00 ET and defeat it.
    """
    return datetime.combine(day, time(0, 0), tzinfo=MARKET_TZ).isoformat()


def parse_option_description(description: Any) -> dict[str, Any] | None:
    """The real contract behind a Questrade option row, or ``None``.

    Questrade puts an internal id in the Symbol column for options
    (``8SVDLK9``) and the actual contract in the Description. Without this every
    contract would be its own opaque symbol, an expiry and a strike would be
    invisible, and the multiplier would be 1 instead of 100 - which is a P&L
    error of two orders of magnitude, not a cosmetic one.
    """
    match = _OPTION_DESCRIPTION.match(str(description or ""))
    if match is None:
        return None
    parts = match.groupdict()
    try:
        strike = float(parts["strike"])
    except (TypeError, ValueError):
        return None
    return {
        "right": parts["right"].upper(),
        "underlying": parts["underlying"].upper(),
        "expiry": f"{parts['year']}{parts['month']}{parts['day']}",
        "strike": strike,
    }


def parse_rows(table: Iterable[Mapping[str, Any]]) -> list[StatementRow]:
    """Map raw heading->value dicts onto :class:`StatementRow`.

    ``sequence`` is the file's own row order and is carried through to the
    execution uid. It is the only intraday ordering information a statement
    has, so it is preserved rather than discarded and re-guessed.
    """
    parsed: list[StatementRow] = []
    for index, raw in enumerate(table):
        row = {
            _COLUMN_ALIASES[_normalize_heading(key)]: value
            for key, value in raw.items()
            if _normalize_heading(key) in _COLUMN_ALIASES
        }
        transaction_date = _coerce_date(row.get("transaction_date"))
        if transaction_date is None:
            continue
        parsed.append(
            StatementRow(
                sequence=index,
                transaction_date=transaction_date,
                settlement_date=_coerce_date(row.get("settlement_date")),
                action=str(row.get("action") or "").strip().upper(),
                symbol=str(row.get("symbol") or "").strip().upper(),
                description=str(row.get("description") or "").strip(),
                quantity=_coerce_float(row.get("quantity")),
                price=_coerce_float(row.get("price")),
                gross_amount=(
                    _coerce_float(row.get("gross_amount"))
                    if str(row.get("gross_amount") or "").strip()
                    else None
                ),
                commission=_coerce_float(row.get("commission")),
                net_amount=(
                    _coerce_float(row.get("net_amount"))
                    if str(row.get("net_amount") or "").strip()
                    else None
                ),
                currency=str(row.get("currency") or "USD").strip().upper() or "USD",
                account_number=str(row.get("account_number") or "").strip(),
                activity_type=str(row.get("activity_type") or "").strip(),
                account_type=str(row.get("account_type") or "").strip(),
            )
        )
    return parsed


def _execution_from_row(row: StatementRow) -> NormalizedExecution | None:
    """One trade row as a normalized execution, or ``None`` if it is not one."""
    side = normalize_side(row.action)
    if side not in {"BUY", "SELL"} or not row.quantity:
        return None

    option = parse_option_description(row.description)
    if option is not None:
        security_type = "OPT"
        symbol = canonical_option_symbol(
            "",
            security_type,
            underlying=option["underlying"],
            expiry=option["expiry"],
            strike=option["strike"],
            right=option["right"],
        )
    else:
        security_type = normalize_security_type("STK")
        symbol = row.symbol
    if not symbol:
        return None

    timestamp = statement_timestamp(row.transaction_date)
    payload = {
        "source": STATEMENT_SOURCE,
        "statement_sequence": row.sequence,
        "action": row.action,
        "questrade_symbol": row.symbol,
        "description": row.description,
        "settlement_date": row.settlement_date.isoformat() if row.settlement_date else "",
        "gross_amount": row.gross_amount,
        "commission": row.commission,
        "net_amount": row.net_amount,
        "account_type": row.account_type,
        # Stated in the row itself so a later reader does not have to know this
        # module's conventions to understand what the timestamp is worth.
        "time_of_day_known": False,
        "fills_may_be_aggregated": True,
    }
    if option is not None:
        payload["option"] = option
        payload["multiplier"] = 100.0
    return NormalizedExecution(
        execution_uid=stable_execution_uid(
            "QUESTRADE",
            row.account_number,
            "",
            STATEMENT_SOURCE,
            row.account_number,
            row.transaction_date.isoformat(),
            symbol,
            side,
            f"{abs(row.quantity):.5f}",
            f"{row.price:.8f}",
            f"{row.commission:.2f}",
            # The file's own row order. Two identical fills on one day are two
            # rows in the statement and must stay two executions; without this
            # they would hash to one uid and half the position would vanish.
            row.sequence,
        ),
        source=STATEMENT_SOURCE,
        broker="QUESTRADE",
        account_number=row.account_number,
        account_label=row.account_type or row.account_number,
        account_type=row.account_type,
        symbol=symbol,
        security_type=security_type,
        currency=row.currency,
        side=side,
        quantity=abs(row.quantity),
        price=row.price,
        timestamp=timestamp,
        trade_date=row.transaction_date.isoformat(),
        # A statement gives ONE cost column, and it is complete: across the
        # first real file, Net == Gross + Commission on every one of 884 trade
        # rows. Splitting it into a guessed commission and a guessed fee would
        # invent a breakdown the file does not contain; the total is what a tax
        # return needs and the total is exact.
        commission=abs(row.commission),
        fees=0.0,
        gross_amount=row.gross_amount,
        net_amount=row.net_amount,
        order_id="",
        exchange_exec_id="",
        raw_json=json.dumps(payload, sort_keys=True, default=str),
    )


def _cash_from_row(row: StatementRow) -> dict[str, Any] | None:
    """One non-trade row as a ``cash_transactions`` row, or ``None`` to skip."""
    if not row.account_number:
        return None
    amount = row.net_amount if row.net_amount is not None else row.commission
    activity_type = classify_activity_type(row.activity_type)
    return {
        "txn_uid": _cash_txn_uid(
            "QUESTRADE",
            row.account_number,
            row.transaction_date.isoformat(),
            activity_type,
            row.symbol,
            amount,
            row.description,
            # Same reasoning as the execution uid: two identical fee rows on
            # one day are two real charges.
            row.sequence,
        ),
        "broker": "QUESTRADE",
        "account_number": row.account_number,
        "txn_date": row.transaction_date.isoformat(),
        "activity_type": activity_type,
        "description": row.description,
        "symbol": row.symbol,
        "amount": amount,
        "currency": row.currency,
        "raw_json": json.dumps(
            {
                "source": STATEMENT_SOURCE,
                "statement_sequence": row.sequence,
                "action": row.action,
                "activity_type": row.activity_type,
                "account_type": row.account_type,
            },
            sort_keys=True,
            default=str,
        ),
    }


def parse_statement(table: Iterable[Mapping[str, Any]]) -> StatementParse:
    """Everything a statement file yields, with what it could not read listed."""
    result = StatementParse()
    accounts: dict[str, dict[str, Any]] = {}
    for row in parse_rows(table):
        if row.account_number:
            account = accounts.setdefault(
                row.account_number,
                {"number": row.account_number, "type": row.account_type, "label": row.account_type},
            )
            if not account.get("type") and row.account_type:
                account["type"] = row.account_type
                account["label"] = row.account_type

        if str(row.activity_type or "").strip().upper() in TRADE_ACTIVITY_TYPES:
            execution = _execution_from_row(row)
            if execution is None:
                result.skipped.append(
                    {
                        "sequence": row.sequence,
                        "reason": "trade row is not a readable buy or sell",
                        "action": row.action,
                        "symbol": row.symbol,
                        "date": row.transaction_date.isoformat(),
                    }
                )
                continue
            result.executions.append(execution)
            result.trade_days.add((row.account_number, row.transaction_date))
            continue

        cash = _cash_from_row(row)
        if cash is None:
            result.skipped.append(
                {
                    "sequence": row.sequence,
                    "reason": "cash row has no account number",
                    "date": row.transaction_date.isoformat(),
                }
            )
            continue
        result.cash.append(cash)
        result.cash_days.add((row.account_number, row.transaction_date))

    result.accounts = list(accounts.values())
    return result


def tax_status_for_account_type(account_type: Any) -> str:
    """The tax vocabulary for a Questrade account-type wording, or ``""``."""
    text = re.sub(r"\s+", " ", str(account_type or "").strip().upper())
    return TAX_STATUS_BY_ACCOUNT_TYPE.get(text, "")


# -- applying it to the store ------------------------------------------------


def days_covered_by_richer_sources(store: Any, account_numbers: Sequence[str]) -> set[tuple[str, date]]:
    """(account, day) pairs a statement must not touch.

    A day already holding an API, Flex, socket, manual or CSV execution has
    real ids, real times and unaggregated fills. Statement rows for that day
    would be the same money under different identities, and the ``execution_uid``
    upsert cannot see that they are duplicates - so the day is refused here
    instead.
    """
    if not account_numbers:
        return set()
    placeholders = ",".join("?" for _ in account_numbers)
    source_placeholders = ",".join("?" for _ in sorted(RICHER_SOURCES))
    with store.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT account_number, trade_date
            FROM raw_executions
            WHERE broker = 'QUESTRADE'
              AND account_number IN ({placeholders})
              AND source IN ({source_placeholders})
            """,
            [*account_numbers, *sorted(RICHER_SOURCES)],
        ).fetchall()
    covered: set[tuple[str, date]] = set()
    for account_number, trade_date in rows:
        day = _coerce_date(trade_date)
        if day is not None:
            covered.add((str(account_number), day))
    return covered


def import_questrade_statement(
    store: Any,
    path: Path,
    *,
    rebuild: bool = True,
    mark_coverage: bool = True,
) -> dict[str, Any]:
    """Read a statement file and apply everything it may safely write.

    Returns a summary the Health tab renders and the import run records:
    counts written, days skipped because a richer source owns them, rows the
    parser could not read, and the span the file covered.

    Coverage is marked COVERED only for days this import actually wrote, and
    only when the day carried trades. A statement listing no trades on a day is
    not evidence that none happened - it may simply be a statement for another
    account - so a quiet day is left alone rather than being claimed.
    """
    table = read_statement_table(Path(path))
    parse = parse_statement(table)
    account_numbers = [str(account["number"]) for account in parse.accounts if account.get("number")]

    start, end = parse.date_range
    run_id = store.start_import_run(
        "QUESTRADE_STATEMENT",
        account_number=",".join(sorted(account_numbers)),
        trigger="trader_file_import",
        coverage_start=start.isoformat() if start else "",
        coverage_end=end.isoformat() if end else "",
    )

    try:
        if parse.accounts:
            store.upsert_accounts(
                "QUESTRADE",
                [
                    {
                        "number": account["number"],
                        "type": account.get("type") or "",
                        "label": account.get("label") or account["number"],
                    }
                    for account in parse.accounts
                ],
            )
            # Seed the tax status only where the trader has not set one. I6:
            # the label is trader-owned and a guessed tax status is a wrong
            # number in a tax record, so this fills a blank and never argues.
            existing = {
                str(row.get("account_number") or ""): str(row.get("tax_status") or "")
                for row in store.list_accounts()
                if str(row.get("broker") or "").upper() == "QUESTRADE"
            }
            for account in parse.accounts:
                number = str(account["number"])
                if existing.get(number):
                    continue
                status = tax_status_for_account_type(account.get("type"))
                if status:
                    store.set_account_tax_status(
                        "QUESTRADE", number, status, source="statement_import"
                    )

        blocked = days_covered_by_richer_sources(store, account_numbers)
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
        if rebuild and written_executions:
            store.rebuild_trades()

        written_days = sorted(
            {
                (execution.account_number, _coerce_date(execution.trade_date))
                for execution in executions
            }
            - {(None, None)},
            key=lambda item: (item[0], item[1]),
        )
        if mark_coverage:
            from journal_coverage import COVERED
            from journal_coverage import mark_coverage as mark

            for account_number, day in written_days:
                if day is None:
                    continue
                mark(
                    store,
                    broker="QUESTRADE",
                    account_number=account_number,
                    day=day,
                    status=COVERED,
                    source=STATEMENT_SOURCE,
                    import_run_id=run_id,
                    message=f"statement import: {Path(path).name}",
                )

        summary = {
            "file": Path(path).name,
            "rows_read": len(table),
            "executions_written": int(written_executions),
            "cash_written": int(written_cash),
            "days_written": len(written_days),
            "days_skipped_richer_source": len(skipped_days),
            "skipped_days": [(account, day.isoformat()) for account, day in skipped_days],
            "unreadable_rows": len(parse.skipped),
            "accounts": sorted(account_numbers),
            "coverage_start": start.isoformat() if start else "",
            "coverage_end": end.isoformat() if end else "",
        }
        store.finish_import_run(
            run_id,
            status="OK",
            imported_executions=int(written_executions),
            message=describe_summary(summary),
        )
        return summary
    except Exception as exc:  # noqa: BLE001 - the run must record its own failure
        store.finish_import_run(
            run_id, status="FAILED", imported_executions=0, message=f"{type(exc).__name__}: {exc}"
        )
        raise


def describe_summary(summary: Mapping[str, Any]) -> str:
    """One line for the import-run ledger and the Health tab status."""
    parts = [
        f"{summary.get('executions_written', 0)} executions",
        f"{summary.get('cash_written', 0)} cash rows",
        f"{summary.get('days_written', 0)} days",
    ]
    skipped = int(summary.get("days_skipped_richer_source") or 0)
    if skipped:
        parts.append(f"{skipped} day(s) left to the API")
    unreadable = int(summary.get("unreadable_rows") or 0)
    if unreadable:
        parts.append(f"{unreadable} row(s) unreadable")
    span_start = summary.get("coverage_start") or ""
    span_end = summary.get("coverage_end") or ""
    span = f" {span_start}..{span_end}" if span_start else ""
    return f"{summary.get('file', 'statement')}{span}: " + ", ".join(parts)


__all__ = [
    "RICHER_SOURCES",
    "STATEMENT_SOURCE",
    "StatementParse",
    "StatementRow",
    "days_covered_by_richer_sources",
    "describe_summary",
    "import_questrade_statement",
    "parse_option_description",
    "parse_rows",
    "parse_statement",
    "read_csv_table",
    "read_statement_table",
    "read_xlsx_table",
    "statement_timestamp",
    "tax_status_for_account_type",
]
