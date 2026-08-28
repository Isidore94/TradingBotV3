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
  day carry nothing that says which came first: the file lists a same-day round
  trip SELL-first 227 times out of 227, which makes that a SORT and not a
  sequence. Left to the assembler's tiebreak - the execution uid, a hash - the
  direction of a same-day round trip was a coin flip, and 86 of 199 came out
  SHORT at random.
  **The description settles it instead.** Questrade writes "... COMMON STOCK
  SHORT." on a short sale and "... COVER SHORT." on the buy that closes one, so
  :func:`leg_rank` orders each row by what it does to the position rather than
  by where it sat in the file. That resolved all 227 - 169 long, 58 short - and
  every one of the 58 carried BOTH markings, so the two halves agree with each
  other rather than being read off one row.
  The money was never at risk either way: a symbol that starts and ends a day
  flat realises the same total P&L whichever way the legs are paired, because
  every leg is closed either way.
* **Nothing positional may reach an execution's identity.** The statement's row
  order is kept in ``raw_json`` as a record of what the file said, and is
  deliberately NOT part of the uid: a January-to-December export lists the same
  January trades as a January-to-August one did, at different row positions.
  When the uid carried the row index, a one-row shift made **884 of 884** real
  trades look new, which is precisely how layering later exports onto earlier
  ones would have doubled the year. Identity is
  :func:`fill_signature` plus an ordinal counted within that signature.

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
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
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


#: Questrade names a short in the Description, and this is the whole reason the
#: long/short label survives an import. "SIMPLY GOOD FOODS COMPANY (THE) COMMON
#: STOCK SHORT. WE ACTED AS AGENT" is a short sale; "VF CORPORATION COVER SHORT.
#: WE ACTED AS AGENT" is the buy that closes one. ``COVER`` is tested first
#: because a cover line contains the word SHORT too.
_COVER_MARKING = re.compile(r"\bCOVER\s+SHORT\b", re.IGNORECASE)
_SHORT_MARKING = re.compile(r"\bSHORT\b\.?(?:\s|$)", re.IGNORECASE)

#: Rank inside one symbol-day: legs that OPEN a position sort before legs that
#: CLOSE one. This is what makes a same-day round trip come out with the right
#: direction, and it is derived from each row on its own rather than from the
#: file, so it is identical in every export that lists the trade.
OPENS_POSITION = 0
CLOSES_POSITION = 1


def short_marking(description: Any) -> str:
    """``"COVER"``, ``"SHORT"`` or ``""`` for one statement description."""
    text = str(description or "")
    if _COVER_MARKING.search(text):
        return "COVER"
    if _SHORT_MARKING.search(text):
        return "SHORT"
    return ""


def leg_rank(side: str, marking: str) -> int:
    """Does this row open a position or close one?

    A statement carries no time of day and lists every same-day round trip
    SELL-first - 227 of 227 in the first real file, which makes that a SORT and
    not a sequence, so the row order says nothing about what happened first.
    Left to the assembler's tiebreak (a uid hash) the direction of a same-day
    round trip was a coin flip: 86 of 199 came out SHORT at random.

    The description settles it. A SELL marked SHORT opened a short; a BUY marked
    COVER SHORT closed one; an unmarked BUY opened a long and an unmarked SELL
    closed one. On the first real file this resolved every one of the 227
    round trips - 169 long, 58 short - and all 58 carried BOTH markings, so the
    two halves agree with each other rather than being read from one row.
    """
    if side == "SELL":
        return OPENS_POSITION if marking == "SHORT" else CLOSES_POSITION
    return CLOSES_POSITION if marking == "COVER" else OPENS_POSITION


def fill_signature(row: StatementRow, symbol: str, side: str) -> tuple[Any, ...]:
    """What makes two statement lines the SAME fill, across different files.

    Everything the statement knows about the fill and nothing about where it sat
    in the file. This is the load-bearing part of layering a later export on an
    earlier one: a January-to-December file lists the same January trades as a
    January-to-August file did, but at different row positions, so anything
    positional in the identity re-imports the whole year as new executions and
    doubles it. Measured on the real file before the fix: a one-row shift made
    884 of 884 trades look new.
    """
    return (
        row.account_number,
        row.transaction_date.isoformat(),
        symbol,
        side,
        f"{abs(row.quantity):.5f}",
        f"{row.price:.8f}",
        f"{row.commission:.2f}",
        row.currency,
    )


def _execution_from_row(row: StatementRow, *, ordinal: int = 0) -> NormalizedExecution | None:
    """One trade row as a normalized execution, or ``None`` if it is not one.

    ``ordinal`` distinguishes genuinely repeated fills - the Nth line in this
    file that is identical to another in every field the statement carries. It
    counts within the fill signature rather than across the file, so it is the
    same number in every export that contains this trade.
    """
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

    signature = fill_signature(row, symbol, side)
    timestamp = statement_timestamp(row.transaction_date)
    marking = short_marking(row.description)
    rank = leg_rank(side, marking)
    payload = {
        "source": STATEMENT_SOURCE,
        "statement_sequence": row.sequence,
        "short_marking": marking,
        "opens_position": rank == OPENS_POSITION,
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
    # The uid doubles as the assembler's intra-day tiebreak. Every statement row
    # on one date shares a timestamp (midnight - a statement has no clock), and
    # `_execution_assembly_sort_key` breaks that tie on the execution uid, so
    # the uid is where the leg order has to live. `rank` first means a position
    # is opened before it is closed; the digest is over the fill's own fields,
    # so the id stays stable across exports while still sorting correctly.
    digest = hashlib.sha256(
        "|".join(str(part) for part in signature).encode("utf-8")
    ).hexdigest()[:16]
    return NormalizedExecution(
        execution_uid=stable_execution_uid(
            "QUESTRADE",
            row.account_number,
            # The Nth identical fill on this day, NOT the row's position in the
            # file. Two identical fills are two real executions and must stay
            # two, or half the position vanishes into one uid; but the count
            # has to be the same in every export that lists this trade, or
            # layering a longer file re-imports the whole overlap.
            f"stmt-{rank}{ordinal:03d}-{digest}",
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


def _cash_from_row(row: StatementRow, *, ordinal: int = 0) -> dict[str, Any] | None:
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
            # one day are two real charges, and the count must not depend on
            # where they sat in the file.
            ordinal,
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
    seen_fills: Counter[tuple[Any, ...]] = Counter()
    seen_cash: Counter[tuple[Any, ...]] = Counter()
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
            probe = _execution_from_row(row)
            if probe is not None:
                key = fill_signature(row, probe.symbol, probe.side)
                seen_fills[key] += 1
                execution = _execution_from_row(row, ordinal=seen_fills[key] - 1)
            else:
                execution = None
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

        cash_signature = (
            row.account_number,
            row.transaction_date.isoformat(),
            row.activity_type,
            row.symbol,
            row.net_amount,
            row.description,
        )
        seen_cash[cash_signature] += 1
        cash = _cash_from_row(row, ordinal=seen_cash[cash_signature] - 1)
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


# -- checking the import against the file it came from -----------------------

#: A per-symbol difference at or below this is arithmetic, not a defect.
#: Questrade books ``Gross Amount`` rounded to the cent while ``rebuild_trades``
#: recomputes price x quantity at full precision, so the two disagree by
#: fractions of a cent per fill. Measured on the first real file: worst symbol
#: 1.17c over 253 closed symbols, total drift $0.42, NET $-0.16 on $4,014.18.
ROUNDING_TOLERANCE = 0.02


def reconcile_statement(store: Any, path: Path) -> dict[str, Any]:
    """Add the statement up by hand and check the journal against it.

    Two independent routes to the same number. The **statement** side is plain
    arithmetic on the file: for a symbol whose quantities net to zero across the
    file, the sum of its Net Amount column IS the realised P&L, because every
    share bought was sold. The **journal** side is what ``rebuild_trades``
    assembled - average-cost matching, leg pairing, multipliers, corrections.
    They share only the file, so a disagreement is an assembly defect, and the
    per-symbol rows say which symbol to look at.

    What this DOES prove: that the position walk, the fill matching, the option
    multipliers and the commission handling turned the file into the right
    money. What it does NOT prove: that the file was parsed correctly in the
    first place - both sides read the same parse. Only the trader's own
    Questrade year-end numbers can close that, which is what makes this a
    demonstration rather than a proof.

    Symbols still holding a position are **excluded, not zeroed**: cash has left
    the account with no realised P&L against it yet, so including them would
    show a loss that is really an open trade. They are counted and their cash is
    reported separately. A symbol carried in from before the file's window nets
    to a non-zero quantity too, so the same exclusion covers it.

    Nothing here writes. It reads the file and the store and returns numbers.
    """
    parse = parse_statement(read_statement_table(Path(path)))

    statement_pnl: dict[tuple[str, str], float] = defaultdict(float)
    statement_commission: dict[tuple[str, str], float] = defaultdict(float)
    position: dict[tuple[str, str], float] = defaultdict(float)
    currency: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_per_key: dict[tuple[str, str], int] = defaultdict(int)
    round_trip_days: dict[tuple[str, str], set[str]] = defaultdict(set)
    sides_by_day: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    markings_by_day: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    short_rows = 0

    for execution in parse.executions:
        key = (execution.account_number, execution.symbol)
        signed = execution.quantity if execution.side == "BUY" else -execution.quantity
        position[key] += signed
        raw = json.loads(execution.raw_json or "{}")
        net = raw.get("net_amount")
        statement_pnl[key] += float(net) if net is not None else 0.0
        statement_commission[key] += abs(execution.commission)
        currency[key].add(execution.currency)
        rows_per_key[key] += 1
        day_key = (execution.account_number, execution.symbol, execution.trade_date)
        sides_by_day[day_key].add(execution.side)
        marking = str(raw.get("short_marking") or "")
        markings_by_day[day_key].add(marking)
        if marking:
            short_rows += 1

    # A same-day round trip is a short exactly when its description says so, so
    # direction is read rather than guessed. What is still worth surfacing is a
    # day where the SAME symbol carried both a marked short round trip and an
    # unmarked long one - a real and legal thing to do, and 3 days of 439 on the
    # trader's own history. The assembler groups a symbol into ONE position, so
    # on those days it blends what were really two trades: the day's money is
    # still exact (everything closed), but the split between the long and the
    # short is not. Naming them lets the trader split them by hand if they care.
    mixed_direction_days: list[dict[str, Any]] = []
    for (account_number, symbol, day), sides in sides_by_day.items():
        if len(sides) < 2:
            continue
        round_trip_days[(account_number, symbol)].add(day)
        markings = markings_by_day[(account_number, symbol, day)]
        if markings & {"SHORT", "COVER"} and "" in markings:
            mixed_direction_days.append(
                {"account": account_number, "symbol": symbol, "date": day}
            )

    journal_pnl: dict[tuple[str, str], float] = defaultdict(float)
    journal_commission: dict[tuple[str, str], float] = defaultdict(float)
    journal_trades: dict[tuple[str, str], int] = defaultdict(int)
    needs_review: dict[tuple[str, str], int] = defaultdict(int)
    for trade in store.list_trades(broker="QUESTRADE"):
        key = (str(trade.get("account_number") or ""), str(trade.get("symbol") or ""))
        journal_pnl[key] += float(trade.get("net_pnl") or 0.0)
        journal_commission[key] += float(trade.get("commission") or 0.0) + float(
            trade.get("fees") or 0.0
        )
        journal_trades[key] += 1
        if str(trade.get("reconcile_status") or "").upper() == "NEEDS_REVIEW":
            needs_review[key] += 1

    closed = {key for key, quantity in position.items() if abs(quantity) < 1e-9}
    open_keys = set(position) - closed

    symbols: list[dict[str, Any]] = []
    for key in sorted(closed):
        account_number, symbol = key
        difference = statement_pnl[key] - journal_pnl.get(key, 0.0)
        symbols.append(
            {
                "account": account_number,
                "symbol": symbol,
                "currency": "/".join(sorted(currency[key])),
                "statement_rows": rows_per_key[key],
                "statement_pnl": round(statement_pnl[key], 4),
                "journal_pnl": round(journal_pnl.get(key, 0.0), 4),
                "difference": round(difference, 4),
                "statement_commission": round(statement_commission[key], 4),
                "journal_commission": round(journal_commission.get(key, 0.0), 4),
                "journal_trades": journal_trades.get(key, 0),
                "needs_review": needs_review.get(key, 0),
                "round_trip_days": sorted(round_trip_days.get(key, ())),
                "beyond_rounding": abs(difference) > ROUNDING_TOLERANCE,
            }
        )
    symbols.sort(key=lambda row: (-abs(row["difference"]), row["account"], row["symbol"]))

    by_account: dict[str, dict[str, Any]] = {}
    for account_number in sorted({key[0] for key in position}):
        account_closed = [key for key in closed if key[0] == account_number]
        account_open = [key for key in open_keys if key[0] == account_number]
        by_account[account_number] = {
            "closed_symbols": len(account_closed),
            "open_symbols": len(account_open),
            "statement_pnl": round(sum(statement_pnl[key] for key in account_closed), 4),
            "journal_pnl": round(sum(journal_pnl.get(key, 0.0) for key in account_closed), 4),
            "difference": round(
                sum(statement_pnl[key] - journal_pnl.get(key, 0.0) for key in account_closed), 4
            ),
            "statement_commission": round(
                sum(statement_commission[key] for key in account_closed + account_open), 4
            ),
            "journal_commission": round(
                sum(journal_commission.get(key, 0.0) for key in account_closed + account_open), 4
            ),
            "open_cash": round(sum(statement_pnl[key] for key in account_open), 4),
            "currencies": sorted({value for key in account_closed for value in currency[key]}),
        }

    start, end = parse.date_range
    return {
        "file": Path(path).name,
        "coverage_start": start.isoformat() if start else "",
        "coverage_end": end.isoformat() if end else "",
        "statement_trade_rows": len(parse.executions),
        "closed_symbols": len(closed),
        "open_symbols": len(open_keys),
        "statement_pnl": round(sum(statement_pnl[key] for key in closed), 4),
        "journal_pnl": round(sum(journal_pnl.get(key, 0.0) for key in closed), 4),
        "difference": round(
            sum(statement_pnl[key] - journal_pnl.get(key, 0.0) for key in closed), 4
        ),
        "statement_commission": round(sum(statement_commission.values()), 4),
        "journal_commission": round(
            sum(journal_commission.get(key, 0.0) for key in position), 4
        ),
        "open_cash": round(sum(statement_pnl[key] for key in open_keys), 4),
        "symbols_beyond_rounding": [row for row in symbols if row["beyond_rounding"]],
        "symbols_missing_from_journal": [
            {"account": key[0], "symbol": key[1]}
            for key in sorted(closed)
            if key not in journal_pnl
        ],
        "needs_review_trades": sum(needs_review.values()),
        "same_day_round_trips": sum(len(days) for days in round_trip_days.values()),
        "short_marked_rows": short_rows,
        "mixed_direction_days": mixed_direction_days,
        "by_account": by_account,
        "symbols": symbols,
        "tolerance": ROUNDING_TOLERANCE,
    }


def reconciliation_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The per-symbol table, flattened for a CSV the trader can open."""
    rows: list[dict[str, Any]] = []
    for row in report.get("symbols") or []:
        flat = dict(row)
        flat["round_trip_days"] = ";".join(flat.get("round_trip_days") or [])
        rows.append(flat)
    return rows


def describe_reconciliation(report: Mapping[str, Any]) -> str:
    """The comparison as a few lines a person can read."""
    lines = [
        f"{report.get('file', 'statement')} "
        f"{report.get('coverage_start', '')}..{report.get('coverage_end', '')}",
        f"{report.get('statement_trade_rows', 0)} trade rows; "
        f"{report.get('closed_symbols', 0)} symbols fully closed, "
        f"{report.get('open_symbols', 0)} still open (excluded).",
        f"Statement adds up to {report.get('statement_pnl', 0.0):,.2f}; "
        f"the journal says {report.get('journal_pnl', 0.0):,.2f}; "
        f"difference {report.get('difference', 0.0):+,.4f}.",
        f"Commission: statement {report.get('statement_commission', 0.0):,.2f}, "
        f"journal {report.get('journal_commission', 0.0):,.2f}.",
    ]
    beyond = report.get("symbols_beyond_rounding") or []
    if beyond:
        worst = ", ".join(f"{row['symbol']} {row['difference']:+.2f}" for row in beyond[:5])
        lines.append(
            f"{len(beyond)} symbol(s) differ by more than "
            f"{report.get('tolerance', 0.02):.2f}: {worst}"
        )
    else:
        lines.append(
            f"Every symbol agrees within {report.get('tolerance', 0.02):.2f} "
            "- the rest is Questrade rounding each row to the cent."
        )
    missing = report.get("symbols_missing_from_journal") or []
    if missing:
        lines.append(f"{len(missing)} closed symbol(s) are in the file but not the journal.")
    if report.get("needs_review_trades"):
        lines.append(
            f"{report['needs_review_trades']} trade(s) are flagged NEEDS_REVIEW "
            "- the journal had to invent an opening fill."
        )
    round_trips = int(report.get("same_day_round_trips") or 0)
    if round_trips:
        lines.append(
            f"{round_trips} same-day round trip(s); long or short is read from the "
            f"description, which marked {report.get('short_marked_rows', 0)} row(s) as "
            "a short sale or a cover."
        )
    mixed = report.get("mixed_direction_days") or []
    if mixed:
        worst = ", ".join(f"{row['symbol']} {row['date']}" for row in mixed[:5])
        lines.append(
            f"{len(mixed)} day(s) held both a short and a long in the same symbol, so "
            f"the journal blends them into one position - the day's money is still "
            f"right, the split between the two is not: {worst}"
        )
    return "\n".join(lines)


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
    "ROUNDING_TOLERANCE",
    "STATEMENT_SOURCE",
    "StatementParse",
    "StatementRow",
    "days_covered_by_richer_sources",
    "describe_reconciliation",
    "fill_signature",
    "leg_rank",
    "describe_summary",
    "import_questrade_statement",
    "parse_option_description",
    "parse_rows",
    "parse_statement",
    "read_csv_table",
    "reconciliation_rows",
    "reconcile_statement",
    "short_marking",
    "read_statement_table",
    "read_xlsx_table",
    "statement_timestamp",
    "tax_status_for_account_type",
]
