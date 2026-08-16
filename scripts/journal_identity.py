"""What makes two broker fills the same instrument, the same execution, or neither.

R7 §9 step 3, root cause B3. Extracted from ``journal_migrate`` because the
vocabulary is not a migration concern: the importers, the assembler and the
migration all have to agree on it, and a module named "migrate" owning the live
import vocabulary would mislead the next reader.

THE DEFECT THIS EXISTS TO FIX

``rebuild_trades`` groups executions by ``(broker, account, symbol,
security_type, currency)``, and nothing normalized ``security_type``. Three
independent spellings reached that key:

* Questrade fell back to ``listingExchange`` when it had no ``securityType``
  (``journal_importers.py:305``), so one AMZN position could be half ``STOCK``
  and half ``NASDAQ`` - two groups that can never net against each other, and a
  pair of trades that stay open forever;
* the IBKR socket spelled the type from ``contract.secType`` (``STK``) while
  Flex spelled it from ``assetCategory`` (also ``STK``, but ``OPT`` vs
  ``Option`` and ``FUT`` vs ``Future`` diverge);
* Questrade says ``Stock``/``Option``, IBKR says ``STK``/``OPT``.

Normalizing at import time alone would fix nothing for data already in the
journal, so the same function runs on the *stored* row when the group key is
built. That is what makes the trader's existing stuck-open pairs net.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

#: The canonical vocabulary. Everything a group key sees is one of these.
CANONICAL_SECURITY_TYPES = frozenset(
    {"STK", "OPT", "FUT", "FOP", "CASH", "BOND", "FUND", "CFD", "WAR", "IND", "CRYPTO", "UNKNOWN"}
)

#: Broker spellings -> canonical. Both brokers' vocabularies, in one place.
SECURITY_TYPE_ALIASES: dict[str, str] = {
    # Equity
    "STK": "STK",
    "STOCK": "STK",
    "STOCKS": "STK",
    "COMMONSTOCK": "STK",
    "COMMON": "STK",
    "EQUITY": "STK",
    "EQUITIES": "STK",
    "ETF": "STK",
    "ADR": "STK",
    "REIT": "STK",
    # Options
    "OPT": "OPT",
    "OPTION": "OPT",
    "OPTIONS": "OPT",
    "EQUITYOPTION": "OPT",
    "INDEXOPTION": "OPT",
    # Futures and futures options
    "FUT": "FUT",
    "FUTURE": "FUT",
    "FUTURES": "FUT",
    "FOP": "FOP",
    "FUTURESOPTION": "FOP",
    # Cash / FX
    "CASH": "CASH",
    "FOREX": "CASH",
    "FX": "CASH",
    "CURRENCY": "CASH",
    # Debt
    "BOND": "BOND",
    "BND": "BOND",
    "BILL": "BOND",
    "DEBT": "BOND",
    # Funds
    "FUND": "FUND",
    "MUTUALFUND": "FUND",
    "MF": "FUND",
    # Other IB categories
    "CFD": "CFD",
    "WAR": "WAR",
    "WARRANT": "WAR",
    "IND": "IND",
    "INDEX": "IND",
    "CRYPTO": "CRYPTO",
    "CRYPTOCURRENCY": "CRYPTO",
}

#: Listing exchanges that reached ``security_type`` only through Questrade's
#: ``listingExchange`` fallback. A value from this set never described an
#: instrument type; it described where the instrument trades, and the row it
#: came from is an equity - Questrade sends a real ``securityType`` for options
#: and futures, so the fallback only ever fired for plain stock.
#:
#: This is the entry that actually reunites the trader's split positions, so it
#: is an explicit auditable list rather than a "does it look like an exchange?"
#: heuristic.
LISTING_EXCHANGES = frozenset(
    {
        "NASDAQ", "NASD", "NYSE", "NYSEAMERICAN", "AMEX", "ARCA", "NYSEARCA", "BATS", "IEX",
        "EDGA", "EDGX", "MEMX", "PHLX", "ISE", "BOX", "NYSENAT", "PSX", "CBOE", "CBOE2",
        "TSX", "TSXV", "CSE", "NEO", "AEQUITAS", "CNSX", "ALPHA", "OMEGA", "PURE", "CHIX",
        "OTC", "OTCBB", "PINK", "OTCMKTS", "GREY",
        "LSE", "ASX", "TSE", "HKEX", "SEHK",
    }
)


def normalize_security_type(value: Any) -> str:
    """Map a broker's security-type spelling onto the canonical vocabulary.

    An unrecognized value is **kept as it was**, uppercased, rather than folded
    into ``UNKNOWN``. Folding would merge two genuinely different instruments
    that happen to share a symbol, and a wrong merge produces a wrong P&L
    silently; leaving them apart produces a visible stuck-open pair the trader
    can fix with a ``REASSIGN_GROUP`` adjustment. Between a silent wrong number
    and a visible wrong shape, this system takes the visible one.
    """
    text = str(value or "").strip().upper()
    if not text:
        return "UNKNOWN"
    compact = text.replace(" ", "").replace("_", "").replace("-", "")
    if compact in SECURITY_TYPE_ALIASES:
        return SECURITY_TYPE_ALIASES[compact]
    if compact in LISTING_EXCHANGES:
        return "STK"
    return text


def canonical_option_symbol(
    symbol: Any,
    security_type: Any = "",
    *,
    underlying: Any = "",
    expiry: Any = "",
    strike: Any = None,
    right: Any = "",
) -> str:
    """One compact OCC-style identity for socket, Flex and reconciliation rows."""
    text = str(symbol or "").strip().upper()
    compact = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"([A-Z0-9.]{1,6})(\d{6})([CP])(\d{8})", compact)
    if match:
        return "".join(match.groups())
    if normalize_security_type(security_type) not in {"OPT", "FOP", "WAR"}:
        return text

    root = re.sub(r"\s+", "", str(underlying or text).strip().upper())
    expiry_digits = re.sub(r"\D", "", str(expiry or ""))
    if len(expiry_digits) == 8:
        expiry_digits = expiry_digits[2:]
    right_text = str(right or "").strip().upper()
    option_right = {"CALL": "C", "PUT": "P"}.get(right_text, right_text[:1])
    try:
        strike_code = f"{int(round(float(strike) * 1000)):08d}"
    except (TypeError, ValueError):
        strike_code = ""
    if root and len(expiry_digits) == 6 and option_right in {"C", "P"} and strike_code:
        return f"{root}{expiry_digits}{option_right}{strike_code}"
    return compact


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """The identity of the position an execution belongs to.

    Normalizing here rather than only at import is the point: the trader's
    existing rows were written with the un-normalized spellings, and a fix that
    only applied to future imports would leave every already-stuck position
    stuck.
    """
    return (
        str(row.get("broker") or "").strip().upper(),
        str(row.get("account_number") or "").strip(),
        canonical_option_symbol(row.get("symbol"), row.get("security_type")),
        normalize_security_type(row.get("security_type")),
        str(row.get("currency") or "").strip().upper(),
    )


def group_key_text(key: tuple[str, str, str, str, str]) -> str:
    """A group key as one string, for adjustments that target a whole position.

    ``trade_adjustments.target_uid`` holds an execution uid for execution-scoped
    actions and this text for ``TRADE_GROUP`` ones. Pipe-separated because none
    of the five fields can contain a pipe: broker, account number, symbol and
    currency are broker identifiers, and the security type is drawn from the
    canonical vocabulary above.
    """
    return "|".join(str(part or "") for part in key)


def stable_execution_uid(prefix: str, account_number: str, exec_id: Any, *fallback_parts: Any) -> str:
    """``PREFIX:account:exec_id``, with a deterministic surrogate when there is no exec id.

    Dropping the symbol and timestamp from the uid (§9 step 2) removed the
    accidental uniqueness they used to provide, so a broker row with no
    execution id can no longer be allowed a random uuid - it would re-import as
    a new execution every night and double the position by a different route
    than B4 did. The surrogate hashes the fields that identify the fill instead.
    """
    prefix = str(prefix or "").strip().upper()
    account = str(account_number or "").strip()
    cleaned_id = str(exec_id or "").strip()
    if cleaned_id:
        return f"{prefix}:{account}:{cleaned_id}"
    blob = "|".join(str(part or "") for part in fallback_parts)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{account}:auto-{digest}"


def classify_execution_source(row: dict[str, Any]) -> str:
    """Which importer wrote this row, read from the shape of its raw payload.

    Nothing recorded the source before v3, so it has to be inferred - and the
    collapse rule depends on telling a Flex row from a socket row. The shapes
    are distinct enough to be safe: the socket importer writes a nested
    ``{"contract": ..., "execution": ...}`` object, Flex writes the statement's
    flat attributes, and Questrade writes the API's own JSON.
    """
    broker = str(row.get("broker") or "").upper()
    raw = _raw_payload(row)
    if broker == "IBKR":
        if "contract" in raw and "execution" in raw:
            return "IBKR_SOCKET"
        if raw.keys() & {"ibExecID", "tradePrice", "assetCategory", "accountId", "tradeID"}:
            return "IBKR_FLEX"
        return ""
    if broker == "QUESTRADE":
        return "QT_API"
    if broker == "MANUAL":
        return "MANUAL"
    return ""


def contract_multiplier(row: dict[str, Any]) -> float:
    """The contract multiplier, from the raw payload or the security type."""
    raw = _raw_payload(row)
    candidates = [raw.get("multiplier")]
    contract = raw.get("contract")
    if isinstance(contract, dict):
        candidates.append(contract.get("multiplier"))
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    if normalize_security_type(row.get("security_type")) in {"OPT", "FOP"}:
        return 100.0
    return 1.0


def _raw_payload(row: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = json.loads(row.get("raw_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}
