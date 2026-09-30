"""Questrade book, read-only: accounts, open positions and cash for the Trade Mentor ``/book``.

Reuses ``journal_importers.QuestradeImporter`` exactly as the night import does: every
request goes through its ``_authorized_get`` (a 401 is answered by its own refresh,
under ``_questrade_refresh_lock()``; this module never refreshes a token itself). Only
three GETs are ever sent: ``v1/accounts``, ``v1/accounts/{id}/positions`` and
``v1/accounts/{id}/balances``. Never an order endpoint. Qt-free, no store writes: the
caller keeps the snapshot, the 15-min freshness and the 1-hour backoff after a failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

#: A snapshot younger than this is "fresh": /book, /check and the 06:20 fetch reuse it.
FRESH_FOR = timedelta(minutes=15)
#: After a failed fetch, no new one for this long (never hammer the token chain).
BACKOFF = timedelta(hours=1)
#: The only paths this module may request (``{n}`` = account number).
ALLOWED_PATHS = ("v1/accounts", "v1/accounts/{n}/positions", "v1/accounts/{n}/balances")


@dataclass(frozen=True)
class BookSnapshot:
    """One point-in-time read of the Questrade book. ``fetched_utc`` is tz-aware UTC ISO."""

    fetched_utc: str
    accounts: tuple[dict[str, Any], ...] = ()
    positions: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {"fetched_utc": self.fetched_utc, "accounts": [dict(a) for a in self.accounts],
                "positions": [dict(p) for p in self.positions], "errors": list(self.errors)}

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookSnapshot":
        return cls(
            fetched_utc=str(data.get("fetched_utc") or ""),
            accounts=tuple(dict(a) for a in data.get("accounts") or () if isinstance(a, Mapping)),
            positions=tuple(dict(p) for p in data.get("positions") or () if isinstance(p, Mapping)),
            errors=tuple(str(e) for e in data.get("errors") or ()),
        )

    @classmethod
    def from_json(cls, text: str | None) -> "BookSnapshot | None":
        if not text:
            return None
        try:
            data = json.loads(text)
        except ValueError:
            return None
        return cls.from_dict(data) if isinstance(data, Mapping) and data.get("fetched_utc") else None

    def age(self, now: datetime) -> timedelta | None:
        try:
            when = datetime.fromisoformat(self.fetched_utc)
        except ValueError:
            return None
        when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        return _aware(now) - when

    def is_fresh(self, now: datetime, fresh_for: timedelta = FRESH_FOR) -> bool:
        age = self.age(now)
        return age is not None and timedelta(0) <= age <= fresh_for


def _aware(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN is unknown


def has_token(importer: Any) -> bool:
    """A refresh token, or an access token with its API server, in local settings."""
    try:
        return bool(importer.refresh_token) or (bool(importer.access_token) and bool(importer.api_server))
    except Exception:  # noqa: BLE001 - unreadable settings = no token
        return False


def account_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    number = str(raw.get("number") or raw.get("accountNumber") or "").strip()
    kind = str(raw.get("type") or raw.get("accountType") or "").strip()
    label = str(raw.get("name") or raw.get("description") or "").strip() or f"{kind or 'Account'} {number}".strip()
    return {"account_number": number, "account_type": kind, "account_label": label,
            "status": str(raw.get("status") or ""), "cash": None, "cash_known": False}


def position_row(account: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """One open position from ``QuestradeImporter.get_positions``; a flat (0 qty) row is None."""
    try:
        payload = json.loads(str(raw.get("raw_json") or "{}"))
    except ValueError:
        payload = {}
    qty = _float(raw.get("quantity"))
    if qty is None:
        qty = _float(payload.get("openQuantity"))
    if not qty:
        return None
    return {
        "account_number": str(account.get("account_number") or raw.get("account_number") or ""),
        "account_label": str(account.get("account_label") or ""),
        "account_type": str(account.get("account_type") or ""),
        "symbol": str(raw.get("symbol") or payload.get("symbol") or "").strip().upper(),
        "security_type": str(raw.get("security_type") or ""),
        "currency": str(raw.get("currency") or ""),
        "open_qty": abs(qty),
        "side": "LONG" if qty > 0 else "SHORT",
        "avg_price": _float(payload.get("averageEntryPrice")),
        "current_price": _float(payload.get("currentPrice")),
        "market_value": _float(payload.get("currentMarketValue")),
    }


def cash_by_currency(payload: Any) -> dict[str, float] | None:
    """``{CUR: cash}`` from a balances payload's ``perCurrencyBalances``; None when absent."""
    rows = payload.get("perCurrencyBalances") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        return None
    out: dict[str, float] = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("currency"):
            value = _float(row.get("cash"))
            if value is not None:
                out[str(row["currency"]).upper()] = value
    return out or None


def fetch_book(now: datetime | None = None, importer: Any = None) -> tuple[BookSnapshot | None, str]:
    """``(snapshot, "")`` or ``(None, reason)``. Never raises. No token -> ``(None, "no token")``.

    The caller stores a failure's time and backs off :data:`BACKOFF` before trying again.
    """
    moment = _aware(now)
    if importer is None:
        try:
            from journal_importers import QuestradeImporter

            importer = QuestradeImporter()
        except Exception as exc:  # noqa: BLE001
            return None, f"importer unavailable ({type(exc).__name__})"
    if not has_token(importer):
        return None, "no token"
    try:
        accounts = [account_row(raw) for raw in importer.get_accounts() or ()]
        accounts = [row for row in accounts if row["account_number"]]
        positions: list[dict[str, Any]] = []
        errors: list[str] = []
        for account in accounts:
            number = account["account_number"]
            for raw in importer.get_positions(number) or ():
                row = position_row(account, raw)
                if row is not None:
                    positions.append(row)
            try:
                cash = cash_by_currency(importer._authorized_get(f"v1/accounts/{number}/balances"))
            except Exception as exc:  # noqa: BLE001 - cash is optional: unknown, never a failed book
                errors.append(f"balances {number}: {type(exc).__name__}")
                cash = None
            account["cash"], account["cash_known"] = cash, cash is not None
    except Exception as exc:  # noqa: BLE001 - one failure = no snapshot and a backoff, never a raise
        reason = f"{type(exc).__name__}: {exc}"[:200]
        logging.warning("Trade Mentor book: Questrade read failed (%s)", reason)
        return None, reason
    stamp = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    logging.info("Trade Mentor book: Questrade read %d account(s), %d position(s)", len(accounts), len(positions))
    return BookSnapshot(stamp, tuple(accounts), tuple(positions), tuple(errors)), ""


def backoff_left(failed_utc: str | None, now: datetime, backoff: timedelta = BACKOFF) -> timedelta | None:
    """Time left in the backoff after a failure at ``failed_utc``; None when not backing off."""
    if not failed_utc:
        return None
    try:
        when = datetime.fromisoformat(str(failed_utc))
    except ValueError:
        return None
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    left = when + backoff - _aware(now)
    return left if left > timedelta(0) else None


def fixture_snapshot(fetched_utc: str = "2026-09-29T14:00:00+00:00",
                     positions: Iterable[Mapping[str, Any]] | None = None) -> BookSnapshot:
    """Two accounts: a TFSA (long only) and a margin account holding a short."""
    accounts = (
        {"account_number": "111", "account_type": "TFSA", "account_label": "TFSA 111", "status": "Active",
         "cash": {"CAD": 2500.0, "USD": 1200.0}, "cash_known": True},
        {"account_number": "222", "account_type": "Margin", "account_label": "Margin 222", "status": "Active",
         "cash": None, "cash_known": False},
    )
    default = (
        {"account_number": "111", "account_label": "TFSA 111", "account_type": "TFSA", "symbol": "NVDA",
         "security_type": "STK", "currency": "USD", "open_qty": 50.0, "side": "LONG", "avg_price": 120.0,
         "current_price": 125.0, "market_value": 6250.0},
        {"account_number": "222", "account_label": "Margin 222", "account_type": "Margin", "symbol": "AMD",
         "security_type": "STK", "currency": "USD", "open_qty": 100.0, "side": "SHORT", "avg_price": 150.0,
         "current_price": 148.0, "market_value": -14800.0},
    )
    return BookSnapshot(fetched_utc, accounts, tuple(dict(p) for p in (default if positions is None else positions)))
