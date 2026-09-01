"""Armed alerts expire on the trading-day clock - Phase 0.12 A2.

Trader, 2026-09-01: an armed watch that never fires used to sit in the Armed
inventory forever. A surface whose job is to say "these are the exact
conditions I am waiting on" stops meaning anything once half of it is a watch
armed six weeks ago on a thesis that has since gone stale.

So an arm now has a life, measured in SESSIONS:

* a manually armed **5-day** extreme watch: 5 trading days;
* a **20-day** extreme watch: 10 trading days;
* everything else armed - D1 level watches, any-bounce watches, manual price
  alerts: 10 trading days.

Three rules this module exists to hold:

* **Sessions, never weekdays.** `market_calendar.trading_days_between` is the
  clock. Weekday arithmetic would count Thanksgiving and would bring a Friday
  arm due on the wrong Friday.
* **Uncertainty never deletes.** When the calendar refuses to answer - a date
  outside its validated range, a corrupt stamp - :func:`is_expired` returns
  ``None`` and the entry STAYS ARMED. Every caller here is about to remove
  something the trader created by hand; failing closed is the only safe
  direction.
* **Nothing is silently lost.** Every expiry appends one row naming the store,
  the symbol, the kind, when it was armed, when it came due and how many
  sessions it was given. The append is best-effort and swallowed on failure:
  an evidence store is never allowed to cost the thing it records.

Pure module - no Qt, no network, no store paths of its own. The callers own
their stores; this owns the policy.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

#: Schema by NAME, never by number (ground rule 5).
SCHEMA_ARMED_ALERT_EXPIRY = "armed_alert_expiry_v1"
STREAM = "armed_alert_expiry"

#: Kinds whose window is not the default. Keyed by the watch `kind` the store
#: already uses, so no caller has to translate.
EXPIRY_TRADING_DAYS_BY_KIND = {
    "new_5d_high": 5,
    "new_5d_low": 5,
    "new_20d_high": 10,
    "new_20d_low": 10,
}

#: Every other armed thing.
DEFAULT_EXPIRY_TRADING_DAYS = 10

#: The kind a price-alert entry is filed under. It has no `kind` of its own -
#: an entry carries an above and/or a below level - so the store names it.
PRICE_ALERT_KIND = "price_alert"


def expiry_trading_days(kind: object) -> int:
    """How many sessions this kind of arm is given."""
    return EXPIRY_TRADING_DAYS_BY_KIND.get(
        str(kind or "").strip(), DEFAULT_EXPIRY_TRADING_DAYS
    )


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def is_expired(armed_at: object, kind: object = "", *, today: date | None = None) -> bool | None:
    """Has this arm run out of sessions?

    ``True``/``False`` when the calendar can answer, ``None`` when it cannot -
    and ``None`` must never be treated as ``True``. See the module docstring.
    """
    armed_day = _as_date(armed_at)
    if armed_day is None:
        return None
    reference = today or date.today()
    try:
        from market_calendar import trading_days_between

        elapsed = trading_days_between(armed_day, reference)
    except Exception as exc:  # SessionCalendarError, or the module unavailable
        # Fail CLOSED. A caller that cannot date an arm keeps it.
        logging.debug("Armed-alert expiry could not date %s: %s", armed_day, exc)
        return None
    return elapsed >= expiry_trading_days(kind)


def expiry_row(
    *,
    store: str,
    symbol: object,
    kind: object,
    armed_at: object,
    expired_at: date | None = None,
    trading_days: int | None = None,
) -> dict[str, Any]:
    """One evidence row. Names what was removed and why it was due."""
    armed_day = _as_date(armed_at)
    return {
        "schema": SCHEMA_ARMED_ALERT_EXPIRY,
        "event": "armed_alert_expired",
        "store": str(store or ""),
        "symbol": str(symbol or "").strip().upper(),
        "kind": str(kind or ""),
        "armed_at": (
            armed_at.isoformat()
            if isinstance(armed_at, (datetime, date))
            else str(armed_at or "")
        ),
        "armed_on": armed_day.isoformat() if armed_day else "",
        "expired_at": (expired_at or date.today()).isoformat(),
        "trading_days": int(
            trading_days if trading_days is not None else expiry_trading_days(kind)
        ),
    }


def partition(
    watches: Iterable[Any],
    *,
    store: str,
    today: date | None = None,
    kind_of: Callable[[Any], str] | None = None,
    armed_at_of: Callable[[Any], Any] | None = None,
    symbol_of: Callable[[Any], str] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Split armed entries into (still armed, expiry rows for the rest).

    The accessors default to the attribute names every watch dataclass in
    `chart_watch` already uses, so the callers pass a list and nothing else.
    An entry the calendar cannot date is KEPT.
    """
    kind_of = kind_of or (lambda item: str(getattr(item, "kind", "") or ""))
    armed_at_of = armed_at_of or (lambda item: getattr(item, "armed_at", None))
    symbol_of = symbol_of or (lambda item: str(getattr(item, "symbol", "") or ""))
    reference = today or date.today()
    kept: list[Any] = []
    expired: list[dict[str, Any]] = []
    for item in watches:
        kind = kind_of(item)
        armed_at = armed_at_of(item)
        verdict = is_expired(armed_at, kind, today=reference)
        if verdict is not True:
            kept.append(item)
            continue
        expired.append(
            expiry_row(
                store=store,
                symbol=symbol_of(item),
                kind=kind,
                armed_at=armed_at,
                expired_at=reference,
            )
        )
    return kept, expired


def default_ledger():
    """The append-only stream these rows live in, or ``None`` when unavailable."""
    try:
        from evidence_ledger import EvidenceLedger

        return EvidenceLedger(stream=STREAM, schema=SCHEMA_ARMED_ALERT_EXPIRY)
    except Exception:  # pragma: no cover - the ledger is optional at runtime
        logging.debug("Armed-alert expiry ledger unavailable", exc_info=True)
        return None


def record_expiries(rows: Sequence[Mapping[str, Any]], *, ledger=None) -> int:
    """Append the rows. Returns how many landed; never raises.

    A failed append loses the row, never the expiry - the trader's Armed
    surface is the product and an unwritable evidence file must not turn a
    routine cleanup into an error they have to work around.
    """
    if not rows:
        return 0
    target = ledger if ledger is not None else default_ledger()
    if target is None:
        return 0
    try:
        return int(target.append_many(rows))
    except Exception:
        logging.debug("Armed-alert expiry rows were not written", exc_info=True)
        return 0
