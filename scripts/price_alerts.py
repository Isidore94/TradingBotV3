"""Position price-level alerts: the wake-me-up watchlist for Evening mode.

The trader enters current positions (or SPY itself) with alert levels above
and/or below the current price the night before a sleep-in morning. While the
GUI runs, the PriceAlertService polls last prices (yfinance 1m bars including
pre/post market) and pushes a phone/watch notification the moment a level is
crossed - urgent priority in Evening mode so it breaks through sleep focus.

Semantics chosen for wake-up safety:

- Each side (above/below) fires ONCE per arm, then disarms itself. A level
  that keeps re-firing every minute while price chops around it would train
  the trader to ignore the channel; one shot per arm means a notification is
  always news. Re-arm from the panel (or Re-arm All for the next night).
- Entries are trader-owned. Nothing here ever removes an entry - the engine
  only flips armed flags and records trigger history (plan.md sec 5 spirit:
  user-entered names are never auto-removed). That is still true of the
  trading-day EXPIRY added 2026-09-01 (Phase 0.12 A2): an alert that has sat
  armed for 10 trading days without firing is DISARMED, never deleted. It
  leaves the Armed surface - which is what the trader asked for - and keeps
  its levels, its note and its trigger history exactly where they were, so
  re-arming it is one click and nothing has to be retyped.
- Everything below is pure (store I/O aside) so the trigger rules are tested
  without Qt or network.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from project_paths import PRICE_ALERT_TRIGGERS_FILE, PRICE_ALERTS_FILE

_STORE_LOCK = threading.Lock()

# History kept per entry - enough to see "what happened overnight" without the
# file growing unbounded.
_MAX_ENTRY_HISTORY = 8


def normalize_price_alert(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """One store entry in canonical shape, or ``None`` for garbage rows."""
    symbol = str(raw.get("symbol") or "").strip().upper()
    if not symbol:
        return None

    def _level(value: Any) -> float | None:
        try:
            level = float(value)
        except (TypeError, ValueError):
            return None
        return level if level > 0 else None

    above = _level(raw.get("above"))
    below = _level(raw.get("below"))
    # A2: when the arm started, so the trading-day expiry has something to
    # measure. A row written before this field existed gets TODAY, never an
    # older guess - guessing backwards would disarm the trader's alerts on
    # the first load after the upgrade.
    armed_at = str(raw.get("armed_at") or "").strip()[:10]
    try:
        date.fromisoformat(armed_at)
    except ValueError:
        armed_at = date.today().isoformat()
    return {
        "symbol": symbol,
        "above": above,
        "below": below,
        "armed_at": armed_at,
        # A side with no level is never armed, whatever the flag says.
        "armed_above": bool(raw.get("armed_above", True)) and above is not None,
        "armed_below": bool(raw.get("armed_below", True)) and below is not None,
        "note": str(raw.get("note") or "").strip(),
        "history": [dict(item) for item in raw.get("history") or [] if isinstance(item, Mapping)][
            -_MAX_ENTRY_HISTORY:
        ],
    }


def load_price_alerts(path: Path = PRICE_ALERTS_FILE) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    normalized = []
    for raw in entries:
        if isinstance(raw, Mapping):
            entry = normalize_price_alert(raw)
            if entry is not None:
                normalized.append(entry)
    return normalized


def save_price_alerts(entries: Iterable[Mapping[str, Any]], path: Path = PRICE_ALERTS_FILE) -> bool:
    cleaned = []
    for raw in entries:
        entry = normalize_price_alert(raw)
        if entry is not None:
            cleaned.append(entry)
    payload = {"entries": cleaned, "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with _STORE_LOCK:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except OSError:
        logging.exception("Price alert store save failed")
        return False


def armed_symbols(entries: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            str(entry.get("symbol") or "").strip().upper()
            for entry in entries
            if entry.get("armed_above") or entry.get("armed_below")
        }
        - {""}
    )


def mark_armed_now(entry: dict[str, Any], *, today: date | None = None) -> dict[str, Any]:
    """Restart one entry's trading-day clock. Mutates and returns the entry.

    Called wherever a side is armed - a new alert, a changed level, "Re-arm
    selected". Without it a re-armed alert would still carry the stamp that
    expired it and would be disarmed again on the next poll.
    """
    entry["armed_at"] = (today or date.today()).isoformat()
    return entry


def expire_stale_alerts(
    entries: Iterable[Mapping[str, Any]],
    *,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Disarm alerts that have sat armed past their trading-day window.

    Returns ``(entries, expiry rows)``. Both sides of one entry share its
    ``armed_at``, so an entry expires whole. An entry with nothing armed is
    already off the Armed surface and writes no row - expiring it again every
    poll would be an unbounded stream of rows about nothing.

    Uncertainty never deletes: an ``armed_at`` the calendar cannot reason
    about leaves the entry armed (see `armed_alert_expiry.is_expired`).
    """
    import armed_alert_expiry

    reference = today or date.today()
    updated: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for raw in entries:
        entry = normalize_price_alert(raw)
        if entry is None:
            continue
        if not (entry["armed_above"] or entry["armed_below"]):
            updated.append(entry)
            continue
        verdict = armed_alert_expiry.is_expired(
            entry.get("armed_at"), armed_alert_expiry.PRICE_ALERT_KIND, today=reference
        )
        if verdict is not True:
            updated.append(entry)
            continue
        entry["armed_above"] = False
        entry["armed_below"] = False
        updated.append(entry)
        rows.append(
            armed_alert_expiry.expiry_row(
                store="price_alerts",
                symbol=entry["symbol"],
                kind=armed_alert_expiry.PRICE_ALERT_KIND,
                armed_at=entry.get("armed_at"),
                expired_at=reference,
            )
        )
    return updated, rows


def evaluate_price_alerts(
    entries: Iterable[Mapping[str, Any]],
    quotes: Mapping[str, Mapping[str, Any]],
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(updated entries, fired triggers) for one polling pass.

    A trigger fires when an armed side's level is crossed by the last price
    (>= for above, <= for below). The fired side disarms in the returned
    entries; callers persist them so the same cross cannot fire twice.
    """
    moment = now or datetime.now()
    updated: list[dict[str, Any]] = []
    triggers: list[dict[str, Any]] = []
    for raw in entries:
        entry = normalize_price_alert(raw)
        if entry is None:
            continue
        quote = quotes.get(entry["symbol"]) or {}
        try:
            last = float(quote.get("last"))
        except (TypeError, ValueError):
            last = None
        if last is not None and last > 0:
            for side in ("above", "below"):
                level = entry[side]
                if level is None or not entry[f"armed_{side}"]:
                    continue
                crossed = last >= level if side == "above" else last <= level
                if not crossed:
                    continue
                trigger = {
                    "symbol": entry["symbol"],
                    "side": side,
                    "level": level,
                    "last": last,
                    "at": moment.strftime("%H:%M:%S"),
                    "date": moment.date().isoformat(),
                    "note": entry["note"],
                }
                entry[f"armed_{side}"] = False
                entry["history"] = (entry["history"] + [trigger])[-_MAX_ENTRY_HISTORY:]
                triggers.append(trigger)
        updated.append(entry)
    return updated, triggers


def format_trigger_message(trigger: Mapping[str, Any]) -> str:
    direction = "ABOVE" if str(trigger.get("side")) == "above" else "BELOW"
    note = str(trigger.get("note") or "").strip()
    note_text = f" ({note})" if note else ""
    return (
        f"{trigger.get('symbol')} {float(trigger.get('last', 0.0)):.2f} crossed {direction} "
        f"your {float(trigger.get('level', 0.0)):.2f} alert level{note_text}"
    )


def append_trigger_log(
    triggers: Iterable[Mapping[str, Any]],
    path: Path = PRICE_ALERT_TRIGGERS_FILE,
) -> None:
    """Trigger history CSV - the briefing's overnight-alerts section reads it."""
    rows = list(triggers)
    if not rows:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["date", "at", "symbol", "side", "level", "last", "note"]
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    except Exception:
        logging.exception("Price alert trigger log append failed")


def todays_triggers(
    now: datetime | None = None,
    path: Path = PRICE_ALERT_TRIGGERS_FILE,
) -> list[dict[str, str]]:
    today = (now or datetime.now()).date().isoformat()
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle) if row.get("date") == today]
    except OSError:
        return []


def fetch_last_quotes(
    symbols: Iterable[str],
    *,
    downloader: Callable[..., Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """{symbol: {"last": float, "at": datetime}} from 1m bars incl. pre/post.

    Pre/post matters here more than anywhere else in the app: a position
    gapping through a level at 05:00 on news is exactly the wake-up this
    watchlist exists for.
    """
    pool = sorted({str(s or "").strip().upper() for s in symbols if str(s or "").strip()})
    if not pool:
        return {}
    downloader = downloader or _default_quote_downloader
    try:
        data = downloader(pool)
    except Exception as exc:
        if log:
            log(f"Price alert quote fetch failed: {exc}")
        return {}
    from autopilot_core import _frame_rows

    quotes: dict[str, dict[str, Any]] = {}
    for symbol in pool:
        try:
            frame = data[symbol] if len(pool) > 1 else data
        except Exception:
            frame = None
        rows = _frame_rows(frame)
        if not rows:
            continue
        quotes[symbol] = {"last": rows[-1]["close"], "at": rows[-1]["dt"]}
    return quotes


def _default_quote_downloader(symbols: list[str]):
    import yfinance as yf

    return yf.download(
        tickers=" ".join(symbols),
        period="1d",
        interval="1m",
        prepost=True,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
