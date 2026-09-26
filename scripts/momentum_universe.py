"""The `momentum_scanner` universe source: strong names from IB market scanners, kept on a rolling list.

Once a day, from the universe rebuild, IB scanner lists of strong US stocks (near 13/26/52-week
highs, today's top % gainers) are pulled on this module's own client id. A sighted name that clears
the price, share-volume and market-cap floors (read from the rebuild's own yfinance metrics; unpriced = unknown =
not admitted) joins the membership and stays for MOMENTUM_KEEP_SESSIONS observed sessions after it
was last sighted, so a name that was strong once is still scanned on its pullback. Capped at
MOMENTUM_MAX_SYMBOLS, most recently sighted first.

"Sessions" are days a refresh succeeded: an IB outage ages nothing and never wipes the list.
One writer: `refresh_membership` (called only by `universe_builder.build_universe`). Scanner
subscriptions only - no orders, no market or historical data.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

from project_paths import MOMENTUM_UNIVERSE_MEMBERSHIP_FILE

MOMENTUM_SOURCE = "momentum_scanner"
#: IB scan codes for strength: close near the 52/26/13-week high, then today's top % gainers.
MOMENTUM_SCAN_CODES = ("HIGH_VS_52W_HL", "HIGH_VS_26W_HL", "HIGH_VS_13W_HL", "TOP_PERC_GAIN")
#: IB's scanner answers at most 50 rows per subscription.
MOMENTUM_SCAN_ROWS = 50
MOMENTUM_MIN_PRICE = 5.0
#: The trader's minimum (2026-09-26): 20-session mean share volume and market cap ($M), the same
#: floors as the base universe screen (`universe_builder.DEFAULT_MIN_*`).
MOMENTUM_MIN_AVG_VOLUME = 1_000_000
MOMENTUM_MIN_MARKET_CAP_M = 1000.0
#: A member stays this many observed sessions after it was last sighted.
MOMENTUM_KEEP_SESSIONS = 60
#: Most names the source may hold (most recently sighted first, then most sightings).
MOMENTUM_MAX_SYMBOLS = 300
#: Own IB client id (Movers uses 9135-9137, Options Chase 9145); retries take the next two.
MOMENTUM_CLIENT_ID = 9140
MEMBERSHIP_SCHEMA = 1
#: Observed refresh sessions kept on file (only the last MOMENTUM_KEEP_SESSIONS matter).
_SESSIONS_KEPT = 400


class MembershipUnreadable(RuntimeError):
    """The membership file exists but cannot be read; it is never overwritten."""


def fetch_momentum_scans() -> dict[str, list[str]]:
    """Symbols per scan code from TWS. Raises `ib_market_scanner.ScannerError` when IB gives nothing."""
    import ib_market_scanner as ims

    scanner = ims.IBMarketScanner(client_id=MOMENTUM_CLIENT_ID)
    try:
        return scanner.run_scans(
            MOMENTUM_SCAN_CODES, rows=MOMENTUM_SCAN_ROWS, above_price=MOMENTUM_MIN_PRICE
        )
    finally:
        scanner.close()


def sighted_symbols(
    results: Mapping[str, list[str]] | None,
    metrics: Mapping[str, tuple[float | None, float | None]],
    *,
    market_caps: Callable[[list[str]], Mapping[str, float]] | None = None,
    min_price: float = MOMENTUM_MIN_PRICE,
    min_avg_volume: float = MOMENTUM_MIN_AVG_VOLUME,
    min_market_cap_m: float = MOMENTUM_MIN_MARKET_CAP_M,
) -> dict[str, list[str]]:
    """Admitted symbol -> the scan codes that sighted it, in scan then rank order.

    `metrics` is symbol -> (last price, 20-session mean share volume). `market_caps` returns
    symbol -> cap in $M for the names that pass price and volume (asked once). A missing
    price, volume or cap is unknown and not admitted.
    """
    sighted: dict[str, list[str]] = {}
    for code, names in (results or {}).items():
        for raw in names or ():
            symbol = str(raw or "").strip().upper()
            if not symbol:
                continue
            price, avg_volume = metrics.get(symbol, (None, None))
            if price is None or avg_volume is None:
                continue
            if float(price) < min_price or float(avg_volume) < min_avg_volume:
                continue
            codes = sighted.setdefault(symbol, [])
            if code not in codes:
                codes.append(str(code))
    caps = dict(market_caps(sorted(sighted)) or {}) if (market_caps and sighted) else {}
    return {
        symbol: codes for symbol, codes in sighted.items()
        if float(caps.get(symbol) or 0.0) >= min_market_cap_m
    }


def empty_membership() -> dict[str, Any]:
    return {"schema": MEMBERSHIP_SCHEMA, "refreshed_on": None, "sessions": [], "members": {}}


def load_membership(path: Path | None = None) -> dict[str, Any]:
    """The stored membership; a missing file is empty. Raises MembershipUnreadable otherwise."""
    target = Path(path or MOMENTUM_UNIVERSE_MEMBERSHIP_FILE)
    if not target.exists():
        return empty_membership()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MembershipUnreadable(f"{target}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("members"), dict):
        raise MembershipUnreadable(f"{target}: not a membership file")
    payload.setdefault("sessions", [])
    return payload


def _ranked(members: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(
        members,
        key=lambda s: (str(members[s].get("last_seen") or ""), int(members[s].get("hits") or 0), s),
        reverse=True,
    )


def active_members(
    membership: Mapping[str, Any],
    *,
    keep_sessions: int = MOMENTUM_KEEP_SESSIONS,
    cap: int = MOMENTUM_MAX_SYMBOLS,
) -> list[str]:
    """Members sighted within the last `keep_sessions` observed sessions, capped, ranked first-kept first."""
    members = membership.get("members") or {}
    sessions = sorted(str(day) for day in membership.get("sessions") or [])
    # Active while at most `keep_sessions` observed sessions have passed since the last sighting.
    cutoff = sessions[-(int(keep_sessions) + 1)] if len(sessions) > int(keep_sessions) else ""
    live = {s: m for s, m in members.items() if str(m.get("last_seen") or "") >= cutoff}
    return _ranked(live)[: int(cap)]


def update_membership(
    membership: Mapping[str, Any],
    sighted: Mapping[str, list[str]],
    *,
    today: date,
    keep_sessions: int = MOMENTUM_KEEP_SESSIONS,
    cap: int = MOMENTUM_MAX_SYMBOLS,
) -> dict[str, Any]:
    """A new membership: today counts as one observed session, sighted names are stamped, then expiry and cap."""
    day = today.isoformat()
    sessions = sorted({str(d) for d in membership.get("sessions") or []} | {day})[-_SESSIONS_KEPT:]
    members = {s: dict(m) for s, m in (membership.get("members") or {}).items()}
    for symbol, codes in sighted.items():
        entry = members.setdefault(symbol, {"first_seen": day, "hits": 0, "scans": []})
        if entry.get("last_seen") != day:
            entry["hits"] = int(entry.get("hits") or 0) + 1
        entry["last_seen"] = day
        entry["scans"] = list(dict.fromkeys([*codes, *(entry.get("scans") or [])]))
    draft = {"schema": MEMBERSHIP_SCHEMA, "refreshed_on": day, "sessions": sessions, "members": members}
    kept = active_members(draft, keep_sessions=keep_sessions, cap=cap)
    draft["members"] = {s: members[s] for s in sorted(kept)}
    return draft


def refresh_membership(
    metrics: Mapping[str, tuple[float | None, float | None]],
    *,
    today: date | None = None,
    path: Path | None = None,
    fetch: Callable[[], dict[str, list[str]]] | None = None,
    market_caps: Callable[[list[str]], Mapping[str, float]] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Refresh the store once per day and return the active members with a report for the ledger.

    Already refreshed today, IB unavailable, or `write=False`: no store write, the stored members
    stand. An unreadable store is never overwritten (its members are unknown, so none are returned).
    """
    started = time.monotonic()
    day = today or date.today()
    target = Path(path or MOMENTUM_UNIVERSE_MEMBERSHIP_FILE)
    report: dict[str, Any] = {"refreshed": False, "sighted": 0, "error": "", "members": []}
    try:
        membership = load_membership(target)
    except MembershipUnreadable as exc:
        logging.warning("Momentum universe membership unreadable; left untouched: %s", exc)
        report["error"] = f"unreadable: {exc}"
        return report
    if write and membership.get("refreshed_on") != day.isoformat():
        try:
            results = (fetch or fetch_momentum_scans)()
        except Exception as exc:
            logging.warning("Momentum universe scanner unavailable; yesterday's membership kept: %s", exc)
            report["error"] = str(exc) or type(exc).__name__
        else:
            sighted = sighted_symbols(results, metrics, market_caps=market_caps)
            membership = update_membership(membership, sighted, today=day)
            from diagnostics.artifact_io import atomic_write_json

            atomic_write_json(target, membership)
            report.update(refreshed=True, sighted=len(sighted), scans={k: len(v) for k, v in results.items()})
    report["members"] = active_members(membership)
    report["elapsed_s"] = round(time.monotonic() - started, 1)
    return report
