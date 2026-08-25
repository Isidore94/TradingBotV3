"""Booked FX rates, from the Bank of Canada, once per (date, currency).

R7 §9 step 8, root cause B8, invariant I5.

Multi-currency P&L was summed unconverted: a USD win and a CAD loss were added
as if the numbers meant the same thing. For a Canadian trader filing Canadian
tax that is not a rounding problem, it is a wrong number.

THE THREE RULES, AND WHY

* **Booked once, at import time, never at render.** A rate fetched when a report
  is opened makes the same trade worth different amounts on different days, and
  a tax figure that moves when you look at it is not a tax figure. Rates land in
  ``fx_rates`` and every reader uses the stored value.
* **A missing rate renders as "unconverted".** Never 0, never silently native.
  A trade whose rate could not be fetched has ``net_pnl_cad = NULL``, which the
  UI shows as unconverted - visible, and impossible to add up by accident.
* **A weekend or holiday carries the prior business day's observation**, and
  records **which** day it came from in ``effective_date``. The BoC publishes no
  rate on days it is closed; carrying back the previous published observation is
  what the CRA accepts, and saying which observation was used is what makes it
  auditable.

The source is the BoC Valet API - free, no key, no rate limit worth the name.
Nothing here ever blocks an import: a failed fetch leaves the rate missing, and
missing is a state this system already knows how to render honestly.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any

import requests

BOC_VALET_URL = "https://www.bankofcanada.ca/valet/observations/FX{currency}CAD/json"
BOC_SOURCE = "BOC_VALET"

#: How far back to look for a published observation when the asked-for day has
#: none. Ten days clears any Canadian long weekend plus a stat holiday run.
MAX_CARRY_BACK_DAYS = 10

#: The one currency that needs no conversion, and the one rate that is a fact
#: rather than an observation.
BASE_CURRENCY = "CAD"


#: Machine-local setting holding the trader's manually entered USD display rate
#: and when it was entered.
MANUAL_USD_RATE_SETTING = "journal_manual_usd_cad_rate"
MANUAL_USD_RATE_STAMP_SETTING = "journal_manual_usd_cad_rate_entered_at"

#: Sanity bounds. USD/CAD has not left this range in living memory, and a
#: fat-fingered 13 or 0.13 silently rescaling every total is exactly the class
#: of error a manual field invites.
MANUAL_USD_RATE_MIN = 0.5
MANUAL_USD_RATE_MAX = 3.0


def manual_usd_rate() -> dict[str, Any] | None:
    """The trader's manually entered USD/CAD rate, or None.

    **This is a DISPLAY convenience and never a booked figure.** Everything
    above this line is point-in-time: a rate is fetched once, at import, for the
    day the trade happened, and stored, because a tax number that moves when you
    look at it is not a tax number. One current rate applied to a year of trades
    is an estimate and nothing more.

    So it is kept deliberately far away from the booked path: it lives in a
    machine-local setting rather than the ``fx_rates`` table, it never touches
    ``net_pnl_cad``, and every total computed from it is labelled as an estimate
    at the entered rate. Nothing here is CRA-facing.
    """
    from project_paths import get_local_setting

    raw = get_local_setting(MANUAL_USD_RATE_SETTING)
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        return None
    if not MANUAL_USD_RATE_MIN <= rate <= MANUAL_USD_RATE_MAX:
        return None
    return {
        "rate_cad_per_usd": rate,
        "entered_at": str(get_local_setting(MANUAL_USD_RATE_STAMP_SETTING) or ""),
        "source": "MANUAL_DISPLAY",
    }


def set_manual_usd_rate(rate: Any) -> dict[str, Any] | None:
    """Store (or clear, on a blank/None) the manual USD display rate.

    Refuses anything outside the sanity bounds rather than storing it: a
    rejected entry is visible immediately, while a stored 13.5 quietly makes
    every USD total wrong by an order of magnitude.
    """
    from project_paths import save_local_setting

    if rate is None or str(rate).strip() == "":
        save_local_setting(MANUAL_USD_RATE_SETTING, "")
        save_local_setting(MANUAL_USD_RATE_STAMP_SETTING, "")
        return None
    value = float(rate)
    if not MANUAL_USD_RATE_MIN <= value <= MANUAL_USD_RATE_MAX:
        raise ValueError(
            f"USD/CAD rate {value} is outside {MANUAL_USD_RATE_MIN}-{MANUAL_USD_RATE_MAX}"
        )
    save_local_setting(MANUAL_USD_RATE_SETTING, value)
    save_local_setting(MANUAL_USD_RATE_STAMP_SETTING, _now_iso())
    return manual_usd_rate()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def fetch_observations(
    currency: str,
    start: date,
    end: date,
    *,
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> dict[date, float]:
    """Published FX{CUR}CAD observations in a range. Network call.

    Returns only days the BoC actually published - weekends and holidays are
    simply absent, which is what :func:`ensure_rates` carries back from.
    """
    code = str(currency or "").strip().upper()
    if not code or code == BASE_CURRENCY:
        return {}
    http = session or requests.Session()
    response = http.get(
        BOC_VALET_URL.format(currency=code),
        params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    series_key = f"FX{code}CAD"
    observations: dict[date, float] = {}
    for row in payload.get("observations", []) if isinstance(payload, dict) else []:
        if not isinstance(row, dict):
            continue
        try:
            day = _as_date(row.get("d"))
            value = float((row.get(series_key) or {}).get("v"))
        except (TypeError, ValueError, KeyError):
            continue
        observations[day] = value
    return observations


def stored_rate(store: Any, day: Any, currency: str) -> dict[str, Any] | None:
    """The booked rate for one (date, currency), or None. No network."""
    code = str(currency or "").strip().upper()
    if code == BASE_CURRENCY:
        return {
            "rate_date": _as_date(day).isoformat(),
            "currency": BASE_CURRENCY,
            "rate_to_cad": 1.0,
            "source": "identity",
            "effective_date": _as_date(day).isoformat(),
        }
    with store.connection() as conn:
        row = conn.execute(
            "SELECT * FROM fx_rates WHERE rate_date = ? AND currency = ?",
            (_as_date(day).isoformat(), code),
        ).fetchone()
    return {key: row[key] for key in row.keys()} if row else None


def missing_pairs(store: Any, pairs: Iterable[tuple[Any, str]]) -> list[tuple[date, str]]:
    """Which (date, currency) pairs are not booked yet. CAD is never missing."""
    wanted = {
        (_as_date(day), str(currency or "").strip().upper())
        for day, currency in pairs
        if str(currency or "").strip().upper() not in {"", BASE_CURRENCY}
    }
    if not wanted:
        return []
    with store.connection() as conn:
        known = {
            (_as_date(row[0]), str(row[1]))
            for row in conn.execute("SELECT rate_date, currency FROM fx_rates")
        }
    return sorted(wanted - known)


def ensure_rates(
    store: Any,
    pairs: Iterable[tuple[Any, str]],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Book every missing (date, currency), carrying back where BoC is closed.

    Never raises. A currency whose fetch fails is reported and left unbooked,
    because "unconverted" is an honest answer and a guessed rate is not.
    """
    summary: dict[str, Any] = {"booked": 0, "carried_back": 0, "unavailable": [], "errors": []}
    outstanding = missing_pairs(store, pairs)
    if not outstanding:
        return summary

    by_currency: dict[str, list[date]] = {}
    for day, currency in outstanding:
        by_currency.setdefault(currency, []).append(day)

    for currency, days in sorted(by_currency.items()):
        first, last = min(days), max(days)
        try:
            observations = fetch_observations(
                currency, first - timedelta(days=MAX_CARRY_BACK_DAYS), last, session=session
            )
        except Exception as exc:  # noqa: BLE001 - an import must not die of this
            summary["errors"].append({"currency": currency, "message": str(exc)})
            continue
        if not observations:
            summary["unavailable"].extend(
                {"currency": currency, "date": day.isoformat(), "reason": "no observations"}
                for day in days
            )
            continue
        rows = []
        for day in sorted(days):
            effective = day
            value = observations.get(day)
            carried = False
            probe = day
            while value is None and (day - probe).days < MAX_CARRY_BACK_DAYS:
                probe -= timedelta(days=1)
                value = observations.get(probe)
                if value is not None:
                    effective = probe
                    carried = True
            if value is None:
                summary["unavailable"].append(
                    {"currency": currency, "date": day.isoformat(), "reason": "no published rate within carry-back"}
                )
                continue
            rows.append((day, value, effective))
            summary["booked"] += 1
            if carried:
                summary["carried_back"] += 1
        if rows:
            with store.connection() as conn:
                for day, value, effective in rows:
                    conn.execute(
                        """
                        INSERT INTO fx_rates(
                            rate_date, currency, rate_to_cad, source, effective_date, fetched_at
                        ) VALUES(?, ?, ?, ?, ?, ?)
                        ON CONFLICT(rate_date, currency) DO UPDATE SET
                            rate_to_cad = excluded.rate_to_cad,
                            source = excluded.source,
                            effective_date = excluded.effective_date,
                            fetched_at = excluded.fetched_at
                        """,
                        (
                            day.isoformat(),
                            currency,
                            float(value),
                            BOC_SOURCE,
                            effective.isoformat(),
                            _now_iso(),
                        ),
                    )
    return summary


#: The one non-CAD currency this system also DISPLAYS in. A USD observation is
#: booked for every session that has trades, not only the sessions that happen
#: to hold a USD trade, because converting a CAD trade INTO USD needs the rate
#: for that CAD trade's own day.
DISPLAY_CURRENCY = "USD"


def rates_needed_for_trades(store: Any) -> list[tuple[date, str]]:
    """The (date, currency) pairs the current trade table needs converting.

    Two jobs, not one. Each trade's own currency is needed to book the
    tax-grade CAD value (I5). **And a USD observation is needed for every
    session that has trades at all**, including CAD-only ones - without it a
    CAD trade can never be displayed in USD, however honest the render seam is.
    That gap is why true USD conversion was deferred in the first place: the
    rate was never asked for, so it was never there to book from.
    """
    with store.connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT trade_date, currency FROM trades
            WHERE COALESCE(trade_date, '') != '' AND UPPER(COALESCE(currency, '')) != 'CAD'
            """
        ).fetchall()
        sessions = conn.execute(
            "SELECT DISTINCT trade_date FROM trades WHERE COALESCE(trade_date, '') != ''"
        ).fetchall()
    pairs: list[tuple[date, str]] = []
    for row in rows:
        try:
            pairs.append((_as_date(row[0]), str(row[1]).upper()))
        except ValueError:
            continue
    for row in sessions:
        try:
            pairs.append((_as_date(row[0]), DISPLAY_CURRENCY))
        except ValueError:
            continue
    return sorted(set(pairs))


def rates_needed_for_executions(store: Any) -> list[tuple[date, str]]:
    """Pairs introduced by raw imports before tonight's rebuild creates trades."""
    with store.connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT trade_date, currency FROM raw_executions
            WHERE COALESCE(trade_date, '') != '' AND UPPER(COALESCE(currency, '')) != 'CAD'
            """
        ).fetchall()
    pairs: list[tuple[date, str]] = []
    for row in rows:
        try:
            pairs.append((_as_date(row[0]), str(row[1]).upper()))
        except ValueError:
            continue
    return pairs


def seed_rate(store: Any, *, day: Any, currency: str, rate_to_cad: float, effective_date: Any = None) -> None:
    """Book a rate directly. For tests and for the one-off manual repair."""
    booked = _as_date(day)
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO fx_rates(rate_date, currency, rate_to_cad, source, effective_date, fetched_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(rate_date, currency) DO UPDATE SET
                rate_to_cad = excluded.rate_to_cad,
                effective_date = excluded.effective_date,
                fetched_at = excluded.fetched_at
            """,
            (
                booked.isoformat(),
                str(currency).upper(),
                float(rate_to_cad),
                BOC_SOURCE,
                _as_date(effective_date or day).isoformat(),
                _now_iso(),
            ),
        )


def describe_coverage(store: Any) -> dict[str, Any]:
    """What the Health tab shows: how much of the journal is convertible."""
    with store.connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        converted = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE net_pnl_cad IS NOT NULL"
        ).fetchone()[0]
        unconverted = [
            {"currency": str(row[0]), "trades": int(row[1])}
            for row in conn.execute(
                """
                SELECT currency, COUNT(*) FROM trades
                WHERE net_pnl_cad IS NULL GROUP BY currency ORDER BY currency
                """
            )
        ]
    return {
        "trades": int(total),
        "converted": int(converted),
        "unconverted": unconverted,
        "booked_rates": _booked_rate_count(store),
    }


def _booked_rate_count(store: Any) -> int:
    with store.connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM fx_rates").fetchone()[0])


def carried_back_rates(store: Any) -> list[dict[str, Any]]:
    """Rates whose observation came from an earlier day, for the audit trail."""
    with store.connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM fx_rates
            WHERE COALESCE(effective_date, '') != '' AND effective_date != rate_date
            ORDER BY rate_date, currency
            """
        ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


__all__ = [
    "BASE_CURRENCY",
    "BOC_SOURCE",
    "BOC_VALET_URL",
    "MAX_CARRY_BACK_DAYS",
    "carried_back_rates",
    "describe_coverage",
    "ensure_rates",
    "fetch_observations",
    "missing_pairs",
    "rates_needed_for_trades",
    "seed_rate",
    "stored_rate",
]
