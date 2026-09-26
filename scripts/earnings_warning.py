"""Earnings warning on SHORT setups (S10b). Display only.

A SHORT within 0-14 calendar days of its next earnings date gets one warning
line, with the scan-factor leaderboard's measured edge for shorts 3-14 days
before earnings (SHORT, horizon 5, SPY-relative). It never hides, sorts or
mutes anything. Unknown days = no warning, never a guess.

`warm_cache()` reads the files and must run off the Qt thread; the `cached_*`
readers and `warning_for_symbol` never open a file. `request_warm()` runs one
warm on a single background thread, at most once per `WARM_INTERVAL_SECONDS`.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

#: Inclusive calendar-day window before earnings in which a short is warned.
WARN_MAX_DAYS = 14
FACTOR_KEY = "days_to_next_earnings"
STAT_SIDE = "SHORT"
STAT_HORIZON = "5"
#: The leaderboard buckets that make up "3-14 d before earnings".
STAT_BUCKETS = ("3 to < 7", "7 to < 14")
WARM_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True)
class ShortEarningsStat:
    """Observation-weighted SPY-relative short return, 3-14 d before earnings."""

    spy_rel_pct: float
    lookback_days: int
    observations: int


def _days_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return int(number)


def is_warned(days_to_next_earnings: Any, side: Any) -> bool:
    """True for a SHORT with a known next earnings date 0-14 days out."""
    if str(side or "").strip().upper() != "SHORT":
        return False
    days = _days_or_none(days_to_next_earnings)
    return days is not None and 0 <= days <= WARN_MAX_DAYS


def short_into_earnings(
    days_to_next_earnings: Any, side: Any, stat: ShortEarningsStat | None = None
) -> str:
    """The warning line, or "" when this row is not a short into earnings."""
    if not is_warned(days_to_next_earnings, side):
        return ""
    days = _days_or_none(days_to_next_earnings)
    if stat is None:
        return f"earnings in {days} d - short into earnings"
    return (
        f"earnings in {days} d - shorts 3-14 d before earnings: "
        f"{stat.spy_rel_pct:+.1f}% vs SPY ({stat.lookback_days} d)"
    )


def badge_text(days_to_next_earnings: Any, side: Any) -> str:
    """Short chip text ("ER 5d"), or "" when there is no warning."""
    if not is_warned(days_to_next_earnings, side):
        return ""
    return f"ER {_days_or_none(days_to_next_earnings)}d"


def stat_from_rows(rows: Any) -> ShortEarningsStat | None:
    """The 3-14 d SHORT h5 stat from leaderboard rows; None unless both buckets are there."""
    found: dict[str, tuple[float, int, int]] = {}
    for row in rows or ():
        if str(row.get("factor_key") or "") != FACTOR_KEY:
            continue
        if str(row.get("side") or "").strip().upper() != STAT_SIDE:
            continue
        if str(row.get("horizon_sessions") or "").strip() != STAT_HORIZON:
            continue
        bucket = str(row.get("value_label") or "").strip()
        if bucket not in STAT_BUCKETS:
            continue
        try:
            value = float(row.get("avg_spy_relative_side_return_pct"))
            count = int(float(row.get("observation_count")))
            lookback = int(float(row.get("lookback_days")))
        except (TypeError, ValueError):
            continue
        if value != value or count <= 0:
            continue
        found[bucket] = (value, count, lookback)
    if any(bucket not in found for bucket in STAT_BUCKETS):
        return None
    total = sum(count for _value, count, _lb in found.values())
    mean = sum(value * count for value, count, _lb in found.values()) / total
    lookback = max(lb for _value, _count, lb in found.values())
    return ShortEarningsStat(spy_rel_pct=mean, lookback_days=lookback, observations=total)


def read_stat(path: Path | None = None) -> ShortEarningsStat | None:
    """Read the stat from the scan-factor leaderboard CSV (file I/O: never on Qt)."""
    target = Path(path) if path is not None else _default_leaderboard_path()
    try:
        with target.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("factor_key") == FACTOR_KEY]
    except OSError:
        return None
    return stat_from_rows(rows)


def future_dates_from_history(payload: Any, *, since: date) -> dict[str, list[date]]:
    """{SYMBOL: sorted earnings dates on or after `since`} from an earnings history payload."""
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    result: dict[str, list[date]] = {}
    for raw_symbol, entry in (symbols or {}).items():
        symbol = str(raw_symbol or "").strip().upper()
        events = entry.get("events") if isinstance(entry, dict) else None
        dates: set[date] = set()
        for event in events if isinstance(events, list) else ():
            text = str((event or {}).get("earnings_date") or "").strip()[:10]
            try:
                parsed = date.fromisoformat(text)
            except ValueError:
                continue
            if parsed >= since:
                dates.add(parsed)
        if symbol and dates:
            result[symbol] = sorted(dates)
    return result


def days_to_next(dates: list[date] | None, today: date) -> int | None:
    """Calendar days from `today` to the first date on or after it; None when unknown."""
    for value in dates or ():
        if value >= today:
            return (value - today).days
    return None


# ---------------------------------------------------------------- the cache
_lock = threading.Lock()
_cache: dict[str, Any] = {
    "stat_signature": None,
    "stat": None,
    "dates_signature": None,
    "dates": {},
    "warmed_at": None,
    "warming": False,
}


def _default_leaderboard_path() -> Path:
    from project_paths import MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE

    return Path(MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE)


def _default_history_path() -> Path:
    from project_paths import EARNINGS_CALENDAR_HISTORY_FILE

    return Path(EARNINGS_CALENDAR_HISTORY_FILE)


def _signature(target: Path) -> tuple:
    try:
        stat = target.stat()
        return (str(target), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(target), None, None)


def market_today() -> date:
    """Today's date on the exchange clock."""
    try:
        from market_calendar import MARKET_TZ

        return datetime.now(MARKET_TZ).date()
    except Exception:  # noqa: BLE001 - fall back to the local date
        return date.today()


def warm_cache(
    leaderboard_path: Path | None = None, history_path: Path | None = None
) -> bool:
    """Re-read the leaderboard and earnings history when they moved. True when changed. Never on Qt."""
    changed = False
    board = Path(leaderboard_path) if leaderboard_path is not None else _default_leaderboard_path()
    signature = _signature(board)
    with _lock:
        stale = signature != _cache["stat_signature"]
    if stale:
        try:
            stat = read_stat(board) if signature[1] is not None else None
        except Exception:  # noqa: BLE001 - a warning line never costs the table
            logging.warning("Earnings warning: leaderboard unreadable; keeping the last stat.", exc_info=True)
        else:
            with _lock:
                changed = changed or stat != _cache["stat"]
                _cache["stat_signature"] = signature
                _cache["stat"] = stat
    history = Path(history_path) if history_path is not None else _default_history_path()
    signature = _signature(history)
    with _lock:
        stale = signature != _cache["dates_signature"]
    if stale:
        try:
            payload = (
                json.loads(history.read_text(encoding="utf-8")) if signature[1] is not None else {}
            )
            # One day of slack so a date read yesterday evening still counts today.
            since = date.fromordinal(market_today().toordinal() - 1)
            dates = future_dates_from_history(payload, since=since)
        except Exception:  # noqa: BLE001
            logging.warning("Earnings warning: history unreadable; keeping the last dates.", exc_info=True)
        else:
            with _lock:
                changed = changed or dates != _cache["dates"]
                _cache["dates_signature"] = signature
                _cache["dates"] = dates
    with _lock:
        _cache["warmed_at"] = time.monotonic()
    return changed


def request_warm() -> None:
    """Start one background warm unless one is running or ran in the last minute."""
    with _lock:
        last = _cache["warmed_at"]
        if _cache["warming"]:
            return
        if last is not None and time.monotonic() - last < WARM_INTERVAL_SECONDS:
            return
        _cache["warming"] = True

    def _run() -> None:
        try:
            warm_cache()
        except Exception:  # noqa: BLE001
            logging.warning("Earnings warning warm failed.", exc_info=True)
        finally:
            with _lock:
                _cache["warming"] = False
                if _cache["warmed_at"] is None:
                    _cache["warmed_at"] = time.monotonic()

    threading.Thread(target=_run, name="earnings-warning-warm", daemon=True).start()


def cached_stat() -> ShortEarningsStat | None:
    """The stat already in memory. Never opens a file."""
    with _lock:
        return _cache["stat"]


def cached_days_to_next_earnings(symbol: Any, *, today: date | None = None) -> int | None:
    """Days to the symbol's next known earnings date from memory; None when unknown."""
    key = str(symbol or "").strip().upper()
    if not key:
        return None
    with _lock:
        dates_map = _cache["dates"]
    dates = dates_map.get(key)
    if not dates and "." in key:
        dates = dates_map.get(key.replace(".", "-"))
    return days_to_next(dates, today or market_today())


def warning_for_symbol(symbol: Any, side: Any, *, today: date | None = None) -> str:
    """The warning line for a symbol and side from memory only ("" when none)."""
    if str(side or "").strip().upper() != "SHORT":
        return ""
    request_warm()
    return short_into_earnings(
        cached_days_to_next_earnings(symbol, today=today), side, cached_stat()
    )


def reset_cache_for_tests() -> None:
    with _lock:
        _cache.update(
            stat_signature=None, stat=None, dates_signature=None, dates={},
            warmed_at=None, warming=False,
        )
