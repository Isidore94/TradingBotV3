"""The Alert Center's "Show" filter for M5 rows (P9, display only).

Three choices: All, Grade B and up, Best right now. It hides M5 rows from what
the trader SEES; every alert is still recorded, still reaches the review-queue
door, the Working-now strip and the evidence files. Rows on names the trader
typed, Focus names, armed watches, price alerts and regime-pause rows always
show. Unknown (grades not loaded, Best list not ranked yet) shows.

S2 (finding F5): a separate switch, default on, hides M5 rows whose alert time
is 09:30-10:00 ET. Top-grade rows (`bypass_grades`, P14) and the always-show
rows above still show; an alert with no timezone-aware time is unknown and shows.

Longs off (the trader 2026-09-26): with `longs_market_gate` on and the market
not on a long's side, LONG rows hide (reason `longs_off`); the always-show rows
and names with an open position still show, and an unknown market shows.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Iterable, Mapping

import project_paths
import setup_grades

#: Machine-local setting (same store as the sector switch).
SETTING_SHOW_FILTER = "alert_show_filter"

ALL = "all"
GRADE_B_UP = "grade_b_up"
BEST_NOW = "best_now"
#: (value, combo label), in menu order.
MODES = (
    (ALL, "Show: All"),
    (GRADE_B_UP, "Show: Grade B and up"),
    (BEST_NOW, "Show: Best right now"),
)
DEFAULT_MODE = GRADE_B_UP
PASSING_GRADES = frozenset({setup_grades.PROVEN, setup_grades.A, setup_grades.B})
#: P14 (trader 2026-09-26, "retire PROVEN"): the grades whose M5 rows pass the
#: tier gate and the first-30 switch - A and up while any day-trade cell has it.
TOP_GRADES = frozenset({setup_grades.PROVEN, setup_grades.A})


def bypass_grades(lookup: Mapping[str, Any] | None) -> frozenset:
    """A and up when any day-trade cell grades A or PROVEN, else B and up; empty before grades load.

    B stands in only while no A exists (F1: a 1:1 bracket has not cleared A).
    """
    if not lookup:
        return frozenset()
    grades = {str((cell or {}).get("grade") or "") for cell in lookup.values()}
    return TOP_GRADES if grades & TOP_GRADES else PASSING_GRADES

#: S2: the "hide the first 30 minutes" switch (machine-local, default on).
SETTING_FIRST30 = "alert_show_hide_first30"
DEFAULT_FIRST30 = True
FIRST30_LABEL = "Hide first 30 min"
#: The `hidden_by_show` detail reason for a first-30 hide.
REASON_FIRST30 = "first30"
_ET_NAME = "America/New_York"
#: The `hidden_by_show` detail reason for a longs-off hide (`longs_market_gate.REASON`).
REASON_LONGS_OFF = "longs_off"
_FIRST30_START = dt_time(9, 30)
_FIRST30_END = dt_time(10, 0)

# How often the typed-name lookup re-stats longs.txt / shorts.txt (seconds).
_STAT_INTERVAL_SECONDS = 30.0
_lock = threading.Lock()
_typed_cache: dict = {"paths": None, "stamp": None, "checked": 0.0, "names": frozenset()}


def mode() -> str:
    """The saved choice; unreadable or unknown keeps the default."""
    try:
        value = str(project_paths.get_local_setting(SETTING_SHOW_FILTER, DEFAULT_MODE) or "")
    except Exception:  # noqa: BLE001 - a preference read never costs a surface
        return DEFAULT_MODE
    return value if value in {key for key, _label in MODES} else DEFAULT_MODE


def set_mode(value: str) -> None:
    value = str(value or "")
    if value not in {key for key, _label in MODES}:
        value = DEFAULT_MODE
    project_paths.save_local_setting(SETTING_SHOW_FILTER, value)


def first30_enabled() -> bool:
    """The saved first-30 switch; unreadable keeps the default (on)."""
    try:
        value = project_paths.get_local_setting(SETTING_FIRST30, DEFAULT_FIRST30)
    except Exception:  # noqa: BLE001 - a preference read never costs a surface
        return DEFAULT_FIRST30
    return value if isinstance(value, bool) else DEFAULT_FIRST30


def set_first30_enabled(enabled: bool) -> None:
    project_paths.save_local_setting(SETTING_FIRST30, bool(enabled))


def alert_time(alert: Any) -> datetime | None:
    """The alert's timezone-aware receive time, or None (unknown)."""
    when = getattr(alert, "received_at", None)
    if isinstance(when, datetime) and when.tzinfo is not None and when.utcoffset() is not None:
        return when
    return None


def in_first30(when: datetime | None) -> bool:
    """True for 09:30:00 <= ET time < 10:00:00; a naive or missing time is False."""
    if not isinstance(when, datetime) or when.tzinfo is None or when.utcoffset() is None:
        return False
    from zoneinfo import ZoneInfo

    clock = when.astimezone(ZoneInfo(_ET_NAME)).time()
    return _FIRST30_START <= clock < _FIRST30_END


def daytrade_grade(lookup: Mapping[str, Any] | None, alert: Any) -> str | None:
    """The M5 row's grade (`setup_grades`), or None before grades load."""
    if not lookup:
        return None
    import working_lately

    payload = getattr(alert, "payload", None)
    feedback = payload.get("feedback") if isinstance(payload, dict) else None
    bounce_types = str((feedback or {}).get("bounce_types") or "")
    if not bounce_types:
        bounce_types = working_lately.alert_priority_key(alert)[0]
    side = str(getattr(alert, "side", "") or "")
    return setup_grades.daytrade_grade_for_alert(lookup, bounce_types, side)


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return {"BUY": "LONG", "SELL": "SHORT", "L": "LONG", "S": "SHORT"}.get(text, text)


def best_keys(entries: Iterable[Any]) -> frozenset:
    """`{(SYMBOL, SIDE)}` from Best-right-now entries (objects or mappings)."""
    keys = set()
    for entry in entries or ():
        get = entry.get if isinstance(entry, Mapping) else (lambda k, e=entry: getattr(e, k, ""))
        symbol = str(get("symbol") or "").strip().upper()
        if symbol:
            keys.add((symbol, _side(get("side"))))
    return frozenset(keys)


def on_best(keys: frozenset, symbol: Any, side: Any) -> bool:
    symbol = str(symbol or "").strip().upper()
    return (symbol, _side(side)) in keys or (symbol, "") in keys


def hide_reason(
    show_mode: str,
    *,
    grade: str | None,
    best: frozenset | None,
    symbol: Any,
    side: Any,
    privileged: bool,
    first30: bool = False,
    when: datetime | None = None,
    bypass: frozenset | None = None,
    longs_off: bool = False,
) -> str:
    """Why this M5 row is hidden: `longs_off`, `first30`, the Show mode, or "" (shows).

    Privileged rows always show; so do `bypass` grades (default A and up, see
    `bypass_grades`) and unknown-grade rows under the first-30 switch. Unknown
    grade / unranked Best shows under the Show mode. ``longs_off`` = the gate
    says longs are off for this row; it hides a LONG whatever its grade.
    """
    if privileged:
        return ""
    if longs_off and _side(side) == "LONG":
        return REASON_LONGS_OFF
    exempt = TOP_GRADES if bypass is None else bypass
    if first30 and in_first30(when) and grade is not None and grade not in exempt:
        return REASON_FIRST30
    if show_mode == ALL:
        return ""
    if show_mode == BEST_NOW:
        return show_mode if best is not None and not on_best(best, symbol, side) else ""
    if grade is None:
        return ""
    return show_mode if grade not in PASSING_GRADES else ""


def hides(
    show_mode: str,
    *,
    grade: str | None,
    best: frozenset | None,
    symbol: Any,
    side: Any,
    privileged: bool,
    first30: bool = False,
    when: datetime | None = None,
    bypass: frozenset | None = None,
    longs_off: bool = False,
) -> bool:
    """True when this M5 row is hidden (see `hide_reason`)."""
    return bool(
        hide_reason(
            show_mode,
            grade=grade,
            best=best,
            symbol=symbol,
            side=side,
            privileged=privileged,
            first30=first30,
            when=when,
            bypass=bypass,
            longs_off=longs_off,
        )
    )


def hidden_text(count: int, new: int = 0, *, first30: int = 0) -> str:
    """"N hidden by Show filter (M New[, K first30])"; "" for none."""
    count = int(count or 0)
    if not count:
        return ""
    first30 = int(first30 or 0)
    extra = f", {first30} {REASON_FIRST30}" if first30 else ""
    return f"{count} hidden by Show filter ({int(new or 0)} New{extra})"


def count_hidden(verdicts: Iterable[tuple[Any, bool, bool]]) -> tuple[int, int]:
    """`(rows, new)` over `(key, hidden, is_new)`; one row per key, first verdict wins."""
    seen: dict = {}
    for key, hidden, is_new in verdicts:
        if key not in seen:
            seen[key] = (bool(hidden), bool(is_new))
    rows = [value for value in seen.values() if value[0]]
    return len(rows), sum(1 for _hidden, is_new in rows if is_new)


def _typed_paths() -> tuple[Path, Path]:
    return (Path(project_paths.LONGS_FILE), Path(project_paths.SHORTS_FILE))


def typed_symbols() -> frozenset:
    """Names in longs.txt / shorts.txt, re-read on a stat change (30 s throttle)."""
    from watchlist_utils import read_watchlist_symbols

    paths = _typed_paths()
    now = time.monotonic()
    with _lock:
        if (
            _typed_cache["paths"] == paths
            and _typed_cache["stamp"] is not None
            and now - _typed_cache["checked"] < _STAT_INTERVAL_SECONDS
        ):
            return _typed_cache["names"]
    stamp = []
    for path in paths:
        try:
            stat = path.stat()
            stamp.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            stamp.append(None)
    stamp = tuple(stamp)
    with _lock:
        if _typed_cache["paths"] == paths and _typed_cache["stamp"] == stamp:
            _typed_cache["checked"] = now
            return _typed_cache["names"]
    names = set()
    for path, part in zip(paths, stamp, strict=True):
        if part is not None:
            names.update(str(s).strip().upper() for s in read_watchlist_symbols(path))
    names = frozenset(name for name in names if name)
    with _lock:
        _typed_cache.update(paths=paths, stamp=stamp, checked=now, names=names)
    return names


def clear_cache() -> None:
    with _lock:
        _typed_cache.update(paths=None, stamp=None, checked=0.0, names=frozenset())
