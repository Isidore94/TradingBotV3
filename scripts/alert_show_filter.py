"""The Alert Center's "Show" filter for M5 rows (P9, display only).

Three choices: All, Grade B and up, Best right now. It hides M5 rows from what
the trader SEES; every alert is still recorded, still reaches the review-queue
door, the Working-now strip and the evidence files. Rows on names the trader
typed, Focus names, armed watches, price alerts and regime-pause rows always
show. Unknown (grades not loaded, Best list not ranked yet) shows.
"""

from __future__ import annotations

import threading
import time
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


def hides(
    show_mode: str,
    *,
    grade: str | None,
    best: frozenset | None,
    symbol: Any,
    side: Any,
    privileged: bool,
) -> bool:
    """True when this M5 row is hidden. Unknown grade / unranked Best shows."""
    if privileged or show_mode == ALL:
        return False
    if show_mode == BEST_NOW:
        return best is not None and not on_best(best, symbol, side)
    if grade is None:
        return False
    return grade not in PASSING_GRADES


def hidden_text(count: int, new: int = 0) -> str:
    """"N hidden by Show filter (M New)"; "" for none."""
    count = int(count or 0)
    if not count:
        return ""
    return f"{count} hidden by Show filter ({int(new or 0)} New)"


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
