"""Exit-window truth (S11, finding F14): when does an M5 alert's move usually end?

Per (bounce type, side) over the outcome log: the share of alerts peaking within
30/60/120 minutes, mean MFE by 60 and 120 minutes and at the close, the give-back
when +1R printed, and the EV of five exit rules with a 1R stop.

Facts only. Nothing here is read by a detector, a score, an alert decision or a
filter; the desk formats these numbers and never enforces them. The night slot
`ai_jobs.exit_windows_night` builds the file; the desk reads it off the Qt thread
(`warm_cache` on a background thread, then memory-only lookups).

Definitions (one alert = one event id with a final row and a settled close R):

* peak minute - the first row whose MFE reaches the alert's largest MFE;
* MFE by T - the MFE on the alert's last row at or before T minutes;
* give-back - final MFE minus final close R, for alerts with any +1R row;
* rules, R per alert: a stop row at or before the exit is -1R. ``hold_to_close``
  is the close R; ``exit_60m`` the close R at 60 min; ``t1_or_60m`` /
  ``t1_or_120m`` the first +1R-or-stop row inside the window (stop wins a tie),
  else the close R at the time stop; ``bracket_1to1`` the first +1R-or-stop row
  all day, else the close R.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

SCHEMA = "exit_windows_v1"
#: Sessions of outcome log read (~2 months; F14 used 08-01 to 09-25).
WINDOW_SESSIONS = 40
#: Rows per pandas chunk: the live log is ~300 MB.
CHUNK_ROWS = 250_000
PEAK_WINDOWS = (30, 60, 120)
RULES = ("hold_to_close", "exit_60m", "t1_or_60m", "t1_or_120m", "bracket_1to1")
USECOLS = (
    "event_id", "event_type", "trade_date", "direction", "entry_price",
    "risk_per_share", "bars_elapsed", "minutes_elapsed", "close_r", "mfe_r",
    "target_1r_hit", "stop_hit", "eod_close",
)
WARM_INTERVAL_SECONDS = 60.0


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _r4(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _mean(values: list[float]) -> float | None:
    return _r4(sum(values) / len(values)) if values else None


# ------------------------------------------------------------------ per alert
def alert_result(rows: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """One alert's numbers, or None when it is open or its close R is unsettled."""
    import setup_grades

    ordered = [row for _i, row in sorted(
        enumerate(rows), key=lambda item: (int(_num(item[1].get("bars_elapsed")) or 0), item[0])
    )]
    finals = [row for row in ordered if str(row.get("event_type") or "").strip().lower() == "final"]
    if not finals:
        return None
    eod_r = setup_grades._eod_r(finals[-1])
    if eod_r is None:
        return None
    steps = []
    for row in ordered:
        minutes, close_r, mfe_r = (_num(row.get(k)) for k in ("minutes_elapsed", "close_r", "mfe_r"))
        if minutes is None or mfe_r is None:
            continue
        steps.append((minutes, close_r, mfe_r, _flag(row.get("target_1r_hit")), _flag(row.get("stop_hit"))))
    if not steps:
        return None
    best = max(step[2] for step in steps)
    peak_minute = next(step[0] for step in steps if step[2] >= best - 1e-9)

    def last_at(limit: float):
        inside = [step for step in steps if step[0] <= limit]
        return inside[-1] if inside else None

    def stopped_by(limit: float) -> bool:
        return any(step[4] for step in steps if step[0] <= limit)

    def first_decisive(limit: float) -> float | None:
        for step in steps:
            if step[0] > limit:
                break
            if step[4]:
                return -1.0
            if step[3]:
                return 1.0
        return None

    def time_exit(limit: float) -> float:
        row = last_at(limit)
        close = row[1] if row is not None and row[1] is not None else eod_r
        return close

    all_day = float("inf")
    ev = {
        "hold_to_close": -1.0 if stopped_by(all_day) else eod_r,
        "exit_60m": -1.0 if stopped_by(60) else time_exit(60),
        "t1_or_60m": first_decisive(60) if first_decisive(60) is not None else time_exit(60),
        "t1_or_120m": first_decisive(120) if first_decisive(120) is not None else time_exit(120),
        "bracket_1to1": first_decisive(all_day) if first_decisive(all_day) is not None else eod_r,
    }
    hit_1r = any(step[3] for step in steps)
    mfe_60, mfe_120 = last_at(60), last_at(120)
    return {
        "peak_minute": peak_minute,
        "mfe_60": mfe_60[2] if mfe_60 else None,
        "mfe_120": mfe_120[2] if mfe_120 else None,
        "mfe_close": best,
        "hit_1r": hit_1r,
        "give_back": (best - eod_r) if hit_1r else None,
        "eod_r": eod_r,
        "ev": ev,
    }


def _cell(key: str, bounce_type: str, side: str, results: list[dict], dates: list[str]) -> dict[str, Any]:
    n = len(results)
    hits = [item for item in results if item["hit_1r"]]
    gives = [item["give_back"] for item in hits]
    return {
        "key": key,
        "bounce_type": bounce_type,
        "side": side,
        "n": n,
        "sessions": len(set(dates)),
        "peak_within": {
            str(limit): _r4(sum(1 for item in results if item["peak_minute"] <= limit) / n)
            for limit in PEAK_WINDOWS
        },
        "mfe_by": {
            "60": _mean([item["mfe_60"] for item in results if item["mfe_60"] is not None]),
            "120": _mean([item["mfe_120"] for item in results if item["mfe_120"] is not None]),
            "close": _mean([item["mfe_close"] for item in results]),
        },
        "hit_1r_n": len(hits),
        "give_back_median": _r4(median(gives)) if gives else None,
        "give_back_mean": _mean(gives),
        "hit_1r_closed_le_0": _r4(sum(1 for item in hits if item["eod_r"] <= 0) / len(hits)) if hits else None,
        "ev": {rule: _mean([item["ev"][rule] for item in results]) for rule in RULES},
    }


def build_payload(
    rows: Iterable[Mapping[str, Any]], *, as_of: str = "", window: tuple[str, str] = ("", "")
) -> dict[str, Any]:
    """The file's payload from outcome rows. Pure: no file, no clock."""
    from held_run_score import bounce_components
    from setup_scoreboard import bounce_type_from_event_id

    by_event: dict[str, list] = defaultdict(list)
    for row in rows or ():
        event_id = str(row.get("event_id") or "").strip()
        if event_id:
            by_event[event_id].append(row)
    groups: dict[tuple[str, str], tuple[list, list]] = {}
    alerts = 0
    for event_id, event_rows in by_event.items():
        result = alert_result(event_rows)
        if result is None:
            continue
        alerts += 1
        first = event_rows[0]
        side = str(first.get("direction") or "").strip().upper()
        trade_date = str(first.get("trade_date") or "").strip()
        keys = [(part, side) for part in bounce_components(bounce_type_from_event_id(event_id))]
        keys.append(("all", side))
        for key in dict.fromkeys(keys):
            bucket = groups.setdefault(key, ([], []))
            bucket[0].append(result)
            bucket[1].append(trade_date)
    cells = [
        _cell(f"{bounce_type}|{side}", bounce_type, side, results, dates)
        for (bounce_type, side), (results, dates) in sorted(groups.items())
    ]
    return {
        "schema": SCHEMA,
        "as_of": str(as_of or ""),
        "window": {"start": window[0], "end": window[1], "sessions": WINDOW_SESSIONS},
        "alerts": alerts,
        "rules": (
            "1R stop on every rule. hold_to_close: close R; exit_60m: close R at 60 min; "
            "t1_or_60m/t1_or_120m: +1R or stop first inside the window, else close R at the time; "
            "bracket_1to1: +1R or stop first all day, else close R. Stop wins a same-row tie."
        ),
        "cells": cells,
    }


# ------------------------------------------------------------------ the read
def read_window_rows(path: Path, window: tuple[str, str]) -> list[dict[str, str]]:
    """The window's rows of the outcome log, read in chunks with only the needed columns.

    Raises when the file is missing or unreadable (the slot records the failure).
    """
    import pandas as pd

    start, end = window
    kept: list[dict[str, str]] = []
    reader = pd.read_csv(
        Path(path), usecols=list(USECOLS), chunksize=CHUNK_ROWS, dtype=str,
        keep_default_na=False, encoding="utf-8",
    )
    for chunk in reader:
        dates = chunk["trade_date"].str.strip()
        chunk = chunk[(dates >= start) & (dates <= end)]
        if not chunk.empty:
            kept.extend(chunk.to_dict("records"))
    return kept


def write_payload(payload: Mapping[str, Any], path: Path) -> Path:
    """Temp file then replace, so a half-written file never replaces the last good one."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temp, target)
    return target


def read_payload(path: Path | None = None) -> dict[str, Any]:
    """The last published payload, `{}` when absent or unreadable. Never on the Qt thread."""
    target = Path(path) if path is not None else _default_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) and payload.get("schema") == SCHEMA else {}


def _default_path() -> Path:
    from project_paths import EXIT_WINDOWS_FILE

    return Path(EXIT_WINDOWS_FILE)


# ------------------------------------------------------------------ lookups
def lookup(payload: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    return {str(cell.get("key")): cell for cell in (payload or {}).get("cells") or ()}


def key_for(bounce_type: Any, side: Any) -> str:
    return f"{str(bounce_type or '').strip().lower()}|{str(side or '').strip().upper()}"


def cell_for_alert(
    cells: Mapping[str, Mapping[str, Any]], bounce_types: Any, side: Any
) -> Mapping[str, Any] | None:
    """The cell of the alert's bounce type with the most alerts, or None."""
    from held_run_score import bounce_components

    best = None
    for part in str(bounce_types or "").replace(";", "-").split("-"):
        for component in bounce_components(part) or ():
            cell = cells.get(key_for(component, side))
            if cell and (best is None or int(cell.get("n") or 0) > int(best.get("n") or 0)):
                best = cell
    return best


# ------------------------------------------------------------------ words
def _signed(value: Any) -> str:
    number = _num(value)
    return "?" if number is None else f"{number:+.2f}R"


def tracker_text(cell: Mapping[str, Any] | None) -> str:
    """The Daytrade Tracker's "Exit by" cell: "peak <= 60 min 49%; +1R/60m -0.05R vs hold -0.30R"."""
    if not cell:
        return ""
    share = _num((cell.get("peak_within") or {}).get("60"))
    ev = cell.get("ev") or {}
    peak = f"peak <= 60 min {share * 100:.0f}%" if share is not None else "peak unknown"
    return f"{peak}; +1R/60m {_signed(ev.get('t1_or_60m'))} vs hold {_signed(ev.get('hold_to_close'))}"


def alert_line(cell: Mapping[str, Any] | None) -> str:
    """One line for the M5 row hover and the chart review header."""
    if not cell:
        return ""
    within = cell.get("peak_within") or {}
    usual = next(
        (limit for limit in PEAK_WINDOWS if (_num(within.get(str(limit))) or 0.0) >= 0.5), None
    )
    peak = (
        f"this family usually peaks inside {usual} min" if usual is not None
        else f"this family usually peaks after {PEAK_WINDOWS[-1]} min"
    )
    ev = cell.get("ev") or {}
    target, hold = _num(ev.get("t1_or_60m")), _num(ev.get("hold_to_close"))
    if target is None or hold is None:
        versus = "exit rules unmeasured"
    elif target >= hold:
        versus = f"+1R or 60 min has beaten holding by {target - hold:.2f}R"
    else:
        versus = f"holding has beaten +1R or 60 min by {hold - target:.2f}R"
    return f"Exit by: {peak}; {versus} (n {int(cell.get('n') or 0)})"


# ------------------------------------------------------------------ desk cache
_lock = threading.Lock()
_cache: dict[str, Any] = {"signature": None, "cells": {}, "warmed_at": None, "warming": False}


def _signature(target: Path) -> tuple:
    try:
        stat = target.stat()
        return (str(target), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(target), None, None)


def warm_cache(path: Path | None = None) -> bool:
    """Re-read the file when it moved. True when the cells changed. Never on the Qt thread."""
    target = Path(path) if path is not None else _default_path()
    signature = _signature(target)
    with _lock:
        stale = signature != _cache["signature"]
    changed = False
    if stale:
        cells = lookup(read_payload(target))
        with _lock:
            changed = cells != _cache["cells"]
            _cache["signature"] = signature
            _cache["cells"] = cells
    with _lock:
        _cache["warmed_at"] = time.monotonic()
    return changed


def request_warm() -> None:
    """Start one background read unless one is running or ran in the last minute."""
    with _lock:
        last = _cache["warmed_at"]
        if _cache["warming"] or (last is not None and time.monotonic() - last < WARM_INTERVAL_SECONDS):
            return
        _cache["warming"] = True

    def _run() -> None:
        try:
            warm_cache()
        except Exception:  # noqa: BLE001 - a fact line never costs the desk
            logging.warning("Exit windows warm failed.", exc_info=True)
        finally:
            with _lock:
                _cache["warming"] = False
                if _cache["warmed_at"] is None:
                    _cache["warmed_at"] = time.monotonic()

    threading.Thread(target=_run, name="exit-windows-warm", daemon=True).start()


def cached_lookup() -> dict[str, Mapping[str, Any]]:
    """The cells already in memory. Never opens a file."""
    with _lock:
        return dict(_cache["cells"])


def cached_alert_line(bounce_types: Any, side: Any) -> str:
    """The alert line from memory only ("" when unknown); kicks a background warm."""
    request_warm()
    return alert_line(cell_for_alert(cached_lookup(), bounce_types, side))


def reset_cache_for_tests() -> None:
    with _lock:
        _cache.update(signature=None, cells={}, warmed_at=None, warming=False)


def line_for_alert(alert: Any) -> str:
    """The alert line for an M5 alert object from memory only ("" when unknown)."""
    payload = getattr(alert, "payload", None)
    feedback = payload.get("feedback") if isinstance(payload, Mapping) else None
    bounce_types = str((feedback or {}).get("bounce_types") or "") if isinstance(feedback, Mapping) else ""
    if not bounce_types:
        import working_lately

        bounce_types = working_lately.alert_priority_key(alert)[0]
    return cached_alert_line(bounce_types, getattr(alert, "side", ""))


def mentor_quote(cells: Mapping[str, Mapping[str, Any]], setup: Any, side: Any) -> str:
    """The Trade Mentor's exit-question quote: the setup's family cell, else all M5 alerts on that side."""
    cell = cell_for_alert(cells, setup, side) if str(setup or "").strip() else None
    name = str(cell.get("bounce_type")) if cell else "all"
    cell = cell or cells.get(key_for("all", side))
    if not cell:
        return ""
    return (
        f"Fact, not a rule: {name} {str(side or '').strip().upper()} M5 alerts - "
        f"{tracker_text(cell)} (n {int(cell.get('n') or 0)})."
    )
