"""The long-setups files (p9): the scan publishes, the desk and the phone report read.

`long_setups` is the pure rule; this is its only I/O. The scan runner is the one writer of
`LONG_SETUPS_FILE` (this scan's rows) and `LONG_SETUPS_HISTORY_FILE` (every scan session's
rows, settled for grading) and `RUNNER_DIP_WATCH_FILE` (the runner dip watch). All are written whole and atomically, so a failed publish
leaves the last good file in place. Readers get None / [] for a missing or unreadable file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import long_setups
import runner_dip_watch

SCHEMA_VERSION = 1


def _paths(path: Path | None, history_path: Path | None) -> tuple[Path, Path]:
    import project_paths

    return (Path(path or project_paths.LONG_SETUPS_FILE),
            Path(history_path or project_paths.LONG_SETUPS_HISTORY_FILE))


def read_long_setups(path: Path | None = None) -> dict[str, Any] | None:
    """The last published scan's payload, or None."""
    target, _history = _paths(path, None)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else None


def read_history(history_path: Path | None = None) -> list[dict[str, Any]]:
    """Every scan session's rows (settled or not), or [] when there is no readable file."""
    _target, history = _paths(None, history_path)
    try:
        payload = json.loads(history.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = payload.get("rows") if isinstance(payload, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def read_earnings_dates(path: Path | None = None) -> dict[str, list[str]]:
    """``{SYMBOL: [iso dates, oldest first]}`` from the earnings-dates cache; {} when unreadable."""
    import project_paths

    try:
        payload = json.loads(Path(path or project_paths.EARNINGS_DATES_CACHE_FILE).read_text(encoding="utf-8"))
        symbols = payload.get("symbols") if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return {}
    out: dict[str, list[str]] = {}
    for symbol, entry in (symbols if isinstance(symbols, dict) else {}).items():
        days = set()
        dates = entry.get("dates") if isinstance(entry, dict) else None
        for text in dates if isinstance(dates, list) else ():
            try:
                days.add(datetime.fromisoformat(str(text)[:10]).date().isoformat())
            except ValueError:
                continue
        if days:
            out[str(symbol).strip().upper()] = sorted(days)
    return out


def publish_long_setups(
    *,
    bars_by_symbol: Mapping[str, Any],
    spy_bars: Any,
    feature_rows: Iterable[Mapping[str, Any]],
    earnings_by_symbol: Mapping[str, Mapping[str, Any]] | None = None,
    atr_by_symbol: Mapping[str, Any] | None = None,
    sector_by_symbol: Mapping[str, Any] | None = None,
    market_cap_by_symbol: Mapping[str, Any] | None = None,
    as_of: Any = None,
    now: datetime | None = None,
    path: Path | None = None,
    history_path: Path | None = None,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Build this scan's long setups, settle the history, and write both files.

    Returns the published payload. Nothing is written without a scan session (``as_of``).
    The earnings-dates cache (`read_earnings_dates`) anchors each leader pullback's earnings AVWAP.
    The same inputs then publish the runner dip watch (`publish_runner_dip_watch`).
    """
    earnings_dates = read_earnings_dates()
    inputs = dict(bars_by_symbol=bars_by_symbol, spy_bars=spy_bars, feature_rows=list(feature_rows or ()),
                  earnings_by_symbol=earnings_by_symbol, atr_by_symbol=atr_by_symbol,
                  sector_by_symbol=sector_by_symbol, market_cap_by_symbol=market_cap_by_symbol,
                  earnings_dates_by_symbol=earnings_dates, as_of=as_of)
    payload = _publish_rows(inputs, now=now, path=path, history_path=history_path)
    # The runner dip watch is its own file: a failure there never costs the long setups.
    try:
        publish_runner_dip_watch(inputs, now=now, path=runner_path)
    except Exception:  # noqa: BLE001 - the last good runner file stays
        logging.warning("Runner dip watch not published for this scan.", exc_info=True)
    return payload


def _publish_rows(inputs: Mapping[str, Any], *, now: datetime | None, path: Path | None,
                  history_path: Path | None) -> dict[str, Any]:
    from diagnostics.artifact_io import atomic_write_json

    target, history = _paths(path, history_path)
    bars_by_symbol, spy_bars = inputs["bars_by_symbol"], inputs["spy_bars"]
    payload = long_setups.build_rows(**inputs)
    if not payload["as_of"]:
        logging.info("Long setups: no completed scan session; nothing published.")
        return payload
    has_current_bar = any(
        isinstance(bars, (list, tuple)) and bars and isinstance(bars[-1], Mapping)
        and str(bars[-1].get("date") or "")[:10] == payload["as_of"]
        for bars in (bars_by_symbol or {}).values()
    )
    previous = read_long_setups(target)
    if not payload["rows"] and not has_current_bar and previous and previous.get("rows"):
        logging.warning("Long setups: no name has a bar for %s; the last good file (%s) is kept.",
                        payload["as_of"], previous.get("as_of"))
        return previous
    payload = {"schema_version": SCHEMA_VERSION,
               "generated_at": (now or datetime.now()).isoformat(timespec="seconds"), **payload}
    atomic_write_json(target, payload)
    settled = long_setups.settle(read_history(history), bars_by_symbol, spy_bars)
    rows = long_setups.upsert_history(settled, payload["rows"])
    atomic_write_json(history, {"schema_version": SCHEMA_VERSION, "rows": rows}, indent=None)
    logging.info("Long setups: %s row(s) for %s, %s promoted (market working: %s).",
                 len(payload["rows"]), payload["as_of"],
                 sum(1 for row in payload["rows"] if row.get("promoted")), payload["market_working"])
    return payload


def read_runner_dip_watch(path: Path | None = None) -> dict[str, Any] | None:
    """The last published runner dip watch, or None."""
    import project_paths

    try:
        payload = json.loads(Path(path or project_paths.RUNNER_DIP_WATCH_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("members"), list) else None


def publish_runner_dip_watch(inputs: Mapping[str, Any], *, now: datetime | None = None,
                             path: Path | None = None) -> dict[str, Any] | None:
    """Build the runner dip watch from the scan's long-setups inputs and write it whole, atomically.

    No scan session writes nothing; a scan where no name has a bar for the session keeps the last
    good file. Returns what was written (or kept), None when nothing was.
    """
    import project_paths
    from diagnostics.artifact_io import atomic_write_json

    target = Path(path or project_paths.RUNNER_DIP_WATCH_FILE)
    payload = runner_dip_watch.build_members(**inputs)
    if not payload["as_of"]:
        return None
    has_current_bar = any(
        isinstance(bars, (list, tuple)) and bars and isinstance(bars[-1], Mapping)
        and str(bars[-1].get("date") or "")[:10] == payload["as_of"]
        for bars in (inputs.get("bars_by_symbol") or {}).values()
    )
    previous = read_runner_dip_watch(target)
    if not payload["members"] and not has_current_bar and previous and previous.get("members"):
        logging.warning("Runner dip watch: no name has a bar for %s; the last good file (%s) is kept.",
                        payload["as_of"], previous.get("as_of"))
        return previous
    payload = {"schema_version": runner_dip_watch.SCHEMA_VERSION,
               "generated_at": (now or datetime.now()).isoformat(timespec="seconds"), **payload}
    atomic_write_json(target, payload)
    logging.info("Runner dip watch: %s runner(s) for %s, %s armed (market working: %s).",
                 len(payload["members"]), payload["as_of"], len(payload["armed"]), payload["market_working"])
    return payload
