"""The AVWAPE quick test files: the scan publishes, the Setup Tracker reads.

`avwape_quick_test` is the pure rule; this is its only I/O. The scan runner is the one writer
of `AVWAPE_QUICK_TEST_FILE` (this scan's rows) and `AVWAPE_QUICK_TEST_HISTORY_FILE` (every scan
session's rows, settled for grading). Both are written whole and atomically, so a failed
publish leaves the last good file in place. Readers get None / [] for a missing or unreadable file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import avwape_quick_test
from long_setups_store import read_earnings_dates

SCHEMA_VERSION = 1


def _paths(path: Path | None, history_path: Path | None) -> tuple[Path, Path]:
    import project_paths

    return (Path(path or project_paths.AVWAPE_QUICK_TEST_FILE),
            Path(history_path or project_paths.AVWAPE_QUICK_TEST_HISTORY_FILE))


def read_avwape_quick_test(path: Path | None = None) -> dict[str, Any] | None:
    """The last published scan's payload, or None."""
    target, _history = _paths(path, None)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else None


def read_history(path: Path | None = None) -> list[dict[str, Any]]:
    """Every scan session's rows (settled or not), or [] when there is no readable file."""
    _target, history = _paths(None, path)
    try:
        payload = json.loads(history.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = payload.get("rows") if isinstance(payload, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def publish_avwape_quick_test(
    *,
    bars_by_symbol: Mapping[str, Any],
    spy_bars: Any,
    feature_rows: Iterable[Mapping[str, Any]] = (),
    atr_by_symbol: Mapping[str, Any] | None = None,
    market_cap_by_symbol: Mapping[str, Any] | None = None,
    as_of: Any = None,
    now: datetime | None = None,
    path: Path | None = None,
    history_path: Path | None = None,
    earnings_dates_path: Path | None = None,
) -> dict[str, Any]:
    """Build this scan's quick-test rows, settle the history, and write both files.

    Nothing is written without a scan session (``as_of``). A scan where no name has a bar for
    the session keeps the last good file. Returns what was written (or kept).
    """
    from diagnostics.artifact_io import atomic_write_json

    target, history = _paths(path, history_path)
    payload = avwape_quick_test.build_rows(
        bars_by_symbol=bars_by_symbol, spy_bars=spy_bars,
        earnings_dates_by_symbol=read_earnings_dates(earnings_dates_path),
        atr_by_symbol=atr_by_symbol, market_cap_by_symbol=market_cap_by_symbol,
        feature_rows=list(feature_rows or ()), as_of=as_of)
    if not payload["as_of"]:
        logging.info("AVWAPE quick test: no completed scan session; nothing published.")
        return payload
    has_current_bar = any(
        isinstance(bars, (list, tuple)) and bars and isinstance(bars[-1], Mapping)
        and str(bars[-1].get("date") or "")[:10] == payload["as_of"]
        for bars in (bars_by_symbol or {}).values()
    )
    previous = read_avwape_quick_test(target)
    if not payload["rows"] and not has_current_bar and previous and previous.get("rows"):
        logging.warning("AVWAPE quick test: no name has a bar for %s; the last good file (%s) is kept.",
                        payload["as_of"], previous.get("as_of"))
        return previous
    payload = {"schema_version": SCHEMA_VERSION,
               "generated_at": (now or datetime.now()).isoformat(timespec="seconds"), **payload}
    atomic_write_json(target, payload)
    settled = avwape_quick_test.settle(read_history(history), bars_by_symbol, spy_bars)
    rows = avwape_quick_test.upsert_history(settled, payload["rows"])
    atomic_write_json(history, {"schema_version": SCHEMA_VERSION, "rows": rows}, indent=None)
    logging.info("AVWAPE quick test: %s row(s) for %s.", len(payload["rows"]), payload["as_of"])
    return payload
