"""The long-setups files (p9): the scan publishes, the desk and the phone report read.

`long_setups` is the pure rule; this is its only I/O. The scan runner is the one writer of
`LONG_SETUPS_FILE` (this scan's rows) and `LONG_SETUPS_HISTORY_FILE` (every scan session's
rows, settled for grading). Both are written whole and atomically, so a failed publish
leaves the last good file in place. Readers get None / [] for a missing or unreadable file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import long_setups

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


def publish_long_setups(
    *,
    bars_by_symbol: Mapping[str, Any],
    spy_bars: Any,
    feature_rows: Iterable[Mapping[str, Any]],
    earnings_by_symbol: Mapping[str, Mapping[str, Any]] | None = None,
    atr_by_symbol: Mapping[str, Any] | None = None,
    sector_by_symbol: Mapping[str, Any] | None = None,
    as_of: Any = None,
    now: datetime | None = None,
    path: Path | None = None,
    history_path: Path | None = None,
) -> dict[str, Any]:
    """Build this scan's long setups, settle the history, and write both files.

    Returns the published payload. Nothing is written without a scan session (``as_of``).
    """
    from diagnostics.artifact_io import atomic_write_json

    target, history = _paths(path, history_path)
    payload = long_setups.build_rows(
        bars_by_symbol=bars_by_symbol, spy_bars=spy_bars, feature_rows=feature_rows,
        earnings_by_symbol=earnings_by_symbol, atr_by_symbol=atr_by_symbol,
        sector_by_symbol=sector_by_symbol, as_of=as_of,
    )
    if not payload["as_of"]:
        logging.info("Long setups: no completed scan session; nothing published.")
        return payload
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
