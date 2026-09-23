"""Append-only, per-session history of the auto market regime at the open.

`autopilot_core.record_opening_environment` calls in here. Two row sources:

- `first_read`: the session's first regime label seen, directional or not
  (the regime before the first shift, which the opening file never keeps).
- `opening_anchor`: the directional label written to `AUTO_OPENING_ENV_FILE`.

`opening_regime_for(session_date)` answers "what was the regime at the open".
Evidence only: nothing here feeds discovery, a detector, a score or an alert.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

SCHEMA = "opening_regime_history_v1"
SOURCE_FIRST_READ = "first_read"
SOURCE_OPENING_ANCHOR = "opening_anchor"

_LOCK = threading.Lock()
#: `(path, session_date, source)` already on file, so the per-scan call is cheap.
_SEEN: set[tuple[str, str, str]] = set()
#: Paths whose file has been read into `_SEEN` this process.
_LOADED: set[str] = set()
_LOG = logging.getLogger(__name__)


def default_history_path() -> Path:
    from project_paths import AUTO_OPENING_REGIME_HISTORY_FILE

    return Path(AUTO_OPENING_REGIME_HISTORY_FILE)


def _aware(when: datetime | None) -> datetime:
    if when is None:
        return datetime.now().astimezone()
    return when if when.tzinfo is not None else when.astimezone()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load(path: Path) -> None:
    key = str(path)
    if key in _LOADED:
        return
    for row in _read_rows(path):
        _SEEN.add((key, str(row.get("session_date") or ""), str(row.get("source") or "")))
    _LOADED.add(key)


def record(
    session_date: date | str,
    label: str,
    source: str,
    *,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> bool:
    """Append one row unless that session already has a row from `source`.

    Returns True when a row was written. Raises on I/O failure; the caller
    logs it and carries on.
    """
    label = str(label or "").strip().lower()
    if not label:
        return False  # no read is unknown, not a regime
    target = Path(path) if path is not None else default_history_path()
    session = session_date.isoformat() if isinstance(session_date, date) else str(session_date)[:10]
    key = (str(target), session, str(source))
    with _LOCK:
        _load(target)
        if key in _SEEN:
            return False
        row = {
            "schema": SCHEMA,
            "session_date": session,
            "label": label,
            "written_at": _aware(now).isoformat(timespec="seconds"),
            "source": str(source),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        _SEEN.add(key)
    return True


def opening_regime_for(
    session_date: date | str,
    *,
    path: Path | str | None = None,
) -> dict[str, Any] | None:
    """That session's opening regime, or None when nothing was recorded.

    `label`/`written_at`/`source` come from the session's earliest row (the
    first read when one exists). `directional_anchor` is the directional label
    kept for the day, "" when the day never went directional.
    """
    target = Path(path) if path is not None else default_history_path()
    session = session_date.isoformat() if isinstance(session_date, date) else str(session_date)[:10]
    rows = [row for row in _read_rows(target) if str(row.get("session_date") or "") == session]
    if not rows:
        return None
    first = next((row for row in rows if row.get("source") == SOURCE_FIRST_READ), rows[0])
    anchor = next((row for row in rows if row.get("source") == SOURCE_OPENING_ANCHOR), None)
    return {
        "session_date": session,
        "label": str(first.get("label") or ""),
        "written_at": str(first.get("written_at") or ""),
        "source": str(first.get("source") or ""),
        "directional_anchor": str(anchor.get("label") or "") if anchor else "",
        "anchor_written_at": str(anchor.get("written_at") or "") if anchor else "",
    }
