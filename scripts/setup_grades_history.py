"""Append-only, dated history of the setup grades (PROVEN/A/B/C/D/New).

`working_lately_service` writes `setup_grades_latest.json` and, right after,
calls `append_snapshot` here. One JSONL file per local calendar day under
`SETUP_GRADES_HISTORY_DIR`; one line per write whose content changed.
`grades_as_of(when)` answers "what grade did this setup have at that moment".

Evidence only: nothing here feeds a detector, a score, an alert or a rank.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "setup_grades_history_v1"

#: Files older than this many calendar days are pruned (600 days is about
#: 413 sessions, so at least 400 sessions are always kept).
KEEP_DAYS = 600

_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")
_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)


def default_history_dir() -> Path:
    from project_paths import SETUP_GRADES_HISTORY_DIR

    return Path(SETUP_GRADES_HISTORY_DIR)


def _aware(when: datetime | None) -> datetime:
    """`when` as a tz-aware datetime; naive values are read as local time."""
    if when is None:
        return datetime.now().astimezone()
    return when if when.tzinfo is not None else when.astimezone()


def _content_hash(grades: Mapping[str, Any]) -> str:
    text = json.dumps(grades, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dated_files(history_dir: Path) -> list[tuple[date, Path]]:
    """`(day, path)` for every history file, oldest first."""
    out: list[tuple[date, Path]] = []
    try:
        entries = list(history_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        match = _FILE_RE.match(entry.name)
        if not match:
            continue
        try:
            out.append((date.fromisoformat(match.group(1)), entry))
        except ValueError:
            continue
    out.sort()
    return out


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
            continue  # a torn last line loses that line only
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _last_hash(history_dir: Path) -> str:
    for _day, path in reversed(_dated_files(history_dir)):
        rows = _read_rows(path)
        if rows:
            return str(rows[-1].get("sha256") or "")
    return ""


def _prune(history_dir: Path, today: date) -> None:
    cutoff = today - timedelta(days=KEEP_DAYS)
    for day, path in _dated_files(history_dir):
        if day >= cutoff:
            break
        try:
            path.unlink()
        except OSError:
            _LOG.warning("Setup grades history: could not prune %s", path, exc_info=True)


def append_snapshot(
    grades: Mapping[str, Any],
    *,
    history_dir: Path | str | None = None,
    now: datetime | None = None,
) -> bool:
    """Append one line for `grades` unless it matches the newest line. True if written.

    Raises on I/O failure; the caller logs it and keeps its own write.
    """
    folder = Path(history_dir) if history_dir is not None else default_history_dir()
    written_at = _aware(now)
    digest = _content_hash(grades)
    with _LOCK:
        if _last_hash(folder) == digest:
            return False
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{written_at.date().isoformat()}.jsonl"
        is_new_file = not target.exists()
        row = {
            "schema": SCHEMA,
            "written_at": written_at.isoformat(timespec="seconds"),
            "as_of": str(grades.get("as_of") or ""),
            "sha256": digest,
            "grades": dict(grades),
        }
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
        if is_new_file:
            _prune(folder, written_at.date())
    return True


def grades_as_of(
    when: datetime,
    *,
    history_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """The newest grades written at or before `when`, never one written after.

    Returns the grades payload plus `written_at` (ISO, tz-aware), or None when
    no snapshot was written by then. A naive `when` is read as local time.
    """
    folder = Path(history_dir) if history_dir is not None else default_history_dir()
    limit = _aware(when)
    # A file is named by its local write date; allow one day of slack for
    # a `when` in another timezone, then compare exact instants.
    last_day = limit.astimezone().date() + timedelta(days=1)
    for day, path in reversed(_dated_files(folder)):
        if day > last_day:
            continue
        best: dict[str, Any] | None = None
        best_at: datetime | None = None
        for row in _read_rows(path):
            try:
                at = datetime.fromisoformat(str(row.get("written_at") or ""))
            except ValueError:
                continue
            if at.tzinfo is None or at > limit:
                continue
            if best_at is None or at >= best_at:
                best, best_at = row, at
        if best is not None and isinstance(best.get("grades"), dict):
            payload = dict(best["grades"])
            payload["written_at"] = str(best.get("written_at") or "")
            return payload
    return None
