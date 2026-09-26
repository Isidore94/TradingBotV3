"""The one join onto the trader's structural regime (S16 item 3). Pure, plus one reader.

Every per-regime number on the desk labels its rows here: the regime of a row is
the trader's structural regime (``structural_regime``) in force on the row's
entry / scan / alert date. Point in time: a row is labelled only by segments
whose ``start_date`` is on or before its date; a row before the first segment is
``unknown``. A segment the trader typed later but dated back (the three past
regimes, a correction that supersedes) DOES label older rows - the trader
authored it, and the regime is the trader's call, not something the desk
inferred.

Nothing here grades; `regime_grades` does, and it never pools two regimes.
"""

from __future__ import annotations

import sqlite3
from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote

import structural_regime

UNKNOWN = "unknown"
#: What a per-regime cell with no rows in the current regime says.
UNTESTED = "untested in this regime"
#: The label the existing pooled grade carries beside the per-regime cells.
ALL_REGIMES = "all regimes"


def _day_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return ""


def timeline(rows: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """The trader's effective segments, oldest first (`structural_regime.effective_segments`)."""
    return structural_regime.effective_segments(rows or ())


class Joiner:
    """``label(day)`` over one timeline, by bisect. Build once per read, reuse per row."""

    def __init__(self, segments: Iterable[Mapping[str, Any]] | None) -> None:
        ordered = sorted(
            (
                (_day_text(segment.get("start_date")), str(segment.get("regime") or "").strip())
                for segment in segments or ()
                if isinstance(segment, Mapping)
            ),
        )
        self._starts = [start for start, regime in ordered if start and regime]
        self._regimes = [regime for start, regime in ordered if start and regime]

    def __bool__(self) -> bool:
        return bool(self._starts)

    def label(self, day: Any) -> str:
        text = _day_text(day)
        if not text:
            return UNKNOWN
        index = bisect_right(self._starts, text)
        return self._regimes[index - 1] if index else UNKNOWN


def label_for(day: Any, segments: Iterable[Mapping[str, Any]] | None) -> str:
    """The regime in force on ``day`` from an effective timeline, or ``unknown``.

    ``segments`` is `timeline(rows)` (effective, never raw rows with supersedes).
    Only segments starting on or before ``day`` count; a backdated segment the
    trader typed later counts too (it is the trader's own label).
    """
    return Joiner(segments).label(day)


def split(
    rows: Iterable[Any],
    date_of: Callable[[Any], Any],
    segments: Iterable[Mapping[str, Any]] | Joiner | None,
) -> dict[str, list[Any]]:
    """``{regime: [rows]}``; a row with no date or before the first segment is ``unknown``."""
    joiner = segments if isinstance(segments, Joiner) else Joiner(segments)
    out: dict[str, list[Any]] = {}
    for row in rows or ():
        out.setdefault(joiner.label(date_of(row)), []).append(row)
    return out


def ordered_regimes(
    present: Iterable[str],
    current: str | None,
    segments: Iterable[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Current regime first, then the others newest segment first, ``unknown`` last.

    The current regime is listed even when no row falls in it (its cells then say
    `UNTESTED`).
    """
    wanted = {str(name) for name in present if name}
    newest: dict[str, str] = {}
    for segment in segments or ():
        regime = str(segment.get("regime") or "").strip()
        start = _day_text(segment.get("start_date"))
        if regime and start > newest.get(regime, ""):
            newest[regime] = start
    out: list[str] = []
    if current and current != UNKNOWN:
        out.append(current)
    others = sorted(
        (name for name in wanted if name not in out and name != UNKNOWN),
        key=lambda name: (newest.get(name, ""), name),
        reverse=True,
    )
    out.extend(others)
    if UNKNOWN in wanted:
        out.append(UNKNOWN)
    return out


def regime_label(regime: str | None) -> str:
    if not regime or regime == UNKNOWN:
        return "regime unknown"
    return structural_regime.label(regime)



# ---------------------------------------------------------------------------
# the one reader (IO): worker threads only
# ---------------------------------------------------------------------------


def select_read_only(db_path: Any, sql: str) -> list[dict[str, Any]]:
    """Rows of one SELECT on the journal opened read-only; ``[]`` when absent or unreadable.

    No migration and no write. The default path is `project_paths.JOURNAL_DB_FILE`.
    IO: never on the Qt thread.
    """
    if db_path is None:
        from project_paths import JOURNAL_DB_FILE

        db_path = JOURNAL_DB_FILE
    path = Path(db_path)
    if not path.is_file():
        return []
    try:
        uri = "file:" + quote(path.as_posix(), safe="/:") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.Error:
        try:  # a UNC path is not a valid URI authority; plain open, SELECT only
            conn = sqlite3.connect(str(path), timeout=5)
        except sqlite3.Error:
            return []
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def read_rows(db_path: Any = None) -> list[dict[str, Any]]:
    """The raw ``structural_regime`` rows; a missing table is no regime yet."""
    return select_read_only(db_path, "SELECT * FROM structural_regime ORDER BY segment_id")


def read_segments(db_path: Any = None) -> list[dict[str, Any]]:
    """`timeline(read_rows(db_path))`. IO: never on the Qt thread."""
    return timeline(read_rows(db_path))


def read_trades(db_path: Any = None) -> list[dict[str, Any]]:
    """The journal's trades, the columns the regime join reads. IO: never on the Qt thread."""
    return select_read_only(
        db_path, "SELECT trade_id, status, direction, opened_at, trade_date, net_pnl FROM trades"
    )
