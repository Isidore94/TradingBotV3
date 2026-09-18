"""One small file per session, so opening Day Review is not a 476 MB read.

TJ-1 item 4. Measured on the live desk 2026-09-17:
`daily_recap_reader._read_intraday_outcomes` streams the WHOLE
`intraday_bounce_outcomes.csv` (476 MB) on every open, with
`master_avwap_session_horizon_outcomes.csv` (31 MB) behind it, and every view
then filters by session AFTERWARDS. On a staged copy of that home folder one
`daily_recap.reload` settled in 16.5 seconds.

The index stores, per session, exactly the two `_Store`s `read_session` would
have built for it:

* ``rows`` - only the rows the views for that session and its lookback window
  actually consult (the latest append per `event_id`, as the streaming reader
  keeps them, in the order it keeps them);
* ``coverage`` - the FULL-FILE :class:`daily_recap_reader.SourceCoverage`,
  carried forward unchanged. It answers "was the file there and how big was it",
  so an index that stored the kept-row count instead would quietly relabel a
  476 MB file as a 40-row one;
* ``raw_rows_by_session`` - the append counts, which the recap summary names.

So an indexed read is the SAME ANSWER off a small file, and that equality is the
contract: `read_session(date, index=build_index(date))` equals
`read_session(date)` on the whole `RecapSession` dataclass.

**It is derived and it may never cost the page.** A missing index means a slow
open; a corrupt one means a slow open; a write that fails is logged and the page
still paints. The one judgement in here is :func:`is_stale`, which is a single
function on purpose: an index built while a horizon was still forming may be
wrong once that session closes, and every reader has to agree about when.

This module imports `daily_recap_reader` and `daily_recap_reader` never imports
this one, except lazily inside `read_session`: one direction, no cycle.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import daily_recap_reader
import project_paths

_log = logging.getLogger(__name__)

#: Schema NAME (ground rule 5). A file that does not carry it is not an index.
SCHEMA = "day_review_index_v1"

#: The two stores the index covers. The other ten are small and are read live,
#: so a note written since the index was built is still on the page.
INDEXED_SOURCES: tuple[str, ...] = ("intraday_outcomes", "session_horizon_outcomes")


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def default_root() -> Path:
    """`DAY_REVIEW_DIR`, read at CALL time so a test can redirect it."""
    return Path(project_paths.DAY_REVIEW_DIR)


def index_path(session_date: str, *, root: Path | None = None) -> Path:
    """`<root>/sessions/<date>/outcomes.json` - one folder per session.

    A folder rather than one flat file per date because the later TJ packets put
    the session's own M5 bars beside this file.
    """
    base = Path(root) if root is not None else default_root()
    return base / "sessions" / str(session_date or "").strip()[:10] / "outcomes.json"


# ---------------------------------------------------------------------------
# serialising one store
# ---------------------------------------------------------------------------
def _moment_text(moment: datetime | None) -> str | None:
    return moment.isoformat() if isinstance(moment, datetime) else None


def _moment(text: Any) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text))
    except ValueError:
        return None


def _coverage_payload(coverage: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(coverage, "name", "")),
        "path": str(getattr(coverage, "path", "")),
        "rows": int(getattr(coverage, "rows", 0) or 0),
        "oldest": _moment_text(getattr(coverage, "oldest", None)),
        "newest": _moment_text(getattr(coverage, "newest", None)),
        "unavailable_reason": str(getattr(coverage, "unavailable_reason", "") or ""),
    }


def _coverage(payload: Mapping[str, Any]) -> daily_recap_reader.SourceCoverage:
    return daily_recap_reader.SourceCoverage(
        name=str(payload.get("name") or ""),
        path=str(payload.get("path") or ""),
        rows=int(payload.get("rows") or 0),
        oldest=_moment(payload.get("oldest")),
        newest=_moment(payload.get("newest")),
        unavailable_reason=str(payload.get("unavailable_reason") or ""),
    )


def store_payload(
    store: Any, rows: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """One `_Store` as JSON: its kept rows, its full-file coverage, its counts.

    `rows` overrides the store's own rows, which is how the builder stores the
    SELECTED session's rows under the WHOLE file's coverage.
    """
    kept = store.rows if rows is None else rows
    return {
        "rows": [dict(row) for row in kept],
        "coverage": _coverage_payload(store.coverage),
        "raw_rows_by_session": dict(getattr(store, "raw_rows_by_session", {}) or {}),
    }


def store_from_payload(payload: Mapping[str, Any]):
    """The `_Store` a stored payload describes. The inverse of `store_payload`."""
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("an index store payload has no rows list")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("an index store payload has no coverage block")
    counts = payload.get("raw_rows_by_session")
    return daily_recap_reader._Store(
        tuple(dict(row) for row in rows if isinstance(row, Mapping)),
        _coverage(coverage),
        raw_rows_by_session={
            str(key): int(value) for key, value in dict(counts or {}).items()
        },
    )


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _horizon_scope(session_date: str, lookback_sessions: int) -> tuple[str, str, str]:
    """`(window_first, window_last, session)` - the scan dates the views read.

    `_recent_swings_view` reads the lookback WINDOW and `_d1_horizon_row` reads
    the SELECTED session, so the index needs both or a D1 decision on the day
    would lose its declared result.
    """
    first, last = daily_recap_reader._lookback_window(session_date, lookback_sessions)
    return first, last, session_date


def build_index(
    session_date: str,
    *,
    lookback_sessions: int = 3,
    sources: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read the two big stores once and write down what this session needs.

    Streams exactly what `read_session` streams - same functions, same identity
    rules - and then narrows the ROWS to the sessions the views consult while
    carrying the coverage forward whole.
    """
    session = str(session_date or "").strip()[:10]
    lookback = max(1, int(lookback_sessions))
    sources = sources or daily_recap_reader.RecapSources()
    moment = now or datetime.now()

    intraday = daily_recap_reader._read_intraday_outcomes(sources.intraday_outcomes)
    horizon = daily_recap_reader._read_csv(
        "session_horizon_outcomes", sources.session_horizon_outcomes, "scan_date"
    )

    first, last, _ = _horizon_scope(session, lookback)
    horizon_rows: list[Mapping[str, Any]] = []
    pending_targets: list[str] = []
    pending = False
    for row in horizon.rows:
        scan_date = daily_recap_reader._session_text(row.get("scan_date"))
        if not (first <= scan_date <= last or scan_date == session):
            continue
        horizon_rows.append(row)
        if daily_recap_reader._measured_return(row) is None:
            pending = True
            target = daily_recap_reader._session_text(row.get("target_session"))
            if target:
                pending_targets.append(target)

    # The intraday sessions the views ask about: the selected one, and the
    # target session of every horizon row in the window (the swing view reads
    # the name's own M5 excursion on the session its pick was measured INTO).
    wanted = {session}
    for row in horizon_rows:
        target = daily_recap_reader._session_text(row.get("target_session"))
        if target:
            wanted.add(target)
    intraday_rows = [
        row
        for row in intraday.rows
        if daily_recap_reader._session_text(row.get("trade_date")) in wanted
    ]

    return {
        "schema": SCHEMA,
        "session_date": session,
        "lookback_sessions": lookback,
        "built_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds")
        if moment.tzinfo is not None
        else moment.isoformat(timespec="seconds"),
        "pending": bool(pending),
        "pending_target_sessions": sorted(set(pending_targets)),
        "intraday_outcomes": store_payload(intraday, intraday_rows),
        "session_horizon_outcomes": store_payload(horizon, horizon_rows),
    }


# ---------------------------------------------------------------------------
# staleness - ONE rule
# ---------------------------------------------------------------------------
def is_stale(index: Mapping[str, Any] | None, *, now: datetime | None = None) -> bool:
    """Could the answer in this file have changed since it was written?

    Only one thing in it can: a horizon row that had not matured when it was
    built. Once the session that row was waiting for has CLOSED, the stored
    answer may be wrong and the page rebuilds rather than printing it. Before
    that it is the same answer, and rebuilding on every open would give the
    index no purpose.

    An index with nothing pending is never stale: the rows it holds are the last
    append of a finished event and a closed session does not reopen.
    """
    if not isinstance(index, Mapping):
        return True
    if not bool(index.get("pending")):
        return False
    targets = [
        str(value)[:10]
        for value in (index.get("pending_target_sessions") or ())
        if str(value or "").strip()
    ]
    try:
        import market_calendar

        completed = market_calendar.last_completed_session(now or datetime.now())
    except Exception:  # noqa: BLE001 - an unreadable calendar rebuilds, never prints
        return True
    if not targets:
        # Nothing named a session to wait for, so the only honest rule left is
        # "a session has closed since it was built".
        built = _moment(index.get("built_at"))
        if built is None:
            return True
        try:
            built_completed = market_calendar.last_completed_session(
                built.replace(tzinfo=None) if built.tzinfo is not None else built
            )
        except Exception:  # noqa: BLE001
            return True
        return completed > built_completed
    try:
        return completed >= date.fromisoformat(min(targets))
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# reading it back
# ---------------------------------------------------------------------------
def stores_for(
    index: Mapping[str, Any] | None,
    *,
    session_date: str,
    lookback_sessions: int,
):
    """`(intraday, horizon)` from a VALID index for this read, else `None`.

    Validated on what it is an index OF, not only on its shape: an index for
    another session, or for another lookback window, holds the wrong horizon
    rows, and a reader that used it anyway would print one session's numbers
    under another's date.
    """
    if not isinstance(index, Mapping):
        return None
    if str(index.get("schema") or "") != SCHEMA:
        return None
    if str(index.get("session_date") or "") != str(session_date or "").strip()[:10]:
        return None
    if int(index.get("lookback_sessions") or 0) != max(1, int(lookback_sessions)):
        return None
    try:
        return tuple(
            store_from_payload(index[name]) for name in INDEXED_SOURCES
        )
    except (KeyError, TypeError, ValueError):
        _log.debug("A Day Review index could not be revived; streaming instead.")
        return None


def read_index(session_date: str, *, root: Path | None = None) -> dict[str, Any] | None:
    """The stored index for one session, or `None`.

    A half-written file after a power cut is uncertainty, never a broken page:
    it is logged and read as absent.
    """
    path = index_path(session_date, root=root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        _log.info("The Day Review index %s could not be read: %s", path, exc)
        return None
    if not isinstance(payload, Mapping) or str(payload.get("schema") or "") != SCHEMA:
        _log.info("The Day Review index %s is not an index of this schema.", path)
        return None
    return dict(payload)


def write_index(index: Mapping[str, Any], *, root: Path | None = None) -> Path | None:
    """Temp-and-rename the index beside its session. `None` when it failed.

    A derived, rebuildable artefact may never cost the page that wanted it, so
    every failure is logged and swallowed - the page has already painted from
    the rows this file was built out of.
    """
    session = str((index or {}).get("session_date") or "").strip()[:10]
    if not session:
        _log.info("A Day Review index with no session was not written.")
        return None
    path = index_path(session, root=root)
    temp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(
            json.dumps(dict(index), default=str, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    except Exception as exc:  # noqa: BLE001 - the index is never worth the page
        _log.info("The Day Review index %s was not written: %s", path, exc)
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return path


__all__ = [
    "INDEXED_SOURCES",
    "SCHEMA",
    "build_index",
    "default_root",
    "index_path",
    "is_stale",
    "read_index",
    "store_from_payload",
    "store_payload",
    "stores_for",
    "write_index",
]
