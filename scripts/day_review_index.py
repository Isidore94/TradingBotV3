"""One small file per session, so opening Day Review is not a 476 MB read.

TJ-1 item 4. Measured on the live desk 2026-09-17:
`daily_recap_reader._read_intraday_outcomes` streams the WHOLE
`intraday_bounce_outcomes.csv` (476 MB) on every open, with
`master_avwap_session_horizon_outcomes.csv` (31 MB) behind it, and every view
then filters by session AFTERWARDS. On a staged copy of that home folder one
`daily_recap.reload` settled in 16.5 seconds.

**How wide it is, and why.** The first cut covered the two biggest stores and left
an indexed read at 2,217 ms, which did not meet gate #145's "under one second". It
now covers every store `read_session` opens whose live file is over a megabyte -
the 476 MB intraday log, the 31 MB horizon CSV, the 14 MB tier CSV and the 1.0 MB
human-focus CSV (`INDEXED_SOURCE_SPECS`) - and the six small ones are still read
LIVE on every open, because a veto, a note, a favorite or a staged pick from a
minute ago has to be on the page. That is a freshness rule, not an optimisation,
and it is why `alert_review_events.jsonl` (0.27 MB, 15 ms) and
`preference_trade_outcomes.csv` (0.44 MB, 8 ms) are deliberately NOT in here: both
are rewritten as the trader and the journal move, and indexing them would buy
23 ms at the price of a page that had stopped listening.

The index stores, per session, exactly the `_Store`s `read_session` would have
built for it:

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
`read_session(date)` on the whole `RecapSession` dataclass, AT BUILD TIME; after an
append the stamp judged out of scope, only the full-file `coverage` of that store
lags the disk, and nothing the page renders reads it. It is ALL OR NOTHING -
an index that carries some of `INDEXED_SOURCES` and not others reads as absent, so
an index written by an earlier build simply causes one slow open and is replaced.

**It is derived and it may never cost the page.** A missing index means a slow
open; a corrupt one means a slow open; a write that fails is logged and the page
still paints. The one judgement in here is :func:`is_stale`, which is a single
function on purpose: an index built while a horizon was still forming may be
wrong once that session closes, and every reader has to agree about when.

This module imports `daily_recap_reader` and `daily_recap_reader` never imports
this one, except lazily inside `read_session`: one direction, no cycle.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import daily_recap_reader
import project_paths

_log = logging.getLogger(__name__)

#: Schema NAME (ground rule 5). A file that does not carry it is not an index.
SCHEMA = "day_review_index_v1"

#: The stores the index covers, and how each one is read and narrowed:
#: `name -> (reader, clock_field, session_field)`. The reader is the SAME
#: function `read_session` uses, so an indexed store is built by the code that
#: would otherwise have streamed it; `session_field` is the column whose value
#: says which trading session a row belongs to, and it is how the slice is cut.
#:
#: The FOUR here are every store `read_session` opens whose live file is over a
#: megabyte (measured 2026-09-17 on a staged copy of the live home folder:
#: 476 MB, 31 MB, 14 MB, 1.0 MB - and 944 ms + 164 ms of a 1,754 ms indexed read
#: came from the last two alone). Everything else stays LIVE, and two of those
#: are named in the docstring above because the reason is not their size.
INDEXED_SOURCE_SPECS: dict[str, tuple[str, str, str]] = {
    "intraday_outcomes": ("intraday", "logged_at", "trade_date"),
    "session_horizon_outcomes": ("csv", "scan_date", "scan_date"),
    "tier_outcomes": ("csv", "run_timestamp", "scan_date"),
    "human_focus_outcomes": ("csv", "updated_at", "trade_date"),
}

#: The order they are declared in, and the set an index must carry to be usable.
INDEXED_SOURCES: tuple[str, ...] = tuple(INDEXED_SOURCE_SPECS)

#: How many sessions' indexes are kept on disk. One is 22-30 MB, the picker
#: offers fifteen sessions, and every one of them is rebuildable - so the folder
#: is pruned to the newest 40 on each write rather than growing for ever.
KEEP_SESSIONS = 40


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def default_root() -> Path:
    """`DAY_REVIEW_DIR`, read at CALL time so a test can redirect it."""
    return Path(project_paths.DAY_REVIEW_DIR)


def stamp_path(session_date: str, *, root: Path | None = None) -> Path:
    """`stamp.json` beside the index - a few hundred bytes, not 22 MB.

    When the stores have only GROWN outside this index's scope, the body is still
    the right answer and the only thing out of date is the stamp. Recording it
    here means the next open compares against the new size instead of reading the
    same tail again, without rewriting the body (reviewer, round 2).
    """
    return index_path(session_date, root=root).with_name("stamp.json")


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


def read_store(name: str, sources: Any):
    """Read one indexed store the way `read_session` reads it. One reader each.

    Named rather than inlined so the builder and the streaming reader cannot
    drift: an index built by a different function from the one it replaces is an
    index that agrees with nothing.
    """
    reader, clock_field, _session_field = INDEXED_SOURCE_SPECS[name]
    path = getattr(sources, name)
    if reader == "intraday":
        return daily_recap_reader._read_intraday_outcomes(path)
    return daily_recap_reader._read_csv(name, path, clock_field)


def sources_stamp(sources: Any) -> dict[str, dict[str, Any]]:
    """`name -> (size, mtime_ns)` for the indexed stores, as they are right now.

    The cheapest honest answer to "is this index still describing those files".
    A warehouse recompute REWRITES the outcome CSVs - the rows for a session that
    closed weeks ago can change - and nothing else in the index would notice: the
    session is closed, nothing is pending, and the stored answer would be printed
    for ever. A missing file is stamped as absent rather than skipped, so a store
    that comes back is a change too.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in INDEXED_SOURCES:
        path = Path(getattr(sources, name))
        try:
            stat = path.stat()
            out[name] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        except OSError:
            out[name] = {"size": -1, "mtime_ns": -1}
    return out


@dataclass(frozen=True)
class _Scope:
    """What an index would have kept, as the tail rule has to ask it.

    `sessions` are the dates where ANY name counts - the selected session and its
    lookback window, which is what the day's own views read. `pairs` are
    `(session, symbol)`: the swing view reads ONE name's excursion on the session
    its pick was measured into, so a target session weeks forward only matters for
    the handful of names the index's own observations reference.

    That distinction is the whole point (reviewer, round 3). Every recent index
    holds target sessions running weeks ahead - six live indexes from 2026-08-28
    to 2026-09-17 all carry 2026-09-18, with targets out to 2026-10-01 - so a
    scope keyed on the SESSION alone made every `trade_date = today` append from
    the M5 scanner a rebuild, which is the cost the tail rule exists to avoid.
    SIDE is deliberately not in the key: a blank side matches either side, so a
    side-blind pair is the conservative answer.
    """

    session: str
    first: str
    last: str
    pairs: frozenset[tuple[str, str]]


def _index_scope(index: Mapping[str, Any]) -> _Scope:
    """The scope a stored index describes.

    Rebuilt from what the index itself CARRIES, never from the live stores: the
    question a tail row has to answer is "would THIS index have kept you".
    """
    session = str(index.get("session_date") or "")[:10]
    lookback = max(1, int(index.get("lookback_sessions") or 3))
    first, last, _ = _horizon_scope(session, lookback)
    pairs: set[tuple[str, str]] = set()
    horizon = index.get("session_horizon_outcomes")
    if isinstance(horizon, Mapping):
        for row in horizon.get("rows") or ():
            if not isinstance(row, Mapping):
                continue
            target = daily_recap_reader._session_text(row.get("target_session"))
            symbol = daily_recap_reader._symbol(row.get("symbol"))
            if target and symbol:
                pairs.add((target, symbol))
    intraday = index.get("intraday_outcomes")
    if isinstance(intraday, Mapping):
        # A late append for an event this index already holds changes the latest
        # append it kept, so its own rows are part of the scope too.
        for row in intraday.get("rows") or ():
            if not isinstance(row, Mapping):
                continue
            stamp = daily_recap_reader._session_text(row.get("trade_date"))
            symbol = daily_recap_reader._symbol(row.get("symbol"))
            if stamp and symbol:
                pairs.add((stamp, symbol))
    return _Scope(session=session, first=first, last=last, pairs=frozenset(pairs))


def _appended_tail_touches(
    name: str, path: Path, stored_size: int, index: Mapping[str, Any]
) -> bool:
    """Did the bytes appended since `stored_size` change what this index says?

    Reads ONLY the tail. The reason this exists: `intraday_bounce_outcomes.csv`
    is appended to all day by the M5 scanner, so a stamp that invalidated on ANY
    change turned one appended row for ANOTHER session into a 12.1 s open and a
    22 MB rewrite with nothing on the page different (reviewer, round 2).

    It reads the tail with the file's OWN header re-attached, so it trusts the
    column NAMES the store declares (`trade_date` / `scan_date` and `symbol`) and
    not their positions; a store that renamed a column would answer "unparseable"
    and rebuild, which is the safe direction.

    The tail is read WHOLE. After a week unopened that tail is a week of appends -
    tens of MB on the intraday log - which is still two orders of magnitude less
    than the 476 MB stream it replaces, and it happens once, on the open that then
    records a fresh stamp.

    Answers True - rebuild - for anything it cannot read as out of scope:

    * a tail that does not begin at a line boundary (the stored size was taken
      mid-append, so the first row is a fragment);
    * a row whose session OR symbol cannot be read;
    * a file with no header to parse the tail against.

    Uncertainty rebuilds; it never assumes.
    """
    session_field = INDEXED_SOURCE_SPECS[name][2]
    scope = _index_scope(index)
    try:
        with path.open("rb") as handle:
            header = handle.readline()
            if not header.strip():
                return True
            if stored_size > 0:
                # The byte before the boundary must be the end of a line, or
                # what follows is half a row.
                handle.seek(stored_size - 1)
                if handle.read(1) not in (b"\n", b"\r"):
                    return True
            handle.seek(max(stored_size, 0))
            tail = handle.read()
    except OSError as exc:
        _log.info("The appended tail of %s could not be read: %s", path, exc)
        return True
    if not tail.strip():
        return False
    try:
        text = (header.decode("utf-8", "replace") + tail.decode("utf-8", "replace")).splitlines()
        rows = list(csv.DictReader(text))
    except Exception:  # noqa: BLE001 - an unreadable tail rebuilds
        _log.info("The appended tail of %s could not be parsed.", path)
        return True
    for row in rows:
        stamp = daily_recap_reader._session_text(row.get(session_field))
        symbol = daily_recap_reader._symbol(row.get("symbol"))
        if not stamp or not symbol:
            return True
        if stamp == scope.session or scope.first <= stamp <= scope.last:
            # The day's own views read every name of the session and its window.
            return True
        if (stamp, symbol) in scope.pairs:
            return True
    return False


def stamp_verdict(
    index: Mapping[str, Any], *, sources: Any
) -> tuple[str, dict[str, dict[str, Any]]]:
    """`("same" | "moved" | "rebuild", the stores' current stamp)`.

    * **same** - every indexed store is byte-for-byte where it was.
    * **moved** - a store only GREW, and none of the appended rows belong to
      this index's scope: its session, its lookback window, or a `(session,
      symbol)` pair its own observations reference. The body is still right; only
      the stamp needs recording (`refresh_stamp`), which is a few hundred bytes
      rather than 22 MB.
    * **rebuild** - a store SHRANK, changed at the SAME SIZE (a warehouse
      recompute rewrites in place), gained an in-scope row, or cannot be read.
      A bare `os.utime` with no size change lands here too: a same-size change
      is a rewrite as far as a stamp can tell, and a touched file with no new
      bytes is rare enough to pay for one rebuild.
    """
    current = sources_stamp(sources)
    stored = index.get("sources_stamp")
    if not isinstance(stored, Mapping) or not stored:
        return "same", current
    verdict = "same"
    for name in INDEXED_SOURCES:
        was = stored.get(name)
        now_stamp = current.get(name)
        if not isinstance(was, Mapping) or not isinstance(now_stamp, Mapping):
            return "rebuild", current
        if dict(was) == dict(now_stamp):
            continue
        old_size = int(was.get("size", -1))
        new_size = int(now_stamp.get("size", -1))
        if old_size < 0 or new_size < 0 or new_size < old_size or new_size == old_size:
            return "rebuild", current
        if _appended_tail_touches(
            name, Path(getattr(sources, name)), old_size, index
        ):
            return "rebuild", current
        verdict = "moved"
    return verdict, current


def _in_scope(value: Any, session: str, first: str, last: str, targets: set[str]) -> bool:
    """Does a row's own session belong to this read?

    Three ways in, and they are the three the views ask about: the SELECTED
    session, the lookback WINDOW (`_recent_swings_view`, `_d1_horizon_row`), and
    the TARGET session a windowed swing observation was measured into (the swing
    view reads that name's own M5 excursion on it). A row outside all three is
    read by nothing, which is why leaving it out cannot change the answer.
    """
    stamp = daily_recap_reader._session_text(value)
    if not stamp:
        return False
    return stamp == session or first <= stamp <= last or stamp in targets


def build_index(
    session_date: str,
    *,
    lookback_sessions: int = 3,
    sources: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read the big stores once and write down what this session needs.

    Streams exactly what `read_session` streams - same functions, same identity
    rules - and then narrows the ROWS to the sessions the views consult while
    carrying every coverage line forward whole.
    """
    session = str(session_date or "").strip()[:10]
    lookback = max(1, int(lookback_sessions))
    sources = sources or daily_recap_reader.RecapSources()
    moment = now or datetime.now()

    stores = {name: read_store(name, sources) for name in INDEXED_SOURCES}
    horizon = stores["session_horizon_outcomes"]

    first, last, _ = _horizon_scope(session, lookback)
    # The horizon store first, because what it says is pending decides both the
    # staleness rule and which OTHER sessions the rest of the slice needs.
    targets: set[str] = set()
    pending_targets: list[str] = []
    pending = False
    for row in horizon.rows:
        if not _in_scope(row.get("scan_date"), session, first, last, set()):
            continue
        target = daily_recap_reader._session_text(row.get("target_session"))
        if target:
            targets.add(target)
        if daily_recap_reader._measured_return(row) is None:
            pending = True
            if target:
                pending_targets.append(target)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "session_date": session,
        "lookback_sessions": lookback,
        "built_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds")
        if moment.tzinfo is not None
        else moment.isoformat(timespec="seconds"),
        "pending": bool(pending),
        "pending_target_sessions": sorted(set(pending_targets)),
        "sources_stamp": sources_stamp(sources),
    }
    for name, store in stores.items():
        session_field = INDEXED_SOURCE_SPECS[name][2]
        kept = [
            row
            for row in store.rows
            if _in_scope(row.get(session_field), session, first, last, targets)
        ]
        payload[name] = store_payload(store, kept)
    return payload


# ---------------------------------------------------------------------------
# staleness - ONE rule
# ---------------------------------------------------------------------------
def is_stale(
    index: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
    sources: Any = None,
) -> bool:
    """Could the answer in this file have changed since it was written?

    With `sources` given, the first question is about the FILES: are they still
    the files this index was built from (`stamp_verdict`)? A warehouse recompute
    rewrites the outcome CSVs and can change the rows of a session that closed
    weeks ago, which none of the clauses below would ever notice. But growth is
    not a rewrite - the M5 scanner appends to the intraday log all day - so an
    APPEND is read at the tail and only an appended row that belongs to this
    index's own scope makes it stale. Without `sources` - a caller that has none,
    and every test that asks about the clock alone - none of that is consulted,
    and an index written before the stamp existed is not failed for lacking one.

    Then two things can change it, and only two:

    * **The session had not closed when it was built.** Then the file it
      describes is still being appended to, so the index is a snapshot of a
      moving target and is always stale. (The post-close tick is what writes the
      one that lasts.)
    * **A horizon row that had not matured has since matured.** A pending row
      matures when the session it was waiting for CLOSES, so the question is
      narrow: did one of the sessions those rows named close SINCE this index
      was built? A pending row whose target had already closed at build time is
      unmeasured for some other reason and waiting will not change it -
      rebuilding for that one would make every index stale the moment it was
      written, which is how a cache ends up costing more than it saves.

    An index with nothing pending is never stale: the rows it holds are the last
    append of a finished event, and a closed session does not reopen.
    """
    if not isinstance(index, Mapping):
        return True
    if sources is not None and stamp_verdict(index, sources=sources)[0] == "rebuild":
        return True
    if not bool(index.get("pending")):
        return False
    built = _moment(index.get("built_at"))
    if built is None:
        return True
    if built.tzinfo is not None:
        built = built.astimezone().replace(tzinfo=None)
    try:
        import market_calendar

        completed = market_calendar.last_completed_session(now or datetime.now())
        built_completed = market_calendar.last_completed_session(built)
    except Exception:  # noqa: BLE001 - an unreadable calendar rebuilds, never prints
        return True
    session = str(index.get("session_date") or "")[:10]
    try:
        if session and date.fromisoformat(session) > built_completed:
            return True
    except ValueError:
        return True
    if completed <= built_completed:
        return False
    targets: list[date] = []
    for value in index.get("pending_target_sessions") or ():
        try:
            targets.append(date.fromisoformat(str(value)[:10]))
        except ValueError:
            return True
    if not targets:
        # Something was pending and a session has closed since; with no target
        # named there is nothing narrower to ask.
        return True
    return any(built_completed < target <= completed for target in targets)


# ---------------------------------------------------------------------------
# reading it back
# ---------------------------------------------------------------------------
def stores_for(
    index: Mapping[str, Any] | None,
    *,
    session_date: str,
    lookback_sessions: int,
) -> dict[str, Any] | None:
    """`{name: _Store}` from a VALID index for this read, else `None`.

    Validated on what it is an index OF, not only on its shape: an index for
    another session, or for another lookback window, holds the wrong horizon
    rows, and a reader that used it anyway would print one session's numbers
    under another's date.

    **ALL OR NOTHING.** An index that carries some of `INDEXED_SOURCES` and not
    others is answered as absent, which is how an index written by an earlier
    build - when this covered two stores rather than four - is treated: the read
    streams and leaves a complete one behind. Half a cache is the shape of bug
    where one store is a week old and the page looks fine.
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
        return {name: store_from_payload(index[name]) for name in INDEXED_SOURCES}
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
    index = dict(payload)
    # The sidecar stamp, when one was recorded after an out-of-scope append. It
    # only ever REPLACES the stamp; a sidecar that is unreadable or belongs to
    # another session is ignored, which costs one tail read.
    stamp_file = stamp_path(session_date, root=root)
    try:
        sidecar = json.loads(stamp_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return index
    except (OSError, json.JSONDecodeError) as exc:
        _log.info("The Day Review stamp %s could not be read: %s", stamp_file, exc)
        return index
    if (
        isinstance(sidecar, Mapping)
        and str(sidecar.get("session_date") or "") == str(index.get("session_date") or "")
        and isinstance(sidecar.get("sources_stamp"), Mapping)
    ):
        index["sources_stamp"] = dict(sidecar["sources_stamp"])
    return index


def refresh_stamp(
    index: Mapping[str, Any],
    *,
    sources: Any = None,
    stamp: Mapping[str, Mapping[str, Any]] | None = None,
    root: Path | None = None,
) -> Path | None:
    """Record the stores' CURRENT stamp beside an index whose body is still right.

    Called after a "moved" verdict. Hand it the `stamp` that verdict was computed
    from: the caller has already stat-ed the stores and read the tail, and doing
    it again would be the second of two identical answers per open (reviewer,
    round 3). With no `stamp`, `sources` is stat-ed here.

    Quiet on failure: without it the next open reads the same tail again, which is
    milliseconds, never a wrong page.
    """
    session = str((index or {}).get("session_date") or "")[:10]
    if not session:
        return None
    if stamp is not None:
        current = {name: dict(value) for name, value in dict(stamp).items()}
        if current == {
            name: dict(value)
            for name, value in dict(index.get("sources_stamp") or {}).items()
        }:
            return None
    else:
        verdict, current = stamp_verdict(index, sources=sources)
        if verdict == "same":
            return None
    path = stamp_path(session, root=root)
    temp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(
            json.dumps(
                {"schema": SCHEMA, "session_date": session, "sources_stamp": current},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    except Exception as exc:  # noqa: BLE001 - a stamp is never worth the page
        _log.info("The Day Review stamp %s was not written: %s", path, exc)
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return path


def _session_has_closed(session: str, now: datetime | None = None) -> bool:
    """Has the session this index describes finished producing rows?

    An index for a session that is still trading describes files that are still
    being appended to, so writing one spends 22-30 MB to cache an answer that is
    already out of date (`is_stale` would refuse it anyway). Today's page streams
    until the post-close tick builds the one that lasts. Unknown reads as NOT
    closed: refusing to write costs a slow open, and writing costs a wrong page.
    """
    try:
        import market_calendar

        return date.fromisoformat(session) <= market_calendar.last_completed_session(
            now or datetime.now()
        )
    except Exception:  # noqa: BLE001
        return False


def _prune(base: Path, keep: int | None = None) -> int:
    """Keep the newest `keep` session folders. Rebuildable, so this is safe.

    `KEEP_SESSIONS` is read at CALL time, not bound as a default, so a test can
    redirect it - the idiom every other tunable in this repo uses.

    Quiet on every failure: a folder that will not delete is a folder that stays,
    never an index that was not written.
    """
    keep = KEEP_SESSIONS if keep is None else int(keep)
    removed = 0
    try:
        sessions = sorted(
            (child for child in (base / "sessions").iterdir() if child.is_dir()),
            key=lambda child: child.name,
            reverse=True,
        )
    except OSError:
        return 0
    for stale in sessions[keep:]:
        try:
            for child in stale.iterdir():
                child.unlink()
            stale.rmdir()
            removed += 1
        except OSError as exc:
            _log.debug("The Day Review index %s was not pruned: %s", stale, exc)
    return removed


def _body(index: Mapping[str, Any]) -> str:
    """The index as it is stored, WITHOUT the stamp of when it was built.

    Two builds of a finished session differ only in `built_at`, and rewriting
    22-30 MB to change one timestamp is churn on the shared home folder.
    """
    payload = {key: value for key, value in dict(index).items() if key != "built_at"}
    return json.dumps(payload, default=str, sort_keys=True)


def _stamp_only(session: str, index: Mapping[str, Any], *, root: Path | None = None) -> None:
    """Record just the stamp of an index whose rows are already on disk."""
    stamp = index.get("sources_stamp")
    if not isinstance(stamp, Mapping):
        return
    path = stamp_path(session, root=root)
    temp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(
            json.dumps(
                {"schema": SCHEMA, "session_date": session, "sources_stamp": dict(stamp)},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    except Exception as exc:  # noqa: BLE001 - a stamp is never worth the page
        _log.info("The Day Review stamp for %s was not written: %s", session, exc)


def write_index(
    index: Mapping[str, Any],
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> Path | None:
    """Temp-and-rename the index beside its session. `None` when nothing was written.

    Three refusals, all of them about a 22-30 MB file on the shared home folder:
    a session that has NOT CLOSED is never indexed, an index whose content is
    UNCHANGED is not rewritten, and the folder is pruned to `KEEP_SESSIONS`.

    A derived, rebuildable artefact may never cost the page that wanted it, so
    every failure is logged and swallowed - the page has already painted from
    the rows this file was built out of.
    """
    session = str((index or {}).get("session_date") or "").strip()[:10]
    if not session:
        _log.info("A Day Review index with no session was not written.")
        return None
    if not _session_has_closed(session, now):
        _log.info(
            "The Day Review index for %s was not written: the session has not closed.",
            session,
        )
        return None
    path = index_path(session, root=root)
    temp = path.with_suffix(".json.tmp")
    try:
        body = _body(index)
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if _body(stored) == body:
                _log.debug("The Day Review index for %s is unchanged.", session)
                return path
            # The BODY is identical apart from the stamp: record the stamp in the
            # sidecar rather than rewriting 22 MB for a few hundred bytes.
            if _body({**dict(stored), "sources_stamp": index.get("sources_stamp")}) == body:
                _stamp_only(session, index, root=root)
                return path
    except Exception:  # noqa: BLE001 - an unreadable old file is simply replaced
        _log.debug("The stored Day Review index could not be compared.", exc_info=True)
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
    # A fresh body carries its own stamp, so any sidecar is now history.
    try:
        stamp_path(session, root=root).unlink(missing_ok=True)
    except OSError:
        pass
    # `index_path` is <base>/sessions/<date>/outcomes.json, so three parents up
    # is the base whether or not a root was named.
    _prune(path.parent.parent.parent)
    return path


__all__ = [
    "INDEXED_SOURCES",
    "INDEXED_SOURCE_SPECS",
    "KEEP_SESSIONS",
    "SCHEMA",
    "build_index",
    "default_root",
    "index_path",
    "is_stale",
    "read_index",
    "read_store",
    "refresh_stamp",
    "sources_stamp",
    "stamp_path",
    "stamp_verdict",
    "store_from_payload",
    "store_payload",
    "stores_for",
    "write_index",
]
