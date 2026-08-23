"""Append-only evidence ledgers - R10.A, ground rules 5 and 7.

An evidence ledger is the **authority** for what happened. A checkpoint (the
BounceBot pending dict, say) is a convenience that can be rebuilt from it; a
ledger is not rebuildable from anything, so it is written first, appended only,
and never rewritten. A correction is a **superseding event**, not an edit.

What that costs, and why each cost is paid here:

* **Schema by NAME, never by number.** `intraday_outcome_event_v1` means one
  thing forever. A changed meaning is a new name, so a report written in March
  and one written in September agree about what a word meant.
* **Every row carries its own time twice**: `event_at` is tz-aware UTC (machine
  order) and `session_date` is the market-local session (trading order). One
  without the other cannot answer "which session was this?" across a
  20:30-local write, and `astimezone` is used rather than `replace(tzinfo=None)`,
  which discards an offset instead of converting through it.
* **Every row says who wrote it** - host, pid, and the run id when the caller
  has one. When two desks ran concurrently on 2026-08-20 nothing in the outcome
  store could say so, and the duplicate rows had to be attributed by inference.
* **A torn line is counted, never skipped silently.** Power loss mid-append
  leaves a partial JSON line; the reader reports it as `unreadable` beside n, so
  a gap can never read as an absence of events.
* **Month segments** (`<stream>-YYYYMM.jsonl`), so retention is a file move and
  a reader never has to open a year to answer a question about a week.

Nothing here interprets an event. It stores and returns them.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SCHEMA_INTRADAY_OUTCOME_EVENT = "intraday_outcome_event_v1"

#: The one place a ledger may live. Cold-pushed by `push_cold_to_das.ps1`.
LEDGER_DIR_NAME = "evidence_ledgers"
#: 13 months hot (R10.0 decision 7 / trader answer Q5); older segments are cold.
HOT_MONTHS = 13


def default_ledger_dir() -> Path:
    from project_paths import RUNTIME_DATA_DIR

    return Path(RUNTIME_DATA_DIR) / LEDGER_DIR_NAME


def _market_tz():
    try:
        from market_calendar import MARKET_TZ

        return MARKET_TZ
    except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.12
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/New_York")


def market_session_date(moment: datetime) -> date:
    """The market-local calendar date of `moment`.

    `astimezone`, never `replace(tzinfo=None)`: a 20:30-local write on the 21st
    is 03:30 UTC on the 22nd, and only a conversion gets that right.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(_market_tz()).date()


@dataclass(frozen=True)
class ReadResult:
    """Rows, and what could not be read. The second half is not optional."""

    rows: tuple[dict, ...] = ()
    unreadable: int = 0
    files: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.rows)

    @property
    def coverage_note(self) -> str:
        if not self.unreadable:
            return f"n={len(self.rows)}"
        return f"n={len(self.rows)} ({self.unreadable} unreadable line(s) - a gap, not an absence)"


@dataclass
class EvidenceLedger:
    """One append-only stream, segmented by month.

    Thread-safe within a process by a lock. **Across** processes the guarantee is
    weaker and deliberately so: an append is a single `write` of one line opened
    in append mode, which POSIX and Windows both keep atomic for small writes, so
    concurrent writers interleave whole rows rather than corrupting each other.
    Ordering between processes is not guaranteed and must not be assumed - that
    is what `event_at` is for.
    """

    stream: str
    schema: str
    directory: Path | None = None
    run_id: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not str(self.stream or "").strip():
            raise ValueError("a ledger needs a stream name")
        if not str(self.schema or "").strip():
            raise ValueError("a ledger needs a schema NAME (never a bare version number)")
        self.directory = Path(self.directory) if self.directory else default_ledger_dir()

    # -- paths ------------------------------------------------------------
    def segment_for(self, moment: datetime) -> Path:
        session = market_session_date(moment)
        return Path(self.directory) / f"{self.stream}-{session.strftime('%Y%m')}.jsonl"

    def segments(self) -> tuple[Path, ...]:
        try:
            return tuple(sorted(Path(self.directory).glob(f"{self.stream}-*.jsonl")))
        except OSError:
            return ()

    # -- writing ----------------------------------------------------------
    def append(self, event: Mapping[str, Any], *, now: datetime | None = None) -> dict:
        """Append one event and return the row exactly as it was written.

        The caller's fields are copied, never mutated, and the ledger's own
        fields are applied **last** so a caller cannot overwrite the schema name,
        the timestamps or the writer identity - a row that can lie about who
        wrote it is not evidence.
        """
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        row = dict(event or {})
        row.update(
            {
                "schema": self.schema,
                "event_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
                "session_date": market_session_date(moment).isoformat(),
                "writer_host": socket.gethostname(),
                "writer_pid": os.getpid(),
            }
        )
        if self.run_id:
            row["run_id"] = self.run_id
        line = json.dumps(row, default=str, separators=(",", ":"), sort_keys=True)
        path = self.segment_for(moment)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return row

    def append_many(self, events: Iterable[Mapping[str, Any]], *, now: datetime | None = None) -> int:
        return sum(1 for event in events if self.append(event, now=now))

    # -- reading ----------------------------------------------------------
    def read(
        self,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        event_types: Iterable[str] | None = None,
    ) -> ReadResult:
        """Every row in `[start, end]` by `session_date`, plus what could not be read.

        A row whose `session_date` is missing or unparseable is **kept** when no
        window is asked for and **excluded and counted as unreadable** when one
        is: a row that cannot say which session it belongs to cannot be claimed
        by a window.
        """
        wanted = {str(name) for name in event_types} if event_types else None
        low = _as_date(start)
        high = _as_date(end)
        rows: list[dict] = []
        unreadable = 0
        files: list[str] = []
        for path in self.segments():
            files.append(path.name)
            try:
                handle = path.open("r", encoding="utf-8")
            except OSError:
                unreadable += 1
                continue
            with handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        # A torn line - power loss mid-append. Counted, because a
                        # silently skipped row makes a gap look like an absence.
                        unreadable += 1
                        continue
                    if not isinstance(row, dict):
                        unreadable += 1
                        continue
                    if wanted is not None and str(row.get("event_type")) not in wanted:
                        continue
                    if low is not None or high is not None:
                        session = _as_date(row.get("session_date"))
                        if session is None:
                            unreadable += 1
                            continue
                        if low is not None and session < low:
                            continue
                        if high is not None and session > high:
                            continue
                    rows.append(row)
        return ReadResult(rows=tuple(rows), unreadable=unreadable, files=tuple(files))

    # -- housekeeping -----------------------------------------------------
    def cold_segments(self, *, today: date | None = None, hot_months: int = HOT_MONTHS) -> tuple[Path, ...]:
        """Segments older than the hot window. **Listed, never deleted here.**

        Naming what is cold is a reader's job; moving it is the cold push's, and
        deleting it is nobody's - the retention decision belongs to the trader.
        """
        moment = today or market_session_date(datetime.now(timezone.utc))
        cutoff = (moment.year * 12 + moment.month) - int(hot_months)
        cold: list[Path] = []
        for path in self.segments():
            stamp = path.stem.rsplit("-", 1)[-1]
            try:
                year, month = int(stamp[:4]), int(stamp[4:6])
            except (ValueError, IndexError):
                continue
            if year * 12 + month <= cutoff:
                cold.append(path)
        return tuple(cold)


def _as_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def intraday_outcome_ledger(directory: Path | None = None, *, run_id: str = "") -> EvidenceLedger:
    """The intraday outcome authority (R10.A).

    The legacy CSV keeps being written beside it during the canary; this is the
    store that will be believed when they disagree.
    """
    return EvidenceLedger(
        stream="intraday_outcome_events",
        schema=SCHEMA_INTRADAY_OUTCOME_EVENT,
        directory=directory,
        run_id=run_id,
    )
