"""Setup permutation key on M5 alerts (WISHLIST P1-4 4a): a shadow sidecar.

The M5 outcome writer (`bounce_bot_lib.legacy`) calls `submit(row)` once per
outcome row, after the row is on disk. The first row of each ``event_id`` is
queued; one daemon worker looks up the PREVIOUS session's last D1 scan row for
the name and side (the backfill's rule, `setup_permutation_backfill`) and
appends one jsonl record to `project_paths.M5_SETUP_KEY_STAMPS_FILE`.

Shadow only: nothing here reads or changes the outcome row, and nothing reads
the stamp for an alert, a grade or a score. `submit` never raises and never
waits; a full queue or a failed lookup loses the stamp, never the row. With no
scan row for the name, or no stamp record at all, the key is unknown.

A scan run on the alert's own day (before the alert) can write rows dated the
previous session (the pre-market and after-close scans use the last completed
bar); those rows count, as they do in the backfill, because they were on disk
before the alert and hold only completed bars.

Memory: only the previous session's lines are kept (raw), only the columns the
rule needs are split out, and only each representative row is parsed whole,
one at a time. Review events are streamed for that session only.

Owner: this module is the only writer of the sidecar (append-only).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "m5_setup_key_stamp_v1"
STATUS_STAMPED = "stamped"
STATUS_NO_SCAN_ROW = "no_scan_row"
STATUS_FAILED = "lookup_failed"
STAMP_FIELDS = ("permutation_key", "permutation_label", "permutation_rule_version")
#: Rows waiting for the worker; past this the stamp is dropped, never the caller delayed.
QUEUE_MAX = 5000
#: How far back from the end of the D1 history the previous session may lie.
MAX_TAIL_BYTES = 256 << 20
#: A failed load is not retried for this long (a persistent failure never re-reads per alert).
FAILURE_BACKOFF_SECONDS = 300.0
#: The columns the backfill's representative rule reads (`representatives_of`, `scan_row_id`).
_RULE_COLUMNS = ("symbol", "side", "last_trade_date", "run_date", "last_close", "run_timestamp", "run_id")
#: Set to "0" to switch the hook off (tests, a bad day).
ENABLED_ENV = "TRADINGBOTV3_M5_SETUP_KEY_STAMP"

_lock = threading.Lock()
_queue: "queue.Queue[dict]" = queue.Queue(maxsize=QUEUE_MAX)
_worker: threading.Thread | None = None
_seen: set[str] = set()
_logged_reasons: set[str] = set()


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _side(direction: Any) -> str:
    text = _text(direction).upper()
    return text if text in {"LONG", "SHORT"} else ""


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "1").strip() != "0"


def _log_once(reason: str) -> None:
    """One warning per distinct reason per process: a broken lookup never floods the log."""
    with _lock:
        if reason in _logged_reasons:
            return
        _logged_reasons.add(reason)
    logging.warning("M5 setup key stamp skipped: %s", reason)


# --- the lookup (worker thread only)


class ScanKeyLookup:
    """Keys for one previous session, cached until the D1 history file changes.

    ``history_path`` is the append-ordered `d1_features_history.csv`; only its
    tail back to the previous session is read. ``context_loader(session)``
    gives the `SessionContext` the backfill would use for that session.
    """

    def __init__(
        self,
        history_path: Path | None = None,
        *,
        context_loader: Callable[[str], Any] | None = None,
        max_tail_bytes: int = MAX_TAIL_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._history_path = history_path
        self._context_loader = context_loader
        self._max_tail_bytes = max_tail_bytes
        self._clock = clock
        self._cache: dict[str, tuple[Any, dict[tuple[str, str], dict]]] = {}
        self._failure: tuple[str, float, Exception] | None = None
        self.loads = 0

    def history_path(self) -> Path:
        if self._history_path is not None:
            return Path(self._history_path)
        from project_paths import D1_FEATURES_HISTORY_FILE

        return Path(D1_FEATURES_HISTORY_FILE)

    def _context(self, session: str):
        if self._context_loader is not None:
            return self._context_loader(session)
        import setup_permutation_context as spc

        return spc.SessionContext.load(session, stream_review_events=True)

    def keys_for_session(self, session: str) -> dict[tuple[str, str], dict]:
        """``{(SYMBOL, SIDE): stamp}`` for every representative row of ``session``. Raises on failure."""

        path = self.history_path()
        stat = path.stat()
        signature = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(session)
        if cached is not None and cached[0] == signature:
            return cached[1]
        failure = self._failure
        if failure is not None and failure[0] == session and self._clock() - failure[1] < FAILURE_BACKOFF_SECONDS:
            raise failure[2]
        try:
            keys = self._load(path, session)
        except FileNotFoundError:
            raise
        except Exception as exc:
            self._failure = (session, self._clock(), exc)
            raise
        self._failure = None
        self._cache = {session: (signature, keys)}  # one session at a time
        self.loads += 1
        return keys

    def _load(self, path: Path, session: str) -> dict[tuple[str, str], dict]:
        import csv
        import io

        import setup_permutation_backfill as bf
        import setup_permutation_context as spc

        fieldnames, lines = spc._tail_lines_for_session(
            path, session, date_columns=("last_trade_date", "run_date"),
            stamp_columns=("run_timestamp", "run_id"), max_bytes=self._max_tail_bytes,
        )
        wanted = [(name, fieldnames.index(name)) for name in _RULE_COLUMNS if name in fieldnames]

        def parse(raw: bytes) -> list[str]:
            return next(csv.reader(io.StringIO(raw.decode("utf-8", errors="replace"))), [])

        def slim(values: list[str]) -> dict[str, str]:
            return {name: values[index] if index < len(values) else "" for name, index in wanted}

        representatives = bf.representatives_of(slim(parse(raw)) for raw in lines)
        context = self._context(session)
        keys: dict[tuple[str, str], dict] = {}
        for index, (symbol, side, rep_session) in representatives.items():
            if rep_session != session:
                continue
            row = dict(zip(fieldnames, parse(lines[index]), strict=False))
            key = bf.key_scan_row(row, symbol, side, session, context)
            keys[(symbol, side)] = {
                "permutation_key": key.compact_key,
                "permutation_label": key.label,
                "permutation_rule_version": key.permutation_rule_version,
                "d1_family": key.family,
                "facets": key.as_dict(),
                "scan_row_id": bf.scan_row_id(row),
            }
        return keys

    def stamp(self, symbol: str, side: str, trade_date: str) -> dict:
        """The sidecar fields for one alert. Never raises: a failure is a blank stamp with its reason."""
        import setup_permutation_backfill as bf

        previous = bf.previous_session_text(trade_date)
        out: dict[str, Any] = {"d1_session": previous, **{name: "" for name in STAMP_FIELDS}}
        if not previous or not symbol or not side:
            out.update(status=STATUS_NO_SCAN_ROW, reason="no previous session, symbol or side")
            return out
        try:
            found = self.keys_for_session(previous).get((symbol, side))
        except FileNotFoundError:
            out.update(status=STATUS_NO_SCAN_ROW, reason="no D1 history file")
            return out
        except Exception as exc:  # noqa: BLE001 - a lookup failure costs the stamp, never the row
            reason = f"{type(exc).__name__}: {exc}"
            _log_once(reason)
            out.update(status=STATUS_FAILED, reason=reason)
            return out
        if found is None:
            out.update(status=STATUS_NO_SCAN_ROW, reason="no scan row for the name and side")
            return out
        out.update(found)
        out["status"] = STATUS_STAMPED
        return out


def record_for(row: Mapping[str, Any], lookup: ScanKeyLookup) -> dict | None:
    """The sidecar record for one outcome row, or None when the row has no event id."""
    event_id = _text(row.get("event_id"))
    if not event_id:
        return None
    symbol = _text(row.get("symbol")).upper()
    side = _side(row.get("direction"))
    trade_date = _text(row.get("trade_date"))[:10]
    return {
        "schema": SCHEMA,
        "event_id": event_id,
        "symbol": symbol,
        "side": side,
        "trade_date": trade_date,
        **lookup.stamp(symbol, side, trade_date),
        "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_record(record: Mapping[str, Any], path: Path | None = None) -> None:
    from project_paths import M5_SETUP_KEY_STAMPS_FILE

    target = Path(path) if path is not None else Path(M5_SETUP_KEY_STAMPS_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n").encode("ascii")
    # Unbuffered: the whole line goes down in one write() call, never a partial line.
    with target.open("ab", buffering=0) as handle:
        handle.write(line)


def read_stamps(path: Path) -> dict[str, dict]:
    """``{event_id: record}``, the FIRST record per event (the one made at alert time)."""
    out: dict[str, dict] = {}
    target = Path(path)
    if not target.is_file():
        return out
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("schema") == SCHEMA:
                out.setdefault(_text(record.get("event_id")), record)
    return out


# --- the hook and its one worker


_lookup: ScanKeyLookup | None = None


def _run() -> None:
    global _lookup
    while True:
        item = _queue.get()
        try:
            if _lookup is None:
                _lookup = ScanKeyLookup()
            record = record_for(item, _lookup)
            if record is not None:
                append_record(record)
        except Exception as exc:  # noqa: BLE001 - the worker outlives any one bad row
            _log_once(f"{type(exc).__name__}: {exc}")
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_run, name="m5-setup-key-stamp", daemon=True)
        _worker.start()


def _market_today() -> str:
    from market_calendar import MARKET_TZ

    return datetime.now(MARKET_TZ).date().isoformat()


def submit(row: Mapping[str, Any]) -> bool:
    """Queue the first outcome row of today's event for stamping. Never raises, never waits.

    Only four plain strings are copied off ``row``; the row itself is not kept.
    """
    try:
        if not enabled():
            return False
        event_id = _text(row.get("event_id"))
        if not event_id or _text(row.get("trade_date"))[:10] != _market_today():
            return False  # an old session's row (the startup sweep) is left to the backfill
        with _lock:
            if event_id in _seen:
                return False
            _seen.add(event_id)
        item = {
            "event_id": event_id,
            "symbol": _text(row.get("symbol")),
            "direction": _text(row.get("direction")),
            "trade_date": _text(row.get("trade_date")),
        }
        _ensure_worker()
        _queue.put_nowait(item)
        return True
    except queue.Full:
        _log_once("queue full")
        return False
    except Exception as exc:  # noqa: BLE001 - the hook never costs the caller
        _log_once(f"{type(exc).__name__}: {exc}")
        return False


def drain(timeout: float = 10.0) -> bool:
    """Wait until the queue is empty (tests and CLIs only; never from the bot loop)."""
    deadline = time.monotonic() + timeout
    while _queue.unfinished_tasks:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def reset_for_tests(lookup: ScanKeyLookup | None = None) -> None:
    global _lookup
    drain(5.0)
    with _lock:
        _seen.clear()
        _logged_reasons.clear()
    _lookup = lookup


__all__ = [
    "SCHEMA",
    "ScanKeyLookup",
    "append_record",
    "drain",
    "read_stamps",
    "record_for",
    "submit",
]
