"""Lossless packing store for ``d1_features_history.csv`` (trader 2026-10-02).

"I want a solution where I lose nothing." The live CSV grows ~20 MB a trading
day and every scan re-reads all of it. This module packs it, without loss, into
one Parquet file per calendar month of ``run_date`` plus a manifest, and can
then trim the live CSV to the last ``DEFAULT_TRIM_KEEP_DAYS`` days.

WHAT "LOSSLESS" MEANS HERE
--------------------------
* Every row, including repeat same-day scans and duplicate rows, in the
  original append order: each archived row carries ``SEQ_COLUMN``, its position
  in the file's whole history (row 0 is the first row ever appended).
* Every column and every cell's exact text: cells are read as strings with no
  type inference and no NaN conversion, so ``""`` stays ``""``, ``1.50`` stays
  ``1.50`` and ``NA`` stays ``NA``. A column the archive file never saw (the
  writer widened the header later) reads back as ``""``, which is what the
  writer's own widening rewrite leaves in those cells.
* One caveat, stated rather than hidden: the WRITER's widening rewrite
  (``legacy._write_d1_feature_history_locked``) re-reads the whole CSV through
  pandas inference and writes it back, which turns ``1.50`` into ``1.5``, a
  blank-holding int column's ``3`` into ``3.0`` and the text ``NA`` into ``""``.
  Rows archived BEFORE such a rewrite keep the text they had when packed - the
  more faithful text. ``read_history(typed=True)`` gives the same numbers either
  way.

THE LIVE FILE AND THE ARCHIVE
-----------------------------
The manifest's ``live_offset`` is the history position of the live file's row 0
(0 until the first trim). Rows ``[0, archived_rows)`` are in the archive; live
rows at or past ``archived_rows`` are not yet. This holds because the live file
only ever changes by appends, by the writer's widening rewrite (same rows, same
order) and by ``trim``, which removes a HEAD of rows already proven in the
archive and records the new offset. Anything else - rows deleted or reordered
by hand - is caught by the key check (``KEY_COLUMNS`` of every live row that is
already archived must match the archive at that position) and refused loudly.

"30 DAYS"
---------
``trim`` removes a row only when ALL of these hold: it is in the archive and
proven cell-for-cell against it; its ``run_date`` parses as ``YYYY-MM-DD``; and
that date is more than ``keep_days`` calendar days before ``today`` (with 30 on
2026-10-02, 2026-09-01 goes and 2026-09-02 stays). It removes only the head of
the file - it stops at the first row that must stay - so a row with a blank or
unparseable ``run_date`` is never dropped and holds everything after it live
(the result says ``stopped_by: "undated"``). The live history is append-ordered
by date (211,005 rows, zero out of order on 2026-10-02), so the head is the
old part.

LOCKS, ATOMICITY, FAILURE
-------------------------
``archive`` and ``trim`` take the writer's own lock
(``local_writer_lock(lock_key_for_path(csv))``, the key
``append_d1_feature_history`` uses) and then the archive's lock, and refuse
when either is unavailable. Every file goes down through a temp sibling and
``os.replace``; the manifest is written last and is the commit. A failed verify
deletes the new files and leaves the last good archive and manifest untouched.

``read_history`` takes no lock. It reads the manifest, then the files, then the
manifest again, and retries when a pack or trim landed in between.

MORE THAN ONE STORE (2026-10-02)
--------------------------------
The same core packs any append-only CSV described by a ``StoreSpec`` (CSV,
archive folder, date column, key columns, the writer's lock, keep days, trim
setting); ``registered_stores()`` lists the ones the CLI and the night slot
use. Everything above holds per store, with two differences for a store whose
writer takes NO lock (the bounce outcomes CSV): archive takes only the
archive's lock and reads a length snapshot cut at the last complete record,
so an append in flight is left for the next run; and trim refuses, always.

CLI: ``python scripts/d1_feature_history_archive.py status|archive|verify|trim
[--store NAME] [--apply]`` - the store defaults to the D1 history; trim is a
dry run unless ``--apply``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

try:  # scripts/ on sys.path (the desk, the night runner, tests)
    from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
except ImportError:  # pragma: no cover - run as a file from elsewhere
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path

_log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = 1
#: The original history position of each archived row (0 = first row ever).
SEQ_COLUMN = "_archive_seq"
DATE_COLUMN = "run_date"
#: The identity checked between a live row and its archived copy.
KEY_COLUMNS = ("run_id", "symbol", "side")
#: The archive file for rows whose run_date is blank or unparseable.
UNDATED = "undated"
#: Calendar days of run_date the live CSV keeps after a trim.
DEFAULT_TRIM_KEEP_DAYS = 30
#: CSV parse block. Bounded memory: one block (plus its parsed copy) at a time.
#: Measured on the 900 MB file (2026-10-02): 16 MB peaked at 1.3 GB RSS, 4 MB
#: at ~0.8 GB for ~7 s more.
CSV_BLOCK_BYTES = 4 << 20
#: How long archive/trim wait for the writer's lock before refusing.
LOCK_TIMEOUT_SECONDS = 60.0
PARQUET_COMPRESSION = "zstd"
_READ_RETRIES = 4
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
#: pandas' default NA strings: its widening rewrite writes each of them back as "".
_PANDAS_NA_TEXT = frozenset(
    {
        "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
        "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
    }
)


class ArchiveError(RuntimeError):
    """The archive could not be changed safely; nothing was changed."""


class ArchiveRefused(ArchiveError):
    """A precondition failed (lock, header, verification); nothing was changed."""


class VerifyFailed(ArchiveError):
    """The archive does not match its source; the last good archive is kept."""


# ---------------------------------------------------------------------------
# Store specs: one append-only CSV and its archive
# ---------------------------------------------------------------------------

#: ``StoreSpec.writer_lock`` value: the writer takes
#: ``local_writer_lock(lock_key_for_path(csv_path))`` around every write.
WRITER_LOCK_PATH = "path"
#: ``StoreSpec.writer_lock`` value: the writer takes no lock. Archive reads a
#: length snapshot cut at the last complete record; trim always refuses.
WRITER_LOCK_NONE = "none"
D1_STORE = "d1_features_history"


@dataclass(frozen=True)
class StoreSpec:
    """One append-only CSV packed by this module.

    ``writer_lock`` is ``WRITER_LOCK_PATH``, ``WRITER_LOCK_NONE`` or a literal
    lock key the writer takes. ``trim_setting`` names the local setting that
    lets the night trim (empty: the night never trims this store).
    """

    name: str
    csv_path: Path
    archive_dir: Path
    date_column: str = DATE_COLUMN
    key_columns: tuple[str, ...] = KEY_COLUMNS
    writer_lock: str = WRITER_LOCK_PATH
    keep_days: int = DEFAULT_TRIM_KEEP_DAYS
    trim_setting: str = ""

    @property
    def writer_lock_key(self) -> str | None:
        if self.writer_lock == WRITER_LOCK_NONE:
            return None
        if self.writer_lock == WRITER_LOCK_PATH:
            return lock_key_for_path(self.csv_path)
        return self.writer_lock

    def with_paths(self, csv_path: Any = None, archive_dir: Any = None) -> StoreSpec:
        return replace(
            self,
            csv_path=Path(csv_path) if csv_path is not None else Path(self.csv_path),
            archive_dir=Path(archive_dir) if archive_dir is not None else Path(self.archive_dir),
        )


def registered_stores() -> dict[str, StoreSpec]:
    """Every store the CLI and the night slot pack, by name (paths resolved now).

    ``intraday_bounce_candidates.csv`` is deliberately NOT here: the bot's
    startup ``compact_bounce_candidates_csv`` rewrites it and drops rows from
    the middle (old near_miss rows), which a position-ordered archive cannot
    follow. It needs that compaction changed first (ask-first file).
    """
    import project_paths as pp

    return {
        D1_STORE: StoreSpec(
            name=D1_STORE,
            csv_path=Path(pp.D1_FEATURES_HISTORY_FILE),
            archive_dir=Path(pp.D1_FEATURES_HISTORY_ARCHIVE_DIR),
            trim_setting="d1_history_trim_enabled",
        ),
        "intraday_bounce_outcomes": StoreSpec(
            name="intraday_bounce_outcomes",
            csv_path=Path(pp.INTRADAY_BOUNCE_OUTCOMES_FILE),
            archive_dir=Path(pp.INTRADAY_BOUNCE_OUTCOMES_ARCHIVE_DIR),
            date_column="trade_date",
            # Measured 2026-10-02: event_id alone repeats 595,631 times in 628,945
            # rows (one event, many milestone rows); with event_type and
            # logged_at it repeats once. A key check needs a match, not uniqueness.
            key_columns=("event_id", "event_type", "logged_at"),
            # `bounce_bot_lib.legacy._append_learning_row` takes no lock.
            writer_lock=WRITER_LOCK_NONE,
        ),
    }


def _spec(store: Any = None, csv_path: Any = None, archive_dir: Any = None) -> StoreSpec:
    """Resolve a store name or spec, then apply explicit path overrides."""
    if isinstance(store, StoreSpec):
        base = store
    else:
        name = store or D1_STORE
        stores = registered_stores()
        if name not in stores:
            raise ArchiveRefused(f"unknown store {name!r}; registered: {sorted(stores)}")
        base = stores[name]
    return base.with_paths(csv_path, archive_dir)


# ---------------------------------------------------------------------------
# Paths, manifest, locks
# ---------------------------------------------------------------------------


def _empty_manifest() -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "generation": 0,
        "archived_rows": 0,
        "live_offset": 0,
        "pending_live_offset": None,
        "columns": [],
        "files": {},
        "updated_at": "",
    }


def load_manifest(archive_dir: Path) -> dict[str, Any]:
    path = Path(archive_dir) / MANIFEST_NAME
    if not path.exists():
        return _empty_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerifyFailed(f"archive manifest unreadable ({exc})") from exc
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        raise VerifyFailed(f"archive manifest has an unknown schema: {data.get('schema')!r}")
    manifest = _empty_manifest()
    manifest.update(data)
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_bytes_atomic(path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with open(temp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        _unlink_quietly(temp)


def _unlink_quietly(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        _log.warning("d1 history archive: could not remove %s (%s)", path, exc)


def _manifest_signature(manifest: dict[str, Any]) -> tuple:
    return (
        manifest.get("generation"),
        manifest.get("archived_rows"),
        manifest.get("live_offset"),
        manifest.get("pending_live_offset"),
    )


@contextmanager
def _locks(spec: StoreSpec, timeout: float):
    """The writer's own lock first (when it has one), then the archive's. Refuses, never waits forever."""
    with ExitStack() as stack:
        try:
            if spec.writer_lock_key is not None:
                stack.enter_context(local_writer_lock(spec.writer_lock_key, timeout_seconds=timeout))
            stack.enter_context(
                local_writer_lock(lock_key_for_path(spec.archive_dir / MANIFEST_NAME), timeout_seconds=timeout)
            )
        except LocalLockUnavailable as exc:
            raise ArchiveRefused(f"writer lock unavailable ({exc}); nothing changed") from exc
        yield


def _now_text(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Reading the CSV as text, in bounded blocks
# ---------------------------------------------------------------------------


def read_header(csv_path: Path) -> list[str]:
    """The live header, or ArchiveRefused. Duplicate or blank names are refused."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration, csv.Error, UnicodeDecodeError) as exc:
        raise ArchiveRefused(f"history header unreadable ({type(exc).__name__}: {exc})") from exc
    if not header or any(not name for name in header):
        raise ArchiveRefused("history header has a blank column name")
    if header[0].startswith("﻿"):
        raise ArchiveRefused("history header starts with a byte-order mark")
    if len(set(header)) != len(header):
        raise ArchiveRefused("history header repeats a column name")
    return header


class _Bounded(io.RawIOBase):
    """The first ``limit`` bytes of a file: what an archive run may read of a
    file a no-lock writer is still appending to."""

    def __init__(self, path: Path, limit: int):
        self._handle = open(path, "rb")
        self._left = int(limit)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self._left <= 0:
            return 0
        view = memoryview(buffer)[: self._left]
        got = self._handle.readinto(view)
        self._left -= got or 0
        return got or 0

    def close(self) -> None:
        self._handle.close()
        super().close()


def _snapshot_end(csv_path: Path, *, writer_locked: bool) -> int:
    """The byte length an archive run reads: complete records only.

    Holding the writer's lock, the file is still and the whole length is used.
    With no writer lock, the length is snapshot ONCE and cut back to the end of
    the last complete record (quote-aware), so a row the writer is halfway
    through appending is never read - it is packed on a later night.
    """
    size = csv_path.stat().st_size
    if writer_locked:
        return size
    end = 0
    quotes = 0
    offset = 0
    with open(csv_path, "rb") as handle:
        while offset < size:
            chunk = handle.read(min(8 << 20, size - offset))
            if not chunk:
                break
            position = chunk.rfind(b"\n")
            while position >= 0:
                if (quotes + chunk.count(b'"', 0, position + 1)) % 2 == 0:
                    end = offset + position + 1
                    break
                position = chunk.rfind(b"\n", 0, position)
            quotes += chunk.count(b'"')
            offset += len(chunk)
    return end


def _iter_csv(
    csv_path: Path,
    header: list[str],
    include: Iterable[str] | None = None,
    block_bytes: int = CSV_BLOCK_BYTES,
    limit: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream the CSV as all-string batches. Exact text: no inference, no NA.

    ``limit``: read only the first ``limit`` bytes (a no-lock store's snapshot).
    """
    include_list = [c for c in header if c in set(include)] if include is not None else None
    if limit is not None and limit <= 0:
        return
    source = _Bounded(csv_path, limit) if limit is not None else None
    try:
        yield from _iter_source(source if source is not None else str(csv_path), header, include_list, block_bytes)
    finally:
        if source is not None:
            source.close()


def _iter_source(source: Any, header: list[str], include_list, block_bytes: int) -> Iterator[pa.RecordBatch]:
    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(block_size=int(block_bytes), use_threads=False),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(
            column_types={name: pa.string() for name in header},
            include_columns=include_list,
            null_values=[],
            strings_can_be_null=False,
            quoted_strings_can_be_null=False,
        ),
    )
    try:
        for batch in reader:
            if batch.num_rows:
                yield batch
    finally:
        reader.close()


class _Cursor:
    """Take exactly ``n`` rows at a time from a stream of record batches."""

    def __init__(self, batches: Iterable[pa.RecordBatch]):
        self._it = iter(batches)
        self._pending: list[pa.RecordBatch] = []
        self._count = 0

    def take(self, n: int) -> pa.Table | None:
        while self._count < n:
            batch = next(self._it, None)
            if batch is None:
                break
            self._pending.append(batch)
            self._count += batch.num_rows
        if not self._pending:
            return None
        table = pa.Table.from_batches(self._pending)
        head, tail = table.slice(0, n), table.slice(n)
        self._pending = tail.to_batches() if tail.num_rows else []
        self._count = tail.num_rows
        return head

    def exhausted(self) -> bool:
        return self.take(1) is None

    def close(self) -> None:
        """Finish the underlying generator now, so Windows lets go of the file."""
        closer = getattr(self._it, "close", None)
        if closer is not None:
            closer()
        self._pending = []


@contextmanager
def _closing_all(cursors: dict[str, _Cursor] | list[_Cursor]):
    try:
        yield cursors
    finally:
        for cursor in list(cursors.values() if isinstance(cursors, dict) else cursors):
            cursor.close()


def _parse_run_date(text: str) -> date | None:
    match = _DATE_RE.match(text or "")
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _month_of(text: str) -> str:
    parsed = _parse_run_date(text)
    return f"{parsed.year:04d}-{parsed.month:02d}" if parsed else UNDATED


def _month_keys(batch: pa.RecordBatch | pa.Table, date_column: str = DATE_COLUMN) -> list[str]:
    if date_column not in batch.schema.names:
        return [UNDATED] * batch.num_rows
    return [_month_of(text) for text in batch.column(date_column).to_pylist()]


def _split_by_month(table: pa.Table, keys: list[str]) -> dict[str, pa.Table]:
    index: dict[str, list[int]] = {}
    for position, key in enumerate(keys):
        index.setdefault(key, []).append(position)
    if len(index) == 1:
        return {next(iter(index)): table}
    return {key: table.take(pa.array(rows, pa.int64())) for key, rows in index.items()}


# ---------------------------------------------------------------------------
# Archive files
# ---------------------------------------------------------------------------


def _archive_schema(columns: list[str]) -> pa.Schema:
    return pa.schema([pa.field(SEQ_COLUMN, pa.int64())] + [pa.field(c, pa.string()) for c in columns])


def _archive_table(live_rows: pa.Table, seqs: list[int], columns: list[str]) -> pa.Table:
    """Live text rows -> an archive table: the seq column, then every column."""
    arrays = [pa.array(seqs, pa.int64())]
    for name in columns:
        if name in live_rows.schema.names:
            arrays.append(live_rows.column(name).combine_chunks())
        else:
            arrays.append(pa.nulls(live_rows.num_rows, pa.string()))
    return pa.Table.from_arrays(arrays, schema=_archive_schema(columns))


def _conform(table: pa.Table, columns: list[str], *, with_seq: bool = True) -> pa.Table:
    """Reorder to ``columns``, adding all-null columns a file never had."""
    names = ([SEQ_COLUMN] if with_seq else []) + list(columns)
    arrays = []
    for name in names:
        if name in table.schema.names:
            arrays.append(table.column(name))
        else:
            arrays.append(pa.nulls(table.num_rows, pa.int64() if name == SEQ_COLUMN else pa.string()))
    return pa.Table.from_arrays(arrays, names=names)


def _file_batches(path: Path, columns: list[str], seq_from: int = 0) -> Iterator[pa.RecordBatch]:
    """An archive file's rows (seq-ascending) from ``seq_from`` on, conformed."""
    handle = pq.ParquetFile(str(path))
    present = [c for c in [SEQ_COLUMN, *columns] if c in handle.schema_arrow.names]
    try:
        for batch in handle.iter_batches(batch_size=8192, columns=present):
            table = pa.Table.from_batches([batch])
            if seq_from:
                table = table.filter(pc.greater_equal(table.column(SEQ_COLUMN), seq_from))
            if table.num_rows:
                yield from _conform(table, columns).to_batches()
    finally:
        handle.close()


def _verify_files(manifest: dict[str, Any], archive_dir: Path) -> list[str]:
    """sha256, row counts, seq ranges, and seqs 0..archived_rows-1 exactly once."""
    problems: list[str] = []
    total = 0
    seq_arrays = []
    for month, entry in sorted(manifest.get("files", {}).items()):
        path = archive_dir / str(entry.get("file") or "")
        if not path.is_file():
            problems.append(f"{month}: archive file {path.name} is missing")
            continue
        if _sha256(path) != entry.get("sha256"):
            problems.append(f"{month}: sha256 of {path.name} does not match the manifest")
            continue
        try:
            seqs = pq.read_table(str(path), columns=[SEQ_COLUMN]).column(SEQ_COLUMN).combine_chunks()
        except Exception as exc:  # noqa: BLE001 - any unreadable file is a problem, reported
            problems.append(f"{month}: {path.name} unreadable ({exc})")
            continue
        if len(seqs) != int(entry.get("rows", -1)):
            problems.append(f"{month}: {len(seqs)} rows on disk, manifest says {entry.get('rows')}")
        values = seqs.to_numpy(zero_copy_only=False)
        if len(values) > 1 and not (values[1:] > values[:-1]).all():
            problems.append(f"{month}: rows are not in original order")
        total += len(values)
        seq_arrays.append(values)
    archived = int(manifest.get("archived_rows", 0))
    if not problems:
        if total != archived:
            problems.append(f"archive holds {total} rows, manifest says {archived}")
        elif seq_arrays:
            import numpy as np

            every = np.sort(np.concatenate(seq_arrays))
            if not (every == np.arange(archived)).all():
                problems.append("archive positions are not exactly 0..archived_rows-1")
    return problems


# ---------------------------------------------------------------------------
# Comparing live rows with archived rows
# ---------------------------------------------------------------------------


def _cells_prove(live_text: str, archived: str | None) -> bool:
    """Is this live cell the archived cell, allowing for the writer's rewrite?"""
    if archived is None:
        return live_text == ""
    if live_text == archived:
        return True
    if live_text == "" and archived in _PANDAS_NA_TEXT:
        return True
    try:
        a, b = float(live_text), float(archived)
    except ValueError:
        return False
    return a == b or (math.isnan(a) and math.isnan(b))


def _keys_match(live_text: str, archived: str | None) -> bool:
    """The key check's equivalence: narrower than the trim proof on purpose.

    Exact text, or a live "" where the archive holds pandas-NA text (or no
    cell): the one thing the writer's widening rewrite does to a key, e.g. a
    ticker literally named ``NA``. Never float equality - '0' and '0.0' are
    different keys.
    """
    if archived is None:
        return live_text == ""
    return live_text == archived or (live_text == "" and archived in _PANDAS_NA_TEXT)


def _compare(live: pa.Table, arch: pa.Table, seqs: list[int], columns: list[str], mode: str) -> str | None:
    """None when ``arch`` holds ``live`` at ``seqs``; else the first difference.

    ``exact``: identical text, and columns the live row lacks are null.
    ``proof``: identical, or equal after the writer's widening rewrite.
    ``keys``: identical, or live "" for archived pandas-NA text (`_keys_match`).
    """
    match = _keys_match if mode == "keys" else _cells_prove
    if arch is None or arch.num_rows != live.num_rows:
        return f"archive has {0 if arch is None else arch.num_rows} rows where the live file has {live.num_rows}"
    got = arch.column(SEQ_COLUMN).to_pylist()
    if got != seqs:
        return f"history positions differ near {seqs[0]} (archive {got[:3]}...)"
    for name in columns:
        if name not in live.schema.names:
            if mode == "exact" and arch.column(name).null_count != arch.num_rows:
                return f"column {name!r} is filled in the archive but absent from the live row"
            continue
        live_col = live.column(name)
        arch_col = arch.column(name)
        if mode == "exact":
            if arch_col.null_count or not arch_col.equals(live_col):
                return f"column {name!r} differs near history position {seqs[0]}"
            continue
        equal = pc.fill_null(pc.equal(arch_col, live_col), False)
        if pc.all(equal).as_py():
            continue
        live_values = live_col.to_pylist()
        arch_values = arch_col.to_pylist()
        for i, ok in enumerate(equal.to_pylist()):
            if not ok and not match(live_values[i], arch_values[i]):
                return (
                    f"column {name!r} at history position {seqs[i]}: live {live_values[i]!r} "
                    f"vs archived {arch_values[i]!r}"
                )
    return None


def _check_live_against_archive(
    spec: StoreSpec,
    header: list[str],
    manifest: dict[str, Any],
    *,
    live_offset: int,
    mode: str,
    stop_row: int | None = None,
    block_bytes: int = CSV_BLOCK_BYTES,
    limit: int | None = None,
) -> tuple[int, str | None]:
    """Walk the live file; compare its already-archived rows with the archive.

    ``mode`` is ``keys`` (only the store's key columns, a narrow read),
    ``proof`` (every column, allowing the writer's rewrite) or ``exact``.
    ``stop_row`` limits the comparison to the first ``stop_row`` live rows;
    ``limit`` is the byte snapshot to read. Returns (live row count, first
    problem or None). Rows the archive does not cover are only counted.
    """
    if mode == "keys":
        columns = [c for c in spec.key_columns if c in header]
        include = set(columns) | ({spec.date_column} if spec.date_column in header else set())
    else:
        columns = list(header)
        include = None
    with _closing_all({}) as cursors:
        return _walk_live(
            spec, header, manifest, cursors, columns=columns, include=include,
            live_offset=live_offset, mode=mode, stop_row=stop_row, block_bytes=block_bytes, limit=limit,
        )


def _walk_live(spec, header, manifest, cursors, *, columns, include, live_offset, mode,
               stop_row, block_bytes, limit) -> tuple[int, str | None]:
    archive_dir = spec.archive_dir
    archived = int(manifest.get("archived_rows", 0))
    files = manifest.get("files", {})
    upto = archived if stop_row is None else min(archived, live_offset + stop_row)
    count = 0
    for batch in _iter_csv(spec.csv_path, header, include=include, block_bytes=block_bytes, limit=limit):
        start = live_offset + count
        count += batch.num_rows
        lo, hi = start, min(start + batch.num_rows, upto)
        if hi <= lo:
            continue
        table = pa.Table.from_batches([batch]).slice(0, hi - lo)
        seqs = list(range(lo, hi))
        keys = _month_keys(table, spec.date_column)
        seq_by_month: dict[str, list[int]] = {}
        for seq, key in zip(seqs, keys, strict=True):
            seq_by_month.setdefault(key, []).append(seq)
        for month, rows in _split_by_month(table, keys).items():
            entry = files.get(month)
            if entry is None:
                return count, f"live rows dated {month} are inside the archived range but no {month} file exists"
            cursor = cursors.get(month)
            if cursor is None:
                month_seqs = seq_by_month[month]
                cursor = _Cursor(_file_batches(archive_dir / entry["file"], columns, seq_from=month_seqs[0]))
                cursors[month] = cursor
            # Keys use their own narrow equivalence (`_keys_match`): the writer's
            # widening rewrite turns a ticker `NA` into "", and that must not
            # make every night refuse. Any other difference still refuses.
            problem = _compare(rows, cursor.take(rows.num_rows), seq_by_month[month], columns, mode)
            if problem:
                return count, f"{month}: {problem}"
    if live_offset + count < archived and stop_row is None:
        return count, (
            f"the live file ends at history position {live_offset + count} but the archive "
            f"covers {archived}: rows were lost or the file was replaced"
        )
    return count, None


def _first_keys(spec: StoreSpec, header: list[str], n: int) -> list[tuple]:
    columns = [c for c in spec.key_columns if c in header]
    rows: list[tuple] = []
    for batch in _iter_csv(spec.csv_path, header, include=columns, block_bytes=1 << 20):
        table = batch.to_pydict()
        for i in range(batch.num_rows):
            rows.append(tuple(table[c][i] for c in columns))
            if len(rows) >= n:
                return rows
    return rows


def _archived_keys(manifest: dict[str, Any], spec: StoreSpec, header: list[str], lo: int, n: int) -> list[tuple]:
    archive_dir = spec.archive_dir
    columns = [c for c in spec.key_columns if c in header]
    hi = min(lo + n, int(manifest.get("archived_rows", 0)))
    if hi <= lo:
        return []
    tables = []
    for entry in manifest.get("files", {}).values():
        if entry.get("seq_max", -1) < lo or entry.get("seq_min", 0) >= hi:
            continue
        table = pq.read_table(str(archive_dir / entry["file"]), columns=[SEQ_COLUMN, *columns])
        seq = table.column(SEQ_COLUMN)
        table = table.filter(pc.and_(pc.greater_equal(seq, lo), pc.less(seq, hi)))
        tables.append(table)
    if not tables:
        return []
    table = pa.concat_tables(tables).sort_by(SEQ_COLUMN).to_pydict()
    return [tuple(table[c][i] for c in columns) for i in range(len(table[SEQ_COLUMN]))]


def _effective_live_offset(manifest: dict[str, Any], spec: StoreSpec) -> int:
    """The live offset, settling a trim that may or may not have reached the file.

    A trim writes ``pending_live_offset`` first, replaces the CSV, then commits.
    If it died between, the live file's first rows say which offset is true.
    """
    committed = int(manifest.get("live_offset", 0))
    pending = manifest.get("pending_live_offset")
    if pending is None:
        return committed
    pending = int(pending)
    try:
        header = read_header(spec.csv_path)
    except ArchiveRefused:
        return committed
    probe = 64
    live = _first_keys(spec, header, probe)
    if not live:
        # Empty live file: the trim removed everything it proved.
        return pending

    def matches(offset: int) -> bool:
        archived = _archived_keys(manifest, spec, header, offset, len(live))
        return bool(archived) and live[: len(archived)] == archived

    # The pre-trim file starts with archived rows at the committed offset; if
    # the live file does not, the replace happened (its head may even be rows
    # the archive has not reached yet, when the trim removed every archived row).
    if matches(committed) and not matches(pending):
        return committed
    if matches(committed):
        _log.warning("d1 history archive: interrupted trim is ambiguous; keeping the committed offset")
        return committed
    return pending


def _settle_pending(manifest: dict[str, Any], spec: StoreSpec) -> dict[str, Any]:
    """Under the locks: write down which offset a dead trim left behind."""
    if manifest.get("pending_live_offset") is None:
        return manifest
    settled = dict(manifest)
    settled["live_offset"] = _effective_live_offset(manifest, spec)
    settled["pending_live_offset"] = None
    _write_manifest(spec.archive_dir / MANIFEST_NAME, settled)
    _log.warning("d1 history archive: settled an interrupted trim at live offset %s", settled["live_offset"])
    return settled


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


def archive(
    csv_path: Any = None,
    archive_dir: Any = None,
    *,
    store: Any = None,
    block_bytes: int = CSV_BLOCK_BYTES,
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pack the live rows not yet archived; verify them; commit the manifest.

    ``store``: a registered store name or a ``StoreSpec`` (default the D1
    history); ``csv_path`` / ``archive_dir`` override its paths. Idempotent:
    with nothing new it writes nothing. Raises ArchiveRefused or VerifyFailed
    (both ArchiveError) with nothing changed.
    """
    spec = _spec(store, csv_path, archive_dir)
    started = time.monotonic()
    try:
        with _locks(spec, lock_timeout):
            before = _file_state(spec.csv_path)
            try:
                result = _archive_locked(spec, block_bytes=block_bytes, now=now)
            except (pa.ArrowInvalid, ArchiveError) as exc:
                _refuse_if_changed_under_read(spec, before, exc)
                raise
    finally:
        _release_memory()
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


class _ChangedUnderRead(RuntimeError):
    """The live file of a no-lock store changed while it was being read."""


def _file_state(csv_path: Path) -> tuple[int, tuple[str, ...] | None]:
    """(size, header) now; header None when it cannot be read."""
    try:
        size = csv_path.stat().st_size
    except OSError:
        return -1, None
    try:
        return size, tuple(read_header(csv_path))
    except ArchiveRefused:
        return size, None


def _changed_since(csv_path: Path, before: tuple[int, tuple[str, ...] | None]) -> bool:
    """Did the file shrink or its header change? An appender only ever grows it."""
    size, header = _file_state(csv_path)
    return size < before[0] or header != before[1]


def _refuse_if_changed_under_read(spec: StoreSpec, before, exc: BaseException) -> None:
    """For a store whose writer takes no lock, turn a read that ran into a
    concurrent rewrite (the bounce writer's in-place ``open("w")`` header
    widening) into a refusal: nothing was written, the next run retries.

    A parse error on a no-lock store is taken as that rewrite even when the
    file looks settled again; a locked store's errors are never touched here -
    a corrupt D1 file still fails loudly.
    """
    if spec.writer_lock_key is not None or isinstance(exc, ArchiveRefused):
        return
    if isinstance(exc, pa.ArrowInvalid) or _changed_since(spec.csv_path, before):
        raise ArchiveRefused(
            f"{spec.name}: the live file changed during the read (its writer takes no lock and "
            f"rewrites the file in place when its header widens); nothing written, retry next run "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def _release_memory() -> None:
    """Hand arrow's pooled memory back: the night loads a 20 GB model after this."""
    try:
        pa.default_memory_pool().release_unused()
    except Exception as exc:  # noqa: BLE001 - best effort only
        _log.debug("d1 history archive: release_unused failed (%s)", exc)


def _archive_locked(spec: StoreSpec, *, block_bytes: int, now: datetime | None) -> dict[str, Any]:
    csv_path, archive_dir = spec.csv_path, spec.archive_dir
    _sweep_stale_temps(csv_path, archive_dir)
    manifest = _settle_pending(load_manifest(archive_dir), spec)
    problems = _verify_files(manifest, archive_dir)
    if problems:
        raise VerifyFailed("archive does not verify; nothing packed: " + "; ".join(problems))
    archived = int(manifest["archived_rows"])
    base = {"ok": True, "archived_rows": 0, "total_archived": archived, "months": {}}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {**base, "reason": "no live history file"}
    # One snapshot for every pass of this run (complete records only when the
    # writer takes no lock), so a concurrent append is simply left for later.
    limit = _snapshot_end(csv_path, writer_locked=spec.writer_lock_key is not None)
    header = read_header(csv_path)
    known = list(manifest.get("columns") or [])
    if header[: len(known)] != known:
        raise ArchiveRefused(
            "the live header no longer starts with the archived columns; the writer only "
            "ever appends columns, so the file was changed by something else"
        )
    live_offset = int(manifest["live_offset"])
    live_rows, problem = _check_live_against_archive(
        spec, header, manifest, live_offset=live_offset, mode="keys", block_bytes=block_bytes, limit=limit
    )
    if problem:
        raise VerifyFailed(f"live file does not match the archive; nothing packed: {problem}")
    new_rows = live_offset + live_rows - archived
    if new_rows <= 0:
        return {**base, "live_rows": live_rows, "reason": "nothing new to pack"}

    generation = int(manifest["generation"]) + 1
    files = manifest.get("files", {})
    writers: dict[str, pq.ParquetWriter] = {}
    temps: dict[str, Path] = {}
    old_rows: dict[str, int] = {}
    added: dict[str, int] = {}
    finals: list[Path] = []
    schema = _archive_schema(header)
    stream = _iter_csv(csv_path, header, block_bytes=block_bytes, limit=limit)
    try:
        # Pass 1: write. Old rows of a touched month are copied first.
        count = 0
        for batch in stream:
            start = live_offset + count
            count += batch.num_rows
            skip = max(0, archived - start)
            if skip >= batch.num_rows:
                continue
            table = pa.Table.from_batches([batch]).slice(skip)
            first = start + skip
            keys = _month_keys(table, spec.date_column)
            seq_by_month: dict[str, list[int]] = {}
            for offset, key in enumerate(keys):
                seq_by_month.setdefault(key, []).append(first + offset)
            for month, rows in _split_by_month(table, keys).items():
                writer = writers.get(month)
                if writer is None:
                    temp = archive_dir / f"{month}.g{generation:06d}.parquet.tmp-{os.getpid()}"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(str(temp), schema, compression=PARQUET_COMPRESSION)
                    writers[month], temps[month] = writer, temp
                    old_rows[month] = 0
                    if month in files:
                        for old in _file_batches(archive_dir / files[month]["file"], header):
                            writer.write_table(pa.Table.from_batches([old], schema=schema))
                            old_rows[month] += old.num_rows
                writer.write_table(_archive_table(rows, seq_by_month[month], header))
                added[month] = added.get(month, 0) + rows.num_rows
        for writer in writers.values():
            writer.close()
        writers.clear()
        if sum(added.values()) != new_rows:
            raise VerifyFailed(f"packed {sum(added.values())} rows, expected {new_rows}")

        # Pass 2: read back and verify, cell for cell.
        for month, temp in temps.items():
            if month in files:
                old = _Cursor(_file_batches(archive_dir / files[month]["file"], header))
                fresh = _Cursor(_file_batches(temp, header))
                with _closing_all([old, fresh]):
                    while True:
                        left = old.take(8192)
                        if left is None:
                            break
                        right = fresh.take(left.num_rows)
                        if right is None or not right.equals(left):
                            raise VerifyFailed(
                                f"{month}: the rewritten file does not carry the old rows unchanged"
                            )
        verify_manifest = {
            "archived_rows": archived + new_rows,
            "files": {month: {"file": temp.name} for month, temp in temps.items()},
        }
        _, problem = _check_live_against_archive_from(
            spec, header, verify_manifest,
            live_offset=live_offset, seq_from=archived, block_bytes=block_bytes, limit=limit,
        )
        if problem:
            raise VerifyFailed(f"packed rows do not read back as written: {problem}")

        # Commit: rename, hash, manifest last.
        new_files = dict(files)
        for month, temp in temps.items():
            final = archive_dir / f"{month}.g{generation:06d}.parquet"
            os.replace(temp, final)
            finals.append(final)
            seqs = pq.read_table(str(final), columns=[SEQ_COLUMN]).column(SEQ_COLUMN)
            new_files[month] = {
                "file": final.name,
                "rows": old_rows[month] + added[month],
                "sha256": _sha256(final),
                "bytes": final.stat().st_size,
                "seq_min": pc.min(seqs).as_py(),
                "seq_max": pc.max(seqs).as_py(),
                "columns": len(header),
            }
        new_manifest = {
            **manifest,
            "generation": generation,
            "archived_rows": archived + new_rows,
            "columns": list(header),
            "files": new_files,
            "updated_at": _now_text(now),
        }
        problems = _verify_files(new_manifest, archive_dir)
        if problems:
            raise VerifyFailed("new archive files do not verify: " + "; ".join(problems))
        _write_manifest(archive_dir / MANIFEST_NAME, new_manifest)
    except BaseException:
        stream.close()
        for writer in writers.values():
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - cleanup after a failure
                pass
        for path in [*temps.values(), *finals]:
            _unlink_quietly(path)
        _log.error("d1 history archive: pack failed; the last good archive is kept", exc_info=True)
        raise
    _collect_garbage(archive_dir, new_manifest)
    return {
        "ok": True,
        "archived_rows": new_rows,
        "total_archived": archived + new_rows,
        "live_rows": live_rows,
        "months": dict(sorted(added.items())),
        "generation": generation,
    }


def _check_live_against_archive_from(
    spec: StoreSpec,
    header: list[str],
    manifest: dict[str, Any],
    *,
    live_offset: int,
    seq_from: int,
    block_bytes: int,
    limit: int | None,
) -> tuple[int, str | None]:
    """Exact comparison of live rows at history positions >= ``seq_from``."""
    with _closing_all({}) as cursors:
        return _walk_from(spec, header, manifest, cursors, live_offset=live_offset,
                          seq_from=seq_from, block_bytes=block_bytes, limit=limit)


def _walk_from(spec, header, manifest, cursors, *, live_offset, seq_from, block_bytes, limit):
    archive_dir = spec.archive_dir
    archived = int(manifest["archived_rows"])
    files = manifest["files"]
    count = 0
    for batch in _iter_csv(spec.csv_path, header, block_bytes=block_bytes, limit=limit):
        start = live_offset + count
        count += batch.num_rows
        lo, hi = max(start, seq_from), min(start + batch.num_rows, archived)
        if hi <= lo:
            continue
        table = pa.Table.from_batches([batch]).slice(lo - start, hi - lo)
        keys = _month_keys(table, spec.date_column)
        seq_by_month: dict[str, list[int]] = {}
        for offset, key in enumerate(keys):
            seq_by_month.setdefault(key, []).append(lo + offset)
        for month, rows in _split_by_month(table, keys).items():
            cursor = cursors.get(month)
            if cursor is None:
                cursor = _Cursor(_file_batches(archive_dir / files[month]["file"], header, seq_from=seq_from))
                cursors[month] = cursor
            problem = _compare(rows, cursor.take(rows.num_rows), seq_by_month[month], header, "exact")
            if problem:
                return count, f"{month}: {problem}"
    for month, cursor in cursors.items():
        if not cursor.exhausted():
            return count, f"{month}: the archive file holds more rows than were packed"
    if live_offset + count < archived:
        return count, "the live file is shorter than the packed range"
    return count, None


def _sweep_stale_temps(csv_path: Path, archive_dir: Path) -> list[str]:
    """Remove temp files a hard-killed archive or trim left behind.

    Called only while holding the archive's lock (and the writer's, where the
    store has one), which every archive and trim holds for as long as its temp
    files exist - so any file matching OUR naming pattern now is an orphan.
    Only these exact patterns are touched: ``<csv>.trim-tmp-<pid>``, and in the
    archive folder ``*.parquet.tmp-<pid>`` and ``manifest.json.tmp-<pid>``.
    """
    removed: list[str] = []
    trim_pattern = re.compile(re.escape(csv_path.name) + r"\.trim-tmp-\d+$")
    archive_pattern = re.compile(r"(\.parquet|" + re.escape(MANIFEST_NAME) + r")\.tmp-\d+$")
    candidates = []
    if csv_path.parent.is_dir():
        candidates += [p for p in csv_path.parent.iterdir() if trim_pattern.fullmatch(p.name)]
    if archive_dir.is_dir():
        candidates += [p for p in archive_dir.iterdir() if archive_pattern.search(p.name)]
    for path in candidates:
        if path.is_file():
            _unlink_quietly(path)
            if not path.exists():
                removed.append(path.name)
                _log.warning("d1 history archive: removed stale temp %s from an interrupted run", path)
    return removed


def _collect_garbage(archive_dir: Path, manifest: dict[str, Any]) -> None:
    """Remove archive files the committed manifest no longer names (superseded)."""
    keep = {entry["file"] for entry in manifest.get("files", {}).values()} | {MANIFEST_NAME}
    for path in archive_dir.iterdir():
        if path.is_file() and path.name not in keep and ".parquet" in path.name:
            _unlink_quietly(path)


# ---------------------------------------------------------------------------
# verify / status
# ---------------------------------------------------------------------------


def verify(
    csv_path: Any = None,
    archive_dir: Any = None,
    *,
    store: Any = None,
    deep: bool = False,
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
    block_bytes: int = CSV_BLOCK_BYTES,
) -> dict[str, Any]:
    """Check the archive files and the live file against them. Never raises.

    Always: every file's sha256, row count and positions. Then, holding the
    writer's lock (where the store has one) so a scan cannot replace the file
    mid-read, the key of every live row the archive covers (``deep``: every
    cell, allowing the writer's widening rewrite), read up to a snapshot of
    complete records. Read-only.
    """
    report: dict[str, Any] = {"ok": False, "problems": [], "deep": bool(deep)}
    try:
        spec = _spec(store, csv_path, archive_dir)
        manifest = load_manifest(spec.archive_dir)
    except ArchiveError as exc:
        report["problems"].append(str(exc))
        return report
    csv_path, archive_dir = spec.csv_path, spec.archive_dir
    report["store"] = spec.name
    report["archived_rows"] = int(manifest["archived_rows"])
    report["files"] = len(manifest.get("files", {}))
    report["problems"].extend(_verify_files(manifest, archive_dir))
    if not report["problems"] and csv_path.exists() and csv_path.stat().st_size:
        try:
            with _locks(spec, lock_timeout):
                before = _file_state(csv_path)
                limit = _snapshot_end(csv_path, writer_locked=spec.writer_lock_key is not None)
                try:
                    header = read_header(csv_path)
                    offset = _effective_live_offset(manifest, spec)
                    count, problem = _check_live_against_archive(
                        spec, header, manifest, live_offset=offset,
                        mode="proof" if deep else "keys", block_bytes=block_bytes, limit=limit,
                    )
                    if problem:
                        _refuse_if_changed_under_read(spec, before, VerifyFailed(problem))
                except (pa.ArrowInvalid, ArchiveError) as exc:
                    _refuse_if_changed_under_read(spec, before, exc)
                    raise
            report["live_rows"] = count
            report["unarchived_rows"] = max(0, offset + count - int(manifest["archived_rows"]))
            if problem:
                report["problems"].append(problem)
        except ArchiveError as exc:
            report["problems"].append(str(exc))
    report["ok"] = not report["problems"]
    return report


def status(csv_path: Any = None, archive_dir: Any = None, *, store: Any = None) -> dict[str, Any]:
    """What the store holds, by manifest and ``stat`` only (cheap)."""
    spec = _spec(store, csv_path, archive_dir)
    csv_path, archive_dir = spec.csv_path, spec.archive_dir
    out: dict[str, Any] = {
        "store": spec.name,
        "csv": str(csv_path),
        "archive_dir": str(archive_dir),
        "writer_lock": "none" if spec.writer_lock_key is None else "held by the writer",
    }
    out["csv_bytes"] = csv_path.stat().st_size if csv_path.exists() else 0
    try:
        manifest = load_manifest(archive_dir)
    except ArchiveError as exc:
        out["manifest_problem"] = str(exc)
        return out
    out.update(
        archived_rows=manifest["archived_rows"],
        live_offset=manifest["live_offset"],
        pending_live_offset=manifest["pending_live_offset"],
        generation=manifest["generation"],
        updated_at=manifest.get("updated_at", ""),
        months={month: entry.get("rows") for month, entry in sorted(manifest["files"].items())},
        archive_bytes=sum(
            (archive_dir / e["file"]).stat().st_size
            for e in manifest["files"].values()
            if (archive_dir / e["file"]).exists()
        ),
    )
    return out


# ---------------------------------------------------------------------------
# trim
# ---------------------------------------------------------------------------


def trim(
    csv_path: Any = None,
    archive_dir: Any = None,
    *,
    store: Any = None,
    keep_days: int | None = None,
    today: date | None = None,
    apply: bool = False,
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
    block_bytes: int = CSV_BLOCK_BYTES,
) -> dict[str, Any]:
    """Drop the archived, proven head of the live CSV older than ``keep_days``.

    ``keep_days`` defaults to the store's. A dry run unless ``apply``. Refuses
    (ArchiveRefused / VerifyFailed, nothing changed) for a store whose writer
    takes no lock (always, whatever any setting says), without the writer's
    lock, with an unreadable header, or when the archive or the rows to remove
    do not verify.
    """
    spec = _spec(store, csv_path, archive_dir)
    if spec.writer_lock_key is None:
        raise ArchiveRefused(
            f"{spec.name}: its writer takes no lock, so the live file cannot be rewritten "
            "without racing an append that would be lost; trim is refused. Turning trim on "
            "needs the writer to take local_writer_lock first."
        )
    keep_days = spec.keep_days if keep_days is None else int(keep_days)
    today = today or date.today()
    read_header(spec.csv_path)  # refuse before taking any lock when the file is unreadable
    try:
        with _locks(spec, lock_timeout):
            return _trim_locked(spec, keep_days=keep_days, today=today, apply=apply, block_bytes=block_bytes)
    finally:
        _release_memory()


def _trim_locked(spec: StoreSpec, *, keep_days: int, today: date, apply: bool, block_bytes: int) -> dict[str, Any]:
    csv_path, archive_dir = spec.csv_path, spec.archive_dir
    date_column = spec.date_column
    _sweep_stale_temps(csv_path, archive_dir)
    manifest = _settle_pending(load_manifest(archive_dir), spec)
    problems = _verify_files(manifest, archive_dir)
    if problems:
        raise VerifyFailed("archive does not verify; nothing trimmed: " + "; ".join(problems))
    header = read_header(csv_path)
    archived = int(manifest["archived_rows"])
    live_offset = int(manifest["live_offset"])
    _, problem = _check_live_against_archive(
        spec, header, manifest, live_offset=live_offset, mode="keys", block_bytes=block_bytes
    )
    if problem:
        raise VerifyFailed(f"live file does not match the archive; nothing trimmed: {problem}")

    # The removable head: archived, dated, and older than keep_days.
    remove = 0
    stopped_by = "end_of_file"
    oldest_kept = ""
    date_cols = [date_column] if date_column in header else []
    live_count = 0
    done = False
    for batch in _iter_csv(csv_path, header, include=date_cols or header[:1], block_bytes=block_bytes):
        texts = batch.column(date_column).to_pylist() if date_cols else [""] * batch.num_rows
        for text in texts:
            if not done:
                parsed = _parse_run_date(text)
                if live_offset + live_count >= archived:
                    stopped_by, done = "unarchived", True
                elif parsed is None:
                    stopped_by, done = "undated", True
                elif (today - parsed).days <= keep_days:
                    stopped_by, done = "recent", True
                else:
                    remove += 1
                if done:
                    oldest_kept = text
            live_count += 1
    result: dict[str, Any] = {
        "applied": False,
        "removed": 0,
        "would_remove": remove,
        "keep": live_count - remove,
        "live_rows": live_count,
        "keep_days": keep_days,
        "today": today.isoformat(),
        "cutoff_removes_before": (today - timedelta(days=keep_days)).isoformat(),
        "stopped_by": stopped_by,
        "first_kept_run_date": oldest_kept,
    }
    if remove == 0:
        return result
    _, problem = _check_live_against_archive(
        spec, header, manifest, live_offset=live_offset, mode="proof",
        stop_row=remove, block_bytes=block_bytes,
    )
    if problem:
        raise VerifyFailed(f"rows to remove are not proven in the archive; nothing trimmed: {problem}")
    if not apply:
        return result

    temp = csv_path.with_name(csv_path.name + f".trim-tmp-{os.getpid()}")
    manifest_path = archive_dir / MANIFEST_NAME
    replaced = False
    try:
        header_bytes, offset = _record_offset(csv_path, remove)
        with open(csv_path, "rb") as source, open(temp, "wb") as target:
            target.write(header_bytes)
            source.seek(offset)
            shutil.copyfileobj(source, target, 1 << 20)
            target.flush()
            os.fsync(target.fileno())
        problem = _same_rows(temp, csv_path, header, skip=remove, block_bytes=block_bytes)
        if problem:
            raise VerifyFailed(f"trimmed copy does not equal the kept rows; nothing trimmed: {problem}")
        pending = {**manifest, "pending_live_offset": live_offset + remove}
        _write_manifest(manifest_path, pending)
        try:
            os.replace(temp, csv_path)
            replaced = True
        except OSError as exc:
            _write_manifest(manifest_path, manifest)
            raise ArchiveError(f"could not replace the live file ({exc}); nothing trimmed") from exc
        committed = {**manifest, "live_offset": live_offset + remove, "pending_live_offset": None}
        _write_manifest(manifest_path, committed)
    finally:
        _unlink_quietly(temp)
        if not replaced:
            _log.error("d1 history trim: not applied; the live file is unchanged")
    result.update(applied=True, removed=remove, would_remove=remove)
    return result


def _read_record(handle) -> bytes | None:
    """One CSV record's raw bytes (quote-aware, RFC 4180 doubling); blank lines skipped."""
    parts: list[bytes] = []
    quotes = 0
    while True:
        line = handle.readline()
        if not line:
            if parts:
                raise ArchiveError("history file ends inside a quoted field")
            return None
        parts.append(line)
        quotes += line.count(b'"')
        if quotes % 2:
            continue
        if len(parts) == 1 and not line.strip(b"\r\n"):
            parts = []
            continue
        return b"".join(parts)


def _record_offset(csv_path: Path, records: int) -> tuple[bytes, int]:
    """The header's bytes and the byte offset where data record ``records`` starts."""
    with open(csv_path, "rb") as handle:
        header = _read_record(handle)
        if header is None:
            raise ArchiveRefused("history header unreadable")
        for _ in range(records):
            if _read_record(handle) is None:
                raise ArchiveError("history file has fewer records than expected")
        return header, handle.tell()


def _same_rows(trimmed: Path, original: Path, header: list[str], *, skip: int, block_bytes: int) -> str | None:
    if read_header(trimmed) != header:
        return "header differs"
    mine = _Cursor(_iter_csv(trimmed, header, block_bytes=block_bytes))
    with _closing_all([mine]):
        return _walk_same(mine, original, header, skip=skip, block_bytes=block_bytes)


def _walk_same(mine: _Cursor, original: Path, header: list[str], *, skip: int, block_bytes: int) -> str | None:
    seen = 0
    for batch in _iter_csv(original, header, block_bytes=block_bytes):
        start = seen
        seen += batch.num_rows
        lo = max(0, skip - start)
        if lo >= batch.num_rows:
            continue
        table = pa.Table.from_batches([batch]).slice(lo)
        other = mine.take(table.num_rows)
        if other is None or not other.equals(table):
            return f"rows differ after original row {start + lo}"
    if not mine.exhausted():
        return "trimmed copy has extra rows"
    return None


# ---------------------------------------------------------------------------
# read_history
# ---------------------------------------------------------------------------


def _date_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    parsed = _parse_run_date(str(value))
    if parsed is None:
        raise ValueError(f"not a date: {value!r}")
    return parsed.isoformat()


def read_history(
    columns: Iterable[str] | None = None,
    since: Any = None,
    until: Any = None,
    *,
    csv_path: Any = None,
    archive_dir: Any = None,
    store: Any = None,
    typed: bool = False,
    block_bytes: int = CSV_BLOCK_BYTES,
):
    """Archive + live as one DataFrame, in original order, with the original text.

    ``columns``: like ``pd.read_csv(usecols=...)`` - returned in file order;
    an unknown name raises ValueError. ``since`` / ``until``: inclusive
    ``run_date`` bounds; when either is given, rows with a blank or unparseable
    ``run_date`` are left out (they have no date to compare). Both push down:
    only the month files in range are opened, only the named columns are read.

    ``typed=False`` (default): every cell is the CSV's text (object dtype, ``""``
    for empty), equal to ``pd.read_csv(path, dtype=str, keep_default_na=False)``.
    ``typed=True``: the same rows re-parsed by ``pd.read_csv(low_memory=False)``,
    so the dtypes are exactly what that call infers from these rows' text. That is
    parity with reading the original CSV whenever the row set is the same; a date
    bound reads fewer rows, and pandas may infer a different dtype from fewer
    rows (an int column whose blanks all fall outside the bounds stays int), as it
    would on a CSV holding only those rows.

    ``store``: a registered store name or spec (default the D1 history); the
    date bounds apply to that store's date column. For a store whose writer
    takes no lock, the live part is read up to the last complete record.
    """
    import pandas as pd

    spec = _spec(store, csv_path, archive_dir)
    since_text, until_text = _date_text(since), _date_text(until)
    bounded = since_text is not None or until_text is not None
    last_error: Exception | None = None
    for _attempt in range(_READ_RETRIES):
        try:
            before = load_manifest(spec.archive_dir)
            frame = _read_once(before, spec, columns, since_text, until_text, bounded, block_bytes)
            after = load_manifest(spec.archive_dir)
        except (FileNotFoundError, pa.ArrowInvalid, OSError, _ChangedUnderRead) as exc:
            last_error = exc
            time.sleep(0.05 * (_attempt + 1))
            continue
        if _manifest_signature(before) == _manifest_signature(after):
            break
    else:
        raise ArchiveError(f"history kept changing under the read ({last_error})")
    if typed:
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        buffer.seek(0)
        return pd.read_csv(buffer, low_memory=False)
    return frame


def _read_once(manifest, spec, columns, since_text, until_text, bounded, block_bytes):
    import pandas as pd

    csv_path, archive_dir, date_column = spec.csv_path, spec.archive_dir, spec.date_column
    state_before = _file_state(csv_path)
    header = read_header(csv_path) if csv_path.exists() and csv_path.stat().st_size else []
    limit = (
        _snapshot_end(csv_path, writer_locked=False) if header and spec.writer_lock_key is None else None
    )
    known = list(manifest.get("columns") or [])
    all_columns = known + [c for c in header if c not in known]
    if columns is None:
        wanted = all_columns
    else:
        requested = list(dict.fromkeys(columns))
        missing = [c for c in requested if c not in all_columns]
        if missing:
            raise ValueError(f"columns not in the history: {missing}")
        wanted = [c for c in all_columns if c in set(requested)]
    need = list(wanted) + ([date_column] if bounded and date_column not in wanted else [])

    pieces: list[pa.Table] = []
    archived = int(manifest.get("archived_rows", 0))
    lo_month = since_text[:7] if since_text else None
    hi_month = until_text[:7] if until_text else None
    filters = []
    if since_text:
        filters.append((date_column, ">=", since_text))
    if until_text:
        filters.append((date_column, "<", (date.fromisoformat(until_text) + timedelta(days=1)).isoformat()))
    for month, entry in sorted(manifest.get("files", {}).items()):
        if bounded and (month == UNDATED or (lo_month and month < lo_month) or (hi_month and month > hi_month)):
            continue
        path = archive_dir / entry["file"]
        schema_names = pq.read_schema(str(path)).names
        present = [c for c in [SEQ_COLUMN, *need] if c in schema_names]
        table = pq.read_table(str(path), columns=present, filters=filters or None)
        pieces.append(_conform(table, need))
    archive_part = (
        pa.concat_tables(pieces).sort_by(SEQ_COLUMN) if pieces else _conform(_archive_schema([]).empty_table(), need)
    )

    live_tables: list[pa.Table] = []
    if header:
        offset = _effective_live_offset(manifest, spec)
        skip = max(0, archived - offset)
        seen = 0
        include = [c for c in need if c in header] or header[:1]
        for batch in _iter_csv(csv_path, header, include=include, block_bytes=block_bytes, limit=limit):
            start = seen
            seen += batch.num_rows
            lo = max(0, skip - start)
            if lo >= batch.num_rows:
                continue
            table = pa.Table.from_batches([batch]).slice(lo)
            if bounded:
                dates = [_date_text_or_none(t) for t in table.column(date_column).to_pylist()] if (
                    date_column in table.schema.names
                ) else [None] * table.num_rows
                mask = [
                    d is not None and (since_text is None or d >= since_text) and (until_text is None or d <= until_text)
                    for d in dates
                ]
                table = table.filter(pa.array(mask, pa.bool_()))
            live_tables.append(_conform(table, need, with_seq=False))
        if seen < skip:
            # The live file holds fewer rows than the archive says it does: a
            # rewrite in progress, a trim between our reads, or real loss.
            # Never return the short frame; retry, then raise.
            raise _ChangedUnderRead(f"live file has {seen} rows, the archive expects at least {skip}")
    if spec.writer_lock_key is None and _changed_since(csv_path, state_before):
        raise _ChangedUnderRead("the live file shrank or its header changed during the read")
    archive_part = archive_part.drop_columns([SEQ_COLUMN])
    combined = pa.concat_tables([archive_part, *live_tables]) if live_tables else archive_part
    combined = combined.select(wanted)
    frame = combined.to_pandas()
    frame = frame.astype(object).where(frame.notna(), "") if len(frame.columns) else frame
    frame = frame.reset_index(drop=True)
    if not len(frame.columns):
        frame = pd.DataFrame(index=pd.RangeIndex(combined.num_rows))
    return frame


def _date_text_or_none(text: str) -> str | None:
    parsed = _parse_run_date(text)
    return parsed.isoformat() if parsed else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="d1_feature_history_archive",
        description="Lossless monthly Parquet packing of append-only CSV stores (default: the D1 history).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "archive", "verify", "trim"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--store", default=D1_STORE,
                         help=f"registered store name (default: {D1_STORE})")
        cmd.add_argument("--csv", default=None, help="live CSV (default: the store's)")
        cmd.add_argument("--archive-dir", default=None, help="archive folder (default: the store's)")
        if name == "verify":
            cmd.add_argument("--deep", action="store_true", help="compare every cell of every live row")
        if name == "trim":
            cmd.add_argument("--keep-days", type=int, default=None, help="default: the store's (30)")
            cmd.add_argument("--today", default=None, help="YYYY-MM-DD (default: today)")
            cmd.add_argument("--apply", action="store_true", help="really rewrite the live file")
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            out = status(args.csv, args.archive_dir, store=args.store)
            code = 0
        elif args.command == "archive":
            out = archive(args.csv, args.archive_dir, store=args.store)
            code = 0
        elif args.command == "verify":
            out = verify(args.csv, args.archive_dir, store=args.store, deep=args.deep)
            code = 0 if out["ok"] else 1
        else:
            today = date.fromisoformat(args.today) if args.today else None
            out = trim(args.csv, args.archive_dir, store=args.store, keep_days=args.keep_days,
                       today=today, apply=args.apply)
            if not args.apply:
                print(f"DRY RUN: would remove {out['would_remove']} rows; pass --apply to trim.")
            code = 0
    except ArchiveError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
