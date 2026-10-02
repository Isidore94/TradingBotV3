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

CLI: ``python scripts/d1_feature_history_archive.py status|archive|verify|trim
[--apply]`` - trim is a dry run unless ``--apply``.
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
# Paths, manifest, locks
# ---------------------------------------------------------------------------


def _paths(csv_path: Any, archive_dir: Any) -> tuple[Path, Path]:
    if csv_path is None or archive_dir is None:
        import project_paths as pp

        csv_path = csv_path if csv_path is not None else pp.D1_FEATURES_HISTORY_FILE
        archive_dir = archive_dir if archive_dir is not None else pp.D1_FEATURES_HISTORY_ARCHIVE_DIR
    return Path(csv_path), Path(archive_dir)


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
def _locks(csv_path: Path, archive_dir: Path, timeout: float):
    """The writer's own lock first, then the archive's. Refuses, never waits forever."""
    with ExitStack() as stack:
        try:
            stack.enter_context(local_writer_lock(lock_key_for_path(csv_path), timeout_seconds=timeout))
            stack.enter_context(
                local_writer_lock(lock_key_for_path(archive_dir / MANIFEST_NAME), timeout_seconds=timeout)
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


def _iter_csv(
    csv_path: Path, header: list[str], include: Iterable[str] | None = None, block_bytes: int = CSV_BLOCK_BYTES
) -> Iterator[pa.RecordBatch]:
    """Stream the CSV as all-string batches. Exact text: no inference, no NA."""
    include_list = [c for c in header if c in set(include)] if include is not None else None
    reader = pacsv.open_csv(
        str(csv_path),
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


def _month_keys(batch: pa.RecordBatch | pa.Table) -> list[str]:
    if DATE_COLUMN not in batch.schema.names:
        return [UNDATED] * batch.num_rows
    return [_month_of(text) for text in batch.column(DATE_COLUMN).to_pylist()]


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


def _compare(live: pa.Table, arch: pa.Table, seqs: list[int], columns: list[str], mode: str) -> str | None:
    """None when ``arch`` holds ``live`` at ``seqs``; else the first difference.

    ``exact``: identical text, and columns the live row lacks are null.
    ``proof``: identical, or equal after the writer's widening rewrite.
    """
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
            if not ok and not _cells_prove(live_values[i], arch_values[i]):
                return (
                    f"column {name!r} at history position {seqs[i]}: live {live_values[i]!r} "
                    f"vs archived {arch_values[i]!r}"
                )
    return None


def _check_live_against_archive(
    csv_path: Path,
    header: list[str],
    manifest: dict[str, Any],
    archive_dir: Path,
    *,
    live_offset: int,
    mode: str,
    stop_row: int | None = None,
    block_bytes: int = CSV_BLOCK_BYTES,
) -> tuple[int, str | None]:
    """Walk the live file; compare its already-archived rows with the archive.

    ``mode`` is ``keys`` (only KEY_COLUMNS, a narrow read), ``proof`` (every
    column, allowing the writer's rewrite) or ``exact``. ``stop_row`` limits the
    comparison to the first ``stop_row`` live rows. Returns (live row count,
    first problem or None). Rows the archive does not cover are only counted.
    """
    if mode == "keys":
        columns = [c for c in KEY_COLUMNS if c in header]
        include = set(columns) | ({DATE_COLUMN} if DATE_COLUMN in header else set())
    else:
        columns = list(header)
        include = None
    with _closing_all({}) as cursors:
        return _walk_live(
            csv_path, header, manifest, archive_dir, cursors, columns=columns, include=include,
            live_offset=live_offset, mode=mode, stop_row=stop_row, block_bytes=block_bytes,
        )


def _walk_live(csv_path, header, manifest, archive_dir, cursors, *, columns, include, live_offset, mode,
               stop_row, block_bytes) -> tuple[int, str | None]:
    archived = int(manifest.get("archived_rows", 0))
    files = manifest.get("files", {})
    limit = archived if stop_row is None else min(archived, live_offset + stop_row)
    count = 0
    for batch in _iter_csv(csv_path, header, include=include, block_bytes=block_bytes):
        start = live_offset + count
        count += batch.num_rows
        lo, hi = start, min(start + batch.num_rows, limit)
        if hi <= lo:
            continue
        table = pa.Table.from_batches([batch]).slice(0, hi - lo)
        seqs = list(range(lo, hi))
        keys = _month_keys(table)
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
            problem = _compare(rows, cursor.take(rows.num_rows), seq_by_month[month], columns,
                               "exact" if mode == "keys" else mode)
            if problem:
                return count, f"{month}: {problem}"
    if live_offset + count < archived and stop_row is None:
        return count, (
            f"the live file ends at history position {live_offset + count} but the archive "
            f"covers {archived}: rows were lost or the file was replaced"
        )
    return count, None


def _first_keys(csv_path: Path, header: list[str], n: int) -> list[tuple]:
    columns = [c for c in KEY_COLUMNS if c in header]
    rows: list[tuple] = []
    for batch in _iter_csv(csv_path, header, include=columns, block_bytes=1 << 20):
        table = batch.to_pydict()
        for i in range(batch.num_rows):
            rows.append(tuple(table[c][i] for c in columns))
            if len(rows) >= n:
                return rows
    return rows


def _archived_keys(manifest: dict[str, Any], archive_dir: Path, header: list[str], lo: int, n: int) -> list[tuple]:
    columns = [c for c in KEY_COLUMNS if c in header]
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


def _effective_live_offset(manifest: dict[str, Any], archive_dir: Path, csv_path: Path) -> int:
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
        header = read_header(csv_path)
    except ArchiveRefused:
        return committed
    probe = 64
    live = _first_keys(csv_path, header, probe)
    if not live:
        # Empty live file: the trim removed everything it proved.
        return pending

    def matches(offset: int) -> bool:
        archived = _archived_keys(manifest, archive_dir, header, offset, len(live))
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


def _settle_pending(manifest: dict[str, Any], archive_dir: Path, csv_path: Path) -> dict[str, Any]:
    """Under the locks: write down which offset a dead trim left behind."""
    if manifest.get("pending_live_offset") is None:
        return manifest
    settled = dict(manifest)
    settled["live_offset"] = _effective_live_offset(manifest, archive_dir, csv_path)
    settled["pending_live_offset"] = None
    _write_manifest(archive_dir / MANIFEST_NAME, settled)
    _log.warning("d1 history archive: settled an interrupted trim at live offset %s", settled["live_offset"])
    return settled


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


def archive(
    csv_path: Any = None,
    archive_dir: Any = None,
    *,
    block_bytes: int = CSV_BLOCK_BYTES,
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pack the live rows not yet archived; verify them; commit the manifest.

    Idempotent: with nothing new it writes nothing. Raises ArchiveRefused or
    VerifyFailed (both ArchiveError) with nothing changed.
    """
    csv_path, archive_dir = _paths(csv_path, archive_dir)
    started = time.monotonic()
    try:
        with _locks(csv_path, archive_dir, lock_timeout):
            result = _archive_locked(csv_path, archive_dir, block_bytes=block_bytes, now=now)
    finally:
        _release_memory()
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def _release_memory() -> None:
    """Hand arrow's pooled memory back: the night loads a 20 GB model after this."""
    try:
        pa.default_memory_pool().release_unused()
    except Exception as exc:  # noqa: BLE001 - best effort only
        _log.debug("d1 history archive: release_unused failed (%s)", exc)


def _archive_locked(csv_path: Path, archive_dir: Path, *, block_bytes: int, now: datetime | None) -> dict[str, Any]:
    manifest = _settle_pending(load_manifest(archive_dir), archive_dir, csv_path)
    problems = _verify_files(manifest, archive_dir)
    if problems:
        raise VerifyFailed("archive does not verify; nothing packed: " + "; ".join(problems))
    archived = int(manifest["archived_rows"])
    base = {"ok": True, "archived_rows": 0, "total_archived": archived, "months": {}}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {**base, "reason": "no live history file"}
    header = read_header(csv_path)
    known = list(manifest.get("columns") or [])
    if header[: len(known)] != known:
        raise ArchiveRefused(
            "the live header no longer starts with the archived columns; the writer only "
            "ever appends columns, so the file was changed by something else"
        )
    live_offset = int(manifest["live_offset"])
    live_rows, problem = _check_live_against_archive(
        csv_path, header, manifest, archive_dir, live_offset=live_offset, mode="keys", block_bytes=block_bytes
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
    stream = _iter_csv(csv_path, header, block_bytes=block_bytes)
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
            keys = _month_keys(table)
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
            csv_path, header, verify_manifest, archive_dir,
            live_offset=live_offset, seq_from=archived, block_bytes=block_bytes,
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
    csv_path: Path,
    header: list[str],
    manifest: dict[str, Any],
    archive_dir: Path,
    *,
    live_offset: int,
    seq_from: int,
    block_bytes: int,
) -> tuple[int, str | None]:
    """Exact comparison of live rows at history positions >= ``seq_from``."""
    with _closing_all({}) as cursors:
        return _walk_from(csv_path, header, manifest, archive_dir, cursors, live_offset=live_offset,
                          seq_from=seq_from, block_bytes=block_bytes)


def _walk_from(csv_path, header, manifest, archive_dir, cursors, *, live_offset, seq_from, block_bytes):
    archived = int(manifest["archived_rows"])
    files = manifest["files"]
    count = 0
    for batch in _iter_csv(csv_path, header, block_bytes=block_bytes):
        start = live_offset + count
        count += batch.num_rows
        lo, hi = max(start, seq_from), min(start + batch.num_rows, archived)
        if hi <= lo:
            continue
        table = pa.Table.from_batches([batch]).slice(lo - start, hi - lo)
        keys = _month_keys(table)
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
    deep: bool = False,
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
    block_bytes: int = CSV_BLOCK_BYTES,
) -> dict[str, Any]:
    """Check the archive files and the live file against them. Never raises.

    Always: every file's sha256, row count and positions. Then, holding the
    writer's lock so a scan cannot replace the file mid-read, the key of every
    live row the archive covers (``deep``: every cell, allowing the writer's
    widening rewrite). Read-only.
    """
    csv_path, archive_dir = _paths(csv_path, archive_dir)
    report: dict[str, Any] = {"ok": False, "problems": [], "deep": bool(deep)}
    try:
        manifest = load_manifest(archive_dir)
    except ArchiveError as exc:
        report["problems"].append(str(exc))
        return report
    report["archived_rows"] = int(manifest["archived_rows"])
    report["files"] = len(manifest.get("files", {}))
    report["problems"].extend(_verify_files(manifest, archive_dir))
    if not report["problems"] and csv_path.exists() and csv_path.stat().st_size:
        try:
            with _locks(csv_path, archive_dir, lock_timeout):
                header = read_header(csv_path)
                offset = _effective_live_offset(manifest, archive_dir, csv_path)
                count, problem = _check_live_against_archive(
                    csv_path, header, manifest, archive_dir, live_offset=offset,
                    mode="proof" if deep else "keys", block_bytes=block_bytes,
                )
            report["live_rows"] = count
            report["unarchived_rows"] = max(0, offset + count - int(manifest["archived_rows"]))
            if problem:
                report["problems"].append(problem)
        except ArchiveError as exc:
            report["problems"].append(str(exc))
    report["ok"] = not report["problems"]
    return report


def status(csv_path: Any = None, archive_dir: Any = None) -> dict[str, Any]:
    """What the store holds, by manifest and ``stat`` only (cheap)."""
    csv_path, archive_dir = _paths(csv_path, archive_dir)
    out: dict[str, Any] = {"csv": str(csv_path), "archive_dir": str(archive_dir)}
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
    keep_days: int = DEFAULT_TRIM_KEEP_DAYS,
    today: date | None = None,
    apply: bool = False,
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
    block_bytes: int = CSV_BLOCK_BYTES,
) -> dict[str, Any]:
    """Drop the archived, proven head of the live CSV older than ``keep_days``.

    A dry run unless ``apply``. Refuses (ArchiveRefused / VerifyFailed, nothing
    changed) without the writer's lock, with an unreadable header, or when the
    archive or the rows to remove do not verify.
    """
    csv_path, archive_dir = _paths(csv_path, archive_dir)
    today = today or date.today()
    read_header(csv_path)  # refuse before taking any lock when the file is unreadable
    try:
        with _locks(csv_path, archive_dir, lock_timeout):
            return _trim_locked(csv_path, archive_dir, keep_days=int(keep_days), today=today,
                                apply=apply, block_bytes=block_bytes)
    finally:
        _release_memory()


def _trim_locked(
    csv_path: Path, archive_dir: Path, *, keep_days: int, today: date, apply: bool, block_bytes: int
) -> dict[str, Any]:
    manifest = _settle_pending(load_manifest(archive_dir), archive_dir, csv_path)
    problems = _verify_files(manifest, archive_dir)
    if problems:
        raise VerifyFailed("archive does not verify; nothing trimmed: " + "; ".join(problems))
    header = read_header(csv_path)
    archived = int(manifest["archived_rows"])
    live_offset = int(manifest["live_offset"])
    _, problem = _check_live_against_archive(
        csv_path, header, manifest, archive_dir, live_offset=live_offset, mode="keys", block_bytes=block_bytes
    )
    if problem:
        raise VerifyFailed(f"live file does not match the archive; nothing trimmed: {problem}")

    # The removable head: archived, dated, and older than keep_days.
    remove = 0
    stopped_by = "end_of_file"
    oldest_kept = ""
    date_cols = [DATE_COLUMN] if DATE_COLUMN in header else []
    live_count = 0
    done = False
    for batch in _iter_csv(csv_path, header, include=date_cols or header[:1], block_bytes=block_bytes):
        texts = batch.column(DATE_COLUMN).to_pylist() if date_cols else [""] * batch.num_rows
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
        csv_path, header, manifest, archive_dir, live_offset=live_offset, mode="proof",
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
    """
    import pandas as pd

    csv_path, archive_dir = _paths(csv_path, archive_dir)
    since_text, until_text = _date_text(since), _date_text(until)
    bounded = since_text is not None or until_text is not None
    last_error: Exception | None = None
    for _attempt in range(_READ_RETRIES):
        try:
            before = load_manifest(archive_dir)
            frame = _read_once(before, csv_path, archive_dir, columns, since_text, until_text, bounded, block_bytes)
            after = load_manifest(archive_dir)
        except (FileNotFoundError, pa.ArrowInvalid, OSError) as exc:
            last_error = exc
            time.sleep(0.05)
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


def _read_once(manifest, csv_path, archive_dir, columns, since_text, until_text, bounded, block_bytes):
    import pandas as pd

    header = read_header(csv_path) if csv_path.exists() and csv_path.stat().st_size else []
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
    need = list(wanted) + ([DATE_COLUMN] if bounded and DATE_COLUMN not in wanted else [])

    pieces: list[pa.Table] = []
    archived = int(manifest.get("archived_rows", 0))
    lo_month = since_text[:7] if since_text else None
    hi_month = until_text[:7] if until_text else None
    filters = []
    if since_text:
        filters.append((DATE_COLUMN, ">=", since_text))
    if until_text:
        filters.append((DATE_COLUMN, "<", (date.fromisoformat(until_text) + timedelta(days=1)).isoformat()))
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
        offset = _effective_live_offset(manifest, archive_dir, csv_path)
        skip = max(0, archived - offset)
        seen = 0
        include = [c for c in need if c in header] or header[:1]
        for batch in _iter_csv(csv_path, header, include=include, block_bytes=block_bytes):
            start = seen
            seen += batch.num_rows
            lo = max(0, skip - start)
            if lo >= batch.num_rows:
                continue
            table = pa.Table.from_batches([batch]).slice(lo)
            if bounded:
                dates = [_date_text_or_none(t) for t in table.column(DATE_COLUMN).to_pylist()] if (
                    DATE_COLUMN in table.schema.names
                ) else [None] * table.num_rows
                mask = [
                    d is not None and (since_text is None or d >= since_text) and (until_text is None or d <= until_text)
                    for d in dates
                ]
                table = table.filter(pa.array(mask, pa.bool_()))
            live_tables.append(_conform(table, need, with_seq=False))
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
        description="Lossless monthly Parquet packing of d1_features_history.csv.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "archive", "verify", "trim"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--csv", default=None, help="history CSV (default: the live one)")
        cmd.add_argument("--archive-dir", default=None, help="archive folder (default: beside the CSV)")
        if name == "verify":
            cmd.add_argument("--deep", action="store_true", help="compare every cell of every live row")
        if name == "trim":
            cmd.add_argument("--keep-days", type=int, default=DEFAULT_TRIM_KEEP_DAYS)
            cmd.add_argument("--today", default=None, help="YYYY-MM-DD (default: today)")
            cmd.add_argument("--apply", action="store_true", help="really rewrite the live file")
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            out = status(args.csv, args.archive_dir)
            code = 0
        elif args.command == "archive":
            out = archive(args.csv, args.archive_dir)
            code = 0
        elif args.command == "verify":
            out = verify(args.csv, args.archive_dir, deep=args.deep)
            code = 0 if out["ok"] else 1
        else:
            today = date.fromisoformat(args.today) if args.today else None
            out = trim(args.csv, args.archive_dir, keep_days=args.keep_days, today=today, apply=args.apply)
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
