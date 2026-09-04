"""The lake write path: 4-step seal, quarantine, compaction, retirement.

Plan sec 8.3 fixes the entire write path as four steps, and nothing here may
add a fifth:

1. write ``part-<uuid>.parquet`` into ``<lake>/_incoming/`` (same volume);
2. hash + validate it (SHA-256, row count, min/max timestamps) by reading the
   staged file back, so an unreadable file never enters the tree;
3. ``os.replace()`` it into its final partition path (atomic on NTFS);
4. append one line to ``manifest_log.jsonl``.

Consequences that the crash matrix in ``tests/test_warehouse_seal.py`` pins:

* a crash during step 1-2 leaves artifacts **only** in ``_incoming/``; nothing
  in the live tree, nothing in the ledger;
* a crash between step 3 and step 4 leaves a complete file that no reader can
  see (reads are manifest-resolved), and :meth:`ResearchStore.reconcile`
  adopts it at startup when its content is not already registered, or
  quarantines it when it is - never a double count, never a deletion.

Partial-publish semantics (sec 8.3, the tracker blackout of 2026-07-13): a
bounded dirty tail is quarantined **per symbol and partition** and the clean
remainder publishes. Only manifest corruption vetoes a publish wholesale.

Nothing in this module reads or writes an operational surface, and every entry
point is inert when ``research_store_dir`` is unset.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

try:  # package import (scripts.research_warehouse.store)
    from . import config
    from .manifest import (
        ACTION_COMPACT,
        ACTION_PUBLISH,
        ACTION_QUARANTINE,
        ACTION_RETIRE,
        ManifestCorruptionError,
        ManifestEntry,
        ManifestLog,
        definitions_git_commit,
        lake_relative,
        utc_now,
    )
    from .schemas import DatasetSpec, dataset_spec, symbol_bucket
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import config  # type: ignore
    from manifest import (  # type: ignore
        ACTION_COMPACT,
        ACTION_PUBLISH,
        ACTION_QUARANTINE,
        ACTION_RETIRE,
        ManifestCorruptionError,
        ManifestEntry,
        ManifestLog,
        definitions_git_commit,
        lake_relative,
        utc_now,
    )
    from schemas import DatasetSpec, dataset_spec, symbol_bucket  # type: ignore

PARQUET_COMPRESSION = "zstd"
RETIRED_RETENTION_DAYS = 30
# A staged part file older than this is a crashed write, not a live one. The
# build job holds a single-flight lock, so the window is generous on purpose.
INCOMING_STALE_SECONDS = 3600
#: Slack for the system clock being coarser than filesystem timestamps, so a
#: file cannot be "newer than now" by clock resolution alone. See `reconcile`.
CLOCK_GRANULARITY_SECONDS = 0.05

QUARANTINE_NAIVE_TIMESTAMP = "NAIVE_TIMESTAMP"
QUARANTINE_PARTITION_KEY = "PARTITION_KEY_UNRESOLVED"
QUARANTINE_SCHEMA_CAST = "SCHEMA_CAST_FAILED"
QUARANTINE_VALIDATOR = "VALIDATOR_REJECTED"
QUARANTINE_ORPHAN_DUPLICATE = "ORPHAN_DUPLICATE_CONTENT"
#: An orphan whose rows are already live in its partition - the signature of a
#: compaction that crashed between its os.replace and its manifest append.
QUARANTINE_ORPHAN_OVERLAPS_LIVE = "ORPHAN_OVERLAPS_LIVE_ROWS"

#: Datasets whose current view is "the latest row per grain, by a time column"
#: rather than by a revision column *inside* the grain: ``outcome_path``
#: supersedes by ``computed_at`` (BD-53) and ``collection_gap`` by
#: ``detected_at`` (BD-60/BD-67). Their grain is therefore *expected* to repeat,
#: so an orphan sharing a grain key with a live row is a legitimate
#: supersession, not the double-count D14 guards against - refusing it would
#: quarantine a recomputed outcome or a gap resolution instead of adopting it.
#: Datasets that carry their revision in the grain (``setup_occurrence``'s
#: ``revision_id``, ``anchor_instance``'s ``system_from``) need no exemption.
SUPERSEDING_DATASETS = frozenset({"collection_gap", "outcome_path"})
QUARANTINE_ORPHAN_UNREADABLE = "ORPHAN_UNREADABLE_FILE"
QUARANTINE_INCOMPLETE_WRITE = "INCOMPLETE_STAGED_WRITE"


class LakeIntegrityError(RuntimeError):
    """A file the manifest declares live is gone, or a compaction disagrees."""


@dataclass
class DirtyRow:
    index: int
    reason: str
    symbol: str
    partition: str
    row: dict


@dataclass
class PublishResult:
    dataset: str
    published: list[ManifestEntry] = field(default_factory=list)
    quarantined: list[ManifestEntry] = field(default_factory=list)
    rows_published: int = 0
    rows_quarantined: int = 0
    dirty: list[DirtyRow] = field(default_factory=list)

    @property
    def partitions(self) -> list[str]:
        return sorted({entry.partition for entry in self.published})


@dataclass
class RetirementResult:
    moved: list[str] = field(default_factory=list)
    skipped_in_use: list[str] = field(default_factory=list)
    purged: list[str] = field(default_factory=list)


@dataclass
class DedupeResult:
    """One partition's answer to "how many rows repeat at the grain?" (BD-96)."""

    dataset: str = ""
    partition: str = ""
    rows_before: int = 0
    rows_after: int = 0
    rows_dropped: int = 0
    #: The COMPACT line that replaced the inputs; None for a dry run or a
    #: partition that had nothing to drop.
    entry: ManifestEntry | None = None


def _duplicate_mask(table: pa.Table, grain) -> pa.Array:
    """True for every row whose grain key equals the row BEFORE it.

    Vectorised and order-dependent: the caller sorts on the grain first (and on
    whatever should win inside a key), so the first row of each run is the one
    kept. A null on either side compares as "different", so uncertainty never
    drops a row.
    """
    n = table.num_rows
    if n == 0:
        return pa.array([], type=pa.bool_())
    columns = [name for name in grain if name in table.column_names]
    if not columns:
        return pa.array([False] * n, type=pa.bool_())
    same = None
    for name in columns:
        column = table.column(name).combine_chunks()
        equal = pc.fill_null(pc.equal(column.slice(1), column.slice(0, n - 1)), False)
        same = equal if same is None else pc.and_(same, equal)
    return pa.concat_arrays([pa.array([False], type=pa.bool_()), same])


@dataclass
class ReconcileResult:
    torn_manifest_tail_repaired: bool = False
    adopted: list[ManifestEntry] = field(default_factory=list)
    quarantined: list[ManifestEntry] = field(default_factory=list)
    missing_live_files: list[str] = field(default_factory=list)
    stale_incoming: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet_file(path: Path) -> pa.Table:
    """Read exactly one file's own columns - never a dataset around it.

    ``pq.read_table`` builds a ParquetDataset, which applies hive partitioning
    by default. A part inside ``.../year=2026/`` therefore comes back with a
    synthetic dictionary-typed ``year`` column taken from the directory name,
    on top of whatever the file actually holds. Two things break on that: a
    compaction concatenates the synthetic column and seals it into the merged
    file, so the merged file no longer matches the frozen schema its siblings
    were written to; and reading that merged file back raises
    ``ArrowTypeError`` when the path key and the now-real column disagree on
    type, which took the startup reconcile down with it.

    Every call site here means "this one file", so read the file.
    """
    return pq.ParquetFile(path).read()


def _as_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _coerce_value(field_type: pa.DataType, value):
    """Light, explicit coercion: ISO strings for dates/timestamps only.

    A naive datetime is *not* coerced - the point-in-time contract requires an
    explicit timezone, so it becomes a quarantined row instead of a guess.
    """
    if value is None:
        return None, ""
    if pa.types.is_timestamp(field_type):
        parsed = _as_datetime(value)
        if parsed is None:
            return None, QUARANTINE_SCHEMA_CAST
        if parsed.tzinfo is None:
            return None, QUARANTINE_NAIVE_TIMESTAMP
        return parsed.astimezone(timezone.utc), ""
    if pa.types.is_date(field_type):
        parsed = _as_date(value)
        if parsed is None:
            return None, QUARANTINE_SCHEMA_CAST
        return parsed, ""
    return value, ""


class ResearchStore:
    """One lake root. The main desktop is its only writer (LD-01)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = ManifestLog(self.root)

    # -- construction -------------------------------------------------------
    @classmethod
    def open(cls, root: Path | None = None) -> "ResearchStore | None":
        """The configured store, or None when the warehouse is disabled.

        Callers treat None as "do nothing"; an unset ``research_store_dir``
        must make every warehouse entry point a total no-op.
        """
        if root is None:
            if not config.warehouse_enabled():
                return None
            root = config.get_research_store_dir()
            if root is None:
                return None
        store = cls(root)
        config.ensure_lake_layout(store.root)
        return store

    # -- layout -------------------------------------------------------------
    @property
    def incoming_dir(self) -> Path:
        return self.root / "_incoming"

    @property
    def quarantine_dir(self) -> Path:
        return self.root / "_quarantine"

    @property
    def retired_dir(self) -> Path:
        return self.root / "_retired"

    def dataset_dir(self, dataset: str) -> Path:
        spec = dataset_spec(dataset)
        return self.root / spec.layer / spec.name

    def partition_dir(self, dataset: str, partition: str) -> Path:
        target = self.dataset_dir(dataset)
        for part in str(partition).split("/"):
            if part:
                target = target / part
        return target

    def partition_of(self, spec: DatasetSpec, row: dict) -> str:
        """The locked partition key for one row (sec 7.1 partition spec)."""
        pieces: list[str] = []
        for dimension in spec.partition_by:
            if dimension in {"year", "month"}:
                stamp = _as_date(row.get(spec.time_column))
                if stamp is None:
                    raise ValueError(f"row has no usable {spec.time_column} for {dimension} partitioning")
                pieces.append(f"year={stamp.year}" if dimension == "year" else f"month={stamp:%Y-%m}")
            elif dimension == "symbol_bucket":
                pieces.append(f"symbol_bucket={symbol_bucket(row.get('symbol') or '')}")
            else:
                value = row.get(dimension)
                if value in (None, ""):
                    raise ValueError(f"row has no {dimension} value for partitioning")
                pieces.append(f"{dimension}={value}")
        return "/".join(pieces)

    # -- the 4-step seal ----------------------------------------------------
    def publish(
        self,
        dataset: str,
        rows,
        *,
        job_id: str = "",
        validate=None,
        git_commit: str | None = None,
        extra: dict | None = None,
    ) -> PublishResult:
        """Seal ``rows`` into ``dataset``; quarantine only what is dirty.

        ``validate(row) -> reason|None`` lets a caller reject rows on domain
        grounds; schema/timezone/partition defects are detected here. Dirty
        rows never abort the clean remainder (sec 8.3). ``extra`` adds caller
        provenance (e.g. a bronze source path and hash) to the manifest line.
        """
        spec = dataset_spec(dataset)
        # Manifest corruption is the wholesale veto - check before any write.
        self.manifest.read_entries()
        records = list(rows.to_pylist()) if isinstance(rows, pa.Table) else [dict(row) for row in rows]
        commit = definitions_git_commit() if git_commit is None else git_commit

        clean: dict[str, list[dict]] = {}
        dirty: list[DirtyRow] = []
        for index, raw in enumerate(records):
            row, reason = self._coerce_row(spec, raw)
            if not reason and validate is not None:
                verdict = validate(row)
                if verdict:
                    reason = f"{QUARANTINE_VALIDATOR}: {verdict}"
            partition = ""
            if not reason:
                try:
                    partition = self.partition_of(spec, row)
                except ValueError as exc:
                    reason = f"{QUARANTINE_PARTITION_KEY}: {exc}"
            if reason:
                dirty.append(
                    DirtyRow(
                        index=index,
                        reason=reason,
                        symbol=str(raw.get("symbol") or "_nosymbol"),
                        partition=partition or self._best_effort_partition(spec, raw),
                        row=raw,
                    )
                )
                continue
            clean.setdefault(partition, []).append(row)

        result = PublishResult(dataset=dataset, dirty=dirty)
        for partition, partition_rows in sorted(clean.items()):
            try:
                table = pa.Table.from_pylist(partition_rows, schema=spec.schema)
            except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
                # Isolate the offenders; the rest of the partition still ships.
                keep, bad = self._isolate_bad_rows(spec, partition_rows, str(exc))
                for dirty_row in bad:
                    dirty_row.partition = partition
                dirty.extend(bad)
                if not keep:
                    continue
                table = pa.Table.from_pylist(keep, schema=spec.schema)
            entry = self._seal_table(
                spec,
                partition,
                table,
                action=ACTION_PUBLISH,
                job_id=job_id,
                git_commit=commit,
                extra=extra,
            )
            result.published.append(entry)
            result.rows_published += entry.row_count

        for entry in self._quarantine_rows(spec, dirty, job_id=job_id, git_commit=commit):
            result.quarantined.append(entry)
            result.rows_quarantined += entry.row_count
        return result

    def _best_effort_partition(self, spec: DatasetSpec, row: dict) -> str:
        """Quarantine still needs a folder when the partition key is the defect."""
        try:
            return self.partition_of(spec, row)
        except (ValueError, TypeError):
            return "unpartitioned"

    def _coerce_row(self, spec: DatasetSpec, raw: dict) -> tuple[dict, str]:
        row: dict = {}
        for pa_field in spec.schema:
            value, reason = _coerce_value(pa_field.type, raw.get(pa_field.name))
            if reason:
                return raw, f"{reason}: {pa_field.name}"
            row[pa_field.name] = value
        return row, ""

    def _isolate_bad_rows(self, spec: DatasetSpec, rows: list[dict], error: str):
        keep: list[dict] = []
        bad: list[DirtyRow] = []
        for index, row in enumerate(rows):
            try:
                pa.Table.from_pylist([row], schema=spec.schema)
            except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
                bad.append(
                    DirtyRow(
                        index=index,
                        reason=f"{QUARANTINE_SCHEMA_CAST}: {exc}",
                        symbol=str(row.get("symbol") or "_nosymbol"),
                        partition="",
                        row=row,
                    )
                )
            else:
                keep.append(row)
        if not bad:  # the failure was at table level, not row level
            raise LakeIntegrityError(f"{spec.name}: table build failed but every row casts alone: {error}")
        return keep, bad

    def _seal_table(
        self,
        spec: DatasetSpec,
        partition: str,
        table: pa.Table,
        *,
        action: str,
        job_id: str,
        git_commit: str,
        supersedes=None,
        extra: dict | None = None,
    ) -> ManifestEntry:
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        staged = self.incoming_dir / f"part-{uuid.uuid4().hex}.parquet"

        # Step 1: stage on the same volume as the final partition path.
        pq.write_table(table, staged, compression=PARQUET_COMPRESSION)

        # Step 2: hash + validate by reading the staged file back.
        digest = _sha256_file(staged)
        verify = _read_parquet_file(staged)
        if verify.num_rows != table.num_rows:
            raise LakeIntegrityError(
                f"{spec.name}/{partition}: staged file has {verify.num_rows} rows, expected {table.num_rows}"
            )
        min_ts, max_ts = self._time_bounds(spec, verify)

        # Step 3: atomic move into the live tree.
        target_dir = self.partition_dir(spec.name, partition)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / staged.name
        os.replace(staged, target)

        # Step 4: one manifest line - the moment the file becomes readable.
        return self.manifest.append(
            action=action,
            dataset=spec.name,
            partition=partition,
            file_path=lake_relative(self.root, target),
            sha256=digest,
            row_count=verify.num_rows,
            min_ts=min_ts,
            max_ts=max_ts,
            supersedes=supersedes or [],
            git_commit=git_commit,
            job_id=job_id,
            **(extra or {}),
        )

    def _time_bounds(self, spec: DatasetSpec, table: pa.Table) -> tuple[str, str]:
        if spec.time_column not in table.column_names or table.num_rows == 0:
            return "", ""
        bounds = pc.min_max(table.column(spec.time_column))
        low = bounds["min"].as_py()
        high = bounds["max"].as_py()
        return ("" if low is None else str(low), "" if high is None else str(high))

    def _quarantine_rows(self, spec: DatasetSpec, dirty: list[DirtyRow], *, job_id: str, git_commit: str):
        """Per-symbol/per-partition quarantine files; nothing is discarded."""
        grouped: dict[tuple[str, str], list[DirtyRow]] = {}
        for item in dirty:
            grouped.setdefault((item.partition or "unpartitioned", item.symbol), []).append(item)
        entries = []
        stamped = utc_now().isoformat()
        for (partition, symbol), items in sorted(grouped.items()):
            target_dir = self.quarantine_dir / spec.name
            for piece in partition.split("/"):
                if piece:
                    target_dir = target_dir / piece
            target_dir = target_dir / _safe_name(symbol)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"quarantine-{uuid.uuid4().hex}.jsonl"
            payload = "".join(
                json.dumps(
                    {"reason": item.reason, "symbol": item.symbol, "quarantined_at": stamped, "row": item.row},
                    default=str,
                )
                + "\n"
                for item in items
            )
            target.write_text(payload, encoding="utf-8")
            entries.append(
                self.manifest.append(
                    action=ACTION_QUARANTINE,
                    dataset=spec.name,
                    partition=partition,
                    file_path=lake_relative(self.root, target),
                    sha256=_sha256_file(target),
                    row_count=len(items),
                    git_commit=git_commit,
                    job_id=job_id,
                    symbol=symbol,
                    reasons=sorted({item.reason.split(":")[0] for item in items}),
                )
            )
        return entries

    # -- compaction and retirement -----------------------------------------
    def compact(self, dataset: str, partition: str, *, job_id: str = "") -> ManifestEntry | None:
        """Merge a partition's live part files behind ONE manifest line.

        The appended COMPACT record registers the replacement and retires its
        inputs simultaneously - that single line is the atomic switch. Physical
        moves into ``_retired/`` happen later and may lag.
        """
        spec = dataset_spec(dataset)
        if not spec.compactable:
            raise ValueError(f"{dataset} is never a compaction input (bronze raw / evidence freeze, sec 8.3)")
        snapshot = self.manifest.resolve(dataset=dataset, partition=partition)
        if len(snapshot.entries) < 2:
            return None
        paths = [self.root / entry.file_path for entry in snapshot.entries]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise LakeIntegrityError(f"{dataset}/{partition}: manifest-live files are missing: {missing}")
        table = pa.concat_tables([_read_parquet_file(path) for path in paths]).combine_chunks()
        expected = snapshot.row_count
        if table.num_rows != expected:
            # Logical reconciliation runs for compaction only (sec 8.3); a
            # mismatch aborts before anything is written or retired.
            raise LakeIntegrityError(
                f"{dataset}/{partition}: compaction reconciliation failed - "
                f"{table.num_rows} rows read, manifest says {expected}"
            )
        return self._seal_table(
            spec,
            partition,
            table,
            action=ACTION_COMPACT,
            job_id=job_id,
            git_commit=definitions_git_commit(),
            supersedes=[entry.file_path for entry in snapshot.entries],
        )

    def duplicate_rows(self, dataset: str, partition: str) -> DedupeResult:
        """Count rows repeated at the dataset's declared grain. Reads grain
        columns only; writes nothing. The dry run behind ``dedupe_partition``."""
        spec = dataset_spec(dataset)
        result = DedupeResult(dataset=dataset, partition=partition)
        table = self.read_table(dataset, partition, columns=list(spec.grain))
        result.rows_before = table.num_rows
        if table.num_rows:
            table = table.sort_by([(name, "ascending") for name in spec.grain])
            result.rows_after = table.num_rows - int(pc.sum(_duplicate_mask(table, spec.grain)).as_py())
        else:
            result.rows_after = 0
        result.rows_dropped = result.rows_before - result.rows_after
        return result

    def dedupe_partition(self, dataset: str, partition: str, *, job_id: str = "") -> DedupeResult:
        """Rewrite a partition with ONE row per grain key, keeping the earliest.

        A COMPACT line whose replacement is SMALLER than its inputs (BD-96). Like
        ``compact`` it is one atomic manifest switch and the inputs are retired,
        not deleted; unlike ``compact`` the row count is allowed to fall, and the
        line records how far (``rows_dropped``) and on which key (``grain``), so
        a reader of the ledger can see that rows were dropped and why.

        "Earliest" is the row with the smallest ``observed_at`` where the
        dataset carries one - the first capture is the evidence, a later twin
        is the tee re-offering it - and otherwise the first in file order.
        Refused for datasets that are never compacted and for the ones whose
        grain legitimately repeats (``SUPERSEDING_DATASETS``).
        """
        spec = dataset_spec(dataset)
        if not spec.compactable:
            raise ValueError(f"{dataset} is never a compaction input (bronze raw / evidence freeze, sec 8.3)")
        if dataset in SUPERSEDING_DATASETS:
            raise ValueError(f"{dataset}: a repeated grain key is this dataset's normal shape; nothing to dedupe")
        result = DedupeResult(dataset=dataset, partition=partition)
        snapshot = self.manifest.resolve(dataset=dataset, partition=partition)
        if not snapshot.entries:
            return result
        paths = [self.root / entry.file_path for entry in snapshot.entries]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise LakeIntegrityError(f"{dataset}/{partition}: manifest-live files are missing: {missing}")
        table = pa.concat_tables([_read_parquet_file(path) for path in paths]).combine_chunks()
        if table.num_rows != snapshot.row_count:
            raise LakeIntegrityError(
                f"{dataset}/{partition}: dedupe reconciliation failed - "
                f"{table.num_rows} rows read, manifest says {snapshot.row_count}"
            )
        result.rows_before = table.num_rows
        if "observed_at" in table.column_names:
            order = [(name, "ascending") for name in spec.grain] + [("observed_at", "ascending")]
            table = table.sort_by(order)
        else:
            table = table.sort_by([(name, "ascending") for name in spec.grain])
        mask = _duplicate_mask(table, spec.grain)
        result.rows_dropped = int(pc.sum(mask).as_py()) if table.num_rows else 0
        result.rows_after = result.rows_before - result.rows_dropped
        if result.rows_dropped == 0:
            return result  # nothing to rewrite: no line, no retirement
        kept = table.filter(pc.invert(mask))
        result.entry = self._seal_table(
            spec,
            partition,
            kept,
            action=ACTION_COMPACT,
            job_id=job_id,
            git_commit=definitions_git_commit(),
            supersedes=[entry.file_path for entry in snapshot.entries],
            extra={"dedupe_grain": list(spec.grain), "rows_dropped": result.rows_dropped},
        )
        return result

    def retire_partition(self, dataset: str, partition: str, *, job_id: str = "", reason: str = "") -> list[str]:
        """Take every live file of a partition out of the live set. One RETIRE line.

        The files are not touched: they leave the live set the instant the line
        lands and ``collect_retired`` moves them into ``_retired/<day>/`` later,
        exactly as a compaction's inputs do, so the partition is restorable by
        repointing the manifest. Returns the retired file paths. This is the
        first half of a REBUILD (BD-96): a derived partition computed from
        polluted inputs is retired whole, then recomputed session by session.
        Refused for datasets that are never compacted (bronze raw / evidence).
        """
        spec = dataset_spec(dataset)
        if not spec.compactable:
            raise ValueError(f"{dataset} is never retired wholesale (bronze raw / evidence freeze, sec 8.3)")
        snapshot = self.manifest.resolve(dataset=dataset, partition=partition)
        paths = [entry.file_path for entry in snapshot.entries]
        if not paths:
            return []
        self.manifest.append(
            action=ACTION_RETIRE,
            dataset=dataset,
            partition=partition,
            file_path="",
            supersedes=paths,
            git_commit=definitions_git_commit(),
            job_id=job_id,
            reason=reason,
        )
        return paths

    def retired_pending(self) -> list[str]:
        """Retired file paths still physically present in the live tree."""
        live = {entry.file_path for entry in self.manifest.resolve().entries}
        pending: list[str] = []
        for entry in self.manifest.read_entries():
            for path in entry.supersedes:
                if path in live or path in pending:
                    continue
                if (self.root / path).exists():
                    pending.append(path)
        return pending

    def collect_retired(self, *, now: datetime | None = None) -> RetirementResult:
        """Garbage collection: move superseded files into ``_retired/<day>/``.

        A Windows sharing violation on a file some reader still has open is
        expected and harmless - reads are manifest-resolved, so the file is
        already invisible. Skip it and retry next run (R6, LD-28).
        """
        stamp = (now or utc_now()).strftime("%Y%m%d")
        result = RetirementResult()
        for relative in self.retired_pending():
            source = self.root / relative
            target = self.retired_dir / stamp / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
            except OSError:
                result.skipped_in_use.append(relative)
                continue
            result.moved.append(relative)
        result.purged = self.purge_retired(now=now)
        return result

    def purge_retired(self, *, now: datetime | None = None, retention_days: int = RETIRED_RETENTION_DAYS) -> list[str]:
        """Drop ``_retired/<yyyymmdd>/`` folders past the rollback window."""
        cutoff = (now or utc_now()).date() - timedelta(days=retention_days)
        purged: list[str] = []
        if not self.retired_dir.exists():
            return purged
        for day_dir in sorted(self.retired_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                day = datetime.strptime(day_dir.name, "%Y%m%d").date()
            except ValueError:
                continue
            if day > cutoff:
                continue
            try:
                shutil.rmtree(day_dir)
            except OSError:
                continue
            purged.append(day_dir.name)
        return purged

    # -- startup reconciliation --------------------------------------------
    def iter_live_tree_files(self):
        for spec in _specs_by_layer(self.root):
            if not spec.exists():
                continue
            for path in sorted(spec.rglob("*.parquet")):
                yield path

    def _grain_keys(self, spec, table) -> set:
        """The dataset's identity tuples for every row of ``table``."""
        columns = [name for name in spec.grain if name in table.column_names]
        if not columns:
            return set()
        values = [table.column(name).to_pylist() for name in columns]
        return set(zip(*values))

    def _overlaps_live_rows(self, dataset: str, partition: str, table) -> bool:
        """Would adopting ``table`` duplicate rows already live in the partition?

        Compared at the dataset's declared grain rather than by file hash: the
        compaction-crash case produces a file that is byte-different from every
        registered one while containing exactly their rows.

        Cost: only the grain columns are read, and only when an unregistered
        file is actually present - reconciliation is not a hot path.
        """
        if dataset in SUPERSEDING_DATASETS:
            # A repeated grain key is this dataset's normal shape, so overlap
            # carries no information about duplication. Deciding on it would
            # quarantine legitimate supersessions.
            return False
        try:
            spec = dataset_spec(dataset)
        except KeyError:
            return False
        candidate = self._grain_keys(spec, table)
        if not candidate:
            return False
        try:
            live = self.read_table(dataset, partition or None, columns=list(spec.grain))
        except (OSError, pa.ArrowInvalid, KeyError):
            # Unreadable live state is uncertainty: refuse to adopt.
            return True
        return bool(candidate & self._grain_keys(spec, live))

    def reconcile(self, *, job_id: str = "", incoming_grace_seconds: int = INCOMING_STALE_SECONDS) -> ReconcileResult:
        """Repair what a crash can leave behind. Safe to run at every startup."""
        result = ReconcileResult()
        result.torn_manifest_tail_repaired = self.manifest.repair_torn_tail()
        entries = self.manifest.read_entries()  # raises on real corruption
        known = {entry.file_path for entry in entries}
        for entry in entries:
            known.update(entry.supersedes)
        live = self.manifest.resolve()
        live_paths = {entry.file_path: entry for entry in live.entries}
        registered_hashes: dict[tuple[str, str], set[str]] = {}
        for entry in live.entries:
            registered_hashes.setdefault((entry.dataset, entry.partition), set()).add(entry.sha256)

        for path in self.iter_live_tree_files():
            relative = lake_relative(self.root, path)
            if relative in known:
                continue
            dataset, partition = self._dataset_partition_from_path(path)
            adopted = None
            reason = QUARANTINE_ORPHAN_UNREADABLE
            if dataset:
                try:
                    table = _read_parquet_file(path)
                    digest = _sha256_file(path)
                except (OSError, pa.ArrowException):
                    # Any unreadable orphan is quarantined, never raised:
                    # reconcile runs at startup, so a single bad file must not
                    # be able to stop the desk. ArrowException is the base of
                    # ArrowInvalid and ArrowTypeError alike.
                    adopted = None
                else:
                    if digest in registered_hashes.get((dataset, partition), set()):
                        # A completed retry already published this content;
                        # adopting it a second time would double-count rows.
                        reason = QUARANTINE_ORPHAN_DUPLICATE
                    elif self._overlaps_live_rows(dataset, partition, table):
                        # BD-03's adopt-don't-discard is right for a publish
                        # retry, whose file is *new* content. A compaction that
                        # crashed between its os.replace and its manifest
                        # append leaves a merged file whose hash matches
                        # nothing registered but whose rows are all still live
                        # in the source parts - adopting it would double-count
                        # every row in the partition, silently (D14).
                        reason = QUARANTINE_ORPHAN_OVERLAPS_LIVE
                    else:
                        spec = dataset_spec(dataset)
                        min_ts, max_ts = self._time_bounds(spec, table)
                        entry = self.manifest.append(
                            action=ACTION_PUBLISH,
                            dataset=dataset,
                            partition=partition,
                            file_path=relative,
                            sha256=digest,
                            row_count=table.num_rows,
                            min_ts=min_ts,
                            max_ts=max_ts,
                            git_commit=definitions_git_commit(),
                            job_id=job_id,
                            reconciled=True,
                        )
                        registered_hashes.setdefault((dataset, partition), set()).add(digest)
                        result.adopted.append(entry)
                        adopted = entry
            if adopted is None:
                result.quarantined.append(
                    self._quarantine_file(path, dataset or "_unknown", partition, reason, job_id=job_id)
                )

        for relative, entry in live_paths.items():
            if not (self.root / relative).exists():
                result.missing_live_files.append(relative)

        # The clock is coarser than the filesystem. Windows' system clock ticks
        # about every 15.6 ms while NTFS stamps mtimes far more finely, so
        # `utc_now()` can round BELOW the mtime of a file written microseconds
        # earlier - and that file then reads as "from the future" and is never
        # quarantined. Harmless against the 3600 s default grace, but with
        # `incoming_grace_seconds=0` it made the outcome a coin flip (measured
        # 3 failures in 6 runs on this desk, reproducing in isolation, 2026-08-15).
        #
        # Widening by one clock tick fixes the real bug rather than the symptom:
        # a file written in the same tick as the check IS stale and should be
        # quarantined. 50 ms is inconsequential beside any real grace period.
        cutoff = (
            utc_now() - timedelta(seconds=incoming_grace_seconds)
        ).timestamp() + CLOCK_GRANULARITY_SECONDS
        if self.incoming_dir.exists():
            for path in sorted(self.incoming_dir.iterdir()):
                if not path.is_file() or path.stat().st_mtime > cutoff:
                    continue
                result.stale_incoming.append(path.name)
                result.quarantined.append(
                    self._quarantine_file(path, "_incoming", "", QUARANTINE_INCOMPLETE_WRITE, job_id=job_id)
                )
        return result

    def _dataset_partition_from_path(self, path: Path) -> tuple[str, str]:
        try:
            relative = Path(path).relative_to(self.root)
        except ValueError:
            return "", ""
        parts = relative.parts
        if len(parts) < 3:
            return "", ""
        dataset = parts[1]
        try:
            dataset_spec(dataset)
        except KeyError:
            return "", ""
        return dataset, "/".join(parts[2:-1])

    def _quarantine_file(self, path: Path, dataset: str, partition: str, reason: str, *, job_id: str) -> ManifestEntry:
        target_dir = self.quarantine_dir / dataset
        for piece in str(partition).split("/"):
            if piece:
                target_dir = target_dir / piece
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        if target.exists():
            target = target_dir / f"{target.stem}-{uuid.uuid4().hex[:8]}{target.suffix}"
        os.replace(path, target)
        return self.manifest.append(
            action=ACTION_QUARANTINE,
            dataset=dataset,
            partition=partition,
            file_path=lake_relative(self.root, target),
            sha256=_sha256_file(target),
            row_count=0,
            git_commit=definitions_git_commit(),
            job_id=job_id,
            reasons=[reason],
        )

    # -- manifest-resolved reads -------------------------------------------
    def resolve_files(self, dataset: str, partition: str | None = None) -> list[Path]:
        snapshot = self.manifest.resolve(dataset=dataset, partition=partition)
        paths = []
        for entry in snapshot.entries:
            path = self.root / entry.file_path
            if not path.exists():
                raise LakeIntegrityError(
                    f"{entry.file_path} is manifest-live but missing on disk; restore before reading."
                )
            paths.append(path)
        return paths

    def open_dataset(self, dataset: str, partition: str | None = None) -> pads.Dataset:
        """A pyarrow dataset over an explicit, manifest-resolved file list.

        The file list is fixed at query start, which is what gives one query a
        consistent snapshot across a concurrent compaction.
        """
        spec = dataset_spec(dataset)
        paths = [str(path) for path in self.resolve_files(dataset, partition)]
        return pads.dataset(paths, schema=spec.schema, format="parquet")

    def read_table(self, dataset: str, partition: str | None = None, columns=None) -> pa.Table:
        spec = dataset_spec(dataset)
        paths = self.resolve_files(dataset, partition)
        if not paths:
            empty = spec.schema.empty_table()
            return empty.select(list(columns)) if columns else empty
        return self.open_dataset(dataset, partition).to_table(columns=list(columns) if columns else None)

    def read_rows(
        self,
        dataset: str,
        partition: str | None = None,
        *,
        columns=None,
        symbols=None,
        occurrence_ids=None,
        recipe_ids=None,
        interval_start_range: tuple[datetime | None, datetime | None] | None = None,
        time_column: str = "interval_start",
    ) -> list[dict]:
        """Rows as dicts, with the narrowing done in Arrow BEFORE Python sees them.

        `read_table(...).to_pylist()` materialises an entire partition as
        Python objects and then lets the caller throw most of it away. Our
        partitions are MONTH-keyed, so that cost grows all month: measured
        2026-08-27, `bar_m5 month=2026-08` held 8,704,108 rows / 408 MB of
        parquet, `to_pylist` cost 1,769 bytes per row = **15.4 GB**, and the
        largest single session in it was 588,778 rows - 6.8% of the month.
        That is why the desk climbed to 8-13 GB after every swing-scan slot
        and fell back minutes later.

        Pushing the same predicate into `Dataset.to_table(filter=...)` lets
        Arrow drop row groups and rows before any Python object exists, so the
        peak scales with the slice the caller actually needs.

        The narrowing is deliberately NOT a free-form expression: only the two
        predicates the readers actually use are offered, so a caller cannot
        express a filter that silently means something other than the Python
        one it replaced.

        ``symbols`` and ``occurrence_ids`` match EXACTLY, with no case folding or stripping,
        because that is what the Python ``symbol in wanted`` checks it replaces
        did; an empty sequence means "no symbol filter", which is what the
        callers pass when no cohort was named. ``occurrence_ids`` is the same
        bounded seam for outcome/context rows, so a 32-bucket research run
        never materialises every historical recipe result. ``interval_start_range`` is
        half-open ``[start, end)``, matching ``rth_open_at <= t < rth_close_at``.

        ``time_column`` names the column that range applies to and defaults to
        ``interval_start``, which every caller before P10 used and which is
        therefore unchanged for all of them. It exists because the bar datasets
        are not the only ones with a time to narrow on: ``setup_occurrence``
        carries ``trigger_at`` and ``event_at`` and no ``interval_start`` at all,
        so a caller that wanted a date window there had either to name its column
        or to pull the year into Python and filter it - which is the thing this
        method exists to stop. It is still a NAME, not an expression: the
        predicate built from it is the same half-open comparison, so it cannot
        silently mean something else.
        """
        paths = self.resolve_files(dataset, partition)
        if not paths:
            return []
        predicate = None
        if symbols:
            wanted = sorted({str(symbol) for symbol in symbols})
            predicate = pads.field("symbol").isin(wanted)
        if occurrence_ids:
            wanted_occurrences = sorted({str(identity) for identity in occurrence_ids})
            clause = pads.field("occurrence_id").isin(wanted_occurrences)
            predicate = clause if predicate is None else (predicate & clause)
        if recipe_ids:
            wanted_recipes = sorted({str(identity) for identity in recipe_ids})
            clause = pads.field("recipe_id").isin(wanted_recipes)
            predicate = clause if predicate is None else (predicate & clause)
        if interval_start_range is not None:
            start, end = interval_start_range
            for bound, operator in ((start, "ge"), (end, "lt")):
                if bound is None:
                    continue
                # Compared as instants: the column is timestamp[us, tz=UTC], so
                # an aware bound in any zone is normalized rather than stripped.
                if isinstance(bound, datetime) and bound.tzinfo is not None:
                    bound = bound.astimezone(timezone.utc)
                field = pads.field(time_column)
                clause = field >= pc.scalar(bound) if operator == "ge" else field < pc.scalar(bound)
                predicate = clause if predicate is None else (predicate & clause)
        table = self.open_dataset(dataset, partition).to_table(
            columns=list(columns) if columns else None, filter=predicate
        )
        return table.to_pylist()

    # -- health -------------------------------------------------------------
    def health_counts(self) -> dict:
        """Feeds the sec-18 Health tiles (quarantine count, manifest integrity)."""
        try:
            entries = self.manifest.read_entries()
            corrupt = ""
        except ManifestCorruptionError as exc:
            return {"manifest_corrupt": str(exc)}
        live = self.manifest.resolve()
        live_paths = {entry.file_path for entry in live.entries}
        known = {entry.file_path for entry in entries}
        for entry in entries:
            known.update(entry.supersedes)
        unmanifested = [
            lake_relative(self.root, path)
            for path in self.iter_live_tree_files()
            if lake_relative(self.root, path) not in known
        ]
        quarantine = self.manifest.quarantine_entries()
        last_seal = next(
            (entry for entry in reversed(entries) if entry.action in {ACTION_PUBLISH, ACTION_COMPACT}),
            None,
        )
        return {
            "manifest_corrupt": corrupt,
            "manifest_seq": live.manifest_seq,
            "live_files": len(live_paths),
            "live_rows": live.row_count,
            # Tile 6: live files not in the manifest must be 0 (_retired/ is
            # expected GC lag and is not counted here).
            "unmanifested_live_files": len(unmanifested),
            "missing_live_files": len([p for p in live_paths if not (self.root / p).exists()]),
            "quarantine_files": len(quarantine),
            "quarantine_rows": sum(entry.row_count for entry in quarantine),
            "retired_pending": len(self.retired_pending()),
            "last_seal_at": last_seal.written_at if last_seal else "",
            "last_seal_dataset": last_seal.dataset if last_seal else "",
        }


def _specs_by_layer(root: Path):
    return [root / "bronze", root / "silver", root / "gold"]


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return cleaned or "_nosymbol"
