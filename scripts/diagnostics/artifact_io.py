"""Shared artifact I/O primitives for diagnostics evidence (plan.md sec 4, 6.1).

Shadow logs, run manifests, coverage summaries and per-day rollups are the
evidence base for every promotion decision, so *how* they are written matters as
much as what they contain. Roughly fifteen modules grew their own private
``mkstemp`` + ``os.replace`` variant, three grew their own ``_config_hash``, and
none of them clean the temp file up on the failure paths - the live diagnostics
directory accumulates orphaned ``tmp*.tmp`` files as proof.

This module is the single implementation of those primitives:

- :func:`atomic_write_json` - crash-safe whole-file replace with **guaranteed**
  temp cleanup on every failure path;
- :func:`append_jsonl` / :func:`append_jsonl_rows` / :func:`read_jsonl` -
  one-record-per-line append logs that survive a previously truncated write;
- :func:`config_hash` - a stable hash over canonical sorted JSON, identical
  across processes, runs and ``PYTHONHASHSEED`` values;
- :func:`prune_by_age` / :func:`prune_by_size` - bounded retention;
- :func:`archive_dated` - **copy** an artifact into a dated archive, preserving
  the original (plan.md sec 6.1 step 6: rotate prior shadow logs *without
  deleting them*);
- :func:`sweep_stale_temp_files` - reclaim orphans left by the old writers.

This module is pure infrastructure: it holds no engine state, makes no trading
decision, and does not change any detector or scoring behavior. Adopting it in
existing writers is deliberately a separate packet.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time as _time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path

__all__ = [
    "TEMP_SUFFIX",
    "append_jsonl",
    "append_jsonl_rows",
    "archive_dated",
    "atomic_write_json",
    "canonical_json",
    "config_hash",
    "diagnostics_dir",
    "diagnostics_path",
    "prune_by_age",
    "prune_by_size",
    "read_jsonl",
    "sweep_stale_temp_files",
]

# Every temp file this module creates ends in TEMP_SUFFIX, so a stale-temp sweep
# can be scoped precisely. It matches the legacy ``tempfile.mkstemp(suffix=".tmp")``
# orphans the older writers left behind, which is intentional.
TEMP_SUFFIX = ".tmp"
_TEMP_PREFIX = "artifact-"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# location
# ---------------------------------------------------------------------------
def diagnostics_dir() -> Path:
    """Machine-local diagnostics root (honors ``TRADINGBOT_DIAGNOSTICS_DIR``).

    Imported lazily, exactly like :mod:`diagnostics.run_manifest` does, because
    importing ``project_paths`` has startup side effects (shared-drive wait,
    legacy migration) that a diagnostics helper must never trigger on import.
    """
    try:
        from project_paths import get_diagnostics_dir

        return get_diagnostics_dir()
    except Exception:
        return Path.home() / ".tradingbotv3" / "diagnostics"


def diagnostics_path(*parts: str) -> Path:
    """Path under the diagnostics root, e.g. ``diagnostics_path("archive", name)``."""
    return diagnostics_dir().joinpath(*parts)


# ---------------------------------------------------------------------------
# canonical JSON + hashing
# ---------------------------------------------------------------------------
def _canonical(value):
    """Reduce ``value`` to JSON-safe primitives with a deterministic ordering.

    Determinism is the whole point: no reliance on dict insertion order, on set
    iteration order (which varies with ``PYTHONHASHSEED``), or on ``repr()`` of
    objects that embed a memory address.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # NaN/inf round-trip through json as NaN/Infinity tokens; normalize them
        # to stable strings so the encoding never depends on the encoder flags.
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, (datetime, date, _time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return f"timedelta:{value.total_seconds()}"
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, Mapping):
        items = [(str(key), _canonical(item)) for key, item in value.items()]
        items.sort(key=lambda pair: pair[0])
        return dict(items)
    if isinstance(value, Set):
        encoded = [_canonical(item) for item in value]
        # Sort by encoding, not by the values themselves: a set may mix types
        # that are not mutually orderable.
        encoded.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return encoded
    if isinstance(value, Sequence):  # list/tuple (str/bytes handled above)
        return [_canonical(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    payload = getattr(value, "__dict__", None)
    if isinstance(payload, Mapping):
        # Plain config objects (MarketStateConfig, engine configs, ...).
        return _canonical({k: v for k, v in payload.items() if not str(k).startswith("_")})
    return str(value)


def canonical_json(obj) -> str:
    """Canonical JSON text for ``obj``: sorted keys, no incidental whitespace."""
    return json.dumps(
        _canonical(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(obj, *, length: int | None = None) -> str:
    """Stable SHA-256 over ``obj``'s canonical JSON.

    Every diagnostics artifact stamps the configuration that produced it
    (plan.md sec 4: "configuration hash"), so this must be reproducible across
    processes, machines and interpreter restarts. Dict insertion order, set
    iteration order and ``PYTHONHASHSEED`` cannot change the result.

    ``length`` truncates the hex digest (e.g. ``length=12`` for log stamps).
    """
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    if length is None:
        return digest
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    return digest[:length]


# ---------------------------------------------------------------------------
# atomic whole-file writes
# ---------------------------------------------------------------------------
def atomic_write_json(path: Path | str, obj, *, indent: int | None = 1, fsync: bool = True) -> Path:
    """Write ``obj`` as JSON to ``path`` atomically, leaving no temp file behind.

    The record is serialized *first*, so a serialization failure (circular
    reference, exploding ``__dict__``) never touches the filesystem at all;
    exotic field types degrade through :func:`_canonical` rather than losing the
    whole artifact. The bytes then go to a temp file in the **same directory**
    (so ``os.replace`` is a same-volume atomic rename) and the rename publishes
    them. A reader therefore sees either the previous file or the complete new
    one - never a half-written artifact.

    Unlike the copies scattered across the codebase, the temp file is removed on
    *every* failure path (serialization, write, fsync, rename, interrupt), which
    is why the live diagnostics directory grows orphaned ``tmp*.tmp`` files.
    """
    path = Path(path)
    payload = json.dumps(obj, indent=indent, default=_canonical)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=_TEMP_PREFIX, suffix=TEMP_SUFFIX)
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
    except BaseException:
        os.close(fd)
        _remove_quietly(tmp_name)
        raise
    try:
        with handle:
            handle.write(payload)
            if not payload.endswith("\n"):
                handle.write("\n")
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        _remove_quietly(tmp_name)
        raise
    return path


def _remove_quietly(target: Path | str) -> None:
    try:
        os.remove(target)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# append-only JSONL
# ---------------------------------------------------------------------------
def append_jsonl(path: Path | str, obj, *, fsync: bool = False) -> Path:
    """Append one JSON record as a single line to ``path``.

    One ``write`` of one complete line keeps concurrent appenders from
    interleaving partial records. If a previous process died mid-line, a
    separating newline is inserted first so the damage stays confined to that
    one line instead of swallowing this record too.

    ``fsync=True`` forces the record to disk; it is off by default because
    shadow loops append on every evaluation.
    """
    return append_jsonl_rows(path, (obj,), fsync=fsync)


def append_jsonl_rows(path: Path | str, rows, *, fsync: bool = False) -> Path:
    """Append many JSON records (one line each) in a single open/close."""
    path = Path(path)
    lines = [json.dumps(row, separators=(",", ":"), default=_canonical) + "\n" for row in rows]
    if not lines:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "\n" if _needs_separating_newline(path) else ""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(prefix + "".join(lines))
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    return path


def _needs_separating_newline(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 0:
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) not in (b"\n", b"\r")
    except OSError:
        return False


def read_jsonl(path: Path | str, *, skip_bad: bool = True) -> list[dict]:
    """Read a JSONL artifact into a list of records.

    Truncated or corrupt lines are skipped by default: a partially flushed tail
    line must never make a whole day of evidence unreadable.
    """
    path = Path(path)
    out: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if skip_bad:
                continue
            raise
    return out


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------
def _matching_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(
        (p for p in directory.glob(pattern) if p.is_file()),
        key=lambda p: (p.stat().st_mtime, p.name),
    )


def prune_by_age(
    directory: Path | str,
    max_age_days: float,
    *,
    pattern: str = "*",
    keep_newest: int = 0,
    now: float | None = None,
) -> int:
    """Delete files in ``directory`` older than ``max_age_days``; return the count.

    Boundary rule: a file whose age is *exactly* ``max_age_days`` is kept; only
    strictly older files go. ``keep_newest`` always spares that many most-recent
    files regardless of age, so a low-traffic artifact never disappears
    entirely. ``max_age_days <= 0`` is a no-op (a guard against a mis-set
    config wiping the evidence base).
    """
    directory = Path(directory)
    if not directory.is_dir() or max_age_days is None or float(max_age_days) <= 0:
        return 0
    cutoff = (time.time() if now is None else float(now)) - float(max_age_days) * 86400.0
    files = _matching_files(directory, pattern)
    keep_newest = max(0, int(keep_newest))
    candidates = files[: len(files) - keep_newest] if keep_newest else files
    removed = 0
    for path in candidates:
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def prune_by_size(
    directory: Path | str,
    max_bytes: int,
    *,
    pattern: str = "*",
    keep_newest: int = 1,
) -> int:
    """Delete oldest files until the matched set fits in ``max_bytes``.

    ``keep_newest`` (default 1) is a floor: the newest artifact is never deleted
    to satisfy a budget, so a single oversized current log cannot leave the
    directory empty. ``max_bytes <= 0`` is a no-op.
    """
    directory = Path(directory)
    if not directory.is_dir() or max_bytes is None or int(max_bytes) <= 0:
        return 0
    files = _matching_files(directory, pattern)
    keep_newest = max(0, int(keep_newest))
    sizes: list[tuple[Path, int]] = []
    for path in files:
        try:
            sizes.append((path, path.stat().st_size))
        except OSError:
            continue
    total = sum(size for _, size in sizes)
    prunable = sizes[: len(sizes) - keep_newest] if keep_newest else sizes
    removed = 0
    for path, size in prunable:
        if total <= int(max_bytes):
            break
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        removed += 1
    return removed


def sweep_stale_temp_files(
    directory: Path | str,
    *,
    min_age_seconds: float = 3600.0,
    pattern: str = f"*{TEMP_SUFFIX}",
    now: float | None = None,
) -> int:
    """Remove orphaned ``*.tmp`` files older than ``min_age_seconds``.

    These are the leftovers of interrupted writes (including the pre-existing
    ones in the live diagnostics directory). The age floor guarantees a
    concurrently running write is never sniped mid-flight.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    cutoff = (time.time() if now is None else float(now)) - max(0.0, float(min_age_seconds))
    removed = 0
    for path in directory.glob(pattern):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# dated archive (non-destructive)
# ---------------------------------------------------------------------------
def _safe_stamp(session_date) -> str:
    if isinstance(session_date, (datetime, date)):
        text = session_date.date().isoformat() if isinstance(session_date, datetime) else session_date.isoformat()
    else:
        text = str(session_date or "").strip()
    text = _SAFE_NAME_RE.sub("-", text).strip("-.")
    if not text:
        raise ValueError("session_date must be a non-empty date or date-like string")
    return text


def archive_dated(
    path: Path | str,
    session_date,
    *,
    archive_dir: Path | str | None = None,
) -> Path | None:
    """COPY ``path`` into a dated archive file and return the archive path.

    plan.md sec 6.1 step 6 requires prior shadow logs be "archived or rotated
    **without deleting them**", so this helper is strictly additive:

    - the source file is left exactly as it was (never moved, never truncated);
    - an existing archive for the same date is never overwritten - a ``-2``,
      ``-3``, ... suffix is appended instead;
    - the copy itself is staged through a temp file and published with
      ``os.replace``, so a half-copied archive is never visible.

    Returns ``None`` when the source does not exist (nothing to preserve).
    Callers that also want the live file emptied must do that themselves,
    *after* confirming the returned archive path exists.
    """
    path = Path(path)
    if not path.is_file():
        return None
    stamp = _safe_stamp(session_date)
    target_dir = Path(archive_dir) if archive_dir is not None else path.parent / "archive"
    target_dir.mkdir(parents=True, exist_ok=True)

    base = f"{path.stem}-{stamp}" if path.stem else stamp
    target = target_dir / f"{base}{path.suffix}"
    counter = 2
    while target.exists():
        target = target_dir / f"{base}-{counter}{path.suffix}"
        counter += 1

    fd, tmp_name = tempfile.mkstemp(dir=str(target_dir), prefix=_TEMP_PREFIX, suffix=TEMP_SUFFIX)
    os.close(fd)
    try:
        shutil.copyfile(path, tmp_name)
        os.replace(tmp_name, target)
    except BaseException:
        _remove_quietly(tmp_name)
        raise
    return target
