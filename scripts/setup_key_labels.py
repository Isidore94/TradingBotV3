"""Short setup-key labels for display rows (P1-5 5a). Display only.

The scan writes one `d1_features.csv` row per symbol (the newest rows of
`d1_features_history.csv`). Once P1-4 stamps a `permutation_label` column there,
this module maps `(SYMBOL, SIDE)` to that label so the setups table and the
digest can show it. No column, no file, or no label = nothing shown.

`warm_cache()` reads the file and must run off the Qt thread; `attach_labels()`
reads only the in-memory map unless `allow_read=True` (workers, tests, CLIs).
"""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Any, Iterable

LABEL_COLUMN = "permutation_label"
#: Longest label shown in a cell or a digest line; longer ones end in "..".
SHORT_LABEL_MAX = 40

_lock = threading.Lock()
_cache: dict[str, Any] = {"signature": None, "labels": {}}


def short_label(value: Any, *, max_len: int = SHORT_LABEL_MAX) -> str:
    """One short display label, or "" when there is nothing to show."""
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return ""
    if len(text) > max_len:
        return text[: max(1, max_len - 2)].rstrip("|_ ") + ".."
    return text


def _default_path() -> Path:
    from project_paths import D1_FEATURES_FILE

    return Path(D1_FEATURES_FILE)


def read_labels(path: Path | None = None) -> dict[tuple[str, str], dict[str, str]]:
    """`{(SYMBOL, SIDE): {"label", "date"}}` from the features file; `{}` without the column."""
    target = Path(path) if path is not None else _default_path()
    out: dict[tuple[str, str], dict[str, str]] = {}
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if LABEL_COLUMN not in (reader.fieldnames or ()):
            return {}
        for row in reader:
            label = short_label(row.get(LABEL_COLUMN))
            symbol = str(row.get("symbol") or "").strip().upper()
            if not label or not symbol:
                continue
            side = str(row.get("side") or "").strip().upper()
            out[(symbol, side)] = {
                "label": label,
                "date": str(row.get("last_trade_date") or "").strip()[:10],
            }
    return out


def warm_cache(path: Path | None = None) -> bool:
    """Re-read the features file when it moved. True when the map changed. Never on Qt."""
    target = Path(path) if path is not None else _default_path()
    try:
        stat = target.stat()
        signature = (str(target), stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = (str(target), None, None)
    with _lock:
        if signature == _cache["signature"]:
            return False
    try:
        labels = read_labels(target) if signature[1] is not None else {}
    except Exception:  # noqa: BLE001 - a label never costs the table
        logging.warning("Setup key labels unreadable; keeping the last map.", exc_info=True)
        return False
    with _lock:
        changed = labels != _cache["labels"]
        _cache["signature"] = signature
        _cache["labels"] = labels
    return changed


def cached_labels() -> dict[tuple[str, str], dict[str, str]]:
    """The map already in memory. Never opens a file."""
    with _lock:
        return _cache["labels"]


def attach_labels(rows: Iterable[Any], *, allow_read: bool = False, path: Path | None = None) -> int:
    """Fill `row.raw["permutation_label"]` where the row has none. Returns rows filled.

    Matched on symbol and side; a features row from another session date is skipped.
    """
    if allow_read:
        warm_cache(path)
    labels = cached_labels()
    if not labels:
        return 0
    filled = 0
    for row in rows or ():
        raw = getattr(row, "raw", None)
        if not isinstance(raw, dict) or short_label(raw.get(LABEL_COLUMN)):
            continue
        key = (
            str(getattr(row, "symbol", "") or "").strip().upper(),
            str(getattr(row, "side", "") or "").strip().upper(),
        )
        entry = labels.get(key)
        if not entry:
            continue
        row_date = str(getattr(row, "last_trade_date", "") or raw.get("last_trade_date") or "")[:10]
        if row_date and entry["date"] and row_date != entry["date"]:
            continue
        raw[LABEL_COLUMN] = entry["label"]
        filled += 1
    return filled


def row_label(row: Any) -> str:
    """The short label a display row carries, or ""."""
    raw = getattr(row, "raw", None)
    if isinstance(row, dict):
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else row
    return short_label((raw or {}).get(LABEL_COLUMN)) if isinstance(raw, dict) else ""


def reset_cache_for_tests() -> None:
    with _lock:
        _cache["signature"] = None
        _cache["labels"] = {}
