"""Short setup-key labels for display rows (P1-5 5a). Display only.

The scan writes one `d1_features.csv` row per symbol (the newest rows of
`d1_features_history.csv`). Once P1-4 stamps a `permutation_label` column there,
this module maps `(SYMBOL, SIDE)` to that label so the setups table and the
digest can show it. No column, no file, or no label = nothing shown.

P12: `permutation_verdicts.json` (beside the features file) marks facet keys as
a weak variant or a promotion candidate. A row whose stamped `permutation_key`
holds a verdict key's facets (same family and side) carries that verdict, and
the chip "(weak variant)" / "(candidate)" shows beside its label. Rank and
annotate only: a verdict never hides a row.

`warm_cache()` reads the files and must run off the Qt thread; `attach_labels()`
reads only the in-memory maps unless `allow_read=True` (workers, tests, CLIs).
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

LABEL_COLUMN = "permutation_label"
KEY_COLUMN = "permutation_key"
#: Set on `row.raw` by `attach_labels`: the verdicts this row's key carries.
VERDICT_FIELD = "permutation_verdicts"
VERDICTS_FILE_NAME = "permutation_verdicts.json"
WEAK = "weak_variant"
CANDIDATE = "promotion_candidate"
CHIPS = {WEAK: "weak variant", CANDIDATE: "candidate"}
#: Longest label shown in a cell or a digest line; longer ones end in "..".
SHORT_LABEL_MAX = 40

_lock = threading.Lock()
_cache: dict[str, Any] = {"signature": None, "labels": {}, "verdict_signature": None, "verdicts": []}


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
    """`{(SYMBOL, SIDE): {"label", "date"[, "key"]}}` from the features file; `{}` without the column."""
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
            entry = {
                "label": label,
                "date": str(row.get("last_trade_date") or "").strip()[:10],
            }
            key = str(row.get(KEY_COLUMN) or "").strip()
            if key:
                entry["key"] = key
            out[(symbol, side)] = entry
    return out


def parse_compact_key(text: Any) -> tuple[str, str, dict[str, str]] | None:
    """`version|family|side|a=b;c=d` -> `(family, SIDE, {facet: value})`, or None."""
    parts = str(text or "").strip().split("|", 3)
    if len(parts) != 4 or not parts[1]:
        return None
    facets = dict(item.split("=", 1) for item in parts[3].split(";") if "=" in item)
    return parts[1], parts[2].strip().upper(), facets


def read_verdicts(path: Path) -> list[dict[str, Any]]:
    """The swing verdicts from `permutation_verdicts.json`; `[]` when there is no file."""
    target = Path(path)
    if not target.is_file():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    out = []
    for verdict in (payload or {}).get("verdicts") or ():
        if not isinstance(verdict, Mapping) or verdict.get("population") != "swing":
            continue
        if verdict.get("verdict") not in CHIPS or not verdict.get("facets"):
            continue
        out.append({
            "verdict": str(verdict["verdict"]),
            "family": str(verdict.get("family") or ""),
            "side": str(verdict.get("side") or "").upper(),
            "facets": {str(k): str(v) for k, v in dict(verdict["facets"]).items()},
            "label": str(verdict.get("label") or ""),
            "horizon": str(verdict.get("horizon") or ""),
            "citation": str(verdict.get("citation") or ""),
        })
    return out


def matching_verdicts(compact_key: Any, verdicts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Verdicts whose facets all sit on this row's key, same family and side."""
    parsed = parse_compact_key(compact_key)
    if parsed is None:
        return []
    family, side, facets = parsed
    return [
        {"verdict": v["verdict"], "label": v["label"], "horizon": v["horizon"], "citation": v["citation"]}
        for v in verdicts
        if v["family"] == family and v["side"] == side
        and all(facets.get(name) == value for name, value in v["facets"].items())
    ]


def _signature(target: Path) -> tuple:
    try:
        stat = target.stat()
        return (str(target), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(target), None, None)


def _default_verdicts_path(features_path: Path | None) -> Path:
    if features_path is not None:
        return Path(features_path).parent / VERDICTS_FILE_NAME
    from project_paths import SETUP_PERMUTATION_VERDICTS_FILE

    return Path(SETUP_PERMUTATION_VERDICTS_FILE)


def warm_verdicts(path: Path | None = None) -> bool:
    """Re-read the verdicts file when it moved. True when the list changed. Never on Qt."""
    target = Path(path) if path is not None else _default_verdicts_path(None)
    signature = _signature(target)
    with _lock:
        if signature == _cache["verdict_signature"]:
            return False
    try:
        verdicts = read_verdicts(target) if signature[1] is not None else []
    except Exception:  # noqa: BLE001 - a chip never costs the table
        logging.warning("Setup key verdicts unreadable; keeping the last list.", exc_info=True)
        return False
    with _lock:
        changed = verdicts != _cache["verdicts"]
        _cache["verdict_signature"] = signature
        _cache["verdicts"] = verdicts
    return changed


def warm_cache(path: Path | None = None, verdicts_path: Path | None = None) -> bool:
    """Re-read the features and verdicts files when they moved. True when a map changed. Never on Qt."""
    verdicts_changed = warm_verdicts(verdicts_path if verdicts_path is not None else _default_verdicts_path(path))
    return _warm_labels(path) or verdicts_changed


def _warm_labels(path: Path | None = None) -> bool:
    target = Path(path) if path is not None else _default_path()
    signature = _signature(target)
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


def cached_verdicts() -> list[dict[str, Any]]:
    """The verdict list already in memory. Never opens a file."""
    with _lock:
        return _cache["verdicts"]


def attach_labels(rows: Iterable[Any], *, allow_read: bool = False, path: Path | None = None) -> int:
    """Fill `row.raw["permutation_label"]` where the row has none. Returns rows filled.

    Matched on symbol and side; a features row from another session date is
    skipped. Also sets `row.raw["permutation_verdicts"]` from the verdict list
    (P12), from the row's own stamped key or else the features row's.
    """
    if allow_read:
        warm_cache(path)
    labels = cached_labels()
    verdicts = cached_verdicts()
    if not labels and not verdicts:
        return 0
    filled = 0
    for row in rows or ():
        raw = getattr(row, "raw", None)
        if not isinstance(raw, dict):
            continue
        key = (
            str(getattr(row, "symbol", "") or "").strip().upper(),
            str(getattr(row, "side", "") or "").strip().upper(),
        )
        entry = labels.get(key)
        row_date = str(getattr(row, "last_trade_date", "") or raw.get("last_trade_date") or "")[:10]
        if entry and row_date and entry["date"] and row_date != entry["date"]:
            entry = None
        if entry and not short_label(raw.get(LABEL_COLUMN)):
            raw[LABEL_COLUMN] = entry["label"]
            filled += 1
        compact = raw.get(KEY_COLUMN) or (entry or {}).get("key")
        matches = matching_verdicts(compact, verdicts) if verdicts and compact else []
        if matches:
            raw[VERDICT_FIELD] = matches
        else:
            raw.pop(VERDICT_FIELD, None)
    return filled


def _raw_of(row: Any) -> dict | None:
    raw = getattr(row, "raw", None)
    if isinstance(row, dict):
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else row
    return raw if isinstance(raw, dict) else None


def row_label(row: Any) -> str:
    """The short label a display row carries, or ""."""
    raw = _raw_of(row)
    return short_label(raw.get(LABEL_COLUMN)) if raw is not None else ""


def row_verdicts(row: Any) -> list[dict[str, Any]]:
    """The verdicts `attach_labels` put on this row, or []."""
    raw = _raw_of(row)
    value = (raw or {}).get(VERDICT_FIELD)
    return [v for v in value if isinstance(v, Mapping)] if isinstance(value, list) else []


def row_chips(row: Any) -> list[str]:
    """Chip texts for this row, weak first: "weak variant", "candidate"."""
    kinds = {str(v.get("verdict") or "") for v in row_verdicts(row)}
    return [CHIPS[kind] for kind in (WEAK, CANDIDATE) if kind in kinds]


def is_weak_variant(row: Any) -> bool:
    return any(v.get("verdict") == WEAK for v in row_verdicts(row))


def display_label(row: Any) -> str:
    """The short label plus its chips, e.g. `sma100_support (weak variant)`; "" when neither."""
    return " ".join([row_label(row), *(f"({chip})" for chip in row_chips(row))]).strip()


def verdict_tooltip(row: Any) -> str:
    """One citation line per verdict the row carries."""
    return "\n".join(
        f"{CHIPS.get(str(v.get('verdict')), '')}: {v.get('citation', '')}" for v in row_verdicts(row)
    )


def weak_variants_last(rows: Sequence[Any], group_of: Callable[[Any], Hashable]) -> list[Any]:
    """Each weak-variant row moves to just after the last non-weak row of its group.

    Stable: every other row keeps its place, weak rows of one group keep their
    order, and a group with no other rows leaves its weak rows where they were.
    Display only: the same rows come back, none hidden.
    """
    groups = [group_of(row) for row in rows]
    weak = [is_weak_variant(row) for row in rows]
    last_peer: dict[Hashable, int] = {}
    for index, (group, is_weak) in enumerate(zip(groups, weak, strict=True)):
        if not is_weak:
            last_peer[group] = index
    out: list[Any] = []
    pending: dict[Hashable, list[Any]] = {}
    for index, row in enumerate(rows):
        group = groups[index]
        if weak[index] and last_peer.get(group, -1) > index:
            pending.setdefault(group, []).append(row)
            continue
        out.append(row)
        if not weak[index] and last_peer.get(group) == index:
            out.extend(pending.pop(group, []))
    return out


def reset_cache_for_tests() -> None:
    with _lock:
        _cache["signature"] = None
        _cache["labels"] = {}
        _cache["verdict_signature"] = None
        _cache["verdicts"] = []
