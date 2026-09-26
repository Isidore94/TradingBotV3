"""The swing table's ONE context payload (p9), read on a worker thread. Never on the Qt thread.

`read_swing_context` gathers what the day's modules already publish, for
`ui.models.swing_columns` to show in the Master AVWAP setups table:

* ``long_setups`` - `long_setups_store.read_long_setups()` (the scan's Long leaders);
* ``sp4`` - the S12 family evidence (`ai_jobs.family_side_evidence.read_payload`);
* ``path`` - S15 swing path facts, per ``SIDE|family``: the median 10-session MFE / MAE in
  ATR over the measured rows (`SWING_PATH_FACTS_FILE`);
* ``exits`` - the S13 exit models per ``SIDE|family``, only when the Setup Tracker's read
  is already cached in this process (`working_lately_service.cached_exit_model_review`) - the
  table never pays for the 800 MB features read itself;
* ``study`` - per ``SYMBOL|SIDE`` on the table's scan date, the S14 study families and the
  strength shadow from the session-horizon index the Working-lately build already holds
  (`working_lately_service.cached_horizon_index`), same rule: never built here;
* ``sources`` - ``{SYMBOL: [universe source]}`` for `momentum_scanner` / `journal_traded`.

Every piece that cannot be read is empty (unknown), never an error. Nothing is written.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from statistics import median
from typing import Any, Mapping

#: `{name: (file key, value)}` - one parse per file change, like the Working-lately cache.
_CACHE: dict[str, tuple[Any, Any]] = {}
PATH_HORIZON = 10


def _file_key(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _cached(name: str, path: Path, build) -> Any:
    key = _file_key(path)
    hit = _CACHE.get(name)
    if hit is not None and hit[0] == key:
        return hit[1]
    value = build() if key is not None else None
    _CACHE[name] = (key, value)
    return value


def input_paths() -> dict[str, Path]:
    import project_paths

    return {
        "long_setups": Path(project_paths.LONG_SETUPS_FILE),
        "sp4": Path(project_paths.FAMILY_SIDE_EVIDENCE_FILE),
        "path": Path(project_paths.SWING_PATH_FACTS_FILE),
        "momentum": Path(project_paths.MOMENTUM_UNIVERSE_MEMBERSHIP_FILE),
        "journal": Path(project_paths.JOURNAL_DB_FILE),
    }


def signature(data_date: str = "") -> tuple:
    """What a new read would depend on: the input stamps, the scan date and the cached
    tracker reads' identities. `stat` calls and dict lookups only - safe on the Qt thread."""
    from ui.services import working_lately_service as wl

    stamps = tuple(_file_key(path) for path in input_paths().values())
    return (stamps, str(data_date or "")[:10], id(wl.cached_exit_model_review()), id(wl.cached_horizon_index()))


def path_medians(rows) -> dict[str, dict[str, Any]]:
    """``{SIDE|family: {mfe_atr, mae_atr, n}}`` over measured `PATH_HORIZON`-session rows."""
    from ui.models.swing_columns import side_family_key

    groups: dict[str, tuple[list[float], list[float]]] = {}
    for row in rows:
        if str(row.get("horizon_sessions") or "").strip() not in {str(PATH_HORIZON), f"{PATH_HORIZON}.0"}:
            continue
        if str(row.get("measured") or "").strip().lower() not in {"true", "1", "yes"}:
            continue
        try:
            mfe, mae = float(row.get("mfe_atr")), float(row.get("mae_atr"))
        except (TypeError, ValueError):
            continue
        if mfe != mfe or mae != mae:
            continue
        pair = groups.setdefault(side_family_key(row.get("side"), row.get("setup_family")), ([], []))
        pair[0].append(mfe)
        pair[1].append(mae)
    return {key: {"mfe_atr": round(median(mfes), 3), "mae_atr": round(median(maes), 3), "n": len(mfes)}
            for key, (mfes, maes) in groups.items()}


def _read_path(path: Path) -> dict[str, dict[str, Any]]:
    def build():
        wanted = ("side", "setup_family", "horizon_sessions", "measured", "mfe_atr", "mae_atr")
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return path_medians({name: row.get(name) for name in wanted} for row in csv.DictReader(handle))

    return _cached("path", path, build) or {}


def _read_long_setups(path: Path) -> dict[str, Any] | None:
    import long_setups_store

    return long_setups_store.read_long_setups(path)


def _read_sp4(path: Path) -> dict[str, Any]:
    from ai_jobs.family_side_evidence import read_payload

    return read_payload(path)


def _exit_cells() -> dict[str, dict[str, Any]]:
    from ui.models.swing_columns import side_family_key
    from ui.services import working_lately_service as wl

    review = wl.cached_exit_model_review() or {}
    return {side_family_key(cell.get("side"), cell.get("family")): dict(cell)
            for cell in review.get("cells") or () if isinstance(cell, Mapping)}


def _study_rows(data_date: str) -> dict[str, dict[str, Any]]:
    from ui.services import working_lately_service as wl

    day = str(data_date or "")[:10]
    index = wl.cached_horizon_index() or {}
    if not day:
        return {}
    out = {}
    for (symbol, side, scan_date), row in list(index.items()):
        if scan_date != day:
            continue
        study = str(row.get("study_families") or "").strip()
        strength = str(row.get("strength_filter") or "").strip()
        if study or strength:
            out[f"{symbol}|{side}"] = {"study_families": study, "strength_filter": strength}
    return out


def _sources(paths: Mapping[str, Path]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    try:
        import momentum_universe

        members = momentum_universe.active_members(momentum_universe.load_membership(paths["momentum"]))
    except Exception:  # noqa: BLE001 - an unreadable membership is no badge
        logging.debug("momentum membership not read", exc_info=True)
        members = []
    for symbol in members:
        out.setdefault(str(symbol).upper(), []).append("momentum_scanner")
    try:
        from universe_builder import JOURNAL_TRADED_SOURCE, journal_traded_symbols

        traded = journal_traded_symbols(db_path=paths["journal"])
    except Exception:  # noqa: BLE001 - an unreadable journal is no badge
        logging.debug("journal traded names not read", exc_info=True)
        traded, JOURNAL_TRADED_SOURCE = {}, "journal_traded"
    for symbol in traded:
        out.setdefault(str(symbol).upper(), []).append(JOURNAL_TRADED_SOURCE)
    return out


def read_swing_context(data_date: str = "") -> dict[str, Any]:
    """The whole payload. Each part fails alone to its empty value."""
    paths = input_paths()
    parts = {
        "long_setups": lambda: _read_long_setups(paths["long_setups"]),
        "sp4": lambda: _read_sp4(paths["sp4"]),
        "path": lambda: _read_path(paths["path"]),
        "exits": _exit_cells,
        "study": lambda: _study_rows(data_date),
        "sources": lambda: _sources(paths),
    }
    out: dict[str, Any] = {"data_date": str(data_date or "")[:10]}
    for name, read in parts.items():
        try:
            out[name] = read()
        except Exception:  # noqa: BLE001 - one part never costs the table
            logging.warning("swing context part %s not read", name, exc_info=True)
            out[name] = None
        if out[name] is None and name != "long_setups":
            out[name] = {}
    return out
