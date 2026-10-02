"""The `history_pack` night slot (trader 2026-10-02): deterministic, no model.

Every night: pack the D1 feature history into its lossless monthly Parquet
archive (`d1_feature_history_archive.archive`, which verifies cell for cell
before it commits), verify the whole archive, and write ONE small JSON report
(`HISTORY_PACK_REPORT_FILE`): what was packed, the verify result, and for each
large live store its size now and at the previous report, by `stat` only.

TRIM STAYS OFF. Seven readers still read `d1_features_history.csv` directly,
so trimming it today would silently shorten their history. The trim runs only
when the local setting `TRIM_SETTING` is true, which a later packet switches
on after those readers move to `read_history`.

A failure returns `failed` and leaves the last good report where it was.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

#: Local setting that lets the night trim the live CSV. Default False.
TRIM_SETTING = "d1_history_trim_enabled"
REPORT_SCHEMA = 1


def trim_enabled() -> bool:
    import project_paths as pp

    value = pp.get_local_setting(TRIM_SETTING, False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return value is True


def default_stores() -> dict[str, Path]:
    """The large live stores whose size the report tracks (name -> path)."""
    import project_paths as pp
    from evidence_ledger import default_ledger_dir

    tracker = Path(pp.MASTER_AVWAP_SETUP_TRACKER_FILE)
    return {
        "d1_features_history.csv": Path(pp.D1_FEATURES_HISTORY_FILE),
        "master_avwap_setup_tracker.json": tracker,
        "master_avwap_setup_tracker.json.bak": tracker.with_name(tracker.name + ".bak"),
        "master_avwap_setup_tracker.sqlite": Path(pp.MASTER_AVWAP_SETUP_TRACKER_DB),
        "intraday_bounce_outcomes.csv": Path(pp.INTRADAY_BOUNCE_OUTCOMES_FILE),
        "intraday_bounce_candidates.csv": Path(pp.INTRADAY_BOUNCE_CANDIDATES_FILE),
        "evidence_ledgers": Path(default_ledger_dir()),
    }


def _size(path: Path) -> tuple[bool, int]:
    """Bytes by stat only; a folder is the sum of the files under it."""
    try:
        if path.is_dir():
            total = 0
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        total += os.stat(os.path.join(root, name)).st_size
                    except OSError:
                        continue
            return True, total
        return True, path.stat().st_size
    except FileNotFoundError:
        return False, 0


def _previous_sizes(report_path: Path) -> dict[str, int]:
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return {str(row["name"]): int(row["bytes"]) for row in data.get("stores", [])}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _store_rows(stores: dict[str, Path], previous: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for name, path in stores.items():
        exists, size = _size(Path(path))
        before = previous.get(name)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": exists,
                "bytes": size,
                "previous_bytes": before,
                "delta_bytes": None if before is None else size - before,
            }
        )
    return rows


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run_history_pack(
    *,
    session_date: str = "",
    csv_path: Any = None,
    archive_dir: Any = None,
    report_path: Any = None,
    stores: dict[str, Path] | None = None,
    today: date | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    import d1_feature_history_archive as arc
    import project_paths as pp

    csv_path = Path(csv_path or pp.D1_FEATURES_HISTORY_FILE)
    archive_dir = Path(archive_dir or pp.D1_FEATURES_HISTORY_ARCHIVE_DIR)
    report_path = Path(report_path or pp.HISTORY_PACK_REPORT_FILE)
    failed = {"status": "failed", "model": "", "outputs": []}
    try:
        packed = arc.archive(csv_path=csv_path, archive_dir=archive_dir)
        checked = arc.verify(csv_path=csv_path, archive_dir=archive_dir)
        if not checked["ok"]:
            return {**failed, "reason": "archive verify failed: " + "; ".join(checked["problems"])
                    + "; last report kept"}
        trim: dict[str, Any] = {"enabled": trim_enabled()}
        if trim["enabled"]:
            trim.update(arc.trim(csv_path=csv_path, archive_dir=archive_dir, today=today, apply=True))
    except arc.ArchiveError as exc:
        _log.error("history_pack: %s", exc)
        return {**failed, "reason": f"history not packed ({type(exc).__name__}: {exc}); last report kept"}
    except Exception as exc:  # noqa: BLE001 - the night goes on; the last report stays
        _log.exception("history_pack: failed")
        return {**failed, "reason": f"history pack failed ({type(exc).__name__}: {exc}); last report kept"}

    tracked = dict(stores if stores is not None else default_stores())
    tracked.setdefault("d1_features_history_archive", archive_dir)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_date": session_date,
        "archive": packed,
        "verify": {k: checked.get(k) for k in ("ok", "problems", "archived_rows", "files",
                                                "live_rows", "unarchived_rows")},
        "trim": trim,
        "stores": _store_rows(tracked, _previous_sizes(report_path)),
    }
    try:
        _write_report(report_path, report)
    except OSError as exc:
        return {**failed, "reason": f"history packed but the report was not written ({exc})"}
    reason = (
        f"packed {packed.get('archived_rows', 0)} rows ({packed.get('total_archived', 0)} archived), "
        f"verify ok; trim {'removed ' + str(trim.get('removed', 0)) + ' rows' if trim['enabled'] else 'off'}"
    )
    return {"status": "ok", "model": "", "reason": reason, "outputs": [str(report_path)]}
