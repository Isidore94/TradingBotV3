"""The `history_pack` night slot (trader 2026-10-02): deterministic, no model.

Every night, for every store registered in
`d1_feature_history_archive.registered_stores()` (the D1 feature history and
the bounce outcomes CSV today): pack it into its lossless monthly Parquet
archive (`archive`, which verifies cell for cell before it commits) and verify
the whole archive. Then write ONE small JSON report (`HISTORY_PACK_REPORT_FILE`):
one result block per store, and for each large live store its size now and at
the previous report, by `stat` only.

TRIM STAYS OFF. Readers still read the live CSVs directly, so trimming would
silently shorten their history. A store is trimmed only when its own local
setting (`StoreSpec.trim_setting`; the D1 one is `TRIM_SETTING`) is true AND
its writer takes a lock; a store whose writer takes no lock (the bounce
outcomes) is never trimmed, whatever any setting says.

One store failing never stops another from packing. Any failure returns
`failed` and leaves the last good report where it was.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

#: Local setting that lets the night trim the D1 history CSV. Default False.
TRIM_SETTING = "d1_history_trim_enabled"
REPORT_SCHEMA = 2


def trim_enabled(setting: str = TRIM_SETTING) -> bool:
    import project_paths as pp

    if not setting:
        return False
    value = pp.get_local_setting(setting, False)
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


def _pack_one(spec, *, today: date | None = None) -> dict[str, Any]:
    """Archive + verify + (only when allowed) trim one store. Raises ArchiveError."""
    import d1_feature_history_archive as arc

    packed = arc.archive(store=spec)
    checked = arc.verify(store=spec)
    if not checked["ok"]:
        raise arc.VerifyFailed("archive verify failed: " + "; ".join(checked["problems"]))
    if spec.writer_lock_key is None:
        trim: dict[str, Any] = {
            "enabled": False,
            "reason": "its writer takes no lock; trim is refused for this store whatever the setting",
        }
    elif not trim_enabled(spec.trim_setting):
        trim = {"enabled": False, "reason": f"local setting {spec.trim_setting or '(none)'} is off"}
    else:
        trim = {"enabled": True}
        trim.update(arc.trim(store=spec, today=today, apply=True))
    return {
        "archive": packed,
        "verify": {k: checked.get(k) for k in ("ok", "problems", "archived_rows", "files",
                                                "live_rows", "unarchived_rows")},
        "trim": trim,
    }


def run_history_pack(
    *,
    session_date: str = "",
    csv_path: Any = None,
    archive_dir: Any = None,
    report_path: Any = None,
    stores: dict[str, Path] | None = None,
    specs: list | None = None,
    today: date | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    import d1_feature_history_archive as arc
    import project_paths as pp

    report_path = Path(report_path or pp.HISTORY_PACK_REPORT_FILE)
    if specs is None:
        if csv_path is not None or archive_dir is not None:
            # An explicit D1 path (a manual run, a test) packs that store only.
            specs = [arc.registered_stores()[arc.D1_STORE].with_paths(csv_path, archive_dir)]
        else:
            specs = list(arc.registered_stores().values())

    blocks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for spec in specs:
        try:
            blocks[spec.name] = _pack_one(spec, today=today)
        except arc.ArchiveError as exc:
            _log.error("history_pack %s: %s", spec.name, exc)
            failures.append(f"{spec.name}: history not packed ({type(exc).__name__}: {exc})")
        except Exception as exc:  # noqa: BLE001 - the night goes on; the last report stays
            _log.exception("history_pack %s: failed", spec.name)
            failures.append(f"{spec.name}: history pack failed ({type(exc).__name__}: {exc})")
    if failures:
        return {"status": "failed", "model": "", "outputs": [],
                "reason": "; ".join(failures) + "; last report kept"}

    tracked = dict(stores if stores is not None else default_stores())
    for spec in specs:
        tracked.setdefault(f"{spec.name}_archive", Path(spec.archive_dir))
    first = blocks[specs[0].name] if specs else {"archive": {}, "verify": {}, "trim": {"enabled": False}}
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_date": session_date,
        "stores_packed": blocks,
        # The first store's block again at the top level (the D1 history on the
        # night slate), where the first version of this report put it.
        "archive": first["archive"],
        "verify": first["verify"],
        "trim": first["trim"],
        "stores": _store_rows(tracked, _previous_sizes(report_path)),
    }
    try:
        _write_report(report_path, report)
    except OSError as exc:
        return {"status": "failed", "model": "", "outputs": [],
                "reason": f"history packed but the report was not written ({exc})"}
    parts = []
    for name, block in blocks.items():
        trim = block["trim"]
        trimmed = f"trim removed {trim.get('removed', 0)}" if trim.get("enabled") else "trim off"
        parts.append(f"{name}: packed {block['archive'].get('archived_rows', 0)} rows "
                     f"({block['archive'].get('total_archived', 0)} archived), verify ok, {trimmed}")
    return {"status": "ok", "model": "", "reason": "; ".join(parts), "outputs": [str(report_path)]}
