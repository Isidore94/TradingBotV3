"""Research-lake location and enablement (plan Phase 0).

The lake is a new append-only storage class (decision 0014): very large
immutable Parquet files on a trader-owned local/DAS disk. It is deliberately
separate from the operational home folder - that folder is for compact state and
is mirrored wholesale to the DAS by the hourly cold push - so a configured path
inside the shared home folder is refused, never silently accepted. (Until
decision 0015 the home folder was Drive-synced and quota was the reason; the
refusal outlived the sync client. See BD-72.)

Resolution order mirrors the shared-store precedent in ``project_paths``:
``TRADINGBOTV3_RESEARCH_DIR`` environment override first, then the
``research_store_dir`` key in ``local_settings.json``. Unset means the entire
warehouse is disabled: every entry point no-ops via :func:`warehouse_enabled`.
"""

from __future__ import annotations

import os
from pathlib import Path

RESEARCH_DIR_ENV = "TRADINGBOTV3_RESEARCH_DIR"
RESEARCH_DIR_SETTING = "research_store_dir"

# Backup targets (sec 8.5 / LD-11). Class A is the irreplaceable-small set,
# mirrored to the home folder AND a backup disk; Class B is the append-only
# lake copy on a SECOND PHYSICAL DISK, never alongside the lake itself. Both are
# unset by default and the build job simply says so rather than guessing a
# destination: a backup written somewhere nobody chose is not a backup.
BACKUP_CLASS_A_ENV = "TRADINGBOTV3_RESEARCH_BACKUP_A"
BACKUP_CLASS_A_SETTING = "research_backup_class_a_dirs"
BACKUP_CLASS_B_ENV = "TRADINGBOTV3_RESEARCH_BACKUP_B"
BACKUP_CLASS_B_SETTING = "research_backup_class_b_dir"

# Directory contract from the plan (sec 8.2). ``manifest_log.jsonl`` is the
# append-only read authority; Phase 1 owns its semantics - here it is only
# created empty so a freshly pointed lake is recognizably a lake.
LAKE_SUBDIRS = (
    "_incoming",
    "_quarantine",
    "_retired",
    "bronze",
    "silver",
    "gold",
    "definitions",
)
LAKE_LEDGERS = ("manifest_log.jsonl", "imported_bundles.jsonl")
SPOOL_DIR_NAME = "research_spool"


def _paths():
    try:
        from scripts import project_paths
    except ImportError:  # scripts/ itself on sys.path (writer_role precedent)
        import project_paths
    return project_paths


def _refuse_shared_home(path: Path) -> None:
    shared_home = Path(_paths().SHARED_HOME_DIR)
    try:
        inside = path.expanduser().resolve().is_relative_to(shared_home.resolve())
    except OSError:
        inside = False
    if inside:
        raise ValueError(
            f"Research store {path} is inside the shared home folder {shared_home}. "
            "The lake is a separate storage class (decision 0014) and must never "
            "live in the operational home folder - point research_store_dir (or "
            f"{RESEARCH_DIR_ENV}) at a local/DAS disk instead."
        )


def get_research_store_dir() -> Path | None:
    """The configured lake root, or None when the warehouse is disabled.

    Raises ValueError for a path inside the shared home folder: that is a
    misconfiguration to surface, not a location to fall back from.
    """
    raw = str(os.environ.get(RESEARCH_DIR_ENV) or "").strip()
    if not raw:
        value = _paths().get_local_setting(RESEARCH_DIR_SETTING)
        raw = value.strip() if isinstance(value, str) else ""
    if not raw:
        return None
    path = Path(raw).expanduser()
    _refuse_shared_home(path)
    return path


def warehouse_enabled() -> bool:
    """False when unset OR misconfigured - callers no-op either way."""
    try:
        return get_research_store_dir() is not None
    except ValueError:
        return False


def research_spool_dir() -> Path:
    """Machine-local write spool (plan sec 8.4); never on the DAS or in the
    home folder - it must survive a file-server outage."""
    return Path(_paths().LOCAL_SETTINGS_DIR) / SPOOL_DIR_NAME


def backup_class_a_dirs() -> list[Path]:
    """Class-A backup destinations; empty means the build job skips the step.

    Accepts a list in ``local_settings.json`` or an ``os.pathsep``-separated
    string in the environment override.
    """
    raw = str(os.environ.get(BACKUP_CLASS_A_ENV) or "").strip()
    if raw:
        values = [part for part in raw.split(os.pathsep) if part.strip()]
    else:
        setting = _paths().get_local_setting(BACKUP_CLASS_A_SETTING)
        if isinstance(setting, str):
            values = [setting] if setting.strip() else []
        elif isinstance(setting, (list, tuple)):
            values = [str(item) for item in setting if str(item).strip()]
        else:
            values = []
    return [Path(value).expanduser() for value in values]


def backup_class_b_dir() -> Path | None:
    """Class-B lake copy destination, or None when unset."""
    raw = str(os.environ.get(BACKUP_CLASS_B_ENV) or "").strip()
    if not raw:
        value = _paths().get_local_setting(BACKUP_CLASS_B_SETTING)
        raw = value.strip() if isinstance(value, str) else ""
    return Path(raw).expanduser() if raw else None


def save_research_store_dir(path: str) -> Path:
    target = Path(path).expanduser()
    _refuse_shared_home(target)
    _paths().save_local_setting(RESEARCH_DIR_SETTING, str(target))
    return target


def clear_research_store_dir() -> None:
    # A null value reads back as unset; avoids reaching into the settings file.
    if _paths().get_local_setting(RESEARCH_DIR_SETTING) is not None:
        _paths().save_local_setting(RESEARCH_DIR_SETTING, None)


def ensure_lake_layout(root: Path | None = None) -> Path:
    """Create the sec-8.2 directory skeleton; idempotent, additive only."""
    lake = root if root is not None else get_research_store_dir()
    if lake is None:
        raise ValueError("No research store configured; nothing to initialize.")
    _refuse_shared_home(lake)
    for name in LAKE_SUBDIRS:
        (lake / name).mkdir(parents=True, exist_ok=True)
    for ledger in LAKE_LEDGERS:
        ledger_path = lake / ledger
        if not ledger_path.exists():
            ledger_path.touch()
    return lake


def get_research_store_details() -> dict[str, str]:
    """Settings/Health surface payload, shaped like get_tracker_storage_details."""
    env_value = str(os.environ.get(RESEARCH_DIR_ENV) or "").strip()
    try:
        resolved = get_research_store_dir()
        error = ""
    except ValueError as exc:
        resolved = None
        error = str(exc)
    return {
        "research_store_dir": str(resolved) if resolved else "",
        "enabled": "yes" if resolved is not None else "no",
        "source": "environment" if env_value else ("local_config" if resolved else "unset"),
        "spool_dir": str(research_spool_dir()),
        "error": error,
    }
