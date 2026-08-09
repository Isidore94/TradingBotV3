"""AI store location and enablement (plan sec 3.3 / 6.1).

The AI store is its own storage class: nightly bulk writes that must not churn
the Drive-synced home folder, and must not be mixed into the DAS research lake
either -- separate trees mean an AI-job bug can never corrupt lake data or an
operational report. A configured path inside the shared home folder is
therefore refused, never silently accepted.

Resolution order mirrors ``research_warehouse/config.py`` and the shared-store
precedent in ``project_paths``: the ``TRADINGBOTV3_AI_STORE_DIR`` environment
override first, then the ``ai_store_dir`` key in ``local_settings.json``. Unset
means the whole batch layer is disabled and every entry point no-ops via
:func:`ai_store_enabled`.

The store is expected to live on a network share, which is a different failure
model from a local disk: it can be asleep, unreachable, or slow (a measured
19.8 s first write while the NAS spins up, then ~40 ms per fsync'd append).
So availability is checked explicitly and reported, never assumed -- a job that
cannot reach the store degrades to "no digest tonight" rather than failing
halfway through a write.
"""

from __future__ import annotations

import os
from pathlib import Path

AI_STORE_DIR_ENV = "TRADINGBOTV3_AI_STORE_DIR"
AI_STORE_DIR_SETTING = "ai_store_dir"

#: Directory contract from the plan (sec 3.3).
AI_STORE_SUBDIRS = ("digests", "briefs", "retros", "models", "logs")

#: Some operations touch a sleeping NAS. Anything under this is normal.
SLOW_SHARE_SECONDS = 30.0


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
            f"AI store {path} is inside the shared home folder {shared_home}. "
            "The AI store is a separate storage class (plan sec 3.3): nightly "
            "bulk writes must not churn the synced operational folder. Point "
            f"{AI_STORE_DIR_SETTING} (or {AI_STORE_DIR_ENV}) at a file server "
            "or local disk instead."
        )


def _refuse_research_lake(path: Path) -> None:
    """The lake and the AI store are separate writers; keep the trees apart."""
    try:
        from research_warehouse.config import get_research_store_dir
    except Exception:
        return
    try:
        lake = get_research_store_dir()
    except ValueError:
        return
    if lake is None:
        return
    try:
        inside = path.expanduser().resolve().is_relative_to(Path(lake).resolve())
    except OSError:
        return
    if inside:
        raise ValueError(
            f"AI store {path} is inside the research lake {lake}. They are "
            "separate storage classes with separate writer components (plan "
            "sec 3.3); keeping the trees apart is what stops an AI-job bug "
            "from corrupting lake data."
        )


def get_ai_store_dir() -> Path | None:
    """The configured AI store root, or None when the batch layer is disabled.

    Raises ValueError for a path inside the shared home folder or the research
    lake: that is a misconfiguration to surface, not a location to fall back
    from.
    """
    raw = str(os.environ.get(AI_STORE_DIR_ENV) or "").strip()
    if not raw:
        value = _paths().get_local_setting(AI_STORE_DIR_SETTING)
        raw = value.strip() if isinstance(value, str) else ""
    if not raw:
        return None
    path = Path(raw).expanduser()
    _refuse_shared_home(path)
    _refuse_research_lake(path)
    return path


def ai_store_enabled() -> bool:
    """False when unset OR misconfigured - callers no-op either way."""
    try:
        return get_ai_store_dir() is not None
    except ValueError:
        return False


def save_ai_store_dir(path: str) -> Path:
    target = Path(path).expanduser()
    _refuse_shared_home(target)
    _refuse_research_lake(target)
    _paths().save_local_setting(AI_STORE_DIR_SETTING, str(target))
    return target


def clear_ai_store_dir() -> None:
    # A null value reads back as unset; avoids reaching into the settings file.
    if _paths().get_local_setting(AI_STORE_DIR_SETTING) is not None:
        _paths().save_local_setting(AI_STORE_DIR_SETTING, None)


def ensure_ai_store_layout(root: Path | None = None) -> Path:
    """Create the sec-3.3 directory skeleton; idempotent, additive only."""
    store = root if root is not None else get_ai_store_dir()
    if store is None:
        raise ValueError("No AI store configured; nothing to initialize.")
    _refuse_shared_home(store)
    _refuse_research_lake(store)
    for name in AI_STORE_SUBDIRS:
        (store / name).mkdir(parents=True, exist_ok=True)
    return store


def store_available(root: Path | None = None, *, read_only: bool = False) -> tuple[bool, str]:
    """Can the store actually be written right now?

    Returns ``(ok, reason)``. A network share that is asleep, offline or
    credential-blocked fails here, before any job starts writing -- which is
    the difference between "no digest tonight" and a half-written artifact.

    ``read_only=True`` reports what can be observed without touching the store:
    no directory creation, no write probe. ``run_ai_jobs.py --status`` uses it.
    A status command that says "print state, run nothing" must not mkdir a
    five-directory skeleton on a NAS and write a probe file -- during market
    hours that is a write the hard rule never authorised, and on a sleeping
    share it is a ~20 s spin-up for a read (checkpoint review 2026-08-08
    second review). The trade-off is stated honestly in the reason: read-only
    availability cannot prove writability, so it never claims to.
    """
    try:
        store = root if root is not None else get_ai_store_dir()
    except ValueError as exc:
        return False, str(exc)
    if store is None:
        return False, f"{AI_STORE_DIR_SETTING} is unset; the AI batch layer is disabled"
    if read_only:
        try:
            reachable = store.is_dir()
        except OSError as exc:
            return False, f"AI store {store} is unreachable: {exc}"
        if not reachable:
            return False, (
                f"AI store {store} does not exist yet; a job run will create it"
                if not store.exists()
                else f"AI store {store} is not a directory"
            )
        return True, f"AI store present at {store} (not write-probed: status is read-only)"
    try:
        ensure_ai_store_layout(store)
    except OSError as exc:
        return False, f"AI store {store} is unreachable or read-only: {exc}"
    probe = store / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"AI store {store} is not writable: {exc}"
    return True, f"AI store ready at {store}"


def _subdir(name: str, *, create: bool) -> Path:
    """One store subdirectory.

    ``create=False`` resolves the path without creating anything, for readers
    that must not write (``run_ai_jobs.py --status``); it raises the same
    ValueError when no store is configured, because "where would it be?" has no
    answer either way.
    """
    if create:
        return ensure_ai_store_layout() / name
    store = get_ai_store_dir()
    if store is None:
        raise ValueError("No AI store configured; nothing to initialize.")
    return store / name


def digests_dir(*, create: bool = True) -> Path:
    return _subdir("digests", create=create)


def briefs_dir(*, create: bool = True) -> Path:
    return _subdir("briefs", create=create)


def retros_dir(*, create: bool = True) -> Path:
    return _subdir("retros", create=create)


def store_logs_dir(*, create: bool = True) -> Path:
    return _subdir("logs", create=create)


def get_ai_store_details() -> dict[str, str]:
    """Settings/Health surface payload, shaped like get_research_store_details."""
    env_value = str(os.environ.get(AI_STORE_DIR_ENV) or "").strip()
    try:
        resolved = get_ai_store_dir()
        error = ""
    except ValueError as exc:
        resolved = None
        error = str(exc)
    return {
        "ai_store_dir": str(resolved) if resolved else "",
        "enabled": "yes" if resolved is not None else "no",
        "source": "environment" if env_value else ("local_config" if resolved else "unset"),
        "error": error,
    }
