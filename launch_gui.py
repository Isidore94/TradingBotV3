#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
# Frozen builds must NOT do this. ROOT_DIR is sys._MEIPASS there, and
# PyInstaller's importer claims any path under _MEIPASS — including a
# subdirectory that does not exist. Inserting <_MEIPASS>/scripts makes it
# resolve the first-party packages to that phantom location, so `bounce_bot_lib`
# imports with __path__ pointing at <_MEIPASS>/scripts/bounce_bot_lib and every
# submodule (learning, legacy, tier_flip, ...) then fails to import. The bundle
# already exposes scripts/ contents as top-level modules, so the path is only
# ever needed when running from a source checkout.
if not getattr(sys, "frozen", False) and str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# faulthandler keeps a borrowed reference to this handle; a module global
# stops the file (and its fd) from being garbage-collected out from under it.
_CRASH_LOG_HANDLE = None


def _enable_crash_log() -> None:
    """Dump every thread's stack to a persistent log on a native crash.

    The GUI has died repeatedly since 2026-07-16 with access violations
    inside python314.dll (Windows Event Log ID 1000, fault offset 0xc06b7) -
    below the interpreter, so app logging never sees a traceback.
    faulthandler writes each thread's Python stack at fault time, so the
    next crash names the exact code path instead of just a DLL offset.
    """
    global _CRASH_LOG_HANDLE
    import faulthandler
    import os
    from datetime import datetime

    try:
        from project_paths import LOCAL_LOG_DIR

        log_dir = Path(LOCAL_LOG_DIR)
    except Exception:
        # project_paths can legitimately fail here (e.g. shared drive not
        # mounted raises at import). Mirror its per-platform local dir without
        # importing it, so the crash log never lands inside the checkout.
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            log_dir = Path(local_appdata) / "TradingBotV3" / "logs"
        elif sys.platform == "darwin":
            log_dir = Path.home() / "Library" / "Application Support" / "TradingBotV3" / "logs"
        else:
            log_dir = Path.home() / ".local" / "share" / "TradingBotV3" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _CRASH_LOG_HANDLE = open(
            log_dir / "gui_crash.log", "a", buffering=1, encoding="utf-8"
        )
        _CRASH_LOG_HANDLE.write(
            f"\n=== GUI start {datetime.now():%Y-%m-%d %H:%M:%S} pid={os.getpid()} ===\n"
        )
        faulthandler.enable(file=_CRASH_LOG_HANDLE, all_threads=True)
    except Exception:
        # Crash logging must never keep the desk from launching.
        faulthandler.enable()


def main() -> int:
    _enable_crash_log()
    # This is the Qt desk's real entrypoint, not a hop through scripts/gui.py.
    # The latter remains only for legacy ``--ui tk`` compatibility.
    from ui import app

    return int(app.main(sys.argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
