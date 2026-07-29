#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
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
        log_dir = Path(os.environ.get("LOCALAPPDATA", str(ROOT_DIR))) / "TradingBotV3" / "logs"
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


def main() -> None:
    _enable_crash_log()
    target = SCRIPTS_DIR / "gui.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
