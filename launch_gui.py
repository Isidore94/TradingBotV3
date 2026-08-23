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
    argv = sys.argv[1:]
    # --selftest before anything else, including the crash log: it must be
    # runnable against a frozen bundle on a machine with no display and no
    # network, and it must not touch the desk's own log files on the way.
    # This is what replaces the trader's post-build click-through - see
    # scripts/selftest.py and packaging/README.md.
    if "--selftest" in argv:
        from selftest import run_selftest

        return run_selftest(verbose="--verbose" in argv or "-v" in argv)

    # --run-scan is how a FROZEN desk spawns its scan child. sys.executable is
    # TradingBotV3.exe there, so the source form (`sys.executable -c "<code>"`)
    # reaches the app's own argument parser and exits 2 without scanning - which
    # is precisely what silently killed every scheduled swing scan on the desk
    # from 2026-08-12. Handled here, before argparse and before the crash log,
    # for the same reason --selftest is: this process is not the desk.
    if "--run-scan" in argv:
        from scan_worker import run as run_scan_worker

        index = argv.index("--run-scan")
        payload = argv[index + 1] if index + 1 < len(argv) else None
        return run_scan_worker(payload)

    _enable_crash_log()
    # R10.A / Sol blocker 3: one desk per machine. `launch_gui_auto.ps1` already
    # refuses a second one, but only on that path - a double-click, a shortcut,
    # a terminal, a second scheduled task and the frozen exe all arrive here
    # instead. R10.0 measured concurrent desks: pid 31848 overlapped three
    # others on 2026-08-20, the worst by 3.8 hours.
    #
    # Defence in depth, not the transaction: the outcome finalizer fences itself
    # and stays correct with two desks running, because this guard can be
    # overridden and a second process can always be started another way.
    from single_instance import OVERRIDE_FLAG, AnotherDeskIsRunning, desk_slot

    # This is the Qt desk's real entrypoint, not a hop through scripts/gui.py.
    # The latter remains only for legacy ``--ui tk`` compatibility.
    from ui import app

    try:
        with desk_slot(allow_second=OVERRIDE_FLAG in argv) as protection:
            print(f"TradingBotV3 desk: {protection}")
            return int(app.main([arg for arg in argv if arg != OVERRIDE_FLAG]) or 0)
    except AnotherDeskIsRunning as exc:
        # Exit 0, like the PowerShell launcher: "already running" is a normal
        # outcome of double-clicking twice, not a failure to report.
        print(str(exc))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
