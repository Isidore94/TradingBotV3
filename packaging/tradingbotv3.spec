# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the PySide6 Trading Desk.

Build from the repo root:

    .\\.venv\\Scripts\\pyinstaller.exe .\\packaging\\tradingbotv3.spec --noconfirm

Produces ``dist/TradingBotV3/TradingBotV3.exe`` (onedir). Onedir rather than
onefile on purpose: the bundle is ~500MB, and onefile would re-extract all of it
into a temp dir on every launch, adding many seconds to startup for no benefit.

Runtime data is NOT bundled. The app keeps reading its machine-local store
(%LOCALAPPDATA%\\TradingBotV3) and the shared store named by ``shared_data_dir``,
exactly as the source checkout does, so the exe and `python launch_gui.py` share
one set of data.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# The runtime hook pins the launched exe. The build process needs the same pin
# before hook-qtpy runs, or a stray developer PyQt5 install makes qtpy collect
# the wrong binding even though Analysis excludes it later.
os.environ["QT_API"] = "pyside6"

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent
SCRIPTS = ROOT / "scripts"

# collect_submodules() imports the package to walk it, so the first-party roots
# have to be importable by the interpreter running this spec — pathex alone is
# not enough, it only affects the analysis. Without this, every first-party
# collect_submodules call raises ModuleNotFoundError and the bundle silently
# ships without the lazily-imported engines.
for _root in (str(ROOT), str(SCRIPTS)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

# `launch_gui.py` puts scripts/ on sys.path and then does `from ui import app`,
# so the analysis needs the same two roots to resolve `ui`, `project_paths`,
# `bounce_bot_lib`, ... as top-level names alongside `market_prep`.
PATHEX = [str(ROOT), str(SCRIPTS)]

# The first-party trees the frozen desk can reach. tests/test_packaging_spec_drift.py
# reads this tuple back and fails when a package appears under scripts/ that is
# neither listed here nor allowlisted there with a reason, so growing the tree
# can no longer silently shrink the bundle.
#
# `desk_link` was here until 2026-08-24 (P1.5): the Desk Link/satellite role was
# retired on 2026-08-08 and its code is now gone from the tree, so the package
# it named no longer exists to collect.
FIRST_PARTY_PACKAGES = (
    "ui",
    "bounce_bot_lib",
    "master_avwap_lib",
    "market_prep",
    "diagnostics",
    "research_warehouse",
    # Phase 0.31: the desk's Trade Mentor now imports the bounded local-AI
    # request owner and grounded market-story reader at button/card time.
    "ai_jobs",
    # R5 (2026-08-17): `indicators` gained its first real importer when the LRSI
    # cross engine wired in - bounce_bot_lib.legacy -> m5_signal_engines ->
    # indicators.efficiency_lrsi. It was allowlisted as unreachable until then.
    "indicators",
    # R10.A (2026-08-22): `ops` must be IN the bundle, not allowlisted out.
    # `operations_audit._evidence_snapshot_check` does `from ops import
    # evidence_snapshot` lazily, and operations_audit renders the frozen exe's
    # System Health page - so a bundle without it dies at exactly the lazy
    # import this guard exists to catch. Its three .ps1 files ride along as
    # assets; they are Task Scheduler entry points rather than anything the exe
    # executes, and shipping them costs ~12 KB and keeps the tree mirrored.
    "ops",
)

# Most first-party packages are collected whole. ``ai_jobs`` is the exception:
# the source-only nightly runner contains optional analysis/UI helpers, while
# the frozen desk reaches only the bounded story reader used by Trade Mentor.
# Pulling the whole tree made qtpy inspect the stray PyQt5 install before the
# PySide6 hook and produced a mixed-Qt bundle. Keep this allowlist exact.
PARTIAL_PACKAGE_MODULES = {
    "ai_jobs": frozenset(("ai_jobs", "ai_jobs.market_story_narration")),
}


def _package_dir(name):
    """market_prep lives at the repo root; everything else under scripts/."""
    candidate = SCRIPTS / name
    return candidate if candidate.is_dir() else ROOT / name


datas = [
    # market_prep/config_loader.py resolves CONFIG_DIR as
    # Path(__file__).parents[1] / "config", which lands at the bundle root.
    (str(ROOT / "config"), "config"),
    # Settings > Testing Plan renders this markdown file at runtime. It lives
    # OUTSIDE scripts/, so the package-asset sweep below never sees it and the
    # spec-drift test (which only walks scripts/) cannot guard it either -
    # hence the explicit rule and the hard failure underneath.
    (str(ROOT / "docs" / "DESK_TESTING_PLAN.md"), "docs"),
]
_TESTING_PLAN = ROOT / "docs" / "DESK_TESTING_PLAN.md"
if not _TESTING_PLAN.is_file():
    raise SystemExit(
        "spec error: docs/DESK_TESTING_PLAN.md not found - Settings > Testing Plan "
        "would ship showing 'plan file not found' on the trader's desk"
    )

# PyInstaller bundles .py only. Modules reach their own assets through
# __file__-relative paths — ui/theme.py loads theme.qss with
# Path(__file__).with_name(), ui/annotations reads its veto vocabulary, and
# research_warehouse reads exploration_cohort.txt — so every non-Python file in
# a bundled package is mirrored at the same relative location. Doing this for
# every package rather than for ui alone is the point: an asset added to a new
# package is covered the day it lands, instead of going missing until the first
# frozen run that needs it.
_assets = []
for _package in FIRST_PARTY_PACKAGES:
    _pkg_dir = _package_dir(_package)
    for _asset in sorted(_pkg_dir.rglob("*")):
        if not _asset.is_file() or _asset.suffix.lower() in (".py", ".pyc"):
            continue
        if "__pycache__" in _asset.parts:
            continue
        _rel = _asset.parent.relative_to(_pkg_dir)
        datas.append((str(_asset), str(Path(_package) / _rel) if _rel.parts else _package))
        _assets.append(_asset)
# The same rule one level up: a non-.py asset sitting at the scripts/ ROOT, next
# to the top-level modules rather than inside a package. `setup_registry.py`
# reads its frozen JSON with a __file__-relative path, and a frozen top-level
# module's __file__ parent IS the bundle root - so these land at "." rather than
# under a package name. Swept rather than named one file at a time, for the same
# reason the package sweep above is: the next root-level asset is covered the day
# it lands instead of the first frozen run that needs it.
for _asset in sorted(SCRIPTS.glob("*")):
    if not _asset.is_file() or _asset.suffix.lower() in (".py", ".pyc"):
        continue
    datas.append((str(_asset), "."))
    _assets.append(_asset)

# The desk renders unstyled without the stylesheet and dies on the missing file,
# so treat its absence as a build failure rather than shipping a broken exe.
if not any(a.name == "theme.qss" for a in _assets):
    raise SystemExit("spec error: ui/theme.qss not found — the desk cannot start without it")
print(f"[spec] package assets bundled: {[a.name for a in _assets]}")
datas += collect_data_files("qtawesome")   # bundled icon fonts
datas += collect_data_files("pyqtgraph")
datas += collect_data_files("certifi")

binaries = []
hiddenimports = []
# Daily Recap reaches the measured report and validated next-test reader only
# on its worker after the trader opens Review. They are top-level modules, so
# package collection cannot see them; make a missing frozen import fail here.
hiddenimports += [
    "measured_report", "research_proposal", "ai_jobs.store", "ai_jobs.digest",
    "ai_jobs.measured_report_publish",
]
# TJ-1: the Day Review page reaches its per-session index and the forecast reader
# only on its worker (the index) and when the trader pastes a brief (the reader).
# Both are top-level modules, for the same reason and with the same consequence as
# the three above: package collection cannot see them, so a missing frozen import
# has to fail here rather than at the first open.
hiddenimports += ["day_review_index", "day_review_bars", "forecast_brief", "walkaway_day"]
# Econ morning brief (2026-09-24): the reminder service's worker imports the
# view builder, which imports the fixed econ parser. Same reason as above.
hiddenimports += ["econ_brief", "econ_events"]
# Day Recap coach: the night's facts slot imports the day record lazily, and the
# record imports the recap store lazily. Same reason as above.
hiddenimports += ["day_session_record", "recap_store", "recap_findability", "setup_grades_history", "opening_regime_history"]
# The UI loads panels/services by name in places, and the engines import each
# other lazily inside functions; collecting the first-party trees outright is
# far cheaper than chasing ModuleNotFoundError one launch at a time.
#
# Deliberately NOT wrapped in try/except: a package that fails to collect here
# is a bundle that starts and then dies at the first lazy import. Fail the
# build loudly instead.
for package in FIRST_PARTY_PACKAGES:
    allowed = PARTIAL_PACKAGE_MODULES.get(package)
    found = collect_submodules(
        package,
        filter=(lambda name, allowed=allowed: name in allowed) if allowed else (lambda _name: True),
    )
    if not found:
        raise SystemExit(f"spec error: collect_submodules({package!r}) found nothing — check sys.path above")
    print(f"[spec] {package}: {len(found)} submodules")
    hiddenimports += found

# DuckDB is the warehouse's optional read-only query engine (LD-04): every slice
# query is answerable through pyarrow, and research_warehouse.queries imports
# duckdb inside the two functions that need it, behind duckdb_available().
# It carries a compiled extension, so collect_all picks up the shared library
# that a bare hiddenimport would leave behind. Absence is not a build error —
# the frozen desk simply reports duckdb_available() False and uses pyarrow.
try:
    import duckdb  # noqa: F401
except ImportError:
    print("[spec] duckdb not installed — the bundle will answer slice queries through pyarrow (LD-04)")
else:
    _dd_datas, _dd_binaries, _dd_hidden = collect_all("duckdb")
    datas += _dd_datas
    binaries += _dd_binaries
    hiddenimports += _dd_hidden
    print(f"[spec] duckdb: {len(_dd_binaries)} binaries, {len(_dd_hidden)} submodules")
# scikit-learn/scipy were force-collected here for a trade-quality model that was
# removed in a73f072. Nothing in the tree has imported sklearn or scipy since, so the
# collection was pulling ~93 MB (scipy 59 + scipy.libs 20 + sklearn 14) of dead weight
# into every bundle - and the unguarded collect_submodules("sklearn") would have failed
# the build outright once the dependency was dropped. If a model ever returns, restore
# the collection with it and say which module imports it.

a = Analysis(
    [str(ROOT / "launch_gui.py")],
    pathex=PATHEX,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(SPEC_DIR / "rthook_qt_api.py")],
    # PyQt5 must not enter the bundle: two Qt bindings in one process is a
    # crash, and qtpy would be free to pick the wrong one. The Qt desk is
    # PySide6-only. PyQt5 left the dependency set on 2026-09-03 with TickerMover.py;
    # the exclude stays as a guard against a stray install.
    excludes=["PyQt5", "PyQt5.sip", "PyQt6", "PySide2", "matplotlib", "pytest", "_pytest"],
    noarchive=False,
    optimize=0,
)

# Codex desktop adds its own PDF/image runtimes to PATH. PyInstaller's binary
# dependency walk can mistake those unrelated DLLs for app dependencies. In
# September 2026 that copied poppler's ICU and libheif's private CRT beside the
# desk, and QtCore then failed at import with a missing procedure. Nothing in
# TradingBotV3 imports from Codex's cache; fence the host toolchain out.
_CODEX_RUNTIME_TOKEN = "\\.cache\\codex-runtimes\\"
_python_ssl_names = {"libcrypto-3-x64.dll", "libssl-3-x64.dll"}
_clean_binaries = []
_foreign_codex_count = 0
_python_ssl_replacements = 0
for _entry in a.binaries:
    if _CODEX_RUNTIME_TOKEN not in str(_entry[1]).lower():
        _clean_binaries.append(_entry)
        continue
    _foreign_codex_count += 1
    _binary_name = Path(_entry[0]).name.lower()
    if _binary_name in _python_ssl_names:
        _python_dll = Path(sys.base_prefix) / "DLLs" / _binary_name
        if not _python_dll.is_file():
            raise SystemExit(f"spec error: Python SSL DLL not found: {_python_dll}")
        _clean_binaries.append((_entry[0], str(_python_dll), _entry[2]))
        _python_ssl_replacements += 1
a.binaries = _clean_binaries
if _foreign_codex_count:
    print(
        f"[spec] fenced {_foreign_codex_count} foreign Codex runtime DLL(s); "
        f"restored {_python_ssl_replacements} Python SSL DLL(s)"
    )

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TradingBotV3",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # A console window is kept deliberately, but NOT for the faulthandler crash
    # log — launch_gui.py points that at a file (logs/gui_crash.log), so it
    # survives either way. It is here because the desk's live diagnostics (IB
    # connect/reconnect, autopilot state, Qt warnings) go to stderr, and this app
    # has a history of dying to native access violations. Flip to False for a
    # windowed launch with no terminal; nothing else needs to change.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TradingBotV3",
)
