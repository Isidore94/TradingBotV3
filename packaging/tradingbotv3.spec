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

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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

datas = [
    # market_prep/config_loader.py resolves CONFIG_DIR as
    # Path(__file__).parents[1] / "config", which lands at the bundle root.
    (str(ROOT / "config"), "config"),
]

# PyInstaller bundles .py only. ui/theme.py reads its Qt stylesheet with
# Path(__file__).with_name("theme.qss"), so every non-Python asset under
# scripts/ui has to be mirrored into the bundle at the same relative location.
_UI_DIR = SCRIPTS / "ui"
_ui_assets = [p for p in _UI_DIR.rglob("*") if p.is_file() and p.suffix.lower() not in (".py", ".pyc")]
for _asset in _ui_assets:
    _rel = _asset.parent.relative_to(_UI_DIR)
    datas.append((str(_asset), str(Path("ui") / _rel) if _rel.parts else "ui"))
# The desk renders unstyled without the stylesheet and dies on the missing file,
# so treat its absence as a build failure rather than shipping a broken exe.
if not any(a.name == "theme.qss" for a in _ui_assets):
    raise SystemExit("spec error: ui/theme.qss not found — the desk cannot start without it")
print(f"[spec] ui assets bundled: {[a.name for a in _ui_assets]}")
datas += collect_data_files("qtawesome")   # bundled icon fonts
datas += collect_data_files("pyqtgraph")
datas += collect_data_files("certifi")

hiddenimports = []
# The UI loads panels/services by name in places, and the engines import each
# other lazily inside functions; collecting the first-party trees outright is
# far cheaper than chasing ModuleNotFoundError one launch at a time.
#
# Deliberately NOT wrapped in try/except: a package that fails to collect here
# is a bundle that starts and then dies at the first lazy import. Fail the
# build loudly instead.
for package in ("ui", "bounce_bot_lib", "master_avwap_lib", "market_prep", "diagnostics", "research_warehouse"):
    found = collect_submodules(package)
    if not found:
        raise SystemExit(f"spec error: collect_submodules({package!r}) found nothing — check sys.path above")
    print(f"[spec] {package}: {len(found)} submodules")
    hiddenimports += found
# scikit-learn and scipy reach for submodules dynamically at predict time.
hiddenimports += collect_submodules("sklearn")
hiddenimports += ["scipy._lib.array_api_compat.numpy.fft", "scipy.special._special_ufuncs"]

a = Analysis(
    [str(ROOT / "launch_gui.py")],
    pathex=PATHEX,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(SPEC_DIR / "rthook_qt_api.py")],
    # PyQt5 must not enter the bundle: two Qt bindings in one process is a
    # crash, and qtpy would be free to pick the wrong one. The Qt desk is
    # PySide6-only; PyQt5 exists in the venv solely for legacy TickerMover.py.
    excludes=["PyQt5", "PyQt5.sip", "PyQt6", "PySide2", "matplotlib", "pytest", "_pytest"],
    noarchive=False,
    optimize=0,
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
