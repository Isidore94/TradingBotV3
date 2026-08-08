# Packaging Notes

The consumer build should eventually create a Windows desktop installer around
the Qt UI.

## Intended Product Surface

- `TradingBotV3.exe` launches the PySide6 UI.
- Runtime data lives in the selected home folder / `%LOCALAPPDATA%`, not inside
  the installed app directory.
- The legacy Tk UI remains available during migration, but it should not be the
  final consumer entrypoint.

## Development Build Sketch

Use a dev environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Smoke the app before packaging:

```powershell
.\.venv\Scripts\python.exe .\scripts\gui.py --ui qt
```

## Building The Exe

```powershell
.\.venv\Scripts\pip.exe install pyinstaller
.\.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm
```

Output: `dist/TradingBotV3/TradingBotV3.exe` (onedir, ~400MB, ~4 min to build).
`dist/` and `build/` are gitignored.

Onedir rather than onefile deliberately: onefile would re-extract the whole
bundle to a temp directory on every launch.

The exe reads the same stores as the source checkout — `%LOCALAPPDATA%\TradingBotV3`
and whatever `shared_data_dir` names — so both share one set of data.

### Things that will bite you

- **Never let a frozen run insert `<ROOT_DIR>/scripts` onto `sys.path`.**
  `ROOT_DIR` is `sys._MEIPASS` when frozen, and PyInstaller's importer claims any
  path under `_MEIPASS` even when the directory does not exist. The first-party
  packages then resolve to `<_MEIPASS>/scripts/<pkg>` and every submodule import
  fails with a baffling `No module named 'bounce_bot_lib.learning'`.
  `launch_gui.py` guards this with `sys.frozen`.
- **`collect_submodules()` imports the package**, so the spec puts the repo root
  and `scripts/` on `sys.path` first. Without that it raises, and a swallowed
  exception ships a bundle that dies at the first lazy import. The spec fails
  the build instead.
- **PyInstaller bundles `.py` only.** `ui/theme.qss` is copied explicitly; the
  spec aborts if it goes missing.
- **PyQt5 is excluded.** Two Qt bindings in one process is a crash, and qtpy
  picks whichever it finds. A runtime hook pins `QT_API=pyside6`.

## Open Packaging Work

- Create app icon and version metadata.
- Flip `console=True` to `False` in the spec for a windowed launch once the
  native-crash investigation is closed out (the crash log is a file either way).
- Trim the bundle: `collect_submodules("sklearn")` drags in sklearn's test
  modules.
- Add installer creation step after the `.exe` build is stable.
