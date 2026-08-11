# Packaging Notes

Document role: **active frozen-build runbook**. Current implementation history is in
the root `CHANGELOG.md`; rebuild triggers and verification procedure live here.

The consumer build should eventually create a Windows desktop installer around
the Qt UI.

## Intended Product Surface

- `TradingBotV3.exe` launches the PySide6 UI.
- Runtime data lives in the selected home folder / `%LOCALAPPDATA%`, not inside
  the installed app directory.
- The legacy Tk UI remains available during migration, but it should not be the
  final consumer entrypoint.

## Development environment

The repo `.venv` is uv-managed and has no pip. Refresh it with:

```powershell
uv pip install -r requirements-dev.txt -c constraints.txt --python .venv\Scripts\python.exe
```

Smoke the app before packaging:

```powershell
.\.venv\Scripts\python.exe .\launch_gui.py
```

## Building The Exe

```powershell
.\.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm
```

Output: `dist/TradingBotV3/TradingBotV3.exe` (onedir, ~400MB, ~4 min to build).
`dist/` and `build/` are gitignored.

Onedir rather than onefile deliberately: onefile would re-extract the whole
bundle to a temp directory on every launch.

The exe reads the same stores as the source checkout — `%LOCALAPPDATA%\TradingBotV3`
and whatever `shared_data_dir` names — so both share one set of data.

After every required rebuild, run:

```powershell
dist\TradingBotV3\TradingBotV3.exe --selftest
```

Current expected result: `selftest OK: 29/29 checks passed (frozen)`.

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
- **PyInstaller bundles `.py` only.** Every non-Python file inside a package in
  `FIRST_PARTY_PACKAGES` is mirrored automatically — `ui/theme.qss`, the veto
  vocabulary, `research_warehouse/exploration_cohort.txt`. An asset outside
  those trees still needs its own `datas` entry. The spec aborts if `theme.qss`
  goes missing.
- **PyQt5 is excluded.** Two Qt bindings in one process is a crash, and qtpy
  picks whichever it finds. A runtime hook pins `QT_API=pyside6`.
- **DuckDB is optional.** `research_warehouse.queries` imports it inside the two
  functions that use it, behind `duckdb_available()`, and pyarrow answers every
  slice query without it (LD-04). The spec collects it when it is installed and
  says so when it is not; neither case fails the build.

### The spec-drift guard

`tests/test_packaging_spec_drift.py` executes this spec with the PyInstaller
collectors stubbed and checks it against the tree: every package under
`scripts/` is either in `FIRST_PARTY_PACKAGES` or in the test's
`PACKAGES_NOT_IN_THE_BUNDLE` with a reason, and every non-`.py` file under
`scripts/` is either bundled or in `ASSETS_NOT_IN_THE_BUNDLE` with a reason.
It runs in the normal suite in under a tenth of a second.

That converts CLAUDE.md's rebuild triggers 2-4 into a test failure at commit
time rather than a `ModuleNotFoundError` on the desk weeks later. It does not
retire the rebuild: it proves the spec still describes the tree, not that
PyInstaller can freeze it. Triggers 1 and 5 — a new third-party dependency and
anything touching `__file__` / `ROOT_DIR` / `sys.path` — are still build-and-run
questions, and a green suite has never been evidence that the exe starts.

## Open Packaging Work

- Create app icon and version metadata.
- Flip `console=True` to `False` in the spec for a windowed launch once the
  native-crash investigation is closed out (the crash log is a file either way).
- Trim the bundle: `collect_submodules("sklearn")` drags in sklearn's test
  modules.
- Add installer creation step after the `.exe` build is stable.
