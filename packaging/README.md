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

The real PyInstaller spec should be added here once the Qt UI reaches enough
parity to be the default:

```powershell
.\.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec
```

## Open Packaging Work

- Create app icon and version metadata.
- Add PyInstaller spec with PySide6, qtawesome, pyqtgraph, config files, and
  required hidden imports.
- Decide whether to include PyQt5 while `TickerMover.py` and the legacy UI still
  exist, or retire/fold them into PySide6 first.
- Add installer creation step after the `.exe` build is stable.
