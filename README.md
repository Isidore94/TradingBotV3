# TradingBotV3

## Prerequisites
- Python 3.10+.
- [Interactive Brokers TWS or IB Gateway](https://www.interactivebrokers.com/en/trading/ib-api.php) running locally with API access enabled on `127.0.0.1:7496`.
- A working Windows desktop session for the GUI. The new consumer UI is PySide6/Qt; the Tk UI remains available during migration.

Install the normal desktop dependencies:

```bash
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependency layers:
- `requirements-core.txt` - headless/mini-PC engines and data services.
- `requirements-gui.txt` - core plus desktop GUI packages.
- `requirements-dev.txt` - GUI plus test and packaging tools.
- `requirements.txt` - compatibility alias for the GUI install.

Run scripts from the repo-local virtual environment:

```powershell
.\.venv\Scripts\python.exe .\scripts\master_avwap.py
```

If you keep personal launcher files such as `run_python_script.ps1` or `run_master_avwap_mini_pc.cmd`, keep them local to your machine. They are intentionally not tracked in git.

## Repository Layout
- `TradingBotV3_GUI.cmd` - Windows launcher for the new PySide6 Trading Desk UI.
- `scripts/gui.py` - compatibility launcher. Defaults to the new PySide6 UI; use `--ui tk` for the legacy Tk UI.
- `scripts/ui/` - new consumer desktop UI.
- `scripts/master_avwap_lib/`, `scripts/bounce_bot_lib/` - trading engines and legacy compatibility modules.
- `market_prep/` - market prep services.
- `docs/` - shipping, cleanup, and future broker architecture notes.
- `packaging/` - Windows `.exe` / installer notes and future PyInstaller files.
- `data/`, `logs/`, `output/` - legacy repo folders; current runtime files are stored under the selected home folder.
- `longs.txt`, `shorts.txt` - Primary shared watchlists consumed by BounceBot and also scanned by Master AVWAP, stored in the selected home folder root.
- `swinglongs.txt`, `shortswings.txt` - Master AVWAP-only swing watchlists. BounceBot does not read these files.

For repo cleanup and shipping direction, see `docs/SHIP_READINESS.md`.
For future multi-broker architecture, see `docs/BROKER_ADAPTERS.md`.

## Required Inputs
- `longs.txt` - one ticker per line for long-side scanning.
- `shorts.txt` - one ticker per line for short-side scanning.
- `swinglongs.txt` - optional Master AVWAP-only long swing tickers.
- `shortswings.txt` - optional Master AVWAP-only short swing tickers.

These files should live in the selected home folder root. The app creates any missing `data`, `logs`, and `output` directories inside that home folder at runtime.

## Running The Bots
- New Qt Trading Desk UI:

  ```powershell
  .\TradingBotV3_GUI.cmd
  ```

  Or launch through Python:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\gui.py
  ```

  To force and save dark mode from the launcher:

  ```powershell
  .\TradingBotV3_GUI.cmd --theme dark
  ```

  This is the target consumer UI. It currently includes the themed shell,
  Trading Desk, Master AVWAP setup table, settings, and placeholders for the
  remaining phased panels. You can also switch between Dark and Light from
  the Settings page.

- Daily AVWAP/previous-AVWAP engine:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\master_avwap.py
  ```

  Generates `output/master_avwap_events.txt` inside the selected home folder and writes diagnostics to `logs/trading_bot.log` there as well. Master AVWAP scans `longs.txt` / `shorts.txt` plus optional `swinglongs.txt` / `shortswings.txt`.

- Always-on mini-PC AVWAP scheduler for a shared home folder:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\master_avwap_mini_pc.py
  ```

  Launches the normal Master AVWAP GUI plus a dedicated `Mini PC` tab. It reuses the full `master_avwap.py` scan logic with the shared-folder watchlists, auto-runs on the default `07:00,08:00,09:00,10:00,11:00,12:00,13:00` schedule, stops at `13:30`, updates the setup tracker, and writes a phone-friendly status file to `master_avwap_mini_pc_status.txt` in the shared home-folder root. Theta plays now appear near the top of that status file.

  Useful flags:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\master_avwap_mini_pc.py --once
  .\.venv\Scripts\python.exe .\scripts\master_avwap_mini_pc.py --dry-run
  .\.venv\Scripts\python.exe .\scripts\master_avwap_mini_pc.py --headless
  .\.venv\Scripts\python.exe .\scripts\master_avwap_mini_pc.py --no-autostart
  .\.venv\Scripts\python.exe .\scripts\master_avwap_mini_pc.py --shutdown-at-end
  ```

  Windows Task Scheduler pattern:
  Program/script: `C:\Users\aaron\Documents\TradingBotV3\.venv\Scripts\python.exe`
  Add arguments: `C:\Users\aaron\Documents\TradingBotV3\scripts\master_avwap_mini_pc.py --headless`
  Start in: `C:\Users\aaron\Documents\TradingBotV3`

  Generic scheduler pattern for any repo script:
  Program/script: `C:\Users\aaron\Documents\TradingBotV3\.venv\Scripts\python.exe`
  Add arguments: `C:\Users\aaron\Documents\TradingBotV3\scripts\your_script.py ...`
  Start in: `C:\Users\aaron\Documents\TradingBotV3`

  The `Mini PC` tab exposes a scheduler status panel, a live preview of the phone status file, and quick access to `Change Home Folder` / `Open Home Folder`. The existing Setup Tracker tab still has the same home-folder controls too.

  If you want the mini PC itself to power off after the scan window, either use `--shutdown-at-end` or create a separate `13:30` Task Scheduler action that runs `shutdown /s /t 0`.

- Consolidated BounceBot + Master AVWAP GUI:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\gui.py --ui tk
  ```

  Legacy Tk GUI. Uses the same selected home folder and writes a shared-root snapshot file named `consolidated_gui_output.txt` so the current GUI outputs can be checked from the synced folder as well. The top-level `Trading` tab contains BounceBot, Master AVWAP, and expanded watchlist editors for shared `longs.txt` / `shorts.txt` plus Master AVWAP-only `swinglongs.txt` / `shortswings.txt`; Market Prep and Ticker Lookup stay in separate tabs.

- Intraday 5-minute bounce detector with optional GUI:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\bounce_bot.py --use_gui
  ```

  Writes detected bouncers to `logs/bouncers.txt`, stores the structured bounce history in `data/intraday_bounces.csv`, and writes runtime diagnostics to `logs/trading_bot.log` inside the selected home folder.

Ensure your IB session is connected before launching either bot so market data requests succeed.

## Notes
- The main rotating app log is `logs/trading_bot.log` in the selected home folder.
- `logs/bouncers.txt` is the lightweight current-session bounce list.
- `logs/rrs_strength_extremes.csv` and `logs/rrs_group_strength_extremes.csv` are data logs for RRS history.

## Syncing Day-To-Day Data Across Devices
- The app can use a per-machine home folder such as Google Drive or OneDrive for day-to-day mutable data.
- In the GUI, open the `Master AVWAP` tab and use `Change Home Folder` to point this computer at your synced folder.
- The home folder stores watchlists, runtime AVWAP data, reports, logs, and setup-tracker files.
- Replaceable download caches now stay in a per-machine local cache directory so Google Drive or OneDrive stays lightweight.
- Place `longs.txt` and `shorts.txt` in the home folder root to share primary BounceBot watchlists across devices.
- Add `swinglongs.txt` and `shortswings.txt` in the same folder for Master AVWAP-only swing candidates.
- Each computer can use a different local path as long as they all point to the same shared cloud folder.
- The chosen folder is saved locally in `%LOCALAPPDATA%\TradingBotV3\local_settings.json`.
- Restart the GUI after changing the home folder so the scripts reload from the new location.
- Avoid running the same writer-heavy workflows from multiple devices at the same time, since JSON/CSV files can still conflict during cloud sync.
