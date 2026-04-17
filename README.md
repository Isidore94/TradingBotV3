# TradingBotV3

## Prerequisites
- Python 3.10+.
- [Interactive Brokers TWS or IB Gateway](https://www.interactivebrokers.com/en/trading/ib-api.php) running locally with API access enabled on `127.0.0.1:7496`.
- A working GUI environment if you plan to use the Tkinter UI in `bounce_bot.py` or the PyQt5 interface in `TickerMover.py`.

Install Python dependencies:

```bash
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run scripts from the repo-local virtual environment:

```powershell
.\.venv\Scripts\python.exe .\scripts\master_avwap.py
```

If you keep personal launcher files such as `run_python_script.ps1` or `run_master_avwap_mini_pc.cmd`, keep them local to your machine. They are intentionally not tracked in git.

## Repository Layout
- `scripts/` - Intraday and daily-running bots ready for use (`master_avwap.py`, `master_avwap_mini_pc.py`, `bounce_bot.py`, `TickerMover.py`, etc.).
- `data/`, `logs/`, `output/` - Legacy repo folders; current runtime files are stored under the selected home folder.
- `longs.txt`, `shorts.txt` - Input watchlists consumed by the scanners, stored in the selected home folder root.

## Required Inputs
- `longs.txt` - one ticker per line for long-side scanning.
- `shorts.txt` - one ticker per line for short-side scanning.

These files should live in the selected home folder root. The app creates any missing `data`, `logs`, and `output` directories inside that home folder at runtime.

## Running The Bots
- Daily AVWAP/previous-AVWAP engine:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\master_avwap.py
  ```

  Generates `output/master_avwap_events.txt` inside the selected home folder and writes diagnostics to `logs/trading_bot.log` there as well.

- Always-on mini-PC AVWAP scheduler for a shared home folder:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\master_avwap_mini_pc.py
  ```

  Launches the normal Master AVWAP GUI plus a dedicated `Mini PC` tab. It reuses the full `master_avwap.py` scan logic with the shared-folder watchlists, auto-runs on the default `07:00,08:00,09:00,10:00,11:00,12:00,13:00` schedule, stops at `13:30`, updates the setup tracker, and writes a phone-friendly status file to `master_avwap_mini_pc_status.txt` in the shared home-folder root.

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
  .\.venv\Scripts\python.exe .\scripts\gui.py
  ```

  Uses the same selected home folder and now writes a shared-root snapshot file named `consolidated_gui_output.txt` so the current GUI outputs can be checked from the synced folder as well.

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
- Place `longs.txt` and `shorts.txt` in the home folder root to share watchlists across devices.
- Each computer can use a different local path as long as they all point to the same shared cloud folder.
- The chosen folder is saved locally in `%LOCALAPPDATA%\TradingBotV3\local_settings.json`.
- Restart the GUI after changing the home folder so the scripts reload from the new location.
- Avoid running the same writer-heavy workflows from multiple devices at the same time, since JSON/CSV files can still conflict during cloud sync.
