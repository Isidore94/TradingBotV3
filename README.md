# TradingBotV3

## Prerequisites
- Python 3.10+.
- [Interactive Brokers TWS or IB Gateway](https://www.interactivebrokers.com/en/trading/ib-api.php) running locally with API access enabled on `127.0.0.1:7496`.
- A working GUI environment if you plan to use the Tkinter UI in `bounce_bot.py` or the PyQt5 interface in `TickerMover.py`.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

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

  ```bash
  python scripts/master_avwap.py
  ```

  Generates `output/master_avwap_events.txt` inside the selected home folder and writes diagnostics to `logs/trading_bot.log` there as well.

- Always-on mini-PC AVWAP scheduler for a shared home folder:

  ```bash
  python scripts/master_avwap_mini_pc.py
  ```

  Reuses the full `master_avwap.py` scan logic with the shared-folder watchlists, runs on the default `07:00,08:00,09:00,10:00,11:00,12:00,13:00` schedule, stops at `13:30`, updates the setup tracker, and writes a phone-friendly status file to `master_avwap_mini_pc_status.txt` in the shared home-folder root.

  Useful flags:

  ```bash
  python scripts/master_avwap_mini_pc.py --once
  python scripts/master_avwap_mini_pc.py --dry-run
  python scripts/master_avwap_mini_pc.py --shutdown-at-end
  ```

  Windows Task Scheduler pattern:
  Program/script: `py`
  Add arguments: `scripts\\master_avwap_mini_pc.py`
  Start in: the repo root folder

  If you want the mini PC itself to power off after the scan window, either use `--shutdown-at-end` or create a separate `13:30` Task Scheduler action that runs `shutdown /s /t 0`.

- Intraday 5-minute bounce detector with optional GUI:

  ```bash
  python scripts/bounce_bot.py --use_gui
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
- The home folder stores watchlists, caches, runtime AVWAP data, reports, logs, and setup-tracker files.
- Place `longs.txt` and `shorts.txt` in the home folder root to share watchlists across devices.
- Each computer can use a different local path as long as they all point to the same shared cloud folder.
- The chosen folder is saved locally in `%LOCALAPPDATA%\TradingBotV3\local_settings.json`.
- Restart the GUI after changing the home folder so the scripts reload from the new location.
- Avoid running the same writer-heavy workflows from multiple devices at the same time, since JSON/CSV files can still conflict during cloud sync.
