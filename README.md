# TradingBotV3

## Prerequisites
- Python 3.10+.
- [Interactive Brokers TWS or IB Gateway](https://www.interactivebrokers.com/en/trading/ib-api.php) running locally with API access enabled on `127.0.0.1:7496`.
- A working GUI environment if you plan to use the Tkinter UI in `bounce_bot.py` or the PyQt5 interface in `TickerMover.py`.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Repository layout
- `scripts/` – Intraday and daily-running bots ready for out-of-the-box use (`master_avwap.py`, `bounce_bot.py`, `TickerMover.py`, etc.).
- `maintenance/` – Workspace cleanup helpers that are intentionally outside the main script directory.
- `data/`, `logs/`, `output/` – Generated artifacts. These are created as needed by the scripts.
- `longs.txt`, `shorts.txt` – Input watchlists consumed by the scanners.

## Required inputs
- `longs.txt` – one ticker per line for long-side scanning.
- `shorts.txt` – one ticker per line for short-side scanning.

These files should live in the repository root. The scripts will create any missing `data`, `logs`, and `output` directories at runtime.

## Running the bots
- Daily AVWAP/previous-AVWAP engine:

  ```bash
  python scripts/master_avwap.py
  ```

  Generates `output/master_avwap_events.txt` and writes diagnostics to the shared app log under `logs/trading_bot.log`.

- Intraday 5-minute bounce detector (with optional GUI):

  ```bash
  python scripts/bounce_bot.py --use_gui
  ```

  Writes detected bouncers to `logs/bouncers.txt`, stores the structured bounce history in `data/intraday_bounces.csv`, and writes runtime diagnostics to `logs/trading_bot.log`.

Ensure your IB session is connected before launching either bot so that market data requests succeed.

## Maintenance tools
To reset generated artifacts to the repository's default state, use the intentionally tucked-away maintenance script:

```bash
python maintenance/tidy_workspace.py --dry-run  # inspect what would be removed
python maintenance/tidy_workspace.py            # perform the cleanup
```

It is kept outside the `scripts/` directory to avoid accidental execution.

## Notes
- AI training/labeling scripts were removed from this repository.
- The main rotating app log is `logs/trading_bot.log`.
- `logs/bouncers.txt` is the lightweight current-session bounce list.
- `logs/rrs_strength_extremes.csv` and `logs/rrs_group_strength_extremes.csv` are data logs for RRS history.

## Syncing day-to-day data across devices
- The app can use a per-machine home folder such as Google Drive or OneDrive for day-to-day mutable data.
- In the GUI, open the `Master AVWAP` tab and use `Change Home Folder` to point this computer at your synced folder.
- The home folder stores watchlists, caches, runtime AVWAP data, reports, logs, and setup-tracker files.
- Place `longs.txt` and `shorts.txt` in the home folder root to share watchlists across devices.
- Each computer can use a different local path as long as they all point to the same shared cloud folder.
- The chosen folder is saved locally in `%LOCALAPPDATA%\TradingBotV3\local_settings.json`.
- Restart the GUI after changing the home folder so the scripts reload from the new location.
- Avoid running the same writer-heavy workflows from multiple devices at the same time, since JSON/CSV files can still conflict during cloud sync.
