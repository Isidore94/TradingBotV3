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

  Generates `output/master_avwap_events.txt` and logs to `logs/`.

- Intraday 5-minute bounce detector (with optional GUI):

  ```bash
  python scripts/bounce_bot.py --use_gui
  ```

  Writes detected bouncers to `logs/bouncers.txt` and `data/intraday_bounces.csv`.

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
- Bounce logs and runtime bot logs are written under `logs/`.
