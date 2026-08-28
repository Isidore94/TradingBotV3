# TradingBotV3

TradingBotV3 is a private Windows-first trading decision-support desk for one trader.
It prepares the market, scans D1 anchored-VWAP swing setups, monitors intraday M5
bounces, surfaces alerts, publishes an Auto/Away phone report, records decisions and
outcomes, and supports controlled research. It never places orders.

## Documentation

The documentation has five entry points:

- [`CHANGELOG.md`](CHANGELOG.md) — what is implemented and the revision history;
- [`plan.md`](plan.md) — what remains, in order, with validation/promotion gates;
- [`CURRENT_CHECKPOINT.md`](CURRENT_CHECKPOINT.md) — active work, current branch, and exact verification checkpoint;
- [`WISHLIST.md`](WISHLIST.md) — candidate integrations that are not authorized work;
- [`docs/README.md`](docs/README.md) — every runbook, specification, decision record,
  and historical document classified by role.

## Current operating model

- `launch_gui.py` starts the PySide6 Trading Desk.
- There is one desk role and no flag to change it. Desk Link/satellite mode and the
  separate mini-PC scanner role were retired on 2026-08-08 and their code was **removed
  on 2026-08-24** (P1.5): no `desk_link` package, no `--satellite`/`--desk-role` flags.
- One desk per machine: `launch_gui.py` takes a machine-local slot, and a second launch
  exits 0 with a message. `--allow-second-instance` overrides it.
- IBKR TWS/Gateway on `127.0.0.1:7496` is the primary market-data source; yfinance is
  the fallback.
- The shared home folder `C:\TradingBotData` — a plain local folder on the desk SSD —
  stores operational watchlists, reports, and evidence logs. There is no cloud sync;
  Google Drive/OneDrive were removed on 2026-08-10 (decision 0015).
- Durable storage is the DAS file server at `\\MINI-PC\Trading Bot Data`. Cold, only-
  growing subtrees are pushed to it hourly; large writes stage on local disk first.
- Per-machine settings, replaceable caches, and diagnostics live under
  `%LOCALAPPDATA%\TradingBotV3`.
- Large Parquet research data lives only in the separately configured research lake on
  the DAS, never inside the `C:\TradingBotData` home folder. With no research path
  configured, the warehouse is disabled.

## Requirements and installation

- Windows 10/11 with Python 3.12+; the desk repo uses a uv-managed Python 3.12
  environment.
- IBKR TWS or IB Gateway with API access enabled on `127.0.0.1:7496`.
- A desktop session for the PySide6 GUI.

The existing repo `.venv` has no `pip`. Install or refresh dependencies with uv:

```powershell
uv pip install -r requirements-dev.txt -c constraints.txt --python .venv\Scripts\python.exe
```

Dependency layers:

- `requirements-core.txt` — headless engines and data services;
- `requirements-gui.txt` — core plus the desktop UI;
- `requirements-dev.txt` — GUI plus tests and packaging;
- `requirements.txt` — compatibility alias for the GUI install;
- `constraints.txt` — reproducible pins.

For macOS, follow [`docs/MACOS_SETUP.md`](docs/MACOS_SETUP.md). The same source tree
is used on both platforms.

## Launch

Start TWS/Gateway first, then:

```powershell
.venv\Scripts\python.exe launch_gui.py
```

The `trading_desk.cmd` launcher wraps the same command. **The source launch is
production** by trader decision (2026-08-26): a pushed commit is live at the next
restart, and the frozen exe is a verification artifact only. If the desk ever returns
to the frozen exe, a fix is not delivered until the exe is rebuilt.

Optional theme override:

```powershell
.venv\Scripts\python.exe launch_gui.py --theme dark
```

The legacy Tk UI remains available only during migration:

```powershell
.venv\Scripts\python.exe scripts\gui.py --ui tk
```

`scripts/master_avwap_mini_pc.py` was removed on 2026-08-24. The named-slot scheduling
shape it established lives on in `scripts/ai_jobs/runner.py`.

## Required watchlists

The selected shared-home root contains plain-text files with one symbol per line:

- `longs.txt` — shared long names for BounceBot and Master AVWAP;
- `shorts.txt` — shared short names for BounceBot and Master AVWAP;
- `swinglongs.txt` — optional Master AVWAP-only long swings;
- `shortswings.txt` — optional Master AVWAP-only short swings.

User-entered names are never automatically removed. The app creates needed runtime
subdirectories inside the selected home.

## Auto/Away and phone alerts

`autopilot_today.txt` is the single verified phone-facing digest in the home folder,
announced over ntfy. It keeps the
safety/freshness header first, then numbered best swing trades, intraday candidates,
and condensed operations. Only the main desk publishes it.

For ntfy:

1. install ntfy on the phone and subscribe to a long random topic;
2. in **Research → Price Alerts**, set the server (normally `https://ntfy.sh`),
   topic, and optional token;
3. send a test push and confirm phone/watch permissions;
4. add cross-up/cross-down levels from Focus or Research.

Each side fires once per arm and stays disarmed until manually re-armed. Price alerts
are last-price crossings, not setup confirmations. See
[`docs/EVENING_MODE_RUNBOOK.md`](docs/EVENING_MODE_RUNBOOK.md) and
[`docs/AWAY_SCANNER_RUNBOOK.md`](docs/AWAY_SCANNER_RUNBOOK.md).

## Verification

Before every commit:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
.venv\Scripts\python.exe scripts\smoke_check.py
```

Check pytest's own exit code. The exact current baseline is in `CURRENT_CHECKPOINT.md`.

The frozen application supports a no-window, no-network engine check:

```powershell
dist\TradingBotV3\TradingBotV3.exe --selftest
```

The check count is a running total that grows as checks are added, not a fixed
number: it was 29 on 2026-08-09 and the unfrozen tree measured 72 on 2026-08-27.
Expect `selftest OK: N/N checks passed (frozen)` and exit 0, and compare N against a
current unfrozen `launch_gui.py --selftest` run rather than a number quoted in a doc.
Read [`packaging/README.md`](packaging/README.md) before rebuilding or changing the
spec.

## Repository layout

- `launch_gui.py` — single operator launcher;
- `scripts/ui/` — PySide6 Trading Desk;
- `scripts/master_avwap.py` and `scripts/master_avwap_lib/` — D1 AVWAP scanner;
- `scripts/bounce_bot.py` and `scripts/bounce_bot_lib/` — M5 bounce detector;
- `market_prep/` — pre-session services;
- `scripts/research_warehouse/` — disabled-by-default research lake;
- `tests/` — deterministic, Qt, broker, network, slow, and packaging coverage;
- `docs/` — indexed runbooks, design references, decisions, and historical records;
- `packaging/` — PyInstaller spec and frozen-build guidance.
