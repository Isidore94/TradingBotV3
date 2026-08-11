# macOS setup — Trading Desk on a Mac

Document role: **active platform runbook**. Run the app in Main mode only; Desk Link
satellite operation is retired.

> **Storage amendment (2026-08-10) — decision
> [0015](decisions/0015-no-cloud-sync-das-file-server-storage.md).** The
> production desk no longer uses Google Drive at all: the home folder is plain
> local storage and the DAS file server is the durable tier. The Google Drive
> sections below therefore describe **dead configuration on the current desk**.
> They are kept because the CloudStorage mount-discovery code still exists and
> still works, so a Mac could still be pointed at a synced folder — but that is
> no longer how this system is set up, and a fresh Mac should simply select a
> local home folder and mount the file server.

The desk runs the same code on macOS as on Windows: one PySide6 GUI
(`scripts/ui/app.py`), the same scanners, the same shared home folder.
There is no mac fork — any change made to the main GUI or the underlying
scripts applies to both platforms on the next `git pull`. Platform
differences are confined to the launcher, the local-state directory, and the
credential store.

## Requirements

- macOS on Apple Silicon or Intel.
- Python **3.12+** (`brew install python` or python.org installer).
- **Trader Workstation (TWS) for macOS** or IB Gateway, running locally.
- Google Drive for desktop (optional but recommended — it hosts the shared
  home folder that syncs watchlists/reports/tracker state across machines).

## One-time setup

```bash
git clone <this repo> && cd TradingBotV3
./setup_macos.command      # creates .venv, installs requirements-gui pinned by constraints.txt
```

(Double-clicking `setup_macos.command` in Finder does the same thing.
`TRADINGBOTV3_PYTHON=/path/to/python3.12 ./setup_macos.command` picks a
specific interpreter.)

## Launching

Run:

```bash
.venv/bin/python launch_gui.py
```

`launch_gui.py` is the single desktop entrypoint on every platform, including
native-crash logging (`gui_crash.log`). Keep the role on Main; satellite operation
is retired.

## IBKR TWS on macOS

Nothing changes in the app: it connects to `127.0.0.1:7496` exactly as on
Windows (`yfinance` remains the fallback bar source). In TWS:

1. **File → Global Configuration → API → Settings**
2. Enable *ActiveX and Socket Clients*.
3. Confirm the socket port is **7496** (live TWS default).
4. Keep *Read-Only API* on if you like — the bot is decision-support only
   and never places orders (plan.md sec 5).

## Shared home folder (Google Drive)

Google Drive for desktop on modern macOS mounts under
`~/Library/CloudStorage/GoogleDrive-<account>/My Drive`. The app detects
that automatically (in addition to the legacy `~/My Drive` / `~/Google
Drive` locations) and uses `<mount>/Trading/TradingBot` as the shared home,
same as `G:\My Drive\Trading\TradingBot` on Windows.

Overrides work as on Windows: the `TRADINGBOTV3_DATA_DIR` environment
variable, or `shared_data_dir` in `local_settings.json` (set from the GUI's
storage settings). If the configured store sits under `CloudStorage` and the
Drive client is not running yet, startup waits up to 120 s for the mount
(`TRADINGBOTV3_DRIVE_WAIT_SECONDS` adjusts, `0` = fail fast) and then fails
with an actionable message — never a silent local fallback, which would fork
the shared tracker/watchlist state.

## Per-machine state

What lives in `%LOCALAPPDATA%\TradingBotV3` on Windows lives in
`~/Library/Application Support/TradingBotV3` on macOS: `local_settings.json`,
`machine_cache/`, `diagnostics/` (run manifests, shadow JSONLs, heartbeat),
and rotating app logs.

## AI provider keys

Keys saved from the A.I. Summary panel go to the **macOS login Keychain**
(generic passwords, service `TradingBotV3/ai-summary/<provider>`) — the
counterpart of Windows Credential Manager. `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` environment variables always win over saved keys.

## Development on macOS

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q  # fully green before commit
.venv/bin/python scripts/smoke_check.py                          # 7/7, offline
```

These mirror the Windows commands in CLAUDE.md (`.venv\Scripts\python.exe …`).
`QT_QPA_PLATFORM=offscreen` keeps the Qt widget tests headless. Note the
dependency pins: `constraints.txt` gives macOS its own `PyQt5-Qt5` pin
(5.15.17 — the Windows 5.15.2 pin has no macOS wheel).

## Still Windows-only

- Task Scheduler registration for the repeating main-desk launch and Local-AI jobs.
- PyInstaller `.exe` verification and its Windows runtime hook.
- Windows Credential Manager storage (Keychain replaces it here).
- The legacy background-thread priority drop in the Auto Pilot wrap-up is a
  no-op off Windows by design.
