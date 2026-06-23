# Shipping Readiness Notes

TradingBotV3 can keep a modular codebase without feeling messy to consumers. The
goal is not to reduce every file; it is to reduce the number of things a user
or installer needs to know about.

## Do We Need This Many Files?

Yes for implementation modules and tests. No for public entrypoints.

The consumer-facing product should eventually expose:

- `TradingBotV3.exe` - the desktop app.
- Optional background/scheduler entrypoint for an always-on mini-PC install.
- A settings/import/export surface inside the app, not loose scripts.

Everything else should be internal app code, developer tooling, tests, or
compatibility shims while the legacy Tk app is retired.

## Current Shape

- `scripts/gui.py` is the compatibility GUI launcher. It now supports
  `--ui qt` for the new PySide6 app and keeps `--ui tk` as the default legacy
  path during migration.
- `scripts/ui/` is the new consumer UI layer.
- `scripts/master_avwap_lib/` and `scripts/bounce_bot_lib/` contain the existing
  trading engines plus legacy compatibility code.
- `market_prep/` is already closer to a service-style package.
- `tests/` is healthy and should stay broad. Consumer polish should not mean
  reducing test coverage.

## Dependency Layers

- `requirements-core.txt` - headless engines, data, broker API, market prep.
- `requirements-gui.txt` - core plus desktop GUI dependencies.
- `requirements-dev.txt` - GUI plus test/build tooling.
- `requirements.txt` - compatibility alias for GUI installs.

For a consumer `.exe`, the PyInstaller build should use the GUI/dev environment.
For a mini-PC/headless setup, prefer `requirements-core.txt` unless the GUI is
needed.

## Repo Touch-Up Direction

1. Keep one primary GUI launcher and one scheduler launcher.
2. Move standalone developer utilities under a future `tools/` or `scripts/dev/`
   area once imports/tests confirm they are not user workflows.
3. Keep compatibility shims until the Qt UI reaches parity, then retire Tk GUI
   modules in one deliberate cleanup.
4. Keep generated data, logs, caches, local IDE folders, and cloud-sync runtime
   files out of git.
5. Before packaging, create a real PyInstaller spec and installer notes under
   `packaging/`.

## What Not To Do Yet

- Do not delete legacy modules just because they are large. `legacy.py` files
  are still active compatibility bridges.
- Do not move broker-specific code until an adapter interface exists and tests
  cover the old behavior.
- Do not make the installer depend on repo-relative watchlists or logs. Runtime
  data should stay under the selected home folder / `%LOCALAPPDATA%`.
