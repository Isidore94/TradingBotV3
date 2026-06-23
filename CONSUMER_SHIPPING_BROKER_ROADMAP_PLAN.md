# Consumer Shipping + Broker Architecture Plan

> Goal: turn TradingBotV3 from a powerful local trading workspace into a
> consumer-installable Windows desktop product, while preserving the working bot
> logic and creating a path toward multiple broker/data providers.

---

## 1. North Star

The shipped product should feel like:

- One Windows desktop app: `TradingBotV3.exe`.
- One optional background/scheduler mode for always-on machines.
- Clean settings for broker connection, storage folder, watchlists, theme, and
  explain/compact modes.
- No user-visible pile of scripts, logs, or implementation details.
- Broker/data-provider support that starts with IBKR but can later add other
  providers without rewriting the scanners or UI.

The repo can still have many files. The consumer should not have to know that.

---

## 2. Guardrails

- Do not rewrite trading/scoring logic during cleanup.
- Keep the current engines runnable after every phase.
- Keep tests green before and after each structural move.
- Keep Tk compatibility until Qt reaches parity.
- Separate public launchers from internal modules.
- Move broker-specific SDK objects behind app-owned interfaces before adding a
  second broker.
- Runtime data belongs in `%LOCALAPPDATA%` or the selected shared home folder,
  not inside the installed app directory.

---

## 3. Target Architecture

Long-term repo shape:

```text
TradingBotV3/
  tradingbot/                 # future installable app package
    app/                      # Qt shell, panels, state
    engines/                  # AVWAP, BounceBot, Market Prep orchestration
    brokers/                  # provider interfaces + adapters
    storage/                  # settings, paths, migrations
    models/                   # app-owned dataclasses/view models
    services/                 # background workers, schedulers, diagnostics
  scripts/                    # thin compatibility launchers only
  config/                     # shipped default config/templates
  docs/                       # developer/product architecture
  packaging/                  # PyInstaller/installer files
  tests/                      # unit/integration coverage
```

Near term, we do not need to rename everything. We can evolve toward this while
preserving `scripts/` compatibility.

---

## 4. Phase 1 - Product Boundary And Repo Hygiene

Purpose: make the repo understandable without moving fragile code.

Work:

- Keep `scripts/gui.py` as the main desktop launcher during migration.
- Keep `scripts/master_avwap_mini_pc.py` as the scheduler/background entrypoint.
- Label everything else as internal, legacy, or developer tooling.
- Keep dependency layers:
  - `requirements-core.txt`
  - `requirements-gui.txt`
  - `requirements-dev.txt`
- Keep `.gitignore` strict for logs, caches, runtime data, IDE files, and build
  output.
- Add a small `docs/ENTRYPOINTS.md` explaining which commands are public.

Acceptance:

- New developer can identify the consumer app launcher in under one minute.
- Full tests pass.
- No behavior change for existing workflows.

---

## 5. Phase 2 - Broker/Data Adapter Boundary

Purpose: make IBKR one provider, not the app architecture.

Work:

- Create `scripts/brokers/` or `scripts/providers/` with app-owned protocols:
  - `MarketDataProvider`
  - `OptionsDataProvider`
  - `ExecutionImporter`
  - `BrokerConnectionProfile`
- Define app-owned dataclasses:
  - `BarFrameRequest`
  - `QuoteSnapshot`
  - `OptionContract`
  - `OptionChain`
  - `ExecutionFill`
  - `BrokerHealth`
- Wrap existing IBKR daily/intraday/option calls behind an `IBKRProvider`.
- Wrap existing Yahoo fallback behind a `YahooDataProvider`.
- Keep old functions as compatibility shims that call the provider.

Order:

1. Daily bars.
2. Intraday bars.
3. Option chains and theta quotes.
4. Journal execution import.
5. Broker health/status reporting.

Acceptance:

- Scans produce the same outputs with the IBKR/Yahoo providers.
- Tests cover provider fallback behavior.
- UI does not import `ibapi` directly.

---

## 6. Phase 3 - Qt UI Parity And Legacy Retirement

Purpose: make the new UI the actual app.

Work:

- Finish Master AVWAP panel:
  - copy lists
  - detail drawer
  - watchlist editor
  - setup tracker summaries
  - factor impact/scenario tables
- Port BounceBot live panel:
  - start/stop
  - connection health
  - live alert feed
  - RRS/timeframe/environment controls
- Port Journal:
  - import runs
  - trade table
  - filters
  - stats
- Port Research:
  - Market Prep
  - Ticker Lookup
- Fold or retire `TickerMover.py`.
- Flip launcher default from Tk to Qt only after parity.
- Remove Tk modules in one deliberate cleanup after a tagged backup/release.

Acceptance:

- Qt UI can perform the daily user workflow without opening Tk.
- Legacy GUI remains available until parity is confirmed.
- Full tests and a manual UI smoke checklist pass.

---

## 7. Phase 4 - Consumer Runtime Experience

Purpose: make the app feel safe and guided outside the developer machine.

Work:

- First-run setup wizard:
  - choose storage/home folder
  - create/watchlist templates
  - choose broker/data provider
  - explain IBKR/TWS requirements
- Settings:
  - broker profiles
  - data directory
  - local cache location
  - theme/density/explain mode
  - diagnostics/export logs
- Add health checks:
  - broker connection
  - market data permission hints
  - missing watchlists
  - stale data warnings
  - disk/cloud-sync warnings
- Add safe import/export:
  - settings export
  - watchlist export
  - journal backup

Acceptance:

- Fresh Windows install can launch without manual repo knowledge.
- Missing broker/data setup produces helpful guidance, not tracebacks.
- Logs and diagnostics are accessible from the UI.

---

## 8. Phase 5 - Packaging And Installer

Purpose: ship an installable Windows app.

Work:

- Create `packaging/tradingbotv3.spec` for PyInstaller.
- Bundle:
  - Qt/PySide6 dependencies
  - theme assets
  - config templates
  - app icon/version metadata
- Exclude:
  - tests
  - local data/log/output folders
  - developer-only tooling
- Add a Windows installer layer:
  - Inno Setup or WiX
  - Start Menu shortcut
  - Desktop shortcut option
  - uninstall support
- Add release checklist:
  - clean venv build
  - full tests
  - app smoke test
  - installer smoke test on a clean Windows user profile
  - checksum/version notes

Acceptance:

- Installer produces a working `TradingBotV3.exe`.
- App starts on a clean Windows machine without the source repo.
- Runtime data is created in the selected user data folder.

---

## 9. Phase 6 - Multi-Broker Expansion

Purpose: add providers without destabilizing the core app.

Candidate order:

1. Data-only provider improvements, such as Polygon or another market-data API.
2. Journal/import provider, such as CSV/manual broker exports.
3. Second live broker provider only after market-data and journal abstractions
   are stable.
4. Order routing only if the product explicitly needs execution, with a separate
   risk review.

Work:

- Add provider capability flags:
  - daily bars
  - intraday bars
  - option chains
  - quotes
  - executions
  - positions
  - orders
- Let Settings show only features supported by the selected provider.
- Add provider-specific diagnostics.
- Keep strategy/scanner logic consuming app-owned data models only.

Acceptance:

- Adding a new provider does not require UI rewrites.
- Provider tests can run without live credentials where possible.
- Unsupported provider features degrade clearly.

---

## 10. Phase 7 - Product Polish

Purpose: make the product credible for non-developer consumers.

Work:

- App branding: icon, name, window title, installer metadata.
- Better empty/error states.
- Explain mode and glossary.
- Compact mode and keyboard shortcuts for experienced traders.
- In-app update/release notes strategy.
- Crash-safe logging and diagnostics bundle.
- Performance pass on startup, scan workers, and report parsing.
- Accessibility and HiDPI pass.

Acceptance:

- First run feels guided.
- Experienced daily workflow is fast.
- Errors are understandable and recoverable.
- The app looks like one cohesive product.

---

## 11. Decision Log Needed

Open owner decisions:

- Is the final product IBKR-only at first, or should data-only providers ship in
  v1?
- Is order execution in scope, or is this scanner/journal/research only?
- Installer technology: Inno Setup, WiX, MSIX, or simple zipped `.exe` first?
- Paid distribution needs: code signing, license keys, auto-update, support
  diagnostics.
- Whether the mini-PC scheduler remains a separate app mode or becomes a Windows
  background service/task created by the installer.

---

## 12. Immediate Next Moves

Recommended next implementation steps:

1. Add `docs/ENTRYPOINTS.md`.
2. Add initial broker provider protocols and dataclasses.
3. Wrap daily/intraday bar fetching behind providers without changing scanner
   outputs.
4. Continue Qt parity work: BounceBot panel next.
5. Add a first PyInstaller smoke spec once Qt can handle the main daily workflow.

