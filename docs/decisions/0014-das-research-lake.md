# 0014 — DAS research lake as a new append-only storage class

Date: 2026-08-03 (trader-directed)
Relates to: 0005 (home-folder/mutable-state policy), 0012 (dependency pinning),
plan.md sec 12 item 13a, docs/ULTIMATE_SETUP_DATABASE_PLAN.md

## Context

The research warehouse (bar archives, feature snapshots, outcome paths) will
grow to tens of GB per year — far past what the Drive-synced home folder can
carry without quota, sync-lock, and DriveFS-wedge problems. The trader owns a
DAS and explicitly accepts large local files.

## Decision

1. A new storage class exists: the research lake — immutable Parquet/Zstd files
   on a trader-owned local/DAS disk at `research_store_dir`
   (`local_settings.json`; `TRADINGBOTV3_RESEARCH_DIR` env override), written
   only by the main desktop's build/import job via the 4-step seal protocol,
   read via `manifest_log.jsonl`. Config: `scripts/research_warehouse/config.py`;
   the warehouse is fully disabled when the path is unset.
2. Decision 0005 remains FULLY IN FORCE for operational mutable data:
   watchlists, reports, JSONL evidence logs, and every live surface stay in the
   Drive home folder / `%LOCALAPPDATA%` exactly as today. Nothing operational
   moves to the lake.
3. The Drive home folder additionally carries: (a) nightly Class A mirrors
   (irreplaceable-small research artifacts), and (b) the future
   `research_inbox/` bundle channel — always whole immutable files, never a
   live database.
4. No mutable database file ever lives in the Drive folder or on the DAS. Any
   `.duckdb` file is a disposable machine-local cache, never shared, never
   authoritative. A configured lake path inside the shared home folder is
   refused at the config layer, never silently accepted.

## Consequences

- One new config key (`research_store_dir`); machine-local write spool at
  `%LOCALAPPDATA%\TradingBotV3\research_spool`.
- Backup: 3-class policy per plan sec 8.5; Class B lake copies live on a second
  physical disk, never in Drive.
- The lake is out of scope for decision 0005's single-file atomic-publish
  rules; its integrity contract is the seal protocol + manifest instead.

## Rejected alternatives

- Extending 0005 to cover the lake (mutable-publish semantics are wrong for an
  append-only archive).
- A shared DuckDB/SQLite database as the store (Windows file locking,
  single-writer concurrency model, Drive sync hazards).
- Hosting the lake inside the Drive home folder (sync latency, DriveFS wedge
  precedent 2026-07-17, quota).
