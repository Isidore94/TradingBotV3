# 0005 — Plain-file storage in a cloud-synced "home folder", no database

Date: backfilled 2026-08-01

> **Status: partially superseded by [0015](0015-no-cloud-sync-das-file-server-storage.md)
> (2026-08-10).** There is no cloud drive anymore. The plain-file, no-database
> decision below **stays fully in force**, and the home folder keeps its path
> (`C:\TradingBotData`) and its role — only the "cloud-synced" premise is dead.
> Everything below is preserved as the original rationale; read 0015 for what is
> true now.

Topology amendment: 2026-08-08 — the main desk is now the sole runtime writer;
cloud sync remains for operational continuity and phone-readable reports, not for a
second live scanner.

## Context
Day-to-day mutable data (watchlists, reports, evidence logs, tracker state) must be
available to the main desk and through small phone-readable/cloud-synced artifacts.

## Decision
All mutable runtime data is plain files (txt watchlists, JSON, CSV, append-only
JSONL) in a user-selected home folder, typically a Google Drive/OneDrive sync
folder. Replaceable download caches and diagnostics stay in a per-machine
`%LOCALAPPDATA%\TradingBotV3` directory; the chosen folder is saved in
`local_settings.json` there. There is no shared mutable database. Decision 0014
separately governs the immutable research lake.

## Rationale
Evident in README: cloud sync gives multi-device sharing and the phone report for
free; caches were moved local "so Google Drive or OneDrive stays lightweight."
Plan.md Phase 2 item P2.1 ("storage and secrets classification") shows the file
layout is a known tradeoff being actively managed, not an accident.
