# 0005 — Plain-file storage in a cloud-synced "home folder", no database

Date: backfilled 2026-08-01

## Context
Day-to-day mutable data (watchlists, reports, evidence logs, tracker state) must be
shared across the desktop, the mini-PC, and the trader's phone.

## Decision
All mutable runtime data is plain files (txt watchlists, JSON, CSV, append-only
JSONL) in a user-selected home folder, typically a Google Drive/OneDrive sync
folder. Replaceable download caches and diagnostics stay in a per-machine
`%LOCALAPPDATA%\TradingBotV3` directory; the chosen folder is saved in
`local_settings.json` there. There is no database.

## Rationale
Evident in README: cloud sync gives multi-device sharing and the phone report for
free; caches were moved local "so Google Drive or OneDrive stays lightweight."
Plan.md Milestone 2 ("supervised storage and secrets migration") shows the file
layout is a known tradeoff being actively managed, not an accident.
