# First-session validation checklist (plan.md sec 6.1 + sec 12 items 3-4)

Operator checklist for the first restarted live session on the
`milestone-1-observability` build. Every step names the command to run and the
artifact that proves it. **A green deterministic test is not live evidence** —
each PASS below must come from inspecting the real artifact after the real
event.

Record results in the table at the bottom. Statuses stay
IMPLEMENTED/GREEN until the corresponding live evidence exists; only then may
`SOL_PROGRESS.md` say LIVE_VALIDATED for that row.

All commands run from `c:\Users\aaron\TradingBotV3` in PowerShell.

---

## A. Writer roles (do once per day-type, on the machine in question)

Only ONE machine is the designated writer on a given day.

**Live-trading day (desktop publishes):**

On DESKTOP-IABHR62:
```
.venv\Scripts\python.exe scripts\writer_role.py --designate-self
```
On MainPC (the mini-PC):
```
.venv\Scripts\python.exe scripts\writer_role.py --secondary DESKTOP-IABHR62
```

**Away day (mini-PC publishes):** the mirror image —
`--designate-self` on MainPC, `--secondary <MINI-PC-HOSTNAME>` on the desktop.
(Get the exact hostname the app uses on each machine with
`.venv\Scripts\python.exe -c "import socket; print(socket.gethostname())"`.)

**Verify on BOTH machines** (status only, writes nothing):
```
.venv\Scripts\python.exe scripts\writer_role.py
```
- PASS: the writer machine prints `role: designated_writer` and **exit code 0**;
  the other prints `role: secondary`, names the writer, and exits **1**.
- FAIL: both print `designated_writer` (they would compete), or the intended
  writer exits non-zero (no phone report would publish).
- The non-zero exit on a non-publishing machine is deliberate — script it into
  any away-day preflight so a forgotten switch is caught before the session.

Restart the GUI after changing a role: the role is read at publish time, but a
restart also starts the review-events v2 shard (section C).

## B. Pre-session (desktop, before open)

1. Branch/commit: `git log --oneline -1` — expect the pushed head you intend
   to run. Worktree clean: `git status --short` prints nothing.
2. Deterministic gate: `.venv\Scripts\python.exe scripts\smoke_check.py` → 7/7.
3. Read-only health: `.venv\Scripts\python.exe scripts\operations_audit.py --no-write`
   - Expect UNHEALTHY overall *before* the first GUI start of the day (stale
     heartbeat etc.) — what matters here is the writer rows match section A and
     no `rollover FAILED` text appears anywhere.
4. Note the current shadow-log sizes and mtimes (for section D's comparison):
   ```
   Get-Item $env:LOCALAPPDATA\TradingBotV3\diagnostics\spy_state_shadow.jsonl,
            $env:LOCALAPPDATA\TradingBotV3\diagnostics\greatness_shadow.jsonl |
     Select-Object Name,Length,LastWriteTime
   ```
5. Confirm no stray TradingBot/scanner process is running (Task Manager, or
   the owned-process row on the Health page once the GUI is up).

## C. Review-events v2 shards (first restarted session on each machine)

After the GUI restart, make one review action in Alert Center (e.g. skip an
alert), then check:
```
Get-ChildItem "G:\My Drive\Trading\TradingBot\alert_review_events" -ErrorAction SilentlyContinue
```
- PASS: a shard file exists whose name embeds THIS machine's installation id,
  and it grows with review actions. The legacy
  `alert_review_events.jsonl` byte count **never changes again** (read-only:
  was 270,476 bytes / 491 lines on 2026-07-30).
- PASS (merged reads): the Review/learning surfaces still show pre-migration
  history alongside new events.
- Repeat on the mini-PC on its next Away day.
- FAIL: no shard after a review action, or the legacy file's size changes.

## D. The first session rollover (the W08/W09 live gate)

The rollover fires on the first shadow evaluation of the NEXT session (or
after a config change) — i.e. typically at the next trading day's first scan
cycle, including after an overnight-down restart. After it fires, on the
desktop:

1. **Archives exist and raw bytes moved, not copied:**
   ```
   Get-ChildItem $env:LOCALAPPDATA\TradingBotV3\diagnostics\shadow_evidence -Recurse |
     Select-Object FullName,Length
   ```
   - PASS: `spy_state_shadow\raw\*.jsonl` and `greatness_shadow\raw\*.jsonl`
     hold the PRIOR sessions' rows; the active logs contain ONLY the new
     session (small files, new mtime).
2. **Summaries + checksums:** each `summaries\*.json` has
   `raw_archive_sha256` and `summary_id`; the operations audit's shadow rows
   reconcile them automatically — run
   `.venv\Scripts\python.exe scripts\operations_audit.py --no-write` and
   confirm NO "do not reconcile", "checksum differs", or "rollover FAILED"
   text in either shadow row.
3. **Counters carried:** the summary for yesterday's date has
   `coverage_present: true` and its `coverage.evaluations` matches what the
   sidecar showed at yesterday's close. Backfilled older sessions will say
   `coverage_present: false` / "coverage counters were unavailable at
   finalization" — **expected, report as INCOMPLETE, not as failure.**
4. **Eligibility honesty:** expected counts for the FIRST rollover on this
   data: SPY 0 eligible (all legacy rows are v2 without `complete_bar_ts`;
   eligibility begins with the first full v4 session), Greatness ≥1 eligible.
   Do not "fix" these numbers; they are the honest baseline.
5. **Failure path (only if a `rollover FAILED` row appears):** recording is
   paused, evidence is preserved, and every later call retries — capture the
   status sidecar's `rollover_failure` block and stop there; do not delete or
   edit any log.

## E. During / after the session (plan.md sec 6.1)

- Glance at heartbeat age on the Health page at open, midmorning, midday,
  late day; confirm `current_job`/`next_job` stay credible.
- After the normal GUI close: owned process/thread row returns to zero; no
  orphaned scanner (sec 6.1 post-session items 2-3).
- Preserve, do not prune: run manifests, heartbeat, job ledger, both shadow
  logs + new `shadow_evidence\` tree, verified Away report + `.meta.json`.
- Record observations only. Do NOT tune thresholds from one session
  (plan.md sec 6.1).

## F. Two-machine / destructive drills (outside market hours ONLY)

Held until sections A-E pass at least once. The matrix is plan.md sec 6.2
rows 1-10; the lease/writer rows (5-8) additionally need both machines and
real Drive sync. Do not run any of these during a session.

---

## Result record

| # | Check | Date | PASS/FAIL | Evidence (path/value) |
|---|-------|------|-----------|-----------------------|
| A | Writer roles verified on both machines | | | |
| C | v2 shard started (desktop) | | | |
| C | v2 shard started (mini-PC) | | | |
| D1 | Raw archives present, active logs clean | | | |
| D2 | Summaries reconcile, no rollover failure | | | |
| D3 | Prior-session counters carried | | | |
| D4 | Eligibility counts honest (SPY 0, Greatness ≥1) | | | |
| E | Clean teardown, artifacts preserved | | | |
