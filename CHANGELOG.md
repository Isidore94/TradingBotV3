# TradingBotV3 implemented history

Last reconciled: **2026-08-15** from the working copy of
`phase05-r2-focus-gating-strength-board` (cut from `phase05-r1-auto-modes-quiet-hours`,
itself branched from `testing-week-2026-08-10`)

Authoritative for: **what exists and the historical sequence of revisions**

Remaining work: [`plan.md`](plan.md)

This is a curated product history, not a raw commit dump. It reconciles the former
status sections in `plan.md`, the accumulated `CURRENT_CHECKPOINT.md` ledger, the GUI
plans, warehouse plans/reviews, dated handoffs, and Git history. Exact current test
counts remain in `CURRENT_CHECKPOINT.md`.

The labels retain their strict meanings: `IMPLEMENTED` means code exists, `GREEN`
means deterministic tests pass, `LIVE_VALIDATED` requires real-session evidence,
and `PROMOTED` requires an explicit champion decision. A feature can be implemented
and green while its live or promotion gate remains open in `plan.md`.

## Current implemented inventory

### Application, runtime, and data ownership

- PySide6 Trading Desk launched by `launch_gui.py`, with the legacy Tk UI retained
  as a compatibility path.
- Main-desk single-process ownership, bounded BounceBot startup/shutdown, generation
  guards, child-process reaping, runtime heartbeat, durable job ledger, typed retry
  budgets, stale-run marking, and a hardened single-instance launch guard that also
  sees the frozen executable.
- User-selected shared home folder for operational text/JSONL/CSV artifacts;
  machine-local settings, caches, and diagnostics under LocalAppData; a separate
  research-lake storage class outside that home folder.
- **No cloud sync (2026-08-10, decision 0015).** Google Drive/OneDrive were removed
  from the system entirely. `C:\TradingBotData` keeps its path and role as a plain
  local folder; the DAS file server `\\MINI-PC\Trading Bot Data` is the durable
  tier, holding the research lake, the AI store, and hourly cold-pushed subtrees.
  Documentation-only change: no path, behavior, or test changed.
- Designated-writer authority, local kernel exclusion, fenced writer lease, atomic
  publication, readback verification, last-good preservation, and bounded archives.
- Main desk is the sole always-on scanner. The former mini-PC scanner and Desk Link
  satellite topology are `RETIRED`; their code remains only pending cleanup.

### Scanning, candidates, and decision support

- Master AVWAP D1 swing scanning with earnings anchors, current/previous AVWAP
  families, running-deviation bands, focus buckets, Expected-R ranking, study tags,
  theta candidates, tracker history, and durable daily-bar storage.
- BounceBot completed-M5 detection with session VWAP/bands, EMA and prior-day
  levels, relative strength/weakness, regime-aware candidate discovery, tiering,
  alerts, outcome tracking, and the day-scoped M5 Focus path.
- BounceBot's sweep runs only inside the session window (open-30m to close+30m by
  default, weekdays); outside it Auto Pilot pauses scanning and holds the IB
  connection open. A manual resume survives until the next boundary.
- CandidateRegistry foundation with provenance, source leases, transitions, atomic
  versioned persistence, and partial shadow adoption. Full authority remains open.
- Industry Board with one single-flight owner, hourly refresh, atomic last-good
  snapshot, numeric sorting, freshness/Health integration, and advisory aligned
  industry-vs-SPY plus stock-vs-primary-industry fields.
- Auto-populate rules for both regimes, previous-day-extreme gating, DESK adoption
  into M5 Focus, and one extension notification per Focus name/day while pullback
  notifications stay active.
- Focus privileges begin only beyond the previous session's directional extreme;
  missing prior-day data grants nothing.
- D1 Focus routes final Favorite/High Conviction upgrades while developing trigger
  evidence remains research-only. Legacy D1 champion alerts are unchanged.

### Charts, review, alerts, and phone surfaces

- Chart-first review flow, current forming D1 preview, D1/M5 shared snapshot widget,
  log scale, crosshair/OHLCV readout, source/age strip, fallback warning, cache
  invalidation, background loading, prewarming, and stall watchdog.
- Chart Review workspace with lookup for any symbol, hidden-by-default Setups drawer,
  keyboard-first LIKE/veto/note/setup-claim capture, versioned veto vocabulary,
  append-only `trader_annotations.jsonl`, and isolated forward veto cohorts.
- Painted D1 S/R, previous-day H/L, projected trendline, SMA/EMA/AVWAP groups,
  machine-local visibility preferences, stable level IDs, click selection, and
  click-to-arm routed through the one `PriceAlertService` writer.
- Chart Review annotations cannot add Focus/watchlist membership or price alerts;
  LIKE records judgement only.
- Visual Alert Center and review queue, chart-armed watches, persistent History,
  structured review decisions, review scoreboard, and annotation-only/FIFO policy
  gate.
- Main-only price-level polling with cross-up/cross-down, one fire per arm, urgent
  ntfy push, persistent main-desk presentation, and manual re-arm.
- Auto modes OFF/DESK/AWAY/EVENING, honest global status, EVENING early scan and
  briefing, and one verified `autopilot_today.txt` with safety/freshness first,
  numbered best swings, intraday candidates, and condensed operations.
- The double-click symbol snapshot popup opens at desk height (2026-08-11): its size
  is taken from the hosting window's frame, or the screen's available area when the
  window is not yet measurable, never smaller than the former fixed 1180x760, and is
  centered on the desk window and clamped inside the screen. Opening geometry only —
  a trader resize survives subsequent double-clicks.
- On 2026-08-10, best swings gained an ntfy report notification; it stays quiet when
  the generated swing section contains no readable setups. Late-opened alerts now
  receive current bars, and the Chart Review Setups column defaults hidden with a
  visible restore control.
- Phone push policy, 2026-08-11 (trader rule): **AWAY is the only mode that pushes**,
  and the Research/Focus price alerts are the single deliberate exception — they keep
  their own always-on urgent channel, unchanged. The EVENING morning-briefing push and
  the retired Desk Link control-reclaim push are now silent outside AWAY; both still
  announce on the desk. The hourly swing push carries the **full favorite and
  high-conviction roster** under the ranked picks, built from the whole current feed
  rather than the top-ten slice, side-split, with `near` excluded and an explicit
  "did not fit" marker if the message ever exceeds the ntfy size ceiling; a roster with
  no ranked picks still sends. A **second hourly push names every stock that fired a D1
  level or event alert since the previous one** (armed D1 levels, D1 event watches,
  Focus D1 flags, and the scanner's ready D1 focus alerts), new-since-last-push rather
  than cumulative, silent on an empty hour, and cleared only on a delivered push so an
  ntfy failure never eats the events. The Alert Center classifies (it owns the D1
  routing rules) and Auto Pilot aggregates and gates, so the phone and the D1 Focus
  feed cannot disagree. Machine-local kill switches: `push_away_swings`,
  `push_away_d1_events`. **Extended 2026-08-14 (packet R1):** EVENING's SPY ±1%
  wake alarm is the *second* deliberate exception — urgent, repeating every five
  minutes while the move holds, stopping on the flip out of EVENING, kill switch
  `push_evening_spy_alarm`.
- Auto-mode matrix, 2026-08-14 (packet R1): discovery is identical in every mode;
  what differs is who is present to act. DESK adopts staged picks immediately;
  AWAY stages and never adopts and queues alerts silently (only the sound is
  suppressed); EVENING runs its early block and then stops scanning entirely,
  staging picks for the wake-up flip; OFF is the only mode that still
  self-applies. Quiet hours confine every automatic starter to weekdays,
  06:00–14:00 local; manual buttons are never gated.
- One Master AVWAP scan action, 2026-08-15 (packet R1). The Shared/Local pair read
  the identical two watchlist files, so `use_shared_watchlists` and the menu choice
  it drove were removed across thirteen files. Cloud-drive *store discovery* went
  with it (decision 0015 amendment); the mount-presence guard stays.

### Journal, explanations, and learning

- Journal schema v2 with append-only opportunity lifecycle events, idempotent broker
  Taken/Closed imports, structured reviews, free-form notes, tags, and analytics.
- Deterministic novice explanations across Setup Tracker, Day Trade Tracker, and
  Move Forensics, plus an evidence-floor-aware “What’s Working” summary.
- Review events partitioned by installation, merged/deduplicated by readers, capture
  audits, preference scoreboard, AI-curated `review_policy.json`, and a permanent
  no-suppression boundary.
- Technical Integrity research hierarchy with point-in-time predictions/outcomes,
  break pressure, calibration report, and no detector/watchlist/alert influence.
- Regime infrastructure evidence for SPY baseline, breadth, Technical Integrity
  follow-ups, and audit tooling. The evidence remains exploratory/non-promotable.

### AI and automation

- Provider-neutral A.I. Summary workspace for OpenAI and Anthropic, explicit evidence
  selection, bounded preview, credential-manager storage, structured/source
  validation, immutable evidence packages, and export-only results.
- Config-gated local OpenAI-compatible provider through Ollama, default off; small and
  medium model tiers verified on the Ryzen main desk with no market-hours inference.
  The local large tier is `RETIRED` (2026-08-10): 27B-class models no longer load
  beside the running desk on the 780M, so its jobs belong to the frontier model. Local
  calls are capped to the tier's context window and fail loudly on server-side prompt
  truncation.
- Separate off-hours `ai_jobs` process and scheduled task, job-ledger integration,
  deterministic evidence coverage, daily advisory summary, per-ticker briefs, full
  artifacts in `ai_store`, and bounded atomic `ai_morning_brief.txt` publication.
- Per-ticker briefs project each symbol out of a full-size base package and then
  ration the projection to the local context window; ticker-roster and bare-name
  lines are discarded as non-evidence, each symbol resolves independently, a symbol
  with no evidence beyond watchlist membership is answered without a model call,
  completions resume on a read-stamp-independent evidence key, the morning file is
  republished after every resolved symbol, and the slot spends at most three attempts
  a session.
- Local-AI Phase 0 is complete. Phase 1 implementation is complete; its five-session
  unattended live gate remains in `plan.md`.

### Durability and catch-up

- Repeating 06:00 Pacific weekday launch task through the session, protected by the
  existing single-instance guard.
- Master AVWAP tracker staleness catch-up from completed prior-session D1 data with
  explicit `data_session` vintage and no automatic scoring-tuner/prior-refit side
  effects.
- Technical Integrity follow-up and breadth-ledger deterministic backfill with
  bounded retries, explicit `capture_mode`, honest gap rows, and live/backfill audit
  separation.
- Frozen snapshots, never-started predictions, and other Tier-C evidence remain
  intentionally non-reconstructed.

### Research warehouse

- Phase 0: research-lake decision record, configuration, home-folder-path refusal, layout,
  and disabled-by-default no-op behavior.
- Phase 1: immutable Parquet store, four-step seal, append-only manifest authority,
  13 frozen schemas, quarantine, compaction, retirement, and crash reconciliation.
- Phase 2: idempotent bronze wraps, daily universe/level snapshots, and completed D1
  projection with source hashes and watermarks.
- Phase 3/3b: zero-extra-request M5 tee, coverage/gap rows, capped spool, capture-only
  pacer, IB backfill transport, nightly/weekly backfill, and trickled yfinance seed.
- Phase 4: versioned XNYS sessions and deterministic M15/M30/H1/W1 aggregation.
- Phase 5: point-in-time daily/intraday feature snapshots and anchor instances using
  champion calculations, including AVWAP parity at 1e-9.
- Phase 6: deterministic occurrence/revision/episode identity and versioned swing and
  intraday outcome simulation with costs, ambiguity bounds, partials, time stops,
  slippage, and open/truncated states.
- Phase 7: manifest-resolved read path and read-only Research panel; DuckDB remains
  optional and pyarrow can answer every slice.
- Phase 8: three-class backups, restore check, single-flight build/status CLI, job
  ledger, and six Health tiles.
- Defect passes repaired outcome supersession, management bounds, feature windows,
  per-bar backfill dedupe, pacing clocks, gap semantics, session identity, compaction
  reads, every job invoker, live tee wiring, and off-GUI-thread spool I/O.
- Phases 0–8 are code-complete on the testing-week branch. The broker check,
  confirmation items, and 20-session pilot remain open.

### Testing, packaging, and platform

- Broad pytest suite, deterministic smoke check, pytest markers, narrow Ruff gates,
  layered requirements with constraints, and Windows/macOS path handling.
- Provider telemetry at IBKR/Yahoo/Nasdaq boundaries with completeness contracts and
  honest UNKNOWN until measured.
- PyInstaller onedir spec, Qt runtime hook, asset/package drift test, lazy-engine
  `--selftest`, and a permanent guard preventing self-test from demanding packages
  deliberately excluded from the bundle.
- The first Windows frozen run found and closed an `ai_jobs` bundle-roster conflict;
  the current frozen self-test is 29/29.
- macOS launcher, CloudStorage Drive discovery, Keychain credentials, and machine-
  local path normalization.

### Shadow challengers

- Side-symmetric SPY market-state/pullback engine runs beside the legacy pause
  detector, emits replayable evidence, and cannot affect candidates, alerts, or rank.
- Greatness Monitor persists ordered touch/wick/close/acceptance/retest/failure/re-arm
  transitions beside legacy D1 alerts and cannot alter the champion path.
- Champion-invariance tests prove enabled, failing, or poisoned shadow engines leave
  production SPY/D1 results unchanged.

Neither challenger is promoted. Their remaining evidence gates are in `plan.md`.

## Revision history

### 2026-08-15 — a Testing Plan tab, and the packaging trap it nearly walked into

`IMPLEMENTED` + `GREEN`. Documentation-and-viewer work; no engine touched.

`docs/DESK_TESTING_PLAN.md` is a plain-language runbook of the current testing
sequence — tonight's quiet-boot check, Monday's six live proofs, then the
after-close checklist, rebuild and merge. Every step gives when to do it, what
to click, the exact log line or screen element that means it worked, what bad
looks like, and what to copy to the AI if it fails. It restates
`CURRENT_CHECKPOINT.md`'s owed proofs for a human reader and carries a header
line requiring it to be updated in the same pass whenever those change.

Settings gains a **Testing Plan** tab rendering that file read-only, with a
Refresh button and the file's last-modified time. It owns no timer, writes no
state and touches no engine. A missing, unreadable or empty file says so
plainly and shows **no cached copy** — a stale runbook read as current would
have the trader checking for log lines the build no longer prints.

**The packaging trap.** The plan is a runtime asset living **outside
`scripts/`**, and nothing existing would have caught that. The spec's
package-asset sweep only mirrors files inside `FIRST_PARTY_PACKAGES`, and
`test_packaging_spec_drift.py` only walks `scripts/`. The frozen desk would
have shipped showing "plan file not found" on the one page the trader opens
when nothing else is behaving — the same shape as the two defects that reached
the desk before (the `ai_jobs` roster clash and the `-c` scan spawn). Now
guarded three ways: an explicit `datas` rule with a hard `SystemExit` if the
file is absent at build time, a new selftest asset check, and a test asserting
the spec rule still exists. The view resolves through `sys._MEIPASS` when
frozen, since a frozen build has no `scripts/` tree to walk up from.

The frozen selftest is therefore **31/31**, not 30/30 — the new check is the
31st, and it was verified by an actual rebuild rather than assumed.

### 2026-08-15 — packet R2: the M5 Focus adoption gate and the strength board

`IMPLEMENTED` + `GREEN`; **not** `LIVE_VALIDATED` — the four live proofs in
`docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` §8 are owed. Built on branch
`phase05-r2-focus-gating-strength-board` (cut from R1, R1.1 merged forward) on
the trader's second explicit redirect ahead of P0.7.

**One gate, three call sites.** `scripts/focus_adoption_gate.py` holds the
combined rule — beyond yesterday's extreme AND on the right side of session
VWAP — and it runs at candidate build, at every staging refresh, and again at
adoption. Session VWAP comes from `chart_snapshot.session_vwap_series` over the
completed bars of the current session, carried on the profile as
`completed_session_vwap`; BounceBot's dynamic and EOD VWAP are deliberately not
used because they blend prior sessions and answer a different question.

The gate reads `last_complete`, not `last`. The previous filter measured the
prev-day break on the **forming** bar, so a pick could be admitted on a break
that bar then closed back inside. The golden fixture landed first and recorded
the effect: five of its seven baseline survivors now fail, each for a stated
reason, and nothing that failed the baseline now passes — the gate only
narrows.

**The queue is re-checked, not trusted.** Each 30-minute refresh re-runs the
gate over everything already staged: picks that have fallen back are evicted
with a logged reason, survivors are re-stamped. An evicted pick may re-propose
the same day if it re-qualifies — the queue says what qualifies now, not what
once did, and a name that pulled back at 10:00 and broke out at 13:00 is the
setup rather than the noise.

**Adoption reads a stored verdict** rather than measuring its own: the Alert
Center runs on the GUI thread and a staged pick is on no watchlist, so
BounceBot holds no bars for it. Failing, missing and >45-minute-old verdicts
are all refused, and a refusal does not mark the pick seen, so a stale verdict
costs one cycle rather than the pick. **This closes the stale-drain gap R1
recorded as an accepted limitation.**

**Focus entries now have an origin.** `focus_auto_picks.json` rides beside the
plain-text focus files (which do not change — the trader edits them by hand)
and marks only what the machine adopted. Absence of a marker reads as
user-entered, so every pre-R2 file is protected by default and a lost or
corrupt sidecar fails toward "the trader owns it". This is what makes
"user-entered names are never automatically removed" structural instead of
aspirational: without a per-entry origin, no removal verb could be written
safely at all.

**"Not today" says which of its two jobs it will do.** On an auto-adopted entry
it reads `✕ Not today - drop pick` and removes that one M5 entry on that one
side; on anything else it keeps its quieter feed-only meaning. The verdict is
recorded as `not_today`, not `dislike` — a same-day pass is not "this name is
bad", and the review-learning scoreboard must not learn that.

**The triple-VWAP desync is repaired by request, not by a second writer.**
BounceBot cuts a watchlist line on a worker thread, so it files a day-scoped
request and the Alert Center's existing poll performs the removal — one owner
per mutable store. The machine's own pick is removed; a name the trader typed
is left alone and the mismatch surfaced, because deleting it would break the
invariant and keeping it quietly would leave them trusting an entry nothing
scans.

**The strength board.** `scripts/strength_scan.py` implements the trader's
TC2000 formula (12-bar body sum × price level ÷ ATR50) with hand-computed
fixtures, plus the percentile cut and the VWAP/15EMA/prev-extreme filters. It
does not touch `real_relative_strength`; the existing RS/RW board is unchanged
beside it. `StrengthBoardService` owns one single-flight refresh on R1's
quiet-hours window and keeps the last good board through a failure;
`StrengthBoardPanel` is a new top-level desk page next to Focus Picks, with
per-row and side-aware adds that re-run the adoption gate at click time and
name any refusal with its reason.

**Transport measured before the cadence was chosen** (spec §10): 27.6 s for all
1,506 symbols at `period=5d`, 100% carrying ≥50 bars. `5d` rather than `1d`
because ATR50 and C50 need 51 completed bars and a `1d` window holds six at
07:00 PT — every symbol would be unmeasurable through the first four hours of
the session being traded. Zero IB traffic, so the locked pacing budget is
untouched. The 15-minute default stands with wide margin; the number was taken
on a Saturday and is recorded as a floor, to be re-measured live.

### 2026-08-15 — R1.1: the repair pass an independent review demanded

`IMPLEMENTED` + `GREEN`. Five code-verified defects from the R1 review, two of
them blockers that would have made the owed live proofs fail as written.

**The boot gate was cosmetic.** Quiet hours gated `AutopilotService.__init__`'s
resume but not `_ensure_bot_running`, which the tick calls every 30 seconds — so
a 21:00 launch logged "nothing starts yet" and connected BounceBot to IB half a
minute later. The gate now lives inside `_ensure_bot_running`, the one place
automation starts the bot, with `force=True` as the manual carve-out that
`force_reconnect` passes. The original test missed it by stopping the timer
before a tick could run; the new one runs a real tick with the clock frozen to a
weekday 21:00.

**The SPY alarm read yesterday's tape.** `_spy_session_bars` calls the last
cached bar's date "today", and the sweep is paused overnight, so on an Evening
morning after a ±1% day the cache still held that move — and the quiet window
opens 30 minutes before the bell. The alarm now refuses a series whose last bar
predates the day it is asked about. Roughly seven false urgent wake-ups per such
morning, none of them ever sent, because the alarm had not yet run live.

**A post-window relaunch silently cancelled the after-close wrap-up.** The quiet
refusal in `_maybe_run_swing_slot` returned before any slot resolution, so slots
still pending after the window closed — a crash, or the 4h39m machine sleep this
desk had on 2026-08-11 — stayed pending forever and `after_close_wrapup_due`
never fired. Slots are now resolved once the window closes, on the same
reasoning as Evening's refused slots. Before the window opens nothing is
resolved: those slots are still going to run.

**EVENING adopted picks immediately**, against the spec, the CLAUDE.md matrix,
the runbook and the CHANGELOG, all of which said it stages until the DESK flip.
It now refuses like AWAY, and stops beeping too — closing the spec §1 alert cell
the R1 build had left unimplemented. The trader is asleep; the SPY alarm is
EVENING's deliberate wake channel.

**The legacy Tk GUI died at construction.** Removing `get_shared_watchlist_paths`
with the shared/local vocabulary left `master_avwap_lib.gui` — which copies
legacy's globals wholesale — raising NameError. Invisible to the suite, which
imports these modules but never constructs them, and to the import-only frozen
self-test. New `tests/test_module_globals_resolve.py` statically resolves every
global that four never-constructed legacy modules read; it was verified to fail
on the un-fixed file before the fix went back in.

Hardening in the same pass: a NaN threshold no longer bypasses the alarm's
threshold test; the quiet-window ⊇ sweep-window containment is now structural
(`auto_scanning_window` widens itself to contain `bouncebot_scan_window`, so two
independent settings keys cannot be configured into contradiction);
`autopilot_auto_arm_due` takes `quiet_hours` so its test no longer depends on a
machine-local setting; the launch self-heal gate and the D1-feed beep site gained
coverage; Qt tests skip rather than silently pass without PySide6; and the
"an early close moves this window" docstring claim is corrected — no early-close
modelling exists anywhere, which is pre-existing and fail-open.

Recorded, not fixed: a corrupt `local_settings.json` still silently re-homes the
store to `%LOCALAPPDATA%`. And whether EVENING should also pause the BounceBot
sweep is now an explicit open question in the spec's new §9 rather than a
silently unbuilt matrix cell — pausing it would also stop the alert stream the
same matrix says EVENING should queue, and remove the prices the strength checks
read, so the build implemented the unambiguous cell and left this one.

### 2026-08-15 — packet R1: quiet hours, the auto-mode matrix, and one scan

`IMPLEMENTED` + `GREEN`; **not** `LIVE_VALIDATED` — the four live proofs in
`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` §6 are owed. Built on branch
`phase05-r1-auto-modes-quiet-hours` after the trader directed R1 ahead of the
P0.7 merge gate. Two commits: the behaviour, then the removal.

**Quiet hours.** `autopilot_core.auto_scanning_window` / `auto_scanning_due`
mirror the proven `bouncebot_scanning_due` pattern — pure window, reason string,
weekend refusal, settings override, fail-open on an unanswerable session lookup —
and now gate every automatic starter: the launch `_self_heal_universe` (which
fired 2.5 s after launch and every 30 minutes with no clock check at all), the
tick-loop universe heal, the boot resume that connected BounceBot to IB whenever
Auto was left ON, the daily 07:00 self-arm (previously unbounded above, so a
21:00 launch self-armed against a closed tape), the open watchlist build, and the
swing slots. Manual work is gated nowhere; `force=True` is the carve-out.
The window is 06:00–14:00 on a normal session and is deliberately a **superset**
of the BounceBot scan window — a literal open-at-06:30 gate would refuse the IB
connect at 06:10 while the sweep window said the sweep could run.

**EVENING prepares the morning and stops.** The open+30 early slot, the
07:00/07:15/07:30 strength checks and the briefing still run; every ordinary
hourly slot is refused and named once in the log, and the open self-build is
skipped without a sticky marker so the wake-up flip to DESK can still build.
Refused slots are marked *done* rather than left pending, because
`after_close_wrapup_due` requires every slot to be done and pending ones would
have silently cancelled the after-close wrap-up for the day.

**AWAY queues instead of adopting.** Auto-populate picks now stage in AWAY as
they already did in DESK and EVENING; OFF is the only remaining self-applying
mode. This reverses the 2026-08-05 rule for AWAY on the trader's reasoning:
nobody is present to *prune*, so a pick applied at 09:00 alerted unwatched all
day. The Alert Center refuses adoption while AWAY and marks nothing seen, so the
whole day drains on the flip to DESK — the same point where packet R2 will add
its freshness re-check. Alerts arrive without a sound; feed, history and the D1
unread badge keep filling, and an unreadable mode file resolves to OFF so a
broken read can never silence the desk.

**EVENING's SPY ±1% wake alarm** — the second deliberate exception to the
AWAY-only push rule, after the always-on price alerts. Reads the champion cached
SPY bars, repeats every five minutes while the condition holds, stops on the flip
out of EVENING, day-rolls its stamp in the Auto Pilot state file, and stamps only
a delivered push. NaN is refused explicitly, because `nan < threshold` is False
and no data at all would otherwise read as a 1% move. Kill switch
`push_evening_spy_alarm`, threshold override `push_evening_spy_alarm_pct`.

**One scan, not two.** `use_shared_watchlists` is gone from thirteen files and
one run-manifest counter: both branches resolved to `(LONGS_FILE, SHORTS_FILE)`
under the same label, so the Shared/Local choice the menu offered was never a
choice. `ScanService.run_shared_watchlist_scan`/`run_local_watchlist_scan`
collapse to `run_watchlist_scan`; the menu pair and the Ctrl+R "Run Shared Scan"
action become one "Run Scan"; the `run_master_with_shared_watchlists` alias is
gone; the scheduler's "shared-watchlist" wording and the false "local project
watchlists" label are gone. The manifest counter had no consumer — checked before
deleting. The job-ledger `config_hash` stays `"shared-v1"` deliberately: an
opaque idempotency token, not user-facing text.

**Cloud-drive store discovery removed.** `project_paths` probed `$GOOGLE_DRIVE`,
`~/My Drive`, `~/Google Drive` and the macOS CloudStorage accounts *at import*
and adopted the first writable one as the operational store whenever
`shared_data_dir` was unset — inert on this desk only because that setting
happens to be set, and directly against decision 0015. The fallback is now
plainly local. The mount-presence guard is kept and renamed
`_wait_for_shared_store` (decision 0015 blessed its macOS half), with the sync-
client instructions removed from its messages. No path moved: the desk still
resolves `C:\TradingBotData` from `local_config`. Decision 0015 carries a dated
amendment explaining which half was harmless and which was not.

Thirty-five new tests in `tests/test_auto_quiet_hours_and_modes.py`. Four
existing tests asserted behaviour this packet reverses and were updated to the
new rule rather than worked around; two more were inverted to prove cloud-drive
discovery can no longer happen.

### 2026-08-15 — trader refinement packets promoted; after-close mechanism identified

Documentation-only pass; no code, path, or test changed. The trader promoted the
2026-08-14 wishlist entries into `plan.md` **Phase 0.5** (packets R1–R6, ranked
R1 auto-modes/quiet-hours first, R2 Focus-gating/strength-board second) with five
ACTIVE specifications added under `docs/` and indexed in `docs/README.md`. Code
for these packets starts only after P0.7 merges.

Two findings from the read-only recon are recorded as knowledge about implemented
behavior:

- **Why Master AVWAP setups "totally change after the close."** The live scan
  applies no completed-bar guard to the daily frame, so AVWAP/sigma bands, ATR20,
  binary bounce/cross gating, two candle-shape score penalties, and an 18-point
  volume-thrust bonus (whose full-day-average denominator makes it structurally
  near-unfireable intraday) all move with today's forming D1 bar; the setup
  tracker is then written at 12:00 PT and wiped/rewritten when the 13:00 slot
  finishes ~13:20-13:28. Full mechanism list with evidence:
  `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md` §4. The trader authorized the
  "full honesty bundle" fix design; nothing is built yet.
- **"Shared scan" is a proven no-op**: `use_shared_watchlists=True/False` resolve
  to the identical watchlist paths, so its removal (Phase 0.5 R1) is a dead-flag
  cleanup, not a behavior change. **Removed 2026-08-15** — see below.

Running the frozen executable as the daily driver disabled the Master AVWAP D1
swing scan completely, for two sessions, without any visible symptom.

- **The defect.** `_run_master_scan_subprocess` spawned
  `[sys.executable, "-c", code]`. Under PyInstaller `sys.executable` is
  `TradingBotV3.exe`, so the flag and code string reached the application's own
  argument parser: `error: unrecognized arguments: -c import faulthandler; …`,
  exit code 2, **one second after each slot fired** against a scan that takes
  17-21 minutes. Every slot from 2026-08-12 07:30 through 2026-08-13 09:00
  failed; the last successful scan was 2026-08-11 13:23:59 (622 setup rows).
- **Why nobody saw it.** Everything that runs in-process was unaffected —
  BounceBot alerts fired, the 07:00 open scan rebuilt the watchlists, Auto Pilot
  wrote its reports — so the desk looked healthy. The cost surfaced one layer
  away: the overnight AI read 11 stale D1 sources and produced briefs about
  truncation.
- **The fix.** New `scripts/scan_worker.py` owns the scan invocation;
  `scan_service.scan_worker_command()` owns the transport, choosing
  `TradingBotV3.exe --run-scan <json payload>` when frozen and the unchanged
  `-c` form from source. `launch_gui.main` answers `--run-scan` before argparse,
  exactly where `--selftest` is handled. Both forms call `scan_worker.run`, so
  work and transport cannot drift apart. A malformed payload raises rather than
  defaulting — guessing would run a different scan than the one requested,
  including the setup-tracker write.
- **The guard that was missing.** `tests/test_scan_worker_spawn.py` really
  spawns a child process and waits for the completion marker, against a stub
  scanner so it stays offline. The spec-drift test inspects bundle *contents*
  and `--selftest` resolves *imports*; neither ever launched anything, which is
  why both passed while the desk could not scan. `scan_worker` is also added to
  the selftest's lazy-import roster.
- Verification: full Windows suite **2738 passed, 19 subtests**, exit 0; smoke
  **7/7**, exit 0. Eleven new tests.

### 2026-08-12 — first-night repair: roster noise, resume identity, crash-safe publish

The 2026-08-11 window was the ticker-briefs packet's owed live proof. It produced
126 briefs covering 101 of 182 symbols, never published a morning file, and exposed
three defects plus one machine fault. Advisory-only throughout: no detector,
scoring, or alert file is in the diff.

- **What the night actually did.** `ai_summary` succeeded first attempt at 22:02:53
  (~170 s, 10 usable sources) — a clean result against the previous night's six
  degraded rounds. `ticker_briefs` ran 22:04:33 → 01:20:08 with zero failures and
  was killed mid-batch. `ai_morning_brief.txt` still held the 2026-08-10 file.
- **TB-5 — a roster line is not evidence about the symbol.** `_extract_ticker_content`
  projected a text source by keeping every *line* containing the symbol, and the
  evidence files are human-readable reports full of copy-paste ticker blobs. Measured
  over the real 2026-08-11 packages: **307,630 of 319,687 projected chars (96.2%)
  were roster text**, median symbol-specific content **42 characters**, and
  `daily.master_events` contributed 174,994 roster chars against 479 chars of real
  content. Lines are now dropped when stripping ticker tokens and list punctuation
  leaves ≤15% residue, and when the line is the bare symbol (Auto Pilot's `longs`
  array is membership wearing a second hat). The residue test is deliberately not a
  ticker count: a tier row carrying eight tickers is pure signal. Measured effect on
  the same data — **166 model calls → 49**, projected payload 319,687 → 26,223 chars,
  and TB-2's membership-only skip now does what sec 6.4b scoped it to do.
- **TB-3 repaired — resume on the evidence, not on when it was read.** The manifest
  now carries a `resume_key` hashing only symbol, session, memberships, and source
  ids with their content. `evidence_hash` keeps its whole-package meaning for
  artifact identity, but it covers `generated_at` and every `as_of`, so it changed on
  every firing and the resume could never match. Manifest schema `v1` → `v2`; a row
  without a `resume_key` is regenerated, never reused.
- **Crash-safe publication.** The morning file is re-rendered and atomically
  republished after every resolved symbol, carrying an explicit
  "Run in progress at the time of writing" note that the final publish drops. A
  publish fault is logged and never costs the batch. The market-session block still
  suppresses publication outright — it is an unconditional stop for the whole job,
  and the last verified file stands.
- **Scheduled-task time limit was defeating its own concurrency guard.**
  `ExecutionTimeLimit` was `PT2H` against an 8-hour window. On 2026-08-11 the 22:00
  run was still briefing at 00:00, so Task Scheduler terminated its PowerShell parent
  and marked the task not-running, letting the 00:00 repetition start a **second**
  runner while the first instance's Python child continued. The session manifest
  records both: from 00:01:54 the rows interleave one-for-one, instance A continuing
  at list position 73 while instance B restarted from position 0, two 12B models
  resident on one iGPU, and 25 symbols briefed twice. Now `PT8H` — the window itself
  — in `scripts/register_ai_jobs_task.ps1` and on the live desk task.
- **Machine fault, not code (trader-owned).** The desk entered Modern Standby 60
  times during the window, 4h39m in total, including an unbroken 01:39:42 → 05:57:09.
  That killed the run and suppressed every task firing from 01:30 to 05:30.
- Verification: full Windows suite **2727 passed, 19 subtests**, exit 0; smoke
  **7/7**, exit 0. Seven new tests.

### 2026-08-11 — symbol snapshot popup opens at desk height

Trader ask: the chart popup that opens on a table double-click should use
essentially the full vertical space the rest of the program uses. It had opened at
a fixed 1180x760 regardless of monitor, so on the desk's screen the stacked D1 and
M5 charts were squeezed into roughly half the available height.

- `SymbolSnapshotDialog.__init__` now calls `_resize_to_desk_height()` instead of the
  hardcoded `resize(1180, 760)`. Height comes from the hosting window's
  `frameGeometry` (minus a title-bar allowance) when that window is visible, and from
  the screen's `availableGeometry` otherwise; it never falls below the old 760.
- The popup is centered horizontally on the desk window and clamped inside the
  screen's available area, so a multi-monitor desk cannot place it off-screen.
- Opening geometry only. The dialog is constructed once per panel and reused, so a
  manual resize persists across subsequent double-clicks within a session.
- Both charts already carry layout stretch 1, so the added height splits evenly
  between D1 and M5; no chart, data, or alert code was touched.
- Verification: full Windows suite **2687 passed, 19 subtests**, exit 0.

### 2026-08-11 — ticker-briefs hardening packet (TB-0..TB-4)

Armed by the trader after reading the first repaired overnight run
(`docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4b, PROPOSED → BUILT). Advisory-only:
nothing in this layer touches scanners, scores, watchlists, alerts, or bot state,
and no detector, scoring, or alert file is in the diff.

- **The measurement that changed the packet's premises.** `ticker_briefs` completed
  all 95 symbols in **5,962 s — ~63 s/call**, on the repaired `gemma3:12b-tbv3ctx`.
  The drafted premise of ~4.75 min/call and a window overrun is **obsolete**: there
  was no overrun. The real finding was content vacuity.
- **TB-0 — project first, budget second.** Every one of those 95 briefs was
  content-free. `run_ticker_briefs` built one base evidence package *already*
  budgeted to the local ceiling (22,000 chars) and projected each symbol out of that
  starved base, so the per-symbol-rich sources had been declared unfunded at 0 chars
  (`setups.current_tracker` 95,806, `setups.current_tiers` 77,124,
  `setups.bounce_learning` 17,995, `market.industry_intraday_rs` 17,833) and the
  funded tables sheared to about one row. MRVL's brief reads "1 of 19 requested
  source(s) usable", the one being its own watchlist membership. The base now carries
  the cloud ceiling so symbol rows survive projection, and the local budget is applied
  to each much smaller per-symbol package through `ai_summary.ration_projected_sources`
  — same unfunded/truncation vocabulary, same truncation tripwire on every local call.
  `run_daily_summary` is untouched and cloud payloads stay byte-identical.
- **TB-1 — per-ticker failure isolation and an honest partial morning file.** Each
  symbol's inference and export is its own unit, with the daily summary's single
  fed-back-error retry applied per symbol for the first time. The morning file
  publishes what completed and states `Briefed N of M. Failed: SYM (reason), …`
  before the first brief. Focus names lead the ordering, so a partial night covers
  Focus first. `ok` only when every symbol resolved; otherwise `degraded`, which the
  runner retries. A mid-batch window closure now publishes the partial instead of
  losing the night; the market session remains an unconditional stop, and the
  unreadable-watchlist refusal is unchanged.
- **TB-2 — membership-only symbols skip the model.** A symbol whose projected package
  holds nothing but `watchlists.membership` gets a deterministic one-line entry and no
  artifact set, and counts as resolved.
- **TB-3 — resumable completion.** Per-symbol completions are recorded in an
  append-only `ticker_briefs_manifest.jsonl` under
  `ai_store/briefs/<year>/<session>/`, keyed by `(session_date, symbol,
  evidence_hash)`. A re-fire regenerates only what changed, ending both the
  restart-at-symbol-1 waste and the duplicate four-file artifact sets; the morning
  file is re-rendered from the manifest, so clearing the failures upgrades `degraded`
  to `ok` on its own. An unreadable manifest regenerates rather than refusing.
- **TB-4 — per-session attempt cap.** `JobSlot.max_attempts` (3 for `ticker_briefs`,
  unlimited elsewhere) plus an identical-error early stop; on reaching either, the
  runner writes one terminal marker — an ordinary `skipped` row carrying
  `terminal: true`, deliberately not a new job status — and every later firing costs
  about a second. Only `failed` and `degraded_no_narrative` rows spend an attempt, so
  a cheap refusal from an unmounted share still self-heals, and `--force` overrides
  the marker. This ends the 11-consecutive-failure grind of 2026-08-09/10.
- Gate handling: separate five-session clocks. `ai_summary`'s clock continues; the
  `ticker_briefs` clock restarts at zero. Live proof owed at the next 22:00 window.
- **Testing-branch integration correction.** The first focused Windows gate after
  fast-forwarding the packet exposed that list evidence truncation measured retained
  rows before prepending its truthful truncation banner, allowing serialized source
  content to exceed the declared local character budget by the banner length. The
  truncator now includes the banner in its allocation. This is an evidence-packaging
  correction; detector, scoring, alert, and daily-summary call-site behavior remain
  unchanged.
- The full Windows gate also exposed a non-hermetic warehouse-tee assertion: it
  counted unrelated background `ResearchStore.open()` calls elsewhere in the pytest
  process although its contract concerns the capture object's own worker. The test
  now scopes the assertion to that worker; production warehouse behavior is
  unchanged.

### 2026-08-10 — testing-week usability and phone-report corrections

- Chart Review opens with its Setups column hidden and exposes a restore control.
- A newly opened alert receives current cached/fetched bars rather than scan-time bars.
- Best swing content can trigger a phone notification after report publication, with
  an explicit no-readable-setups quiet gate.
- The existing live market commentary journal request was recorded as roadmap item;
  it is not implemented.
- Consolidated repository guidance into implemented history, a phase-gated remaining
  roadmap, a precise current checkpoint, a classified documentation index, and a
  non-authoritative wishlist. `CLAUDE.md`/`AGENTS.md` now mandate the read/update
  sequence for every AI handoff.
- **Designated writer configured on the main desk.** `autopilot_today.txt` had not
  published since 2026-07-30 because the retired desktop was still the last recorded
  holder and no writer was named on the mini-PC; the lease correctly fail-closed
  rather than publishing from an unconfigured machine. Consequence: an entire Auto/Away
  session produced no phone digest and no swing push, since the push is tied to a
  *verified* publish. `writer_role.py --designate-self` fixed it.
- **Research warehouse enabled.** `research_store_dir` was unset, so a full session of
  capture was silently discarded. Now `\\MINI-PC\Trading Bot Data\research_lake`, with
  the sec-8.2 layout created and the machine-local spool at
  `%LOCALAPPDATA%\TradingBotV3\research_spool`.
- **Overnight AI jobs repaired.** Three independent faults, all found by reading the
  job ledger rather than the scheduler's hex code:
  (a) the task ran `pythonw.exe`, a GUI-subsystem binary, and exited `0xC0000142`
  with its stdout/stderr discarded — now a logged PowerShell wrapper
  (`scripts/run_ai_jobs.ps1`) over console `python.exe`, with the runner's real exit
  code propagated and both streams captured to `%LOCALAPPDATA%\TradingBotV3\logs\`;
  `register_ai_jobs_task.ps1` updated so re-registering cannot reintroduce it;
  (b) `ticker_briefs` had failed six consecutive nights with truncated JSON because
  the local server capped prompts at 2,048 tokens while the app sends up to 80,000
  chars of evidence — the medium tier now points at a derived `gemma3:12b-tbv3ctx`
  (`num_ctx 12288`), measured at 6,147 prompt tokens against 2,051 before;
  (c) after those failures the job then *skipped* every remaining run for reserving
  120 min against a shrinking window, so the ledger showed skips and hid the failures.
- **Local AI summarization made truthful about its own limits.** The evidence cap
  is now resolved per call site (`evidence_budget_for`): local calls use
  `ai_local_evidence_budget_chars` (default 22,000, derived from the 12288 context
  minus generation and scaffold, with headroom for the retry), while
  `MAX_TOTAL_EVIDENCE_CHARS` (80,000) stays the cloud ceiling — cloud request
  payloads remain byte-identical, test-asserted. A truncation tripwire compares the
  server's reported `usage.prompt_tokens` against what was sent and raises a named
  error instead of parsing output built on a sheared prompt; it is silent when the
  server omits usage, and raises rather than retries because a retry sends more.
  Token usage now reaches the job ledger for the daily summary and per-ticker slots.
  The local large tier is retired: 27B-class models no longer load beside the
  running desk on the 780M, so policy drafts and retros move to the frontier model.
  A Phase 2 design packet (sec 6.4a) is proposed and awaits trader sign-off; no
  digest schema was built or frozen.
- **Cloud sync removed from the system (decision 0015).** Google Drive/OneDrive are no
  longer part of the design. `C:\TradingBotData` is a plain local folder at the same
  path; the DAS `\\MINI-PC\Trading Bot Data` is the durable tier. Decisions 0005, 0006
  and 0014 carry superseded/amendment banners rather than being rewritten, since the
  mechanisms they justify still exist. Documentation and comments only — 2647 tests
  and 7/7 smoke unchanged. Known consequence: cloud sync was the only off-site copy of
  the Class A backup set, so off-site redundancy is now an explicit open gap.
- **BounceBot's intraday sweep is confined to the session window** (trader-directed,
  2026-08-10). Auto Pilot re-enabled scanning on every 30-second tick with no clock
  check of any kind, so a desk left running swept the watchlists — 95 to 150 names,
  measured at roughly eight full sweeps an hour — straight through the night and the
  weekend against prices frozen since the close. `bouncebot_scanning_due` now derives
  a window from `market_session` (session ± configurable 30-minute warm-up and
  wind-down; 06:00-13:30 on a normal Pacific session) and the sweep pauses outside it.
  The pause is `set_scanning_enabled(False)`, whose branch in the strategy loop skips
  `ensure_connected` and every symbol request, so the IB traffic stops while the
  connection stays up and the open needs no reconnect. Three details are deliberate:
  the check runs *before* the tick's weekend and Auto-Pilot-OFF short-circuits, which
  are the paths that would otherwise let a Friday sweep run all weekend; it acts only
  on a boundary transition, so a deliberate manual resume holds instead of being undone
  one tick later; and it fails **open**, because an unanswerable session lookup must
  never be the reason the bot sits out a trading day. Settings
  `qt_bouncebot_scan_session_only` (default on),
  `qt_bouncebot_scan_preopen_minutes`/`qt_bouncebot_scan_postclose_minutes` (default
  30). No detector, score, threshold, or alert rule changed — only *when* the existing
  scan runs. `SCAN_OUTSIDE_MARKET_HOURS` in `bounce_bot_lib/legacy.py` was left exactly
  as it was: it gates per-symbol bounce *detection*, not the data fetching that
  produces the traffic, so flipping it would have cost detection without saving requests.

### 2026-08-09 — testing-week integration, chart completion, and frozen proof

- Integrated chart performance, Chart Review capture, warehouse Phases 1–8 and
  defect repairs, A3 shared chart, A4 paint lines, A5 click-to-arm, Local-AI Phase 1
  completion, and capture-stream hardening.
- Added packaging spec-drift coverage and frozen self-test. The real Windows build
  exposed the excluded-`ai_jobs` contradiction; the roster and disjointness guard
  were corrected.
- Desk surveys confirmed 62/62 stored trendlines projectable/fresh and 0/171 red
  horizontal levels clearing the shared strength threshold across three symbols.
- Recorded the Windows desk gate: 2611 passed, 7 subtests, smoke 7/7, frozen 29/29.

### 2026-08-08 — single-main topology, durability, local AI, and captured judgement

- Retired the Desk Link/satellite and separate mini-PC operating roles; the Ryzen
  desk became the sole always-on scan and AI host.
- Built and repaired durability steps 1–4, including tracker-vintage honesty,
  bounded recovery, and frozen-process launch protection.
- Built Local-AI Phase 0 and the scheduled Phase 1 foundation, then hardened evidence
  budgets, missing-source reporting, session identity, and publication rules.
- Built Chart Review decision capture, veto vocabulary/cohorts, and the workspace
  shell; added chart background loading and stall protection.
- Reviewed/merged packaging work and recorded the testing-week branch.

### 2026-08-03 to 2026-08-04 — remote surfaces and research warehouse

- Consolidated Auto/Away output into one swing-first verified phone digest and added
  main-origin price alerts over ntfy.
- Implemented all three Desk Link tiers, then later retired the topology on 2026-08-08.
- Locked the Ultimate Setup Intelligence Database design and implemented warehouse
  Phases 0–8 plus two review/defect passes.
- Added the DAS research-lake storage class, immutable store, capture/aggregation,
  features, occurrences/outcomes, readout, backups, Health integration, and job
  invokers without production influence.

### 2026-07-30 to 2026-08-02 — observability, live controls, and platform support

- Finished Milestone-1 observability packets: champion-invariance guards, enforced
  fixture contracts, shared diagnostics I/O, shadow coverage/retention, writer
  coordination, honest Health, lifecycle ownership, review sharding, provider
  telemetry, and first-session runbooks.
- Added regime evidence collection, breadth ledger, Technical Integrity outcomes,
  stale-tail D1 recovery, and off-GUI Health work.
- Added DESK/AWAY/EVENING workflows, previous-day gates, chart watches, Focus review
  actions, phone price alerts, and the flagship post-earnings candle break.
- Added macOS setup, CloudStorage/Keychain support, machine-local path normalization,
  UI scaling, and the now-retired Desk Link implementation.

### 2026-07-22 to 2026-07-29 — chart-first desk and review learning

- Added broader RS/industry measurements, market internals, recalibrated Technical
  Integrity, and expanded Auto candidate coverage.
- Built D1/M5 snapshot charts across setups, RS, and industry surfaces, then added
  chart navigation, log scale, forming D1 preview, AVWAP/SMA overlays, and caching.
- Built the visual review queue, armed chart watches, D1 Focus toggles, alert dock,
  strength tape, and persistent D1 event alerts.
- Added review-event capture, preference scoreboard, AI policy handoff, annotation-
  only guidance, Focus strength board, and Phase-0 learning audits.
- Fixed recurring Python/Qt crash paths with faulthandler and GUI-thread GC ownership.

### 2026-07-10 to 2026-07-18 — runtime foundation and product trust

- Restored a deterministic baseline, removed dormant defects, added smoke checks,
  manifests, lifecycle ownership, job ledger, heartbeat, writer lease, and verified
  Away publication.
- Added pure SPY state, aligned RS, CandidateRegistry, and Greatness engines; wired
  SPY and Greatness only in shadow.
- Added trustworthy Industry Board refresh, Master opportunity dedupe, Auto-vs-user
  environment separation, automatic Entry Assist, final-upgrade D1 Focus, novice
  explanations, journal v2, provider-neutral A.I. Summary, and advisory industry RS.
- Reworked day-trade evidence, RVOL, setup tiers, D1 zone/rubric feeds, and daily-
  trend gates while retaining golden/evidence controls for behavior changes.

### 2026-07-01 to 2026-07-08 — research breadth and early Auto Pilot

- Expanded study/playbook families, tracker replay, industry indexes, universe
  building, broker journal imports, and Qt Universe/Industry surfaces.
- Added the setup encyclopedia, Bounce learning tiers, Expected-R ranking spine,
  Alert Command Center, delayed ORB/EMA/VWAP workflows, Auto Pilot, self-healing
  universe, outcome measurement, tracked auto watchlists, and pick feedback.

### 2026-06 — durable data, Qt desk, and Focus Picks

- Added durable D1/H1 stores, gap-aware delta fetching, cache warming, multi-year S/R
  levels, industry/HTF/cloud/structure studies, and theta support improvements.
- Began the Tk-to-PySide6 migration and built the Qt Trading Desk.
- Added FocusPickStore, top-level Focus UI, Master/Bounce integrations, D1 upgrade
  gates, human focus tracking, and the Human Picks tracker.
- Adopted Google Drive as the default operational shared-home pattern.

### 2026-03 to 2026-05 — unified desktop workflow

- Consolidated the AVWAP and BounceBot GUIs, shared home-folder watchlists and data,
  market-session scheduling, ranking/tracker tools, local caches, and swing lists.
- Added the original mini-PC scheduler (retired 2026-08-08), tracker synchronization,
  theta candidate ranking/explanations, D1 watchlist integration, and expanded Market
  Prep/AI reporting.

### 2026-02 — intraday and market-context expansion

- Integrated RRS into BounceBot, configurable EMA bounce monitoring, all-symbol
  bounce checks, sector/industry classification, earnings-gap anchors, GUI controls,
  and anchor persistence.

### 2026-01 and 2025-11/12 — initial system

- Established Master AVWAP and BounceBot, earnings-anchor refresh, AVWAP cross/bounce
  events, signal exports, yfinance fallback, grouped output, and early historical
  evaluation/labeling/trade-outcome tooling.
- Added moving-average and D1 summaries, TickerMover integration, trade logging, and
  the first dependency/runtime structure.

## Retired or superseded implementations

- Desk Link satellite relay/control and the separate mini-PC scanner role are retired
  as of 2026-08-08. The code remains pending a scoped cleanup.
- H1 alerts were retired; H1 now confirms D1 tracker picks.
- The old DESK approval queue for auto-populate was superseded by direct day-scoped
  M5 Focus adoption.
- The legacy shared review-event ledger is read-only; per-installation shards are the
  current writer path.
- The legacy Tk UI remains only for migration compatibility and is not the product
  direction.
- Historical plans and handoffs listed as such in `docs/README.md` are evidence, not
  current execution authority.
