# Current checkpoint

This file is the frequently refreshed active-work, branch, and verification stamp.

- Implemented inventory and revision history: [`CHANGELOG.md`](CHANGELOG.md)
- Remaining work and gates: [`plan.md`](plan.md)
- Supporting-document roles: [`docs/README.md`](docs/README.md)

## Active work — read this before choosing a task

There may be only one active build item unless `plan.md` explicitly identifies an
elapsed evidence lane that can run in parallel.

| Field | Current value |
|---|---|
| Roadmap phase | **P0 — validate and merge the testing-week branch** |
| Active packet | **TICKER-BRIEFS HARDENING — TB-0..TB-6** (`docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4b; armed by the trader 2026-08-11, first-night repair TB-5/TB-6 built 2026-08-12) |
| Scope | `scripts/ai_jobs/briefs.py`, `runner.py`, `ledger.py`, one additive helper in `scripts/ai_summary.py`, `scripts/register_ai_jobs_task.ps1`, `tests/test_ai_ticker_briefs.py`, `tests/test_ai_jobs_runner.py`. No detector, scoring, or alert file touched; output stays advisory-only |
| State | **Integrated and green on `testing-week-2026-08-10`** (2727 passed / 19 subtests / smoke 7/7, both exit 0). **Live proof owed again: the 2026-08-12 22:00 window.** The 08-11 night proved TB-0, broke on TB-3, and exposed a task time limit that defeated its own concurrency guard plus 4h39m of machine sleep |
| Side item landed | **Snapshot popup opens at desk height** (2026-08-11, trader ask) — UI geometry only, no detector/scoring/alert file touched; baseline unchanged at 2687 passed / smoke not re-run (no non-Qt path affected) |
| Side item landed | **Phone push policy + two richer pushes** (2026-08-11, trader ask, design confirmed before editing per the ask-first rule) — AWAY is now the only mode that pushes (price alerts stay the always-on exception), the swing push carries the full favorite/high-conviction roster, and a second hourly push names the D1 level events since the last one. New baseline **2720 passed / 19 subtests / smoke 7/7**, both exit 0 |
| Next action after this packet | **P0.2–P0.4** live gates, plus the ticker-briefs morning check below. P0.1 re-baseline is done |
| Do not start yet | Phase 1 cleanup or any Phase 2+ feature/foundation item |

A newly arriving AI resumes the active packet if it is unfinished. If it is complete,
it performs the stated next action. It does not select a different roadmap item
without explicit trader direction.

## Branch

- Branch: **`testing-week-2026-08-10`**
- Current commit before this documentation pass: **`1f41af1`**
- Base: `main` at `7d85a27`
- State: **not merged to `main`; no PR recorded**
- Testing-week intent: Mon–Wed Auto/Away and baseline observation; Thu–Fri
  live-session validation; merge only after a `plan.md` Section 6 day passes.

## Last full Windows desk gate

Recorded at the 2026-08-09/10 desk re-baseline (`60119e8`):

| Check | Result |
|---|---|
| pytest | **2611 passed, 7 subtests passed**, exit 0 |
| JUnit | 2618 cases, 0 failures, 0 errors, 0 skipped |
| smoke | **7/7**, exit 0 |
| frozen self-test | **29/29**, exit 0 |
| Python | repo-local uv-managed **3.12** environment |

The frozen run found a real packaging-roster conflict: `ai_jobs` was deliberately
excluded from the bundle but required by self-test. The roster was corrected and a
permanent disjointness test added. The 29/29 figure is therefore the correct current
expectation, not the older 30/30 text in historical handoffs.

## Changes after that gate

The following commits landed after the recorded full gate and require coverage by the
next normal full run; none changes the frozen package inventory:

- `07395a0` — Chart Review Setups column defaults hidden and can be restored.
- `bfc8850` — a late-opened alert receives current bars.
- `4907b6f` — a published best-swing report can notify the phone.
- `1f41af1` — the swing push stays quiet when no readable setups exist.
- documentation consolidation: `CHANGELOG.md` for implemented history, `plan.md` for
  remaining work, `docs/README.md` for classification, and the renamed
  `CURRENT_CHECKPOINT.md` for active state;
- mandatory AI read/update workflow in `CLAUDE.md`/`AGENTS.md`, phase-gated roadmap
  ordering, and the new non-authoritative `WISHLIST.md`.

The documentation packet does not change the recorded automated baseline. Markdown
verification consists of link resolution, `git diff --check`, control-document
consistency, and confirmation that tracked edits remain Markdown-only.

## Re-baseline and desk configuration — 2026-08-10 (evening)

**P0.1 is satisfied for the four post-gate commits above.** Full Windows run on the
working tree:

| Check | Result |
|---|---|
| pytest | **2647 passed, 7 subtests passed**, exit 0 (109s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging-trigger change since 29/29 |

Re-run after the decision-0015 documentation/comment pass: **2647 passed**,
**smoke 7/7**, unchanged. That pass edited Markdown, docstrings, comments, and two
user-facing strings only; no behavior, path, or test changed.

**Current baseline after the local-AI evidence-budget packet:**

| Check | Result |
|---|---|
| pytest | **2659 passed, 19 subtests passed**, exit 0 (104s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging trigger (no new package, no new runtime asset, no new dependency) |

Twelve new tests cover the budget resolver and its fallbacks, the cloud ceiling
staying untouched, the derivation itself (worst-case retry prompt must fit the
context left after generation), the truncation tripwire firing/staying silent, and
ledger usage recording.

**Current baseline after the BounceBot scan-window packet (2026-08-10, late):**

| Check | Result |
|---|---|
| pytest | **2672 passed, 19 subtests passed**, exit 0 (104s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging trigger (no new package, no new runtime asset, no new dependency) |

Thirteen new tests in `tests/test_bouncebot_scan_window.py` cover the window bounds,
the overnight and weekend refusals, the settings escape hatch and margin fallbacks,
and the four service transitions that matter: the close pauses a running sweep, an
after-hours start pauses on its first tick without needing a boundary crossing, a
manual resume survives subsequent ticks, and a broken session lookup changes nothing.

**Why this packet exists.** The trader reported the bot "running all night prompting
the API constantly". Reading the artifacts found two independent causes, and the loud
one was not the AI layer:

1. **BounceBot swept all night** — Auto Pilot's 30-second tick re-enabled scanning
   with no clock check, and `trading_bot.log` showed ~830-900 metric lines/hour for
   147 symbols, about eight full sweeps an hour, continuing hours past the close with
   IB answering `HMDS data farm connection is broken` and RRS timeouts. **Fixed here.**
2. **`ticker_briefs` retried all night** — see the open question below. **Not fixed.**

No metered API was involved in either: every unattended AI call is hardcoded
`provider="local"` against Ollama on localhost. OpenAI and Anthropic are reached only
from GUI buttons.

### Resolved — overnight AI job cadence (armed and built 2026-08-11)

The ticker-briefs hardening packet was **armed by the trader on 2026-08-11** after the
first overnight run and is **built** on this branch. The question below is kept because
its premises were partly wrong, and the correction is the useful part.

**What the first repaired night (2026-08-10/11) actually showed.** `ticker_briefs`
completed **all 95 symbols in 5,962 s — ~63 s/call**, not the ~4.75 min/call recorded
below. There was no window overrun. Instead **every one of the 95 briefs was
content-free**: the base evidence package was budgeted to the local ceiling *before*
the per-symbol projection, so the per-symbol-rich sources were unfunded at 0 chars
(`setups.current_tracker` 95,806 chars, `setups.current_tiers` 77,124,
`setups.bounce_learning` 17,995, `market.industry_intraday_rs` 17,833) and the funded
tables were sheared to about one row. MRVL's brief reads **"1 of 19 requested source(s)
usable"**, the one being its own watchlist membership. That is TB-0, and it was the
defect worth an hour and a half of GPU time to fix.

**Built:** TB-0 project-then-budget; TB-1 per-ticker failure isolation with an honest
partial morning file (`Briefed N of M. Failed: …` in the header); TB-2 deterministic
membership-only skip; TB-3 resumable completion keyed by
`(session_date, symbol, evidence_hash)`; TB-4 a three-attempt per-session cap with an
identical-error early stop. `run_daily_summary` is untouched, so the two jobs now run
**separate five-session clocks**: `ai_summary`'s continues, `ticker_briefs`' restarts
at zero.

**Live proof owed — the next 22:00 window.** In the morning check: coverage counts
above one usable source per brief, statements citing real evidence, a morning-file
header stating the outcome, at most three `ticker_briefs` ledger rows for the session
(with a `terminal: true` row if it stopped early), and exactly one artifact set per
symbol under `ai_store/briefs/<year>/<session>/tickers/<symbol>/`.

**~~Known defect, reported not yet fixed (2026-08-11 evening review).~~ FIXED
2026-08-12 — and it fired live first.** TB-3's cross-firing reuse could never
trigger on the desk: the projected package's `evidence_hash` covers `generated_at`
and every source's read stamp, so identical evidence hashed differently on every
firing. On the night of 2026-08-11 a second runner instance restarted from symbol 1
and re-briefed 25 symbols, leaving 25 duplicate artifact sets on the DAS. The
manifest now carries a `resume_key` over stable fields only (symbol, session,
memberships, source ids + content); `evidence_hash` keeps its whole-package meaning
for artifact identity. Manifest schema `v1` → `v2`; a row without a `resume_key` is
regenerated, never reused.

**Queued, not built (trader-approved 2026-08-11):** the **nightly journal pull** —
a third `journal_import` runner slot ahead of `ai_summary` so the summary reads a
journal already containing the session's trades. Spec with design decisions (Flex
over socket at night, Questrade token-rotation race stated, one-writer statement,
zero-execution `ok`) in `docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4c. Build only
after the 6.4b live proof passes and the trader says go.

**Integration correction (2026-08-11).** Fast-forwarding the hardening packet onto
the testing branch exposed a real 27-character ceiling overrun in the first focused
Windows run: list truncation budgeted retained rows before prepending its truncation
banner. `_truncate_to_budget` now budgets the banner too. Focused/full verification
is green: **74 focused tests**, **2687 full-suite tests plus 19 subtests**, and
**smoke 7/7**, all exit 0. The full gate also exposed a test-only warehouse-tee
hermeticity issue: its assertion observed every store open in the pytest process
rather than the tee worker it claimed to test. The assertion is now worker-scoped;
no warehouse runtime behavior changed.

<details>
<summary>The original open question, as written on 2026-08-10 (premises now corrected
above)</summary>

The 30-minute task repeat is **not** a work cadence; it is a retry ladder, and on a
healthy night sixteen of the seventeen firings read the ledger and exit in about a
second. Lengthening the interval would therefore save nothing and weaken the
self-heal. Two real defects sit behind the symptom instead:

- **A failing job has no attempt cap.** Only `ok` is a canonical completion, so a
  deterministic failure retries on every firing for the rest of the window. On the
  night of 2026-08-09/10 `ticker_briefs` failed **11 consecutive times at 9-16 minutes
  each — about 111 minutes of local inference that produced nothing.** A per-session
  attempt cap (2-3) would keep the self-heal for transient faults (NAS asleep,
  endpoint down) and end the grind.
- **`ticker_briefs` cannot finish as scoped.** It calls the model once per unique
  Focus/watchlist symbol — **95 today** — and publishes the morning file only after
  every one succeeds. At the observed ~4.75 min per call that is **~7.5 hours against
  an 8-hour window**, while the slot reserves only 120 minutes. It needs a symbol cap,
  incremental publication, or both.

Neither is fixed. Deferred deliberately: the 22:00 window on 2026-08-10 is the first
run with the repaired `gemma3:12b-tbv3ctx` model and is the live proof the AI-jobs
repair is owed, so the night was left alone rather than changed hours before it.

**Contingency drafted (2026-08-10, late):** the repair plan for both defects — plus
per-ticker failure isolation with an honest partial morning file, a deterministic
membership-only skip, and resumable per-symbol completion — is fully specified as the
**ticker-briefs hardening packet**, `docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4b, with
a pointer in `plan.md` P3.3. It is PROPOSED, not authorized: the trader arms it after
reading the 2026-08-11 morning ledger (or later five-session evidence). An arriving AI
must not build it without that direction. This documentation pass is Markdown-only;
the recorded automated baseline (2672 passed / smoke 7/7) is unchanged.

</details>

**Current baseline after the ticker-briefs hardening packet (2026-08-11):**

| Check | Result |
|---|---|
| pytest | **2682 passed, 5 skipped, 19 subtests passed**, exit 0 (106s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging trigger (no new package, no new runtime asset, no new dependency) |

Recorded on a Linux container (Python 3.12, `TZ=America/Vancouver`,
`QT_QPA_PLATFORM=offscreen`); the 5 skips are the Windows-only cases the desk runs, so
the desk figure should read **2687 passed**. Fifteen new tests cover TB-0's
project-then-budget proof and its budget ceilings, the partial-publish header,
membership-only skip, resume-by-evidence-hash, and the attempt cap with its terminal
marker.

**Windows integration gate after the budget and hermeticity corrections:**

| Check | Result |
|---|---|
| focused | **74 passed**, exit 0 |
| pytest | **2687 passed, 19 subtests passed**, exit 0 (126s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging trigger |

**Current baseline after the phone-push policy packet (2026-08-11):**

| Check | Result |
|---|---|
| pytest | **2720 passed, 19 subtests passed**, exit 0 (119s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | **29/29**, exit 0 — rebuilt at the trader's request, not by a packaging trigger |

Thirty-three new tests across `tests/test_away_push_roster_and_d1.py` (roster
membership, bucket-spelling collapse, the honest trim marker, and the D1 push
formatting/capping) and `tests/test_away_push_gating.py` (the AWAY-only gate on both
pushes, once-per-hour cadence, a failed send keeping its events, the kill switch, the
Alert Center classifier, and the panel signal firing on both D1 routing paths). Two
existing tests were updated rather than worked around: the Desk Link reclaim push now
declares AWAY (with a new sibling test proving it stays quiet in DESK), and the day-roll
test asserts yesterday's unsent D1 events are cleared.

**Live proof owed:** the next AWAY session — a swing push whose roster matches the
Setup Tracker's Favorite + High Conviction rows, a D1 push naming only events from that
hour, and silence on the swing/D1 channels while the desk sits in DESK or EVENING.

**Trader-verified on the phone, 2026-08-11 20:0x.** One real push built from the live
feed (593 rows, `data_date` 2026-08-11, source `focus`) delivered `ok: True`: five ranked
HC longs plus the full roster — HC 12 long / 7 short, FAV 30 long / 6 short, 55 names,
nothing trimmed. The D1 push is NOT yet proven: its queue only fills from live alerts in
the running desk.

**Documentation close-out (2026-08-11, Markdown only).** The push policy is now stated
where an operator or an arriving AI will actually meet it: `CLAUDE.md`/`AGENTS.md` core
loop (with the rule that a new ntfy sender must gate on AWAY or justify itself),
`docs/AWAY_SCANNER_RUNBOOK.md`, `docs/EVENING_MODE_RUNBOOK.md`, a `docs/FIRST_SESSION_CHECKLIST.md`
row, and `plan.md` P0.3. No file was added, removed, or reclassified, so `docs/README.md`
is unchanged; `WISHLIST.md` is untouched (no trader-directed idea moved). The recorded
baseline above still stands — this pass changed no code, path, or test.

### Desk rebuilt and relaunched onto the push-policy build — 2026-08-11 20:15

The frozen exe was the running desk (pid 35676, started 19:02); the python desk pid 32620
named earlier in this file was already gone. Rebuilt at the trader's request rather than
on a packaging trigger: graceful `CloseMainWindow`, `pyinstaller … --noconfirm` exit 0,
**frozen self-test 29/29 exit 0**, relaunch. **Running pid is now 2552** (started
20:15:20), heartbeat fresh at the 30-second cadence from 20:16:05. `dist/` is gitignored,
so the rebuild is verification only and no commit artifact.

### Desk restarted onto the scan-window build — 2026-08-10 21:19

The desk was closed gracefully (`CloseMainWindow`, so `closeEvent` ran its panel
shutdowns and released the writer lease) and relaunched through
`scripts/launch_gui_auto.ps1`, the same path the 06:00 task uses. **Running pid is now
32620** (started 21:19:22); it supersedes pid 17984 named below. Auto Pilot resumed ON
from saved state and BounceBot started and connected to IB as before.

Verified on the live desk immediately after:

- `bouncebot_scan_window` resolves to **06:00-13:30** from the real machine settings,
  with the verdict `False` at 21:20 and `True` at 09:45.
- **Zero `Metrics ->` sweep lines in `trading_bot.log` after the restart**, watched to
  fifteen minutes — the previous build would have run two full sweeps in that time.
  The whole log went quiet at 21:19:48 after the startup sequence (18 lines total, all
  of them start-up) against ~830-900 lines/hour beforehand.
- Sustained CPU fell from ~57% of a core to ~17% (and that figure still includes the
  start-up burst).
- `heartbeat.json` stays fresh at the 30-second cadence under the new pid, so the tick
  loop still reaches its end; `writer_role.py` still resolves
  `designated_writer / may publish True`, so the 07:00 publish proof is unaffected.

There is no "scanning paused" line in the Auto Pilot log, and that is the correct
outcome rather than a missing one: a freshly started BounceBot begins with scanning
already disabled, so the window gate simply never enables it and there is no state
change to announce. The startup IB traffic that remains (`$VOLD`/TICK recorder
contract verification) is the market-internals recorder, not the sweep.

Still owed by P0.3: the two live boundary crossings (a resume at 06:00, a pause at
13:30) and confirmation that the session itself is unchanged.

Three desk misconfigurations were found and fixed by inspecting the first
testing-week session's artifacts. All three were machine-local settings lost when the
old desktop was retired; none was a code defect:

1. **Designated writer was unset** — `autopilot_today.txt` had not published since
   2026-07-30, so the whole 2026-08-10 session produced no phone digest and no swing
   push. Fixed with `writer_role.py --designate-self` (NucBox_K8_Plus). The desk was
   restarted at 19:37 local to pick it up (pid 17984 then; superseded by the 21:19
   restart above — the designation is a saved setting and survives both), and
   `writer_role.py` now resolves `designated_writer / may publish True`, exit 0.
   **Not yet proven end to end:** `hourly_away_report_slot_due` returns nothing once
   the hour is past the session close, so no publish was due at restart time.
   `writer_health.json` consequently still carries its pre-fix 15:18 payload — that
   file is rewritten on a *publish attempt*, not at startup, so a stale copy here is
   expected and is **not** evidence the fix failed.
2. **`research_store_dir` was unset** — the warehouse was fully disabled and captured
   nothing. Now `\\MINI-PC\Trading Bot Data\research_lake`, layout created, and the
   restarted desk is the first process to run with it enabled. Capture is proven by
   the next scan writing under the lake, not by configuration alone.
3. **ntfy was already configured and works** — verified by test push (`ok: True`) at
   both `default` and `urgent` priority. Delivery to the iPhone banner/sound is an
   iOS-side setting and is **not yet confirmed by the trader**.

**AI jobs repaired 2026-08-10 (evening).** The task now exits 0 when run through the
scheduler. Details in `CHANGELOG.md`; the live proof is the 22:00 window tonight, and
`%LOCALAPPDATA%\TradingBotV3\logs\ai_jobs-<date>.log` will now carry any failure.
Two AI-layer caveats remain unproven and must be checked against tomorrow's ledger:

- ~~Context smaller than the evidence cap~~ — **closed the same evening.** Local
  calls now cap evidence at `ai_local_evidence_budget_chars` (22,000) and a
  truncation tripwire fails loudly if the server still sees less than was sent. The
  cloud ceiling is untouched.
- ~~The large tier cannot load~~ — **accepted and designed around.** The local large
  tier is retired (plan sec 2); policy drafts and retros belong to the frontier
  model. Revisit triggers recorded: Ollama Vulkan allocator work, ROCm on gfx1103,
  or more RAM.
- **Phase 2 design packet is PROPOSED, not approved.** `docs/LOCAL_AI_AUTOMATION_PLAN.md`
  sec 6.4a. Its six open questions need trader answers before any digest code is
  written — question 1 ("what counts as winning": R at scenario close, MFE/MAE, or
  both) is a trading judgement and is the one the whole fact pack hangs on.

### What the next session must confirm

Four fixes are configured and unit-verified but have **not** completed a live cycle.
None could be proven on the evening of 2026-08-10; all resolve by 09:00 on 08-11:

| Fix | Proof to look for | When |
|---|---|---|
| Designated writer | `autopilot_today.txt.meta.json` names `NucBox_K8_Plus` with a current `verified_at` — it still names the retired `DESKTOP-IABHR62` at 2026-07-30 | 07:00 publish |
| Swing phone push | an ntfy notification carrying numbered swings | 09:00 (push start hour) |
| Research warehouse | new files appearing under the lake root | first scan |
| AI jobs | `ai_jobs-20260811.log` records a completed `ai_summary` / `ticker_briefs` | 22:00-06:00 window |
| BounceBot scan window | **Requires a desk restart first** — the running pid predates the change. Then: one "scanning resumed" line at 06:00, one "scanning paused" at 13:30, and no symbol sweep in `trading_bot.log` after it | 06:00 and 13:30 |

If the 07:00 publish does not happen, read `writer_health.json` first: it will then be
fresh, and its `reason` names the exact gate that refused.

Still open on the desk, not blocking the week:

- `technical_integrity_events.jsonl` is ~247 MB and is never pruned (~10 MB/session).
- Off-site backup: cloud sync was the only off-site Class A copy (decision 0015).
- One flaky test observed once (2026-08-10, late):
  `test_warehouse_seal.py::test_stale_staged_files_are_quarantined_not_deleted`
  failed in a full run, passed in isolation and on the immediate full re-run
  (2672 passed, exit 0). Its zero-grace mtime-vs-clock comparison
  (`store.reconcile`, `incoming_grace_seconds=0`) is timing-sensitive under
  suite load. Candidate for the P1.1 hermeticity packet; not repaired here.

## What the 2026-08-11 window measured, and what was repaired — 2026-08-12

The packet's owed live proof ran and is **partial**. Ledger and manifest evidence:

| | Result |
|---|---|
| `ai_summary` | **ok at 22:02:53**, first attempt, ~170 s, 10 usable sources — against six degraded rounds the night before |
| `ticker_briefs` | **no completion row.** 126 briefs / 101 unique symbols of 182, 0 failures, 22:04:33 → 01:20:08, killed mid-batch |
| `ai_morning_brief.txt` | **never published** — still the 2026-08-10 file, because publication happened only after the loop |
| TB-0 | **Confirmed.** MDB's real brief: 7 of 19 usable, 0 unfunded (08-10 was 4 of 19 with 5 unfunded) |
| TB-1 / TB-2 / TB-4 | Not exercised — 0 failures, and every membership-only name sits past list position 100 |
| TB-3 | **Proven broken**, 25 symbols with two rows and two distinct `evidence_hash` values |

Three defects and one machine fault, all now addressed except the last:

1. **TB-5 — roster noise.** 96.2% of everything sent to the model (307,630 of
   319,687 chars) was ticker name-dumps matched line-wise; median symbol-specific
   content 42 chars; only 18 of 166 symbols had a real scan line. Fixed by a
   residue test, not a ticker count. Measured effect: **166 model calls → 49**.
2. **TB-3** — see the repaired entry above.
3. **TB-6 — publication only after the loop.** Now republished after every resolved
   symbol, with an explicit in-progress note; the market-session block still
   suppresses publication outright.
4. **`ExecutionTimeLimit` was `PT2H` against an 8-hour window** — it terminated the
   22:00 run's parent at 00:00, freeing `IgnoreNew` so the 00:00 repetition started a
   second runner while the first instance's Python child kept going. The manifest
   shows the two interleaving one-for-one from 00:01:54. Now `PT8H` in
   `scripts/register_ai_jobs_task.ps1` **and applied to the live desk task**.
5. **Machine sleep — trader-owned, not code.** 60 Modern Standby transitions during
   the window, **4h39m asleep**, including an unbroken 01:39:42 → 05:57:09 that
   killed the run and suppressed every firing from 01:30 to 05:30. The trader is
   raising the sleep setting. **Until that is confirmed, no overnight result is
   evidence about the AI layer.**

**The 2026-08-12 morning check.** Expect ~49 model calls against ~160 symbols (the
rest membership-only), roughly an hour of inference rather than 3.5, exactly one
`ticker_briefs` ledger row, a morning file dated 2026-08-12 **without** the
in-progress note, no duplicate artifact sets, and briefs that cite `daily.market_prep`
scan lines and `setups.tier_performance` rows rather than complaining about
truncation. `setups.current_tracker` is a known remaining gap: it arrives as one
JSON line, so line-based projection is still all-or-nothing for it.

## Immediate live gates

- **P0.1:** ~~run the complete Windows automated gate~~ — **done 2026-08-10**
  (2647 passed / smoke 7/7). Re-run before merge if further code lands.
- **P0.2–P0.4:** run the single-main session checklist, Away/ntfy validation, and
  observability rollover.
- **P0.5:** run the durability mid-session restart/backfill drill.
- **P0.6:** start Local-AI's five-session clock and the warehouse broker/live/pilot
  sequence.
- **P0.7:** merge only after the live-validation day and applicable rechecks pass.

Do not add historical detail here. When a change lands, update `CHANGELOG.md`; when a
gate remains, update `plan.md`.
