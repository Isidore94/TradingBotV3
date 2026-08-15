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
| Roadmap phase | **Phase 0.5 R1 + R1.1 + R2 built; R7 in progress** (trader redirects 2026-08-15) — P0's live gates are unchanged and still owed |
| **Active packet** | **R7 JOURNAL RELIABILITY AND UX** (`docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`) — building in its §9 commit order. Progress: **step 0 done**. R2 below is code-complete and waiting on its four live proofs; it is no longer the build item |
| **Working branch** | **`phase05-r7-journal-reliability-ux`**, cut from the R2 tip `8d25c92`. **Built in a linked worktree at `..\TradingBotV3-r7` — the main `C:\Users\Aaron\TradingBotV3` checkout stays on the R2 branch, because the desk's scheduled task runs the desk from it.** Run tests with the main repo's venv python and the worktree as cwd |
| Previous branch | **`phase05-r2-focus-gating-strength-board`**, cut from `phase05-r1-auto-modes-quiet-hours` with the R1.1 repair merged forward; pushed. Merging it brings the testing week, R1, R1.1 and R2 together — and R7 on top of that |
| Scope | R2 added `scripts/focus_adoption_gate.py`, `scripts/strength_scan.py`, `ui/services/strength_board_service.py`, `ui/panels/strength_board_panel.py`; edited `autopilot_core.py`, `focus_picks.py`, `pick_feedback.py`, `ui/services/focus_service.py`, `ui/panels/alert_center_panel.py`, `ui/widgets/alert_chart_review.py`, `bounce_bot_lib/legacy.py`, `ui/app.py`. Ask-first approval taken before the first edit |
| State | **Green on the R7 branch: 2931 passed / 19 subtests**, exit 0 (2921 at the R2 tip + 10 from step 0). Smoke and the frozen selftest are unchanged from R2 and are re-run before any merge. Nothing observed live yet — every live proof below is **UNKNOWN**, not PASS |
| Next action | **R7 §9 step 1** (hygiene: A10 precedence, B5 strict timestamps, A4 timeout). **Monday's sequence below is unchanged and still owed** — R7 does not replace it, and R7's own live migration/backfill wait for it |
| Do not start yet | **R3–R6** — their redirects were never given. **R8 code** — waits for R7's build to complete (then branch from the R7 tip). Also Phase 1 cleanup and any Phase 2+ item |
| Doc-only addendum (2026-08-15, late) | Phase 0.5 gained packets **R7 (journal reliability + UX)** and **R8 (Weekend Prep)**: specs written, WISHLIST/plan.md/docs README reconciled (incl. the P3.3 nightly-journal-pull promotion into R7 and the P5.4 narrowing). **Markdown-only — the release candidate, gates, and baseline above are unchanged** |
| **R7 redirect (2026-08-15, second of the day)** | The trader explicitly authorized **R7 code to start now**, ahead of the P0.7 merge: branch **`phase05-r7-journal-reliability-ux` cut from the R2 tip** — same redirect pattern as R1/R2, recorded in `plan.md` Phase 0.5 preamble and the R7 spec header. Rationale: R7/R8 touch journal/weekend surfaces, not the scanning/alerting/Focus path Monday's proofs cover. **The desk keeps running the R2 branch via the scheduled task until the validation day passes — do not switch the desk branch without disarming that task.** R1/R2's eight live proofs remain owed and are inherited by the eventual stack merge. R7's own trader-present steps (live DB migration, full backfill) must NOT run on the desk before Monday's validation passes |

## R7 build progress — `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` §9

Each step is its own green commit, pushed. A step is not done until
`pytest tests/ -q` passes **by its own exit code**.

| §9 step | State | Evidence |
|---|---|---|
| 0 Characterization fixture | **DONE** | `tests/fixtures/journal_rebuild_trades_v1.json` + `tests/test_journal_characterization.py`; 2931 passed, exit 0 |
| 1 Hygiene (A10, B5, A4) | **DONE** | `tests/test_journal_import_hygiene.py` (34 tests); 2965 passed, exit 0 |
| 2 v3 migration + uid migration | **DONE** | `scripts/journal_migrate.py` + `tests/test_journal_migration.py` (26 tests); 2991 passed / smoke 7/7, exit 0 |
| 3 Group-key normalization | **DONE** | `scripts/journal_identity.py` + `tests/test_journal_identity.py` (34 tests); 3025 passed / smoke 7/7, exit 0. Golden regenerated with a note: 10 trades → 9 |
| 4 Assembly changes | pending | |
| 5–10 Adjustments, coverage, activities, FX, reconcile, nightly slot | pending | |
| 11–13 Journal UI | pending | |
| 14 Governance close-out | pending | |

**The golden fixture is the packet's spine.** It freezes what `rebuild_trades`
does today, six known defects included, and it is regenerated only by
`tests/journal_characterization.py` with the change written into the fixture's
`intentional_difference` field in the same commit. It was verified to fail: a
trial `CLOSED_PARTIAL` status change turned three assertions red, and was
reverted.

**Step 1 finding — the ibapi timestamp gap is latent, not live.** The old parser
did not understand ibapi **10.x**'s `"20260804 09:31:00 US/Eastern"` execution
time and answered `pacific_now()` for it, which would have stamped every socket
fill with the import time. The desk is unaffected today: `constraints.txt` pins
**`ibapi==9.81.1.post1`**, whose `"20260804  09:31:00"` form the old parser did
read. So this is a defect that fires on an ibapi upgrade, not one already in the
live journal — recorded that way rather than as a live data-corruption finding.
Verified by running the pre-fix module directly against both spellings.

**Step 2 changed the golden once, on the record.** Schema v3 adds five columns
to every trade row (`net_pnl_cad`, `fx_rate`, `fx_rate_date`,
`reconcile_status`, `anchor_execution_uid`), all NULL or empty until steps 4, 8
and 9 populate them. **No assembled value moved**: legs, opportunity events and
the summary are byte-identical and every shared trade column matches, verified
column by column before regenerating. The note is in the fixture's
`intentional_difference` field, and the generator now **refuses to write a
changed golden without one**.

**The live journal DB has not been touched.** Everything above ran against
fixture and temporary databases. `journal_migrate.py` defaults to a dry run
against a throwaway copy, and a test asserts the live file is byte-identical
afterwards and that no backup is taken (because nothing changed). The real
migration is a trader-present step and waits for Monday.

**Deferred out of step 3, deliberately — one spec conflict.** Spec §5 fix 3 puts
"the manual-execution dialog gains real broker/account pickers" in this step,
but that dialog exists **only in the legacy Tk tab** (`scripts/journal_tab.py`),
which spec §7 says stays untouched — and the Qt panel has no manual-entry dialog
at all yet. The data layer already accepts a real broker/account
(`manual_execution_from_fields` honours them), so the missing half is purely
UI and belongs to the Qt Trades tab in **step 11**. Recorded rather than
silently skipped.

**One suite run exited 3 (a crash, not a failure) during step 3, then passed on
re-run.** This is the documented Qt/worker-thread hazard in `tests/conftest.py`
("12/12 is a real improvement over 8/10 but it is not a proof of thread
safety"); the leaked `run_strategy` worker threads it names are the suspected
cause. It is **not** attributable to R7 — no R7 file touches Qt — but it is
recorded here because a crash is not a pass, and P1.1 owns the fix.

**Trader-present steps ahead — the build stops and asks at each** (spec §9):
Flex token setup (§8) before step 7 goes live, account tax-status labeling after
step 11, the first live migration + full backfill, and the reconciliation-week
sign-off. **The live migration and full backfill do not run on the desk before
Monday's validation day passes** — everything until then is built against
fixture and copied DBs.

## Merge safeguards — read before Monday

### Release candidate

Monday tests **the tip of `phase05-r2-focus-gating-strength-board`**. The last
commit that changed code or tests is the R2.3 fix **"Give each return to the
desk an identity its timestamp cannot collide"** (`90ba0d4`, committed
2026-08-15 13:11:19 PT); anything after it on this branch is documentation, so
the running behaviour Monday exercises is exactly the tree the three gates
below were run against.

Stated that way on purpose: the SHA above is re-stated **only** because the
external provenance check needs commit time and executable mtime side by side.
**The rule is unchanged: if a commit changes code or tests, all three gates
re-run and this whole section is updated — a stale line here is worse than
none.**

| Check | Result | When |
|---|---|---|
| pytest | **2921 passed / 19 subtests**, exit 0 | 2026-08-15, after R2.3 |
| smoke | **7/7**, exit 0 | 2026-08-15, after R2.3 |
| frozen rebuild + selftest | **`selftest OK: 31/31 checks passed (frozen)`**, exit 0 | 2026-08-15, after R2.3 |

**Provenance, on its face:** last code commit `90ba0d4` at **13:11:19 PT**;
`dist\TradingBotV3\TradingBotV3.exe` mtime **13:13:54 PT** — the executable
postdates the last code commit. Commits after `90ba0d4` on this branch are
Markdown-only (verify with `git show --stat`); the R2.2 round's executable had
been built 21 seconds *before* its tip, which is why the ordering is now
recorded here explicitly rather than left derivable.

The R2.3 fix changed code, so it is a **new** release candidate and all three
gates were re-run against it — including the frozen rebuild, even though no
packaging trigger applied. The frozen one is never optional: it is the gate that
caught the `ai_jobs` roster clash and the `-c` scan-spawn defect when the suite
could not.

### Rollback points

| Point | SHA | What it is |
|---|---|---|
| Pre-R1 | **`e18757e`** | Tip of `testing-week-2026-08-10`. The build that ran the desk before any Phase 0.5 work |
| Pre-everything | **`7d85a27`** | `main`. Last known-good merged trunk |
| Pre-R2 | `4389961` | Tip of R1+R1.1, if only R2 needs backing out |

Ancestry is linear — `main` → `testing-week` → R1 → R2 — so any of these is a
clean checkout, not a revert.

**The rolled-back build reports `selftest OK: 30/30`, not 31/31, and that is
correct** — the 31st check is the one bundling `docs/DESK_TESTING_PLAN.md`, which
did not exist at `e18757e`. `docs/DESK_TESTING_PLAN.md` §3.4 now says so in plain
language, because a 6am reader watching the count drop would otherwise read a
successful rollback as a broken one.

### Rollback drill — EXECUTED 2026-08-15

Run once, unattended, with no desk process running:

| Step | Result |
|---|---|
| Disarm `TradingBotV3 0700 Launch` | `Ready` → `Disabled` |
| Check out the pre-R1 rollback SHA `e18757e` | clean, no conflicts |
| Verify the rolled-back build starts | `selftest OK: 30/30 checks passed`, exit 0 (30 not 31: the testing-plan check did not exist at that SHA — the count moving is *correct*) |
| Return to the release candidate | back at `bf1ab89`, `selftest OK: 31/31` |
| Re-arm the launch task | `Disabled` → `Ready` |

All three TradingBotV3 tasks confirmed `Ready` afterwards (`0700 Launch`,
`AI Jobs`, `Push cold data to DAS`).

**What the drill did NOT prove:** a full GUI launch. The selftest is the
designed proxy — it imports every lazily-loaded engine and loads every
`__file__`-relative asset with no window and no network — but it is not a
double-click. If the trader wants that certainty before Monday, one manual
launch at `e18757e` is the missing step; the mechanical path around it is
proven.

**The order matters and is the point:** disarm first. The launch task starts
the desk from source, so checking out another SHA while it is armed can have
the task launch a half-swapped tree.

### Live proofs are UNKNOWN until observed

Nothing in the tables below has been run on a live session. They are
**UNKNOWN**, and UNKNOWN is a result — `plan.md` sec 6 requires recording it as
such. A green test suite does not upgrade any of them, and none may be written
as PASS in `CHANGELOG.md` without preserved real-session evidence.

## Monday sequence — 2026-08-17

Do these in order. **Nothing merges until (a) and (b) both pass.**

**The trader can read all of this on the desk**: Settings ▸ Testing Plan renders
`docs/DESK_TESTING_PLAN.md`, a plain-language version of the same sequence. That
file restates the proofs below for a human reader and **must be updated in the
same pass whenever they change**.

### (a) Run the live proofs on THIS build, during the real session

Both packets' proof tables are below — four for R1, four for R2. They are written
against the finished build, not against what either packet did mid-flight; the
AWAY proof in particular changed when R2 landed.

Two are already actionable outside the session: the R1 quiet-boot proof (a ~21:00
launch, which the trader is running the evening of 2026-08-15) and the R2 "Not
today" proof (needs an auto-adopted M5 entry, so it needs a session first).

Record every result, including UNKNOWNs, without rewriting the outcome
(`plan.md` sec 6).

### (b) Run the plan.md sec 6 first-session checklist

`docs/FIRST_SESSION_CHECKLIST.md`, which already carries the four R1 rows added
2026-08-15. It has **no R2 rows** — use the R2 proof table below alongside it
rather than assuming the checklist covers this build.

### (c) Only if both pass: P0.7 merges the stack into `main`, in order

Three branches, each a superset of the one before, so the order is not optional:

```
testing-week-2026-08-10   ->  main
phase05-r1-auto-modes-quiet-hours (carries R1 + R1.1)  ->  main
phase05-r2-focus-gating-strength-board                  ->  main
```

Merging R2 alone would carry all three, but merging in order keeps the history
readable and lets a single step be reverted.

**Gates to re-run at merge time, on `main` after the final merge:**

| Gate | Command | Expected |
|---|---|---|
| Full suite | `.venv\Scripts\python.exe -m pytest tests/ -q` | **2919 passed / 19 subtests**, exit 0 — check pytest's own exit code, not a piped tail |
| Smoke | `.venv\Scripts\python.exe scripts/smoke_check.py` | **7/7**, exit 0 |
| Frozen rebuild | `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm` | exit 0, ~4 min unattended. **Already green after R2.2** — repeat only if code lands after that |
| Frozen selftest | `dist\TradingBotV3\TradingBotV3.exe --selftest` | **31/31**, exit 0, output ending `(frozen)`. **Already green after R2.2** |

**Is a packaging trigger pending? No — but rebuild anyway.** Checked all five
triggers across the whole stack (`e18757e..HEAD`): no new third-party dependency,
no new non-`.py` runtime asset, no new top-level *package* under `scripts/`
(`focus_adoption_gate.py` and `strength_scan.py` are modules, reached by static
analysis through eager imports; the two new UI files sit inside `scripts/ui`,
already collected), no new dynamic string import, and no `__file__`/`ROOT_DIR`/
`sys.path` change. The spec-drift test passes. **The rebuild is still required**
because CLAUDE.md mandates one before every merge to `main`, and because:

> **Correction, 2026-08-15:** every "frozen selftest 30/30" recorded for R1, R1.1
> and R2 was actually the **source** selftest (`launch_gui.py --selftest`, whose
> output carries no `(frozen)` suffix), against a `dist/` built 2026-08-13 that
> predated all three packets. **Resolved the same day — see the frozen rebuild
> below.** Re-run it at merge time only if code lands after that rebuild: this is
> the gate that has historically caught what the suite could not, finding the
> `ai_jobs` roster clash on 2026-08-09 and the `-c` scan-spawn defect on
> 2026-08-13.

### Frozen rebuild and REAL frozen selftest — 2026-08-15

Five rebuilds, all green. The first was the run three packets of notes had
mislabeled; the second was forced by the testing-plan asset; the third was the
R2.1 release candidate `bf1ab89`; the fourth was the R2.2 tip — built 21
seconds before its final commit, which the external review correctly refused as
provenance; the fifth is the current R2.3 candidate, built after `90ba0d4`.

| # | Time | Result |
|---|---|---|
| 1 | 09:58 | `selftest OK: **30/30** checks passed **(frozen)**`, exit 0 |
| 2 | 10:27 | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 |
| 3 | 11:0x | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 — on `bf1ab89` |
| 4 | 13:0x | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 — superseded: exe predated its tip by 21 s |
| 5 | 13:13 | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 — **current, after code commit `90ba0d4` (13:11:19)** |

Rebuilds 4 and 5 were run **without a packaging trigger**, because a code commit
makes a new release candidate and CLAUDE.md requires a rebuild before merging to
`main`. The count is unchanged at 31, which is the expected result: neither R2.2
nor R2.3 added a dependency, asset, package or dynamic import.

**31, not 30, and that is the point.** The Testing Plan tab renders
`docs/DESK_TESTING_PLAN.md`, a runtime asset that lives **outside `scripts/`**.
The spec's package-asset sweep only mirrors files inside `FIRST_PARTY_PACKAGES`,
and `test_packaging_spec_drift.py` only walks `scripts/` — so **neither would
have noticed it going missing**, and the frozen desk would have shipped showing
"plan file not found" on the one page the trader opens when nothing else is
behaving. Three things now guard it: an explicit `datas` rule with a hard
`SystemExit` if the file is absent at build time, a new selftest asset check
(the 31st), and a test asserting the spec rule still exists. Confirmed present
in the bundle at `dist/TradingBotV3/_internal/docs/DESK_TESTING_PLAN.md`.

That trigger is trigger 2 in the CLAUDE.md list ("new non-`.py` runtime asset"),
plus trigger 5 (`__file__`-relative resolution — the view resolves through
`sys._MEIPASS` when frozen, since a frozen build has no `scripts/` tree to walk
up from).

| Check | Result |
|---|---|
| `pyinstaller .\packaging\tradingbotv3.spec --noconfirm` | **exit 0** |
| `dist\TradingBotV3\TradingBotV3.exe --selftest` | **`selftest OK: 31/31 checks passed (frozen)`**, exit 0 |

The `(frozen)` suffix is the whole point: the source selftest prints the same
count without it, which is how three packets of notes recorded a run that had
never happened. Any future entry claiming a frozen result must quote the suffix.

What this build collected: `ui` 109 submodules, `bounce_bot_lib` 12,
`master_avwap_lib` 26, `market_prep` 23, `diagnostics` 6, `research_warehouse`
19, `desk_link` 7, `duckdb` 39, plus the three package assets
(`veto_reasons_v1.json`, `theme.qss`, `exploration_cohort.txt`). R2's two new
top-level modules (`focus_adoption_gate`, `strength_scan`) and its two new UI
modules are in the bundle and import cleanly under it — which is what this run
was needed to prove and no packaging-trigger analysis could.

The desk was running from source, so nothing had to be closed — and no desk
process was running at all for rebuild 4. `dist/` and `build/` are gitignored, so
this is verification only and never a commit artifact.

**This satisfies the frozen gate for the current tree** (rebuild 4, on the R2.2
tip). Re-run it at merge time only if code lands after that.

### ~~Known blocker for the merge gate~~ — FIXED 2026-08-15

`tests/test_warehouse_seal.py::test_stale_staged_files_are_quarantined_not_deleted`
no longer fails intermittently, and the merge gate has **no rerun-until-green
carve-out**. Any test failure on Monday is a real failure.

It was never flakiness. `reconcile` compared `st_mtime > cutoff` where
`cutoff = utc_now() - grace`, and Windows' system clock ticks about every
15.6 ms while NTFS stamps mtimes far more finely — so `utc_now()` could round
BELOW the mtime of a file written microseconds earlier, and that file read as
"from the future" and was never quarantined. The earlier "timing-sensitive
under suite load" note was wrong: load was never the variable, and it
reproduced in isolation at 3 failures in 6 runs.

Fixed in `store.py` with a 50 ms clock-granularity slack (trader-approved
before the edit; recorded as a warehouse build decision). Verified by 20
consecutive passes of the previously flaky test plus a new deterministic
reproducer that writes and reconciles back to back 25 times.

### R2.2 review pass — 2026-08-15 (four items from the final external review)

Four items, each its own green commit, plus one refinement of item 1 found while
reviewing it. Two changed behaviour, one is documentation with a test that keeps
it honest, one reconciled the desk runbook.

| # | What | Where |
|---|---|---|
| 1 | **The flip drain is explicitly locked.** The AWAY/EVENING → DESK flip records its own moment; adoption refuses any verdict stamped before it (`pending_pick_gate_ok(..., not_before=)`). A failed re-verification now retries every 60 s, five times, instead of falling through to the ordinary stored-verdict drain — the 2-bar lag bound is defense in depth, no longer the only lock. Giving up after five is safe because the barrier holds and the 30-minute staging refresh stamps post-flip verdicts. A follow-up commit closed the DESK → AWAY → DESK mid-flight case: an attempt remembers which flip it answers, so a newer return is owed its own measurement rather than inheriting one whose bars predate it | `alert_center_panel.py`, `autopilot_core.py`, spec §11.1 |
| 2 | **One 14:00 boundary.** `auto_scanning_due` used an inclusive datetime endpoint, `_auto_work_due`'s fallback used `hour < 14`; at 14:00:00.000000 they disagreed. Both now call `within_auto_scanning_window` over `auto_quiet_hours_fallback_window`, inclusive at both ends. Test pins the exact microsecond at both call sites and was verified to fail against the old spelling | `autopilot_core.py`, `autopilot_service.py`, R1 spec §4 |
| 3 | **The two-bar tolerance is recorded as an accepted exposure**, with its backstop named: BounceBot's four-close triple-VWAP invalidation plus the desync repair removes a bad adoption within ~4 completed bars. A test pins both constants so the documented bound cannot quietly stop being true. No behaviour changed | `autopilot_core.py` comment, spec §11.2 |
| 4 | **The runbook stopped contradicting this file.** It claimed 31/31 at 09:58 where this file says 30/30 — the checkpoint was right, provable from the build: the only selftest change since `e18757e` is the testing-plan asset check added at 10:38, so the runbook was claiming its own bundling was verified before the file existed. Also removed its stale "known flaky test, just re-run it" carve-out and added the rollback section with the 30/30 explanation | `docs/DESK_TESTING_PLAN.md` |

**Not done, and deliberately:** item 3 offered `max_bar_lag = 1` as an
alternative. The trader's note left that as their call, so the accepted-exposure
documentation was built as written and the constant is unchanged. Switching it
later is a one-line change plus the golden-fixture update.

### R2 live proofs owed

None has run. From `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` §8:

| Proof | What to look for |
|---|---|
| Eviction | One staged pick evicted for falling back through VWAP or the previous-day extreme: `Focus gate evicted N staged long pick(s): SYM (not above session VWAP)` in the Auto Pilot log. Silent on the desk by design — the log is the record |
| Adoption refusal | One pick refused at adoption, in `trading_bot.log`: `Focus gate refused N staged pick(s) at adoption`. A verdict older than 45 min reads `gate check is NN min old` |
| Scoped "Not today" | On an auto-adopted M5 entry the button reads `✕ Not today - drop pick` and removes only that entry; the trader's own picks, the swing list and the other side are all still there afterwards. On a name the trader typed the button keeps its old feed-only wording and Focus is untouched |
| Strength board | A board session the trader confirms matches the TC2000 scan's character (~20–40/side). **Re-measure the fetch during market hours** — §10's 27.6 s was taken on a Saturday and is a floor, not a worst case. Decide the RVOL column then; it is specified but deliberately not built |

**Deferred deliberately:** RVOL for the surviving ~20–40 rows a side. Specified
in §9, not built — the trader decides on the first live board session whether
they miss it, and the fetch is cheap only at survivor scale.

### R1 live proofs owed

None of these has run. Each is one observation on the desk:

| Proof | What to look for |
|---|---|
| Quiet hours | Launch at ~21:00 on a weekday with Auto left ON. `autopilot.log` says `Auto Pilot is ON from saved state, but nothing starts yet`; no IB connect, no universe rebuild, no self-arm. A manual scan from the same desk still runs |
| EVENING stop | An EVENING day: the open+30 slot and the 07:00/07:15/07:30 checks run, then one `Evening mode: swing slot(s) … not run` line per refused hourly slot and no further scan. The after-close wrap-up still fires |
| AWAY discipline | An AWAY session: picks do not reach `longs.txt`/`shorts.txt`, alerts arrive silently while the feed and D1 badge fill, and the flip back to DESK adopts **only picks re-measured since the flip** — R2 changed this proof and R2.2 tightened it, so anything staged hours ago and no longer qualifying is refused rather than adopted. If the re-check itself fails, the status line says `Retrying in 60s` and **nothing adopts** — that is also a pass |
| SPY wake alarm | One real ±1% EVENING day, or force it by setting `push_evening_spy_alarm_pct` low: an urgent push, a repeat no sooner than 5 minutes, and silence after flipping out of EVENING |

**~~Known limitation, deliberate~~ — CLOSED by R2 (2026-08-15).** The
AWAY/EVENING→DESK drain no longer adopts an un-revalidated backlog: every staged
pick carries a gate verdict from the most recent 30-minute refresh, and adoption
refuses anything failing, missing, or older than 45 minutes. The AWAY live proof
below is written against that behaviour, not the R1 behaviour it replaced.

### R1 build review — 2026-08-15 (independent five-dimension review; findings code-verified)

**All five findings are FIXED as of the R1.1 pass below.** The list is kept
because the defects are the useful record, not the fact that they closed.

Overall: the architecture is right, fail-open holds at every consumer, the manual
carve-outs are real, the alarm's dedupe/day-roll/restart mechanics are solid, the
shared-scan parity claim is proven against the base commit, no existing test was
weakened, and CLAUDE.md/AGENTS.md are byte-identical. But an **R1.1 fix pass is
required before the live proofs are attempted and before R2 stacks on top** —
the following were verified against the code, not just claimed:

1. **BLOCKER — the boot gate is defeated by the tick.** `_tick` calls
   `self._ensure_bot_running()` ungated (`autopilot_service.py:450`), so a 21:00
   boot with Auto left ON logs "nothing starts yet" and then connects BounceBot
   to IB 30 seconds later. Live proof #1 above will fail as written; every doc
   stating "no IB connect until the window opens" currently describes behavior
   the code does not have. The suite stayed green because the boot test stops
   the timer before a tick can run — the fix needs a test that runs a tick.
2. **BLOCKER — the EVENING SPY alarm fires on YESTERDAY's move pre-open.**
   `_maybe_push_spy_alarm` (`autopilot_service.py:1869-1872`) trusts
   `_spy_session_bars(cached_only=True)` with no bar-date check, and its only
   session gate is the quiet window, which opens 30 minutes before the open. On
   any EVENING morning after a ±1% day, ~7 false urgent wake-ups fire on stale
   data before the first new-session bar (all night if quiet hours are disabled).
   Fix at the data read: refuse a series whose last bar predates `now.date()`.
   Every alarm test stubs `_spy_session_bars`; add one with stale-dated bars.
3. **IMPORTANT — a post-14:00 relaunch silently cancels the after-close
   wrap-up.** The quiet refusal in `_maybe_run_swing_slot`
   (`autopilot_service.py:953-955`) returns before any slot resolution, so slots
   still pending after 14:00 (crash or sleep before the close slot — a 4h39m
   sleep happened on this desk 2026-08-11) stay pending forever and
   `after_close_wrapup_due` never fires that day. Same rationale as the EVENING
   marked-done decision; apply it on the post-window side.
4. **IMPORTANT — EVENING picks still adopt into M5 Focus immediately.**
   `_poll_auto_pick_pending` refuses only AWAY
   (`alert_center_panel.py:1612`); the spec §1/§3.3, CLAUDE.md matrix, EVENING
   runbook, and CHANGELOG all state EVENING stages until the DESK flip. Make the
   code match the documented rule.
5. **IMPORTANT — the legacy Tk GUI dies at construction.** `gui.py:1040` still
   calls `get_shared_watchlist_paths`, which the removal deleted from
   `legacy.py`'s import block; `gui.py` acquires its globals from `legacy`, so
   construction raises NameError. One-line import fix. Invisible to the suite
   (tests import but never construct) and to the import-only frozen selftest.

### R1.1 repair pass — 2026-08-15 (all five findings closed)

| # | Fix | Proof |
|---|---|---|
| 1 | Quiet hours moved **into** `_ensure_bot_running`, the one place automation starts the bot; `force=True` is the manual carve-out and `force_reconnect` passes it | `test_the_tick_cannot_undo_the_boot_refusal` runs a real tick with the clock frozen to a weekday 21:00; `test_the_reconnect_button_starts_the_bot_at_any_hour` |
| 2 | The alarm refuses a SPY series whose last bar predates the day being asked about — stale cache is not a move | `test_yesterdays_cached_move_never_wakes_the_trader` (and the same +3% once today's tape prints it still fires) |
| 3 | `_resolve_slots_after_window` marks still-pending slots done once the window closes, so the after-close wrap-up survives a crash or a long sleep. Before the window opens nothing is resolved | `test_slots_left_pending_past_the_window_are_resolved` |
| 4 | `_poll_auto_pick_pending` refuses `("AWAY", "EVENING")`; EVENING also stops beeping, closing the spec §1 alert cell | `test_away_and_evening_refuse_to_adopt_staged_picks`, `test_evening_queues_alerts_without_a_sound` |
| 5 | `gui.py` uses `LONGS_FILE, SHORTS_FILE` instead of the deleted helper | New `tests/test_module_globals_resolve.py` statically resolves every global four never-constructed legacy modules read — verified to fail on the un-fixed file before the fix went back in |

Hardening taken in the same pass: NaN threshold guard on the alarm; the
quiet-window ⊇ sweep-window containment is now **structural** (`auto_scanning_window`
widens itself to contain `bouncebot_scan_window`, so two independent settings keys
cannot contradict each other); `autopilot_auto_arm_due` takes `quiet_hours` and the
arm test pins it, so a desk with quiet hours disabled no longer turns that test red;
`MainWindow._self_heal_universe`'s gate and the D1-feed beep site now have coverage;
the Qt tests **skip** instead of silently passing without PySide6; the false
"an early close moves this window" docstring claim is corrected (no early-close
modelling exists anywhere — pre-existing, and fail-open since the window is only
ever too long).

**Baseline after R1.1: 2785 passed / 19 subtests / smoke 7/7 / source selftest
30/30**, all exit 0. (Recorded at the time as "frozen"; it was the source run —
`launch_gui.py --selftest`, whose output carries no `(frozen)` suffix.)

Still owed, recorded not fixed: a corrupt `local_settings.json` silently re-homes
the store to `%LOCALAPPDATA%` (wants one loud stderr line plus atomic settings
writes); and the spec §1 EVENING **sweep** cell is now explicitly unresolved in
that spec's new §9 rather than silently unbuilt — the recommendation there is to
leave the sweep running, and the trader decides before the EVENING live proof is
recorded as passed.

Original hardening list from the review, for reference: NaN threshold
bypasses the alarm's threshold test (guard `threshold != threshold` like
`day_pct`); the quiet-window⊇sweep-window containment is enforced nowhere at
runtime (two independent settings keys; clamp or log the contradiction);
`test_autopilot_auto_arm_due_daily_hands_off_rules` reads the machine-local
`qt_auto_quiet_hours` setting and goes red on any desk that disables quiet hours
(pin `quiet_hours=True`); `MainWindow._self_heal_universe`'s gate and the D1-feed
beep site have zero coverage; five Qt tests silently pass (not skip) without
PySide6; the spec §1 matrix retains two EVENING cells (sweep "then quiet",
alerts "queue") the build never implemented and §8 never settled — reconcile or
build; a corrupt `local_settings.json` still silently re-homes the store to
`%LOCALAPPDATA%` (one loud stderr line + atomic settings writes); the
"early close moves this window" docstring/CHANGELOG claim is false —
`get_market_session_window` hardcodes regular hours (pre-existing, fail-open).

### Previous packet — ticker-briefs hardening (TB-0..TB-6)

| Field | Value |
|---|---|
| State | **Integrated and green on `testing-week-2026-08-10`**. **Live proof still owed: the 2026-08-12 22:00 window.** The 08-11 night proved TB-0, broke on TB-3, and exposed a task time limit that defeated its own concurrency guard plus 4h39m of machine sleep |
| Side item landed | **Snapshot popup opens at desk height** (2026-08-11) — UI geometry only |
| Side item landed | **Phone push policy + two richer pushes** (2026-08-11) — AWAY became the only pushing mode; R1 has since added EVENING's SPY alarm as the second exception |

A newly arriving AI resumes the active packet if it is unfinished. If it is complete,
it performs the stated next action. It does not select a different roadmap item
without explicit trader direction.

## Planning pass — 2026-08-15 (documentation only)

**Superseded the same day**: the trader then directed R1 to be built, and it was.
See the active-work table above. This section is kept for the recon findings it
records, which are still the current understanding.

The trader promoted the 2026-08-14 `WISHLIST.md` entries and directed a build
foundation for the next implementer. Recorded in this pass:

- **`plan.md` Phase 0.5 (R1–R6)** inserted with the trader's ranked order
  (R1 auto modes/quiet hours first, R2 Focus gating + strength board second) and
  five ACTIVE specs under `docs/` (indexed in `docs/README.md`).
- Eight trader decisions captured in the specs and `WISHLIST.md` (demote+label
  never hide; v1 extension rules; existing universe; build order; full pre-close
  honesty bundle; prior-anchor AVWAP line; checked = recorded decisions;
  Not-today removes just the M5 entry).
- **After-close investigation COMPLETE** (read-only): the live Master AVWAP scan
  scores today's forming D1 bar (no completed-bar guard in `runner.py`), and the
  setup tracker is written at 12:00 PT then wiped and rewritten by the ~13:24
  close-slot finish. Mechanisms with file:line evidence are in
  `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md` §4. No fix is built.
- Verification: Markdown-only pass — link resolution, `git diff --check`,
  control-document consistency. The recorded automated baseline (2738 passed /
  19 subtests / smoke 7/7 / source selftest 30/30) is **unchanged**.
- Housekeeping note: untracked `desk_report.xml` at the repo root is generated
  pytest JUnit output from the 2026-08-09 desk gate — left untracked; P1.5 owns
  gitignoring desk JUnit artifacts.

The active build item above (P0 live gates) is unchanged; Phase 0.5 code starts
only after P0.7 merges.

## Branch

- Working branch: **`phase05-r1-auto-modes-quiet-hours`** (R1; pushed to origin)
- Parent: **`testing-week-2026-08-10`** at `e18757e`
- Base: `main` at `7d85a27`
- State: **neither branch merged to `main`; no PR recorded**
- The R1 branch is a strict superset of `testing-week-2026-08-10`, so the desk's
  scheduled tasks that run from source are unaffected by a checkout. The standing
  rule still holds: disarm the scheduled task before switching branches on the
  desk.
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
- ~~One flaky test~~ **FIXED 2026-08-15.** `test_stale_staged_files_are_quarantined_not_deleted` was never flaky: `reconcile` compared a file's mtime against a coarser system clock, so a file written microseconds earlier read as "from the future". Both earlier notes here were wrong - it was not "observed once" (3 in 6) and not load-
  related (it reproduced in isolation). See the merge-safeguards section above.

## URGENT — the frozen desk cannot scan (found and fixed 2026-08-13)

The desk switched to `dist\TradingBotV3\TradingBotV3.exe` as its daily driver on
2026-08-12. The frozen build spawned its scan child as `sys.executable -c <code>`,
which under PyInstaller means `TradingBotV3.exe -c …` — rejected by the app's own
argument parser, exit 2, one second after each slot fired. **Every Master AVWAP D1
swing scan failed from 2026-08-12 07:30 through 2026-08-13 09:00.** Last success:
2026-08-11 13:23:59, 622 setup rows.

Nothing else broke, which is why it went unnoticed: BounceBot, the 07:00 open scan,
Auto Pilot and the away report all run in-process. The visible cost was one layer
away — the overnight AI read 11 stale D1 sources.

**Code fix is committed and green** (`scripts/scan_worker.py`,
`scan_service.scan_worker_command`, `launch_gui --run-scan`, `selftest` roster,
`tests/test_scan_worker_spawn.py`), and the desk was **rebuilt 2026-08-13 11:00:25**
after the trader closed it:

| Check | Result |
|---|---|
| pytest | **2738 passed, 19 subtests**, exit 0 |
| smoke | **7/7**, exit 0 |
| frozen selftest | **30/30**, exit 0 — was 29/29; `scan_worker` is the added check |
| frozen `--run-scan` dispatch | **verified** — a deliberately malformed payload now fails inside `scan_worker.parse_payload`, where the old build answered `TradingBotV3.exe: error: unrecognized arguments: -c …` |

**Still owed: one real slot on the desk** — `Swing scan for slot HH:MM finished at …
(N setup rows)` in `autopilot.log`. Nothing before that proves a full scan runs
end to end under the frozen build; the checks above prove only that the child
starts and reaches the scanner. Until then the fallback is running from source
(`scripts/launch_gui_auto.ps1`), where the `-c` form is correct.

Also owed once a slot passes: the D1 sources have been stale since 2026-08-11
13:23:59, so tonight's AI window is the first that can read fresh evidence. A brief
that still cites truncation after a good scan day means something else is wrong.

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
  (2647 passed / smoke 7/7), and **re-run 2026-08-15 on the R1 branch**
  (2773 passed / 19 subtests / smoke 7/7 / source selftest 30/30, all exit 0).
  Re-run again before merge if further code lands.
- **P0.2–P0.4:** run the single-main session checklist, Away/ntfy validation, and
  observability rollover.
- **P0.5:** run the durability mid-session restart/backfill drill.
- **P0.6:** start Local-AI's five-session clock and the warehouse broker/live/pilot
  sequence.
- **P0.7:** merge only after the live-validation day and applicable rechecks pass.
  **Three** branches now queue for `main` - `testing-week-2026-08-10`, the R1
  branch (carrying R1.1) built on it, and the R2 branch built on that. The exact
  order, the gates to re-run, and the packaging-trigger answer are in the Monday
  sequence at the top of this file.

Do not add historical detail here. When a change lands, update `CHANGELOG.md`; when a
gate remains, update `plan.md`.
