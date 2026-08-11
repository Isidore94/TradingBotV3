# Checkpoint review — 2026-08-08 (items 13b/13c build, branches A/B/C)

Document role: **historical review and merge record**. Later repairs and merges are
summarized in the root `CHANGELOG.md`; open work is reconciled in `plan.md`.

Reviewer: independent checkpoint review of the three unmerged build branches
(`durability-catchup` = A, `local-ai-phase-0` = B, `local-ai-phase-1` = C),
conducted with direct inspection of the diffs, task scripts, and branch
topology — not from the building agent's self-report alone.

**VERDICT: PROCEED WITH CONDITIONS.** Tonight's unattended run proceeds as
the trial it should be; nothing merges and no new phase starts until the
conditions below are met.

## 1. Merge plan and conditions

Order: **A → B → C**, sooner than the default packet rhythm, because the
current state is the risky one: production runs from an unmerged branch
checkout (the live AI-jobs scheduled task points into the working tree, which
is checked out on C).

| Branch | Merges when |
|---|---|
| A `durability-catchup` | (i) the 06:00 PT task-start fix (sec 3.1) is folded in, and (ii) one deliberate restart drill on a session day yields a HEALTHY audit with a nonzero backfill count |
| B `local-ai-phase-0` | immediately after A — default-off with byte-identical-when-unset proven by test; a live-validation day adds nothing to an inert change |
| C `local-ai-phase-1` | after its first successful unattended week (scheduler, window logic, NAS publish surviving contact without supervision) |

Standing rules until all three are merged:

- **No branch switches on the desk without disarming the AI-jobs scheduled
  task first** — a checkout switch silently changes (or breaks) what runs
  overnight.
- The loose uncommitted packaging edits floating across the tree must be
  committed or stashed off it.

## 2. Ruling: the detector-file touch (instruction question)

**Defensible outcome, genuine process breach, no undoing needed.** All three
hunks in `bounce_bot_lib/legacy.py` were read: they add the breadth gap-fill
and follow-up chain sweep inside the evidence-capture threads with
trigger-gated, degrade-to-logging error handling; zero detection or scoring
logic is touched, and the placement was forced by the one-owner-per-thread
invariant. But the standing rule's operative clause was "if a step *seems* to
need it, stop and ask" — the ambiguity was the trigger, the agent recognized
the ambiguity (it flagged the touch in its report), and proceeding anyway
substituted its judgment on exactly the class of call reserved to the trader.

**Rule restated, now file-scoped (recorded in CLAUDE.md):** any edit to a
file housing detector/scoring/alert code gets asked about first, even for
capture-side changes. Cheap to obey; removes the judgment call entirely.

Residual technical item: both sweeps issue historical requests on the
**shared IB client** at close/startup — bounded, but the restart drill must
confirm a sweep coinciding with a scan does not trip IB pacing.

## 3. Amendments required to Branch A before merge

### 3.1 The 07:00 task start is wrong on a Pacific desk

Verified in the branch: repetition was added but the start time is still
07:00 **local**. The desk is US Pacific; the open is 06:30 PT — so the
launcher and its self-heal repetition miss the first 30 minutes of every
session (the highest-value half hour) and idle ~4 hours past the close.
Fix: start ≈ 06:00 PT, shorten the repetition span to end near the close;
re-register once after merge.

### 3.2 Single-instance guard is now load-bearing ~40×/day

The guard matches *python* processes running `launch_gui.py`. Correct today,
but if the desk is ever launched any other way (frozen `TradingBotV3.exe`, a
rename, another interpreter), the guard silently fails and the repetition
starts a **second live desk**: double IB connections, duplicate writers to
every shared export — the one failure mode that corrupts live
decision-support rather than merely losing evidence. This is the review's
**biggest-risk** item. Controls, cheapest first: (a) two-minute drill — fire
the task while the desk is running, confirm "nothing to do"; (b) harden the
guard onto the existing writer-lock machinery, which is the authoritative
"is a desk alive" signal, instead of a process-name regex.

## 4. Ruling: the evidence question (`capture_mode` mid-collection)

**No taint, no reset.** The field is additive with absence-means-live; the
reconstruction path touches only *measurement* rows (+30/60/90 windows,
breadth bars), which are deterministic functions of completed bars.
Prediction-side rows are never reconstructed by construction (Tier C). The
counting declaration is written now — into
`docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md`, not deferred to the future
study:

> A session counts toward the 40-session evidence floor iff its collection
> audit is HEALTHY and all prediction-side events (level tests started /
> resolved) were captured live. `capture_mode: "backfill"` on measurement
> rows (follow-up windows, breadth bars) disqualifies nothing. Rows
> predating the field are live.

## 5. Sequencing of open work

1. **Weekend:** fold the 06:00 PT fix into A; run the fire-while-running
   guard drill; write the evidence declaration (done with this review).
2. **Next morning:** verify tonight's unattended run; any failure disarms
   the task until fixed.
3. **Next quiet session:** restart drill (also observes IB pacing during a
   sweep) → merge A → merge B.
4. Finish Phase 1 (per-ticker briefs, morning summary) on C; let its
   unattended week accumulate; merge C.
5. **Not yet:** Phase 2 digest ledger, Phase 4 policy drafting, journal
   enrichment. Nothing new stacks on unmerged branches, and the
   policy-drafting gate does not begin until the scheduler has a boring week
   behind it.

## 6. Confidence and open inspection

Medium. Raised to high by: tonight's run succeeding (or failing
informatively), a HEALTHY-with-backfill restart drill on a real session, and
a quiet unattended week from the scheduler. Open inspection item: the NAS
publish path in `ai_jobs/store.py` — confirm the twice-observed UNC
path-mangling failure mode is structurally prevented (temp-write-verify-
rename on the share, no shell path assembly), since it now runs ~27×/night
unwatched.

---

## ADDENDUM — Sol 5.6 second review, and the repair-and-merge program

Recorded 2026-08-08, after the program below was executed. This section is
the record of what the second review found and what was done about it; the
sections above are unchanged and remain the original review's verdict.

### What the second review confirmed

Two **P0s**, both in Branch A's setup-tracker staleness catch-up, both
verified against the source and both now repaired and merged:

1. **Scoring side effects in an unattended recovery path.**
   `backfill_setup_tracker_from_recent_sessions` ended by running
   `run_priority_scoring_tuner(apply_changes=True)` and
   `calibrate_expected_r_prior_anchors(persist=True)` — both of which rewrite
   *live* scoring inputs. The automatic catch-up fires unattended, minutes
   before a scan, so a recovery path was retuning the scoring model on its way
   past. The packet's claim that the catch-up was "timing-only" was true of the
   replay and false of the call, and the characterization test could not see
   the difference because it mocked both refits into silence.
   **Repaired:** `run_scoring_side_effects` parameter, default `True` so the
   manual GUI backfill is unchanged; the runner's automatic catch-up passes
   `False`. The tests no longer mock the refits away — the stubs record, and
   the tests assert which path invokes them.

2. **Data vintage read off a write clock.**
   `get_setup_tracker_last_update_session` derived the reflected session from
   `updated_at`, which is only the wall clock at write time. The catch-up
   breaks exactly that assumption: a Friday-morning run rebuilds *Thursday's*
   tracker, the heuristic reports Friday as reflected, and the genuine Friday
   refresh is suppressed for the whole next session.
   **Repaired:** `save_setup_tracker_payload` takes an explicit
   `data_session` — the completed session whose bars produced the payload —
   and `update_setup_tracker_from_scan` passes the scan date it evaluated.
   `updated_at` remains the fallback for legacy payloads only.

Four further findings, all repaired: single-attempt permanent data gaps in
both Tier B recovery paths; a follow-up row for an empty window stamped up to
90 minutes before the absence could be known; follow-up gaps and outcome
coverage invisible inside a HEALTHY verdict; and the single-instance guard
blind to the frozen build. On Branch C: three ways around the "no local
inference during market hours" hard rule, and an evidence packager that was
honest about what it had and silent about what it did not.

### Merge strategy, amended

The original review sequenced merges behind a live-validation session and a
quiet unattended week (sec 5 above). **Amended per Sol: merge early.** The
repairs are the thing that makes the branches safe to run, and leaving them on
unmerged branches while production ran from an unmerged checkout was itself a
risk — the working tree had to stay off `main` and could not be switched
without disarming a scheduled task. Merged in the reviewed order, each with
the full suite green and smoke 7/7:

| Branch | Merge | Suite at merge |
|---|---|---|
| A `durability-catchup` | `5d835ab` | 1901 passed |
| B `local-ai-phase-0` | `b40cad7` | 1927 passed |
| C `local-ai-phase-1` | `13f6e7b` | 2002 passed |

What the original sequencing bought is **not** waived, only re-ordered: 13c is
still **not `LIVE_VALIDATED`**. The mid-session restart drill (audit HEALTHY
with a nonzero backfill count, on a real session) remains the outstanding half
of its exit gate, and Phase 1's exit gate still needs its unattended week.
Commit `9037c5f` (WIP packaging) was **not** merged and stays on
`integration-test` only.

### The AI task during the repairs

**TradingBotV3 AI Jobs** was disabled for the duration and re-enabled only
after a controlled proof run on the real desk (2026-08-08 18:07 and 18:10 PT,
inside the open weekend window, run manually rather than by the scheduler).
The proof is the reason the packaging repair matters:

> package `fa65dceda419d37d`, session 2026-08-08 —
> **7 of 18 sources usable**, 10 unfunded, 1 missing, 5 stale, 5 truncated.
> Ledger row `ok`. Published brief states every one of those, by source id,
> in a `[system]`-prefixed data-quality section the model did not write.

Under the pre-repair code the same package reported "18 source(s)". The ten
unfunded ones were already carrying nothing — the first-come budget had zeroed
them — but they were labelled `available`, so the brief was built on a seventh
of the evidence it claimed. That is now visible instead of implied.

### Open items this addendum does not close

- **The evidence budget is undersized for a local model.**
  `MAX_TOTAL_EVIDENCE_CHARS` is 80,000, tuned when every token was metered.
  Ten real sources — including `setups.playbooks` (404 KB),
  `setups.type_stats` (298 KB) and `setups.scan_factors` (259 KB) — cannot be
  funded at that ceiling. Raising it is a trader decision, not a repair, so it
  was left alone and is stated here instead.
- **`setups.current_tracker` is 762 MB on disk.** Observed while reading the
  proof's coverage block. Whatever else that implies, it means the tracker
  reaches the AI package only as its first 16,000 characters.
- **The journal source is not session-scoped.** It is queried live and spans
  all history, so it is correctly never "stale" — but in the proof run the
  model narrated a 2026-06-18 trade in a summary headed 2026-08-08. Staleness
  detection cannot catch this; scoping the journal query, or telling the
  prompt which session the journal rows should be read against, would.
- The original review's open inspection item (UNC path handling in
  `ai_jobs/store.py`) is untouched and still open.

---

## SECOND ADDENDUM — Sol 5.6 verification review (2026-08-09 record)

The first addendum recorded repairs. This one records what happened when
those repairs were *verified*: one of them had never reached production, and
several of the controls written into the first packet turned out to be
warnings rather than controls. Repair packet 2 is the response.

### Verdict

The 2026-08-08 repairs were real but incompletely landed. The load-bearing
finding is that a fix can pass its own test and do nothing, and that the way
to catch that is to make the test go through the production path rather than
the shape the fix happens to produce.

### Surviving P0 — the vintage never reached the loader

`save_setup_tracker_payload` wrote `data_session`; `load_setup_tracker_payload`
dropped it. The loader rebuilds the payload field by field from a fixed
default rather than copying the stored dict, so anything it does not list is
discarded on read — and **every** caller that resolves the vintage, including
`compute_setup_tracker_catchup_plan`'s no-argument path, goes through it. The
catch-up therefore kept falling back to the `updated_at` write clock, exactly
the behaviour the repair was meant to end.

The tests were the other half of the defect: they asserted on `json.loads` of
the file, so they passed while production did nothing. They now round-trip
through the loader. With the loader line reverted, three of them fail:

```
FAILED test_the_loader_carries_the_vintage_it_was_saved_with
  AssertionError: None != '2026-08-06' : the loader dropped the vintage,
  so the fix never reached production
FAILED test_morning_catchup_from_thursday_does_not_mark_friday_reflected
FAILED test_the_default_catchup_path_reads_the_vintage_off_disk
3 failed, 22 passed
```

`tracker_staleness_catchup` was set to `False` in `local_settings.json` for
the duration and re-enabled only after this packet merged.

### The other nine findings

1. **Session identity had no calendar.** `session_date_for` did weekday
   arithmetic, so a Saturday run filed its work under Saturday — three `ok`
   rows claimed coverage of 2026-08-08, a date the exchange never opened.
   Now: `scripts/market_calendar.py` (weekends plus the ten scheduled NYSE
   closures with observance rules, Good Friday from the Gregorian Easter
   algorithm, Juneteenth from 2022; tested against the exchange's published
   closures for 2024–2027; no new dependency, no network). A run resolves the
   most recent session whose close is at or before run time and **fails
   closed** if the calendar cannot answer. `--force` writes `manual_test`,
   which is outside `CANONICAL_COMPLETION_STATUSES` and so never satisfies the
   completion check. The three Saturday rows were annotated with an appended
   `correction` row (`2026-08-08T19:07:10-07:00`); the ledger is append-only,
   so the originals stand and the artifacts they produced are kept — only
   their claim to cover a session is retracted.
2. **The journal was not session-scoped.** Flagged as an open item in the
   first addendum; now closed. Records are filtered to the session by the
   store's own SQL, an empty result is an honest empty source, and the journal
   is treated as what it is — a database that is current only through its last
   successful import. A stalled import makes it stale and hides its old rows,
   and a `[system]` import-health row reports last import, newest execution,
   lag in days and the session's row count.
3. **`observed_at` and `content_through` are now separate.** One is when the
   process read the source; the other is the newest record inside it, derived
   from content where the format allows and from mtime only when it does not.
   Staleness is judged on `content_through` alone, so a file rewritten nightly
   with unchanged data no longer reads as current.
4. **`data_quality` is machine-owned.** Out of the model's schema, out of the
   prompt, rejected by the validator if returned anyway. The model keeps
   `risk_notes`.
5. **Stale sources leave the model package.** They used to stay in with a
   warning notice; the model narrated them as the session's data regardless.
   A warning the model may disregard is not a control.
6. **Retry semantics.** Partial responses now retry like failures — the
   completeness check compares bars received against bars the matured windows
   owe. Retry entitlement is **per symbol and spans sweeps**; the old shared
   wall-clock budget rationed the wrong thing, letting the first few failures
   consume it so every symbol after them got one attempt and a permanent gap.
   A symbol with entitlement left is **deferred** (still pending, no gap row),
   and the sweep marker is written only when nothing is deferred.
7. **Unknown job status failed open.** It coerced to `ok`, so an unrecognised
   status was filed as a trustworthy completion and never retried. It records
   `failed` and names the status it did not recognise.
8. **Unbounded reads and the wrong funding order.** The 762 MB tracker was
   read whole to keep 16,000 characters, and those characters were a list of
   *March* watchlists because the file leads with `daily_watchlists`. Reads
   stream to a byte cap; oversized files are identified by size, mtime and a
   digest of the capped content; the tracker is packaged as a bounded
   most-recent extract from the compact scoring snapshot, or declared
   unavailable — never head-sliced. Within `setup_trackers` the analytic
   sub-sources fund first and the raw tracker last.
9. **The pre-open reserve was zero.** The session block protected the session
   but not the run-up to it. `ai_preopen_guard_minutes` defaults to 15, and
   because the reserve lives inside the session block, `--force` cannot spend
   it.

### What this addendum does not close

- The evidence budget is still 80,000 characters, still undersized for a local
  model, and still a trader decision rather than a repair.
- `setups.current_tracker` is still 762 MB on disk. The packaging no longer
  chokes on it, but nothing has been done about the file itself.
- The unscheduled-closure limit of a rules-based calendar stands and is stated
  in `market_calendar`: a day the exchange shut unexpectedly reads as a
  session. Nothing derivable from rules can fix that.
- The original review's UNC-path inspection item in `ai_jobs/store.py` is
  still open.
