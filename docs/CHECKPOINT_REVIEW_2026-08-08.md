# Checkpoint review — 2026-08-08 (items 13b/13c build, branches A/B/C)

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
