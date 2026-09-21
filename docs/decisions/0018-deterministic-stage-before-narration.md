# 0018 — The overnight run does its deterministic work before it narrates

Date: 2026-09-04

Amends nothing in decisions 0001–0017. It replaces one written-down ordering
rule: `ai_jobs.runner.default_slots()`'s "later phases append to this list;
they never reorder", which `CLAUDE.md`/`AGENTS.md` also carried. That rule was
never a safety property — it was a promise that nobody would silently move a
slot without saying why. This record is the saying-why.

Authorized by the trader on 2026-09-04 (*"please review and implement the
suggested changes"*) over
[`docs/analysis/PROJECT_PROCESS_REVIEW_2026-09-04.md`](../analysis/PROJECT_PROCESS_REVIEW_2026-09-04.md),
findings 6 and 7. Built as packet Q4 on `claude/q4-overnight-gates`.

## Context

The nightly slate ran in the order the slots were written, which for the first
two slots was "first, because they were first".

| Slot | Reserve | Model? |
|---|---|---|
| `journal_import` | 5 min | no |
| `journal_auto_tag` | 5 min | no |
| **`ai_summary`** | **dynamic — up to ~170 min in chunked mode** | **yes** |
| **`ticker_briefs`** | **120 min** | **yes** |
| seven cohort / audit / join slots | 5 min each | no |
| `evidence_report` | 5 min | no |
| `daily_digest` | 10 min | facts no, narration yes |
| `journal_enrichment`, `review_policy_draft`, `setup_research` | 20 / 10 / 20 min | gated |

Three facts made that order a real risk rather than an aesthetic one:

1. **The window is finite and a slot that cannot fit its reserve records
   SKIPPED** (`runner.py`, the reservation check). It is not queued, not
   shortened, not retried later in the night — the night simply does not do it.
2. **The night has actually run long.** On 2026-09-01 the run took six hours.
   The two narration slots hold up to two and a half hours of reserve between
   them and sat ahead of every deterministic slot, so the work most likely to
   be skipped was the cheap, deterministic, evidence-writing work.
3. **No deterministic slot reads either narration slot's output.** Verified at
   the code level, not inferred from the docs: nothing under `ai_jobs/` opens
   the summary's or the briefs' published files. `daily_digest` imports
   `ai_summary` as a LIBRARY to narrate its own fact pack; it reads no file
   either slot wrote. The dependency that would have justified the old position
   does not exist.

The cost of the old order is asymmetric. A skipped narration is a convenience
lost for one night and regenerable the next. A skipped cohort grade, sidecar
completion, preference join or fact pack is a **hole in an append-only forward
record** — the session it would have measured is over, and Phase 2's collection
window (ten consecutive clean sessions) restarts.

## Decision

**`default_slots()` runs in three stages.**

1. **Deterministic** — `journal_import`, `journal_auto_tag`,
   `veto_cohort_grading`, `like_cohort_grading`, `sidecar_completion`,
   `pass_cohort_grading`, `rejection_cohort_grading`, `note_vocabulary_audit`,
   `preference_trade_outcomes`, `evidence_report`, `daily_digest`.
2. **Narration** — `ai_summary`, then `ticker_briefs`.
3. **Model-gated** — `journal_enrichment`, `review_policy_draft`,
   `setup_research`.

**The relative order inside stage 1 is unchanged.** It is the appended-only
order every previous packet argued for, and the comments in `default_slots()`
still carry each position's reason: the import is first because everything
reads the journal, tagging is second for the same reason read from the other
end, `sidecar_completion` precedes `pass_cohort_grading` because it feeds it,
`preference_trade_outcomes` follows the cohorts whose outcome files it reads,
`evidence_report` follows all of them, and `daily_digest` is last in the stage
because its fact pack describes the night the stage just had.

**The narration pair moves as a UNIT** and keeps its internal order.
**Every slot's reserve and retry budget is unchanged.**

**The replacement rule, and it is now the one in `CLAUDE.md`/`AGENTS.md`:**
*the order is decision 0018's — deterministic slots, then the digest, then
narration, then the model-gated slots; a later phase appends inside its stage
and never reorders across stages.*

The order is pinned in exactly one place in the tests —
`EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py` — so the next packet
edits one tuple rather than the six index assertions this change had to touch.

## What is NOT decided here

* **No deadline or resume redesign.** The reservation check, the retry budgets,
  the attempt caps and the window logic are untouched. This record moves slots;
  it does not change how the runner decides to run one.
* **`ticker_briefs` keeps its resumable design.** Its chunking, its own
  high-water marks and its attempt cap are exactly as TB-0..TB-6 left them.
  Moving it later in the night does not make it less resumable and does not
  change what it resumes from.
* **No automatic frontier synthesis.** `weekly_synthesis` remains in
  `optional_slots()`, never in the nightly slate, exactly as
  `docs/LOCAL_AI_AUTOMATION_PLAN.md` §7.3 left it.
* **`daily_digest` stays where it is in the deterministic stage and stays
  deterministic in its FACTS.** Its narration is that slot's second artifact,
  it reads only the fact pack, and a dead model still leaves the pack written
  and the row `degraded_no_narrative`. Nothing about that changed.
* **Nothing here reaches a detector, a score, an alert, a watchlist, Focus, the
  review queue or `review_policy.json`.** This is a job ORDER, and the jobs it
  orders are the same jobs.

## Reopen triggers

* **A deterministic slot ever needs a narration slot's OUTPUT file.** Then the
  stage boundary is wrong and the dependency, not the stage, decides. Say which
  file, and move the reader — not the stage.
* **Narration starts being skipped for want of window on a majority of
  nights.** The trade this record makes is "narration loses the window before
  evidence does"; if narration then never runs, the answer is a shorter or
  cheaper narration, or a second window, not a return to the old order.
* **The window stops being the binding constraint** — a faster endpoint, a
  longer window, or `ai_summary` losing its chunked mode. Then the ordering
  buys nothing and could be revisited; it still costs nothing, so there would
  have to be another reason.
* **A future stage appears** (a fourth kind of slot). Then this record's list
  of three stages needs a fourth entry, and it is an edit here first.

## References

* `scripts/ai_jobs/runner.py` — `default_slots()`, the three-stage list.
* `tests/test_ai_jobs_runner.py` — `EXPECTED_SLOT_ORDER`, the single pin.
* `docs/LOCAL_AI_AUTOMATION_PLAN.md` §3.4, §6.4c, §7 — the window, the
  journal-pull exception, the deterministic slots.
* `docs/analysis/PROJECT_PROCESS_REVIEW_2026-09-04.md` findings 6 and 7.

## Amendment 2026-09-19 — a night's slate, and which night it is (TJ-13A)

Authorized by the trader on 2026-09-19 (decision 0021 answers 18-19; plan.md
§12.4 TJ-13 items 5, 6 and 8). Built on `claude/tj13a-night-slates`, merged at
`9eaae1dd`. It amends WHICH SLOTS a night holds. It moves no stage boundary and
reorders no slot: `EXPECTED_SLOT_ORDER` is unchanged and every slate below is a
subsequence of it. It changes ONE retry budget — `ai_summary` gains
`max_attempts=3`, for the reason given at the end.

**The order in this record is the order WITHIN a night.** A slate is a CHOICE of
slots, made first; the three stages then order whatever was chosen.

**Three night kinds**, named on the exchange calendar by
`ai_jobs.runner.night_kind()` from the evening the night STARTED — noon ET splits
one night from the next, so Saturday night's 02:00 firing is still Saturday's:

* `weeknight` — the evening was a session.
* `sunday` — it was not and the next day is. A Monday holiday moves this to
  Monday night.
* `saturday` — neither. A Friday holiday starts it a night early, and in that
  week Friday AND Saturday night are both `saturday`; the second is the resume
  night.

**Their slates** (`ai_jobs.runner.slots_for()`):

| night | slate |
|---|---|
| weeknight | the whole slate except `ai_summary` |
| saturday | the whole slate, plus `weekly_synthesis` at the end of stage 3 |
| sunday | stage 1, plus a retry of any slot this weekend ATTEMPTED, did not finish, and is still inside its cap |

A slot that never ran is not "owed": Sunday is the backlog, not a second weekly
slate. Both weekend nights key to the same session date — Friday's — which is
what makes the ledger the honest record of what the weekend still owes.

**Why `ai_summary` leaves the weeknight slate.** Measured on the live ledger
2026-09-19: 12,453–18,540 s a night, ending `degraded_no_narrative` on 09-15,
09-16, 09-17 and 09-18. It is the slot the night cannot afford five times a
week; on Saturday it has the whole night in front of it. **It therefore runs for
ONE session a week — Friday's, on Saturday night — so Monday through Thursday
sessions get no AI summary at all.** Its new `max_attempts=3` is three attempts
at that one session: fail-fast (below) made a failing summary cheap, and cheap
plus unbounded is a loop — a weekend night's ~16 firings ran a degrading summary
TEN times in one simulated Saturday before the cap.

**Why `weekly_synthesis` stops needing a typed command.** This record's "What is
NOT decided here" said it "remains in `optional_slots()`, never in the nightly
slate". That clause is superseded FOR THE SATURDAY SLATE ONLY: in 476 ledger
rows it had never run, because the only way to reach it was
`run_ai_jobs.py --weekly-synthesis`. A safeguard nobody can reach past is not a
safeguard. `optional_slots()` is still constructed per call and still absent
from `default_slots()`, so it cannot leak onto a weeknight.

**A typed `--slot` is not filtered by the slate.** It resolves against
`default_slots() + optional_slots()` — every registered slot — and an unknown
name is an error exit that lists the valid ones. A slate is what the night does
UNATTENDED; naming a slot is the operator's explicit choice. This widens what
can be NAMED and never what can RUN: a `uses_model` slot named by day is still
refused by the night-only window.

**One reopen trigger is added:** if a night kind is ever chosen by anything
other than the exchange calendar, `night_kind` is the one place it changes.

## Amendment 2026-09-19 — `miss_contrast` joins stage 1, above the pair that closes it (TJ-15)

Built as packet TJ-15 (`plan.md` 12.4), merged into `lead/p033-integration2`;
the slot's position is the lead's integration fix `1f260ffa`.

**The stage-1 list, as `default_slots()` now spells it:** `journal_import`,
`journal_auto_tag`, `veto_cohort_grading`, `like_cohort_grading`,
`sidecar_completion`, `pass_cohort_grading`, `rejection_cohort_grading`,
`note_vocabulary_audit`, `preference_trade_outcomes`, `evidence_report`,
`daily_digest`, `theta_pick_grading` (WS-TH), **`miss_contrast` (TJ-15)**,
`market_story_rollups` (WS-10D), `measured_report` (WS-RP).

**`miss_contrast` sits after the cohort graders and BEFORE the
`market_story_rollups` + `measured_report` pair.** It reads what a decision
turned OUT to be, so it belongs after the graders; it calls no model
(`uses_model=False`, `max_attempts=3`, `reserve_minutes=5.0`) and nothing below
it reads its pack, so it stays ahead of `ai_summary`. Its place ABOVE the
closing pair is load-bearing twice over:

* WS-10D and WS-RP each pin `measured_report` DIRECTLY after
  `market_story_rollups`, and `tests/test_veto_cohort_grading.py` pins the whole
  slate. The packet first appended the slot after that pair; three order
  assertions went red in the full suite and the lead moved it above them.
* `runner._STAGE_ONE_LAST_SLOT` is `measured_report` and
  `runner._deterministic_stage` walks the slate up to and INCLUDING that name,
  so a slot appended AFTER it is not in stage 1 by that function's reckoning and
  silently leaves the **Sunday** slate, however deterministic it is.

`_STAGE_ONE_LAST_SLOT` is unchanged and `measured_report` still closes the
stage. Nothing in `miss_contrast` reads the measured report and nothing in the
measured report reads this pack, so only the order was ever a choice — and only
one of the two orders runs on a Sunday. **A later packet appending inside stage
1 lands in the same place, for the same reason**, and runs `-k "slot or stage or
slate"` plus the three pinning test files above, not only
`tests/test_ai_jobs_runner.py`.

## Addendum 2026-09-20 — `read_grades_mature` joins stage 1, between `theta_pick_grading` and `miss_contrast` (TJ-10)

Built as packet TJ-10 (`plan.md` 12.4), merged into `lead/p033-integration2`
(`57b44ca9`); the slot is registered in that packet, not by a later fix.

**The stage-1 list, as `default_slots()` now spells it:** `journal_import`,
`journal_auto_tag`, `veto_cohort_grading`, `like_cohort_grading`,
`sidecar_completion`, `pass_cohort_grading`, `rejection_cohort_grading`,
`note_vocabulary_audit`, `preference_trade_outcomes`, `evidence_report`,
`daily_digest`, `theta_pick_grading` (WS-TH), **`read_grades_mature` (TJ-10)**,
`miss_contrast` (TJ-15), `market_story_rollups` (WS-10D),
`measured_report` (WS-RP).

**`read_grades_mature` sits directly after `theta_pick_grading` and before
`miss_contrast`.** It closes the market reads whose horizon has matured — a
five-session call made on Friday cannot be graded until the fifth session closes
— so it is the same kind of work as the cohort graders (a decision, measured
after the fact) and belongs beside them. It calls no model
(`uses_model=False`, `max_attempts=3`, `reserve_minutes=2.0`), so it stays ahead
of `ai_summary`; nothing below it reads the read ledger and the ledger reads
nothing above it, so only the position was ever a choice. It is ABOVE the
closing pair for the reason the 2026-09-19 amendment gives: `_STAGE_ONE_LAST_SLOT`
is `measured_report`, `_deterministic_stage` walks up to and INCLUDING that name,
and a slot appended after it silently leaves the **Sunday** slate however
deterministic it is. `_STAGE_ONE_LAST_SLOT` is unchanged.

A night that cannot measure writes NOTHING: a missing reads directory, an
unreadable ledger or a missing daily store each give an `ok` row with a reason in
under a second, never an exception into the runner and never a verdict, because
an `unmeasured` result may not supersede a `pending` row (plan.md sec 5). The
slot is idempotent across the task's 30-minute re-firings and the only file it
writes is `DAY_REVIEW_READS_DIR/<session>.jsonl`. At integration the lead also
added the name to TJ-13B's two weeknight-slate pins in
`tests/test_tj13b_local_large_provider.py`; the order pins to run remain
`tests/test_ai_jobs_runner.py` (`EXPECTED_SLOT_ORDER`),
`tests/test_veto_cohort_grading.py`, `tests/test_ws_10d_market_story.py` and
`tests/test_ws_rp_shared_report.py`.

## Addendum 2026-09-20 — `prediction_contrast` joins stage 1 and `observation_tags` joins stage 2 (TJ-16)

Built as packet TJ-16 (`plan.md` 12.4), merged into `lead/p033-integration2`
(`c4a760e5`). Two slots, one in each stage, and the stage boundary itself is
untouched: `_STAGE_ONE_LAST_SLOT` is still `measured_report`.

**`prediction_contrast` sits DIRECTLY after `miss_contrast`**, still above the
`market_story_rollups` + `measured_report` pair that closes stage 1. It is
deterministic (`uses_model=False`, `max_attempts=3`, `reserve_minutes=5.0`) and
reads the read ledger that `read_grades_mature`, two slots above it, has already
closed for the night — so the order is a data dependency, not a preference. It
is above the closing pair for the reason the 2026-09-19 amendment gives: a slot
appended after `measured_report` silently leaves the **Sunday** slate however
deterministic it is.

**`observation_tags` sits INSIDE stage 2, after `ai_summary` and before
`ticker_briefs`.** After `ai_summary` because WS-10D pins `measured_report`
directly before it and that pair must stay adjacent; before `ticker_briefs`
because the briefs hold 120 minutes of reserve while this is seconds of work per
note, so queueing behind them would cost the tags a whole night for nothing. It
loads a local MEDIUM model, so `uses_model=True` is declared honestly and
`--force` may not buy it the daytime clock (TJ-13A item 1); it has no
deterministic half, so no `model_free_kwargs`. A rejected reply publishes
nothing and the last verified file stands byte-identical. **Nothing it writes
reaches a detector, score, alert, watchlist, Focus, the review queue or
`review_policy.json`**, and its codes reach `prediction_contrast` only on the
NEXT night, the tagger being stage 2 and the contrast stage 1.

**The narration pair, and the seventh order pin.** `ai_summary` and
`ticker_briefs` keep their ORDER, but they are no longer adjacent: the lead
amended `tests/test_opt_in_evidence_scopes.py` at this merge so that only
`day_review_narration` (TJ-4) and `observation_tags` (TJ-16) may sit between
them, and nothing else. The order pins to run are therefore SEVEN files —
`tests/test_ai_jobs_runner.py` (`EXPECTED_SLOT_ORDER`),
`tests/test_veto_cohort_grading.py`, `tests/test_ws_10d_market_story.py`,
`tests/test_ws_rp_shared_report.py`, `tests/test_tj13b_local_large_provider.py`,
`tests/test_tj13b_probe_guards.py` and `tests/test_opt_in_evidence_scopes.py` —
with `-k "slot or stage or slate or order"` beside them.

**EXTENDED 2026-09-20 (TJ-5): the pair now holds THREE slots, and there is an
eighth pin.** `week_review_narration` (TJ-5, merged into
`lead/p033-integration2` `1b9d77e0`) joins stage 2 between `observation_tags`
and `ticker_briefs`, so the slots that may sit between `ai_summary` and
`ticker_briefs` are, IN THIS ORDER: **`day_review_narration`,
`observation_tags`, `week_review_narration`** — and nothing else. It is ahead of
the briefs because they reserve 120 minutes while the week story is what the
trader opens Weekend Prep to read, and it cannot move further forward because
`measured_report` sits directly before `ai_summary` and two files pin that pair.
It is SATURDAY-slate only, through the EXISTING `runner.WEEKEND_ONLY_SLOTS` and
no second constant, and Sunday offers it only when Saturday attempted it and did
not finish. It loads a local model (`uses_model=True`, no `model_free_kwargs`,
so `--force` may not buy it the daytime clock) and its reserve is
`TIMEOUT_SECONDS / 60 + RESERVE_MARGIN_MINUTES`. **`tests/test_tj5_week_slot_and_slate.py`
is now an EIGHTH order pin that a future slot packet must respect**: it asserts
the ADJACENCY `observation_tags` → `week_review_narration` → `ticker_briefs`,
so a new stage-2 slot inserted anywhere among those three fails there even when
`EXPECTED_SLOT_ORDER` has been updated. The three files amended at this merge
were `tests/test_ai_jobs_runner.py`, `tests/test_veto_cohort_grading.py` and
`tests/test_opt_in_evidence_scopes.py`; the other five needed no edit.

**EXTENDED again 2026-09-20 (TJ-6): stage 3 now ends
`…, setup_research, improvement_ideas`, and NINE places pin slot order.**
`improvement_ideas` (TJ-6, merged into `lead/p033-integration2` `a7809d7c`) is
appended LAST inside stage 3, behind `setup_research` — a later phase appending
inside its own stage, which is this decision's rule. It is model-gated
(`uses_model=True`, no `model_free_kwargs`, so `--force` may not buy it the
daytime clock), `max_attempts=2`, `RESERVE_MINUTES = 10.0`, and it is NOT in
`WEEKEND_ONLY_SLOTS`: the packet says up to three ideas A NIGHT, so every
night's slate carries it and Sunday offers it only when the weekend left it
owed. It sits at the END because it READS what the night just wrote — the day
packs, the stories and the contrast packs — and because it is the one slot whose
output nothing else consumes: **nothing it writes reaches a detector, score,
alert, watchlist, Focus, the review queue or `review_policy.json`**, and only
the trader's own click on the card keeps an idea. A night with nothing to cite
answers `skipped` BEFORE any model load; a night that asked ends `ok` (with an
asked marker beside the store) so the 30-minute task asks once.
**The NINTH order pin is `tests/test_setup_research_pipeline.py`** (lead
amendment `be435cd7`): its `test_setup_research_is_appended_to_the_nightly_slate`
asserted `setup_research` was the LAST slot outright and now asserts that
`setup_research` still ends stage 3 with **only `improvement_ideas` allowed to
follow it**, and that it still sits after `review_policy_draft`. The nine pins
are therefore `tests/test_ai_jobs_runner.py`, `tests/test_veto_cohort_grading.py`,
`tests/test_ws_10d_market_story.py`, `tests/test_ws_rp_shared_report.py`,
`tests/test_tj13b_local_large_provider.py`, `tests/test_tj13b_probe_guards.py`,
`tests/test_opt_in_evidence_scopes.py`, `tests/test_tj5_week_slot_and_slate.py`
and `tests/test_setup_research_pipeline.py`. The four files amended at this merge
were `tests/test_ai_jobs_runner.py`, `tests/test_veto_cohort_grading.py`,
`tests/test_tj13b_local_large_provider.py` and `tests/test_tj13b_probe_guards.py`
(the name went into `set_aside`, never into `pinned_at_e8c04f88`); the ninth was
found by the full suite, in a file TJ-6 never touched.
