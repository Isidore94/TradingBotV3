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
