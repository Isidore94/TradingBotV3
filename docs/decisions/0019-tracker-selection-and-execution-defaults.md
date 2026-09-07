# 0019 — The Setup Tracker's three replay defaults become the repaired ones

Date: 2026-09-06

Amends nothing in decisions 0001–0018. It moves three DEFAULTS that packets ST3
and ST4 (2026-09-06) had deliberately left on the shipped behaviour while the
evidence was gathered. Decision 0008 (the frozen anchored-VWAP sigma formula) is
untouched: nothing here changes one line of `calc_anchored_vwap_bands` or
`calc_anchored_vwap_band_history`. Decision 0009 (golden fixtures before a
detector change) is honoured by construction — every old golden is still pinned
and is now read with the old policy NAMED.

## Who decided what

The trader, 2026-09-06 ~21:15 PT, verbatim:

> *"Yes a trade not yet completed should say pending. A second entry after a
> first close is its own trade yes. 3. This one is up to your discretion."*

That is decisions (1) and (2). Decision (3) — the execution convention and the
level knowledge — was taken by the lead session under the discretion the trader
granted, on the reasoning below.

The trader's covering instruction for the whole ST series still stands and is
not overridden here: *"Do not restart the live desk, rewrite live evidence,
change order execution boundaries, or claim promotion from this prompt alone."*
Nothing in this record promotes a setup, and nothing in it reaches a detector, a
score, an alert, a watchlist, Focus, the review queue or `review_policy.json`.

## The three defaults

| Axis | Was | Is | Where |
|---|---|---|---|
| Selection | `closed_first_v1` | **`first_actionable_v2`** | `master_avwap_lib/selection_policy.py` |
| Execution | `literal_level_v1` | **`gap_aware_v2`** | `master_avwap_lib/execution_convention.py` |
| Level knowledge | `same_session_v1` | **`prior_session_v2`** | `master_avwap_lib/execution_convention.py` |

Each v1 name keeps its value, stays selectable by keyword, and is the arm
`scripts/tracker_selection_compare.py` / `scripts/tracker_execution_compare.py`
call "old".

## Why

**(1) Pending stays pending.** `_summarize_tracker_setup_outcome` reported an
OPEN representative's R as the mean of whichever ALTERNATE exit plans on the
same stop had closed. A position still on the books was therefore printed as a
realized result — usually a loss, because a stop closes before a target does. On
the 2026-09-03 mirror, 271 of 2,712 v2 episodes had a pending representative and
were being graded anyway: 252 as losses, 19 as wins. The same 2,249 theses move
from **61.6% to 72.0% favorable**, and **94% of the mean-R move comes from
pending-stays-pending alone**. It is not a new opinion about exits; it is the
removal of a number the desk never measured.

**(2) A second entry after a first close is its own trade.** The tracker
rescans a thesis every day it still looks like a setup, and
`_dedupe_recent_tracker_family_rows` collapsed those rows by *"prefer a row that
has closed, then the earliest scan date"* — a selection that reads the OUTCOME.
A later rescan that happened to resolve beat the earlier row the trader could
actually have taken. `first_actionable_v2` fixes the episode BEFORE any outcome
is known: attempt 1 is the earliest scan row, and attempt k+1 opens only when
the previous attempt's representative had already closed. The ST4 comparison
found **2,249 → 2,712 episodes**, the 463 extra being genuine second attempts.

**(3) Lead's discretion: both execution repairs.** Each is a pure correctness
fix, not a modelling preference:

* `gap_aware_v2` — the replay booked a hard stop AT the stop level even when the
  whole bar traded below it. Entry 100, risk 5, stop 95, next bar
  O80/H85/L79/C82 booked −1.014R where the honest fill is the open at 80 for
  −4.014R. That is the tail of the distribution being deleted. v2 books the open
  on a gap, clamps when there is no open, and books NOTHING off a candle whose
  own four prices contradict each other — counting the skip rather than reading
  it as "not hit". It is symmetric: a target that gaps through fills BETTER.
* `prior_session_v2` — an intrabar high/low test was run against
  `band_history[<that same day>]`, whose anchored-VWAP bands are computed with
  that day's own bar folded into the cumulative sums. The level is knowable only
  at that day's close. Close-based decisions (the two-closes protective stop,
  the maximum-hold force close) keep day D's levels, because at the close they
  ARE known.

On 794 setups the ST3 comparison measured **472 changed**: execution-only moved
89 setups and raw expectancy **−0.098 → −0.081**; levels-only moved 458 and
**−0.098 → −0.149**. The levels repair makes the number WORSE, and that is the
point — keeping a known look-ahead in the champion's replay because the honest
number is uglier is exactly the failure `docs/decisions/0016` goal 8 names.

## What this restates, and what it does not

**History IS restated, by construction.** The tracker rebuilds every record on
each persisted write, so the first write after this lands re-grades the stored
history under the new defaults. That is the intent, not a side effect: the old
numbers were measuring something the trader could not have traded.

A compact scoring projection written under the OLD defaults carries no policy
stamp. It is read AS-IS — the cache is the only copy of the answer for the live
scoring path — and the next persisted tracker write rebuilds it. The write logs
`policies: selection=… execution=… levels=…` so which generation is on disk is
readable rather than inferred.

**Survivorship caveat, recorded because it bounds claim (2).** The 463 second
attempts win about 80% AFTER the fix, but that population exists only because
the scanner re-listed the thesis, which it does when the setup still looks
good. Those attempts are not a random sample of "re-entries" and the 80% is not
an edge estimate. It is the count of what the tracker was previously hiding.

**Every record now NAMES its policies.** Before, a stamp was written only for a
non-default policy and popped otherwise. After a default flip an absent stamp is
ambiguous, so `execution_convention` and `level_knowledge` are written on every
record and `selection_policy` on every family row and outcome summary.

## Rollback

One switch by name, in two files: set `DEFAULT_SELECTION_POLICY`,
`DEFAULT_EXECUTION_CONVENTION` and `DEFAULT_LEVEL_KNOWLEDGE` back to the v1
constants. The v1 code paths are unchanged and are pinned by
`tests/fixtures/st3_replay_golden.json` and
`tests/fixtures/st4_family_rows_golden.csv`, both now read with the v1 policy
passed explicitly. The new defaults are pinned by
`tests/fixtures/st7_v2_default_golden.json` and
`tests/fixtures/st7_family_rows_v2_default_golden.csv`, both frozen on `main`
at `68762909` through the explicit v2 keywords — before the flip existed, so
neither is a self-portrait of the code it checks.

## Live gate

Gate #84 (`CURRENT_CHECKPOINT.md`): the first persisted tracker write after
merge logs `policies: selection=first_actionable_v2 execution=gap_aware_v2
levels=prior_session_v2`, every record carries the three stamps, the recent
family rows' `n_pending` is non-zero, no row whose `representative_status` is
`pending` is graded, and the Setup Types / recent tables re-rank in the ST4
comparison's order.
