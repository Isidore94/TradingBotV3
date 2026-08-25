# Post-attack authorization — 2026-08-25 (afternoon)

**Hand-committed and frozen.** Trader decisions made after Fable's independent
review of Sol's adversarial pass (`SOL_ATTACK_2026-08-24.md`; review ruling
ACCEPT WITH BLOCKERS). Recorded verbatim so the repair packets cite one source.

## Decision A — what makes a sweep-finalized trade usable (trading rule)

The after-close sweep finalizes with a blank eod-hold `close_r` by design
(`no_eod_close`); the stop-out R and the last-measured close live in
`context.exit`. Trader rule: **a stop-out counts at its `stop_exit_r`; any
other sweep-finalized trade counts at the R of its last measured close.** The
two are reported as their own frozen exit policies (`stop_exit`,
`last_measured`) beside `eod_hold`, **never blended**, each with the full
ground-rule-10 statistics. A trade is `usable` when at least one policy has a
measured value; `unresolved` rows stay unusable. Numbers the trader has seen
will move (today's 656 finals go from 0 usable to whatever is measured) and
the change is printed before/after, never silent.

## Decision B — ask-first answered for `bounce_bot_lib/legacy.py`

Two evidence-side repairs are authorized in the fenced file, nothing else:

1. **Milestone recovery must not erase a recorded stop** (Sol T3 / Fable B3):
   `stop_hit` is `any()` across the trade's milestone rows; the exit is taken
   from the earliest stop-hit row (R10.0's stop-first decision governs). The
   35 already-written finals from the 2026-08-24 sweep are **tagged by a
   versioned reader-side rule, never rewritten** (ground rule 5).
2. **Signal-bar recovery must match the event's bar** (Sol T2 / Fable B4):
   `_signal_bar_dict` also requires `bar.dt == event.bar_time`; a mismatch
   yields the fallback.

Golden fixtures byte-identical before and after; no alert, tier, fold, digest
or queue behavior changes.

## Decision C — unfenced builds and record corrections authorized

- Wire the AWAY Recap to the Alert Center backing list (one call at page
  selection) with a `MainWindow`-level regression test.
- Correct the record: the restarted-process outcome sweep **ran on
  2026-08-25** (`swept_at 14:27:36-07:00`, 656/656 finalized, 0 failed, 0
  commit-failed, `pending_after 0`); the three documents that say the canary
  FAILED are wrong. It started 52 minutes after its due time for a cause not
  yet found; the observation is recorded, the gate is the trader's to accept.
- Restore `DESK_TESTING_PLAN.md` §2.3 to its honest state: the AWAY staging
  half observed, the flip-back-to-DESK half still owed.

## Not decided / not authorized

No change to any threshold, detector, scorer or alert; no scheduling "repair"
(the due logic is correct — an investigation of what held the loop from 13:30
to 14:27 is authorized, a fix is not); Questrade's acceptance of the DateTime
format is read from tonight's ledger, not assumed.
