# CHANGELOG archive - 2026-09-06 (ST6)

Moved out of `CHANGELOG.md`'s Recent changes on 2026-09-08 so that section holds two build days. History only; never context.

### 2026-09-06 - ST6: one Working-lately snapshot, four surfaces, and a switch that only reorders (branch `claude/st6-working-lately`, not merged)

**Re-review fix round (same day, same branch).** Four blockers and eight
advisories; fourteen tests in `tests/test_st6_fix_round.py`, every one proven RED
against the reviewed tip first.

- **The switch is a VIEW, not a mutation.** Both the M5 bar and the waiting
  review list re-sorted their BACKING lists, so turning the switch off could not
  put them back: the bar returned early when disabled and stayed sorted for the
  session, and `_review_queue` was rebound to the sorted order permanently. The
  bar now keeps `_arrival` and draws a view of it; the review queue is never
  reordered and `_next_review_index` picks the next chart by rank instead.
- **The cap applies to the arrival list.** The `MAX_ROWS` trim ran on the sorted
  list, so the switch decided WHICH alert stopped existing on the bar.
- **The day-trade bound is on `held_run_score` itself.** It was the bootstrap of
  the held episodes' MFEs - a different quantity - so live it printed
  `held x ran 1.21 (>= 2.070)`, a lower bound ABOVE its own statistic, and
  ranking on it crowned a different cell from the headline's leader.
  `evidence_stats.session_block_statistic_bootstrap` resamples whole sessions and
  recomputes a caller's statistic; `Segment.score_bootstrap` recomputes
  hold_rate x trimmed-mean MFE per draw. The day-trade kind ranks on the
  STATISTIC with a declared `LEADER_MARGIN_HELD_RUN_R` (0.10, score units) -
  the 0.05 win-rate margin is a margin on a quantity bounded in [0, 1].
- **`swing_favorable` is dated by its MEASURED session** (`future_scan_date` /
  `target_session`), not by the entry: dated by the entry, a horizon-5 file whose
  newest scan was 5 sessions back was stale by construction and the kind could
  never lead. And `snapshot_line` prints all three kinds ALWAYS - a withheld kind
  says `no evidence - <reason>` rather than vanishing from the strip.
- Advisories: the observational caveat counts one KIND's cells; the payload went
  from 133 KB to 44,411 bytes on a pessimistic 150-cell fixture (`_kind_policy`
  lifts every field a kind's cells all agree on, losslessly, and the declared cap
  is 48,000); a counts-only export says `concentration unmeasured`; the strip
  builds its tooltip once, sets no stylesheet and renders 150 cells in under
  5 ms; `AutopilotService.setupTrackerWritten` is the CLOSE-SLOT trigger the
  manual scan service never fires; `built_at` and every event `ts` are
  market-local and aware; gate #83 says what a PASS looks like on day one.
- The snapshot now FEEDS `panel_verdicts`, so ST2's one-computation-per-page
  design holds with the shared reading as its source.
