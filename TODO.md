# TODO

The next work, in order. Delete an item when it's done; don't archive it. Only the
trader adds items. An idea in `WISHLIST.md` becomes work only when the trader moves it
here.

## Waiting on live checks (nothing to build)

- **Read the owed live gates** in `docs/GATES.md`, newest first. Green tests don't
  count as live proof.
- **SP4 score challenger:** freeze a prospective challenger before looking at any
  results. The first trial is 20 new entry sessions plus a 5-session outcome wait, with
  success, downside and rollback limits fixed in advance. No weight tuning from the
  4-session sample.
- **TJ-8 cleanup** (after gates #145–#150 pass): delete
  `ui/panels/market_journal_panel.py`, `daily_recap_panel.py`, the page class in
  `away_recap_panel.py` (the phone digest path stays), their dead tests and the
  four-pane capture reader. Update `docs/RULES.md` and the packaging drift guard. No
  behaviour change.

## Planned, not scheduled

- **TJ-12F:** read the Focus-add and armed-alert lanes in `trade_origin`. Today an
  unread lane looks like `unplanned`.
- **TJ-14C:** give the `quick_like_followup` Mentor kind a reader, then wake it by
  setting `dormant_until=""`. Build the reader first.
- **TJ-13B Sunday follow-ups:** add suggested setup tags for old untagged trades and a
  short week-ahead note. TJ-5 must also read week_review_plan when the week slot
  lands. The first large-model probe is still live gate #158.
- **TJ-6M:** add mentor_answer_mix_rate to TJ-6's closed MEASURABLES, pool the
  Mentor answer mix in day_report_card, and add it to the registry test.
- **TJ-14B grader_gap question:** TJ-10 is merged and the field is on every grade
  row, so this small follow-up is ready to wake the question.
- **TJ-7 follow-ups:** build a verifier rule so the night never pairs a mood with
  a result; gate #175 stands in until then. Re-offered Mentor subjects keep their
  old label and combo items when the prompt or options change.
- **TJ-9E follow-ups** (not authorized yet):
  - Day pack and stories don't read exit fields yet.
  - The `CLOSED_PARTIAL` status is misspelled in 4 places (ask-first:
    `setup_environment_evidence.py:480`, `weekend_prep_panel.py:2426/4076`,
    `journal_feed.py:1076`).
  - `ai_summary._journal_source` lets raw exit words ride into narration next to P&L.
  - `journal_feed._store()` is a module-global cache.

## Housekeeping

- Split the giant files, starting with `ui/panels/alert_center_panel.py` (8k lines).
  The detector `legacy.py` files are ask-first and need golden fixtures first.
- Dead-script review (recon 2026-09-22): these have no importer outside their own tests.
  Delete them only with the trader's yes, since some may be hand-run tools:
  - `tracker_execution_compare.py`, `tracker_selection_compare.py`, `d1_level_store_survey.py`
  - `sector_cohort_divergence.py`, `diagnostics/observability_trends.py`, `gui_output.py`
  - `build_avwap_band_variant_fixture.py`, `build_mixed_unit_avwap_fixture.py`, `build_sector_cohort_fixture.py`
- **142 live gates are still marked owed** (`docs/GATES.md`); many are weeks old. The
  trader should mark whole batches as passed or dropped.
