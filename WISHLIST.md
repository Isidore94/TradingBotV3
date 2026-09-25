# Plan to 4.5/5 on every goal (defined 2026-09-24, trader's request)

The build plan from the 2026-09-24 deep review of `main`. An item becomes authorized work
when the trader moves it to `TODO.md` or says "go" on it in chat. Finished packets are
deleted from this file; `git log` and `CHANGELOG.md` have them.

**Done 2026-09-24/25.** Every packet P0-1 to P2-11 is merged (gates #209-#256). What is
left needs time, a live day, or the trader's word:

- **P1-4 4e Promotion.** A setup-key facet that passes hold-out two weeks running may
  become a named sub-family in `setup_grades`. Ask-first, golden fixtures, trader's word,
  one at a time. Earliest: two Saturdays of `permutation_report.json`.
- **P1-4 4f skipped facets.** Heikin-Ashi, SMI, LRSI D1/H1, H4 pullback depth, ATR
  percentile, 52-week-low distance, level respect count, closes vs level, setup age,
  sector RS rank, weekly-options flag, D1 zone-arm: none is on the scan row at scan time.
  Each needs the value written by the scan first.
- **P2-8 internals tape.** HYG and MAGS are not in the post-close day tape (HYG's daily
  cache is stale since June), so the evening internals axis uses VXX, RSP and TLT only.
- **P2-11 leftovers.** Measure the 43 desk timers with `thread_cpu_gauge` for one live day
  before touching any. The 13 `except: pass` sites in `bounce_bot_lib` (listed in the
  P2-11 merge) are ask-first. `ai_credentials.py` and `secret_store.py` are two
  Credential Manager paths; unify later.

## Decisions the trader made on 2026-09-24/25

- Risk unit for sizing: fixed dollars per trade; no account size stored (6c).
- Ticker briefs: Saturday only, fewer names, 7-day reuse cache (3b).
- Scan files (`master_avwap_lib/*`) may be edited for the wishlist, with no output change.
- M5 setup keys: stamped in a sidecar keyed by event_id; the outcome CSV is untouched.
- Old branches: deleted; four with unmerged work kept as `archive/*` tags.
