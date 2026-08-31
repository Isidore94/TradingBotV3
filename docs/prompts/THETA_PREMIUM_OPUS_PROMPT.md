# Theta premium optimization, Phase 0.11 — build prompt (Opus)

Paste everything below the line into a fresh Claude Code session in the repo on
the main desk, model set to Opus. Paste the handoff back to the Fable session
for review. Authorized: `plan.md` Phase 0.11 (trader, 2026-08-31 chat — premium
floor "on a 200 dollar stock I'd want at least 1 dollar, ideally 2", "2 weeks
max for put sells, 3 weeks for credit spread", "tighter spread the better …
within reason", "all stocks in the universe that meet the theta rules", and
"spreads are a spectrum … #1 priority is still areas of support").

---

You are building `plan.md` **Phase 0.11 — Theta premium optimization** in the
TradingBotV3 repo on my trading desk. Read `CLAUDE.md` first and follow its
mandatory documentation workflow: `CURRENT_CHECKPOINT.md` "Active state at a
glance", `plan.md` §5 and the Phase 0.11 body, and search `CHANGELOG.md`'s
`Current implemented inventory` for "theta". The theta chain is the sold-put and
put-credit-spread **recommendation** report the Master AVWAP scan writes — it
never executes anything, and that stays true.

## The problem you are fixing

The theta report today surfaces ~$0.25 credits with untradeable spreads. Root
causes, verified in code on 2026-08-31 (line numbers are same-day anchors —
re-find by symbol name, never trust them blindly):

- The target IS $0.25: `THETA_PUT_TARGET_TOTAL_CREDIT = 100.0` /
  `THETA_PUT_MAX_CONTRACTS = 4` → `THETA_PUT_TARGET_MIN_CREDIT = 0.25`, cusp
  `0.15` (`scripts/master_avwap_lib/legacy.py:487-491`). Credit is judged in
  flat dollars, never relative to the strike.
- The final sort in `_rank_sold_put_option_recommendations`
  (`legacy.py:19096-19105`) orders by `(status, strike ASCENDING, market_days,
  covered supports, credit, rank_score)` — within "recommended" it prefers the
  LOWEST strike, i.e. the deepest-OTM, cheapest qualifying option, every time.
- A wide spread is only a soft penalty capped at 18 points
  (`legacy.py:19037-19038`); nothing ranks it down hard.
- No symbol-level premium-richness thinking: the quote budget (240 quotes /
  360 s, `legacy.py:510-511`) is spent in `base_score` order
  (`legacy.py:15888`), so dead-vol names burn quotes that rich names never get.

## Trader decisions (locked 2026-08-31) — the spec

- **D1 Premium floor is a percent of the strike.** Minimum credit **0.5% of
  the strike** ($1 on a $200 stock); **1.0%** is the ideal tier ($2). Plus an
  absolute floor of **$0.40 per contract** so cheap stocks never show pennies.
  Below both → the row is filtered from the report, not shown "below_target".
- **D2 Tiers.** `recommended` = yield ≥ 1.0% of strike AND credit ≥ $0.40;
  `cusp` = yield ≥ 0.5% AND ≥ $0.40. The $100/4-contract framing
  (`_contracts_needed_for_target_credit`) becomes display-only info, never a
  qualifying bar.
- **D3 Time.** Sold puts stay ≤ 10 market days (unchanged). PCS extends to
  **15 market days** (3 weeks); its minimum (4) is unchanged.
- **D4 Spread is a spectrum, not a gate.** No new hard block. Wide spreads are
  ranked HEAVILY lower (monotonic, uncap or raise the 18-point cap so a truly
  wide spread sinks to the bottom); tight spreads get a boost. Keep the
  existing `bid_wide_spread` credit-source behavior (`legacy.py:18821-18841`).
- **D5 Ranking priority, in order: (1) support, (2) premium, (3) spread.**
  Support = major SMAs (50/100/200) still above the strike — **1 required
  (unchanged eligibility), 2 is a big rank boost** — plus the rest of the
  covered stack (AVWAP family, trendline, HV levels) via
  `support_quality_score`. Premium = yield per market day, so a nearer expiry
  paying the same percent wins (this replaces the flat
  `THETA_DTE_PENALTY_PER_MARKET_DAY`). The strike-ascending sort key is
  removed. Support quality must dominate: a 2-SMA-covered strike at $1.20
  outranks a 1-SMA strike at $2.00.
- **D6 Universe.** Every scanned symbol already gets theta-evaluated
  (`runner.py:1461-1497`; universe longs join full scans via
  `UNIVERSE_LONGS_FILE`, `runner.py:502-511`), so "all stocks in the universe
  that meet the theta rules" already holds at evaluation time. What you change
  is budget ALLOCATION: pre-rank the enrichment work list so `thetalongs.txt`
  names are pinned first (the trader's own list), then names most able to pay
  the floor, estimated from data already on the row (ATR20/close as the vol
  proxy — **no new network calls**). Low-capacity names sort last but are
  never silently dropped; the existing budget-exhaustion path (support-only
  fallback + note) already handles the tail honestly.

## Hard rules

1. **Recommendation-only, and the scope is fenced.** `legacy.py` houses the
   champion detectors. You may edit ONLY the theta functions (everything
   matched by `theta`/`_pcs_`/`_sold_put_` plus the option-quote helpers
   `_option_quote_*`, `_strike_support_context`, `_contracts_needed_*`,
   `_format_theta_premium_target`, `write_theta_put_report` and its
   extractors), the theta enrichment work-list ordering in
   `enrich_theta_rows_with_ib_option_premiums`, and the theta UI files
   (`scripts/ui/models/theta.py`, `theta_table_model.py`,
   `panels/theta_panel.py`, `services/theta_feed.py`). Any other edit in
   `legacy.py`/`runner.py` — including "harmless" refactors of shared helpers —
   requires asking the trader first. This prompt is the trader's ask-first
   authorization for the named surface only.
2. **No detector, scorer, alert, watchlist, Focus, or queue behavior changes.**
   `evaluate_theta_put_candidate`'s eligibility (≥3 supports, ≥1 major SMA,
   earnings buffer) is unchanged except where D1–D5 say otherwise. R9.4
   semantics are untouched: `thetalongs.txt` moves `theta_side` only, never
   `side` (`runner.py:660-666`).
3. **IB pacing is untouched.** Budget constants
   (`THETA_OPTION_ENRICHMENT_MAX_QUOTES`/`_MAX_SECONDS`, delays, timeouts)
   keep their values; quotes stay snapshot-only (IB rejects snapshot requests
   with generic ticks — the comment at `legacy.py:15284` is load-bearing, so
   no open-interest tick requests).
4. **Fail-before-fix.** Every behavior change ships with a test proven to fail
   on the un-fixed code (stash the fix, run, unstash) — say so per test in the
   handoff. Existing theta tests live in `tests/test_master_avwap_setups.py`,
   `tests/test_theta_longs_list.py`, `tests/test_qt_theta_market_prep.py`,
   `tests/test_gui_output.py`.
5. **Never break the tree.** The desk launches from this checkout. Check
   `git status` before anything (sessions share the checkout). Branch
   `claude/theta-premium` from current `main` HEAD. Commit small and green,
   push after each commit. Before each commit: `.venv\Scripts\python.exe -m
   pytest tests/ -q` fully green (baseline 5590 passed / 72 subtests, will
   have grown), `-m ruff check .` clean, and `scripts/smoke_check.py` 7/7.
   No packaging trigger is expected (logic-only in existing modules) — do not
   rebuild the exe.
6. Chat to me in very short, simple lines (CLAUDE.md "How to talk to the
   trader"). Depth goes in docs and commit messages.

## Packets, in order

### T1 — the floor becomes relative (D1, D2)

Facts: constants at `legacy.py:487-491`; tiering at `19027-19036`; the
per-price flat-dollar target text `_format_theta_premium_target` at `18183`;
the report header stating the old $100/4-contract rule at `19479-19487`;
sub-floor sold-put handling around `15530-15548`.

Build: new constants (min 0.5% of strike, ideal 1.0%, absolute $0.40); tier per
D2; store `credit_pct_of_strike` and `credit_pct_per_market_day` on every
ranked row; contracts-for-$100 kept as display info only; below-floor rows
filtered from the report with a counted reason (reuse the
`_mark_theta_row_filtered` idiom); `_format_theta_premium_target` restated in
percent terms; report header text updated to the new rules. Apply the same
percent floor to the PCS short leg only if the credit-to-width ratios
(`THETA_PCS_TARGET_CREDIT_WIDTH_RATIO` 0.20 / cusp 0.12) don't already imply a
stricter bar — if they do, leave PCS tiering alone and say so in the handoff.

Tests: $200 stock — $2.00 recommended, $1.00 cusp, $0.25 gone from the report;
$20 stock — $0.15 gone on the $0.40 floor even though it is 0.75% of strike;
assert tier behavior, never literal constant values.

### T2 — ranking: support first, then yield, then spread (D4, D5)

Facts: the sort at `19096-19105`; `rank_score` composition at `19047-19057`;
`covered_major_sma_support_count` already computed by `_strike_support_context`
(`18728-18797`, `require_major_sma_support=True` from `18994`).

Build: within each status tier, order by support first
(`covered_major_sma_support_count` with a step boost at 2+, then
`support_quality_score`), then yield per market day, then spread (graded,
heavy, monotonic — replaces the capped 18-point penalty AND the flat DTE
penalty). Remove strike-ascending as a sort key. Moneyness penalty may stay.

Tests, as constructed quote rows with known orderings: (a) 2-SMA $1.20 beats
1-SMA $2.00; (b) equal support → higher yield-per-day wins even at the higher
strike; (c) equal support and yield → tighter spread wins; (d) a 40%-spread
quote still appears, below an otherwise-equal 12%-spread quote; (e) the old
failure — two recommended rows where the deeper-OTM one is cheaper — now ranks
the richer one first.

### T3 — PCS gets three weeks (D3)

Facts: `THETA_PCS_MAX_EXPIRATION_MARKET_DAYS = 10` at `legacy.py:500`; the
sold-put max (`:487`) stays 10; expiration filtering must use the PCS constant
on the PCS path only.

Tests: an expiration 13 market days out is eligible for PCS and not for sold
puts; 16 market days is eligible for neither.

### T4 — spend the quote budget on names that can pay (D6)

Facts: the work list sorts by `base_score` alone at `legacy.py:15888`; budget
exhaustion already degrades honestly (support-only fallback, `15906-15925`);
`atr20` and `last_close` are already on every row.

Build: order = `thetalongs` rows first (the row already carries
`theta_list_source`), then descending estimated premium capacity (an ATR%-based
estimate of whether a support-defended strike can pay 0.5% in ≤10 market days —
document the formula in a comment and store the estimate on the row), then
`base_score`. No row is dropped for a low estimate; no new network calls.

Tests: thetalongs pin ahead of higher-scored watchlist names; high-ATR% ahead
of low at equal base_score; the exhaustion path still writes the support-only
fallback for tail rows.

### T5 — the report and the panel say the new things

Facts: writer `write_theta_put_report` (`legacy.py:19471+`) and its extractors
(`extract_theta_rows_from_report`, `extract_theta_reason_risk_rows`) feed
`scripts/gui_output.py`, `scripts/ui/services/theta_feed.py`, and the Qt theta
model/panel. `ThetaTableModel` full-resets on refresh — leave that shape alone
(Phase 0.9 owns Qt fluidity); columns only.

Build: surface per row — credit, credit % of strike, yield per week, spread %,
credit source, SMA-above-strike count (with the 2+ boost visible), and the
support stack summary that already exists. Round-trip: whatever the writer
emits, the extractors parse, and the Qt model shows.

Tests: extractor round-trips the new fields; the Qt model exposes the new
columns with sane display strings; `test_gui_output.py` highlighting still
passes.

### T6 — reconcile the docs

Per CLAUDE.md: `CURRENT_CHECKPOINT.md` (glance block + dated entry, live gate
owed: one desk scan whose theta report shows percent-floored, support-first
rows and DRAM's list still labelled `via thetalongs.txt`), `CHANGELOG.md`
inventory, `plan.md` Phase 0.11 status, and `docs/README.md` only if you add a
file. `CLAUDE.md`/`AGENTS.md` should not need changes; if you believe they do,
stop and ask.

## Handoff

Paste back to Fable: per-packet summary, every test named with its
failed-first proof, the final baseline counts, the branch tip hash, and
anything you deliberately did not do with the reason.
