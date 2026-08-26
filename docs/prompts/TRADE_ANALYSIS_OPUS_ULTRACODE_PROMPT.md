# Trade-analysis prompt (Opus + ultracode)

Paste everything below the line into a fresh Claude Code session on the main desk
(where the Drive home folder and `%LOCALAPPDATA%\TradingBotV3` are reachable), with
the model set to Opus. The word "ultracode" in the prompt opts in to multi-agent
workflows. Update the AEP date/details if you reuse this on a later day.

---

ultracode

You are running on my trading desk inside the TradingBotV3 repo. This is an
**analysis-and-recommendation task only** — do not edit any detector, scoring, or
alert code, do not tune thresholds, and do not change any file without asking me
first (CLAUDE.md file-scoped ask-first rule applies). Your deliverable is a written
report plus a ranked recommendation list.

## Context you must load first
Read `plan.md` sections 5 (invariants), 6 (live validation), 7 (shadow evidence and
promotion ladder), and 12 (work queue); `docs/SETUPS_MAJOR.md`; `docs/SETUPS_TEST.md`;
`docs/REVIEW_LEARNING_LOOP.md`; and `CHANGELOG.md` beside `CURRENT_CHECKPOINT.md`
(the pair that replaced `SOL_PROGRESS.md`). Every promote/demote
recommendation you make must be expressed in terms of the Section 7.1 ladder (which
rung the setup is on, which gate is or isn't met) — never recommend jumping rungs,
and never suggest a shadow engine influence live decisions.

## The data (last 20 trading days, ~2026-08-01 onward)
Resolve real paths through `scripts/project_paths.py` rather than guessing:
- **Review-learning funnel**: `ALERT_REVIEW_EVENTS_FILE` (+ the per-day
  `alert_review_events/` dir), `INTRADAY_BOUNCE_OUTCOMES_FILE`, and
  `scripts/review_learning.py` (use `build_episodes`, `attach_bounce_outcomes`,
  `attach_forward_returns` instead of re-parsing by hand).
- **Pick feedback**: `PICK_FEEDBACK_FILE` (`pick_feedback.jsonl`).
- **Journal**: `JOURNAL_DB_FILE` (`trade_journal.sqlite3`) — my actual trades,
  R-multiples, and notes; `scripts/journal_analytics.py` has existing aggregation.
- **Shadow evidence**: `%LOCALAPPDATA%\TradingBotV3\diagnostics\` —
  `spy_state_shadow.jsonl`, `greatness_shadow.jsonl`, `run_manifests\`,
  `job_ledger.jsonl` (use manifests to know which scans actually ran and from
  which bar source).
- **Digests**: `autopilot_today.txt` history and anything under the shared home
  folder `output/` directory.
- **Market context**: `market_environment_annotations.jsonl`, `review_policy.json`.

Start with a **data-inventory phase**: for each source report row counts, date
coverage, and gaps inside the window. Missing data is uncertainty, never
confirmation — say explicitly when a conclusion is starved of evidence, and never
draw conclusions from a handful of episodes.

## Task 1 — Setup scoreboard: promote / demote / watch
For every setup family named in `docs/SETUPS_MAJOR.md` and `docs/SETUPS_TEST.md`,
build the funnel over the window: signals fired → surfaced to me → taken/dismissed
(review events) → outcome (bounce outcomes, forward returns, journal trades).
Compute hit rate, average and tail R, dismissal-regret (dismissed alerts that then
worked), and how outcomes split by regime (`spy_state_shadow`, environment
annotations), time of day, and RVOL bucket where the episode data carries them.
Then give me a ranked table: **promote candidates** (with the exact Section 7 gate
evidence they now satisfy and what's still missing), **demote candidates** (with
the evidence they're failing), and **insufficient-data** (with what capture is
missing and how to get it). Cross-check each claim against raw rows — every
promote/demote must cite counts and concrete episodes, not vibes.

## Task 2 — New setup mining
Find winners my systems missed or under-ranked: journal trades not attributable to
any alert, dismissed/ignored alerts with strong forward returns, and repeated
shapes in the review notes. Cluster them into 1-3 candidate setup archetypes. For
each, write a spec in the `docs/SETUPS_TEST.md` style (trigger, context filter,
invalidation, measurement plan) so it can enter the ladder at rung 1-2 as a
measured study family — shadow-only, zero scoring influence.

## Task 3 — Faster / earlier high-quality trades
For each *winning* alert in the window, measure the lag from the first
objectively-detectable precondition to the moment it hit my phone or screen, and
attribute the lag: scan cadence, completed-bar confirmation, ranking/digest
publishing, or detection logic itself. Separate honest latency (the completed-bars
invariant is non-negotiable) from removable latency (cadence, ordering, delivery).
Propose concrete earliness improvements — e.g. pre-arming price alerts at computed
levels so the trigger is event-driven instead of scan-driven, earlier "forming
setup" watch states that are clearly labeled preview, or cadence changes — each
with its cost, its IB pacing impact, and which invariants it must respect. Also
audit the funnel for quality: where do weak candidates leak through and dilute my
attention, and what ranking evidence from Task 1 would tighten it?

## Task 4 — Day-trading deep dive (the AEP case)
My day trading is the weak leg. On **2026-08-22, AEP was a slam-dunk day trade**
that the system did not put in front of me with conviction. Reconstruct AEP's
intraday tape for that session (M5 + D1 context; yfinance is fine, note the bar
source), plus SPY context from the shadow logs. Establish, honestly and
point-in-time (only information available at each simulated decision moment):
1. What the ideal entry, stop, and exit were, and what made it a slam dunk.
2. Whether AEP was on any watchlist, whether BounceBot or any scan evaluated it
   that day (check run manifests), and what each detector saw and why it stayed
   quiet or ranked it low.
3. Define the **archetype** this trade belongs to, then check the window for other
   instances of the same archetype and how many the system caught.
4. Write a candidate intraday detector/watch spec for the archetype — entering at
   ladder rung 1-2 with a measurement harness per the `docs/SETUPS_TEST.md`
   pattern — including what would have surfaced AEP by what time, and the
   false-positive load that detection breadth would have cost across the window.

## Orchestration
Use workflows: fan out parallel readers per data source for the inventory; run the
four tasks as parallel analysis lenses once inventory lands; adversarially verify
every promote/demote claim and the AEP conclusions against raw evidence rows
(spawn skeptics prompted to refute); finish with a completeness critic asking what
data source or setup family was skipped. All timestamps with explicit timezones.

## Deliverables
1. `docs/analysis/TRADE_REVIEW_2026-08-22.md` (ask me before creating the dir if
   it doesn't exist): data inventory + coverage gaps; the scoreboard with
   promote/demote/watch and ladder-gate status; new-setup specs; the earliness
   audit with proposed improvements; the AEP case study and archetype spec.
2. A short ranked list of proposed `plan.md` Section 12 queue items (proposals for
   me to accept — do not edit plan.md).
3. An explicit "Questions for Aaron" section for every judgment call you couldn't
   settle from data.

Do not touch detector/scoring/alert files, do not add suppression anywhere near
`review_policy.json`, and remember chart/queue ordering stays FIFO. If any data
source is unreadable, report it as a gap and keep going.
