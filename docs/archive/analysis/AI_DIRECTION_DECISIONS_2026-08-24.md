# AI-layer direction decisions — 2026-08-24 (evening)

**Hand-committed and frozen.** This records trader decisions made in
conversation on 2026-08-24, following the morning's
`AI_LAYER_REVIEW_2026-08-24.md` and the five AI-P build commits. It is a
decision record, not a roadmap: plan.md still owns build order, and the R10
packet specs in plan.md Phase 0.7 remain the governing text for R10.B–R10.I.

## 1. The summaries' intended reader is a later LLM (trader statement)

The trader stated the purpose of the overnight summaries directly: "mostly for
an LLM to read later to save it time to get the view of each day" — the
two-tier north star (local distills nightly; frontier reads the distillations
periodically). This resolves the review's open decision 1 and **partially
reframes its §2 verdicts**:

- `ticker_briefs` and `ai_summary` are **NOT parked**. Their UNKNOWN verdict
  becomes "reader is designed but does not exist yet" rather than "no reader".
  They stay on the nightly slate.
- The artifact actually designed for that reader — one small per-session
  day-view file — is the **Daily Digest Ledger** (LOCAL_AI_AUTOMATION_PLAN
  §3.2 / Phase 2), which remains blocked on the trader answering §6.4a's six
  open questions, not on data or engineering. The briefs are per-ticker
  archive, not the day view.
- Caution retained from the review: TB-5 measured pre-fix brief content at
  96.2% roster noise, and post-fix content quality has never been measured. A
  future-LLM reader raises the stakes on content quality; one spot-audit of a
  recent night's briefs is still owed before the archive is treated as an
  asset.

## 2. The intraday market journal is R10.H (already specced, not yet built)

The trader's ask — an intraday market journal beside the D1 SPY analysis
journal, giving an AI a perspective on each day — maps exactly onto packets
already registered in plan.md Phase 0.7:

- **R10.H**: `market_journal.jsonl` (`market_journal_entry_v1`), one
  writer/service, two surfaces (Trading Desk "Journal" tab after Capture, M5
  default in-session, Ctrl+Enter commits; and the left-nav "Market Journal"
  page with six D1 charts, the RVOL ≥ 1.2 overlay, the environment timeline
  with auto-vs-manual agreement rate, calendar strip, day-context table).
- **R10.G**: `daily_market_context.jsonl` — the machine half beside the
  trader's written half.
- **R10.I**: the opt-in `market_journal` AI scope. Free-text journal entries
  reach an AI scope **opt-in only** (recorded trader decision, plan.md
  L1118-1126 — unchanged).

**Superseded in two places on 2026-08-27, by the trader, in as many words.**
The record above stands as history; these two lines are what is true now.

1. *"i also expect the AI to get access to these notes for the daily summary
   function."* The R10.I opt-in is **reversed**: `market_journal` is in
   `briefs.DEFAULT_SCOPES`. Only the trader could reverse a recorded trader
   decision, and did. `TICKER_BRIEF_SCOPES` stopped being an alias for it and
   keeps the original four — a session-level entry inside a per-symbol packet
   is the TB-0/TB-5 failure mode named in §3 below.
2. *"this should capture more stuff, such as SPY charts, what they looked like
   when the auto mode flipped, my entries, what the charts looked like when i
   inputted entries, what the D1 looked like."* The "six D1 charts" surface in
   the R10.H line above is **replaced** by a per-entry capture: the symbol's
   M5/D1 and SPY's M5/D1 as BARS at the moment the note was written
   (`scripts/market_journal_capture.py`, sidecar + a `market_journal_chart_v1`
   digest row), plus a machine-authored row for every auto-mode flip. Six live
   charts answer "what does this symbol look like now"; the trader asked what
   they were looking at when they wrote it, which is a different question and
   the only one a journal can answer later.

## 3. Walk-away and setup-tracker AI reads: opt-in scopes over deterministic outputs

Approved direction, on the `pick_feedback`/`trader_judgement` precedent
(registered but not nightly, exercised via `run_ai_jobs.py --scopes …`):

- A `walkaway` evidence scope reading `run_walkaway_analysis` **output** — the
  analysis itself stays deterministic.
- A setup-performance scope reading `setup_scoreboard.py` output and, once
  R10.C lands, the evidence report. **Never the raw tracker**: TB-0/TB-5
  measured the tracker's text projection contributing zero symbol-specific
  content while starving every analysis it led. Pointing a model at the 960 MB
  payload or its roster dump is a measured failure mode, not a caution.

## 4. Build authorization and one sequencing override (trader, 2026-08-24)

The trader authorized building **everything currently queued that is feasibly
buildable without more live data**: R10.B, R10.C, R10.D, R10.E, R10.F, R10.G,
R10.H, the two opt-in scopes above, the AWAY day-recap packet (§5 below), and
— by explicit override — R10.I's machinery ahead of its two-week collection
gate.

Scope of the override, stated precisely so it cannot creep:

- **Waived: R10.I's build sequencing** ("after two weeks of R10.A collection").
  The `evidence_report` slot and the `market_journal` scope may be built now.
- **NOT waived: the evidence-quality gate on claims** (plan.md Phase 0.7
  gates). Until two weeks of R10.A/B collection exist, every report the slot
  emits must state its n, label everything `discovery`, and say in words that
  the collection window is not met. A report over a near-empty ledger is
  honest scaffolding, never a finding.
- **NOT waivable and not waived**: plan.md sec 5 invariants, R10 ground rules
  1–12, the mechanics canaries (one live session per packet touching a live
  writer — R10.B, R10.E, R10.G, R10.H still owe theirs after building), and
  §7.3's bar on frontier calls and nightly reads of raw streams.
- **Phase 2 (digest) stays unbuilt** — its blocker is the §6.4a sign-off
  questions, which are decisions, not data; nothing here overrides that.

## 5. New trader requirement: the AWAY day happens without you, the review should not

Trader statement: many of these features assume DESK presence, but on
Auto/AWAY days the trader wants to come in **after the fact** and (a) write
the day's D1 analysis, (b) adjust Focus picks, and (c) see the bot's best
picks from the day, for review.

Recorded as authorized work (the "AWAY day recap" packet), with the
constraints that keep it inside existing contracts:

- (a) is R10.H used after the fact: a journal entry written in the evening
  about that session carries `session_date` = the session and `created_at` =
  when it was actually written, tz-aware — **never backdated** (point-in-time
  honesty; an entry about Friday written Saturday says so).
- (b) uses the existing owners: `FocusService` and the Alert Center perform
  every mutation; a trader-entered adjustment is trader-owned (no
  `focus_auto_picks.json` marker), and nothing new writes a Focus store. A
  manual after-the-fact adopt is a **trader action** — the R2 auto-adoption
  gate and its post-flip-verdict rule govern the machine's adoptions, not the
  trader's, exactly as the Strength Board's click-time re-check pattern
  already established for surfacing gate state without blocking the trader.
- (c) is presentation over stores that already exist — the day's staged auto
  picks, the alert history/review queue, `autopilot_today.txt`'s numbered best
  swing trades, and the D1 level/event classifications the AWAY push already
  aggregates. **No new detector, score, ranking, or writer**: the recap shows
  what the day already produced, ranked the way the AWAY push already ranks
  it. Where it must touch files housing alert code, the file-scoped ask-first
  rule applies unchanged.

## 6. What this document does not do

It promotes nothing, waives no live gate, and changes no runtime behavior. The
owed items are unchanged and restated for visibility: R10.V's live scan day,
the R10.A mechanics canary (`outcome_sweep_autorun` flip — also the start of
R10.I's claims clock), R9's four live proofs, the 08-21 fluidity gates, R7
gates 1/3/6, R8 §10's one real weekend, the Questrade token paste, and nothing
merging to `main` until a live-session validation day passes.
