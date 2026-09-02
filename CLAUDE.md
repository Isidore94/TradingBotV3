# TradingBotV3 — AI context index

TradingBotV3 is a Windows desktop decision-support system for one trader's day and
swing trading. It does everything except execute orders: pre-session market prep,
candidate discovery (D1 anchored-VWAP swing scans + intraday 5-min bounce detection),
live monitoring with alerts, unattended Auto/Away scanning with a phone report, a
journal, and a controlled research/promotion program for new setups. Order execution
is permanently out of scope (plan.md sec 1).

## How to talk to the trader (trader rule 2026-08-26)

**Write every message to the trader as if they are five years old.** Very short.
Very simple words. One idea per sentence. Say what you did, what is broken, and
what they need to do - nothing else. No long lists, no tables, no section
headers, no code words unless the trader has to type them. If a message is
longer than about ten short lines, cut it. Detail belongs in the docs and the
commit message, not in the chat. This rule is for chat output only; docs, code
comments and commit messages keep their normal depth.

## Mandatory documentation workflow for every AI

**Read narrow, not everything.** This workflow used to say "read `CHANGELOG.md`,
`plan.md`, `CURRENT_CHECKPOINT.md`" without bounds; those files reached 1 MB combined
(~260k tokens) and the instruction stopped being followable. An agent that cannot read
its brief skims it and then appends to it, which is what grew the files. The bounded
read below is the instruction — widen it only when the narrow read leaves a real
question open.

Before proposing, planning, or changing anything:

1. `CURRENT_CHECKPOINT.md` — read the **"Active state at a glance"** block at the top:
   branch, active roadmap items, last verified baseline, open gates, next action.
   That block is the brief. Read the dated entries below it only for the item you are
   actually touching; if a dated entry contradicts the block, the dated entry wins.
2. `plan.md` — Sections 5 (invariants), 6 (live validation) and 7 (promotion), then the
   phase order in Section 12. Read the body of your phase only.
3. `CHANGELOG.md` — search `Current implemented inventory` (the ~250-line contract at the
   top) for the feature you are about to touch, so you do not rebuild landed work.
   **Search it; do not read it end to end.**
4. `docs/README.md` — open only the active specification, runbook, and decision
   records relevant to the selected item. Historical documents are evidence, not
   current authority.
5. Inspect the source, tests, Git status/history, and runtime artifacts needed to
   verify that the documentation still matches reality. **When the docs and the code
   disagree, the code is the fact and the doc is the defect** — fix the doc, and say so.

Archived history is deliberately outside this read and must never be pulled into it
wholesale: [`docs/CHECKPOINT_ARCHIVE_2026-08.md`](docs/CHECKPOINT_ARCHIVE_2026-08.md)
and
[`docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md)
are evidence for one specific question, not context to load.

`WISHLIST.md` contains ideas, not authorized work. Never implement directly from it.
An item enters the build sequence only when the trader explicitly moves it into
`plan.md`.

Before editing, state the exact roadmap/checkpoint item, what already exists, what
remains, governing documents, expected files, tests, and whether the ask-first rule
applies. Do not skip to a later phase because it is easier or more interesting.

After every repository change, reconcile the documentation before handoff:

- always update `CURRENT_CHECKPOINT.md` with the active item, working state, and
  verification result (or explicitly state why the baseline is unchanged);
- update `CHANGELOG.md` when behavior, contracts, architecture, operations, or an
  implementation status changed;
- remove, narrow, or advance the corresponding `plan.md` work while retaining any
  live-validation or promotion gate still owed;
- update the governing detailed spec/decision record when its contract or rationale
  changed;
- update `WISHLIST.md` only for trader-directed idea additions, removals, or
  promotions; an AI may recommend a change but must not silently promote one;
- update `docs/README.md` whenever a Markdown file is added, removed, renamed, or
  reclassified;
- keep `CLAUDE.md` and `AGENTS.md` identical whenever operating instructions change;
- **refresh the "Active state at a glance" block** in `CURRENT_CHECKPOINT.md` — a stale
  block is worse than none, because it is the one thing the next agent trusts;
- **keep the active files small.** Write the shortest entry that carries the decision
  and its evidence; detail belongs in the governing spec, not in a fourth retelling.
  When `CURRENT_CHECKPOINT.md` passes ~1,500 lines, move the entries older than the
  oldest open gate into the dated archive under `docs/` and leave a pointer. Same rule
  for `CHANGELOG.md`'s revision history. Archiving is maintenance, not a new document.

Do not create another roadmap, progress ledger, handoff, or status file. The root
control set is `CLAUDE.md`/`AGENTS.md`, `CHANGELOG.md`, `plan.md`,
`CURRENT_CHECKPOINT.md`, `WISHLIST.md`, and `docs/README.md`.

## Core loop / data flow

Each rule below is binding as written. The incident, measurements and trader
conversation behind every one are preserved verbatim in
[`docs/DESK_INTERNALS.md`](docs/DESK_INTERNALS.md) — **read the matching entry there
before changing the behaviour a rule governs.**

**Shape**
- Entry: `launch_gui.py` → `scripts/ui/app.py` (PySide6 Trading Desk). One desk role, no flag to change it — Desk Link/satellite and the mini-PC scanner were retired 2026-08-08 and their code removed 2026-08-24 (no `desk_link`, no `ui/satellite.py`, no `master_avwap_mini_pc.py`, no `--satellite`/`--desk-role`). `scripts/gui.py --ui tk` is the legacy Tk UI.
- Market data: IBKR TWS/Gateway `127.0.0.1:7496` (`ibapi`) primary, `yfinance` fallback; bar source tracked per scan (`docs/BROKER_ADAPTERS.md`).
- Engines: `scripts/master_avwap.py` (+`master_avwap_lib/`) D1 AVWAP swing scanner; `scripts/bounce_bot.py` (+`bounce_bot_lib/`) intraday M5 bounce detector; `market_prep/` pre-session services.
- Inputs: plain-text watchlists (`longs.txt`, `shorts.txt`, `swinglongs.txt`, `shortswings.txt`) in the shared home folder.
- Storage: home folder `C:\TradingBotData` is a plain LOCAL folder — **there is no cloud drive** (removed 2026-08-10, decision 0015). Per-machine caches/diagnostics under `%LOCALAPPDATA%\TradingBotV3` (`scripts/project_paths.py`). The DAS `\\MINI-PC\Trading Bot Data` is the durable tier; **write local first, move to the DAS after**, so an outage costs throughput and never correctness.
- Shadow engines (`market_state.py`, `greatness_monitor`) emit JSONL promotion evidence only. Review-learning loop: Alert Center decisions → `alert_review_events.jsonl` → `review_learning.py` → AI-curated `review_policy.json` → chart annotations (`docs/REVIEW_LEARNING_LOOP.md`).

**Research warehouse** (Phases 0–8 built; contract `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`, decisions `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`, identities `docs/RESEARCH_WAREHOUSE_ERD.md`)
- Shadow-only additive evidence: **zero detector/score/alert influence.** Lives at `research_store_dir`, a separate storage class NEVER inside `C:\TradingBotData` (unset = disabled).
- **The build runs inside the desk process, so reads are session-scoped.** Partitions are MONTH-keyed: narrow through `ResearchStore.read_rows` (Arrow-side `symbols` / `interval_start_range`), never by filtering a materialised list.
- **Never widen `_run_outcomes` to a date filter** — its walk runs FORWARD across sessions (BD-66/BD-69/BD-74).
- A SNAPSHOT over 64 MB is stored whole but **never `json.loads`-ed**; answer the UNCHANGED watermark from a chunked hash before any `read_bytes` (BD-73).
- Growth is month-keyed: it worsens all month and resets on the 1st. Check the calendar before treating a new report as new.
- **H2 exists again** (`TIMEFRAME_MINUTES`, `DERIVED_TIMEFRAMES`, BD-78): the locked plan cut it for having no consumer and the Phase 0.12 higher-timeframe LRSI study is one. RTH is 6.5 h, so H2/H4 end each session with a STUB - published as evidence, **excluded from the LRSI input**.
- **The HTF LRSI grid is 16 diagnostic recipes and never a Cartesian search** (`outcomes.HTF_LRSI_RECIPES`): 4 timeframes × 4 entries, one stop model, one target. Long and short legs read the **SAME unmirrored series** - cross-up 50/20, cross-down 50/80 - because the formula clamps at 0 and the mirrored-close idiom is a DIFFERENT feature (BD-79). Live `CROSS_LEVELS` stays `(20, 50)`; `RESEARCH_CROSS_LEVELS` is additive and shadow-only.
- **The setup registry is frozen DATA and is not authoritative yet** (P7, 2026-09-01). `scripts/setup_registry.py` loads `setup_registry_v1.json` - one entry per setup, keyed `setup_id@version` - joining the FIVE places that name a setup: `_FAMILY_TAGS` (the canonical id), `setup_docs`, the playbook study, the claim picklist, and `legacy.py`'s `*_STUDY_FAMILY` constants (eight families are named ONLY there). Regenerate with `scripts/build_setup_registry.py --write` and review the DIFF; it is never rebuilt at import. It **resolves no disagreement** - those are `known_divergences` - and **fills no column its sources do not establish**, so supported sides and timeframe roles are deliberately blank. An unknown name RAISES rather than defaulting to `GENERAL`. **`plan.md P4.1` is where it becomes authoritative**; until then nothing in production imports it. Its sibling `research_warehouse/trial_ledger.py` records one append-only row per registered grid BEFORE any outcome is inspected, and `register` refuses to rewrite an existing `trial_id`.

**Alert Center, review queue and capture**
- **The charts own the review pane**; between them and the tab strip there is at most ONE slim row (the verb row). **The arm bar stays UNDER the chart** — placement is a HOST decision via `AlertChartReview(dock_arm_bar=…, dock_capture_rail=…)`. **Never propose moving the arm bar without asking.** `CaptureRail.action_shortcuts()` is rebound at panel scope; a `QShortcut` inside a hidden tab never fires, and two bindings for one sequence fire NEITHER.
- **A VETO and a LIKE each retire the chart; a NOTE never does.** "Veto D1 — but M5 today" writes a veto row and emits a REQUEST; the panel performs the Focus placement, so the Focus store keeps one writer. Place first, retire second; a failed placement still retires. What a LIKE offers is `MAIN_CLAIM_GROUP` + `EXTRA_CLAIM_IDS` in `ui/annotations/setup_claims.py`; keys run in list order so learned digits never move.
- **A day-trade PASS is a note, not a veto, and it never retires the chart** (trader, 2026-08-31: *"I really like this stock for a daytrade but it has this ONE issue"*). Multi-select reason codes come from a SEPARATE vocabulary family, `ui/annotations/vocabularies/pass_reasons_v*.json` — never folded into the veto list, whose cohorts are already accruing forward returns. Codes are written in VOCABULARY order, not click order. When the desk already holds the symbol's M5 bars the row references one session of them through a sidecar (`ui/annotations/pass_bars.py`, written BEFORE the row so a reference never lies); with nothing cached the row still writes, timestamp only. **A capture click never fetches.** A pass does **not** mark the symbol "Reviewed today" - `pick_feedback._ANNOTATION_DECISIONS` stays `veto`/`like_claim`/`note`, because that set is read by the scanner report and several badges (trader decision, 2026-08-31; both this and "never retires" are DECIDED, not open).
- **A LIKE has two modes and only one of them names a setup** (P9, 2026-09-02, trader: *"anytime I like and claim a setup or like a day trade setup I just want to let the bot and the future AI know 'something about this was good'"*). **Alt+L** writes a QUICK like - `like_mode: "quick"`, no claim, no why - and **Alt+K** the claimed one, which still requires both. **The key is instant and the BUTTON prompts**: the chart's verb row and the rail both carry a quick-like button that opens a box for an OPTIONAL note (cancel records nothing), while Alt+L never prompts - a key that stops to ask is not a one-key verb. An optional note is not R9.2(a)'s required why returning: that rule is about a CLAIM. R9.2(a)'s why-required is superseded for the quick path ONLY. A quick like retires the chart, records `like_advance` and marks the symbol reviewed exactly as a claimed one does, and **places nothing** - a like carries zero privileges (plan.md P3.1). It grades under `like_unclaimed` and contributes a **LINK** to the auto-tagger, never a tag, because it names no setup. `like_mode` is ADDITIVE (schema stays 1) and its ABSENCE reads as `claimed` - a claim was required until P9. **A capture sidecar is completed after the close** by the `sidecar_completion` slot into a NEW file (`m5_bars_completed_ref`); the original `m5_bars_ref` still means "what the desk held at the click" and is never rewritten.
- **Every verdict has a forward record, and no two verdicts are combined into one** (P5, 2026-09-01; corrected R1): veto, like, **pass** and **rejection** (`not_today` / `dislike`). Each family DOES get a pooled base row from `human_focus_tracking` - for the rejection family that row is two verdicts recorded on two different populations, so it is LABELLED wherever it is shown and must never be read as either verdict. The rejection source names its lane (`focus__m5_not_today`, `focus__swing_dislike`), and the double underscore right after `focus_` is load-bearing. A day-trade pass is MULTI-SELECT, so it grades in one cohort per reason code AND in the pooled `pass_all` — **the code cohorts overlap and must never be summed**, and only `pass_all`'s n counts passes. `unfavorite` is never graded (a membership change, not a verdict) and a rejection's free-text `reason` is carried verbatim and **never coded by machine**. A pass's same-session grade is BLANK with a stated `intraday_unmeasured_reason` whenever the sidecar cannot reach the entry bar — never a zero. **`update_human_focus_outcomes`'s `pick_key` defaults to the existing identity**; only a multi-source cohort passes `pick_key_with_source`.
- **Veto vocabulary is versioned and codes are never reused** (`ui/annotations/vocabularies/`). Cohort identity on write is `(vocab_version, reason_code)`; rows are never rewritten. Pooling equivalent definitions happens only when the rollup is rebuilt (`_rebuild_pooled_performance`), never at write time. **Never assert a literal `vocab_version` in a test** — assert against the loaded vocabulary.
- **PROVEN is the top alert class and BANGER no longer exists** (trader, 2026-09-01). BANGER was a literal token match with no producer anywhere in the tree; its tier-gate bypass, always-sound and repetition escalations are removed and `is_banger` is gone from `RepetitionLedger.consider` rather than ignored. The `banger` review-event column stays as a constant `False` so historical rows and the schema id are unchanged. `REGIME_BANGER_*` in `legacy.py` is a regime-pause threshold — a different thing, untouched.
- **The LRSI M5 alerts are RETIRED and their evidence is not** (trader, 2026-09-01). `LRSI_M5_ALERTS_RETIRED` gates the EMIT seam in `_emit_lrsi_cross_alert`, the H1 shape: detection, the candidate row, `intraday_bounce_outcomes.csv`, the learning tier and the PROVEN stamp all still run; only `gui_callback` is skipped. **Never flip `M5_SIGNAL_TYPE_DEFAULTS` for these two** — that toggle gates DETECTION and would stop the evidence, not the noise. `log_bounce_to_file` still runs (unlike H1's) because `AutoTagger` reads `INTRADAY_BOUNCES_CSV`. The higher-timeframe warehouse study is the measurement the trader asked for.
- **Feed repetition control is display only and withholds nothing.** One live row per symbol+side+day; repeats fold with an ×N badge. Focus-privileged, trader-armed, entry-assist and ready-D1 output bypass the fold and the digest. The backing list is written BEFORE any repetition decision. **No suppression field exists in this chain.**
- **Movers-only chart review** is a DEFAULT-ON PRESENTATION filter: it hides and counts, never deletes, mutes, or writes `review_policy.json`. Both legs (prev-day extreme via `focus_adoption_gate.mover_state`, and VWAP side) are asked at SHOW time and re-measured before the next chart. D1 recommendations carry a third leg (`scripts/sma_trend_gate.py`). UNKNOWN always SHOWS, tagged `unmeasured`.
- **Intraday alerts are a list beside the chart, not a queue in front of it** (`ui/widgets/m5_alert_bar.py`, LEFT column). **Clicking from one row to the next is a SKIP, never a re-queue** — only a place-holder is re-inserted; the other writes a `skip` event, because `shown` is the denominator for P(take | shown). Routing lives in `_is_m5_review_alert` inside `_enqueue_review_alert`, AFTER the AWAY branch; everything upstream is untouched. **A click away IS a pass and that is the intended meaning** (trader, 2026-09-01: "clicking away = a pass ... set alerts / add to focus and then move on") — never "fix" it into a take or into silence, and never rename `clicked_away_from_m5_alert`, which `review_learning` keys on.
- **"Holding highs" is measured in ATR and it expires.** Distance is **1.0 ATR**, never a percentage (M5 ATR ran 14× across one batch). A row is good for **15 minutes** from the later of the alert and the last new extreme, then is deleted **from the review queue only** — History, `alert_review_events.jsonl` and tracker outcomes keep it, so the rule stays gradeable. **Uncertainty never deletes.** Being AT the extreme is holding regardless and needs no ATR. The detector gate additionally requires the M5 Focus adoption gate (`passes_focus_adoption_gate`), called and never restated. With-trend rows auto-join M5 Focus (`scripts/regime_pause_focus.py`, DESK only) and skip the review chart.

**Focus, gating and modes**
- **M5 Focus adoption gate** — one definition in `scripts/focus_adoption_gate.py`: beyond yesterday's extreme AND right side of session VWAP, on the last **completed** M5 bar, **UNKNOWN always failing**. Session VWAP comes from `chart_snapshot.session_vwap_series`, never BounceBot's dynamic/EOD VWAP. Stored verdicts expire at 45 min or 2 completed bars.
- **A Focus pick's automatic D1 alerts are PULLBACKS only** (trader, 2026-09-01). `_poll_focus_d1_interest` evaluates `D1_PULLBACK_KINDS` and nothing else; the EXTENSION set (new 5d/20d extreme, SMA break, AVWAPE / 1σ break) fires only when the trader ARMED it, through `_poll_d1_event_watches` - a different poll, which is what makes double-firing impossible. The gate is at the flag-GENERATION seam: an extension kind is never constructed, so nothing is suppressed downstream. Supersedes the 2026-08-05 one-extension-per-day ration.
- **An armed alert expires, in TRADING days.** 5 sessions for a manually armed 5d extreme watch, 10 for a 20d one, 10 for everything else armed (D1 level, any-bounce, manual price alerts). The clock is `market_calendar.trading_days_between`, **never weekday arithmetic**; policy lives once in `scripts/armed_alert_expiry.py`. **Uncertainty never deletes** - a date the calendar refuses keeps the entry armed. Every expiry appends a row naming store, symbol, kind, `armed_at`, `expired_at`. A price alert is **DISARMED, never deleted**, and arming restarts its clock. Expiry runs at the head of the poll that already owns each store - no new timer.
- **A quiet Focus pick FADES after 10 trading days** and the fade is reversible. The clock starts at add time (`focus_pick_clocks.json`) and is reset by a fired Focus D1 flag, an armed-watch hit or the trader's "★ keep". It applies to swing AND M5 picks **including the trader's own** - an explicit 2026-09-01 authorization, scoped to Focus and routed through the store's own removal path so a hand-maintained watchlist line is still never touched. Faded is not deleted: `focus_faded.json` + an append-only row, restored with a FRESH clock or discarded. A faded swing favorite appends a RETRACTION with origin `focus_fade`, never an edit, and **no `pick_feedback` verdict is ever written for a fade**. `FocusPickStore` is the single writer; the check runs on the day roll plus a half-hourly timer, **never inside the 60 s poll**. The "Faded review (N)" walkthrough goes through `_enqueue_review_alert` with `FOCUS_FADED_TAG`, which bypasses movers-only exactly as `FOCUS_REVIEW_TAG` does.
- **Focus provenance:** `focus_auto_picks.json` marks machine-adopted entries. **Absence of a marker means the trader owns it**, and only marked entries are reachable by "Not today" or desync repair.
- **Today's swing picks are the trader's own list, and they get two writes** (`swing_favorites.jsonl` + swing Focus, `ui/widgets/swing_favorites_bar.py`). The Focus write goes FIRST and must not fail; the append-only evidence row goes second and its failure is swallowed. **Never write an auto-adoption marker for one** — absence of a marker is what makes it theirs. The Focus like-origin is **`vetted`**, so they grade as their own `human_focus_swing_vetted` cohort instead of mixing with every other hand-typed swing name. A removal appends a RETRACTION, never an edit. The "taken" badge is a display-only join against the TRADE journal, off the Qt thread, and it never prepares the journal schema. The strip shares the M5 alerts column with the alert bar and is always the BOTTOM of it (the trader runs workspace mode, where that surface is a column rather than a tab); the two share a **draggable** split with its own settings key and no collapse.
- **Auto-mode matrix** (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`): discovery is identical in every mode; what changes is who is present. **DESK** adopts staged picks immediately. **AWAY** stages, never adopts, and **does NOT accumulate a review queue** — the return surface is the EOD recap. **EVENING** runs the early slot, strength checks and briefing, then stops; queues silently. **OFF** does nothing automatic, including no auto-pick adoption.
- **Quiet hours:** every **automatic** starter is gated on `autopilot_core.auto_scanning_due`, fail-open. **Manual buttons are never gated.**
- **Phone push:** **AWAY is the only Auto mode that pushes routine output**, with exactly two exceptions — Research/Focus price alerts (every mode) and EVENING's SPY ±1% wake alarm. Gate any new ntfy sender on `auto_mode == AWAY` or state why it belongs with those two.
- **The adoption gate compares timestamps at one seam** (`_gate_moment`): normalize by ATTACHING market-local to the naive side, **never by stripping the aware side**.

**Performance and correctness on the Qt thread**
- **Nothing expensive belongs on the Qt thread, and "expensive" includes a stylesheet.** Lists **diff, never rebuild**; widget variants live in `theme.qss` keyed on object names and dynamic properties; materialization goes through `ChartDataService.cached_bar_dicts`. **The theme sizes fonts in px**, so `QFont.pointSizeF()` is `-1` and arithmetic on it is a bug.
- **A burst of one signal is ONE reaction, and the coalescing lives at the LISTENER.** `focusChanged` still fires per store mutation; every listener wraps its rebuild in `ui.timer_utils.SignalCoalescer` (200 ms **leading-edge** window - later requests fold in and never restart it, so a trickle cannot starve). The DESK adoption drain adopts at most `AUTO_ADOPT_BATCH_LIMIT` (10) staged picks per cycle: **pacing only, nothing withheld, no pick dropped, a deferred pick is never marked seen.**
- **All cyclic GC runs on the GUI thread** (`gc.disable()` is process-wide, so that timer is the only collector). Activity may DELAY a sweep but **never CANCEL one** — every wait carries a deadline in ticks. Any future "wait for quiet" here needs a bound. **The startup heap is swept once and then `gc.freeze()`d** (2026-08-31, `main()` after the window shows): the widget tree, the theme and every import can never be garbage, and re-walking them was 6.5 min of that day's freeze. Collect BEFORE freezing — the other order makes startup garbage immortal.
- **A candle's four prices carry an invariant:** `low <= open, close <= high`. A bar that breaks it is drawn dashed, hollow and clamped, kept out of the scale, and logged — **never silently dropped.**
- **The scan cycle is timed, and the instrument must never become a scheduler.** `ScanCycleClock` measures and formats and decides nothing; a test fails if it ever calls `sleep`, `wait`, `start` or `Thread`.

**Evidence, journal and statistics**
- **A human-focus pick is identified by its CATEGORY as well as its name** (2026-09-01). `human_focus_tracking._pick_key` is `(trade_date, symbol, side, category slot)`, the slot being the base source with any like-origin suffix stripped — so `focus_swing_vetted` and `focus_swing` are ONE swing membership and a re-snapshot never duplicates a row. Without the category a name on both lists lost one row, and `human_focus_swing_vetted` had **0 rows in 4,083**. **Every join over these files uses `pick_source_family`**; a walkaway replays ONE position per (date, symbol, side).
- **A like, a veto AND a pass merge into their cohorts on the same click, through one helper** (`capture_rail._merge_cohort_safely`; the pass's half landed with P5 and this sentence did not follow it). They are read side by side, so a difference between them must come from the data. Failure degrades to a status suffix — the annotation row is already on disk and both merges are idempotent. The nightly slot stays.
- **A pre-versioning veto pools with the version that INTRODUCED its code**, never with the lowest version overall — `compressed` arrived in v2 and its three pre-versioning picks graded alone forever. Pooling stays inside `_rebuild_pooled_performance`; rows are never rewritten. **Never assert a literal `vocab_version`** — load the vocabulary and discover the late codes.
- **The review scoreboard grades every explicit decision, and `r_gap` is report-only.** An action joins `TAKE_ACTIONS`/`REJECT_ACTIONS` on what its WRITER does, never on its name (`veto_day_trade` is a REJECT: the D1 chart shown was vetoed). Machine events, `*_fired`, `*_expired` and every `disarm_*` stay out. `r_gap` fires on the R difference alone, never the take rate, and is deliberately absent from `draft_policy_from_state`, `review_guidance` and the AI evidence package. **Coded vetoes annotate the `dislike_reason` dimension and never re-resolve an episode**; a side disagreement is skipped, never guessed.
- **Evidence stores are never allowed to cost the thing they record** — a failed append loses the event, never the pick, tracker save or trade. **The one exception is a journal WRITE, which fails loudly.**
- Ground rule 10's statistics contract lives once in `scripts/evidence_stats.py`; `outcome_semantics.claim_kind` decides what may be averaged as a trade at all (**59% of the outcome store is annotations**).
- **A sweep-finalized trade counts under the policy that MEASURED it.** `setup_scoreboard.exit_policy_r` derives `eod_hold` / `stop_exit` / `last_measured` as separate columns. **They are never blended**, and every eod-hold view reads `r_eod_hold`, never `close_r`. `usable` = at least one policy measured the row.
- **The Market Journal is what the trader thought; the Journal is what they traded** — two stores, deliberately not merged. **An entry is never backdated**: `written_after_the_session` is COMPUTED. A capture joins by `entry_id` from outside, so a note never waits on a chart. Both surfaces use `shared_journal_service()`.
- **Auto-tagging has two lanes and they never compete.** `journal_analytics.AutoTagger` answers "which of my setups was this?" from the scanner's own output files and LEADS both the stored summary and the candidate list; `journal_trade_shape` derives facts (hold, entry session, execution shape, instrument) from the trade's own timestamps and legs, so imported history — where the scan files cannot reach — is tagged instead of blank. **No tag is ever derived from the outcome**, or every per-tag statistic becomes circular. Unmeasurable emits NO tag. Candidate ordering is by LANE, never confidence: shape tags are facts carrying 1.0 and would otherwise bury every setup match. Accepting a suggestion drops that SUGGESTION from the queue, never the trade.
- **A third lane offers what the trader already SAID, and it is not a link** (P6, 2026-09-01). `trader_capture` matches the trade's OWN window - open date to close date, never the fuzzy neighbourhood - and outranks every fuzzy source; a rejection is PREFIXED (`vetoed:`/`passed:`) so it can never read as an endorsement. `context_row_id` is a POINTER FOR A READER: plan.md P5.3/P5.4 own the canonical opportunity id and a second one must never be invented. The same rule governs `preference_trade_outcomes`, which joins statements to trades - **every row renders its match confidence or says "no match"**, because a trade on the same name that week may have been taken for another reason. A dimension resting on almost nothing (under 10% confirmed-tag coverage) SAYS SO and is **never hidden**.
- **The trader owns `trade_annotations`, and there is exactly ONE machine writer** (P6a, 2026-09-01). `scripts/journal_bulk_tag.py` may write a setup tag for a CLOSED trade that has no confirmed one, and only as `tag_status='provisional'` - `confirmed` / `provisional` / `needs_review` are the three lanes, and existing rows became `confirmed` through the column's DEFAULT. The refusal to overwrite a confirmed row lives in `JournalStore.apply_provisional_tags`, **not in the caller**. It never promotes a shape tag, and it **never writes `tag_corrections`** - that table is the trader's feedback TO the tagger, so only an EDIT teaches it and agreeing with a guess must not. Below its threshold it writes NO tag, only a `needs_review` marker. **"My setups" counts confirmed tags only**; `provisional setups` is a separate analytics group and the two are never blended.
- **The Questrade refresh chain has ONE owner**: `refresh_access_token` holds `local_writer_lock`, **re-reads the token inside the lock**, and saves atomically; a 401 from someone else's rotation picks up THEIR access token rather than burning a refresh. A failed refresh saves nothing. The attempt cap counts failures against a DAY, not a cause. Repair is a TRADER action. Not every FAILED day is repairable — 44 of 45 predate the executions retention horizon.
- **A broker statement is authoritative for money and blind to time.** `journal_statement_import` reads Questrade's activity export (.xlsx via `zipfile`+`ElementTree`, no `openpyxl`, so no packaging trigger) for the days the executions endpoint's retention horizon can no longer reach. `Net == Gross + Commission` on every trade row, so the ONE commission column is the whole cost and is never split into a guessed fee. It carries **no time of day** (every row is 12:00:00 AM), so executions are written at MIDNIGHT MARKET-LOCAL and `journal_trade_shape.is_date_only` refuses to name a session for them — a date-only round trip is a `day_trade`, **never a `scalp`**. Options come from the DESCRIPTION (the Symbol column is a Questrade internal id; trusting it loses the 100 multiplier). **Long vs short comes from the DESCRIPTION too** — Questrade writes `STOCK SHORT.` and `COVER SHORT.`, and `leg_rank` orders each row by what it does to the position; row order is a SORT, not a sequence (227 of 227 round trips list the sell first), and the uid tiebreak had been deciding direction by coin flip. **Nothing positional may reach an execution uid** or a later, longer export re-imports the whole overlap — identity is `fill_signature` plus an ordinal within it, and the uid's `rank` prefix is also the assembler's intra-day tiebreak. **A statement NEVER writes into a (broker, account, day) a richer source already covers** — the two give one fill different uids, so the upsert cannot see the duplicate; the day is refused and counted. `reconcile_statement` adds the file up by hand and compares: journal P&L is recomputed from price x qty and drifts from Questrade's rounded cents by **-$0.24 on $5,299 realised across 428 closed symbols**, worst symbol 1.2c, **commission exact to the cent**.
- **An IBKR transaction file is read separately, and its money is in the BASE currency.** `journal_ib_transactions` reads IBKR's SECTIONED csv (a per-section header; a plain `DictReader` misaligns every later table). `Price` is USD while `Gross`/`Net` are CAD, so a passed-through row computes a USD gross and subtracts a CAD commission — costs are converted with the rate the row itself implies, `|Gross| / |qty x price x multiplier|` (608 rows ran 1.3553-1.4527, the USD/CAD band), recorded as evidence and **never booked into `fx_rates`**, which is BoC-only. **Account numbers arrive MASKED** (`U***2524`): `resolve_account_number` unmasks only when EXACTLY ONE known account fits, else keeps the mask and reports it — a guess splits one position in two. An `Assignment` IS a fill (side from the description). Options already arrive OCC. One "Import statement file..." button serves both brokers and **reads the broker from the file**, never from its name.
- **Commission carries a SIGN, and the importer owns it.** `upsert_executions` and the assembly path no longer `abs()`: every importer already normalizes a charge to a positive cost, so it is a no-op for Questrade/Flex/socket/CSV/manual. What `abs()` was losing is a broker CREDIT — 18 of 609 IBKR fills — which it turned into a charge, overstating the year's cost by twice the credit. That single sign was the **entire** $2.17 by which the IB file and the journal disagreed; with it fixed, IB reconciles to **-0.0000 across 150 closed symbols** with commission equal to four decimals.
- **A broker file outranks the live sync on MONEY, and never on time** (trader decision 2026-08-28). Neither broker's downloadable file carries a time of day, so a blanket override would discard every intraday timestamp the journal has. `journal_file_authority` compares the two per `(account, day)` on **computed signed cash** — `sign x qty x price x multiplier - commission - fees`, never a Gross/Net column, because Questrade reports in the trade's currency and IBKR in the base currency and the two are not comparable. The sync KEEPS a day they agree on (its times survive); the file TAKES a day they do not, retiring the sync's rows with append-only `VOID_EXECUTION` adjustments (I3 — nothing is deleted, a superseding record undoes it). A day the file does not mention is a gap, not a disagreement, and is never touched. Tolerance is **per fill**, not flat, because Questrade rounds each row to the cent. "Check a statement..." runs the same comparison as a DRY RUN so the trader sees which days would move before any do.
- **The tax number is the BROKER's, never ours** (trader decision 2026-08-28: *"Statement is source of truth for final pnl/tax purposes"*). Every other P&L in the journal is RECOMPUTED — average-cost matching, price x qty — which is what makes per-setup statistics possible and also drifts from the broker's cent-rounded figures (-$0.24 on $5,299 across the year). `journal_tax_report` recomputes NOTHING: it sums `raw_executions.net_amount`, the broker's own statement of each fill's cash, and for a FLAT position that sum IS the realised P&L, so no cost-basis model is needed or used. It **refuses** rather than estimates — an open position, one whose opening fill was invented (`SYNTHETIC_OPEN`), or one with any fill lacking a stated amount is EXCLUDED and named with its reason. CAD converts per fill at the booked BoC rate; an unbooked date withholds that position's CAD total rather than guessing. A `VOID_EXECUTION` row never reaches a total. The recomputed figure sits beside it as a cross-check, never blended.
- **Address home-folder stores by their `project_paths` named constants** — resolving by name under the wrong root shipped a blank page for six days.
- **The overnight runner's `veto_cohort_grading` slot is deterministic and calls no model.** Later phases append to `default_slots()`; they never reorder. Nothing in this chain may reach a detector, score, alert, watchlist, Focus, the review queue or `review_policy.json`.

**Charts and boards**
- D1 charts carry a volume underlay and an earnings ribbon drawn INSIDE the price view; neither votes on the price range. Earnings headroom is reserved for EVERY symbol. The cache holds no future dates, so the next report is projected and labelled `est`. Payloads are built on the ChartDataService worker, never on the paint path.
- Chart paint lines: `scripts/chart_levels.py` builds the `levels` payload on the **worker**, never the paint path.
- Price alerts: Focus and Research share one `PriceAlertService`. The `read_only` mode survives the Desk Link removal and now has no production caller.
- **The group RS/RW tape owns its own clock** (`scripts/group_rrs.py` pure formula + `ui/services/group_tape_service.py`): ONE batched `yfinance` download per 5-minute tick, no retry inside the tick, **zero IB traffic and no `legacy.py` change**. Session filter is completed M5 bars **plus a same-date filter**; a window without `length + 2` bars is `None` and draws NOTHING (0.0 would claim "in line with SPY"). The RS Window tab still reads `rrsSnapshotChanged` — it answers a different question.
- **M5 Strength Board:** batched yfinance over `universe_all.txt`, **zero IB traffic**. Every board add re-runs the adoption gate at click time. **A row click charts into the Visual Alert Review pane** via `chart_symbol` (the lookup box's door), never `_enqueue_review_alert` (the scanner's door, which drops in AWAY, drops parked symbols, diverts M5 to the alert bar and can hide a row behind movers-only). **It is a section under the Desk's Strength window, not a page** (trader, 2026-08-31) - one `StrengthBoardService` owned by `MainWindow`, hosted through `AlertCenterPanel.attach_strength_board`, **starting closed** because the alert column's 360 px floor is width the charts would otherwise lose. Its RS/RW half retired with the page; the Alert Center's RS/RW Board tab is the surviving view.
- **One completed-bar rule** (`scripts/completed_bars.py`): `bar_start + bar_minutes <= now`, **inclusive**, timezone-converted with `astimezone` and **never** `replace(tzinfo=None)`. BounceBot's ad-hoc copies migrate opportunistically, never as a silent change to a shipped detector.
- Pure indicator modules (`scripts/indicators/`): completed bars in, immutable tuples out, `None` for anything unmeasurable. **No importer yet — the first one fires the packaging trigger.**
- Auto/Away phone output: `autopilot_today.txt` is the single verified home-folder digest, safety/freshness header first.

## Hard invariants (plan.md sec 5 — never violate)
- Decision-support only: never add order execution.
- Legacy SPY pause detection and D1 wick alerts are the champions; shadow engines must never influence live decisions until plan.md sec 7 promotion gates pass.
- No detector/scoring behavior change without golden-result fixtures first (plan.md Sections 5 and 7).
- Never swap `calc_anchored_vwap_bands`' σ formula — every consumer is calibrated to the running-deviation variant.
- Completed bars only for state transitions; a forming bar is preview. Missing data is uncertainty, never confirmation.
- User-entered watchlist names are never auto-removed (CandidateRegistry enforces this; keep it true in any new writer).
- One component owns each timer/thread/job/mutable shared export; a failed publish never destroys the last verified report.
- Point-in-time research uses only information available at the simulated decision time; timestamps carry explicit timezones.
- `review_policy.json` ranks and annotates only — it deliberately has no suppression field; do not add one.

## Tech stack + key deps
- Python ≥3.12 (desk `.venv` measured 3.12.13, a uv-managed CPython built 2026-08-07; the repo venv
  has no `pip` — install with `uv pip install -r … -c constraints.txt --python .venv\Scripts\python.exe`),
  Windows-first with macOS support (`docs/MACOS_SETUP.md`; same code, no fork — platform differences live in launchers, `project_paths.py`, and `ai_credentials.py`), repo-local `.venv`.
- `PySide6`/`qtawesome`/`pyqtgraph` — new Trading Desk UI (`PyQt5` remains only for legacy `TickerMover.py`); Tk — legacy GUI.
- `ibapi` — IBKR market data; `yfinance` — fallback bars; `pandas`/`pyarrow` — bar frames and arrow-backed columns.
- `feedparser` — news RSS for market prep; `openai` — provider-neutral one-way advisory summaries (`scripts/ai_summary.py`, `market_prep/services/ai_service.py`).
- `pytest` (markers: `network`, `broker`, `slow`, `qt`), `ruff` (narrow defect-class select), `pyinstaller` — packaging, via `packaging/tradingbotv3.spec`.
- Layered installs: `requirements-core.txt` (headless) ⊂ `-gui` ⊂ `-dev`, pinned by `constraints.txt` for reproducibility.

## Commands
- Test (before every commit): `.venv\Scripts\python.exe -m pytest tests/ -q` — must be fully green; current baseline lives in `CURRENT_CHECKPOINT.md`. Check pytest's own exit code, not a piped tail's. macOS/Linux: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q` (Qt tests need the offscreen platform when headless).
- Lint (before every commit): `.venv\Scripts\python.exe -m ruff check .` — must be `All checks passed`. Clean since 2026-08-31; it was 1,703 findings the first time it was ever run here, four of them real bugs. Narrow select by design (`E9`, `F63`, `F7`, `F82`, `F401`) — widen as the legacy cores shrink. **Fix the code, not the config**; a `# noqa` needs the reason beside it (the two in `test_qt_alert_center.py` are availability probes, and `technical_integrity.row_capture_mode` is a re-export).
- Smoke (offline, deterministic): `.venv\Scripts\python.exe scripts/smoke_check.py` — 7/7.
- Run: `.venv\Scripts\python.exe launch_gui.py` (Windows; also the `trading_desk.cmd` launcher — **this is the production launch**, see Frozen exe rebuild policy) or `.venv/bin/python launch_gui.py` (macOS/Linux; `./setup_macos.command` once first). There is no desk-role choice any more — the Desk Link tab and its flags were removed 2026-08-24. IB TWS/Gateway runs on the main desk.
- **One desk per machine** (R10.A, 2026-08-23): `launch_gui.py` takes a
  machine-local slot (`scripts/single_instance.py`) and a second launch prints
  "another TradingBotV3 desk is already running" and exits 0. `--selftest` and
  `--run-scan` are outside the guard; `--allow-second-instance` overrides it. It
  fails OPEN if the machine has no exclusion primitive - the outcome
  finalization transaction fences itself independently, so the guard is defence
  in depth rather than the thing correctness rests on.
- Audits: `scripts/operations_audit.py` (runtime), `scripts/review_capture_audit.py` (capture readiness) — both also render in System Health.
- No deploy pipeline: the user runs the app from this repo on `main`. Never leave the working tree broken.

## Frozen exe rebuild policy
Build: `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm` → `dist/TradingBotV3/TradingBotV3.exe`
(onedir, ~400MB, ~4 min). `dist/` and `build/` are gitignored, so the exe is never a commit artifact —
rebuilding is verification only, and skipping it can never leave the tree broken.

**Rebuilding is not the same as delivering, and a build that completes is not a build that
runs.** `dist/` being gitignored means an unrebuilt commit cannot break the *tree*; it also
means it cannot reach the *desk*. Always start the exe (or its `--selftest`) after building —
success from PyInstaller is not evidence that Windows will let it launch.

**The desk runs from source** — `.venv\Scripts\python.exe launch_gui.py`, normally via the
`trading_desk.cmd` launcher — **by trader decision (2026-08-26), and it stays that way until a
deliberate rebuild + frozen selftest is scheduled.** The original reason was that **Windows Smart
App Control was enforced** (`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy` →
`VerifiedAndReputablePolicyState = 1`) and refused the unsigned local build with "An Application
Control policy has blocked this file" — open from 2026-08-19, and on 2026-08-21 the trader was
launching the exe. **On 2026-08-26 the registry reads OFF** (`VerifiedAndReputablePolicyState = 0`,
`SAC_PreviousState = 1`, `SAC_EnforcementReason = 6`). Read the registry value, never recall it. **SAC verdicts are per file hash**, so one build can run for days while the next
is refused — never assume the last successful frozen run generalizes. SAC has no exclusion list;
the only exits are a reputable code-signing certificate or turning SAC off, which cannot be undone
without reinstalling Windows. That is the trader's call.

While this holds, **the source launch IS production**: a pushed commit is live at the trader's next
restart and the exe is a verification artifact only. If the trader ever returns to the frozen exe,
it becomes production again and so does the delivery gap it carries — a fix the trader will actually
use is not delivered until the exe is rebuilt, which is what kept the `d0aebd5` responsiveness
repair off the desk overnight on 2026-08-20 and made that night's `ui_stalls.jsonl` a pre-fix
baseline rather than diagnostic evidence.

- **Do NOT rebuild per commit.** ~4 min machine time plus 5-10 min of the user's click-through is not
  worth it on the ~90% of commits that cannot affect freezing. Logic changes inside existing modules
  are invisible to PyInstaller.
- **Rebuild before each merge to `main`** (same point as the plan.md sec 6 live-validation day), and
  immediately when a change hits a trigger below. Ask the user before spending their time on the
  click-through; the build itself is unattended.
- **Both guards are now BUILT** (2026-08-09, branch `claude/a4-paint-lines-packaging-nug5km`):
  - `tests/test_packaging_spec_drift.py` executes the spec with the PyInstaller API stubbed and
    asserts every top-level `scripts/` package is in its `collect_submodules` list and every
    non-`.py` runtime asset is covered by a `datas` rule. It found the spec five packages behind
    the tree (`ai_jobs`, `desk_link`, `gui_app`, `indicators`, `market_prep_gui`). `desk_link` was
    bundled from then until P1.5 **removed the package entirely (2026-08-24)**; `indicators` and
    `ops` are bundled; the rest are documented allowlist entries — each unreachable from
    `launch_gui.py`, the frozen entry point.
    **Fix the spec, never the test** — deliberate omissions go in its documented allowlists.
  - `launch_gui.py --selftest` (`scripts/selftest.py`) imports every lazily-loaded engine and loads
    every `__file__`-relative asset (theme.qss, the veto vocabulary), no window and no network,
    exiting non-zero with every failure named. Run it against the FROZEN exe:
    `dist\TradingBotV3\TradingBotV3.exe --selftest`. Expect `selftest OK: N/N checks passed (frozen)` and exit 0 - **N is a running total that grows as checks are added, not a fixed number**; it was 29 on 2026-08-09, 30 later, and the unfrozen tree measured 72 on 2026-08-27. Compare the run against the *current* unfrozen count, never against a number recalled from a doc - that is what replaces the trader's click-through (desk-verified 2026-08-09).
  - The two lists must never contradict each other: a package in `PACKAGES_NOT_IN_THE_BUNDLE` cannot
    also be in `selftest.LAZY_ENGINE_MODULES`, because the frozen exe genuinely does not contain it.
    The unfrozen suite cannot see such a clash — a repo checkout imports anything under `scripts/` —
    so `test_the_selftest_never_demands_a_package_the_bundle_excludes` now asserts the two are
    disjoint. It exists because `ai_jobs` was in both, the unfrozen selftest passed 30/30 all week,
    and the desk's first frozen run (2026-08-09) was the first execution anywhere to catch it.
  - Between them, triggers 2-4 below are now caught by the normal test run.
- **Triggers — a change of these kinds can break the bundle, so rebuild and run the frozen selftest:**
  1. New third-party dependency (`requirements-*.txt` / `constraints.txt`) — may need hiddenimports or `collect_data_files`. **Not** covered by the guards.
  2. New non-`.py` runtime asset. The spec mirrors every `FIRST_PARTY_PACKAGES` tree plus `config/`; an asset outside those silently goes missing. *(spec-drift test catches it)*
  3. New top-level package under `scripts/` that is imported lazily — the spec's `collect_submodules` list is hardcoded. *(spec-drift test catches it)*
  4. New dynamic import by string name (`importlib`, name-keyed panel/service lookup) in an uncollected package. *(add the module to `selftest.LAZY_ENGINE_MODULES` — but only if a frozen run can actually reach it; see the disjointness rule above)*
  5. Any change touching `__file__` / `ROOT_DIR` / `sys.path` — `ROOT_DIR` is `sys._MEIPASS` when frozen. **Not** fully covered; the selftest checks the phantom-root assumption only.
- Read `packaging/README.md` "Things that will bite you" before touching the spec or any of the above.
  The signature failure is a bundle that starts fine and dies at the first lazy import, so "it launched"
  is not proof; the selftest is what exercises the engines.

## Working agreement for agents
- Follow the mandatory documentation workflow above. `plan.md` owns build order;
  `CURRENT_CHECKPOINT.md` owns the active item. Do not re-implement anything in
  `CHANGELOG.md` or implement anything directly from `WISHLIST.md`.
- `main` is the trunk; branch per milestone/packet, merge back after a live-session validation day passes (plan.md sec 6).
- Commit small and green; push after each commit. If a task will exceed usage limits, commit and push so another agent can take over from a green state.
- First live session on any new build: run plan.md sec 6 checklist; do NOT tune thresholds from one session.
- **File-scoped ask-first rule** (checkpoint review 2026-08-08): any edit to a file housing detector/scoring/alert code is asked about BEFORE it is made — even for capture-side or evidence-only changes in that file. Ambiguity is the trigger to ask, not a license to judge.
- While unmerged branch code runs in production via a scheduled task (see `docs/CHECKPOINT_REVIEW_2026-08-08.md`): never switch branches on the desk without disarming that task first.

## Where to read more
- `CHANGELOG.md` — **`Current implemented inventory` is the contract: search it before building.** `Recent changes` holds the last two build days; older entries are archived under `docs/`.
- `docs/DESK_INTERNALS.md` — the incident, measurements and trader conversation behind every `Core loop / data flow` rule, verbatim. Read the matching entry before changing what a rule governs.
- `plan.md` — remaining roadmap and single source of truth for unfinished work. Sec 5 invariants, sec 6 live validation, sec 7 promotion ladder, sec 12 ordered work queue. Read before any feature work.
- `CURRENT_CHECKPOINT.md` — **read the `Active state at a glance` block**: branch, active items, baseline, open gates, next action. Dated entries below are the record behind it.
- `WISHLIST.md` — trader-visible candidate integrations and deferred ideas; never an implementation queue.
- `docs/README.md` — classifies every supporting file as active runbook/reference or historical evidence.
- `docs/BRANCH_HISTORY.md` — what each development branch held and where it landed; the containment proof (`git merge-base --is-ancestor <branch> main`) required before deleting one, and the branches deliberately left open.
- `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` (+ historical `GUI_LEARNING_PROGRESS.md` pointer) — preserved GUI learning design; never overrides plan.md Sections 5–7 or Phase order.
- `GUI_PRODUCT_PLAN.md` — historical consumer GUI product design reference.
- `docs/decisions/` — backfilled decision records; read before changing a library, storage, or architecture choice.
- **`docs/decisions/0016-trader-vision-and-priorities.md` — the trader's goals and their twelve answers of 2026-09-02, the tie-breaker for every prioritisation call: names before entries, win rate as the swing headline, MFE after a held level for day trades, "what is working lately" on the Trading Desk never in Research, likes are training data. Read it before proposing or ordering work.**
- `docs/REVIEW_LEARNING_LOOP.md` — how the AI reads review artifacts and writes `review_policy.json`.
- Phase 0.5 trader refinement packets (promoted 2026-08-15; **R3–R8 authorized by the trader's 2026-08-15 weekend redirect and the 2026-08-18 integration redirect**, consolidated onto `main` on 2026-08-26): `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` (R1 — **BUILT**, 3 live proofs still owed), `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` (R2 — **BUILT**, 3 live proofs still owed), `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md` (R3 — **CLOSED 2026-08-16**; §4.3.5 volume-thrust normalization deferred by trader decision; shadow week owed), `docs/DESK_CHART_UNIFICATION_PLAN.md` (R4 — **BUILT 2026-08-16**, §8 exit gate owed; its two held items resolved 2026-08-18), `docs/M5_SIGNAL_ENGINES_PLAN.md` (R5 — **CODE COMPLETE 2026-08-18**: every engine built, the four newest alert types default OFF, so §7's per-engine desk session now decides audibility rather than existence), R6 (**CLOSED 2026-08-18**; only the stall-watchdog diagnostic week is owed), `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` (R7) and `docs/WEEKEND_PREP_PLAN.md` (R8) — both **BUILT**, live gates owed. **`CURRENT_CHECKPOINT.md` opens with the 2026-08-18 morning report — read that first.** Phases 1–7 remain NOT authorized.
- `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` — LOCKED implementation plan for the DAS research warehouse (capture policy + IB pacing budget, 13-table schemas, Phases 0-8 build order, 28 locked decisions in its Section 23 — do not re-litigate them; open items live only in its confirmation register).
- `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md` — the warehouse builder decision log (BD-01…): every implementation choice the locked plan left open, with rationale and reopen triggers. Read before changing warehouse internals; add a BD entry when you make a new one. `docs/RESEARCH_WAREHOUSE_ERD.md` is its dataset/identity map.
- `docs/SETUPS_MAJOR.md` / `docs/SETUPS_TEST.md` — AI-stated understanding of the production setups and the study/research setups, under trader review; fold corrections back in.
- `docs/FIRST_SESSION_CHECKLIST.md`, `docs/AWAY_SCANNER_RUNBOOK.md`, `docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md` — operational runbooks for live sessions.
- `docs/FOCUS_PRICE_ALERTS_PROPOSAL.md`, `docs/EVENING_MODE_RUNBOOK.md` — Focus price-alert delivery and ntfy phone setup. `docs/MULTI_MACHINE_DESK_PROPOSAL.md` is historical and is now the ONLY record of that design: Desk Link/satellites retired 2026-08-08, code removed 2026-08-24.
- `docs/LOCAL_AI_AUTOMATION_PLAN.md` — local LLM batch layer on the always-on main desk: automated AI summaries, daily digest ledger, journal enrichment, review-policy curation, frontier synthesis. Advisory-only; no inference during market hours. **Section 7 (2026-08-20)** owns the deterministic `veto_cohort_grading` slot and the opt-in `trader_judgement` scope, including what is deliberately NOT built (the weekly synthesis: cadence decided, gated on two weeks of graded rows, unauthorized).
- `docs/DURABILITY_CATCHUP_PLAN.md` — durability design: self-healing launch task, deterministic backfill with `capture_mode` provenance, never-reconstruct boundary, and the Master AVWAP tracker staleness override.
- `docs/CHART_REVIEW_WORKSPACE_PLAN.md` — Chart Review workspace and trader decision capture: `trader_annotations.jsonl` schema v1, the versioned veto vocabulary, veto forward-tracking cohorts, and why a lookup never writes a watchlist. The stream is analysis-only evidence — it must never mute, suppress, score, gate, or alert.
- `docs/MACOS_SETUP.md` — running the desk on macOS (native TWS, Keychain keys). Its Google Drive mount-discovery sections are dead since decision 0015; the code is harmless and stays until a macOS run is actually needed.
- `docs/decisions/0015-no-cloud-sync-das-file-server-storage.md` — no cloud sync; the DAS is the durable tier. Read it before touching storage paths, the writer lease, or backup rules.
- `docs/SHIP_READINESS.md`, `docs/BROKER_ADAPTERS.md`, `packaging/README.md` — shipping direction and future multi-broker architecture.
- Runtime facts: main desk is an always-on Ryzen 7 8845HS mini-PC (32GB DDR5, Radeon 780M iGPU — local-LLM host) and does everything; the former i5-8600K/3080 Ti desktop is powered down most days (discord/chat, at most ad-hoc alternative scanning — never an always-on or writer role). Storage is a DAS file server at `\\MINI-PC\Trading Bot Data`, expandable to ~100TB, holding `research_lake/`, `ai_store/`, and the cold-pushed `data/`, `output/`, `logs/`, `away_report_archive/` subtrees. Full scan ≈ 28.5 min, network-bound (measured on the old desk); the 8845HS measured 17–21 min on 2026-08-10 over 1,097 symbols. Post-session artifacts under `%LOCALAPPDATA%\TradingBotV3\diagnostics\` (`run_manifests\`, `spy_state_shadow.jsonl`, `greatness_shadow.jsonl`, `job_ledger.jsonl`, `heartbeat.json`).

`AGENTS.md` is a copy of this file (symlinks don't survive Windows checkouts) — edit CLAUDE.md, then re-copy.
