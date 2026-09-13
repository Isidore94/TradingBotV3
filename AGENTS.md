# TradingBotV3 — AI context index

TradingBotV3 is a Windows desktop decision-support system for one trader's day and
swing trading. It does everything except execute orders: pre-session market prep,
candidate discovery (D1 anchored-VWAP swing scans + intraday 5-min bounce detection),
live monitoring with alerts, unattended Auto/Away scanning with a phone report, a
journal, and a controlled research/promotion program for new setups. Order execution
is permanently out of scope (plan.md sec 1).

## Agent routing - read before doing work

- **The agent team.** Read [`docs/AGENT_TEAM.md`](docs/AGENT_TEAM.md) before delegating: `.claude/agents/` serves Claude; `.codex/agents/` serves Codex. Tester proves failures, builder fixes, reviewer reproduces; the lead checks handoffs against the diff and alone integrates, with code workers isolated from the desk checkout.
- **Codex delegation (trader, 2026-09-09).** Astra owns planning, orchestration and final acceptance; explicitly delegate bounded recon and simple work to Luna, and implementation, tests and independent review to Terra wherever useful work can run alongside the lead. Use explicit model selection and narrow context, escalate Luna to Terra before asking Astra for a focused decision, and never silently spend on Astra helpers; `.codex/config.toml` and the runbook own defaults and runtime fallback details. These Codex rules do not change Claude's model choices.

## How to talk to the trader (trader rule 2026-08-26)

**Write every message to the trader as if they are five years old.** Very short.
Very simple words. One idea per sentence. Say what you did, what is broken, and
what they need to do - nothing else. No long lists, no tables, no section
headers, no code words unless the trader has to type them. If a message is
longer than about ten short lines, cut it. Detail belongs in the docs and the
commit message, not in the chat. This rule is for chat output only; docs, code
comments and commit messages keep their normal depth.

## Workspace memory (WISHLIST 11, trader 2026-09-12)

Long form and the authority reconciliation: `docs/DESK_INTERNALS.md` "Workspace memory
is recall, never authority".

- **`MEMORY.md` at the root is a ROUTING INDEX only** (name -> detail file -> trigger keywords, never a fact); the detail lives under `memory/`. At idle boot read the standing instructions and `MEMORY.md`; once a task exists the narrow reads below apply unchanged.
- **Before answering about prior work, decisions, dates, people or preferences, search memory first:** route through `MEMORY.md`, read the narrowest detail file, at most five sources, cite file, tag and date. A live-status question is answered from the checkpoint and the code, never from memory.
- **Memory is recall, never authority.** `CURRENT_CHECKPOINT.md` is the brief, `plan.md` the build order, `CHANGELOG.md` the inventory, `docs/decisions/` the contracts and the code the fact; a detail file outranks its index. No recalled line authorizes a detector change, overrides a decision, promotes a WISHLIST item or bypasses the ask-first rule.
- **Every non-blank line in `people/`, `projects/` and `decisions/` carries `[stated]` / `[observed]` / `[inferred]` / `[suggested]`, a date and a source**; only the trader's own words support `[stated]`; never keys, account numbers, live counts or machine status. An inferred lesson becomes a rule only after three weighted independent signals across two sessions; a trader correction applies at once.
- **Supersede in place** (strike the old line with its date, the replacement beside it); update `MEMORY.md` in the same commit; consolidate before 15,000 characters per file. Recon, reviewer and tester never write memory - they hand the lead a sourced line; the lead integrates. Claude's auto-memory is private scratch.

## Mandatory documentation workflow for every AI

**Read narrow, not everything.** The bounded read below is the instruction — widen it
only when the narrow read leaves a real question open. An agent that cannot read its
brief skims it and then appends to it, which is what grew these files to 1 MB once.

Before proposing, planning, or changing anything:

1. `CURRENT_CHECKPOINT.md` — read the **"Active state at a glance"** block at the top:
   branch, active roadmap items, last verified baseline, open gates, next action. That
   block is the brief. Read a dated entry below it only for the item you are touching;
   if a dated entry contradicts the block, the dated entry wins.
2. `plan.md` — Sections 5 (invariants), 6 (live validation) and 7 (promotion), then the
   phase table at the head of Section 12. Read the body of your phase only.
3. `CHANGELOG.md` — **search** `Current implemented inventory` for the feature you are
   about to touch, so you do not rebuild landed work. Never read it end to end.
4. `docs/README.md` — one line per file; open only the spec, runbook and decision
   records for the selected item.
5. Inspect the source, tests, Git status/history and runtime artifacts needed to verify
   that the documentation still matches reality. **When the docs and the code disagree,
   the code is the fact and the doc is the defect** — fix the doc, and say so.

`docs/archive/` is history (checkpoint, changelog and roadmap archives, retired designs,
built prompts, frozen August reviews). Open one file there for one specific question;
**never load it as context** and never read an entry there as an open gate.

`WISHLIST.md` contains ideas, not authorized work. Never implement directly from it. An
item enters the build sequence only when the trader explicitly moves it into `plan.md`.

Before editing, state the exact roadmap/checkpoint item, what already exists, what
remains, governing documents, expected files, tests, and whether the ask-first rule
applies. Do not skip to a later phase because it is easier or more interesting.

After every repository change, reconcile the documentation before handoff:

- update `CURRENT_CHECKPOINT.md` with the active item, working state and verification
  result (or say why the baseline is unchanged), and **refresh the glance block** — a
  stale block is worse than none;
- update `CHANGELOG.md` when behavior, contracts, architecture, operations or an
  implementation status changed;
- remove, narrow or advance the corresponding `plan.md` work while retaining any
  live-validation or promotion gate still owed;
- update the governing spec or decision record when its contract or rationale changed;
- update `WISHLIST.md` only for trader-directed idea changes; an AI may recommend a
  promotion but must not silently make one;
- update `docs/README.md` whenever a Markdown file is added, moved, removed or
  reclassified;
- keep `CLAUDE.md` and `AGENTS.md` identical whenever operating instructions change;
- **keep the active files small.** Size rules: `CLAUDE.md` under ~45 KB (a rule here is
  one to three sentences and names its seam; its story, numbers and quotes go in
  `docs/DESK_INTERNALS.md`); the glance block
  under ~6 KB with one line per gate; `CURRENT_CHECKPOINT.md`'s dated entries hold the
  last **three build days** and older ones move to `docs/archive/`; `CHANGELOG.md`'s
  Recent changes holds two; a BUILT phase in `plan.md` is a stub pointing at the
  roadmap archive. Archiving is maintenance, not a new document.

Do not create another roadmap, progress ledger, handoff, or status file. The root
control set is `CLAUDE.md`/`AGENTS.md`, `CHANGELOG.md`, `plan.md`,
`CURRENT_CHECKPOINT.md`, `WISHLIST.md`, and `docs/README.md`; `MEMORY.md` + `memory/` sit
beside it as recall, never status (see Workspace memory). Prompts, reports and
assessments go in chat or an artifact, never a committed `.md` (trader rule 2026-09-04).

## Core loop / data flow

Each rule below is binding as written. The incident, measurements and trader
conversation behind every one are in [`docs/DESK_INTERNALS.md`](docs/DESK_INTERNALS.md)
— **read the matching entry there before changing the behaviour a rule governs**, and
change both places when a rule changes.

**Shape**
- Entry: `launch_gui.py` → `scripts/ui/app.py` (PySide6 Trading Desk). One desk role, no flag; Desk Link/satellite, the mini-PC scanner, the Tk UI, `TickerMover.py` and `PyQt5` are all REMOVED, not dormant.
- Market data: IBKR TWS/Gateway `127.0.0.1:7496` (`ibapi`) primary, `yfinance` fallback, bar source tracked per scan (`docs/BROKER_ADAPTERS.md`). On the desk the D1 scan's daily bars are PINNED to Yahoo by `local_settings.json` `daily_bars_source: "yahoo"` — a manifest full of Yahoo daily-bar successes is the pin working, not IB failing. IB serves intraday bars and the champion's M5 loop.
- Engines: `scripts/master_avwap.py` (+`master_avwap_lib/`) D1 AVWAP swing scanner; `scripts/bounce_bot.py` (+`bounce_bot_lib/`) intraday M5 bounce detector; `market_prep/` pre-session services.
- Inputs: plain-text watchlists (`longs.txt`, `shorts.txt`, `swinglongs.txt`, `shortswings.txt`) in the shared home folder.
- Storage: `C:\TradingBotData` is a plain LOCAL folder — no cloud drive (decision 0015). Per-machine caches under `%LOCALAPPDATA%\TradingBotV3` (`scripts/project_paths.py`); address home-folder stores by their `project_paths` named constants. The DAS `\\MINI-PC\Trading Bot Data` is the durable tier: **write local first, move to the DAS after.**
- Shadow engines (`market_state.py`, `greatness_monitor`) emit JSONL promotion evidence only. Review-learning loop: Alert Center decisions → `alert_review_events.jsonl` → `review_learning.py` → AI-curated `review_policy.json` → chart annotations (`docs/REVIEW_LEARNING_LOOP.md`).

**Research warehouse** (contract `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`, decisions `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`, identities `docs/RESEARCH_WAREHOUSE_ERD.md`)
- Shadow-only additive evidence with **zero detector/score/alert influence**, at `research_store_dir`, never inside `C:\TradingBotData` (unset = disabled).
- The post-scan build runs in an owned CHILD PROCESS at below-normal priority (F1, BD-95), never a thread; reads are session-scoped and MONTH-keyed through `ResearchStore.read_rows`, never by filtering a materialised list.
- **Never widen `_run_outcomes` to a date filter** — its walk runs FORWARD across sessions (BD-66/69/74).
- The seal de-duplicates at the dataset grain and counts what it drops; repair is `research_warehouse.cli dedupe --apply` (dry run by default); derived rows from a duplicated month are wrong in VALUE and need a rebuild (BD-96/97).
- A SNAPSHOT over 64 MB is stored whole but never `json.loads`-ed; the UNCHANGED watermark is a chunked hash (BD-73).
- H2/H4 exist for the HTF LRSI study (BD-78), end each session with a STUB excluded from the LRSI input, and the grid is the 16 diagnostic recipes in `outcomes.HTF_LRSI_RECIPES`, never a Cartesian search (BD-79). Live `CROSS_LEVELS` stays `(20, 50)`.
- `anchor_instance` comes from `earnings_avwap_anchors.csv`, fed by the SCAN through `runner.bridge_earnings_anchor_caches_to_csv` -> `append_anchor_candidates` (append-only, de-duplicated on ticker + anchor_date, new rows at the END, failure logged never raised). Nothing live reads the CSV; never trim it.
- A reconstructed anchor is LABELLED and never promotion evidence (BD-99/100): `AnchorChoice` `observed` / `reconstructed`, carried by `feature_snapshot_daily.anchor_knowledge` and `outcome_path.path_kind`; repair order build -> `rebuild-daily-features` -> `recompute-outcomes` -> `band-coverage`.
- **The daily snapshot carries BOTH AVWAP band families and they never share a column** (M4, BD-102): `avwape_*` is the frozen champion, `avwap_variant_*` the challenger computed from the SAME bars independently of the champion (a NULL band is "not measured"); `swing_house_variant_v1`'s `outcome_definition_id` fences it out of every `house_default_v1` reader. Shadow only. Long form: DESK_INTERNALS "M4".
- The setup registry (`scripts/setup_registry.py`, `setup_registry_v1.json`) is frozen DATA regenerated by `build_setup_registry.py --write`; an unknown name RAISES; nothing in production imports it before `plan.md P4.1`. `trial_ledger.register` writes one append-only row per grid BEFORE any outcome is read.
- The like-link payload field is `match_basis` and `LikeLink.from_payload` is its only reader, strict in both directions; the dataset is `bronze_like_occurrence_link`.
- **The `setup_research` narration is a BOUNDED view and its selection is a SIZE rule, never a ranking by result** (N3, BD-101): `_bounded_narration_view` fills by `stats.n` descending then name; **no R statistic may enter that key** (gate #43); `narrated K of N` is stated everywhere it is read. Long form: DESK_INTERNALS "N3".

**Alert Center, review queue and capture**
- The charts own the review pane; at most ONE slim verb row sits between them and the tab strip. **The arm bar stays UNDER the chart** (host decision via `AlertChartReview(dock_arm_bar=…)`); never propose moving it without asking. Rail shortcuts are rebound at panel scope — a `QShortcut` in a hidden tab never fires and two bindings for one sequence fire neither.
- **A VETO retires the chart, a CLAIMED like ADVANCES it, a QUICK like and a NOTE move nothing** (trader, 2026-09-04): `vetoRetireRequested` -> `_retire_after_veto` (ONE row); `likeRecorded` -> `_after_like`; `likeAdvanceRequested` -> `_advance_after_like` -> `_advance_review_queue` (parks and drops nothing); both likes write `like_advance` because `review_learning.TAKE_ACTIONS` keys on it. Long form: DESK_INTERNALS "T1".
- **A day-trade PASS is a note, not a veto, and never retires the chart.** Its codes are a SEPARATE vocabulary family (`ui/annotations/vocabularies/pass_reasons_v*.json`); cached M5 bars are referenced through a sidecar written BEFORE the row (`ui/annotations/pass_bars.py`); a capture click never fetches; a pass does not mark the symbol "Reviewed today".
- **A LIKE has two modes and only one names a setup** (P9): **Alt+L** is the QUICK like (`like_mode: "quick"`, never prompts), **Alt+K** the CLAIMED like; an absent `like_mode` reads `claimed`; a like carries zero privileges and contributes a LINK to the auto-tagger, never a tag. `sidecar_completion` writes a NEW file (`m5_bars_completed_ref`) and **its read is AWARE** (N1): `pass_bars.desk_zone()` is ATTACHED to a naive `dt`, never stripped. Long form: DESK_INTERNALS "P9 / N1".
- **Every verdict has a forward record and no two verdicts are combined** (P5): veto, like, pass, rejection (`focus__m5_not_today` / `focus__swing_dislike`, the double underscore load-bearing); pass cohorts overlap and only `pass_all` counts passes; `unfavorite` is never graded; an unreachable pass grade is BLANK with `intraday_unmeasured_reason`, never zero.
- Veto vocabulary is versioned and codes are never reused; cohort identity on write is `(vocab_version, reason_code)`, rows are never rewritten, pooling happens only in `_rebuild_pooled_performance`. **Never assert a literal `vocab_version` in a test.**
- **PROVEN is the top alert class; BANGER no longer exists.** The `banger` review-event column stays a constant `False`; `REGIME_BANGER_*` in `legacy.py` is a regime-pause threshold, untouched.
- **The LRSI M5 alerts are RETIRED and their evidence is not.** `LRSI_M5_ALERTS_RETIRED` gates only the EMIT seam in `_emit_lrsi_cross_alert`; detection, the outcome row and the PROVEN stamp still run. **Never flip `M5_SIGNAL_TYPE_DEFAULTS` for these two** — that would stop the evidence.
- Feed repetition control is display only and withholds nothing: one live row per symbol+side+day, repeats fold with an ×N badge, privileged output bypasses the fold, the backing list is written BEFORE any repetition decision. **No suppression field exists in this chain.**
- Movers-only chart review is a default-on PRESENTATION filter: hides and counts, never deletes, mutes or writes `review_policy.json`; both legs are asked at SHOW time; UNKNOWN always SHOWS, tagged `unmeasured`.
- Intraday alerts are a list beside the chart (`ui/widgets/m5_alert_bar.py`, LEFT column), not a queue in front of it: clicking from one row to the next is a SKIP, never a re-queue, and **a click away IS a pass** (trader, 2026-09-01) - never "fix" it, never rename `clicked_away_from_m5_alert`. Routing is `_is_m5_review_alert` inside `_enqueue_review_alert`, after the AWAY branch.
- "Holding highs" is measured in ATR (1.0 ATR, never a percentage), expires 15 minutes after the later of the alert and the last new extreme, and is deleted from the review queue only; uncertainty never deletes. With-trend rows auto-join M5 Focus (`scripts/regime_pause_focus.py`, DESK only).

**Focus, gating and modes**
- **M5 Focus adoption gate** — one definition in `scripts/focus_adoption_gate.py`: beyond yesterday's extreme AND right side of session VWAP on the last **completed** M5 bar, UNKNOWN always failing; session VWAP from `chart_snapshot.session_vwap_series`. Stored verdicts expire at 45 min or 2 completed bars.
- **A Focus pick's automatic D1 alerts are PULLBACKS only** (`_poll_focus_d1_interest` evaluates `D1_PULLBACK_KINDS`); the extension set fires only when the trader ARMED it, through `_poll_d1_event_watches`. The gate is at flag GENERATION, so nothing is suppressed downstream.
- **An armed alert expires in TRADING days** (5 for a 5d extreme watch, 10 otherwise) counted by `market_calendar.trading_days_between`, policy once in `scripts/armed_alert_expiry.py`; uncertainty never deletes, every expiry appends a row, a price alert is DISARMED never deleted, and expiry runs at the head of the poll that owns each store.
- **A quiet Focus pick FADES after 10 trading days**, reversibly (`focus_pick_clocks.json`), through the store's own removal path so a watchlist line is never touched; faded is not deleted (`focus_faded.json` + append-only row), a faded swing favorite appends a RETRACTION, no `pick_feedback` verdict is written; `FocusPickStore` is the single writer, checked on the day roll and a half-hourly timer. Long form: DESK_INTERNALS "A Focus pick that never speaks fades".
- Focus provenance: `focus_auto_picks.json` marks machine-adopted entries; **absence of a marker means the trader owns it**, and only marked entries are reachable by "Not today" or desync repair.
- Today's swing picks (`ui/widgets/swing_favorites_bar.py`) get two writes - swing Focus FIRST and must not fail, then the append-only `swing_favorites.jsonl` row whose failure is swallowed; never an auto-adoption marker; like-origin `vetted`; a removal appends a RETRACTION. The strip is the BOTTOM of the M5 alerts column.
- Auto-mode matrix (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`): discovery is identical in every mode; DESK adopts staged picks immediately, AWAY stages and never adopts (its return surface is the EOD recap), EVENING runs the early slot and briefing then stops, OFF does nothing automatic.
- Quiet hours: every **automatic** starter is gated on `autopilot_core.auto_scanning_due`, fail-open. **Manual buttons are never gated.**
- Phone push: **AWAY is the only Auto mode that pushes routine output**, with two exceptions — Research/Focus price alerts (every mode) and EVENING's SPY ±1% wake alarm. Gate any new ntfy sender on `auto_mode == AWAY` or say why it belongs with those two.
- **The Trade Mentor prompts only a PRESENT trader** (WS-TM, WISHLIST 10J steps 1-2): whole-hour reads from 07:00 Pacific in DESK / EVENING / OFF, never in AWAY, never a push; an answer is one dated Market Journal row, a missed hour is recorded as skipped, and the Settings checkbox ships OFF. It is the one narrow fixed-time exception to the quiet-hours rule (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` amendment 2026-09-12). Long form: DESK_INTERNALS "TM - a prompt is a slot, an answer is a dated row".
- The adoption gate compares timestamps at one seam (`_gate_moment`): attach market-local to the naive side, never strip the aware side.

**Performance and correctness on the Qt thread**
- **Nothing expensive belongs on the Qt thread, and "expensive" includes a stylesheet.** Lists diff, never rebuild; widget variants live in `theme.qss` keyed on object names and dynamic properties; materialization goes through `ChartDataService.cached_bar_dicts`. The theme sizes fonts in px, so `QFont.pointSizeF()` is `-1`.
- **A burst of one signal is ONE reaction, coalesced at the LISTENER** (`ui.timer_utils.SignalCoalescer`, 200 ms leading-edge). The DESK adoption drain adopts at most `AUTO_ADOPT_BATCH_LIMIT` (10) staged picks per cycle: pacing only, nothing withheld, no pick dropped, a deferred pick never marked seen.
- **All cyclic GC runs on the GUI thread** (`gc.disable()` is process-wide). Activity may DELAY a sweep but never CANCEL one; every wait carries a deadline in ticks. The startup heap is swept once and then `gc.freeze()`d in `main()` after the window shows — collect BEFORE freezing.
- A candle's four prices carry an invariant (`low <= open, close <= high`); a bar that breaks it is drawn dashed, hollow and clamped, kept out of the scale, and logged — never silently dropped.
- `ScanCycleClock` measures and formats and decides nothing; a test fails if it ever calls `sleep`, `wait`, `start` or `Thread`.
- The exchange calendar is memoized; the stall watchdog's cap is per HOUR; every thread's CPU time is measured once a minute (`ui/thread_cpu_gauge.py`, always on), so a thread holding the lock leaves the GUI stack innocent. A recon that rates a timer from its docstring has not measured it.
- The research M5 tee de-duplicates BEFORE any per-bar work (`capture_m5_tee`: identity pass, then work for survivors) against a per-symbol high-water mark persisted in `tee_high_water.json`, never reset by a clock (BD-96).
- **The scanner's hold on the interpreter is measured and being removed one seam at a time** (SN, `docs/DESK_INTERNALS.md` "SN"): SN5/SN6 (breathe, trader picks first), SN4 (the feed diffs itself), SN3 (one RRS pass) and SN2 (new bars only) are built; SN1 (the scanner in its own process) waits for their live proof by the trader's own prompt; the duplicate fourth `_record_environment_focus_history` call is a scoring input and stays until asked.
- The daily pick scorecard streams both CSVs on its own worker (`autopilot-scorecard`) and writes `picks_scored_at` only on success; `read_scorecard_inputs` keeps today's rows only; three failures give up for the day (`picks_scoring_failed_at`) (Q5).

**Evidence, journal and statistics**
- A human-focus pick is identified by its CATEGORY as well as its name: `human_focus_tracking._pick_key` is `(trade_date, symbol, side, category slot)`; every join over these files uses `pick_source_family`; a walkaway replays ONE position per (date, symbol, side).
- A like, a veto AND a pass merge into their cohorts on the same click through one helper (`capture_rail._merge_cohort_safely`); failure degrades to a status suffix; the nightly slot stays.
- A pre-versioning veto pools with the version that INTRODUCED its code, inside `_rebuild_pooled_performance`; rows are never rewritten.
- The review scoreboard grades every explicit decision; an action joins `TAKE_ACTIONS`/`REJECT_ACTIONS` on what its WRITER does, never its name (`veto_day_trade` is a REJECT); machine events, `*_fired`, `*_expired` and `disarm_*` stay out; `r_gap` is report-only and absent from `draft_policy_from_state`, `review_guidance` and the AI evidence package.
- **Evidence stores are never allowed to cost the thing they record** - a failed append loses the event, never the pick, tracker save or trade. **The one exception is a journal WRITE, which fails loudly.**
- Ground rule 10's statistics contract lives once in `scripts/evidence_stats.py`; `outcome_semantics.claim_kind` decides what may be averaged as a trade.
- A sweep-finalized trade counts under the policy that MEASURED it (`setup_scoreboard.exit_policy_r`: `eod_hold` / `stop_exit` / `last_measured` never blended); **`unresolved` means UNMEASURED** (M2), a measured sweep is `swept_measured`, every reader goes through `outcome_semantics.terminal_kind`, and the champion's eod-hold cells take `eod_complete` only (golden in `tests/test_setup_scoreboard.py`). Long form: DESK_INTERNALS "M2".
- The Market Journal is what the trader thought; the Journal is what they traded - two stores, never merged, both through `shared_journal_service()`; `written_after_the_session` is COMPUTED, never backdated; a capture joins by `entry_id`.
- Auto-tagging has four lanes that never compete and are ordered by LANE, never confidence: `trader_capture` (what the trader already SAID inside the trade's own window, outranking every other source), `trader_note` (a setup they NAMED in the Market Journal inside that window - whole-token vocabulary, `match_basis = note:<id>` and the quoted span, opposite-side notes never match, a date-only fill is `unmeasured`), `journal_analytics.AutoTagger`'s scanner lane (which setup), and `journal_trade_shape` (facts from the trade's own timestamps and legs). No tag is ever derived from the outcome and unmeasurable emits NO tag. The rejection prefixes, `context_row_id`, the note lane's window and the match-confidence rule: DESK_INTERNALS "The four auto-tagging lanes".
- The trader owns `trade_annotations`; the ONE machine writer is `scripts/journal_bulk_tag.py`, writing only `tag_status='provisional'` for a CLOSED trade with no confirmed tag (the refusal to overwrite lives in `JournalStore.apply_provisional_tags`), never a shape tag, never `tag_corrections`; below threshold only a `needs_review` marker. "My setups" counts confirmed tags only.
- **`preference_trade_outcomes` matches inside 10 SESSIONS** and **`trade_level_summary` sums P&L once per `trade_id`**; **`journal_exposure` reads bias from the legs** (a LONG option is never a bullish setup); **`personal_evidence_summary` partitions by STATUS with `uncertain` a CROSS-CUTTING label that pools no money**, naming no best setup below `MIN_REPORTABLE_N`. Long form: DESK_INTERNALS "The personal-evidence reader rules".
- The Questrade refresh chain has ONE owner (`refresh_access_token` under `local_writer_lock`, token re-read inside the lock, atomic save, a failed refresh saves nothing, the cap counts failures per DAY). Repair is a TRADER action.
- **A broker file is authoritative for money and blind to time.** `journal_statement_import` and `journal_ib_transactions` write executions at MIDNIGHT market-local and `journal_trade_shape.is_date_only` refuses to name their session; identity is `fill_signature` plus an ordinal; **commission carries a SIGN the importer owns - nothing downstream may `abs()` it**. Long form: DESK_INTERNALS "A broker file is authoritative for money".
- **A broker file outranks the live sync on MONEY and never on time** (`journal_file_authority`, per `(account, day)` on computed signed cash, tolerance per fill): the sync KEEPS a day they agree on, the file TAKES a day they do not through append-only `VOID_EXECUTION` rows, an unmentioned day is untouched. "Check a statement..." is the same comparison as a DRY RUN.
- **The tax number is the BROKER's**: `journal_tax_report` sums `raw_executions.net_amount`, recomputes nothing, refuses rather than estimates (open, `SYNTHETIC_OPEN`, amount-less and unbooked-FX positions EXCLUDED and named), and shows the recomputed figure beside it, never blended.
- The setup tracker is mirrored into a SQLite record store after every JSON save (`scripts/tracker_store.py`, shadow, never able to fail the save); the JSON is still the truth and no reader loads from SQLite until gate #57, then ONE AT A TIME (decision 0017).
- **The tracker replay has a versioned execution convention and level knowledge** (ST3/ST7, `master_avwap_lib/execution_convention.py`): default since 2026-09-06 is `gap_aware_v2` / `prior_session_v2`; `literal_level_v1` / `same_session_v1` remain by name, every record stamped. Long form: DESK_INTERNALS "ST3".
- **Which observation of a thesis gets graded is a NAMED policy** (ST4/ST7): `selection_policy` on every family row, `first_actionable_v2` the default since 2026-09-06 (decision 0019), `closed_first_v1` by name; a COMPACTED record is `undatable_exit` and **a compact projection's `_scoring_outcome_summary` IS the record**. Long form: DESK_INTERNALS "ST4 / ST7".
- **The AVWAP band challenger is measured through the CATCH-UP path too** (M1): `build_anchor_band_variant_meta` in `legacy.py` serves BOTH the live scan and the tracker catch-up; never add a third builder; the Band variant tab prints coverage from the EXPORT's own counts, never the 1.1 GB tracker. Long form: DESK_INTERNALS "M1".
- **The control, study and experimental-exit populations are SURFACED, LABELLED, and never mixed with picks** (M5): three Setup Tracker tabs over three CSVs from the tracker's own guarded save pass, win rate first with `n` and the ONE Wilson bound, `experimental` a COLUMN, champion aggregates pinned byte-identical. Long form: DESK_INTERNALS "M5".
- The overnight runner's `veto_cohort_grading` slot is deterministic and calls no model. **Stage order is decision 0018's**: deterministic slots, digest, narration, then model-gated slots (`EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`); a later phase appends inside its stage. Nothing here may reach a detector, score, alert, watchlist, Focus, the review queue or `review_policy.json`.
- The digest gate has TWO halves (Q4): `clean_digest_sessions` counts CONSECUTIVE clean exchange sessions, and `digest_audit_approval.json` is written **only** by `python -m ai_jobs.digest approve-audit`, never by a nightly job; `gate_met = window_met and audit_recorded` and `journal_enrichment` refuses until both hold. Long form: DESK_INTERNALS "Q4".
- `entry_index.json` is the deterministic compact handoff written beside the packs at the end of `run_daily_digest` (temp-and-rename, a failure never fails the digest): four sections never merged, `changes_vs_prior_window` by FLOOR STATUS only, trials UNRANKED, every `pack_path` the file the numbers were READ from.

**Charts and boards**
- D1 charts carry a volume underlay and an earnings ribbon drawn INSIDE the price view; neither votes on the price range; the next report is projected and labelled `est`. Payloads and `scripts/chart_levels.py`'s `levels` are built on the ChartDataService worker, never the paint path.
- Focus and Research share one `PriceAlertService`; its `read_only` mode has no production caller.
- The group RS/RW tape owns its own clock (`scripts/group_rrs.py` + `ui/services/group_tape_service.py`): ONE batched `yfinance` download per 5-minute tick, no retry inside the tick, zero IB traffic; a window without `length + 2` completed same-date M5 bars is `None` and draws nothing.
- **M5 Strength Board:** batched yfinance over `universe_all.txt` plus the four watchlists, zero IB traffic, relative volume SESSION-RELATIVE and outside the seven fenced formula functions; parity rows auto-join M5 Focus through `_auto_adopt_strength_board` (DESK only, the ONE adoption gate, never removing) and every Focus add is injected into `longs.txt` / `shorts.txt` by `FocusPickStore._inject_into_shared`. Long form: DESK_INTERNALS "The M5 Strength Board's auto-adoption".
- **Every ticker click on the Trading Desk charts into the centre Visual Alert Review pane** through `chart_symbol`, never `_enqueue_review_alert`; a board chart holds NO place in the waiting list (`_is_manual_chart_look` on `MANUAL_CHART_TAG`) and `show_board_symbol` is the popup door for a board on ANOTHER page. Long form: DESK_INTERNALS "Every ticker click lands on the centre chart".
- **The Desk's Strength window is ONE flat scrolling page** (`ui/widgets/strength_page.py`, trader 2026-09-07): no tabs, no collapsible sections, one scrollbar, every board sized to its content (`FIT_ROWS_CAP`), one `StrengthBoardService` owned by `MainWindow`. Long form: DESK_INTERNALS "The Strength window is one flat page".
- **One completed-bar rule** (`scripts/completed_bars.py`): `bar_start + bar_minutes <= now`, inclusive, converted with `astimezone` and never `replace(tzinfo=None)`. BounceBot's ad-hoc copies migrate opportunistically, never as a silent change to a shipped detector.
- Pure indicator modules (`scripts/indicators/`): completed bars in, immutable tuples out, `None` for anything unmeasurable.
- Auto/Away phone output: `autopilot_today.txt` is the single verified home-folder digest, safety/freshness header first.

**Headline statistics and the priority switch (V3, decision 0016)**
- **The priority switch reorders and never withholds, and it is BUILT** (ST6): `local_settings` key `prioritise_working_lately`, default OFF, read AT SORT TIME never at write time, stably re-ordering the M5 list, the WAITING review list and the setups table by the snapshot's verdict order with ties keeping arrival order; the tier gate, movers-only and the repetition fold are untouched (identical-visible-rows test).
- **ONE Working-lately snapshot, four surfaces, and nothing called proven** (ST6): `working_lately.build_snapshot` is PURE (`snapshot_id` a sha1 over cells + policy lines + `as_of`), `ui/services/working_lately_service.py` owns the build and `snapshot_latest.json`; dependence is answered by REFUSING (`CONCENTRATION_LIMIT` 0.5, `pool_cells` RAISES across kinds, `observational leader among K cells`). Long form: DESK_INTERNALS "ST6".
- **Win rate leads every trader-facing SWING surface** (with `n` and `swing_headline`'s Wilson lower bound, sorted by the bound); **MFE-after-a-held-level leads every DAY-TRADE surface**. ONE WILSON (z 1.96; `expected_r.py`'s 1.28 is fenced), integer counts at each table's grain (ST2), ONE leader through `working_lately.select_leader`, and the tier outcomes `win` is a FAVORABLE-DIRECTION flag, not a stop-rule win (ST1, `swing_evidence.read_eligible_rows` the ONE reader). Long form: DESK_INTERNALS "Headline statistics - the ST1/ST2 clauses".
- **The day-trade headline is `held_run_score`**: P(held in the first 30 min) x trimmed-mean MFE_R of the held ones, ONE formula on every surface; **held is MEASURED held** (Q1: `hold_rate` = held / MEASURED, the unmeasured shown never assumed); My Decisions rows use `ALL_DIRECTIONS`, a pooled cell never an average of two. Long form: DESK_INTERNALS "Q1".
- **The AWAY digest ranks swing picks by the tracker's record**: Wilson lower bound on the family's realized win rate at `evidence_stats.SWING_HORIZON_SESSIONS` (5) inside `lately_window()`, expected R as tiebreak, ungraded families last, the near cap applied after ranking; the bucket is printed, never ranked on. Points order only when the trader's Points switch is ON (WS-PT4).
- **Research is not a trader surface - except its Results page** (decision 0016 answer 7, amended 2026-09-06; `scripts/research_results.py`): the readable full readout over ONE ST6 snapshot, four populations never pooled, computing no new statistic; its window control APPLIES to My trades (`in_window`, on `closed_at`) and is DISABLED on Bot, whose cells carry their own `window_sessions` (DESK_INTERNALS "G5"). Nothing the trader must see may live only in the other eight tabs.
- **"Lately" is ONE number in trading sessions**: `evidence_stats.LATELY_SESSIONS` (20) walked on the exchange calendar; `review_learning.DEFAULT_WINDOW_SESSIONS` IS it; Weekend Prep's week is `WEEK_SESSIONS` (5); every surface says **sessions**.

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
- Python ≥3.12 (desk `.venv` measured 3.12.13, a uv-managed CPython; the repo venv has no `pip` — install with `uv pip install -r … -c constraints.txt --python .venv\Scripts\python.exe`), Windows-first with macOS support (`docs/MACOS_SETUP.md`; same code, no fork), repo-local `.venv`.
- `PySide6`/`qtawesome`/`pyqtgraph` — the Trading Desk UI, the only UI (`PyQt5` is excluded by the spec as a guard).
- `ibapi` — IBKR market data; `yfinance` — fallback bars; `pandas`/`pyarrow` — bar frames and arrow-backed columns.
- `feedparser` — news RSS for market prep; `openai` — provider-neutral one-way advisory summaries (`scripts/ai_summary.py`, `market_prep/services/ai_service.py`).
- `pytest` (markers: `network`, `broker`, `slow`, `qt`), `ruff` (narrow defect-class select), `pyinstaller` — packaging, via `packaging/tradingbotv3.spec`.
- Layered installs: `requirements-core.txt` (headless) ⊂ `-gui` ⊂ `-dev`, pinned by `constraints.txt`.

## Commands
- Test (before every commit): `.venv\Scripts\python.exe -m pytest tests/ -q` — must be fully green; current baseline lives in `CURRENT_CHECKPOINT.md`. Check pytest's own exit code, not a piped tail's. macOS/Linux: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q`.
- Lint (before every commit): `.venv\Scripts\python.exe -m ruff check .` — must be `All checks passed`. Narrow select by design (`E9`, `F63`, `F7`, `F82`, `F401`). **Fix the code, not the config**; a `# noqa` needs the reason beside it.
- Smoke (offline, deterministic): `.venv\Scripts\python.exe scripts/smoke_check.py` — 7/7.
- Run: `.venv\Scripts\python.exe launch_gui.py` (Windows; also `trading_desk.cmd` — **this is the production launch**, see Frozen exe rebuild policy) or `.venv/bin/python launch_gui.py` (macOS/Linux; `./setup_macos.command` once first).
- **One desk per machine** (`scripts/single_instance.py`): a second launch prints "another TradingBotV3 desk is already running" and exits 0; `--selftest` and `--run-scan` are outside the guard; `--allow-second-instance` overrides it; it fails OPEN without an exclusion primitive.
- Audits: `scripts/operations_audit.py` (runtime), `scripts/review_capture_audit.py` (capture readiness) — both also render in System Health.
- No deploy pipeline: the user runs the app from this repo on `main`. Never leave the working tree broken.

## Frozen exe rebuild policy

Long form, with the Smart App Control history, in `docs/DESK_INTERNALS.md` ("Frozen exe rebuild policy, long form").

Build: `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm` → `dist/TradingBotV3/TradingBotV3.exe` (onedir, ~420 MB, ~4 min). `dist/` and `build/` are gitignored; rebuilding is verification only.

**The desk runs from SOURCE** (`trading_desk.cmd` → `launch_gui.py`) by trader decision (2026-08-26), so **a pushed commit is live at the trader's next restart** and the exe is a verification artifact. If the trader ever returns to the frozen exe, a fix is not delivered until the exe is rebuilt. Smart App Control verdicts are per file hash; **read the registry value, never recall it** (`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy` → `VerifiedAndReputablePolicyState`).

- **Do NOT rebuild per commit.** Rebuild before each merge to `main` and when a trigger below is hit; ask before spending the trader's time on the click-through. **A build that completes is not a build that runs** - always run `dist\TradingBotV3\TradingBotV3.exe --selftest` and expect `selftest OK: N/N checks passed (frozen)`, N compared against the unfrozen count.
- Guards: `tests/test_packaging_spec_drift.py` (packages and non-`.py` assets; deliberate omissions in `PACKAGES_NOT_IN_THE_BUNDLE`) and `launch_gui.py --selftest` (`scripts/selftest.py`, imports every lazily-loaded engine). The two lists stay disjoint.
- **Triggers:** (1) a new third-party dependency; (2) a non-`.py` runtime asset outside the first-party trees plus `config/`; (3) a new top-level package under `scripts/` imported lazily; (4) a dynamic import by string name in an uncollected package (`selftest.LAZY_ENGINE_MODULES` only if a frozen run can reach it); (5) anything touching `__file__` / `ROOT_DIR` / `sys.path` - `ROOT_DIR` is `sys._MEIPASS` when frozen.
- Read `packaging/README.md` "Things that will bite you" before touching the spec. "It launched" is not proof; the selftest exercises the engines.

## Working agreement for agents
- **Edit surgically.** Use `Edit` for a small or medium change; rewrite a file only when it is short or most of it is changing.
- Follow the mandatory documentation workflow above. `plan.md` owns build order; `CURRENT_CHECKPOINT.md` owns the active item. Do not re-implement anything in `CHANGELOG.md` or implement anything directly from `WISHLIST.md`.
- `main` is the trunk; branch per milestone/packet, merge back after a live-session validation day passes (plan.md sec 6). Commit small and green; push after each commit.
- First live session on any new build: run plan.md sec 6 checklist; do NOT tune thresholds from one session.
- **A scratch script never resolves the live home folder by accident** (incident 2026-09-05: a reviewer's harness run outside pytest overwrote the live setup tracker; restored from the mirror). Any script that imports anything under `scripts/` outside pytest sets `TRADINGBOTV3_DATA_DIR` to a scratch directory BEFORE the import and aborts if `project_paths.DATA_DIR` resolves under `C:\TradingBotData`; test harnesses under `tests/` are never imported outside pytest, whose `conftest.py` is what points them at a test directory; a scratch export patches `MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE` itself, not an alias of it.
- **File-scoped ask-first rule:** any edit to a file housing detector/scoring/alert code is asked about BEFORE it is made — even for capture-side or evidence-only changes in that file. Ambiguity is the trigger to ask, not a license to judge.
- If a scheduled task ever runs unmerged branch code on the desk again, disarm it before switching branches there.

## Where to read more
- `CHANGELOG.md` — **`Current implemented inventory` is the contract: search it before building.** `Recent changes` holds the last two build days.
- `docs/DESK_INTERNALS.md` — the incident, measurements and trader conversation behind every `Core loop / data flow` rule. Read the matching entry before changing what a rule governs.
- `plan.md` — remaining roadmap; sec 5 invariants, sec 6 live validation, sec 7 promotion ladder, sec 12 ordered work queue.
- `CURRENT_CHECKPOINT.md` — the `Active state at a glance` block, then the last three build days.
- `WISHLIST.md` — ideas and their open trader questions; never an implementation queue.
- `MEMORY.md` — the routing index into `memory/` (trader statements, project lessons,
  decisions, dated notes); recall only — read the matching detail file and cite it.
- `docs/README.md` — one line per file; `docs/archive/` is history, never context.
- `docs/decisions/` — accepted constraints, read before changing a library, storage or architecture choice. **`0016-trader-vision-and-priorities.md` is the tie-breaker for every prioritisation call**: names before entries, win rate as the swing headline, MFE after a held level for day trades, "what is working lately" on the Trading Desk never in Research, likes are training data.
- `docs/BRANCH_HISTORY.md` — what each branch held and where it landed; the containment proof before deleting one.
- Active specs and runbooks: one line per file in `docs/README.md`; open only the spec, runbook and decision records for the item in hand.
- Runtime facts: the main desk is an always-on Ryzen 7 8845HS mini-PC (32 GB, Radeon 780M iGPU, local-LLM host); the old i5/3080 Ti desktop is never a writer; the DAS at `\\MINI-PC\Trading Bot Data` holds `research_lake/`, `ai_store/` and the cold-pushed subtrees; a full scan is 17-21 min over 1,097 symbols; post-session artifacts under `%LOCALAPPDATA%\TradingBotV3\diagnostics\`.

`AGENTS.md` is a copy of this file (symlinks don't survive Windows checkouts) — edit CLAUDE.md, then re-copy.
