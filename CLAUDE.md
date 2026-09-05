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
`CURRENT_CHECKPOINT.md`, `WISHLIST.md`, and `docs/README.md`. Prompts, reports and
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
- The post-scan build runs in an owned CHILD PROCESS at below-normal priority (F1, BD-95), never a thread — a CPU-bound thread holds the GIL and no priority trick frees the GUI. Reads are session-scoped; partitions are MONTH-keyed, narrowed through `ResearchStore.read_rows` (`symbols` / `interval_start_range`), never by filtering a materialised list. Growth resets on the 1st.
- **Never widen `_run_outcomes` to a date filter** — its walk runs FORWARD across sessions (BD-66/69/74).
- The seal de-duplicates at the dataset grain and counts what it drops (`SealResult.rows_deduplicated`; `SUPERSEDING_DATASETS` exempt). Repair is `research_warehouse.cli dedupe --apply` (dry run by default); derived bars and intraday features computed from a duplicated month are wrong in VALUE and need a rebuild, not a dedupe (BD-96/97).
- A SNAPSHOT over 64 MB is stored whole but never `json.loads`-ed; the UNCHANGED watermark is answered from a chunked hash (BD-73).
- H2/H4 exist for the HTF LRSI study (BD-78) and end each session with a STUB, published and excluded from the LRSI input. The HTF LRSI grid is 16 diagnostic recipes (`outcomes.HTF_LRSI_RECIPES`), never a Cartesian search; both legs read the same unmirrored series (BD-79). Live `CROSS_LEVELS` stays `(20, 50)`.
- `anchor_instance` comes from `earnings_avwap_anchors.csv`, which the SCAN feeds through `runner.bridge_earnings_anchor_caches_to_csv` → `append_anchor_candidates` (append-only, de-duplicated on ticker + anchor_date, new rows at the END, failure logged never raised). Nothing live reads the CSV; never trim it — the newest two dates per ticker are the current and previous anchors.
- A reconstructed anchor is LABELLED and never promotion evidence (BD-99/100): `AnchorChoice` is `observed` / `reconstructed`, `feature_snapshot_daily.anchor_knowledge` carries it (NULL reads `legacy`), `outcome_path.path_kind` names the swing path and is excluded from BD-98's unchanged-comparison. Repair order: build → `rebuild-daily-features` (dry run by default) → `recompute-outcomes` → `band-coverage`. A terminal outcome row is re-simulated only with `force`.
- The setup registry (`scripts/setup_registry.py`, `setup_registry_v1.json`) is frozen DATA, regenerated by `build_setup_registry.py --write` with the diff reviewed, resolves no disagreement and fills no column its sources do not establish; an unknown name RAISES. It becomes authoritative at `plan.md P4.1`; nothing in production imports it yet. `trial_ledger.register` writes one append-only row per grid BEFORE any outcome is read and refuses to rewrite a `trial_id`.
- The like-link payload field is `match_basis` and `LikeLink.from_payload` is its only reader, strict in both directions; the dataset is `bronze_like_occurrence_link`.

**Alert Center, review queue and capture**
- The charts own the review pane; at most ONE slim verb row sits between them and the tab strip. **The arm bar stays UNDER the chart** (host decision via `AlertChartReview(dock_arm_bar=…)`); never propose moving it without asking. Rail shortcuts are rebound at panel scope — a `QShortcut` in a hidden tab never fires and two bindings for one sequence fire neither.
- **A VETO retires the chart, a CLAIMED like ADVANCES it, a QUICK like and a NOTE move nothing** (trader, 2026-09-04). A rail veto uses its own verb (`vetoRetireRequested` → `_retire_after_veto`) and writes ONE row; the "✕ Not today" button writes an uncoded row and opens the note box. A quick like is `likeRecorded` → `_after_like`; a claimed like is `likeAdvanceRequested` → `_advance_after_like` → `_advance_review_queue`; an advance parks nothing and drops nothing. Both write `like_advance` through `_record_like_advance` because `review_learning.TAKE_ACTIONS` keys on it. "Veto D1 — but M5 today" writes a veto row and emits a REQUEST; the panel places (first), then retires (second) through the box-free verb.
- **A day-trade PASS is a note, not a veto, and never retires the chart.** Its multi-select codes are a SEPARATE vocabulary family (`ui/annotations/vocabularies/pass_reasons_v*.json`), written in vocabulary order; cached M5 bars are referenced through a sidecar written BEFORE the row (`ui/annotations/pass_bars.py`); a capture click never fetches; a pass does not mark the symbol "Reviewed today" (`pick_feedback._ANNOTATION_DECISIONS` stays `veto`/`like_claim`/`note`).
- **A LIKE has two modes and only one names a setup** (P9). **Alt+L** is the QUICK like (`like_mode: "quick"`, no claim, no why, never prompts); **Alt+K** is the CLAIMED like (the claim is the whole gesture; the why is optional since T2). The quick-like BUTTONS prompt for an optional note; the key never does. `like_mode` is additive and its absence reads `claimed`. A like carries zero privileges, grades under `like_unclaimed` when quick, and contributes a LINK to the auto-tagger, never a tag. `sidecar_completion` finishes a capture sidecar into a NEW file (`m5_bars_completed_ref`); `m5_bars_ref` is never rewritten.
- **Every verdict has a forward record and no two verdicts are combined** (P5): veto, like, pass, rejection (`focus__m5_not_today` / `focus__swing_dislike`, the double underscore load-bearing). The rejection family's pooled base row is LABELLED and never read as either verdict. Pass code cohorts overlap and are never summed; only `pass_all` counts passes. `unfavorite` is never graded; a rejection's free-text reason is never machine-coded. A pass grade the sidecar cannot reach is BLANK with `intraday_unmeasured_reason`, never zero. `update_human_focus_outcomes`'s `pick_key` defaults to the existing identity.
- Veto vocabulary is versioned and codes are never reused; cohort identity on write is `(vocab_version, reason_code)`, rows are never rewritten, pooling happens only in `_rebuild_pooled_performance`. **Never assert a literal `vocab_version` in a test.**
- **PROVEN is the top alert class; BANGER no longer exists.** The `banger` review-event column stays a constant `False`; `REGIME_BANGER_*` in `legacy.py` is a regime-pause threshold, untouched.
- **The LRSI M5 alerts are RETIRED and their evidence is not.** `LRSI_M5_ALERTS_RETIRED` gates only the EMIT seam in `_emit_lrsi_cross_alert`; detection, the outcome row and the PROVEN stamp still run. **Never flip `M5_SIGNAL_TYPE_DEFAULTS` for these two** — that would stop the evidence.
- Feed repetition control is display only and withholds nothing: one live row per symbol+side+day, repeats fold with an ×N badge, privileged output bypasses the fold; the backing list is written BEFORE any repetition decision. **No suppression field exists in this chain.**
- Movers-only chart review is a default-on PRESENTATION filter: hides and counts, never deletes, mutes or writes `review_policy.json`; both legs are asked at SHOW time; UNKNOWN always SHOWS, tagged `unmeasured`.
- Intraday alerts are a list beside the chart (`ui/widgets/m5_alert_bar.py`, LEFT column), not a queue in front of it. Clicking from one row to the next is a SKIP, never a re-queue; **a click away IS a pass** (trader, 2026-09-01) — never "fix" it, and never rename `clicked_away_from_m5_alert`. Routing is `_is_m5_review_alert` inside `_enqueue_review_alert`, after the AWAY branch.
- "Holding highs" is measured in ATR (1.0 ATR, never a percentage) and expires after 15 minutes from the later of the alert and the last new extreme, deleted from the review queue only. Uncertainty never deletes. With-trend rows auto-join M5 Focus (`scripts/regime_pause_focus.py`, DESK only).

**Focus, gating and modes**
- **M5 Focus adoption gate** — one definition in `scripts/focus_adoption_gate.py`: beyond yesterday's extreme AND right side of session VWAP on the last **completed** M5 bar, UNKNOWN always failing; session VWAP from `chart_snapshot.session_vwap_series`. Stored verdicts expire at 45 min or 2 completed bars.
- **A Focus pick's automatic D1 alerts are PULLBACKS only** (`_poll_focus_d1_interest` evaluates `D1_PULLBACK_KINDS`); the extension set fires only when the trader ARMED it, through `_poll_d1_event_watches`. The gate is at flag GENERATION, so nothing is suppressed downstream.
- **An armed alert expires in TRADING days** (5 for a 5d extreme watch, 10 for a 20d one and for everything else), counted by `market_calendar.trading_days_between`, policy once in `scripts/armed_alert_expiry.py`. Uncertainty never deletes; every expiry appends a row; a price alert is DISARMED, never deleted. Expiry runs at the head of the poll that owns each store.
- **A quiet Focus pick FADES after 10 trading days**, reversibly (`focus_pick_clocks.json`, reset by a fired flag, a watch hit or "★ keep"); applies to the trader's own picks too, routed through the store's own removal path so a watchlist line is never touched. Faded is not deleted (`focus_faded.json` + append-only row); a faded swing favorite appends a RETRACTION; no `pick_feedback` verdict is written for a fade. `FocusPickStore` is the single writer; the check runs on the day roll and a half-hourly timer, never in the 60 s poll.
- Focus provenance: `focus_auto_picks.json` marks machine-adopted entries; **absence of a marker means the trader owns it**, and only marked entries are reachable by "Not today" or desync repair.
- Today's swing picks (`ui/widgets/swing_favorites_bar.py`) get two writes — swing Focus FIRST and must not fail, then the append-only `swing_favorites.jsonl` row whose failure is swallowed. Never write an auto-adoption marker for one; like-origin is `vetted` (cohort `human_focus_swing_vetted`); a removal appends a RETRACTION. The "taken" badge is a display-only join off the Qt thread. The strip is the BOTTOM of the M5 alerts column behind a draggable split.
- Auto-mode matrix (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`): discovery is identical in every mode; what changes is who is present. DESK adopts staged picks immediately; AWAY stages, never adopts, and does NOT accumulate a review queue (its return surface is the EOD recap); EVENING runs the early slot, strength checks and briefing, then stops; OFF does nothing automatic.
- Quiet hours: every **automatic** starter is gated on `autopilot_core.auto_scanning_due`, fail-open. **Manual buttons are never gated.**
- Phone push: **AWAY is the only Auto mode that pushes routine output**, with two exceptions — Research/Focus price alerts (every mode) and EVENING's SPY ±1% wake alarm. Gate any new ntfy sender on `auto_mode == AWAY` or say why it belongs with those two.
- The adoption gate compares timestamps at one seam (`_gate_moment`): attach market-local to the naive side, never strip the aware side.

**Performance and correctness on the Qt thread**
- **Nothing expensive belongs on the Qt thread, and "expensive" includes a stylesheet.** Lists diff, never rebuild; widget variants live in `theme.qss` keyed on object names and dynamic properties; materialization goes through `ChartDataService.cached_bar_dicts`. The theme sizes fonts in px, so `QFont.pointSizeF()` is `-1`.
- **A burst of one signal is ONE reaction, coalesced at the LISTENER** (`ui.timer_utils.SignalCoalescer`, 200 ms leading-edge). The DESK adoption drain adopts at most `AUTO_ADOPT_BATCH_LIMIT` (10) staged picks per cycle: pacing only, nothing withheld, no pick dropped, a deferred pick never marked seen.
- **All cyclic GC runs on the GUI thread** (`gc.disable()` is process-wide). Activity may DELAY a sweep but never CANCEL one; every wait carries a deadline in ticks. The startup heap is swept once and then `gc.freeze()`d in `main()` after the window shows — collect BEFORE freezing.
- A candle's four prices carry an invariant (`low <= open, close <= high`); a bar that breaks it is drawn dashed, hollow and clamped, kept out of the scale, and logged — never silently dropped.
- `ScanCycleClock` measures and formats and decides nothing; a test fails if it ever calls `sleep`, `wait`, `start` or `Thread`.
- The exchange calendar is memoized; the stall watchdog's cap is per HOUR; every thread's CPU time is measured once a minute (`ui/thread_cpu_gauge.py`, always on), so a thread holding the lock leaves the GUI stack innocent. A recon that rates a timer from its docstring has not measured it.
- The research M5 tee de-duplicates BEFORE any per-bar work (`capture_m5_tee`: identity pass, then work for survivors) against a per-symbol high-water mark persisted in `tee_high_water.json`, never reset by a clock (BD-96).
- The daily pick scorecard streams both CSVs on its own worker (`autopilot-scorecard`) and writes `picks_scored_at` only on success; `read_scorecard_inputs` keeps today's rows only; three failures give up for the day (`picks_scoring_failed_at`) (Q5).

**Evidence, journal and statistics**
- A human-focus pick is identified by its CATEGORY as well as its name: `human_focus_tracking._pick_key` is `(trade_date, symbol, side, category slot)`, the slot being the base source with any like-origin suffix stripped. Every join over these files uses `pick_source_family`; a walkaway replays ONE position per (date, symbol, side).
- A like, a veto AND a pass merge into their cohorts on the same click through one helper (`capture_rail._merge_cohort_safely`); failure degrades to a status suffix; the nightly slot stays.
- A pre-versioning veto pools with the version that INTRODUCED its code, inside `_rebuild_pooled_performance`; rows are never rewritten.
- The review scoreboard grades every explicit decision; an action joins `TAKE_ACTIONS`/`REJECT_ACTIONS` on what its WRITER does, never on its name (`veto_day_trade` is a REJECT). Machine events, `*_fired`, `*_expired` and `disarm_*` stay out. `r_gap` is report-only, fires on the R difference alone, and is absent from `draft_policy_from_state`, `review_guidance` and the AI evidence package. Coded vetoes annotate `dislike_reason` and never re-resolve an episode.
- **Evidence stores are never allowed to cost the thing they record** — a failed append loses the event, never the pick, tracker save or trade. **The one exception is a journal WRITE, which fails loudly.**
- Ground rule 10's statistics contract lives once in `scripts/evidence_stats.py`; `outcome_semantics.claim_kind` decides what may be averaged as a trade (59% of the outcome store is annotations).
- A sweep-finalized trade counts under the policy that MEASURED it: `setup_scoreboard.exit_policy_r` keeps `eod_hold` / `stop_exit` / `last_measured` as separate columns, never blended; every eod-hold view reads `r_eod_hold`.
- The Market Journal is what the trader thought; the Journal is what they traded — two stores, never merged, both through `shared_journal_service()`. An entry is never backdated: `written_after_the_session` is COMPUTED. A capture joins by `entry_id` from outside.
- Auto-tagging has three lanes that never compete and are ordered by LANE, never confidence: `journal_analytics.AutoTagger` (which setup, from the scanner's own files), `journal_trade_shape` (facts from the trade's own timestamps and legs), and `trader_capture` (what the trader already SAID inside the trade's own window, outranking every fuzzy source, a rejection prefixed `vetoed:` / `passed:`). No tag is ever derived from the outcome; unmeasurable emits NO tag; `context_row_id` is a pointer, and plan.md P5.3/P5.4 own the canonical opportunity id. `preference_trade_outcomes` shows its match confidence on every row or says "no match".
- The trader owns `trade_annotations`; the ONE machine writer is `scripts/journal_bulk_tag.py`, writing only `tag_status='provisional'` for a CLOSED trade with no confirmed tag (the refusal to overwrite lives in `JournalStore.apply_provisional_tags`), never a shape tag, never `tag_corrections`; below threshold only a `needs_review` marker. "My setups" counts confirmed tags only.
- The Questrade refresh chain has ONE owner (`refresh_access_token` under `local_writer_lock`, token re-read inside the lock, atomic save, a failed refresh saves nothing, the cap counts failures per DAY). Repair is a TRADER action.
- **A broker file is authoritative for money and blind to time.** `journal_statement_import` (Questrade `.xlsx`, no `openpyxl`) and `journal_ib_transactions` (IBKR sectioned csv, USD price / CAD money, masked accounts unmasked only when exactly one fits) write executions at MIDNIGHT market-local; `journal_trade_shape.is_date_only` refuses to name a session for them. Side and options come from the DESCRIPTION; identity is `fill_signature` plus an ordinal, never positional; a statement never writes into a (broker, account, day) a richer source already covers. Commission carries a SIGN and the importer owns it — nothing downstream may `abs()` it.
- **A broker file outranks the live sync on MONEY and never on time** (`journal_file_authority`, per `(account, day)` on computed signed cash, tolerance per fill): the sync KEEPS a day they agree on, the file TAKES a day they do not through append-only `VOID_EXECUTION` rows, an unmentioned day is untouched. "Check a statement..." is the same comparison as a DRY RUN.
- **The tax number is the BROKER's**: `journal_tax_report` sums `raw_executions.net_amount`, recomputes nothing, refuses rather than estimates (open, `SYNTHETIC_OPEN`, amount-less and unbooked-FX positions are EXCLUDED and named), and shows the recomputed figure beside it, never blended.
- The setup tracker is mirrored into a SQLite record store after every JSON save (`scripts/tracker_store.py`, `tracker_storage_shadow` default ON, never able to fail the save) and the JSON is still the truth; no reader loads from SQLite until gate #57, then readers move ONE AT A TIME (decision 0017).
- The overnight runner's `veto_cohort_grading` slot is deterministic and calls no model. **Stage order is decision 0018's: deterministic slots, then the digest, then narration, then model-gated slots**; a later phase appends inside its stage and never reorders across stages (`EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`). Nothing in this chain may reach a detector, score, alert, watchlist, Focus, the review queue or `review_policy.json`.
- The digest gate has TWO halves (Q4): `clean_digest_sessions` counts CONSECUTIVE clean exchange sessions walked through `market_calendar.previous_session` (clean = `is_session` plus an EMPTY `unavailable`), and `digest_audit_approval.json` is written **only** by `python -m ai_jobs.digest approve-audit` (from `scripts/`), never by a nightly job. `gate_met = window_met and audit_recorded`; `journal_enrichment` refuses until both are true; the System Health strip shows `sessions_consecutive_clean`.
- `entry_index.json` is the deterministic compact handoff written beside the packs at the end of `run_daily_digest` (temp-and-rename; a failure never fails the digest): four sections never merged, `changes_vs_prior_window` by FLOOR STATUS only with both pack counts, trials UNRANKED, every `pack_path` the file the numbers were READ from (newest superseding sibling). `repo_commit()` resolves HEAD through a `gitdir:` pointer because agents build in worktrees.

**Charts and boards**
- D1 charts carry a volume underlay and an earnings ribbon drawn INSIDE the price view; neither votes on the price range; earnings headroom is reserved for every symbol; the next report is projected and labelled `est`. Payloads and `scripts/chart_levels.py`'s `levels` are built on the ChartDataService worker, never the paint path.
- Focus and Research share one `PriceAlertService`; its `read_only` mode has no production caller.
- The group RS/RW tape owns its own clock (`scripts/group_rrs.py` + `ui/services/group_tape_service.py`): ONE batched `yfinance` download per 5-minute tick, no retry inside the tick, zero IB traffic, no `legacy.py` change; a window without `length + 2` completed same-date M5 bars is `None` and draws nothing. The RS Window tab still reads `rrsSnapshotChanged`.
- **M5 Strength Board:** batched yfinance over `universe_all.txt` PLUS the four trader watchlists, zero IB traffic; relative volume is SESSION-RELATIVE and is not one of the seven fenced formula functions (byte-identical to the R8 baseline); D1 SMA floors read `2y` with today's forming bar dropped. Its parity rows auto-join M5 Focus (`_auto_adopt_strength_board`: DESK only, empty `failed_floors` only, the ONE adoption gate re-run per row, skipping `_ignored_symbols` and `FocusPickStore.declined_today`, one `add_many` per side plus `mark_auto_adopted`, never `FocusService.add`, never removing). Every Focus add is injected into `longs.txt` / `shorts.txt` by `FocusPickStore._inject_into_shared` and a removal un-injects it.
- **Every ticker click on the Trading Desk charts into the centre Visual Alert Review pane** through `chart_symbol`, never `_enqueue_review_alert` (panels carry a `set_chart_sink` that `set_mode` points at `chart_symbol` in workspace mode). **A board chart holds NO place in the waiting list and is never re-queued or skip-counted** (`_is_manual_chart_look` on `MANUAL_CHART_TAG`); looking at a WAITING name takes it out for good. `show_board_symbol` is the popup door for a board on ANOTHER page. The RS/RW board (starts OPEN) and the Strength section (starts closed) are sections under the Desk's Strength window, one `StrengthBoardService` owned by `MainWindow`.
- **One completed-bar rule** (`scripts/completed_bars.py`): `bar_start + bar_minutes <= now`, inclusive, converted with `astimezone` and never `replace(tzinfo=None)`. BounceBot's ad-hoc copies migrate opportunistically, never as a silent change to a shipped detector.
- Pure indicator modules (`scripts/indicators/`): completed bars in, immutable tuples out, `None` for anything unmeasurable.
- Auto/Away phone output: `autopilot_today.txt` is the single verified home-folder digest, safety/freshness header first.

**Headline statistics and the priority switch (V3, decision 0016)**
- **The priority switch reorders and never withholds — and it is NOT BUILT YET** (V4). When built it is display-only, sorting the review queue, the M5 list and the setups table; the tier gate, movers-only and repetition control stay untouched. The identical-visible-rows test is owed WITH the switch.
- **Win rate leads every trader-facing SWING surface** (first, with `n` and a Wilson lower bound from `swing_headline`, sorting by the bound, mean R beside it); **MFE-after-a-held-level leads every DAY-TRADE surface**. ONE WILSON: `swing_headline`'s z (1.96); `master_avwap_lib/expected_r.py`'s 1.28 is a parameter inside a fenced file. Still owed: the Setup Tracker's Setup Types tab (its CSV carries no win column).
- **The day-trade headline is `held_run_score`**: P(held in the first 30 min) × trimmed-mean MFE_R of the held ones; ONE formula reaches every surface (the tracker joins `dimension_summaries`; the M5 row reads `alert_cell` + `alert_suffix`); segments are spelled the AGGREGATOR's way; `UNDERIVED_DIMENSIONS` separates "the log cannot answer" from "reachable, not derived". **Held is MEASURED held** (Q1): `measured_held` / `measured_broken` / `pending` / `unmeasured`, `hold_rate` = held / MEASURED, the unmeasured counted and shown, never assumed; the D1 dimension is the ALIGNED same-session setup from `master_avwap_tracker_scoring_snapshot.json` (index expires on the day roll), missing snapshot = UNKNOWN. My Decisions rows use `ALL_DIRECTIONS`, a pooled cell accumulated from the episodes, never an average of two cells. The window is `evidence_stats.lately_window`, gaps reported.
- **The AWAY digest ranks swing picks by the tracker's record**: Wilson lower bound on the family's realized win rate from `master_avwap_tier_outcomes.csv` inside `lately_window()` at ONE declared horizon (`evidence_stats.SWING_HORIZON_SESSIONS`, 5), expected R as tiebreak, ungraded families below every graded one, `stale_horizon` rows dropped, the near cap applied after ranking; the bucket is printed, never ranked on.
- **The Research tab is not a trader surface.** Nothing the trader must see may live only there.
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

- **Do NOT rebuild per commit.** Rebuild before each merge to `main` and immediately when a change hits a trigger below; ask before spending the trader's time on the click-through. **A build that completes is not a build that runs** — always run `dist\TradingBotV3\TradingBotV3.exe --selftest` and expect `selftest OK: N/N checks passed (frozen)`, N compared against the current unfrozen count.
- Guards: `tests/test_packaging_spec_drift.py` (every top-level `scripts/` package in `collect_submodules`, every non-`.py` asset under a `datas` rule; deliberate omissions in `PACKAGES_NOT_IN_THE_BUNDLE`) and `launch_gui.py --selftest` (`scripts/selftest.py`, imports every lazily-loaded engine). The two lists must stay disjoint.
- **Triggers:** (1) a new third-party dependency — not covered by the guards; (2) a non-`.py` runtime asset outside the first-party trees plus `config/`; (3) a new top-level package under `scripts/` imported lazily; (4) a dynamic import by string name in an uncollected package (add it to `selftest.LAZY_ENGINE_MODULES` only if a frozen run can reach it); (5) anything touching `__file__` / `ROOT_DIR` / `sys.path` — `ROOT_DIR` is `sys._MEIPASS` when frozen.
- Read `packaging/README.md` "Things that will bite you" before touching the spec. "It launched" is not proof; the selftest exercises the engines.

## Working agreement for agents
- **Edit surgically.** Use `Edit` for a small or medium change; rewrite a file only when it is short or most of it is changing.
- **The agent team.** A session builds and reviews through the sub-agents in `.claude/agents/` (`tester` writes the failing tests first, `builder` makes them pass, `reviewer` reproduces, `recon` looks things up); the contract is [`docs/AGENT_TEAM.md`](docs/AGENT_TEAM.md). Read it before spawning one. The lead checks every handoff against `git diff --stat`. Builders and reviewers work in their own worktrees and never touch the desk's checkout; the lead session merges. `.codex/agents/` holds the same four roles for Codex.
- Follow the mandatory documentation workflow above. `plan.md` owns build order; `CURRENT_CHECKPOINT.md` owns the active item. Do not re-implement anything in `CHANGELOG.md` or implement anything directly from `WISHLIST.md`.
- `main` is the trunk; branch per milestone/packet, merge back after a live-session validation day passes (plan.md sec 6). Commit small and green; push after each commit.
- First live session on any new build: run plan.md sec 6 checklist; do NOT tune thresholds from one session.
- **File-scoped ask-first rule:** any edit to a file housing detector/scoring/alert code is asked about BEFORE it is made — even for capture-side or evidence-only changes in that file. Ambiguity is the trigger to ask, not a license to judge.
- If a scheduled task ever runs unmerged branch code on the desk again, disarm it before switching branches there.

## Where to read more
- `CHANGELOG.md` — **`Current implemented inventory` is the contract: search it before building.** `Recent changes` holds the last two build days.
- `docs/DESK_INTERNALS.md` — the incident, measurements and trader conversation behind every `Core loop / data flow` rule. Read the matching entry before changing what a rule governs.
- `plan.md` — remaining roadmap; sec 5 invariants, sec 6 live validation, sec 7 promotion ladder, sec 12 ordered work queue.
- `CURRENT_CHECKPOINT.md` — the `Active state at a glance` block, then the last three build days.
- `WISHLIST.md` — ideas and their open trader questions; never an implementation queue.
- `docs/README.md` — one line per file; `docs/archive/` is history, never context.
- `docs/decisions/` — accepted constraints, read before changing a library, storage or architecture choice. **`0016-trader-vision-and-priorities.md` is the tie-breaker for every prioritisation call**: names before entries, win rate as the swing headline, MFE after a held level for day trades, "what is working lately" on the Trading Desk never in Research, likes are training data.
- `docs/BRANCH_HISTORY.md` — what each branch held and where it landed; the containment proof before deleting one.
- Active specs: the Phase 0.5 packets R1–R8 (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`, `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`, `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md`, `docs/DESK_CHART_UNIFICATION_PLAN.md`, `docs/M5_SIGNAL_ENGINES_PLAN.md`, `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`, `docs/WEEKEND_PREP_PLAN.md`), the warehouse trio (`docs/ULTIMATE_SETUP_DATABASE_PLAN.md`, `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`, `docs/RESEARCH_WAREHOUSE_ERD.md`), `docs/LOCAL_AI_AUTOMATION_PLAN.md`, `docs/REVIEW_LEARNING_LOOP.md`, `docs/CHART_REVIEW_WORKSPACE_PLAN.md`, `docs/AVWAP_BAND_VARIANT_STUDY.md`, `docs/DURABILITY_CATCHUP_PLAN.md`, `docs/SETUPS_MAJOR.md` / `docs/SETUPS_TEST.md`.
- Runbooks: `docs/FIRST_SESSION_CHECKLIST.md`, `docs/AWAY_SCANNER_RUNBOOK.md`, `docs/EVENING_MODE_RUNBOOK.md`, `docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md`, `docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md`, `docs/MACOS_SETUP.md`, `docs/SHIP_READINESS.md`, `docs/BROKER_ADAPTERS.md`, `packaging/README.md`.
- Runtime facts: the main desk is an always-on Ryzen 7 8845HS mini-PC (32 GB, Radeon 780M iGPU — local-LLM host) and does everything; the old i5/3080 Ti desktop is powered down most days and never a writer. Storage is a DAS at `\\MINI-PC\Trading Bot Data` holding `research_lake/`, `ai_store/` and the cold-pushed subtrees. A full scan measured 17–21 min over 1,097 symbols on the 8845HS. Post-session artifacts under `%LOCALAPPDATA%\TradingBotV3\diagnostics\`.

`AGENTS.md` is a copy of this file (symlinks don't survive Windows checkouts) — edit CLAUDE.md, then re-copy.
